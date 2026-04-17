from __future__ import annotations

import math
import os
import tempfile
import time
import urllib.parse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from app.assets import resolve_artifact
from app.config import settings
from app.errors import AudioDecodeError, MissingAudioPathError, SongNotFoundError
from app.repository import fetch_song_audio, save_song_embedding
from app.storage import get_bucket


SR = 22050
SEGMENT_SEC = 30
N_MELS = 128
MAX_LEN = 1300
MIN_ANALYSIS_SAMPLES = max(2048, int(SR * settings.audio_min_analysis_sec))
EMBED_DIM = 128
NUM_CLASSES = 10

torch.backends.cudnn.benchmark = True


class AudioCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        emb = self.embedding(x)
        if return_embedding:
            return emb
        return self.classifier(emb)


class AudioEmbeddingService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = self.device == "cuda"
        self.segment_batch_size = settings.audio_segment_batch
        self.download_retries = settings.audio_download_retries

        model_path = resolve_artifact(
            local_path=Path(settings.audio_model_path),
            download_url=settings.audio_model_url or None,
            cache_path=settings.artifact_cache_dir / "best_audio_embedding_model.pt",
            label="Audio model",
        )

        self.model = AudioCNN(EMBED_DIM, NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @staticmethod
    def to_blob_path(audio_path: str) -> str:
        if not audio_path:
            return ""

        path = urllib.parse.unquote(audio_path.strip().lstrip("/"))
        while path.startswith("uploads/uploads/"):
            path = path.replace("uploads/uploads/", "uploads/", 1)
        if not path.startswith("uploads/"):
            path = f"uploads/{path}"
        return path

    @staticmethod
    def extract_logmel(wave: np.ndarray, sr: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wave,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=N_MELS,
        )
        logmel = librosa.power_to_db(mel)
        if logmel.shape[1] < MAX_LEN:
            pad = MAX_LEN - logmel.shape[1]
            logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant")
        else:
            logmel = logmel[:, :MAX_LEN]
        return logmel

    @staticmethod
    def prepare_audio_segment(segment: np.ndarray) -> np.ndarray:
        if segment.size == 0:
            return np.zeros(MIN_ANALYSIS_SAMPLES, dtype=np.float32)

        segment = np.asarray(segment, dtype=np.float32)
        if segment.shape[0] >= MIN_ANALYSIS_SAMPLES:
            return segment

        repeats = math.ceil(MIN_ANALYSIS_SAMPLES / segment.shape[0])
        return np.tile(segment, repeats)[:MIN_ANALYSIS_SAMPLES]

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            return np.zeros_like(vector, dtype=np.float32)
        return vector / norm

    @staticmethod
    def get_temp_suffix(blob_path: str) -> str:
        _, ext = os.path.splitext(blob_path or "")
        if not ext or len(ext) > 10:
            return ".audio"
        return ext.lower()

    def download_to_tempfile(self, blob_path: str) -> str:
        fd, temp_path = tempfile.mkstemp(suffix=self.get_temp_suffix(blob_path))
        os.close(fd)
        try:
            blob = get_bucket().blob(blob_path)
            blob.download_to_filename(temp_path)
            return temp_path
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

    def download_with_retry(self, blob_path: str) -> str:
        last_error = None
        for attempt in range(self.download_retries + 1):
            try:
                return self.download_to_tempfile(blob_path)
            except Exception as exc:
                last_error = exc
                time.sleep(0.6 * (attempt + 1))
        raise last_error

    @torch.no_grad()
    def embed_audio_file(self, file_path: str) -> np.ndarray:
        try:
            wave, sr = librosa.load(file_path, sr=SR, mono=True)
        except Exception as exc:
            raise AudioDecodeError(f"Failed to decode audio file: {exc}") from exc

        if wave is None or wave.size == 0:
            raise AudioDecodeError("Audio file is empty.")

        wave = np.asarray(wave, dtype=np.float32)
        seg_len = SEGMENT_SEC * sr
        tensors = []
        weights = []

        for start in range(0, len(wave), seg_len):
            raw_seg = wave[start : start + seg_len]
            if len(raw_seg) == 0:
                continue

            seg = self.prepare_audio_segment(raw_seg)
            logmel = self.extract_logmel(seg, sr)
            tensors.append(torch.tensor(logmel, dtype=torch.float32))
            weights.append(max(len(raw_seg), 1))

        if not tensors:
            raise AudioDecodeError("No usable audio segments found.")

        embeddings = []
        for i in range(0, len(tensors), self.segment_batch_size):
            batch = torch.stack(tensors[i : i + self.segment_batch_size], dim=0)
            batch = batch.unsqueeze(1).to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                batch_emb = self.model(batch, return_embedding=True)
            embeddings.append(batch_emb.detach().cpu().numpy())

        stacked = np.vstack(embeddings)
        weight_arr = np.asarray(weights, dtype=np.float32)
        return self.l2_normalize(np.average(stacked, axis=0, weights=weight_arr))

    def embed_song(self, song_id: int) -> dict:
        row = fetch_song_audio(song_id)
        if not row:
            raise SongNotFoundError(f"Song {song_id} not found or deleted.")

        blob_path = self.to_blob_path(row.get("audio_path") or "")
        if not blob_path:
            raise MissingAudioPathError(f"Song {song_id} does not have a valid audio_path.")

        temp_path = self.download_with_retry(blob_path)
        try:
            vector = self.embed_audio_file(temp_path)
            save_song_embedding(song_id, "audio", vector.tolist())
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        return {
            "song_id": song_id,
            "type": "audio",
            "dimension": int(vector.shape[0]),
            "device": self.device,
            "blob_path": blob_path,
        }
