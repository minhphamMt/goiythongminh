from __future__ import annotations

import unicodedata
import re

import fasttext
import numpy as np
from sklearn.preprocessing import normalize

from app.assets import resolve_artifact
from app.config import settings
from app.errors import MetadataNotFoundError, SongNotFoundError
from app.repository import fetch_song_metadata, save_song_embedding


class MetadataEmbeddingService:
    def __init__(self) -> None:
        model_path = resolve_artifact(
            local_path=settings.metadata_model_path,
            download_url=settings.metadata_model_url or None,
            cache_path=settings.artifact_cache_dir / "metadata_fasttext.bin",
            label="Metadata model",
        )
        self.model = fasttext.load_model(str(model_path))
        self.embed_dim = self.model.get_dimension()

    @staticmethod
    def normalize_text(text: str | None) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text.lower())
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build_metadata_text(self, song: dict) -> str:
        title = self.normalize_text(song.get("title"))
        artist = self.normalize_text(song.get("artistsNames"))
        album = self.normalize_text(song.get("albumTitle"))
        genres = " ".join(f"genre_{genre_id}" for genre_id in song.get("genreIds", []))

        return "\n".join(
            [
                title,
                f"{artist} {artist} {artist}".strip(),
                f"{album} {album}".strip(),
                f"{genres} {genres} {genres}".strip(),
            ]
        ).strip()

    def compute_embedding(self, song: dict) -> np.ndarray:
        text = self.build_metadata_text(song)
        words = text.split()

        if not words:
            raise MetadataNotFoundError("Song metadata is empty after normalization.")

        vectors = np.array([self.model.get_word_vector(word) for word in words], dtype=np.float32)
        return normalize(vectors.mean(axis=0, keepdims=True))[0]

    def embed_song(self, song_id: int) -> dict:
        song = fetch_song_metadata(song_id)
        if not song:
            raise SongNotFoundError(f"Song {song_id} not found or deleted.")

        vector = self.compute_embedding(song)
        save_song_embedding(song_id, "metadata", vector.tolist())
        return {
            "song_id": song_id,
            "type": "metadata",
            "dimension": int(vector.shape[0]),
        }
