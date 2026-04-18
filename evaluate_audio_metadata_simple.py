from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Set

import fasttext
import numpy as np

from app.database import connection


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_MODEL_PATH = BASE_DIR / "metadata_fasttext_light.bin"
VALID_EVAL_MODES = ("combined", "artist", "album", "genre")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple offline evaluation for metadata and audio recommendation models."
    )
    parser.add_argument(
        "--metadata-model-path",
        default=str(DEFAULT_METADATA_MODEL_PATH),
        help="Path to the trained metadata fastText .bin model.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K used for ranking metrics.",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=99,
        help="Number of negative candidates sampled for each query.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3000,
        help="Maximum number of query songs to evaluate. Use 0 or negative to use all eligible songs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--limit-songs",
        type=int,
        default=0,
        help="Optional DB limit for quick experiments.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Metadata weight in the hybrid score. Hybrid = alpha * metadata + (1 - alpha) * audio.",
    )
    parser.add_argument(
        "--eval-modes",
        default="combined",
        help="Comma-separated evaluation modes: combined, artist, album, genre.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save JSON results.",
    )
    return parser.parse_args()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_metadata_text(song: dict) -> str:
    title = normalize_text(song.get("title"))
    artist = normalize_text(song.get("artist_name"))
    album = normalize_text(song.get("album_title"))
    genres = " ".join(f"genre_{genre_id}" for genre_id in song.get("genre_ids", set()))

    return "\n".join(
        [
            title,
            f"{artist} {artist} {artist}".strip(),
            f"{album} {album}".strip(),
            f"{genres} {genres} {genres}".strip(),
        ]
    ).strip()


def parse_vector(raw_value) -> np.ndarray:
    if raw_value is None:
        raise ValueError("Embedding vector is missing.")

    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")

    data = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    vector = np.asarray(data, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape={vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Embedding vector contains non-finite values.")
    return vector


def parse_eval_modes(raw: str) -> list[str]:
    modes = []
    seen = set()
    for part in raw.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in VALID_EVAL_MODES:
            raise ValueError(
                f"Invalid eval mode '{mode}'. Allowed values: {', '.join(VALID_EVAL_MODES)}"
            )
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    if not modes:
        raise ValueError("At least one evaluation mode must be provided.")
    return modes


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return matrix / norms


def compute_metadata_embedding(model: fasttext.FastText._FastText, song: dict) -> np.ndarray:
    text = build_metadata_text(song)
    words = text.split()
    if not words:
        return np.zeros(model.get_dimension(), dtype=np.float32)

    vectors = np.asarray([model.get_word_vector(word) for word in words], dtype=np.float32)
    return l2_normalize(vectors.mean(axis=0, keepdims=True))[0]


def fetch_song_rows(limit_songs: int) -> List[dict]:
    query = """
        SELECT
            s.id,
            s.title,
            s.artist_id,
            a.name AS artist_name,
            COALESCE(s.album_id, 0) AS album_id,
            COALESCE(al.title, '') AS album_title,
            genre_map.genre_ids,
            audio.vector AS audio_vector
        FROM songs s
        JOIN artists a
            ON s.artist_id = a.id
        LEFT JOIN albums al
            ON s.album_id = al.id
        LEFT JOIN (
            SELECT
                sg.song_id,
                GROUP_CONCAT(DISTINCT sg.genre_id ORDER BY sg.genre_id) AS genre_ids
            FROM song_genres sg
            GROUP BY sg.song_id
        ) genre_map
            ON genre_map.song_id = s.id
        JOIN song_embeddings audio
            ON s.id = audio.song_id AND audio.type = 'audio'
        WHERE s.is_deleted = 0
        ORDER BY s.id
    """

    params: tuple = ()
    if limit_songs > 0:
        query += "\nLIMIT %s"
        params = (limit_songs,)

    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    for row in rows:
        raw_genres = row.get("genre_ids")
        row["genre_ids"] = {int(x) for x in str(raw_genres).split(",") if x} if raw_genres else set()
    return rows


def build_dataset(rows: Sequence[dict], metadata_model: fasttext.FastText._FastText) -> Dict[str, object]:
    song_ids: List[int] = []
    artist_ids: List[int] = []
    album_ids: List[int] = []
    genre_sets: List[Set[int]] = []
    metadata_vectors: List[np.ndarray] = []
    audio_vectors: List[np.ndarray] = []
    skipped = 0

    for row in rows:
        try:
            metadata_vec = compute_metadata_embedding(metadata_model, row)
            audio_vec = parse_vector(row["audio_vector"])
        except Exception:
            skipped += 1
            continue

        song_ids.append(int(row["id"]))
        artist_ids.append(int(row.get("artist_id") or 0))
        album_ids.append(int(row.get("album_id") or 0))
        genre_sets.append(set(row.get("genre_ids") or set()))
        metadata_vectors.append(metadata_vec)
        audio_vectors.append(audio_vec)

    if not song_ids:
        raise RuntimeError("No songs with both metadata and audio vectors were prepared.")

    return {
        "song_ids": song_ids,
        "artist_ids": np.asarray(artist_ids, dtype=np.int32),
        "album_ids": np.asarray(album_ids, dtype=np.int32),
        "genre_sets": genre_sets,
        "metadata_matrix": l2_normalize(np.vstack(metadata_vectors)),
        "audio_matrix": l2_normalize(np.vstack(audio_vectors)),
        "skipped": skipped,
    }


def relevance_score(query_idx: int, candidate_idx: int, artist_ids, album_ids, genre_sets) -> int:
    score = 0
    if artist_ids[candidate_idx] == artist_ids[query_idx]:
        score += 3
    if album_ids[query_idx] > 0 and album_ids[candidate_idx] == album_ids[query_idx]:
        score += 2
    if genre_sets[query_idx] & genre_sets[candidate_idx]:
        score += 1
    return score


def build_indices(
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    artist_to_indices: dict[int, set[int]] = {}
    album_to_indices: dict[int, set[int]] = {}
    genre_to_indices: dict[int, set[int]] = {}

    for idx, artist_id in enumerate(artist_ids):
        artist_to_indices.setdefault(int(artist_id), set()).add(idx)

        album_id = int(album_ids[idx])
        if album_id > 0:
            album_to_indices.setdefault(album_id, set()).add(idx)

        for genre_id in genre_sets[idx]:
            genre_to_indices.setdefault(int(genre_id), set()).add(idx)

    return artist_to_indices, album_to_indices, genre_to_indices


def get_positive_pool(
    query_idx: int,
    mode: str,
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    artist_to_indices: dict[int, set[int]],
    album_to_indices: dict[int, set[int]],
    genre_to_indices: dict[int, set[int]],
) -> set[int]:
    if mode == "artist":
        positives = set(artist_to_indices.get(int(artist_ids[query_idx]), set()))
    elif mode == "album":
        album_id = int(album_ids[query_idx])
        positives = set(album_to_indices.get(album_id, set())) if album_id > 0 else set()
    elif mode == "genre":
        positives = set()
        for genre_id in genre_sets[query_idx]:
            positives |= genre_to_indices.get(int(genre_id), set())
    elif mode == "combined":
        positives = set(artist_to_indices.get(int(artist_ids[query_idx]), set()))
        album_id = int(album_ids[query_idx])
        if album_id > 0:
            positives |= album_to_indices.get(album_id, set())
        for genre_id in genre_sets[query_idx]:
            positives |= genre_to_indices.get(int(genre_id), set())
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    positives.discard(query_idx)
    return positives


def choose_positive_idx(
    query_idx: int,
    mode: str,
    positive_pool: set[int],
    rng: random.Random,
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
) -> int:
    if mode == "combined":
        scored = [
            (
                candidate_idx,
                relevance_score(query_idx, candidate_idx, artist_ids, album_ids, genre_sets),
            )
            for candidate_idx in positive_pool
        ]
        best_score = max(score for _, score in scored)
        best_candidates = [idx for idx, score in scored if score == best_score]
        return rng.choice(best_candidates)

    if mode == "genre":
        overlap_counts = [
            (candidate_idx, len(genre_sets[query_idx] & genre_sets[candidate_idx]))
            for candidate_idx in positive_pool
        ]
        best_overlap = max(count for _, count in overlap_counts)
        best_candidates = [idx for idx, count in overlap_counts if count == best_overlap]
        return rng.choice(best_candidates)

    return rng.choice(list(positive_pool))


def build_eval_cases(
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    *,
    mode: str,
    negative_samples: int,
    max_queries: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    all_indices = list(range(len(artist_ids)))
    rng.shuffle(all_indices)
    all_index_set = set(all_indices)

    artist_to_indices, album_to_indices, genre_to_indices = build_indices(
        artist_ids,
        album_ids,
        genre_sets,
    )

    cases: list[dict] = []
    relation_counts = {
        "same_artist": 0,
        "same_album": 0,
        "shared_genre": 0,
    }

    for query_idx in all_indices:
        positive_pool = get_positive_pool(
            query_idx,
            mode,
            artist_ids,
            album_ids,
            genre_sets,
            artist_to_indices,
            album_to_indices,
            genre_to_indices,
        )
        negative_pool = list(all_index_set - positive_pool - {query_idx})

        if not positive_pool or not negative_pool:
            continue

        positive_idx = choose_positive_idx(
            query_idx,
            mode,
            positive_pool,
            rng,
            artist_ids,
            album_ids,
            genre_sets,
        )

        sampled_negatives = (
            rng.sample(negative_pool, negative_samples)
            if len(negative_pool) > negative_samples
            else list(negative_pool)
        )

        cases.append(
            {
                "query_idx": query_idx,
                "positive_idx": positive_idx,
                "candidate_indices": sampled_negatives + [positive_idx],
            }
        )

        if artist_ids[positive_idx] == artist_ids[query_idx]:
            relation_counts["same_artist"] += 1
        if album_ids[query_idx] > 0 and album_ids[positive_idx] == album_ids[query_idx]:
            relation_counts["same_album"] += 1
        if genre_sets[positive_idx] & genre_sets[query_idx]:
            relation_counts["shared_genre"] += 1

        if max_queries > 0 and len(cases) >= max_queries:
            break

    if not cases:
        raise RuntimeError("No valid evaluation cases were created.")

    return cases, relation_counts


def evaluate_cases(embeddings: np.ndarray, cases: Sequence[dict], k: int) -> Dict[str, float]:
    hits = []
    ndcgs = []
    mrrs = []

    for case in cases:
        query_idx = case["query_idx"]
        positive_idx = case["positive_idx"]
        candidate_indices = case["candidate_indices"]

        query_vector = embeddings[query_idx]
        candidate_vectors = embeddings[candidate_indices]
        scores = candidate_vectors @ query_vector

        ranked_indices = np.argsort(-scores)
        ranked_items = [candidate_indices[i] for i in ranked_indices]

        rank = ranked_items.index(positive_idx)
        hits.append(1.0 if rank < k else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 2) if rank < k else 0.0)
        mrrs.append(1.0 / (rank + 1))

    return {
        "hit_rate": float(np.mean(hits)),
        "ndcg": float(np.mean(ndcgs)),
        "mrr": float(np.mean(mrrs)),
    }


def evaluate_hybrid_cases(
    metadata_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    cases: Sequence[dict],
    k: int,
    alpha: float,
) -> Dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha}")

    hits = []
    ndcgs = []
    mrrs = []

    for case in cases:
        query_idx = case["query_idx"]
        positive_idx = case["positive_idx"]
        candidate_indices = case["candidate_indices"]

        meta_query = metadata_embeddings[query_idx]
        meta_candidates = metadata_embeddings[candidate_indices]
        meta_scores = meta_candidates @ meta_query

        audio_query = audio_embeddings[query_idx]
        audio_candidates = audio_embeddings[candidate_indices]
        audio_scores = audio_candidates @ audio_query

        scores = alpha * meta_scores + (1.0 - alpha) * audio_scores

        ranked_indices = np.argsort(-scores)
        ranked_items = [candidate_indices[i] for i in ranked_indices]

        rank = ranked_items.index(positive_idx)
        hits.append(1.0 if rank < k else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 2) if rank < k else 0.0)
        mrrs.append(1.0 / (rank + 1))

    return {
        "hit_rate": float(np.mean(hits)),
        "ndcg": float(np.mean(ndcgs)),
        "mrr": float(np.mean(mrrs)),
    }


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    metadata_model_path = Path(args.metadata_model_path).resolve()
    if not metadata_model_path.exists():
        raise FileNotFoundError(f"Metadata model file not found: {metadata_model_path}")
    eval_modes = parse_eval_modes(args.eval_modes)
    if not 0.0 <= args.hybrid_alpha <= 1.0:
        raise ValueError("--hybrid-alpha must be in [0, 1].")

    print(f"Loading metadata model: {metadata_model_path}")
    metadata_model = fasttext.load_model(str(metadata_model_path))

    print("Fetching songs and audio embeddings from DB...")
    rows = fetch_song_rows(args.limit_songs)
    print(f"Loaded {len(rows)} rows.")

    print("Building metadata/audio matrices...")
    dataset = build_dataset(rows, metadata_model)
    print(
        f"Prepared {len(dataset['song_ids'])} songs "
        f"(skipped {dataset['skipped']} invalid rows)."
    )

    results = {
        "config": {
            "metadata_model_path": str(metadata_model_path),
            "metadata_dimension": int(metadata_model.get_dimension()),
            "audio_dimension": int(dataset["audio_matrix"].shape[1]),
            "k": args.k,
            "negative_samples": args.negative_samples,
            "max_queries": args.max_queries,
            "seed": args.seed,
            "limit_songs": args.limit_songs,
            "eval_modes": eval_modes,
        },
        "dataset": {
            "songs_used": len(dataset["song_ids"]),
            "invalid_rows_skipped": int(dataset["skipped"]),
        },
        "evaluations": {},
    }

    print("Creating evaluation cases and scoring models...")
    for mode in eval_modes:
        cases, relation_counts = build_eval_cases(
            dataset["artist_ids"],
            dataset["album_ids"],
            dataset["genre_sets"],
            mode=mode,
            negative_samples=args.negative_samples,
            max_queries=args.max_queries,
            seed=args.seed,
        )
        print(f"Built {len(cases)} evaluation cases for mode={mode}.")

        metadata_metrics = evaluate_cases(dataset["metadata_matrix"], cases, args.k)
        audio_metrics = evaluate_cases(dataset["audio_matrix"], cases, args.k)
        hybrid_metrics = evaluate_hybrid_cases(
            dataset["metadata_matrix"],
            dataset["audio_matrix"],
            cases,
            args.k,
            args.hybrid_alpha,
        )

        print(
            f"\n[{mode}] Metadata | HR@{args.k}={metadata_metrics['hit_rate']:.4f} "
            f"| NDCG@{args.k}={metadata_metrics['ndcg']:.4f} | MRR={metadata_metrics['mrr']:.4f}"
        )
        print(
            f"[{mode}] Audio    | HR@{args.k}={audio_metrics['hit_rate']:.4f} "
            f"| NDCG@{args.k}={audio_metrics['ndcg']:.4f} | MRR={audio_metrics['mrr']:.4f}"
        )
        print(
            f"[{mode}] Hybrid   | HR@{args.k}={hybrid_metrics['hit_rate']:.4f} "
            f"| NDCG@{args.k}={hybrid_metrics['ndcg']:.4f} | MRR={hybrid_metrics['mrr']:.4f} "
            f"(alpha={args.hybrid_alpha:.2f})"
        )

        results["evaluations"][mode] = {
            "cases": len(cases),
            "positive_relations": relation_counts,
            "metadata": metadata_metrics,
            "audio": audio_metrics,
            "hybrid": hybrid_metrics,
        }

    if args.output:
        ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
