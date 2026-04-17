from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import fasttext
import numpy as np

from app.database import connection


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "metadata_fasttext_light.bin"
DEFAULT_TOPK = "5,10"
DEFAULT_RELEVANCE_FIELDS = "artist,album,genre"
DEFAULT_RELEVANCE_WEIGHTS = "artist=3,album=2,genre=1"
VALID_RELEVANCE_FIELDS = ("artist", "album", "genre")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained metadata fastText model for song recommendation."
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained fastText .bin model.",
    )
    parser.add_argument(
        "--topk",
        default=DEFAULT_TOPK,
        help="Comma-separated list of K values to evaluate. Example: 5,10",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1000,
        help="Maximum number of query songs to evaluate. Use 0 or negative to use all eligible songs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling query songs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of query songs per similarity batch.",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="Minimum number of relevant songs required for a song to be used as a query.",
    )
    parser.add_argument(
        "--limit-songs",
        type=int,
        default=0,
        help="Optional limit on number of songs loaded from DB for quick experiments.",
    )
    parser.add_argument(
        "--relevance-fields",
        default=DEFAULT_RELEVANCE_FIELDS,
        help="Comma-separated relevance criteria. Allowed values: artist, album, genre.",
    )
    parser.add_argument(
        "--relevance-weights",
        default=DEFAULT_RELEVANCE_WEIGHTS,
        help="Weights used for graded relevance. Example: artist=3,album=2,genre=1",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save JSON evaluation results.",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("No valid integer values found.")
    return sorted(set(values))


def parse_component_list(raw: str) -> List[str]:
    values = []
    seen = set()
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in VALID_RELEVANCE_FIELDS:
            raise ValueError(
                f"Invalid relevance field '{part}'. Allowed values: {', '.join(VALID_RELEVANCE_FIELDS)}"
            )
        if part not in seen:
            values.append(part)
            seen.add(part)
    if not values:
        raise ValueError("At least one relevance field must be provided.")
    return values


def parse_weight_map(raw: str, active_fields: Sequence[str]) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("Invalid relevance weight format. Expected items like artist=3.")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key not in VALID_RELEVANCE_FIELDS:
            raise ValueError(
                f"Invalid relevance weight key '{key}'. Allowed values: {', '.join(VALID_RELEVANCE_FIELDS)}"
            )
        weights[key] = int(value.strip())

    return {field: weights.get(field, 1) for field in active_fields}


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


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return matrix / norms


def compute_embedding(model: fasttext.FastText._FastText, song: dict) -> np.ndarray:
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
            genre_map.genre_ids
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


def build_dataset(rows: Sequence[dict], model: fasttext.FastText._FastText) -> Dict[str, object]:
    song_ids: List[int] = []
    titles: List[str] = []
    artist_ids: List[int] = []
    artist_names: List[str] = []
    album_ids: List[int] = []
    album_titles: List[str] = []
    genre_sets: List[Set[int]] = []
    metadata_vectors: List[np.ndarray] = []

    skipped = 0

    for row in rows:
        try:
            vector = compute_embedding(model, row)
        except Exception:
            skipped += 1
            continue

        song_ids.append(int(row["id"]))
        titles.append(row.get("title") or "")
        artist_ids.append(int(row.get("artist_id") or 0))
        artist_names.append(row.get("artist_name") or "")
        album_ids.append(int(row.get("album_id") or 0))
        album_titles.append(row.get("album_title") or "")
        genre_sets.append(set(row.get("genre_ids") or set()))
        metadata_vectors.append(vector)

    if not song_ids:
        raise RuntimeError("No songs could be embedded for evaluation.")

    metadata_matrix = l2_normalize(np.vstack(metadata_vectors))

    return {
        "song_ids": song_ids,
        "titles": titles,
        "artist_ids": np.asarray(artist_ids, dtype=np.int32),
        "artist_names": artist_names,
        "album_ids": np.asarray(album_ids, dtype=np.int32),
        "album_titles": album_titles,
        "genre_sets": genre_sets,
        "metadata_matrix": metadata_matrix,
        "skipped": skipped,
    }


def build_inverted_indices(
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    artist_to_indices: Dict[int, Set[int]] = defaultdict(set)
    album_to_indices: Dict[int, Set[int]] = defaultdict(set)
    genre_to_indices: Dict[int, Set[int]] = defaultdict(set)

    for idx, artist_id in enumerate(artist_ids):
        artist_to_indices[int(artist_id)].add(idx)

        album_id = int(album_ids[idx])
        if album_id > 0:
            album_to_indices[album_id].add(idx)

        for genre_id in genre_sets[idx]:
            genre_to_indices[int(genre_id)].add(idx)

    return artist_to_indices, album_to_indices, genre_to_indices


def build_relevant_set(
    query_idx: int,
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    artist_to_indices: Dict[int, Set[int]],
    album_to_indices: Dict[int, Set[int]],
    genre_to_indices: Dict[int, Set[int]],
    relevance_fields: Sequence[str],
) -> Set[int]:
    relevant: Set[int] = set()

    if "artist" in relevance_fields:
        relevant |= artist_to_indices[int(artist_ids[query_idx])]

    album_id = int(album_ids[query_idx])
    if "album" in relevance_fields and album_id > 0:
        relevant |= album_to_indices[album_id]

    if "genre" in relevance_fields:
        for genre_id in genre_sets[query_idx]:
            relevant |= genre_to_indices[int(genre_id)]

    relevant.discard(query_idx)
    return relevant


def graded_relevance(
    query_idx: int,
    candidate_idx: int,
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    relevance_fields: Sequence[str],
    relevance_weights: Dict[str, int],
) -> int:
    score = 0

    if "artist" in relevance_fields and artist_ids[candidate_idx] == artist_ids[query_idx]:
        score += relevance_weights["artist"]

    if (
        "album" in relevance_fields
        and album_ids[query_idx] > 0
        and album_ids[candidate_idx] == album_ids[query_idx]
    ):
        score += relevance_weights["album"]

    if "genre" in relevance_fields and genre_sets[query_idx] & genre_sets[candidate_idx]:
        score += relevance_weights["genre"]

    return score


def select_query_indices(
    total_songs: int,
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    artist_to_indices: Dict[int, Set[int]],
    album_to_indices: Dict[int, Set[int]],
    genre_to_indices: Dict[int, Set[int]],
    relevance_fields: Sequence[str],
    max_queries: int,
    seed: int,
    min_relevant: int,
) -> List[int]:
    eligible = []

    for idx in range(total_songs):
        relevant = build_relevant_set(
            idx,
            artist_ids,
            album_ids,
            genre_sets,
            artist_to_indices,
            album_to_indices,
            genre_to_indices,
            relevance_fields,
        )
        if len(relevant) >= min_relevant:
            eligible.append(idx)

    if not eligible:
        raise RuntimeError("No eligible query songs found for evaluation.")

    if max_queries and max_queries > 0 and len(eligible) > max_queries:
        rng = random.Random(seed)
        eligible = rng.sample(eligible, max_queries)

    eligible.sort()
    return eligible


def build_query_ground_truth(
    query_indices: Sequence[int],
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    artist_to_indices: Dict[int, Set[int]],
    album_to_indices: Dict[int, Set[int]],
    genre_to_indices: Dict[int, Set[int]],
    relevance_fields: Sequence[str],
    relevance_weights: Dict[str, int],
) -> Dict[int, Dict[str, object]]:
    truth: Dict[int, Dict[str, object]] = {}

    for query_idx in query_indices:
        relevant = build_relevant_set(
            query_idx,
            artist_ids,
            album_ids,
            genre_sets,
            artist_to_indices,
            album_to_indices,
            genre_to_indices,
            relevance_fields,
        )
        grades = {
            idx: graded_relevance(
                query_idx,
                idx,
                artist_ids,
                album_ids,
                genre_sets,
                relevance_fields,
                relevance_weights,
            )
            for idx in relevant
        }
        grades = {idx: grade for idx, grade in grades.items() if grade > 0}

        truth[query_idx] = {
            "relevant": set(grades.keys()),
            "grades": grades,
        }

    return truth


def batched_topk(
    embeddings: np.ndarray,
    query_indices: Sequence[int],
    max_k: int,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    results: Dict[int, np.ndarray] = {}

    for start in range(0, len(query_indices), batch_size):
        batch_query_indices = np.asarray(query_indices[start : start + batch_size], dtype=np.int32)
        batch_vectors = embeddings[batch_query_indices]
        similarities = batch_vectors @ embeddings.T

        row_ids = np.arange(len(batch_query_indices))
        similarities[row_ids, batch_query_indices] = -np.inf

        partial = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
        partial_scores = similarities[row_ids[:, None], partial]
        local_order = np.argsort(-partial_scores, axis=1)
        topk_sorted = partial[row_ids[:, None], local_order]

        for local_row, query_idx in enumerate(batch_query_indices):
            results[int(query_idx)] = topk_sorted[local_row]

    return results


def average_precision_at_k(binary_hits: Sequence[int], relevant_count: int) -> float:
    if relevant_count <= 0:
        return 0.0

    running_hits = 0
    precision_sum = 0.0
    for rank, hit in enumerate(binary_hits, start=1):
        if hit:
            running_hits += 1
            precision_sum += running_hits / rank

    denominator = min(relevant_count, len(binary_hits))
    if denominator <= 0:
        return 0.0
    return precision_sum / denominator


def dcg_from_relevances(relevances: Sequence[int]) -> float:
    dcg = 0.0
    for rank, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        dcg += (2**rel - 1) / math.log2(rank + 1)
    return dcg


def evaluate_model(
    model_name: str,
    embeddings: np.ndarray,
    query_indices: Sequence[int],
    ground_truth: Dict[int, Dict[str, object]],
    artist_ids: np.ndarray,
    album_ids: np.ndarray,
    genre_sets: Sequence[Set[int]],
    relevance_fields: Sequence[str],
    topk_values: Sequence[int],
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    max_k = max(topk_values)
    topk_predictions = batched_topk(embeddings, query_indices, max_k=max_k, batch_size=batch_size)

    metric_sums: Dict[int, Dict[str, float]] = {
        k: {
            "hit_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
            "artist_hit_rate": 0.0,
            "genre_hit_rate": 0.0,
            "album_hit_rate": 0.0,
        }
        for k in topk_values
    }

    for query_idx in query_indices:
        relevant: Set[int] = ground_truth[query_idx]["relevant"]
        grades: Dict[int, int] = ground_truth[query_idx]["grades"]
        ideal_relevances = sorted(grades.values(), reverse=True)
        predictions = topk_predictions[query_idx]

        for k in topk_values:
            top_items = predictions[:k]
            binary_hits = [1 if idx in relevant else 0 for idx in top_items]
            retrieved_hits = sum(binary_hits)

            graded_hits = [grades.get(int(idx), 0) for idx in top_items]
            ideal_topk = ideal_relevances[:k]

            artist_hit = any(artist_ids[int(idx)] == artist_ids[query_idx] for idx in top_items)
            album_hit = (
                album_ids[query_idx] > 0
                and any(album_ids[int(idx)] == album_ids[query_idx] for idx in top_items)
            )
            genre_hit = any(bool(genre_sets[int(idx)] & genre_sets[query_idx]) for idx in top_items)

            metric_sums[k]["hit_rate"] += 1.0 if retrieved_hits > 0 else 0.0
            metric_sums[k]["precision"] += retrieved_hits / k
            metric_sums[k]["recall"] += retrieved_hits / max(len(relevant), 1)
            metric_sums[k]["map"] += average_precision_at_k(binary_hits, len(relevant))

            dcg = dcg_from_relevances(graded_hits)
            idcg = dcg_from_relevances(ideal_topk)
            metric_sums[k]["ndcg"] += dcg / idcg if idcg > 0 else 0.0

            metric_sums[k]["artist_hit_rate"] += 1.0 if artist_hit else 0.0
            metric_sums[k]["genre_hit_rate"] += 1.0 if genre_hit else 0.0
            metric_sums[k]["album_hit_rate"] += 1.0 if album_hit else 0.0

    query_count = len(query_indices)
    averaged = {}
    for k in topk_values:
        averaged[str(k)] = {
            metric_name: metric_value / query_count
            for metric_name, metric_value in metric_sums[k].items()
        }

    print_model_summary(model_name, averaged, relevance_fields)
    return averaged


def print_model_summary(
    model_name: str,
    results: Dict[str, Dict[str, float]],
    relevance_fields: Sequence[str],
) -> None:
    print(f"\n=== {model_name} ===")
    proxy_metric_map = {
        "artist": ("ArtistHit", "artist_hit_rate"),
        "album": ("AlbumHit", "album_hit_rate"),
        "genre": ("GenreHit", "genre_hit_rate"),
    }
    for k_text, metrics in results.items():
        print(
            (
                f"K={k_text:>2} | "
                f"Hit={metrics['hit_rate']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"NDCG={metrics['ndcg']:.4f} | "
                f"MAP={metrics['map']:.4f}"
            )
        )
        proxy_parts = [
            f"{label}={metrics[key]:.4f}"
            for field, (label, key) in proxy_metric_map.items()
            if field in relevance_fields
        ]
        if proxy_parts:
            print(f"      {' | '.join(proxy_parts)}")


def build_relevance_description(
    relevance_fields: Sequence[str],
    relevance_weights: Dict[str, int],
) -> Tuple[str, str]:
    binary_parts = {
        "artist": "same artist",
        "album": "same album",
        "genre": "shared genre",
    }
    graded_parts = [f"{field}={relevance_weights[field]}" for field in relevance_fields]
    binary_desc = " OR ".join(binary_parts[field] for field in relevance_fields)
    graded_desc = ", ".join(graded_parts)
    return binary_desc, graded_desc


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    topk_values = parse_int_list(args.topk)
    relevance_fields = parse_component_list(args.relevance_fields)
    relevance_weights = parse_weight_map(args.relevance_weights, relevance_fields)

    print(f"Loading model: {model_path}")
    print(f"Model size: {format_size(model_path.stat().st_size)}")
    model = fasttext.load_model(str(model_path))
    print(f"Model dimension: {model.get_dimension()}")

    print("Fetching songs from DB...")
    rows = fetch_song_rows(args.limit_songs)
    print(f"Loaded {len(rows)} songs from DB.")

    print("Computing metadata embeddings...")
    dataset = build_dataset(rows, model)
    song_count = len(dataset["song_ids"])
    print(
        f"Prepared {song_count} songs with metadata embeddings "
        f"(skipped {dataset['skipped']} rows)."
    )

    artist_to_indices, album_to_indices, genre_to_indices = build_inverted_indices(
        dataset["artist_ids"],
        dataset["album_ids"],
        dataset["genre_sets"],
    )

    query_indices = select_query_indices(
        total_songs=song_count,
        artist_ids=dataset["artist_ids"],
        album_ids=dataset["album_ids"],
        genre_sets=dataset["genre_sets"],
        artist_to_indices=artist_to_indices,
        album_to_indices=album_to_indices,
        genre_to_indices=genre_to_indices,
        relevance_fields=relevance_fields,
        max_queries=args.max_queries,
        seed=args.seed,
        min_relevant=args.min_relevant,
    )

    query_ground_truth = build_query_ground_truth(
        query_indices,
        dataset["artist_ids"],
        dataset["album_ids"],
        dataset["genre_sets"],
        artist_to_indices,
        album_to_indices,
        genre_to_indices,
        relevance_fields,
        relevance_weights,
    )

    binary_desc, graded_desc = build_relevance_description(
        relevance_fields,
        relevance_weights,
    )

    print(
        f"Evaluating {len(query_indices)} query songs "
        f"with Top-K={','.join(str(k) for k in topk_values)}."
    )
    print(
        f"Binary relevance: {binary_desc}. "
        f"Graded relevance: {graded_desc}."
    )

    metrics = evaluate_model(
        model_name=model_path.name,
        embeddings=dataset["metadata_matrix"],
        query_indices=query_indices,
        ground_truth=query_ground_truth,
        artist_ids=dataset["artist_ids"],
        album_ids=dataset["album_ids"],
        genre_sets=dataset["genre_sets"],
        relevance_fields=relevance_fields,
        topk_values=topk_values,
        batch_size=args.batch_size,
    )

    results = {
        "config": {
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "model_dimension": model.get_dimension(),
            "topk": topk_values,
            "max_queries": args.max_queries,
            "evaluated_queries": len(query_indices),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "min_relevant": args.min_relevant,
            "limit_songs": args.limit_songs,
            "relevance_fields": relevance_fields,
            "relevance_weights": relevance_weights,
        },
        "dataset": {
            "songs_loaded": len(rows),
            "songs_embedded": song_count,
            "invalid_rows_skipped": int(dataset["skipped"]),
            "query_count": len(query_indices),
        },
        "metrics": metrics,
    }

    if args.output:
        ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
