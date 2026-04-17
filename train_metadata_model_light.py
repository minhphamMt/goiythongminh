from __future__ import annotations

import argparse
import math
import unicodedata
import re
from pathlib import Path

import fasttext

from app.database import connection


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = BASE_DIR / ".cache" / "metadata_corpus_light.txt"
DEFAULT_OUTPUT_PATH = BASE_DIR / "metadata_fasttext_light.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a smaller fastText model for metadata embeddings."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save the trained .bin model.",
    )
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS_PATH),
        help="Path to save the generated metadata corpus.",
    )
    parser.add_argument(
        "--model",
        choices=["skipgram", "cbow"],
        default="skipgram",
        help="fastText unsupervised model type.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Embedding dimension. Lower values reduce model size.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="Training epochs.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Ignore words with fewer than this many occurrences.",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=50000,
        help="Number of hash buckets for subwords. Lower values reduce size.",
    )
    parser.add_argument(
        "--minn",
        type=int,
        default=2,
        help="Minimum character n-gram length.",
    )
    parser.add_argument(
        "--maxn",
        type=int,
        default=4,
        help="Maximum character n-gram length. Set 0 to disable subwords.",
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=4,
        help="Number of training threads.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate.",
    )
    parser.add_argument(
        "--word-ngrams",
        type=int,
        default=1,
        help="Max length of word n-grams.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of songs for quick experiments.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of songs fetched from DB per query.",
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
    artist = normalize_text(song.get("artistsNames"))
    album = normalize_text(song.get("albumTitle"))
    genres = " ".join(f"genre_{genre_id}" for genre_id in song.get("genreIds", []))

    return "\n".join(
        [
            title,
            f"{artist} {artist} {artist}".strip(),
            f"{album} {album}".strip(),
            f"{genres} {genres} {genres}".strip(),
        ]
    ).strip()


def fetch_song_batch(*, last_id: int, batch_size: int, remaining: int | None) -> list[dict]:
    limit = batch_size if remaining is None else min(batch_size, remaining)
    if limit <= 0:
        return []

    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.id,
                    MAX(s.title) AS title,
                    MAX(a.name) AS artistsNames,
                    MAX(COALESCE(al.title, '')) AS albumTitle,
                    GROUP_CONCAT(g.id) AS genreIds
                FROM songs s
                JOIN artists a ON s.artist_id = a.id
                LEFT JOIN albums al ON s.album_id = al.id
                LEFT JOIN song_genres sg ON sg.song_id = s.id
                LEFT JOIN genres g ON g.id = sg.genre_id
                WHERE s.is_deleted = 0
                  AND s.id > %s
                GROUP BY s.id
                ORDER BY s.id
                LIMIT %s
                """,
                (last_id, limit),
            )
            rows = cursor.fetchall()

    for row in rows:
        genre_ids = row.get("genreIds")
        row["genreIds"] = list(map(int, genre_ids.split(","))) if genre_ids else []
    return rows


def build_corpus(corpus_path: Path, *, batch_size: int, limit: int) -> tuple[int, int]:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    total_songs = 0
    total_lines = 0
    last_id = 0
    remaining = limit if limit > 0 else None

    with corpus_path.open("w", encoding="utf-8") as fh:
        while True:
            batch = fetch_song_batch(last_id=last_id, batch_size=batch_size, remaining=remaining)
            if not batch:
                break

            for song in batch:
                text = build_metadata_text(song)
                if text:
                    fh.write(text + "\n")
                    total_lines += 1
                total_songs += 1

            last_id = batch[-1]["id"]
            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break

            print(f"Collected {total_songs} songs into corpus...")

    return total_songs, total_lines


def train_model(args: argparse.Namespace, corpus_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training fastText model with settings:")
    print(
        {
            "model": args.model,
            "dim": args.dim,
            "epoch": args.epoch,
            "minCount": args.min_count,
            "bucket": args.bucket,
            "minn": args.minn,
            "maxn": args.maxn,
            "thread": args.thread,
            "lr": args.lr,
            "wordNgrams": args.word_ngrams,
        }
    )

    model = fasttext.train_unsupervised(
        input=str(corpus_path),
        model=args.model,
        dim=args.dim,
        epoch=args.epoch,
        minCount=args.min_count,
        bucket=args.bucket,
        minn=args.minn,
        maxn=args.maxn,
        thread=args.thread,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
    )
    model.save_model(str(output_path))


def format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus).resolve()
    output_path = Path(args.output).resolve()

    print(f"Building corpus at: {corpus_path}")
    total_songs, total_lines = build_corpus(
        corpus_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"Corpus ready. Songs scanned: {total_songs}, lines written: {total_lines}")

    if total_lines == 0:
        raise SystemExit("Corpus is empty. Nothing to train.")

    train_model(args, corpus_path, output_path)

    output_size = output_path.stat().st_size if output_path.exists() else 0
    approx_ram = int(math.ceil(output_size * 1.25))

    print(f"Saved model to: {output_path}")
    print(f"Model size: {format_size(output_size)}")
    print(f"Estimated RAM to load: about {format_size(approx_ram)}")
    print("If you change the dimension, remember to re-embed existing metadata vectors.")


if __name__ == "__main__":
    main()
