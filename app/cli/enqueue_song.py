from __future__ import annotations

import argparse
import json

from app.repository import enqueue_embedding_job, song_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue embedding jobs for one song.")
    parser.add_argument("song_id", type=int, help="Song ID to enqueue")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["metadata", "audio"],
        default=["metadata", "audio"],
        help="Job types to enqueue",
    )
    parser.add_argument("--source", default="manual", help="requested_source value")
    parser.add_argument("--priority", type=int, default=100, help="Job priority")
    args = parser.parse_args()

    if not song_exists(args.song_id):
        raise SystemExit(f"Song {args.song_id} not found or deleted.")

    jobs = [
        enqueue_embedding_job(
            args.song_id,
            job_type,
            requested_source=args.source,
            priority=args.priority,
        )
        for job_type in args.types
    ]
    print(json.dumps(jobs, indent=2, default=str))


if __name__ == "__main__":
    main()
