from __future__ import annotations

import argparse
import json

from app.repository import enqueue_missing_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue missing embedding jobs.")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["metadata", "audio"],
        default=["metadata", "audio"],
        help="Job types to enqueue",
    )
    parser.add_argument("--source", default="sweep", help="requested_source value")
    parser.add_argument("--priority", type=int, default=10, help="Job priority")
    args = parser.parse_args()

    counts = {
        job_type: enqueue_missing_jobs(
            job_type,
            requested_source=args.source,
            priority=args.priority,
        )
        for job_type in args.types
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
