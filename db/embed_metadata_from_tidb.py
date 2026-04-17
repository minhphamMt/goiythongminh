import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.metadata import MetadataEmbeddingService
from app.repository import enqueue_missing_jobs
from app.workers.common import drain_jobs_once


def main() -> None:
    queued = enqueue_missing_jobs("metadata", requested_source="legacy-batch", priority=50)
    print(f"Queued metadata jobs: {queued}")

    service = MetadataEmbeddingService()
    processed = drain_jobs_once("metadata", service.embed_song, worker_name="metadata-batch")
    print(f"Processed metadata jobs: {processed}")


if __name__ == "__main__":
    main()
