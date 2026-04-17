import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.audio import AudioEmbeddingService
from app.repository import enqueue_missing_jobs
from app.workers.common import drain_jobs_once


def main() -> None:
    queued = enqueue_missing_jobs("audio", requested_source="legacy-batch", priority=50)
    print(f"Queued audio jobs: {queued}")

    service = AudioEmbeddingService()
    processed = drain_jobs_once("audio", service.embed_song, worker_name="audio-batch")
    print(f"Processed audio jobs: {processed}")


if __name__ == "__main__":
    main()
