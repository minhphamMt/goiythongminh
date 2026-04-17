from app.core.metadata import MetadataEmbeddingService
from app.workers.common import run_worker_loop


def main() -> None:
    service = MetadataEmbeddingService()
    run_worker_loop("metadata", service.embed_song)


if __name__ == "__main__":
    main()
