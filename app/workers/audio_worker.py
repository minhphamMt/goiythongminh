from app.core.audio import AudioEmbeddingService
from app.workers.common import run_worker_loop


def main() -> None:
    service = AudioEmbeddingService()
    run_worker_loop("audio", service.embed_song)


if __name__ == "__main__":
    main()
