from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


def _get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str, *, default: Path) -> Path:
    raw = value.strip() if value else ""
    if not raw:
        return default

    path = Path(raw)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_user: str
    db_pass: str
    db_name: str
    db_ca_path: Path
    gcs_bucket: str
    gcs_project_id: str
    gcs_key_file: Path | None
    gcs_service_account_json: str
    metadata_model_path: Path
    metadata_model_url: str
    audio_model_path: Path
    audio_model_url: str
    artifact_cache_dir: Path
    audio_download_workers: int
    audio_prefetch: int
    audio_segment_batch: int
    audio_download_retries: int
    audio_min_analysis_sec: float
    worker_idle_sleep_seconds: float
    worker_retry_delay_seconds: int
    worker_max_attempts: int
    run_metadata_worker: bool
    run_audio_worker: bool
    supervisor_poll_seconds: float
    supervisor_stop_timeout_seconds: int
    api_host: str
    api_port: int
    api_token: str


settings = Settings(
    db_host=_get_env("DB_HOST", required=True),
    db_port=int(_get_env("DB_PORT", "4000")),
    db_user=_get_env("DB_USER", required=True),
    db_pass=_get_env("DB_PASS", required=True),
    db_name=_get_env("DB_NAME", required=True),
    db_ca_path=(BASE_DIR / "db" / "ca.pem").resolve(),
    gcs_bucket=_get_env("GCS_BUCKET", required=True),
    gcs_project_id=_get_env("GCS_PROJECT_ID", required=True),
    gcs_key_file=(
        _resolve_path(
            _get_env("GCS_KEY_FILE", ""),
            default=(BASE_DIR / "firebase-service-account.json").resolve(),
        )
        if _get_env("GCS_KEY_FILE", "")
        else ((BASE_DIR / "firebase-service-account.json").resolve() if (BASE_DIR / "firebase-service-account.json").exists() else None)
    ),
    gcs_service_account_json=_get_env("GCS_SERVICE_ACCOUNT_JSON", ""),
    metadata_model_path=_resolve_path(
        _get_env("METADATA_MODEL_PATH", "metadata_fasttext.bin"),
        default=(BASE_DIR / "metadata_fasttext.bin").resolve(),
    ),
    metadata_model_url=_get_env("METADATA_MODEL_URL", ""),
    audio_model_path=_resolve_path(
        _get_env("AUDIO_MODEL_PATH", "best_audio_embedding_model.pt"),
        default=(BASE_DIR / "best_audio_embedding_model.pt").resolve(),
    ),
    audio_model_url=_get_env("AUDIO_MODEL_URL", ""),
    artifact_cache_dir=_resolve_path(
        _get_env("ARTIFACT_CACHE_DIR", ".cache"),
        default=(BASE_DIR / ".cache").resolve(),
    ),
    audio_download_workers=int(_get_env("AUDIO_DOWNLOAD_WORKERS", "10")),
    audio_prefetch=int(_get_env("AUDIO_PREFETCH", "40")),
    audio_segment_batch=int(_get_env("AUDIO_SEGMENT_BATCH", "32")),
    audio_download_retries=int(_get_env("AUDIO_DOWNLOAD_RETRIES", "2")),
    audio_min_analysis_sec=float(_get_env("AUDIO_MIN_ANALYSIS_SEC", "5")),
    worker_idle_sleep_seconds=float(_get_env("WORKER_IDLE_SLEEP_SECONDS", "3")),
    worker_retry_delay_seconds=int(_get_env("WORKER_RETRY_DELAY_SECONDS", "30")),
    worker_max_attempts=int(_get_env("WORKER_MAX_ATTEMPTS", "3")),
    run_metadata_worker=_get_bool_env("EMBEDDING_RUN_METADATA_WORKER", True),
    run_audio_worker=_get_bool_env("EMBEDDING_RUN_AUDIO_WORKER", True),
    supervisor_poll_seconds=float(_get_env("SUPERVISOR_POLL_SECONDS", "5")),
    supervisor_stop_timeout_seconds=int(_get_env("SUPERVISOR_STOP_TIMEOUT_SECONDS", "20")),
    api_host=_get_env("EMBEDDING_API_HOST", "0.0.0.0"),
    api_port=int(os.getenv("PORT") or _get_env("EMBEDDING_API_PORT", "8000")),
    api_token=_get_env("EMBEDDING_API_TOKEN", ""),
)
