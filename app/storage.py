from __future__ import annotations

import json
from pathlib import Path

from google.cloud import storage

from app.config import settings


_bucket = None


def get_bucket():
    global _bucket

    if _bucket is None:
        if settings.gcs_service_account_json:
            info = json.loads(settings.gcs_service_account_json)
            client = storage.Client.from_service_account_info(
                info,
                project=settings.gcs_project_id,
            )
        elif settings.gcs_key_file:
            key_path = Path(settings.gcs_key_file).resolve()
            if not key_path.exists():
                raise FileNotFoundError(f"GCS key file not found: {key_path}")

            client = storage.Client.from_service_account_json(
                str(key_path),
                project=settings.gcs_project_id,
            )
        else:
            client = storage.Client(project=settings.gcs_project_id)

        _bucket = client.bucket(settings.gcs_bucket)

    return _bucket
