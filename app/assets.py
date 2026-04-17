from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    ensure_parent_dir(destination)
    fd, temp_name = tempfile.mkstemp(
        prefix="artifact-",
        suffix=destination.suffix,
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        temp_path.replace(destination)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return destination


def resolve_artifact(
    *,
    local_path: Path | None,
    download_url: str | None,
    cache_path: Path,
    label: str,
) -> Path:
    if local_path and local_path.exists():
        return local_path

    if cache_path.exists():
        return cache_path

    if download_url:
        print(f"Downloading {label} from {download_url}")
        return download_file(download_url, cache_path)

    attempted = str(local_path) if local_path else str(cache_path)
    raise FileNotFoundError(
        f"{label} not found at {attempted}. "
        f"Provide the file locally or set a download URL for it."
    )
