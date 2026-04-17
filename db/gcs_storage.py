import os
import pathlib
from dotenv import load_dotenv
from google.cloud import storage

# =========================
# LOAD ENV
# =========================
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# =========================
# REQUIRED ENV
# =========================
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_KEY_FILE = os.getenv("GCS_KEY_FILE")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID")

if not GCS_BUCKET:
    raise ValueError("Missing GCS_BUCKET in .env")

if not GCS_KEY_FILE:
    raise ValueError("Missing GCS_KEY_FILE in .env")

key_path = pathlib.Path(GCS_KEY_FILE).resolve()

if not key_path.exists():
    raise FileNotFoundError(f"Key file not found at: {key_path}")

# =========================
# INIT STORAGE (FIXED)
# =========================
storage_client = storage.Client.from_service_account_json(
    str(key_path),
    project=GCS_PROJECT_ID
)

bucket = storage_client.bucket(GCS_BUCKET)

# =========================
# UPLOAD BUFFER
# =========================
def upload_buffer(key: str, buffer: bytes, content_type: str):
    blob = bucket.blob(key)
    blob.upload_from_string(
        buffer,
        content_type=content_type
    )

    return {
        "path": "/" + key.replace("uploads", "")
    }

# =========================
# DOWNLOAD BLOB
# =========================
def download_blob_as_bytes(blob_path: str) -> bytes:
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()

# =========================
# TEST
# =========================
def test_connection():
    try:
        print("✅ GCS Connected")
        print("Bucket:", bucket.name)
    except Exception as e:
        print("❌ GCS Connection Failed:", e)
        raise


if __name__ == "__main__":
    test_connection()
