# Embedding Service

Service nay giai quyet day du 2 nhu cau:

1. Chon mot bai hat cu the roi enqueue job de embed `metadata` va/hoac `audio`.
2. Quet toan bo database, bai nao chua co embedding thi enqueue job va de worker xu ly nen.

Ban da co the deploy theo kieu `1 Render Web Service duy nhat`. Service do se tu chay:

- API de web goi enqueue va xem trang thai
- metadata worker
- audio worker

Tat ca duoc supervisor boi [app/render_service.py](/d:/goiythongminh/embedding_service/app/render_service.py:1).

## Kien truc

- `app/render_service.py`: start API + metadata worker + audio worker trong cung mot deploy.
- `app/api.py`: HTTP API cho web goi enqueue va xem trang thai.
- `app/workers/metadata_worker.py`: worker giu FastText model trong RAM, xu ly metadata jobs.
- `app/workers/audio_worker.py`: worker giu PyTorch model trong RAM/GPU, xu ly audio jobs.
- `app/core/`: logic embed dung chung.
- `app/repository.py`: queue job trong TiDB va luu embedding vao `song_embeddings`.
- `sql/001_embedding_jobs.sql`: schema bang `embedding_jobs`.

## Artifact va secret

Project da duoc sua de khong bat buoc commit model va secret vao repo.

Co 2 cach cap model:

- local path:
  - `METADATA_MODEL_PATH`
  - `AUDIO_MODEL_PATH`
- download URL:
  - `METADATA_MODEL_URL`
  - `AUDIO_MODEL_URL`

Neu model khong co san o `PATH`, service se thu download ve `ARTIFACT_CACHE_DIR` roi moi khoi dong worker.

Co 2 cach cap GCS credentials:

- secret file / local file:
  - `GCS_KEY_FILE`
- JSON string trong env:
  - `GCS_SERVICE_ACCOUNT_JSON`

Nhung file sau khong nen dua len GitHub:

- `metadata_fasttext.bin`
- `best_audio_embedding_model.pt`
- `firebase-service-account.json`
- `.env`

## API

### 1. Enqueue mot bai

```http
POST /songs/{song_id}/embed
Content-Type: application/json
X-Embedding-Token: <token-neu-ban-bat-auth>

{
  "types": ["metadata", "audio"],
  "priority": 100,
  "requested_source": "manual"
}
```

### 2. Sweep bai chua embed

```http
POST /embeddings/sweep-missing
Content-Type: application/json
X-Embedding-Token: <token-neu-ban-bat-auth>

{
  "types": ["metadata", "audio"],
  "priority": 10,
  "requested_source": "sweep"
}
```

### 3. Xem trang thai cua bai

```http
GET /songs/{song_id}/embedding-status
X-Embedding-Token: <token-neu-ban-bat-auth>
```

### 4. Xem chi tiet mot job

```http
GET /jobs/{job_id}
X-Embedding-Token: <token-neu-ban-bat-auth>
```

## Chay local

Chay tat ca trong mot lenh:

```bash
cd embedding_service
pip install -r requirements.txt
python -m app.render_service
```

Neu can chay rieng tung phan de debug:

```bash
cd embedding_service
python -m app.api
python -m app.workers.metadata_worker
python -m app.workers.audio_worker
```

## Lenh CLI ho tro

Enqueue mot bai:

```bash
cd embedding_service
python -m app.cli.enqueue_song 123 --types metadata audio
```

Sweep bai thieu embedding:

```bash
cd embedding_service
python -m app.cli.enqueue_missing --types metadata audio
```

## Batch wrappers tuong thich

Hai file cu van duoc giu lai, nhung gio da dung he thong jobs moi:

```bash
cd embedding_service
python db/embed_metadata_from_tidb.py
python db/embed_audio_from_firebase.py
```

## Render

Chi can tao `1 Web Service`.

Cau hinh nen dung:

- Root Directory: `embedding_service`
- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python -m app.render_service`
- Health Check Path: `/health`
- Number of instances: `1`

Blueprint mau nam o [render.example.yaml](/d:/goiythongminh/embedding_service/render.example.yaml:1).

Neu deploy len Render de tranh loi vuot gioi han GitHub:

- khong commit `metadata_fasttext.bin`
- khong commit `firebase-service-account.json`
- neu can, dua model len 1 URL private/public phu hop va set `METADATA_MODEL_URL`, `AUDIO_MODEL_URL`
- neu dung secret file tren Render thi dat `GCS_KEY_FILE` tro vao duong dan secret file do
- neu dung env secret thi dat `GCS_SERVICE_ACCOUNT_JSON` bang noi dung JSON service account

## Bien moi truong bo sung

Ngoai bien DB/GCS co san, service ho tro:

- `WORKER_IDLE_SLEEP_SECONDS=3`
- `WORKER_RETRY_DELAY_SECONDS=30`
- `WORKER_MAX_ATTEMPTS=3`
- `METADATA_MODEL_PATH=`
- `METADATA_MODEL_URL=`
- `AUDIO_MODEL_PATH=`
- `AUDIO_MODEL_URL=`
- `ARTIFACT_CACHE_DIR=.cache`
- `GCS_KEY_FILE=`
- `GCS_SERVICE_ACCOUNT_JSON=`
- `EMBEDDING_RUN_METADATA_WORKER=true`
- `EMBEDDING_RUN_AUDIO_WORKER=true`
- `SUPERVISOR_POLL_SECONDS=5`
- `SUPERVISOR_STOP_TIMEOUT_SECONDS=20`
- `EMBEDDING_API_HOST=0.0.0.0`
- `EMBEDDING_API_PORT=8000`
- `EMBEDDING_API_TOKEN=` optional

Ghi chu:

- Tren Render, app se uu tien bien `PORT` do Render cap thay vi `EMBEDDING_API_PORT`.
- Neu muon tam tat 1 worker, co the set `EMBEDDING_RUN_METADATA_WORKER=false` hoac `EMBEDDING_RUN_AUDIO_WORKER=false`.

## Ghi chu deploy

- `metadata_fasttext.bin` cua ban rat lon, vi vay khong nen commit len GitHub. Hay dua model qua URL hoac noi luu tru rieng.
- `firebase-service-account.json` va `.env` khong nen commit cong khai.
- Audio worker can moi truong co `ffmpeg`/audio codec day du.
- Vi API va 2 worker chay trong cung mot service, hay giu `numInstances = 1` de tranh tao job xu ly trung tren nhieu instance.
