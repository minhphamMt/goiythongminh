from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.repository import (
    enqueue_embedding_job,
    enqueue_missing_jobs,
    ensure_schema,
    get_job,
    get_song_status,
    song_exists,
)


app = FastAPI(title="Embedding Service", version="1.0.0")


def require_token(x_embedding_token: str | None = Header(default=None)) -> None:
    if settings.api_token and x_embedding_token != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid embedding API token.")


class EnqueueSongRequest(BaseModel):
    types: list[Literal["metadata", "audio"]] = Field(default_factory=lambda: ["metadata", "audio"])
    priority: int = 100
    requested_source: str = "manual"


class SweepRequest(BaseModel):
    types: list[Literal["metadata", "audio"]] = Field(default_factory=lambda: ["metadata", "audio"])
    priority: int = 10
    requested_source: str = "sweep"


@app.on_event("startup")
def on_startup() -> None:
    ensure_schema()


@app.get("/health")
def healthcheck() -> dict:
    return {"ok": True, "service": "embedding-service"}


@app.post("/songs/{song_id}/embed", dependencies=[Depends(require_token)])
def enqueue_song(song_id: int, payload: EnqueueSongRequest) -> dict:
    if not song_exists(song_id):
        raise HTTPException(status_code=404, detail=f"Song {song_id} not found or deleted.")

    jobs = [
        enqueue_embedding_job(
            song_id,
            job_type,
            requested_source=payload.requested_source,
            priority=payload.priority,
        )
        for job_type in payload.types
    ]
    return {"song_id": song_id, "jobs": jobs}


@app.post("/embeddings/sweep-missing", dependencies=[Depends(require_token)])
def enqueue_missing(payload: SweepRequest) -> dict:
    counts = {
        job_type: enqueue_missing_jobs(
            job_type,
            requested_source=payload.requested_source,
            priority=payload.priority,
        )
        for job_type in payload.types
    }
    return {"requested_source": payload.requested_source, "counts": counts}


@app.get("/songs/{song_id}/embedding-status", dependencies=[Depends(require_token)])
def song_status(song_id: int) -> dict:
    if not song_exists(song_id):
        raise HTTPException(status_code=404, detail=f"Song {song_id} not found or deleted.")
    return get_song_status(song_id)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_status(job_id: int) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host=settings.api_host, port=settings.api_port)
