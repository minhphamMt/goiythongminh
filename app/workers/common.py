from __future__ import annotations

import os
import socket
import time
from typing import Callable

from app.config import settings
from app.errors import EmbeddingServiceError
from app.repository import (
    claim_next_job,
    ensure_schema,
    mark_job_completed,
    mark_job_failed,
    reschedule_job,
)


def build_worker_name(job_type: str) -> str:
    return f"{job_type}-worker@{socket.gethostname()}:{os.getpid()}"


def run_worker_loop(
    job_type: str,
    handler: Callable[[int], dict],
    *,
    idle_sleep_seconds: float | None = None,
) -> None:
    ensure_schema()
    worker_name = build_worker_name(job_type)
    sleep_seconds = idle_sleep_seconds or settings.worker_idle_sleep_seconds

    print(f"Starting {worker_name}")

    while True:
        job = claim_next_job(job_type, worker_name)
        if not job:
            time.sleep(sleep_seconds)
            continue

        song_id = job["song_id"]
        job_id = job["id"]
        attempts = job["attempt_count"]
        max_attempts = job["max_attempts"]

        print(f"[{worker_name}] processing job_id={job_id} song_id={song_id} type={job_type}")
        try:
            result = handler(song_id)
            mark_job_completed(job_id)
            print(f"[{worker_name}] completed job_id={job_id} song_id={song_id} result={result}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            retryable = not isinstance(exc, EmbeddingServiceError) or exc.retryable

            if retryable and attempts < max_attempts:
                reschedule_job(job_id, message, settings.worker_retry_delay_seconds)
                print(
                    f"[{worker_name}] retry scheduled job_id={job_id} song_id={song_id} "
                    f"attempt={attempts}/{max_attempts} error={message}"
                )
            else:
                mark_job_failed(job_id, message)
                print(
                    f"[{worker_name}] failed job_id={job_id} song_id={song_id} "
                    f"attempt={attempts}/{max_attempts} error={message}"
                )


def drain_jobs_once(
    job_type: str,
    handler: Callable[[int], dict],
    *,
    worker_name: str | None = None,
    max_jobs: int | None = None,
) -> int:
    ensure_schema()
    worker_name = worker_name or f"batch-{build_worker_name(job_type)}"
    processed = 0

    while True:
        if max_jobs is not None and processed >= max_jobs:
            break

        job = claim_next_job(job_type, worker_name)
        if not job:
            break

        song_id = job["song_id"]
        job_id = job["id"]
        attempts = job["attempt_count"]
        max_attempts = job["max_attempts"]

        print(f"[{worker_name}] processing job_id={job_id} song_id={song_id} type={job_type}")
        try:
            result = handler(song_id)
            mark_job_completed(job_id)
            print(f"[{worker_name}] completed job_id={job_id} song_id={song_id} result={result}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            retryable = not isinstance(exc, EmbeddingServiceError) or exc.retryable

            if retryable and attempts < max_attempts:
                reschedule_job(job_id, message, settings.worker_retry_delay_seconds)
                print(
                    f"[{worker_name}] retry scheduled job_id={job_id} song_id={song_id} "
                    f"attempt={attempts}/{max_attempts} error={message}"
                )
            else:
                mark_job_failed(job_id, message)
                print(
                    f"[{worker_name}] failed job_id={job_id} song_id={song_id} "
                    f"attempt={attempts}/{max_attempts} error={message}"
                )

        processed += 1

    return processed
