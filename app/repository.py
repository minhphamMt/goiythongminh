from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from app.config import settings
from app.database import connection


JOB_TYPE_METADATA = "metadata"
JOB_TYPE_AUDIO = "audio"
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
SUPPORTED_JOB_TYPES = {JOB_TYPE_METADATA, JOB_TYPE_AUDIO}


def ensure_schema() -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_jobs (
                    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    song_id BIGINT NOT NULL,
                    job_type VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    requested_source VARCHAR(32) NOT NULL DEFAULT 'manual',
                    priority INT NOT NULL DEFAULT 0,
                    attempt_count INT NOT NULL DEFAULT 0,
                    max_attempts INT NOT NULL DEFAULT 3,
                    available_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    worker_name VARCHAR(128) NULL,
                    claim_token VARCHAR(64) NULL,
                    last_error TEXT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    started_at DATETIME NULL,
                    finished_at DATETIME NULL,
                    last_enqueued_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_embedding_jobs_song_type (song_id, job_type),
                    KEY idx_embedding_jobs_claim (job_type, status, available_at, priority, id),
                    KEY idx_embedding_jobs_song (song_id)
                )
                """
            )


def _validate_job_type(job_type: str) -> None:
    if job_type not in SUPPORTED_JOB_TYPES:
        raise ValueError(f"Unsupported job type: {job_type}")


def song_exists(song_id: int) -> bool:
    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM songs
                WHERE id = %s
                  AND is_deleted = 0
                LIMIT 1
                """,
                (song_id,),
            )
            return cursor.fetchone() is not None


def fetch_song_metadata(song_id: int) -> dict[str, Any] | None:
    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.id,
                    MAX(s.title) AS title,
                    MAX(a.name) AS artistsNames,
                    MAX(COALESCE(al.title, '')) AS albumTitle,
                    GROUP_CONCAT(g.id) AS genreIds
                FROM songs s
                JOIN artists a ON s.artist_id = a.id
                LEFT JOIN albums al ON s.album_id = al.id
                LEFT JOIN song_genres sg ON sg.song_id = s.id
                LEFT JOIN genres g ON g.id = sg.genre_id
                WHERE s.id = %s
                  AND s.is_deleted = 0
                GROUP BY s.id
                """,
                (song_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            genre_ids = row.get("genreIds")
            row["genreIds"] = list(map(int, genre_ids.split(","))) if genre_ids else []
            return row


def fetch_song_audio(song_id: int) -> dict[str, Any] | None:
    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, audio_path
                FROM songs
                WHERE id = %s
                  AND is_deleted = 0
                LIMIT 1
                """,
                (song_id,),
            )
            return cursor.fetchone()


def save_song_embedding(song_id: int, embed_type: str, vector: list[float]) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO song_embeddings (song_id, type, vector)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE vector = VALUES(vector)
                """,
                (song_id, embed_type, json.dumps(vector)),
            )


def has_embedding(song_id: int, embed_type: str) -> bool:
    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM song_embeddings
                WHERE song_id = %s
                  AND type = %s
                LIMIT 1
                """,
                (song_id, embed_type),
            )
            return cursor.fetchone() is not None


def enqueue_embedding_job(
    song_id: int,
    job_type: str,
    *,
    requested_source: str = "manual",
    priority: int = 100,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    _validate_job_type(job_type)
    ensure_schema()

    if max_attempts is None:
        max_attempts = settings.worker_max_attempts

    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_jobs (
                    song_id,
                    job_type,
                    status,
                    requested_source,
                    priority,
                    max_attempts,
                    last_enqueued_at
                )
                VALUES (
                    %s, %s, 'pending', %s, %s, %s, CURRENT_TIMESTAMP
                )
                ON DUPLICATE KEY UPDATE
                    requested_source = VALUES(requested_source),
                    priority = GREATEST(priority, VALUES(priority)),
                    max_attempts = VALUES(max_attempts),
                    attempt_count = CASE
                        WHEN status = 'running' THEN attempt_count
                        ELSE 0
                    END,
                    status = CASE
                        WHEN status = 'running' THEN status
                        ELSE 'pending'
                    END,
                    available_at = CASE
                        WHEN status = 'running' THEN available_at
                        ELSE CURRENT_TIMESTAMP
                    END,
                    worker_name = CASE
                        WHEN status = 'running' THEN worker_name
                        ELSE NULL
                    END,
                    claim_token = CASE
                        WHEN status = 'running' THEN claim_token
                        ELSE NULL
                    END,
                    last_error = CASE
                        WHEN status = 'running' THEN last_error
                        ELSE NULL
                    END,
                    started_at = CASE
                        WHEN status = 'running' THEN started_at
                        ELSE NULL
                    END,
                    finished_at = CASE
                        WHEN status = 'running' THEN finished_at
                        ELSE NULL
                    END,
                    last_enqueued_at = CURRENT_TIMESTAMP
                """,
                (song_id, job_type, requested_source, priority, max_attempts),
            )

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_jobs
                WHERE song_id = %s
                  AND job_type = %s
                """,
                (song_id, job_type),
            )
            return cursor.fetchone()


def enqueue_missing_jobs(
    job_type: str,
    *,
    requested_source: str = "sweep",
    priority: int = 10,
    max_attempts: int | None = None,
) -> int:
    _validate_job_type(job_type)
    ensure_schema()

    if max_attempts is None:
        max_attempts = settings.worker_max_attempts

    if job_type == JOB_TYPE_METADATA:
        select_sql = """
            SELECT
                s.id,
                %s AS job_type,
                'pending' AS status,
                %s AS requested_source,
                %s AS priority,
                %s AS max_attempts,
                CURRENT_TIMESTAMP AS last_enqueued_at
            FROM songs s
            LEFT JOIN song_embeddings e
                ON s.id = e.song_id
               AND e.type = 'metadata'
            WHERE s.is_deleted = 0
              AND e.song_id IS NULL
        """
    else:
        select_sql = """
            SELECT
                s.id,
                %s AS job_type,
                'pending' AS status,
                %s AS requested_source,
                %s AS priority,
                %s AS max_attempts,
                CURRENT_TIMESTAMP AS last_enqueued_at
            FROM songs s
            LEFT JOIN song_embeddings e
                ON s.id = e.song_id
               AND e.type = 'audio'
            WHERE s.is_deleted = 0
              AND s.audio_path IS NOT NULL
              AND TRIM(s.audio_path) <> ''
              AND e.song_id IS NULL
        """

    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO embedding_jobs (
                    song_id,
                    job_type,
                    status,
                    requested_source,
                    priority,
                    max_attempts,
                    last_enqueued_at
                )
                {select_sql}
                ON DUPLICATE KEY UPDATE
                    requested_source = VALUES(requested_source),
                    priority = GREATEST(priority, VALUES(priority)),
                    max_attempts = VALUES(max_attempts),
                    attempt_count = CASE
                        WHEN status = 'running' THEN attempt_count
                        ELSE 0
                    END,
                    status = CASE
                        WHEN status = 'running' THEN status
                        ELSE 'pending'
                    END,
                    available_at = CASE
                        WHEN status = 'running' THEN available_at
                        ELSE CURRENT_TIMESTAMP
                    END,
                    worker_name = CASE
                        WHEN status = 'running' THEN worker_name
                        ELSE NULL
                    END,
                    claim_token = CASE
                        WHEN status = 'running' THEN claim_token
                        ELSE NULL
                    END,
                    last_error = CASE
                        WHEN status = 'running' THEN last_error
                        ELSE NULL
                    END,
                    started_at = CASE
                        WHEN status = 'running' THEN started_at
                        ELSE NULL
                    END,
                    finished_at = CASE
                        WHEN status = 'running' THEN finished_at
                        ELSE NULL
                    END,
                    last_enqueued_at = CURRENT_TIMESTAMP
                """,
                (job_type, requested_source, priority, max_attempts),
            )
            return cursor.rowcount


def claim_next_job(job_type: str, worker_name: str) -> dict[str, Any] | None:
    _validate_job_type(job_type)
    ensure_schema()

    while True:
        claim_token = uuid.uuid4().hex
        with connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_jobs
                    SET
                        status = 'running',
                        worker_name = %s,
                        claim_token = %s,
                        started_at = CURRENT_TIMESTAMP,
                        finished_at = NULL,
                        attempt_count = attempt_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = (
                        SELECT id
                        FROM (
                            SELECT id
                            FROM embedding_jobs
                            WHERE job_type = %s
                              AND status = 'pending'
                              AND available_at <= CURRENT_TIMESTAMP
                            ORDER BY priority DESC, last_enqueued_at ASC, id ASC
                            LIMIT 1
                        ) AS candidate
                    )
                      AND status = 'pending'
                    """,
                    (worker_name, claim_token, job_type),
                )

                if cursor.rowcount == 0:
                    return None

            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM embedding_jobs
                    WHERE claim_token = %s
                    LIMIT 1
                    """,
                    (claim_token,),
                )
                row = cursor.fetchone()

            if row:
                return row


def mark_job_completed(job_id: int) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET
                    status = 'completed',
                    claim_token = NULL,
                    last_error = NULL,
                    finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (job_id,),
            )


def mark_job_failed(job_id: int, error_message: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET
                    status = 'failed',
                    claim_token = NULL,
                    last_error = %s,
                    finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (error_message, job_id),
            )


def reschedule_job(job_id: int, error_message: str, delay_seconds: int) -> None:
    available_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET
                    status = 'pending',
                    worker_name = NULL,
                    claim_token = NULL,
                    last_error = %s,
                    available_at = %s,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (error_message, available_at, job_id),
            )


def get_song_status(song_id: int) -> dict[str, Any]:
    ensure_schema()

    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT job_type, status, attempt_count, max_attempts, last_error,
                       updated_at, started_at, finished_at, requested_source, priority
                FROM embedding_jobs
                WHERE song_id = %s
                """,
                (song_id,),
            )
            jobs = {row["job_type"]: row for row in cursor.fetchall()}

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT type
                FROM song_embeddings
                WHERE song_id = %s
                """,
                (song_id,),
            )
            embedding_types = {row["type"] for row in cursor.fetchall()}

    def _job_state(job_type: str) -> dict[str, Any]:
        job = jobs.get(job_type)
        return {
            "job_type": job_type,
            "has_embedding": job_type in embedding_types,
            "job_status": job["status"] if job else None,
            "attempt_count": job["attempt_count"] if job else 0,
            "max_attempts": job["max_attempts"] if job else settings.worker_max_attempts,
            "last_error": job["last_error"] if job else None,
            "updated_at": job["updated_at"].isoformat() if job and job["updated_at"] else None,
            "started_at": job["started_at"].isoformat() if job and job["started_at"] else None,
            "finished_at": job["finished_at"].isoformat() if job and job["finished_at"] else None,
            "requested_source": job["requested_source"] if job else None,
            "priority": job["priority"] if job else None,
        }

    return {
        "song_id": song_id,
        "metadata": _job_state(JOB_TYPE_METADATA),
        "audio": _job_state(JOB_TYPE_AUDIO),
    }


def get_job(job_id: int) -> dict[str, Any] | None:
    ensure_schema()
    with connection(autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_jobs
                WHERE id = %s
                LIMIT 1
                """,
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            for key in ("created_at", "updated_at", "started_at", "finished_at", "available_at"):
                if row.get(key):
                    row[key] = row[key].isoformat()
            return row
