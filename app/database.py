from __future__ import annotations

from contextlib import contextmanager

import pymysql

from app.config import settings


def get_connection(*, autocommit: bool = False):
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_pass,
        database=settings.db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=autocommit,
        ssl={"ca": str(settings.db_ca_path)},
    )


@contextmanager
def connection(*, autocommit: bool = False):
    conn = get_connection(autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def cursor(*, autocommit: bool = False):
    with connection(autocommit=autocommit) as conn:
        with conn.cursor() as cur:
            yield cur
