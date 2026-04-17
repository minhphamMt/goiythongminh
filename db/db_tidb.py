import os
from contextlib import contextmanager

import pymysql
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# =========================
# REQUIRED ENV
# =========================
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 4000))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

if not DB_HOST:
    raise ValueError("Missing DB_HOST in .env")

# =========================
# SSL CONFIG
# =========================
SSL_CONFIG = {
    "ca": os.path.join(BASE_DIR, "ca.pem")
}


# =========================
# CONNECTION FACTORY
# =========================
def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        ssl=SSL_CONFIG,
    )


@contextmanager
def get_cursor():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
    finally:
        cursor.close()
        conn.close()


# =========================
# TEST CONNECTION
# =========================
def test_connection():
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1 AS ok")
            row = cursor.fetchone()
            print("TiDB Connected:", row)
    except Exception as e:
        print("TiDB Connection Failed:", e)
        raise


if __name__ == "__main__":
    test_connection()
