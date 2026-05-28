import logging
import os
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_MIN_NEGATIVE_COUNT = int(os.getenv("FEEDBACK_MIN_NEGATIVE_COUNT", "1"))
SQLITE_FEEDBACK_PATH = Path(
    os.getenv("FEEDBACK_SQLITE_PATH", "data/search_feedback.sqlite3")
)


def normalize_feedback_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _get_database_url() -> str | None:
    return os.getenv("FEEDBACK_DATABASE_URL") or os.getenv("DATABASE_URL")


def _ensure_postgres_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS search_feedback (
                query_norm TEXT NOT NULL,
                query_text TEXT NOT NULL,
                hadits_id TEXT NOT NULL,
                action TEXT NOT NULL,
                client_id TEXT NOT NULL,
                source TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (query_norm, hadits_id, client_id)
            )
            """
        )
    conn.commit()


def _ensure_sqlite_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_feedback (
            query_norm TEXT NOT NULL,
            query_text TEXT NOT NULL,
            hadits_id TEXT NOT NULL,
            action TEXT NOT NULL,
            client_id TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (query_norm, hadits_id, client_id)
        )
        """
    )
    conn.commit()


def _connect_sqlite() -> sqlite3.Connection:
    SQLITE_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_FEEDBACK_PATH)
    _ensure_sqlite_table(conn)
    return conn


def _connect_postgres(db_url: str):
    import psycopg2

    return psycopg2.connect(db_url)


def record_search_feedback(
    *,
    query: str,
    hadits_id: str,
    action: str,
    client_id: str,
    source: str | None = None,
) -> None:
    query_norm = normalize_feedback_query(query)
    if action not in {"irrelevant", "clear"}:
        raise ValueError("Action feedback tidak valid")

    db_url = _get_database_url()
    if db_url:
        conn = _connect_postgres(db_url)
        try:
            _ensure_postgres_table(conn)
            with conn.cursor() as cur:
                if action == "clear":
                    cur.execute(
                        """
                        DELETE FROM search_feedback
                        WHERE query_norm = %s AND hadits_id = %s AND client_id = %s
                        """,
                        (query_norm, hadits_id, client_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO search_feedback (
                            query_norm, query_text, hadits_id, action, client_id, source
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (query_norm, hadits_id, client_id)
                        DO UPDATE SET
                            query_text = EXCLUDED.query_text,
                            action = EXCLUDED.action,
                            source = EXCLUDED.source,
                            updated_at = NOW()
                        """,
                        (query_norm, query, hadits_id, action, client_id, source),
                    )
            conn.commit()
        finally:
            conn.close()
        return

    conn = _connect_sqlite()
    try:
        if action == "clear":
            conn.execute(
                """
                DELETE FROM search_feedback
                WHERE query_norm = ? AND hadits_id = ? AND client_id = ?
                """,
                (query_norm, hadits_id, client_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO search_feedback (
                    query_norm, query_text, hadits_id, action, client_id, source
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (query_norm, hadits_id, client_id)
                DO UPDATE SET
                    query_text = excluded.query_text,
                    action = excluded.action,
                    source = excluded.source,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (query_norm, query, hadits_id, action, client_id, source),
            )
        conn.commit()
    finally:
        conn.close()


def get_irrelevant_feedback_ids(query: str) -> set[str]:
    query_norm = normalize_feedback_query(query)
    db_url = _get_database_url()

    try:
        if db_url:
            conn = _connect_postgres(db_url)
            try:
                _ensure_postgres_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT hadits_id, COUNT(DISTINCT client_id)
                        FROM search_feedback
                        WHERE query_norm = %s AND action = 'irrelevant'
                        GROUP BY hadits_id
                        HAVING COUNT(DISTINCT client_id) >= %s
                        """,
                        (query_norm, FEEDBACK_MIN_NEGATIVE_COUNT),
                    )
                    return {row[0] for row in cur.fetchall()}
            finally:
                conn.close()

        conn = _connect_sqlite()
        try:
            rows = conn.execute(
                """
                SELECT hadits_id, COUNT(DISTINCT client_id)
                FROM search_feedback
                WHERE query_norm = ? AND action = 'irrelevant'
                GROUP BY hadits_id
                HAVING COUNT(DISTINCT client_id) >= ?
                """,
                (query_norm, FEEDBACK_MIN_NEGATIVE_COUNT),
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Gagal membaca feedback pencarian: {str(e)}")
        return set()
