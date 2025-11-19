import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "high_scores.sqlite3")


def _resolve_path(path: str | None = None) -> str:
    if path:
        return path
    # Allow env override at runtime
    return os.environ.get("HIGH_SCORES_PATH", DB_DEFAULT_PATH)


@contextmanager
def _conn(path: str | None = None):
    path = _resolve_path(path)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str | None = None):
    with _conn(path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS high_scores (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              name TEXT NOT NULL,
              score REAL NOT NULL,
              best_val_loss REAL NOT NULL,
              duration INTEGER NOT NULL,
              request_idx INTEGER,
              log BLOB NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_high_scores_score_ts
            ON high_scores(score ASC, ts ASC)
            """
        )


def insert_high_score(
    name: str,
    score: float,
    best_val_loss: float,
    duration: int,
    request_idx: int | None,
    log_bytes: bytes,
    ts: str | None = None,
    path: str | None = None,
):
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    with _conn(path) as conn:
        conn.execute(
            (
                "INSERT INTO high_scores (ts, name, score, best_val_loss, "
                "duration, request_idx, log) VALUES (?,?,?,?,?,?,?)"
            ),
            (
                ts,
                name,
                float(score),
                float(best_val_loss),
                int(duration),
                request_idx,
                sqlite3.Binary(log_bytes),
            ),
        )


def get_top_scores(limit: int = 10, path: str | None = None):
    with _conn(path) as conn:
        cur = conn.execute(
            (
                "SELECT name, score, ts FROM high_scores "
                "ORDER BY score ASC, ts ASC LIMIT ?"
            ),
            (limit,),
        )
        return cur.fetchall()


def get_all_scores(offset: int = 0, limit: int = 100, path: str | None = None):
    with _conn(path) as conn:
        cur = conn.execute(
            (
                "SELECT id, ts, name, score, best_val_loss, duration, "
                "request_idx FROM high_scores ORDER BY score ASC, ts ASC "
                "LIMIT ? OFFSET ?"
            ),
            (limit, offset),
        )
        return cur.fetchall()


def get_log_by_id(row_id: int, path: str | None = None) -> bytes | None:
    with _conn(path) as conn:
        cur = conn.execute("SELECT log FROM high_scores WHERE id=?", (row_id,))
        row = cur.fetchone()
        return row[0] if row else None


def get_rank(score: float, path: str | None = None) -> tuple[int, int]:
    """Return (rank, total) for a given score.

    Rank is 1-based and defined as 1 + number of rows with a strictly lower
    score. Ties share the same rank bucket.
    """
    with _conn(path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM high_scores")
        total = int(cur.fetchone()[0])
        cur = conn.execute(
            "SELECT COUNT(*) FROM high_scores WHERE score < ?",
            (float(score),),
        )
        less = int(cur.fetchone()[0])
        return less + 1, total


def delete_high_score(row_id: int, path: str | None = None) -> bool:
    with _conn(path) as conn:
        cur = conn.execute("DELETE FROM high_scores WHERE id=?", (row_id,))
        return cur.rowcount > 0
