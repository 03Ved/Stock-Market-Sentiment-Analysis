import json
import sqlite3
import time
import logging
from pathlib import Path

from config.settings import DB_PATH

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 0.1  # seconds


def get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory set to Row for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_connection_with_retry() -> sqlite3.Connection:
    """Return a connection, retrying on transient 'database is locked' errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return get_connection()
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc) and attempt < _MAX_RETRIES:
                log.warning(f"[db] connection locked (attempt {attempt}), retrying...")
                time.sleep(_RETRY_DELAY * attempt)
            else:
                raise
    return get_connection()  # fallback — should not reach here


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    schema = _SCHEMA_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(schema)
    print(f"[db] Initialised database at {DB_PATH}")


# ---------------------------------------------------------------------------
# Universal document insert (used by all scrapers)
# ---------------------------------------------------------------------------

def insert_doc(conn, doc: dict) -> bool:
    """
    Insert a document into the DB using INSERT OR IGNORE for deduplication.
    Returns True if the row was newly inserted, False if it already existed.
    """
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO documents
            (id, source_id, source_weight, url, headline, body,
             published_at, scraped_at, tickers, sentiment, low_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc["id"],
            doc["source_id"],
            doc["source_weight"],
            doc["url"],
            doc["headline"],
            doc["body"],
            doc["published_at"],
            doc["scraped_at"],
            json.dumps(doc["tickers"]),
            json.dumps(doc["sentiment"]) if doc["sentiment"] is not None else None,
            int(doc.get("low_confidence", (doc.get("sentiment") or {}).get("low_confidence", False))),
        ),
    )
    return cursor.rowcount == 1


def insert_doc_with_retry(doc: dict) -> bool:
    """
    Insert a document with automatic retry on 'database is locked'.
    Opens and closes its own connection.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with get_connection() as conn:
                return insert_doc(conn, doc)
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc) and attempt < _MAX_RETRIES:
                log.warning(f"[db] insert locked (attempt {attempt}), retrying...")
                time.sleep(_RETRY_DELAY * attempt)
            else:
                raise
    return False


if __name__ == "__main__":
    init_db()
