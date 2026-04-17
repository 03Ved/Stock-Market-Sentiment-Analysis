"""
Shared SQL query functions for the Market Sentiment Analyzer.

Centralises the ticker-document fetch logic that was previously duplicated
between aggregation/aggregator.py and api/main.py.
"""

import json
import time
import sqlite3
import logging
from datetime import datetime

from db.database import get_connection

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 0.15  # seconds


def _execute_with_retry(fn):
    """Execute a DB function with retry on 'database is locked'."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc) and attempt < _MAX_RETRIES:
                log.warning(f"DB locked (attempt {attempt}/{_MAX_RETRIES}), retrying...")
                time.sleep(_RETRY_DELAY * attempt)
            else:
                raise
    return None


# ---------------------------------------------------------------------------
# Fetch scored documents for a given ticker within a time window
# ---------------------------------------------------------------------------

_TICKER_DOCS_MINIMAL_SQL = """
    SELECT source_weight, published_at, sentiment, low_confidence
    FROM   documents
    WHERE  published_at >= ?
      AND  sentiment IS NOT NULL
      AND  json_type(tickers) = 'array'
      AND  EXISTS (
          SELECT 1 FROM json_each(tickers)
          WHERE  value = ?
      )
    ORDER BY published_at DESC
"""

_TICKER_DOCS_FULL_SQL = """
    SELECT id, source_id, source_weight, url, headline,
           published_at, sentiment, low_confidence
    FROM   documents
    WHERE  published_at >= ?
      AND  sentiment IS NOT NULL
      AND  json_type(tickers) = 'array'
      AND  EXISTS (
          SELECT 1 FROM json_each(tickers) WHERE value = ?
      )
    ORDER BY published_at DESC
"""


def fetch_docs_for_ticker(
    conn,
    symbol: str,
    since_iso: str,
    columns: list[str] | None = None,
) -> list[dict]:
    """
    Fetch scored documents mentioning `symbol` published since `since_iso`.

    Args:
        conn:      Active SQLite connection (caller manages lifecycle).
        symbol:    NSE ticker symbol, e.g. 'RELIANCE'.
        since_iso: ISO 8601 UTC string cutoff, e.g. '2024-01-01T00:00:00'.
        columns:   Optional list of column names to return.  If None, returns
                   all columns needed by the API layer.

    Returns:
        List of dicts keyed by column name.
    """
    if columns is None:
        col_clause = (
            "id, source_id, source_weight, url, headline, "
            "published_at, sentiment, low_confidence"
        )
    else:
        col_clause = ", ".join(columns)

    sql = f"""
        SELECT {col_clause}
        FROM   documents
        WHERE  published_at >= ?
          AND  sentiment IS NOT NULL
          AND  json_type(tickers) = 'array'
          AND  EXISTS (
              SELECT 1 FROM json_each(tickers) WHERE value = ?
          )
        ORDER BY published_at DESC
    """
    rows = conn.execute(sql, (since_iso, symbol)).fetchall()

    docs = []
    for row in rows:
        d = dict(row)
        if "sentiment" in d and isinstance(d["sentiment"], str):
            d["sentiment"] = json.loads(d["sentiment"])
        docs.append(d)
    return docs


def fetch_ticker_docs_minimal(ticker: str, since: datetime) -> list[dict]:
    """
    Fetch scored documents mentioning `ticker` since `since`.

    Returns only the columns needed for aggregation:
    source_weight, published_at, sentiment dict, low_confidence flag.
    """
    since_iso = since.isoformat()

    def _query():
        with get_connection() as conn:
            return conn.execute(
                _TICKER_DOCS_MINIMAL_SQL, (since_iso, ticker)
            ).fetchall()

    rows = _execute_with_retry(_query)
    if rows is None:
        return []

    docs = []
    for row in rows:
        sentiment = json.loads(row["sentiment"])
        docs.append({
            "source_weight":  row["source_weight"],
            "published_at":   row["published_at"],
            "sentiment":      sentiment,
            "low_confidence": bool(row["low_confidence"]),
        })
    return docs


def fetch_ticker_docs_full(ticker: str, since: datetime) -> list[dict]:
    """
    Fetch scored documents mentioning `ticker` since `since`.

    Returns all columns needed by the API layer (id, source_id, url,
    headline, etc.) in addition to sentiment data.
    """
    since_iso = since.isoformat()

    def _query():
        with get_connection() as conn:
            return conn.execute(
                _TICKER_DOCS_FULL_SQL, (since_iso, ticker)
            ).fetchall()

    rows = _execute_with_retry(_query)
    if rows is None:
        return []

    docs = []
    for row in rows:
        sentiment = json.loads(row["sentiment"])
        docs.append({
            "id":             row["id"],
            "source_id":      row["source_id"],
            "source_weight":  row["source_weight"],
            "url":            row["url"],
            "headline":       row["headline"],
            "published_at":   row["published_at"],
            "sentiment":      sentiment,
            "low_confidence": bool(row["low_confidence"]),
        })
    return docs


# ---------------------------------------------------------------------------
# Pipeline helper queries (used by orchestrator)
# ---------------------------------------------------------------------------

def get_unlinked_docs(limit: int = 50) -> list[dict]:
    """Fetch documents that have not yet been through ticker linking."""
    def _query():
        with get_connection() as conn:
            return conn.execute(
                "SELECT id, headline, body FROM documents WHERE tickers = '[]' LIMIT ?",
                (limit,),
            ).fetchall()

    rows = _execute_with_retry(_query)
    if rows is None:
        return []
    return [{"id": r["id"], "headline": r["headline"], "body": r["body"]} for r in rows]


def get_unscored_docs(limit: int = 50) -> list[dict]:
    """Fetch documents that have not yet been through sentiment scoring."""
    def _query():
        with get_connection() as conn:
            return conn.execute(
                "SELECT id, headline, body FROM documents WHERE sentiment IS NULL LIMIT ?",
                (limit,),
            ).fetchall()

    rows = _execute_with_retry(_query)
    if rows is None:
        return []
    return [{"id": r["id"], "headline": r["headline"], "body": r["body"]} for r in rows]


def update_tickers(doc_id: str, tickers: list[str]) -> None:
    """Update the tickers column for a document."""
    def _update():
        with get_connection() as conn:
            conn.execute(
                "UPDATE documents SET tickers = ? WHERE id = ?",
                (json.dumps(tickers), doc_id),
            )
    _execute_with_retry(_update)


def update_sentiment(doc_id: str, sentiment: dict, low_confidence: bool) -> None:
    """Update the sentiment and low_confidence columns for a document."""
    def _update():
        with get_connection() as conn:
            conn.execute(
                "UPDATE documents SET sentiment = ?, low_confidence = ? WHERE id = ?",
                (json.dumps(sentiment), int(low_confidence), doc_id),
            )
    _execute_with_retry(_update)
