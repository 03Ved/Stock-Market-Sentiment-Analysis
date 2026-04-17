"""
Sentiment aggregation layer.

Combines source reliability weights, exponential time-decay, and FinBERT
confidence into a single per-ticker score for each time window.
"""

import math
from datetime import datetime, timezone, timedelta

from config.settings import (
    LAMBDA_DECAY_24H,
    LAMBDA_DECAY_7D,
    MIN_DOCS_FOR_SCORE,
)
from db.database import get_connection
from db.queries import fetch_ticker_docs_minimal


# ---------------------------------------------------------------------------
# Core weighting + aggregation (pure functions — no DB access)
# ---------------------------------------------------------------------------

def effective_weight(doc: dict, now: datetime, lambda_decay: float = 0.3) -> float:
    """
    Compute the effective weight of a single document.

        weight = source_weight × exp(-λ × age_hours) × confidence

    Args:
        doc:          Unified-schema dict with published_at, source_weight,
                      and sentiment (must have 'confidence' key).
        now:          Current UTC datetime (timezone-aware).
        lambda_decay: Decay constant. 0.3 for 24h window, 0.05 for 7d.
    """
    published = datetime.fromisoformat(doc["published_at"])
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    age_hours      = (now - published).total_seconds() / 3600
    time_weight    = math.exp(-lambda_decay * age_hours)
    confidence     = doc["sentiment"]["confidence"]

    return doc["source_weight"] * time_weight * confidence


def aggregate_ticker(docs: list[dict], now: datetime, lambda_decay: float = 0.3) -> dict | None:
    """
    Compute weighted-average sentiment score for a list of documents.

    Returns None if fewer than MIN_DOCS_FOR_SCORE documents are provided —
    never surface a score based on insufficient evidence.

    Returns:
        {
          "score":        float,   # weighted avg of sentiment scores, in [-1, 1]
          "doc_count":    int,
          "total_weight": float,
          "confidence":   "high" | "low",
        }
    """
    if len(docs) < MIN_DOCS_FOR_SCORE:
        return None

    weights = [effective_weight(d, now, lambda_decay) for d in docs]
    scores  = [d["sentiment"]["score"] for d in docs]

    total_weight = sum(weights)
    if total_weight == 0:
        return None

    weighted_score = sum(w * s for w, s in zip(weights, scores)) / total_weight

    return {
        "score":        round(weighted_score, 4),
        "doc_count":    len(docs),
        "total_weight": round(total_weight, 4),
        "confidence":   "high" if len(docs) >= 10 else "low",
    }


# ---------------------------------------------------------------------------
# DB-backed ticker scorer
# ---------------------------------------------------------------------------

def score_ticker(ticker: str, window: str = "24h") -> dict | None:
    """
    Compute and cache the aggregated sentiment score for `ticker`.

    Args:
        ticker: NSE symbol, e.g. "RELIANCE".
        window: "24h" or "7d".

    Returns aggregated score dict, or None if insufficient data.
    Also writes result to ticker_sentiment_cache.
    """
    now = datetime.now(timezone.utc)

    if window == "24h":
        since        = now - timedelta(hours=24)
        lambda_decay = LAMBDA_DECAY_24H
    elif window == "7d":
        since        = now - timedelta(days=7)
        lambda_decay = LAMBDA_DECAY_7D
    else:
        raise ValueError(f"Unsupported window: {window!r}. Use '24h' or '7d'.")

    docs   = fetch_ticker_docs_minimal(ticker, since)
    result = aggregate_ticker(docs, now, lambda_decay)

    # Cache regardless of None so the API knows when it was last attempted
    _write_cache(ticker, window, result, now)

    return result


def _write_cache(ticker: str, window: str, result: dict | None, now: datetime) -> None:
    score     = result["score"]     if result else None
    doc_count = result["doc_count"] if result else 0
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ticker_sentiment_cache
                (ticker, window, score, doc_count, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, window, score, doc_count, now.isoformat()),
        )


# ---------------------------------------------------------------------------
# Batch scorer — refresh cache for a list of tickers
# ---------------------------------------------------------------------------

def refresh_cache(tickers: list[str]) -> None:
    """Recompute and cache both windows for every ticker in the list."""
    for ticker in tickers:
        for window in ("24h", "7d"):
            score_ticker(ticker, window)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pull a few scored tickers from the DB to test with real data
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT value AS ticker
            FROM documents, json_each(tickers)
            WHERE sentiment IS NOT NULL
            LIMIT 5
            """
        ).fetchall()

    tickers = [r["ticker"] for r in rows]

    if not tickers:
        print("No scored documents in DB yet. Run the NLP pipeline first.")
    else:
        for t in tickers:
            for w in ("24h", "7d"):
                result = score_ticker(t, w)
                print(f"  {t} [{w}]: {result}")
