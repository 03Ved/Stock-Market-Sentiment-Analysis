"""
FastAPI REST layer for the Market Sentiment Analyzer.

Endpoints:
    GET /health
    GET /search?q=
    GET /ticker/{symbol}?window=24h|7d|{N}h
    GET /ticker/{symbol}/history?hours=168&resolution=auto
    GET /explain/{doc_id}
"""

import json
import re
import math
from contextlib import asynccontextmanager
from config.settings import BASE_DIR
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from aggregation.aggregator import aggregate_ticker, effective_weight
from config.settings import LAMBDA_DECAY_24H, LAMBDA_DECAY_7D, MIN_DOCS_FOR_SCORE
from db.database import get_connection, init_db
from db.queries import fetch_ticker_docs_full
from nlp.ticker_linker import SYMBOL_SET, NAME_TO_SYMBOL

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Market Sentiment API",
    description="NSE/BSE sentiment scores derived from financial news and Reddit.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_window(window: str) -> tuple[float, float]:
    """
    Parse a time window string into (hours, lambda_decay_rate).

    Supports presets: '1h', '6h', '12h', '24h', '3d', '7d', '30d'
    Supports custom: any '{N}h' format like '48h', '72h'

    Returns:
        Tuple of (window_hours, lambda_decay_rate)

    Raises:
        HTTPException(400) if window string is invalid.
    """
    presets = {
        "1h":  (1,    1.0),
        "6h":  (6,    0.5),
        "12h": (12,   0.4),
        "24h": (24,   0.30),
        "3d":  (72,   0.10),
        "7d":  (168,  0.05),
        "30d": (720,  0.01),
    }
    if window in presets:
        return presets[window]
    match = re.match(r"^(\d+)h$", window)
    if match:
        hours = int(match.group(1))
        if hours < 1 or hours > 8760:
            raise HTTPException(status_code=400, detail=f"Window hours must be 1-8760, got {hours}")
        decay = max(0.01, 0.30 * (24 / max(hours, 1)))
        return (float(hours), decay)
    raise HTTPException(
        status_code=400,
        detail=f"Invalid window format: '{window}'. Use presets (1h,6h,12h,24h,3d,7d,30d) or custom like '48h'.",
    )


_WINDOW_RE = re.compile(r"^(\d+)(h|d)$")


def _parse_window(window: str) -> tuple[timedelta, float]:
    """
    Parse a window string like '24h', '7d', '48h', '3d' into
    (timedelta, lambda_decay).

    Falls back to the fixed presets for '24h' and '7d'.
    For dynamic values, interpolates lambda: λ = 0.3 * (24 / hours),
    capped to [0.01, 1.0].
    """
    if window == "24h":
        return timedelta(hours=24), LAMBDA_DECAY_24H
    if window == "7d":
        return timedelta(days=7), LAMBDA_DECAY_7D

    m = _WINDOW_RE.match(window)
    if not m:
        raise ValueError(f"Invalid window format: {window!r}")

    value = int(m.group(1))
    unit  = m.group(2)
    hours = value if unit == "h" else value * 24

    if hours < 1:
        raise ValueError("Window must be at least 1 hour.")

    delta = timedelta(hours=hours)
    # Interpolate decay: heavier for short windows, lighter for long
    lambda_decay = max(0.01, min(1.0, 0.3 * (24 / hours)))
    return delta, lambda_decay


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _top_articles(docs: list[dict], now: datetime, lambda_decay: float, n: int = 5) -> list[dict]:
    """Return the top-n articles sorted by effective weight (most influential first)."""
    scored = sorted(
        docs,
        key=lambda d: effective_weight(d, now, lambda_decay),
        reverse=True,
    )
    return [
        {
            "id":           d.get("id", ""),
            "headline":     d["headline"],
            "url":          d["url"],
            "source_id":    d["source_id"],
            "published_at": d["published_at"],
            "sentiment":    {
                "score":      round(d["sentiment"]["score"], 4),
                "confidence": round(d["sentiment"]["confidence"], 4),
            },
        }
        for d in scored[:n]
    ]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    with get_connection() as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        scored    = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE sentiment IS NOT NULL"
        ).fetchone()[0]
        linked    = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE tickers != '[]'"
        ).fetchone()[0]
    return {
        "status": "ok",
        "doc_count": doc_count,
        "scored_count": scored,
        "linked_count": linked,
    }


# ---------------------------------------------------------------------------
# GET /search?q=
# ---------------------------------------------------------------------------

@app.get("/search")
def search(q: str = Query(..., min_length=1, description="Ticker or company name query")):
    """
    Return matching NSE ticker symbols for autocomplete.
    Searches both the symbol set and company name mappings.
    """
    q_upper = q.upper().strip()
    q_lower = q.lower().strip()

    matches: list[dict] = []
    seen: set[str] = set()

    # Exact company aliases must beat substring matches such as HCL Infosystems.
    exact_symbol = NAME_TO_SYMBOL.get(q_lower)
    if q_upper in SYMBOL_SET:
        exact_symbol = q_upper
    if exact_symbol:
        matches.append({"symbol": exact_symbol, "name": q_lower, "match_type": "exact"})
        seen.add(exact_symbol)

    # Exact symbol prefix match
    for sym in sorted(SYMBOL_SET):
        if sym.startswith(q_upper) and sym not in seen:
            matches.append({"symbol": sym, "match_type": "symbol"})
            seen.add(sym)
        if len(matches) >= 10:
            break

    # Company name / alias substring match
    for name, sym in NAME_TO_SYMBOL.items():
        if q_lower in name and sym not in seen:
            matches.append({"symbol": sym, "name": name, "match_type": "name"})
            seen.add(sym)
        if len(matches) >= 20:
            break

    return {"query": q, "results": matches[:20]}


# ---------------------------------------------------------------------------
# GET /ticker/{symbol}?window=24h|7d|{N}h
# ---------------------------------------------------------------------------

@app.get("/ticker/{symbol}")
def ticker_sentiment(
    symbol: str,
    window: str = Query("24h"),
):
    """
    Return aggregated sentiment score for a ticker and its top-5 driving articles.
    Accepts dynamic windows: '24h', '7d', '48h', '3d', etc.
    """
    symbol = symbol.upper()
    if symbol not in SYMBOL_SET:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in NSE master.")

    window_hours, lambda_decay = parse_window(window)  # raises HTTPException on bad input
    now   = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_hours)

    docs   = fetch_ticker_docs_full(symbol, since)
    result = aggregate_ticker(docs, now, lambda_decay)

    # Live-fetch fallback: if fewer than MIN_DOCS_FOR_SCORE docs exist, try
    # pulling fresh articles.
    live_fetched = False
    if len(docs) < MIN_DOCS_FOR_SCORE:
        # Primary fallback: Google News RSS (free, no API key, works for all stocks)
        try:
            from ingestion.google_news_scraper import fetch_google_news_for_ticker
            import json as _json
            # Load company name lookup
            ticker_names = {}
            try:
                with open(BASE_DIR / "data" / "ticker_names.json", "r", encoding="utf-8") as _f:
                    ticker_names = _json.load(_f)
            except Exception:
                pass

            company_name = ticker_names.get(symbol)
            new_count = fetch_google_news_for_ticker(symbol, company_name)

            if new_count > 0:
                docs   = fetch_ticker_docs_full(symbol, since)
                result = aggregate_ticker(docs, now, lambda_decay)
                live_fetched = True
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning(f"Google News fallback failed for {symbol}: {exc}")

        # Secondary fallback: MarketAux (needs API key, rate-limited)
        if not live_fetched and len(docs) < MIN_DOCS_FOR_SCORE:
            try:
                from ingestion.live_fetch import live_fetch_for_ticker
                new_count = live_fetch_for_ticker(symbol)
                if new_count > 0:
                    docs   = fetch_ticker_docs_full(symbol, since)
                    result = aggregate_ticker(docs, now, lambda_decay)
                    live_fetched = True
            except Exception as exc:
                import logging as _log
                _log.getLogger(__name__).warning(f"MarketAux fallback failed for {symbol}: {exc}")

    if result is None:
        return {
            "symbol":       symbol,
            "window":       window,
            "status":       "insufficient_data",
            "detail":       f"Fewer than {MIN_DOCS_FOR_SCORE} scored documents found.",
            "doc_count":    len(docs),
            "articles":     _top_articles(docs, now, lambda_decay) if docs else [],
            "live_fetched": live_fetched,
        }

    return {
        "symbol":       symbol,
        "window":       window,
        "score":        result["score"],
        "doc_count":    result["doc_count"],
        "total_weight": result["total_weight"],
        "confidence":   result["confidence"],
        "articles":     _top_articles(docs, now, lambda_decay),
        "live_fetched": live_fetched,
    }


# ---------------------------------------------------------------------------
# GET /ticker/{symbol}/history?hours=168&resolution=auto
# ---------------------------------------------------------------------------

@app.get("/ticker/{symbol}/history")
def ticker_history(
    symbol: str,
    hours: int = Query(168, ge=1, le=8760, description="Lookback window in hours"),
    resolution: str = Query("auto"),
):
    """
    Return time-bucketed sentiment scores for the past `hours` hours.

    Resolution controls bucket size:
      - 'auto': 1h if hours <= 24, 6h if hours <= 168, 24h otherwise
      - '1h', '6h', '24h': explicit bucket sizes
    """
    symbol = symbol.upper()
    if symbol not in SYMBOL_SET:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in NSE master.")

    # Determine bucket size in hours
    if resolution == "auto":
        if hours <= 24:
            bucket_hours = 1
        elif hours <= 168:
            bucket_hours = 6
        else:
            bucket_hours = 24
    elif resolution == "1h":
        bucket_hours = 1
    elif resolution == "6h":
        bucket_hours = 6
    elif resolution in ("24h", "1d"):
        bucket_hours = 24
    else:
        raise HTTPException(status_code=400, detail=f"Invalid resolution: '{resolution}'. Use 'auto', '1h', '6h', or '1d'.")

    now   = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    docs = fetch_ticker_docs_full(symbol, since)

    # Build time buckets
    n_buckets = max(1, math.ceil(hours / bucket_hours))
    points = []

    for i in range(n_buckets):
        bucket_start = since + timedelta(hours=i * bucket_hours)
        bucket_end   = min(now, since + timedelta(hours=(i + 1) * bucket_hours))

        bucket_docs = [
            d for d in docs
            if bucket_start <= _utc_datetime(d["published_at"]) < bucket_end
        ]

        # Use within-bucket decay relative to bucket end
        lambda_decay = max(0.01, min(1.0, 0.3 * (24 / bucket_hours)))
        result = aggregate_ticker(bucket_docs, bucket_end, lambda_decay)

        points.append({
            "timestamp": bucket_start.isoformat(),
            "score":     result["score"] if result else None,
            "doc_count": len(bucket_docs),
        })

    return {
        "symbol":     symbol,
        "hours":      hours,
        "resolution": f"{bucket_hours}h",
        "points":     points,
    }


# ---------------------------------------------------------------------------
# GET /explain/{doc_id}
# ---------------------------------------------------------------------------

@app.get("/explain/{doc_id}")
def explain_article(doc_id: str):
    """
    Return token-level attribution scores for a document's headline.
    Uses Integrated Gradients for interpretability.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT headline, sentiment FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    headline = row["headline"]

    try:
        from nlp.explainer import get_explainer
        explainer = get_explainer()
        attributions = explainer.explain(headline)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Explainer module not available. Install required dependencies.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explainer error: {exc}")

    sentiment = json.loads(row["sentiment"]) if row["sentiment"] else None

    return {
        "doc_id":       doc_id,
        "headline":     headline,
        "sentiment":    sentiment,
        "attributions": attributions,
    }
