"""
On-demand live fetch for a single NSE ticker symbol.

Calls the MarketAux API in real time, inserts new documents, then immediately
runs ticker linking and sentiment scoring so the results are ready to aggregate.
Used as a fallback by the API when the database has insufficient data for a ticker.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

import requests

from config.settings import MARKETAUX_API_KEY
from db.database import get_connection, insert_doc
from nlp.ticker_linker import link_tickers
from nlp.sentiment import score_document

logger = logging.getLogger(__name__)

_MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
_SOURCE_WEIGHT = 0.75


def live_fetch_for_ticker(symbol: str) -> int:
    """
    Fetch, insert, link, and score fresh MarketAux articles for a single NSE symbol.

    Args:
        symbol: NSE ticker symbol, e.g. 'RELIANCE'.

    Returns:
        Number of newly inserted and scored documents. Returns 0 on any error
        or if MARKETAUX_API_KEY is not set. Never raises.
    """
    if not MARKETAUX_API_KEY:
        return 0

    try:
        resp = requests.get(
            _MARKETAUX_URL,
            params={
                "api_token":      MARKETAUX_API_KEY,
                "symbols":        f"NSE:{symbol}",
                "language":       "en",
                "limit":          10,
                "filter_entities": "true",
            },
            timeout=15,
        )
        resp.raise_for_status()

        articles = resp.json().get("data", [])
        if not articles:
            logger.info(f"live_fetch: no articles returned for {symbol}")
            return 0

        inserted_and_scored = 0

        for article in articles:
            url = article.get("url", "")
            headline = article.get("title", "")
            if not url or not headline:
                continue

            body = article.get("description")

            # Normalise timestamp
            published_raw = article.get("published_at", "")
            if published_raw and not published_raw.endswith("Z") and "+" not in published_raw:
                published_raw += "Z"

            # Extract NSE tickers from MarketAux entity tags
            nse_tickers = [
                entity["symbol"].replace("NSE:", "").strip()
                for entity in article.get("entities", [])
                if entity.get("exchange") == "NSE" and entity.get("symbol")
            ]

            doc = {
                "id":           hashlib.sha256(url.encode()).hexdigest(),
                "source_id":    "marketaux",
                "source_weight": _SOURCE_WEIGHT,
                "url":          url,
                "headline":     headline,
                "body":         body,
                "published_at": published_raw or datetime.now(timezone.utc).isoformat(),
                "scraped_at":   datetime.now(timezone.utc).isoformat(),
                "tickers":      nse_tickers,
                "sentiment":    None,
            }

            # Insert — skip if already in DB
            with get_connection() as conn:
                was_new = insert_doc(conn, doc)

            if not was_new:
                continue

            doc_id = doc["id"]

            # Augment tickers: merge MarketAux tags with rule/fuzzy linker
            linked = link_tickers(headline, body)
            merged_tickers = list(dict.fromkeys(nse_tickers + linked))

            # Score sentiment
            sentiment = score_document({"headline": headline, "body": body})

            # Persist enriched fields
            with get_connection() as conn:
                conn.execute(
                    "UPDATE documents SET tickers = ?, sentiment = ?, low_confidence = ? WHERE id = ?",
                    (
                        json.dumps(merged_tickers),
                        json.dumps(sentiment),
                        int(sentiment.get("low_confidence", False)),
                        doc_id,
                    ),
                )

            inserted_and_scored += 1
            logger.info(f"live_fetch: inserted+scored doc {doc_id[:12]}… for {symbol}")

        logger.info(f"live_fetch: {symbol} → {inserted_and_scored} new docs")
        return inserted_and_scored

    except Exception as exc:
        logger.error(f"live_fetch_for_ticker({symbol}) failed: {exc}")
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = live_fetch_for_ticker("RELIANCE")
    print(f"live_fetch smoke test: {n} new documents inserted and scored")
