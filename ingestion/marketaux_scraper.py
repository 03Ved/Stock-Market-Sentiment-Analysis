"""
Fetches financial news from the MarketAux API.
MarketAux returns pre-tagged entities with exchange identifiers,
so ticker linkage is provided directly for NSE-listed stocks.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

import requests

from config.settings import MARKETAUX_API_KEY, MARKETAUX_BASE_URL, SOURCE_WEIGHTS
from db.database import insert_doc, get_connection

logger = logging.getLogger(__name__)


def fetch_marketaux(tickers: list[str] | None = None, limit: int = 50) -> int:
    """
    Fetch Indian financial news from MarketAux.

    Args:
        tickers: Optional list of NSE symbols to filter by. If None, fetches all Indian news.
        limit: Max articles to fetch (MarketAux caps at 100 per request on free tier).

    Returns:
        Number of new documents inserted.
    """
    if not MARKETAUX_API_KEY:
        logger.warning("MARKETAUX_API_KEY not set. Skipping MarketAux fetch.")
        return 0

    params = {
        "api_token": MARKETAUX_API_KEY,
        "countries": "in",
        "language": "en",
        "limit": min(limit, 100),
        "filter_entities": "true",
    }
    if tickers:
        params["symbols"] = ",".join(f"NSE:{t}" for t in tickers)

    try:
        resp = requests.get(MARKETAUX_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"MarketAux API request failed: {e}")
        return 0

    data = resp.json()
    articles = data.get("data", [])
    if not articles:
        logger.info("MarketAux returned 0 articles.")
        return 0

    inserted = 0
    with get_connection() as conn:
        for article in articles:
            url = article.get("url", "")
            if not url:
                continue

            # Extract NSE tickers from MarketAux entity tags
            nse_tickers = []
            for entity in article.get("entities", []):
                symbol = entity.get("symbol", "")
                exchange = entity.get("exchange", "")
                if exchange == "NSE" and symbol:
                    clean_symbol = symbol.replace("NSE:", "").strip()
                    if clean_symbol:
                        nse_tickers.append(clean_symbol)

            published_raw = article.get("published_at", "")
            # MarketAux timestamps are ISO 8601; ensure UTC suffix
            if published_raw and not published_raw.endswith("Z") and "+" not in published_raw:
                published_raw += "Z"

            doc = {
                "id": hashlib.sha256(url.encode()).hexdigest(),
                "source_id": "marketaux",
                "source_weight": SOURCE_WEIGHTS.get("marketaux", 0.75),
                "url": url,
                "headline": article.get("title", ""),
                "body": article.get("description"),
                "published_at": published_raw or datetime.now(timezone.utc).isoformat(),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "tickers": nse_tickers,  # Pre-tagged by MarketAux
                "sentiment": None,
            }

            if not doc["headline"]:
                continue

            try:
                was_new = insert_doc(conn, doc)
                if was_new:
                    inserted += 1
            except Exception as e:
                logger.error(f"Failed to insert MarketAux doc {url}: {e}")
                continue

    logger.info(f"MarketAux: fetched {len(articles)}, inserted {inserted} new")
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = fetch_marketaux()
    print(f"MarketAux smoke test: {count} new documents inserted")
