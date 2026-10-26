"""
Google News RSS scraper for Indian stock market news.

Fetches headlines from Google News RSS feeds using multiple search queries
per ticker to get diverse, opinion-rich results instead of generic roundups.
No API key required, no rate limits, works for all NSE stocks.

Optional integrations (all fall back silently if keys are absent):
  - GEMINI_API_KEY: AI-powered query generation and headline relevance filtering
  - GNEWS_API_KEY:  Supplementary articles from GNews API (100 req/day free tier)
"""

import hashlib
import os
import json
import logging
import re
import time
import urllib.parse
import email.utils
from datetime import datetime, timezone

import feedparser
import requests

from db.database import get_connection, insert_doc
from nlp.sentiment import score_document

logger = logging.getLogger(__name__)

# Headlines containing these phrases are generic roundups that score neutral — skip them
_SKIP_PATTERNS = re.compile(
    r"most traded|top traded|top gainers|top losers|stocks to watch|stocks to buy|"
    r"stocks in focus|stocks in news|stocks in the news|check out top|these are among|"
    r"among most traded|market wrap|market summary|market roundup|buzzing stocks",
    re.IGNORECASE,
)

# Hardcoded fallback queries — 8 diverse angles covering analysis + price + results
_FALLBACK_SUFFIX = [
    "stock analysis",
    "share price outlook",
    "quarterly results",
    "buy sell recommendation",
    "target price",
    "earnings growth",
    "investor sentiment",
    "market news today",
]


def _clean_headline(title: str) -> str:
    """Strip the ' - SourceName' suffix from Google News headlines."""
    if " - " in title:
        parts = title.split(" - ")
        if len(parts) > 1:
            return " - ".join(parts[:-1]).strip()
    return title.strip()


def _strip_html(text: str) -> str:
    """Remove HTML tags from RSS summary text."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", text).strip()


def _build_fallback_queries(name: str, symbol: str) -> list[str]:
    """Return 8 hardcoded queries plus 2 symbol-based queries, deduplicated."""
    queries = [f"{name} {suffix}" for suffix in _FALLBACK_SUFFIX]
    # Symbol-based extras catch articles that use the ticker code directly
    queries += [f"{symbol} NSE stock", f"{symbol} share price"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def generate_smart_queries(symbol: str, company_name: str) -> list[str]:
    """
    Use Gemini to generate targeted search queries that will find
    opinionated/analytical articles rather than generic roundups.

    When Gemini is available: returns up to 8 queries (4 Gemini + 4 hardcoded),
    deduplicated and capped at 8.
    When Gemini is unavailable: returns 10 hardcoded queries (8 name-based + 2
    symbol-based).

    Never raises.
    """
    import os
    api_key = os.getenv("GEMINI_API_KEY", "")
    name = company_name or symbol

    if not api_key or os.getenv("ENABLE_GEMINI_ENRICHMENT", "0") != "1":
        return _build_fallback_queries(name, symbol)[:3]

    try:
        from agent.gemini import generate_text

        prompt = f"""Generate exactly 4 Google News search queries to find opinionated financial articles about {company_name or symbol} ({symbol}), an Indian stock listed on NSE.

Rules:
- Queries should find articles with STRONG sentiment (bullish/bearish opinions, analyst calls, earnings reactions)
- Do NOT generate queries that would find generic "most traded stocks today" lists
- Each query should be 3-6 words
- Include the company name in each query
- Focus on: analyst ratings, earnings, management decisions, sector outlook, controversies

Return ONLY the 4 queries, one per line, no numbering, no explanation."""

        response_text = generate_text(prompt)
        gemini_queries = [q.strip() for q in response_text.strip().split("\n") if q.strip()]
        gemini_queries = [q for q in gemini_queries if 5 < len(q) < 80][:4]

        if len(gemini_queries) >= 2:
            logger.info(f"Gemini generated {len(gemini_queries)} queries for {symbol}: {gemini_queries}")

            # Append 4 hardcoded queries to ensure coverage even if Gemini picks unusual angles
            hardcoded = [f"{name} {suffix}" for suffix in _FALLBACK_SUFFIX[:4]]
            combined = gemini_queries + hardcoded

            # Deduplicate, preserve order, cap at 8
            seen: set[str] = set()
            unique: list[str] = []
            for q in combined:
                key = q.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(q)
            return unique[:8]

    except Exception as exc:
        logger.warning(f"Gemini query generation failed for {symbol}: {exc}")

    return _build_fallback_queries(name, symbol)[:3]


def filter_relevant_headlines(headlines: list[str], symbol: str, company_name: str) -> set[int]:
    """
    Use Gemini to filter which headlines are actually relevant and opinionated
    about the target stock.

    Returns set of indices to KEEP.
    Falls back to keeping all headlines if Gemini is unavailable.
    """
    import os
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key or os.getenv("ENABLE_GEMINI_ENRICHMENT", "0") != "1" or len(headlines) == 0:
        return set(range(len(headlines)))

    try:
        from agent.gemini import generate_text

        numbered = "\n".join(f"{i}: {h}" for i, h in enumerate(headlines))
        prompt = f"""Below are news headlines fetched for the stock {company_name or symbol} ({symbol}).

{numbered}

Return ONLY the numbers of headlines that:
1. Are specifically about {company_name or symbol} (not just mentioning it in a list)
2. Are substantive company news, analysis, or forward-looking statements
3. May be positive, negative, OR neutral; never exclude a headline for neutral sentiment

Return the numbers as comma-separated values. If none qualify, return "NONE".
No explanation needed."""

        response_text = generate_text(prompt)
        text = response_text.strip()

        if text == "NONE":
            return set()

        indices: set[int] = set()
        for part in text.replace(" ", "").split(","):
            try:
                idx = int(part)
                if 0 <= idx < len(headlines):
                    indices.add(idx)
            except ValueError:
                continue

        logger.info(f"Gemini kept {len(indices)}/{len(headlines)} headlines for {symbol}")
        return indices if indices else set(range(len(headlines)))

    except Exception as exc:
        logger.warning(f"Gemini headline filter failed for {symbol}: {exc}")
        return set(range(len(headlines)))


def fetch_gnews_for_ticker(symbol: str, company_name: str | None = None) -> int:
    """
    Fetch articles from GNews API as a supplementary source.
    Free tier: 100 requests/day, 10 articles per request.

    Returns number of newly inserted and scored documents.
    Returns 0 silently if GNEWS_API_KEY is not set or on any error.
    """
    from config.settings import GNEWS_API_KEY
    if not GNEWS_API_KEY:
        return 0

    search_term = company_name or symbol
    params = {
        "q":      f"{search_term} stock",
        "lang":   "en",
        "country": "in",
        "max":    10,
        "apikey": GNEWS_API_KEY,
    }

    try:
        resp = requests.get("https://gnews.io/api/v4/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"GNews API failed for {symbol}: {exc}")
        return 0

    articles = data.get("articles", [])
    inserted = 0

    for article in articles:
        article_url = article.get("url", "")
        title = article.get("title", "")
        if not article_url or not title:
            continue

        # Apply same skip-phrase filter as the RSS scraper
        if _SKIP_PATTERNS.search(title):
            logger.debug(f"GNews: skipping roundup headline: {title[:60]}")
            continue

        doc = {
            "id":           hashlib.sha256(article_url.encode()).hexdigest(),
            "source_id":    "gnews",
            "source_weight": 0.70,
            "url":          article_url,
            "headline":     title,
            "body":         article.get("description"),
            "published_at": article.get("publishedAt", datetime.now(timezone.utc).isoformat()),
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
            "tickers":      [symbol],
            "sentiment":    None,
        }

        try:
            with get_connection() as conn:
                was_new = insert_doc(conn, doc)

            if not was_new:
                continue

            result = score_document({"headline": title, "body": doc["body"]})

            with get_connection() as conn:
                conn.execute(
                    "UPDATE documents SET sentiment = ?, low_confidence = ? WHERE id = ?",
                    (
                        json.dumps(result),
                        int(result.get("low_confidence", False)),
                        doc["id"],
                    ),
                )

            inserted += 1
            logger.info(
                f"gnews: scored doc {doc['id'][:12]}… "
                f"score={result.get('score', 0):+.3f} for {symbol}"
            )

        except Exception as exc:
            logger.error(f"GNews insert/score failed for '{title[:40]}': {exc}")
            continue

    logger.info(f"gnews: fetched {len(articles)}, inserted {inserted} for {symbol}")
    return inserted


def fetch_google_news_for_ticker(symbol: str, company_name: str = None) -> int:
    """
    Fetch news from Google News RSS using multiple queries for diverse coverage,
    then supplement with GNews API if configured.

    Pipeline:
      1. Generate queries (Gemini if available, else 10 hardcoded)
      2. Fetch RSS feeds (up to 10 entries per query), apply skip-phrase filter
      3. Gemini relevance filter on collected headlines (optional second pass)
      4. Insert and score surviving entries
      5. Supplement with GNews API (optional)

    Args:
        symbol: NSE ticker symbol, e.g. 'RELIANCE'.
        company_name: Optional human-readable company name for better results.

    Returns:
        Total number of newly inserted and scored documents.
        Returns 0 on any error. Never raises to the caller.
    """
    try:
        queries = generate_smart_queries(symbol, company_name or symbol)

        # ── Pass 1: collect entries from all RSS feeds, apply skip filter ──────
        all_entries: list[tuple[str, object]] = []  # (headline, feedparser entry)
        inserted_count = 0

        for qi, query in enumerate(queries[:3]):
            encoded_query = urllib.parse.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

            logger.info(f"Fetching Google News RSS ({qi+1}/{len(queries)}) for {symbol}: q={query}")

            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except Exception as exc:
                logger.warning(f"Failed to parse feed for query '{query}': {exc}")
                continue

            for entry in feed.entries[:10]:  # Up to 10 per query
                try:
                    headline = _clean_headline(entry.title)

                    if _SKIP_PATTERNS.search(headline):
                        logger.debug(f"Skipping roundup headline: {headline[:60]}")
                        continue

                    all_entries.append((headline, entry))

                except Exception as exc:
                    logger.warning(f"Error reading entry: {exc}")
                    continue

            if qi < len(queries) - 1:
                time.sleep(0.5)

        if not all_entries:
            logger.info(f"google_news: no entries survived skip filter for {symbol}")
        else:
            # ── Pass 2: Gemini relevance filter ───────────────────────────────────
            headlines_only = [h for h, _ in all_entries]
            keep_indices = filter_relevant_headlines(headlines_only, symbol, company_name or symbol)

            # ── Pass 3: insert and score surviving entries ─────────────────────────
            inserted_count = 0

            for i, (headline, entry) in enumerate(all_entries[:24]):
                if i not in keep_indices:
                    logger.debug(f"Gemini filtered out: {headline[:60]}")
                    continue

                try:
                    doc_id = hashlib.sha256(entry.link.encode()).hexdigest()
                    body = _strip_html(entry.get("summary", None))

                    try:
                        pub_dt = email.utils.parsedate_to_datetime(entry.published)
                        published_at = pub_dt.astimezone(timezone.utc).isoformat()
                    except Exception:
                        published_at = datetime.now(timezone.utc).isoformat()

                    doc = {
                        "id":           doc_id,
                        "source_id":    "google_news",
                        "source_weight": 0.70,
                        "url":          entry.link,
                        "headline":     headline,
                        "body":         body,
                        "published_at": published_at,
                        "scraped_at":   datetime.now(timezone.utc).isoformat(),
                        "tickers":      [symbol],
                        "sentiment":    None,
                    }

                    with get_connection() as conn:
                        existing = conn.execute("SELECT sentiment FROM documents WHERE id = ?", (doc_id,)).fetchone()
                    if existing is not None and existing["sentiment"] is not None:
                        continue
                    result = score_document({"headline": headline, "body": body})
                    doc["sentiment"] = result
                    with get_connection() as conn:
                        insert_doc(conn, doc)

                    with get_connection() as conn:
                        conn.execute(
                            "UPDATE documents SET sentiment = ?, low_confidence = ? WHERE id = ?",
                            (
                                json.dumps(result),
                                int(result.get("low_confidence", False)),
                                doc_id,
                            ),
                        )

                    inserted_count += 1
                    logger.info(
                        f"google_news: scored doc {doc_id[:12]}… "
                        f"score={result.get('score', 0):+.3f} for {symbol}"
                    )

                except Exception as exc:
                    logger.warning(f"Error processing entry '{headline[:40]}': {exc}")
                    continue

            logger.info(f"google_news: {symbol} → {inserted_count} new docs from RSS")

        # ── Step 4: supplement with GNews API ─────────────────────────────────
        gnews_count = 0
        try:
            gnews_count = fetch_gnews_for_ticker(symbol, company_name)
        except Exception as exc:
            logger.warning(f"GNews supplementary fetch failed for {symbol}: {exc}")

        total_inserted = inserted_count + gnews_count
        logger.info(f"google_news+gnews: {symbol} → {total_inserted} total new docs")
        return total_inserted

    except Exception as exc:
        logger.error(f"fetch_google_news_for_ticker({symbol}) failed: {exc}", exc_info=True)
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = fetch_google_news_for_ticker("IDEA", "Vodafone Idea")
    print(f"IDEA: {count} articles")
    count = fetch_google_news_for_ticker("RELIANCE", "Reliance Industries")
    print(f"RELIANCE: {count} articles")
    count = fetch_google_news_for_ticker("HDFCBANK", "HDFC Bank")
    print(f"HDFCBANK: {count} articles")
