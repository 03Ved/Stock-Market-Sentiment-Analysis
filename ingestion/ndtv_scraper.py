import hashlib
import json
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from config.settings import SOURCE_WEIGHTS, NDTV_URL, NDTV_REQUEST_DELAY
from db.database import get_connection, insert_doc

_SOURCE_ID = "ndtv_profit"
_WEIGHT = SOURCE_WEIGHTS[_SOURCE_ID]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Candidate CSS selectors for article cards — tried in order.
# NDTV Profit periodically updates its front-end; this list makes the scraper
# resilient to minor layout changes.
_CARD_SELECTORS = [
    "div.story-card",
    "div.news-card",
    "div.article-card",
    "li.story-list-item",
    "div[class*='StoryCard']",
    "div[class*='story-card']",
    "div[class*='newscard']",
    "article",
]

# Candidate selectors for the headline anchor inside a card
_HEADLINE_SELECTORS = ["h2 a", "h3 a", "h1 a", "a.story-title", "a[class*='title']", "a"]

# Candidate selectors for the timestamp
_TIME_SELECTORS = ["time", "span.date", "span[class*='date']", "span[class*='time']", "div[class*='date']"]


def _make_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _parse_timestamp(tag) -> str:
    """Extract an ISO 8601 UTC string from a timestamp tag, best-effort."""
    if tag is None:
        return datetime.now(timezone.utc).isoformat()

    # <time datetime="..."> is the cleanest path
    dt_attr = tag.get("datetime", "")
    if dt_attr:
        try:
            # Strip timezone suffix for naive parse, then mark UTC
            dt_str = dt_attr.replace("Z", "+00:00")
            return datetime.fromisoformat(dt_str).astimezone(timezone.utc).isoformat()
        except ValueError:
            pass

    return datetime.now(timezone.utc).isoformat()


def _find_cards(soup: BeautifulSoup) -> list:
    """Try each card selector and return the first non-empty result."""
    for sel in _CARD_SELECTORS:
        cards = soup.select(sel)
        if cards:
            return cards
    return []


def fetch_ndtv(url: str = NDTV_URL) -> int:
    """
    Scrape article headlines from NDTV Profit markets page.
    Returns number of new documents inserted.

    Note: ndtvprofit.com uses Cloudflare protection. This scraper will return 0
    when the server returns 403. Run behind a residential proxy or browser
    automation if consistent access is needed.
    """
    scraped_at = datetime.now(timezone.utc).isoformat()

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.RequestException as exc:
        print(f"[ndtv] Request failed — {exc}")
        return 0

    if resp.status_code == 403:
        print(
            f"[ndtv] 403 Forbidden — site is behind bot protection. "
            f"Skipping this cycle."
        )
        return 0

    if not resp.ok:
        print(f"[ndtv] HTTP {resp.status_code} — skipping.")
        return 0

    time.sleep(NDTV_REQUEST_DELAY)

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = _find_cards(soup)

    if not cards:
        # Fallback: collect all sufficiently long anchor texts that look like articles
        print("[ndtv] No cards found via selectors — falling back to anchor scan.")
        cards = soup.find_all("a", href=True)

    new_count = 0
    seen_urls: set = set()

    with get_connection() as conn:
        for card in cards:
            # Locate headline anchor
            anchor = None
            for sel in _HEADLINE_SELECTORS:
                anchor = card.select_one(sel) if hasattr(card, "select_one") else (
                    card if card.name == "a" else None
                )
                if anchor:
                    break

            if anchor is None:
                continue

            headline = anchor.get_text(strip=True)
            if not headline or len(headline) < 15:
                continue

            href = anchor.get("href", "")
            if not href:
                continue

            # Normalise relative URLs
            if href.startswith("/"):
                href = "https://www.ndtvprofit.com" + href
            if not href.startswith("http"):
                continue

            if href in seen_urls:
                continue
            seen_urls.add(href)

            time_tag = card.find(*_TIME_SELECTORS) if hasattr(card, "find") else None
            published_at = _parse_timestamp(time_tag)

            doc = {
                "id":            _make_id(href),
                "source_id":     _SOURCE_ID,
                "source_weight": _WEIGHT,
                "url":           href,
                "headline":      headline,
                "body":          None,   # no body — headline only for NDTV
                "published_at":  published_at,
                "scraped_at":    scraped_at,
                "tickers":       [],
                "sentiment":     None,
            }

            if insert_doc(conn, doc):
                new_count += 1

    print(f"[ndtv] {new_count} new / {len(seen_urls)} unique URLs parsed")
    return new_count


if __name__ == "__main__":
    from db.database import init_db
    init_db()
    total = fetch_ndtv()
    print(f"[ndtv] Done. New documents: {total}")
