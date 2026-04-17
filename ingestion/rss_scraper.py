import hashlib
from datetime import datetime, timezone

import feedparser
import requests

from config.settings import RSS_FEEDS, SOURCE_WEIGHTS
from db.database import get_connection, insert_doc


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def _make_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _parse_published(entry) -> str:
    """Return ISO 8601 UTC string from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def fetch_rss(source_id: str, url: str) -> int:
    """
    Fetch and store all entries from a single RSS feed.
    Returns the number of new documents inserted.
    """
    weight = SOURCE_WEIGHTS[source_id]
    scraped_at = datetime.now(timezone.utc).isoformat()

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except requests.RequestException as exc:
        print(f"[rss] {source_id}: request failed — {exc}")
        return 0

    entries = feed.get("entries", [])

    if not entries:
        print(f"[rss] {source_id}: no entries returned (status={resp.status_code})")
        return 0

    new_count = 0
    with get_connection() as conn:
        for entry in entries:
            link = entry.get("link", "")
            if not link:
                continue

            doc = {
                "id":           _make_id(link),
                "source_id":    source_id,
                "source_weight": weight,
                "url":          link,
                "headline":     entry.get("title", "").strip(),
                "body":         entry.get("summary", None),
                "published_at": _parse_published(entry),
                "scraped_at":   scraped_at,
                "tickers":      [],
                "sentiment":    None,
            }

            if not doc["headline"]:
                continue

            if insert_doc(conn, doc):
                new_count += 1

    print(f"[rss] {source_id}: {new_count} new / {len(entries)} fetched")
    return new_count


def fetch_all_rss() -> int:
    """Fetch all configured RSS feeds. Returns total new documents inserted."""
    total = 0
    for source_id, url in RSS_FEEDS.items():
        total += fetch_rss(source_id, url)
    return total


if __name__ == "__main__":
    from db.database import init_db
    init_db()
    total = fetch_all_rss()
    print(f"[rss] Done. Total new documents: {total}")
