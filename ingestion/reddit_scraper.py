import requests, hashlib
import time
from datetime import datetime, timezone

from config.settings import SUBREDDITS as _CFG_SUBREDDITS, REDDIT_USER_AGENT
from db.database import get_connection, insert_doc

HEADERS = {"User-Agent": REDDIT_USER_AGENT or "market_sentiment_academic/0.1"}

# Kept for backwards compatibility — scheduler uses fetch_all_reddit()
SUBREDDITS = {
    "reddit_india_invest": ("IndiaInvestments",   0.40),
    "reddit_indian_stock": ("IndianStockMarket",  0.35),
    "reddit_sensex":       ("Sensex",             0.30),
}


def fetch_reddit(source_id, subreddit_name, weight, limit=50):
    """Fetch new posts and return them as a list of unified-schema dicts."""
    url = f"https://www.reddit.com/r/{subreddit_name}/new.json?limit={limit}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[reddit] {source_id}: request failed — {exc}")
        return []

    posts = resp.json()["data"]["children"]
    scraped_at = datetime.now(timezone.utc).isoformat()

    docs = []
    for post in posts:
        p = post["data"]
        body = p.get("selftext", "").strip()
        # Use permalink so self-posts get a stable, canonical URL
        permalink = p.get("permalink", "")
        post_url = f"https://www.reddit.com{permalink}" if permalink else p.get("url", "")
        if not post_url:
            continue
        doc_id = hashlib.sha256(post_url.encode()).hexdigest()
        docs.append({
            "id":            doc_id,
            "source_id":     source_id,
            "source_weight": weight,
            "url":           post_url,
            "headline":      p.get("title", "").strip(),
            "body":          body if len(body) > 20 else None,
            "published_at":  datetime.fromtimestamp(
                                 p["created_utc"], tz=timezone.utc
                             ).isoformat(),
            "scraped_at":    scraped_at,
            "tickers":       [],
            "sentiment":     None,
        })
    return docs


def fetch_all_reddit() -> int:
    """Fetch all configured subreddits and insert into DB. Returns new doc count."""
    total = 0
    for source_id, (subreddit, weight) in _CFG_SUBREDDITS.items():
        docs = fetch_reddit(source_id, subreddit, weight)
        with get_connection() as conn:
            new = sum(insert_doc(conn, d) for d in docs if d["headline"])
        print(f"[reddit] {source_id} (r/{subreddit}): {new} new / {len(docs)} fetched")
        total += new
        time.sleep(2)
    return total