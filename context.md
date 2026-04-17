# Market Sentiment Analyzer — Indian Markets (NSE/BSE)
## Claude Code Project Context

> This file is the single source of truth for the project. Read it fully before writing any code.
> When asked to implement a specific part, refer back to the relevant section here.

---

## Project Overview

An academic NLP + deep learning project that collects financial news and social media content
about Indian stock market tickers (NSE/BSE), runs transformer-based sentiment analysis on it,
aggregates scores with source-reliability weighting and time decay, and serves a simple
dashboard showing per-ticker sentiment.

**Course context:** Partial fulfillment for a Deep Learning and Natural Language Processing course.
The DL requirement is met by fine-tuning FinBERT. The NLP requirement is met across the full
pipeline (tokenization, entity linking, classification, model comparison).

**Target market:** NSE and BSE (Indian exchanges). No SEC filings, no US-centric sources.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | |
| Data store | SQLite | Simple, no server, sufficient for academic scale |
| Scraping | `feedparser`, `requests`, `beautifulsoup4` | RSS-first, minimal scraping |
| Reddit | `praw` | Official API wrapper, free read-only |
| Scheduling | `apscheduler` | Cron-like scheduling inside Python process |
| NLP / DL | `transformers`, `torch` | HuggingFace FinBERT |
| Backend API | `fastapi` + `uvicorn` | Lightweight REST layer |
| Dashboard | `streamlit` | Fast to build, good for academic demos |
| Env management | `python-dotenv` | Credentials out of code |

---

## Repository Structure

```
market-sentiment/
├── .env                        # credentials — never commit
├── .env.example                # template to commit
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py             # SOURCE_WEIGHTS, RSS_FEEDS, constants
│
├── data/
│   └── nse_symbols.csv         # NSE symbol master (downloaded once)
│
├── db/
│   ├── schema.sql              # table definitions
│   └── database.py             # connection helper, init function
│
├── ingestion/
│   ├── __init__.py
│   ├── rss_scraper.py          # fetch_rss()
│   ├── reddit_scraper.py       # fetch_reddit()
│   ├── ndtv_scraper.py         # fetch_ndtv() — no RSS, BS4 scrape
│   └── scheduler.py            # APScheduler setup, scrape_all()
│
├── nlp/
│   ├── __init__.py
│   ├── ticker_linker.py        # rule-based NSE/BSE entity linking
│   ├── sentiment.py            # FinBERT inference
│   ├── fine_tune.py            # fine-tuning script (run once)
│   └── baseline.py             # VADER baseline for comparison
│
├── aggregation/
│   ├── __init__.py
│   └── aggregator.py           # score_ticker(), time-decay, weighting
│
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI app, endpoints
│
└── dashboard/
    └── app.py                  # Streamlit dashboard
```

---

## Environment Variables

The `.env` file must contain:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=market_sentiment_academic/0.1
DB_PATH=./market_sentiment.db
```

The `.env.example` file should contain the same keys with empty values. Commit only `.env.example`.

---

## Part 1 — Data Ingestion

### Unified Document Schema

Every source must produce records matching this shape before storage.
Downstream parts (NLP, aggregation) consume only this schema — never raw source data.

```python
{
  "id":             str,   # sha256 of url — dedup key
  "source_id":      str,   # e.g. "et_markets"
  "source_weight":  float, # reliability weight from config
  "url":            str,
  "headline":       str,
  "body":           str | None,  # may be None for RSS summary-only entries
  "published_at":   str,  # ISO 8601, UTC
  "scraped_at":     str,  # ISO 8601, UTC
  "tickers":        list,  # [] initially, filled by Part 2
  "sentiment":      None   # filled by Part 3
}
```

### Source Weights

Defined in `config/settings.py`. These are fixed, manually assigned — not learned.
Rationale: higher for outlets with professional editorial standards.

```python
SOURCE_WEIGHTS = {
    "et_markets":          0.90,
    "livemint":            0.88,
    "business_standard":   0.85,
    "moneycontrol":        0.80,
    "ndtv_profit":         0.70,
    "reddit_india_invest": 0.40,
    "reddit_indian_stock": 0.35,
    "reddit_sensex":       0.30,
}
```

### RSS Feeds

```python
RSS_FEEDS = {
    "et_markets":        "https://economictimes.indiatimes.com/markets/rss.cms",
    "livemint":          "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "moneycontrol":      "https://www.moneycontrol.com/rss/marketnews.xml",
}
```

`feedparser.parse(url)` handles all of these. Each entry provides `.title`, `.link`, `.summary`,
`.published`. Use `.summary` as `body`. If `published` is missing, fall back to `datetime.utcnow()`.

### NDTV Profit

No RSS feed. Scrape with `requests` + `BeautifulSoup`. Target:
`https://www.ndtvprofit.com/markets`

Parse article cards — each has a headline anchor tag and a timestamp. Extract headline + URL only
(no body needed). Same schema output, `source_id = "ndtv_profit"`, `weight = 0.70`.

Respect a 2-second delay between requests. Add a realistic `User-Agent` header.

### Reddit

Three subreddits. Use PRAW read-only mode (no login required beyond client credentials).

```python
SUBREDDITS = {
    "reddit_india_invest": ("IndiaInvestments",   0.40),
    "reddit_indian_stock": ("IndianStockMarket",  0.35),
    "reddit_sensex":       ("Sensex",             0.30),
}
```

Fetch `.new(limit=50)` per subreddit — recency matters, not engagement rank.
Use post title as `headline`. Use `post.selftext` as `body` only if `len(selftext) > 20`.
`published_at` = `datetime.utcfromtimestamp(post.created_utc).isoformat()`.

Do not ingest comments. Too noisy, insufficient signal.

### Deduplication

Use SQLite's `INSERT OR IGNORE` on the `id` (sha256 of url) column.
A story appearing on multiple outlets gets stored once — whichever arrives first wins.
No additional dedup logic needed at this scale.

### Scheduling

Run `scrape_all()` every 15 minutes on weekdays during NSE market hours (09:00–16:00 IST).
Run every 60 minutes outside market hours and on weekends.

```python
scheduler.add_job(scrape_all, "cron",
    day_of_week="mon-fri",
    hour="9-16", minute="*/15", timezone="Asia/Kolkata")

scheduler.add_job(scrape_all, "cron",
    day_of_week="mon-fri",
    hour="0-8,16-23", minute="0", timezone="Asia/Kolkata")

scheduler.add_job(scrape_all, "cron",
    day_of_week="sat,sun",
    minute="0", timezone="Asia/Kolkata")
```

---

## Part 2 — Ticker Linking (NER)

Rule-based. No trained model — the NSE/BSE symbol universe is bounded and knowable.

### Symbol Master

Download `EQUITY_L.csv` from https://www.nseindia.com/market-data/securities-available-for-trading

This CSV contains `SYMBOL`, `NAME OF COMPANY`, `SERIES`, `ISIN NUMBER` columns.
Parse it into two lookup dicts:

```python
SYMBOL_SET:   set of strings  # {"RELIANCE", "TCS", "INFY", ...}
NAME_TO_SYMBOL: dict          # {"Reliance Industries": "RELIANCE",
                              #  "Tata Consultancy Services": "TCS", ...}
```

The `name_to_symbol` dict must also include common short names and abbreviations:
- "TCS" → "TCS" (already a symbol)
- "Infosys" → "INFY"
- "Reliance" → "RELIANCE"
- "HDFC Bank" → "HDFCBANK"
- "Wipro" → "WIPRO"

These common-name mappings supplement the CSV — maintain a small hand-curated `aliases.json`
for the top 50 most-mentioned companies.

### Linking Logic

For each document, scan `headline + " " + (body or "")`:

1. Check for direct symbol mentions: uppercase tokens 2–10 chars that are in `SYMBOL_SET`
2. Check for company name mentions: sliding window over tokens matched against `NAME_TO_SYMBOL`
3. Normalize all matches to their exchange symbol
4. Deduplicate the result list

Store as a JSON array in the `tickers` column: `'["RELIANCE", "TCS"]'`

Documents with zero ticker matches are kept but deprioritized during aggregation
(they contribute to market-wide sentiment but not ticker-specific sentiment).

---

## Part 3 — Sentiment Analysis

### Model: FinBERT

Checkpoint: `ProsusAI/finbert` on HuggingFace.
Output: probability triple `(positive, negative, neutral)`.
Always retain raw probabilities — do not collapse to a label.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

def score_text(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    labels = ["positive", "negative", "neutral"]
    # finbert label order: positive=0, negative=1, neutral=2
    return {
        "positive": probs[0],
        "negative": probs[1],
        "neutral":  probs[2],
        "score":    probs[0] - probs[1],  # scalar in [-1, 1]
        "confidence": max(probs)
    }
```

The `score` field (`positive - negative`) is the scalar that aggregation consumes.
`confidence` = `max(probs)`. Documents with confidence < 0.55 are stored but flagged
as low-confidence and down-weighted during aggregation.

### Input text

Feed `headline` only by default. If `body` is present and headline confidence < 0.6,
concatenate `headline + ". " + body[:400]` and re-score.

### Fine-tuning (academic contribution)

Fine-tune the off-the-shelf FinBERT checkpoint on a small labeled dataset of
Indian financial news headlines.

**Dataset construction:**
- Pull ~500–800 headlines from your stored documents
- Label manually as positive / negative / neutral using a simple annotation tool or spreadsheet
- Indian financial headlines often have Hinglish terms ("bazaar", "loot sale", "bharat") —
  label these; vanilla FinBERT trained on US text handles them poorly
- 70/15/15 train/val/test split

**Fine-tuning script** (`nlp/fine_tune.py`):
- Use HuggingFace `Trainer` API
- `AdamW`, lr=2e-5, 3 epochs, batch size 16
- Save best checkpoint by validation F1
- Log train/val loss curves — include in project report

**Baseline comparison** (`nlp/baseline.py`):
- VADER (rule-based lexicon, `vaderSentiment` library)
- Apply to same test set
- Report accuracy, F1 (macro), and confusion matrices for both VADER and FinBERT
- This comparison is the academic evaluation section

---

## Part 4 — Aggregation

### Per-ticker sentiment score

For a given ticker `T` and time window `W` (e.g. last 24h):

1. Fetch all documents where `tickers` contains `T` and `published_at` is within `W`
2. For each document compute an effective weight:

```python
import math
from datetime import datetime, timezone

def effective_weight(doc, now, lambda_decay=0.3):
    age_hours = (now - doc["published_at"]).total_seconds() / 3600
    time_weight = math.exp(-lambda_decay * age_hours)
    confidence_weight = doc["sentiment"]["confidence"]
    return doc["source_weight"] * time_weight * confidence_weight
```

3. Compute weighted average score:

```python
def aggregate_ticker(docs, now):
    weights = [effective_weight(d, now) for d in docs]
    scores  = [d["sentiment"]["score"] for d in docs]
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    weighted_score = sum(w * s for w, s in zip(weights, scores)) / total_weight
    return {
        "score":        round(weighted_score, 4),   # in [-1, 1]
        "doc_count":    len(docs),
        "total_weight": round(total_weight, 4),
        "confidence":   "high" if len(docs) >= 5 else "low"
    }
```

4. Return `None` if fewer than 2 documents exist for the ticker in the window.
   Surface this as "insufficient data" in the dashboard — never show a score based on 1 article.

### Time windows

Compute for two windows: 24h and 7d. Both shown in the dashboard.
7d score uses a smaller `lambda_decay` (0.05) so older articles still meaningfully contribute.

### Source weight rationale (for project report)

| Source | Weight | Justification |
|---|---|---|
| ET Markets | 0.90 | Largest Indian financial news outlet, professional editorial |
| LiveMint | 0.88 | Backed by HT Media, dedicated markets desk |
| Business Standard | 0.85 | Strong institutional readership |
| MoneyControl | 0.80 | High traffic, some advertorial content |
| NDTV Profit | 0.70 | Broad coverage, less specialist |
| Reddit (IndiaInvestments) | 0.40 | Community content, some expert users |
| Reddit (IndianStockMarket) | 0.35 | More retail-focused |
| Reddit (Sensex) | 0.30 | Smallest, most speculative |

---

## Part 5 — API and Dashboard

### FastAPI endpoints (`api/main.py`)

```
GET /ticker/{symbol}
    ?window=24h|7d
    Returns: aggregated sentiment score, doc_count, top 5 source articles

GET /ticker/{symbol}/history
    ?days=7
    Returns: daily sentiment scores for sparkline chart

GET /search?q={query}
    Returns: list of matching ticker symbols (for autocomplete)

GET /health
    Returns: {"status": "ok", "doc_count": N}
```

### Streamlit Dashboard (`dashboard/app.py`)

Single-page app with:

1. **Ticker search bar** — autocomplete against symbol master
2. **Sentiment gauge** — a number in [-1, 1] color-coded (green > 0.1, red < -0.1, gray otherwise)
3. **24h vs 7d toggle** — switch between time windows
4. **Source breakdown** — bar chart of contribution by source (source_weight × doc_count)
5. **Recent articles list** — top 5 documents driving the score, with headline, source, timestamp, individual sentiment
6. **7-day sparkline** — line chart of daily sentiment score

---

## Database Schema (`db/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    source_weight   REAL NOT NULL,
    url             TEXT NOT NULL,
    headline        TEXT NOT NULL,
    body            TEXT,
    published_at    TEXT NOT NULL,
    scraped_at      TEXT NOT NULL,
    tickers         TEXT NOT NULL DEFAULT '[]',
    sentiment       TEXT,
    low_confidence  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_published_at ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_source_id    ON documents(source_id);

CREATE TABLE IF NOT EXISTS ticker_sentiment_cache (
    ticker      TEXT NOT NULL,
    window      TEXT NOT NULL,       -- "24h" or "7d"
    score       REAL,
    doc_count   INTEGER,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (ticker, window)
);
```

---

## Implementation Order for Claude Code

Work in this exact sequence. Do not skip ahead.

1. `requirements.txt` and `.env.example`
2. `config/settings.py` — SOURCE_WEIGHTS, RSS_FEEDS, constants
3. `db/schema.sql` and `db/database.py` — schema creation, connection helper
4. `ingestion/rss_scraper.py` — fetch_rss(), insert_doc()
5. `ingestion/reddit_scraper.py` — fetch_reddit()
6. `ingestion/ndtv_scraper.py` — fetch_ndtv() with BS4
7. `ingestion/scheduler.py` — scrape_all(), APScheduler config
8. `data/` — download and commit nse_symbols.csv + aliases.json
9. `nlp/ticker_linker.py` — load symbol master, link_tickers()
10. `nlp/sentiment.py` — load FinBERT, score_text(), batch_score()
11. `nlp/baseline.py` — VADER wrapper matching same interface
12. `nlp/fine_tune.py` — Trainer-based fine-tuning script
13. `aggregation/aggregator.py` — effective_weight(), aggregate_ticker()
14. `api/main.py` — FastAPI app with all endpoints
15. `dashboard/app.py` — Streamlit dashboard

Test each file before moving to the next. Run ingestion for at least one cycle and verify
documents are appearing in the DB before building the NLP layer on top.

---

## Key Constraints and Decisions to Remember

- **Headline-first NLP.** Score headlines by default. Use body only as fallback.
- **Weights travel with documents.** `source_weight` is stored in every row and used at aggregation, not at ingestion.
- **Confidence gating.** Never surface a ticker score to the UI if doc_count < 2 or all documents are low_confidence.
- **UTC everywhere internally.** Convert to IST only in the dashboard display layer.
- **RSS-first.** Prefer feedparser over BeautifulSoup scraping wherever possible. Less brittle.
- **SQLite is fine.** Do not introduce PostgreSQL or any server-side DB. This is a single-machine academic project.
- **No Twitter/X.** API access is too restricted. Note as a scope limitation in the report.
- **FinBERT label order:** index 0 = positive, 1 = negative, 2 = neutral. Verify this against the model config before using.
- **NSE symbol master needs periodic refresh.** Note this in README; for the project, downloading once is fine.