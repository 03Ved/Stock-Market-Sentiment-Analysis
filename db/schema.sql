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
