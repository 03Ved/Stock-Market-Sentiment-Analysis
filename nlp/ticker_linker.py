"""
Rule-based NSE/BSE ticker linker.

Loads the NSE symbol master (data/nse_symbols.csv) and hand-curated aliases
(data/aliases.json) into two lookup structures, then scans document text for
direct symbol mentions and company name mentions.
"""

import csv
import json
import re
from pathlib import Path

from rapidfuzz import process, fuzz

_BASE = Path(__file__).resolve().parent.parent
_CSV_PATH     = _BASE / "data" / "nse_symbols.csv"
_ALIASES_PATH = _BASE / "data" / "aliases.json"

# ---------------------------------------------------------------------------
# Build lookup structures at import time (once per process)
# ---------------------------------------------------------------------------

def _load_symbol_master():
    symbol_set: set[str] = set()
    name_to_symbol: dict[str, str] = {}

    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym  = row["SYMBOL"].strip()
            name = row["NAME OF COMPANY"].strip()
            symbol_set.add(sym)
            # Map full company name → symbol (title-cased for matching)
            name_to_symbol[name.lower()] = sym

    # Overlay hand-curated aliases (these take priority for common short names)
    with open(_ALIASES_PATH, encoding="utf-8") as f:
        aliases: dict[str, str] = json.load(f)
    for alias, sym in aliases.items():
        name_to_symbol[alias.lower()] = sym

    return symbol_set, name_to_symbol


SYMBOL_SET, NAME_TO_SYMBOL = _load_symbol_master()

# Pre-sort alias keys longest-first so longer phrases match before sub-phrases
_NAME_KEYS_SORTED = sorted(NAME_TO_SYMBOL.keys(), key=len, reverse=True)

# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------

_UPPERCASE_TOKEN = re.compile(r"\b([A-Z][A-Z0-9\-&]{1,9})\b")

# Pre-compile word-boundary regex for every name/alias key (once at import)
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE), NAME_TO_SYMBOL[key])
    for key in _NAME_KEYS_SORTED
]


def link_tickers(headline: str, body: str | None = None) -> list[str]:
    """
    Scan headline (and optionally body) for NSE ticker mentions.

    Strategy:
      1. Direct symbol scan  — uppercase tokens 2–10 chars in SYMBOL_SET.
      2. Name/alias scan     — case-insensitive word-boundary match against
                               NAME_TO_SYMBOL keys, longest match first.
      3. Fuzzy match         — rapidfuzz token_sort_ratio on capitalized noun
                               phrases, score_cutoff=85.

    Returns a deduplicated list of NSE symbols (e.g. ["RELIANCE", "TCS"]).
    """
    text = headline + " " + (body[:500] if body else "")
    found: list[str] = []

    # --- Pass 1: direct uppercase symbol tokens ---
    for m in _UPPERCASE_TOKEN.finditer(text):
        token = m.group(1)
        if token in SYMBOL_SET:
            found.append(token)

    # --- Pass 2: company name / alias word-boundary scan ---
    for pattern, sym in _NAME_PATTERNS:
        if pattern.search(text):
            found.append(sym)

    # --- Pass 3: fuzzy match on capitalized noun phrases ---
    candidates = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    for candidate in candidates:
        if len(candidate) < 4:  # Skip very short matches
            continue
        result_fuzzy = process.extractOne(
            candidate,
            list(NAME_TO_SYMBOL.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85,
        )
        if result_fuzzy:
            match_name, score, _ = result_fuzzy
            found.append(NAME_TO_SYMBOL[match_name])

    return list(dict.fromkeys(found))  # Deduplicate preserving order


# ---------------------------------------------------------------------------
# Batch helper used by the NLP pipeline
# ---------------------------------------------------------------------------

def link_document(doc: dict) -> list[str]:
    """Convenience wrapper — accepts a unified-schema document dict."""
    return link_tickers(doc["headline"], doc.get("body"))


if __name__ == "__main__":
    tests = [
        ("Reliance Industries posts record quarterly profit", None),
        ("TCS and Infosys both beat estimates; HDFC Bank dips", None),
        ("Adani Ports expands capacity as JSW Steel eyes acquisition", None),
        ("SBI raises FD rates ahead of RBI meet", None),
        ("Zomato rebrands to Eternal, market cap crosses 2 lakh crore", None),
        ("No known tickers in this headline about monsoon forecast", None),
    ]
    print(f"Loaded {len(SYMBOL_SET)} symbols, {len(NAME_TO_SYMBOL)} name mappings\n")
    for headline, body in tests:
        tickers = link_tickers(headline, body)
        print(f"  {headline[:65]!r}")
        print(f"    -> {tickers}\n")
