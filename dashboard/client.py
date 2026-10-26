"""Shared dashboard/chat access, in-process by default or via an external API."""
import os

import requests


def get_data(path: str, **params) -> dict:
    base = os.getenv("API_BASE_URL", "").rstrip("/")
    if base:
        response = requests.get(f"{base}{path}", params=params, timeout=180)
        response.raise_for_status()
        return response.json()

    from api import main
    if path == "/health":
        return main.health()
    if path == "/search":
        return main.search(q=params["q"])
    if path.startswith("/explain/"):
        return main.explain_article(path.rsplit("/", 1)[-1])
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "ticker":
        if len(parts) == 3 and parts[2] == "history":
            return main.ticker_history(parts[1], hours=params.get("hours", 168),
                                       resolution=params.get("resolution", "auto"))
        return main.ticker_sentiment(parts[1], window=params.get("window", "24h"))
    raise ValueError(f"Unsupported data path: {path}")
