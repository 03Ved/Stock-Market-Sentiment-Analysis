"""
Sentiment-aware chatbot using Google Gemini API.

Answers questions about stock market sentiment using live data from
the Market Sentiment API. Provides context-aware responses with
real-time sentiment scores and article analysis.

Usage:
    from agent.chat import chat
    response = chat("How is Reliance doing?", [])
"""

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_ticker_data(symbol: str, window: str = "7d") -> dict:
    """Fetch sentiment data for a ticker from our API."""
    try:
        r = requests.get(
            f"{API_BASE}/ticker/{symbol}",
            params={"window": window},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        logger.error(f"Failed to fetch ticker data for {symbol}: {exc}")
    return {"error": f"Could not fetch data for {symbol}"}


def search_ticker(query: str) -> list[str]:
    """Search for ticker symbols matching a query string."""
    try:
        r = requests.get(f"{API_BASE}/search", params={"q": query}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [item["symbol"] for item in data.get("results", [])[:5]]
    except Exception as exc:
        logger.debug(f"Ticker search failed for '{query}': {exc}")
    return []


def build_context(user_message: str) -> str:
    """
    Extract ticker references from user message, fetch their sentiment data,
    and build a context string for the LLM.
    """
    words = user_message.upper().split()
    context_parts = []
    seen_symbols = set()

    for word in words:
        cleaned = word.strip("?.,!:;'\"()[]")
        if len(cleaned) < 2 or not cleaned.replace("&", "").replace("-", "").isalpha():
            continue
        results = search_ticker(cleaned)
        if not results:
            continue

        symbol = results[0]
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)

        data = get_ticker_data(symbol, "7d")
        if "error" in data:
            continue

        score = data.get("score")
        doc_count = data.get("doc_count", 0)
        confidence = data.get("confidence", "unknown")
        articles = data.get("articles", [])

        article_summaries = ""
        for a in articles[:3]:
            s = a.get("sentiment", {}).get("score", "N/A")
            if isinstance(s, float):
                s = f"{s:+.3f}"
            article_summaries += (
                f"\n  - \"{a.get('headline', 'N/A')}\" "
                f"(score: {s}, source: {a.get('source_id', 'N/A')})"
            )

        score_str = f"{score:+.3f}" if isinstance(score, (int, float)) else "N/A"
        context_parts.append(
            f"Data for {symbol} (7-day window):\n"
            f"  Sentiment score: {score_str} (range: -1 bearish to +1 bullish)\n"
            f"  Based on {doc_count} articles, confidence: {confidence}\n"
            f"  Recent articles:{article_summaries}"
        )

    if context_parts:
        return "LIVE SENTIMENT DATA:\n" + "\n\n".join(context_parts)
    return "No specific ticker data found for this query."


def chat(user_message: str, history: list[dict]) -> str:
    """
    Send a message to Gemini with sentiment context.

    Args:
        user_message: The user's question.
        history: List of {"role": "user"|"assistant", "content": str} dicts.

    Returns:
        The assistant's response string.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return (
            "⚠️ Gemini API key not configured. "
            "Add GEMINI_API_KEY to your .env file. "
            "Get a key at https://aistudio.google.com/apikey"
        )

    try:
        import google.generativeai as genai
    except ImportError:
        return "⚠️ google-generativeai package not installed. Run: pip install google-generativeai"

    genai.configure(api_key=api_key)

    # Build context from live data
    context = build_context(user_message)

    system_prompt = (
        "You are a financial sentiment analysis assistant for the Indian stock market (NSE/BSE). "
        "You have access to real-time sentiment data from news articles and social media.\n\n"
        "IMPORTANT RULES:\n"
        "- Always cite specific sentiment scores and article counts when discussing a stock.\n"
        "- If sentiment is between -0.1 and 0.1, call it 'neutral' or 'mixed'.\n"
        "- If sentiment is > 0.1, call it 'positive' or 'bullish'.\n"
        "- If sentiment is < -0.1, call it 'negative' or 'bearish'.\n"
        "- NEVER give investment advice. Always say 'this is sentiment data, not investment advice.'\n"
        "- Keep responses concise — 3-5 sentences max unless the user asks for detail.\n"
        "- If you don't have data for a stock, say so honestly.\n\n"
        f"{context}"
    )

    model = genai.GenerativeModel("gemini-2.0-flash")

    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    try:
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(
            f"[System context: {system_prompt}]\n\nUser question: {user_message}"
        )
        return response.text
    except Exception as exc:
        logger.error(f"Gemini API error: {exc}")
        return f"⚠️ Error communicating with Gemini: {str(exc)}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()
    response = chat("How is Reliance doing?", [])
    print(f"Response:\n{response}")
