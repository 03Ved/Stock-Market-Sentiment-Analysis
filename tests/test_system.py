"""Deterministic checks use a temporary database and never call live providers."""
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import main
from aggregation.aggregator import aggregate_ticker, effective_weight
from db import database
from dashboard.client import get_data
from nlp.ticker_linker import link_tickers


@pytest.fixture
def db(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()
    return database


def document(identifier="one", hours=1, score=0.8, published=None):
    now = datetime.now(timezone.utc)
    return dict(id=identifier, source_id="google_news", source_weight=0.7,
                url=f"https://example.com/{identifier}", headline="Reliance reports record profit",
                body=None, published_at=published or (now-timedelta(hours=hours)).isoformat(),
                scraped_at=now.isoformat(), tickers=["RELIANCE"],
                sentiment=dict(score=score, confidence=0.9))


def test_deduplication_and_confidence_flag(db):
    doc = document()
    doc["sentiment"]["low_confidence"] = True
    with db.get_connection() as conn:
        assert db.insert_doc(conn, doc)
        assert not db.insert_doc(conn, doc)
        assert conn.execute("SELECT low_confidence FROM documents").fetchone()[0] == 1


def test_weight_and_aggregate():
    now = datetime.now(timezone.utc)
    doc = document(published=(now-timedelta(hours=2)).isoformat())
    assert effective_weight(doc, now) == pytest.approx(0.7*0.9*math.exp(-0.6))
    assert aggregate_ticker([doc], now) is None
    assert aggregate_ticker([doc, doc], now)["score"] == 0.8
    doc["low_confidence"] = True
    assert aggregate_ticker([doc, doc], now) is None


def test_future_timestamp_does_not_boost_weight():
    now = datetime.now(timezone.utc)
    doc = document(published=(now+timedelta(days=1)).isoformat())
    assert effective_weight(doc, now) == pytest.approx(0.63)


def test_ticker_linking():
    assert {"RELIANCE", "TCS"} <= set(link_tickers("Reliance and TCS report earnings"))


def test_health_search_and_missing_articles(db):
    with TestClient(main.app) as client:
        assert client.get("/health").json()["doc_count"] == 0
        assert any(r["symbol"] == "RELIANCE" for r in client.get("/search?q=reliance").json()["results"])
        assert client.get("/ticker/NOT_A_SYMBOL").status_code == 404
        assert client.get("/ticker/RELIANCE?window=0h").status_code == 400
        assert client.get("/ticker/RELIANCE/history?hours=0").status_code == 422
        assert client.get("/ticker/RELIANCE/history?resolution=bad").status_code == 400
        assert client.get("/explain/missing").status_code == 404


def test_ticker_api_and_local_client_agree(db):
    with db.get_connection() as conn:
        db.insert_doc(conn, document("one"))
        db.insert_doc(conn, document("two", score=-0.2))
    with TestClient(main.app) as client:
        data = client.get("/ticker/reliance?window=7d").json()
        assert data["doc_count"] == 2
        assert -1 <= data["score"] <= 1
        assert len(data["articles"]) == 2
        assert get_data("/ticker/RELIANCE", window="7d")["score"] == data["score"]


def test_history_includes_partial_final_bucket_and_utc_z(db):
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with db.get_connection() as conn:
        db.insert_doc(conn, document("one", published=stamp))
        db.insert_doc(conn, document("two", published=stamp))
    points = main.ticker_history("RELIANCE", hours=25, resolution="6h")["points"]
    assert len(points) == 5
    assert points[-1]["doc_count"] == 2
    assert points[-1]["score"] == 0.8


def test_external_client_timeout_and_params(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com/")
    with patch("dashboard.client.requests.get") as request:
        request.return_value.json.return_value = {"results": []}
        assert get_data("/search", q="reliance") == {"results": []}
        request.assert_called_once_with("https://api.example.com/search", params={"q": "reliance"}, timeout=180)


def test_streamlit_dashboard(db):
    from streamlit.testing.v1 import AppTest
    with db.get_connection() as conn:
        db.insert_doc(conn, document("one"))
        db.insert_doc(conn, document("two"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert any(m.label == "Sentiment Score" for m in app.metric)
    app.selectbox[0].set_value("24 hours").run()
    assert not app.exception
    app.text_input[0].set_value("not-a-known-company-zzzz").run()
    assert not app.exception


def test_chat_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from agent.chat import chat
    assert "not configured" in chat("Reliance?", [])


def test_google_news_ingestion_retries_failed_scoring(db, monkeypatch):
    import ingestion.google_news_scraper as news
    rss = b"""<rss version="2.0"><channel><title>Test</title>
      <item><title>Reliance earnings rise - Example</title><link>https://example.com/earnings</link>
      <pubDate>Tue, 27 Oct 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    monkeypatch.setattr(news, "generate_smart_queries", lambda *a: ["Reliance news"])
    monkeypatch.setattr(news, "fetch_gnews_for_ticker", lambda *a: 0)
    monkeypatch.setattr(news, "filter_relevant_headlines", lambda *a: {0})
    with patch.object(news.requests, "get") as request:
        request.return_value.content = rss
        with patch.object(news, "score_document", side_effect=RuntimeError("temporary failure")):
            assert news.fetch_google_news_for_ticker("RELIANCE") == 0
        with patch.object(news, "score_document", return_value={"score": 0.7, "confidence": 0.9}):
            assert news.fetch_google_news_for_ticker("RELIANCE") == 1
            assert news.fetch_google_news_for_ticker("RELIANCE") == 0
        assert request.call_args.kwargs["timeout"] == 15


def test_gemini_payload_and_failure(monkeypatch):
    from agent.gemini import generate_text
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    with patch("agent.gemini.requests.post") as request:
        request.return_value.ok = True
        request.return_value.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Answer"}]}}]}
        assert generate_text("Hello") == "Answer"
        assert "test-model" in request.call_args.args[0]
        assert request.call_args.kwargs["timeout"] == 45
        request.return_value.ok = False
        request.return_value.status_code = 429
        with pytest.raises(RuntimeError, match="429"):
            generate_text("Hello")
