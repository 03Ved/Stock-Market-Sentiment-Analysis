"""
Batch-processes all unscored documents through the NLP pipeline.
Called by the scheduler after each ingestion cycle.
"""

import json
import logging
from db.database import get_connection
from nlp.ticker_linker import link_tickers
from nlp.sentiment import score_document

logger = logging.getLogger(__name__)


def get_unlinked_docs(conn, batch_size: int = 50) -> list[tuple]:
    """Fetch docs where tickers is still '[]', ordered oldest-first."""
    cursor = conn.execute(
        """
        SELECT id, headline, body
        FROM documents
        WHERE tickers = '[]'
        ORDER BY published_at ASC
        LIMIT ?
        """,
        (batch_size,),
    )
    return cursor.fetchall()


def get_unscored_docs(conn, batch_size: int = 50) -> list[tuple]:
    """Fetch docs where sentiment IS NULL, ordered oldest-first."""
    cursor = conn.execute(
        """
        SELECT id, headline, body, tickers
        FROM documents
        WHERE sentiment IS NULL
        ORDER BY published_at ASC
        LIMIT ?
        """,
        (batch_size,),
    )
    return cursor.fetchall()


def run_ticker_linking(conn, batch_size: int = 50) -> int:
    """Link tickers for all unlinked documents. Returns count of docs that got at least one ticker."""
    docs = get_unlinked_docs(conn, batch_size)
    if not docs:
        return 0
    linked_count = 0
    for doc_id, headline, body in docs:
        try:
            tickers = link_tickers(headline, body)
            if tickers:
                conn.execute(
                    "UPDATE documents SET tickers = ? WHERE id = ?",
                    (json.dumps(tickers), doc_id),
                )
                linked_count += 1
        except Exception as e:
            logger.error(f"Ticker linking failed for doc {doc_id}: {e}")
            continue
    conn.commit()
    logger.info(f"Linked tickers for {linked_count}/{len(docs)} documents")
    return linked_count


def run_sentiment_scoring(conn, batch_size: int = 50) -> int:
    """Score sentiment for all unscored documents. Returns count scored."""
    docs = get_unscored_docs(conn, batch_size)
    if not docs:
        return 0
    scored_count = 0
    for doc_id, headline, body, tickers_json in docs:
        try:
            doc = {"headline": headline, "body": body}
            result = score_document(doc)
            conn.execute(
                "UPDATE documents SET sentiment = ?, low_confidence = ? WHERE id = ?",
                (json.dumps(result), int(result.get("low_confidence", False)), doc_id),
            )
            scored_count += 1
        except Exception as e:
            logger.error(f"Sentiment scoring failed for doc {doc_id}: {e}")
            continue
    conn.commit()
    logger.info(f"Scored {scored_count}/{len(docs)} documents")
    return scored_count


def run_full_pipeline() -> dict:
    """
    Run the complete NLP pipeline on all unprocessed documents.
    Returns dict with counts: {"linked": int, "scored": int}.
    """
    conn = get_connection()
    total_linked = 0
    total_scored = 0
    try:
        # Phase 1: Link tickers — process ALL unlinked docs in one large batch
        # (docs that get no tickers stay '[]' so we avoid infinite loops by
        # fetching the total count once and using it as the batch size)
        unlinked_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE tickers = '[]'"
        ).fetchone()[0]
        if unlinked_count > 0:
            total_linked += run_ticker_linking(conn, batch_size=unlinked_count)

        # Phase 2: Score sentiment in batches until none remain
        while True:
            n = run_sentiment_scoring(conn, batch_size=50)
            total_scored += n
            if n == 0:
                break

        logger.info(f"Pipeline complete: {total_linked} linked, {total_scored} scored")
    finally:
        conn.close()
    return {"linked": total_linked, "scored": total_scored}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_full_pipeline()
    print(f"Orchestrator result: {result}")
