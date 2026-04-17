"""
Bootstrap silver labels from pretrained FinBERT predictions.

Runs pretrained FinBERT on all documents in the database.
Documents where confidence > 0.85 are labeled as 'silver' training data.
Outputs a CSV compatible with the fine-tuning script.
"""

import csv
import json
import logging
from db.database import get_connection
from nlp.sentiment import score_text

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def bootstrap_silver_labels(output_path: str = "data/silver_labels.csv") -> int:
    """
    Generate silver-labeled dataset from high-confidence FinBERT predictions.

    Args:
        output_path: Path to write the CSV.

    Returns:
        Number of silver-labeled rows written.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT headline, sentiment FROM documents WHERE headline IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    silver_rows = []
    for headline, sentiment_json in rows:
        if not headline or not headline.strip():
            continue

        # If already scored, use existing scores
        if sentiment_json:
            sent = json.loads(sentiment_json) if isinstance(sentiment_json, str) else sentiment_json
        else:
            # Score with pretrained FinBERT
            try:
                sent = score_text(headline)
            except Exception as e:
                logger.warning(f"Failed to score '{headline[:50]}': {e}")
                continue

        confidence = sent.get("confidence", 0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Determine label from highest probability
        probs = {
            "positive": sent.get("positive", 0),
            "negative": sent.get("negative", 0),
            "neutral":  sent.get("neutral", 0),
        }
        label = max(probs, key=probs.get)
        silver_rows.append({"headline": headline.strip(), "label": label, "confidence": round(confidence, 4)})

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "label", "confidence"])
        writer.writeheader()
        writer.writerows(silver_rows)

    logger.info(f"Wrote {len(silver_rows)} silver-labeled rows to {output_path}")
    return len(silver_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = bootstrap_silver_labels()
    print(f"Bootstrap complete: {count} silver labels")
