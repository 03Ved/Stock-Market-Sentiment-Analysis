"""
FinBERT-based sentiment scorer.

Model: ProsusAI/finbert  (HuggingFace)
Label order (verified against model config): 0=positive, 1=negative, 2=neutral

score_text()  — score a single string, returns probability dict
batch_score() — score a list of strings efficiently
"""

import json
import logging
from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from config.settings import LOW_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_LABELS = ["positive", "negative", "neutral"]   # index 0, 1, 2

# ---------------------------------------------------------------------------
# Lazy model loading — loaded once on first use
# ---------------------------------------------------------------------------

_tokenizer: BertTokenizer | None = None
_model: BertForSequenceClassification | None = None


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    import os
    model_path = _MODEL_NAME  # default: ProsusAI/finbert
    fine_tuned_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "finbert-indian-markets", "best")
    if os.path.exists(os.path.join(fine_tuned_path, "config.json")):
        model_path = fine_tuned_path
        log.info(f"Using fine-tuned model from {model_path}")
    else:
        log.info(f"Fine-tuned model not found, using pretrained {model_path}")
    _tokenizer = BertTokenizer.from_pretrained(model_path)
    _model = BertForSequenceClassification.from_pretrained(model_path)
    _model.eval()

    # Verify label order matches expectation
    id2label = _model.config.id2label
    expected = {0: "positive", 1: "negative", 2: "neutral"}
    for idx, label in expected.items():
        actual = id2label.get(idx, "").lower()
        if actual != label:
            log.warning(
                f"FinBERT label mismatch at index {idx}: "
                f"expected '{label}', got '{actual}'. "
                f"Scores may be incorrect — check model config."
            )
    log.info("FinBERT loaded.")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_text(text: str) -> dict:
    """
    Score a single text string with FinBERT.

    Returns:
        {
          "positive":   float,   # probability
          "negative":   float,
          "neutral":    float,
          "score":      float,   # positive - negative, in [-1, 1]
          "confidence": float,   # max(probs)
        }
    """
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return {
        "positive":   probs[0],
        "negative":   probs[1],
        "neutral":    probs[2],
        "score":      probs[0] - probs[1],
        "confidence": max(probs),
    }


def batch_score(texts: list[str], batch_size: int = 16) -> list[dict]:
    """
    Score a list of texts in mini-batches.

    Returns a list of dicts in the same order as the input.
    """
    _load_model()
    results: list[dict] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = _tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = _model(**inputs).logits

        probs_batch = torch.softmax(logits, dim=1).tolist()
        for probs in probs_batch:
            results.append({
                "positive":   probs[0],
                "negative":   probs[1],
                "neutral":    probs[2],
                "score":      probs[0] - probs[1],
                "confidence": max(probs),
            })

    return results


# ---------------------------------------------------------------------------
# Document-level scorer (implements headline-first, body fallback logic)
# ---------------------------------------------------------------------------

def score_document(doc: dict) -> dict:
    """
    Score a unified-schema document.

    Strategy (from context.md):
      - Score headline only by default.
      - If body is present AND headline confidence < 0.6,
        re-score with headline + ". " + body[:400].

    Returns the sentiment dict to be stored in doc["sentiment"].
    Also returns low_confidence=True if confidence < LOW_CONFIDENCE_THRESHOLD.
    """
    headline_result = score_text(doc["headline"])

    if doc.get("body") and headline_result["confidence"] < 0.6:
        combined = doc["headline"] + ". " + doc["body"][:400]
        result = score_text(combined)
    else:
        result = headline_result

    result["low_confidence"] = result["confidence"] < LOW_CONFIDENCE_THRESHOLD
    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_sentences = [
        "Reliance Industries reports record quarterly profit, beats estimates",
        "HDFC Bank shares crash 8% after RBI imposes restrictions",
        "Markets remain flat ahead of quarterly earnings season",
        "Infosys raises FY26 revenue guidance on strong deal wins",
        "Adani Group faces regulatory scrutiny over governance concerns",
    ]

    print("\n--- Single scoring ---")
    for sent in test_sentences:
        r = score_text(sent)
        bar = "+" if r["score"] > 0 else "-"
        print(f"  [{bar}{r['score']:+.3f} conf={r['confidence']:.2f}] {sent[:65]}")

    print("\n--- Batch scoring (same texts) ---")
    results = batch_score(test_sentences, batch_size=4)
    for sent, r in zip(test_sentences, results):
        print(f"  score={r['score']:+.3f}  conf={r['confidence']:.2f}  {sent[:55]}")
