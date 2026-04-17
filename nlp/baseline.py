"""
VADER baseline sentiment scorer.

Provides the same interface as nlp/sentiment.py so the two can be swapped
in evaluation code without changes.

VADER compound score is in [-1, 1]. We map it to the same output schema:
  - score      = compound  (pos - neg analogue)
  - positive   = pos
  - negative   = neg
  - neutral    = neu
  - confidence = abs(compound)  — how decisively VADER committed

Used for academic comparison against fine-tuned FinBERT in the evaluation
section of the project report.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import LOW_CONFIDENCE_THRESHOLD

_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Core scorer — same signature as nlp/sentiment.score_text()
# ---------------------------------------------------------------------------

def score_text(text: str) -> dict:
    """
    Score a single text string with VADER.

    Returns the same schema as nlp/sentiment.score_text():
        {
          "positive":   float,
          "negative":   float,
          "neutral":    float,
          "score":      float,   # compound, in [-1, 1]
          "confidence": float,   # abs(compound)
        }
    """
    scores = _analyzer.polarity_scores(text)
    return {
        "positive":   scores["pos"],
        "negative":   scores["neg"],
        "neutral":    scores["neu"],
        "score":      scores["compound"],
        "confidence": abs(scores["compound"]),
    }


def batch_score(texts: list[str], batch_size: int = 16) -> list[dict]:
    """
    Score a list of texts. batch_size is accepted for interface parity
    with the FinBERT scorer but has no effect (VADER is CPU-only, instant).
    """
    return [score_text(t) for t in texts]


def score_document(doc: dict) -> dict:
    """
    Score a unified-schema document with VADER.

    Mirrors the headline-first / body-fallback logic from sentiment.py
    so both scorers are evaluated on identical input text.
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
# CLI smoke test — same sentences used in sentiment.py for easy comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_sentences = [
        "Reliance Industries reports record quarterly profit, beats estimates",
        "HDFC Bank shares crash 8% after RBI imposes restrictions",
        "Markets remain flat ahead of quarterly earnings season",
        "Infosys raises FY26 revenue guidance on strong deal wins",
        "Adani Group faces regulatory scrutiny over governance concerns",
    ]

    print("--- VADER baseline scores ---")
    for sent in test_sentences:
        r = score_text(sent)
        bar = "+" if r["score"] > 0 else "-"
        print(f"  [{bar}{r['score']:+.3f} conf={r['confidence']:.2f}] {sent[:65]}")
