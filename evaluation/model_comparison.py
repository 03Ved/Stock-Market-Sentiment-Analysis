"""
Model comparison evaluation.

Compares three approaches on the test set:
  1. Pretrained FinBERT (ProsusAI/finbert)
  2. Fine-tuned FinBERT (models/finbert-indian-markets/best/)
  3. VADER rule-based baseline

Outputs evaluation/model_comparison.json and evaluation/confusion_matrix.json.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_test_data():
    """Load combined gold + silver labels and return the test split."""
    frames = []

    gold_path = "data/labeled_headlines.csv"
    if os.path.exists(gold_path):
        gold = pd.read_csv(gold_path, encoding="utf-8")
        gold["source"] = "gold"
        frames.append(gold)

    silver_path = "data/silver_labels.csv"
    if os.path.exists(silver_path):
        silver = pd.read_csv(silver_path, encoding="utf-8")
        silver["source"] = "silver"
        frames.append(silver)

    if not frames:
        raise FileNotFoundError("No label files found")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("source", ascending=True)
    df = df.drop_duplicates(subset=["headline"], keep="first")
    df = df[df["label"].isin(LABEL2ID.keys())].reset_index(drop=True)

    # Use same split as fine-tuning (seed=42, 70/15/15)
    _, X_temp, _, y_temp = train_test_split(
        df["headline"].tolist(),
        df["label"].map(LABEL2ID).values,
        test_size=0.30,
        random_state=42,
        stratify=df["label"].map(LABEL2ID).values,
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
    )
    return X_test, y_test


def evaluate_pretrained_finbert(texts, labels):
    """Evaluate pretrained FinBERT."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Evaluating pretrained FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.append(logits.argmax(-1).item())

    return np.array(preds)


def evaluate_finetuned_finbert(texts, labels):
    """Evaluate fine-tuned FinBERT."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = "models/finbert-indian-markets/best"
    if not os.path.exists(model_path):
        logger.warning(f"Fine-tuned model not found at {model_path}")
        return None

    logger.info("Evaluating fine-tuned FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    preds = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.append(logits.argmax(-1).item())

    return np.array(preds)


def evaluate_vader(texts, labels):
    """Evaluate VADER sentiment analyzer."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    logger.info("Evaluating VADER baseline...")
    analyzer = SentimentIntensityAnalyzer()

    preds = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound > 0.05:
            preds.append(0)  # positive
        elif compound < -0.05:
            preds.append(1)  # negative
        else:
            preds.append(2)  # neutral

    return np.array(preds)


def run_comparison():
    """Run full model comparison and save results."""
    os.makedirs("evaluation", exist_ok=True)

    X_test, y_test = load_test_data()
    logger.info(f"Test set: {len(X_test)} samples")

    results = {}

    # 1. Pretrained FinBERT
    try:
        preds_pre = evaluate_pretrained_finbert(X_test, y_test)
        results["Pretrained FinBERT"] = {
            "f1_macro": round(float(f1_score(y_test, preds_pre, average="macro")), 4),
            "f1_weighted": round(float(f1_score(y_test, preds_pre, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_test, preds_pre)), 4),
        }
        logger.info(f"Pretrained FinBERT: {results['Pretrained FinBERT']}")
    except Exception as exc:
        logger.error(f"Pretrained FinBERT evaluation failed: {exc}")

    # 2. Fine-tuned FinBERT
    try:
        preds_ft = evaluate_finetuned_finbert(X_test, y_test)
        if preds_ft is not None:
            results["Fine-tuned FinBERT"] = {
                "f1_macro": round(float(f1_score(y_test, preds_ft, average="macro")), 4),
                "f1_weighted": round(float(f1_score(y_test, preds_ft, average="weighted")), 4),
                "accuracy": round(float(accuracy_score(y_test, preds_ft)), 4),
            }
            logger.info(f"Fine-tuned FinBERT: {results['Fine-tuned FinBERT']}")

            # Save confusion matrix from the best model (fine-tuned)
            cm = confusion_matrix(y_test, preds_ft)
            cm_data = {
                "matrix": cm.tolist(),
                "labels": ["positive", "negative", "neutral"],
            }
            cm_path = "evaluation/confusion_matrix.json"
            with open(cm_path, "w", encoding="utf-8") as f:
                json.dump(cm_data, f, indent=2)
            logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as exc:
        logger.error(f"Fine-tuned FinBERT evaluation failed: {exc}")

    # 3. VADER baseline
    try:
        preds_vader = evaluate_vader(X_test, y_test)
        results["VADER Baseline"] = {
            "f1_macro": round(float(f1_score(y_test, preds_vader, average="macro")), 4),
            "f1_weighted": round(float(f1_score(y_test, preds_vader, average="weighted")), 4),
            "accuracy": round(float(accuracy_score(y_test, preds_vader)), 4),
        }
        logger.info(f"VADER Baseline: {results['VADER Baseline']}")
    except Exception as exc:
        logger.error(f"VADER evaluation failed: {exc}")

    # Save comparison
    out_path = "evaluation/model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Model comparison saved to {out_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_comparison()
    print("\n=== Model Comparison Results ===")
    for model_name, metrics in results.items():
        print(f"  {model_name}: F1={metrics['f1_macro']:.4f}  Acc={metrics['accuracy']:.4f}")
