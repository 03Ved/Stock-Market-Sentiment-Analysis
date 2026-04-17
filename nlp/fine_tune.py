"""
Fine-tune FinBERT on Indian financial headlines with proper evaluation.

Key improvements over v1:
- WeightedTrainer with class-balanced cross-entropy loss
- Supports both gold labels (labeled_headlines.csv) and silver labels
- Proper train/val/test stratified split
- Computes macro F1, weighted F1, accuracy, and confusion matrix
- Saves learning curves to evaluation/training_curves.json
- Saves confusion matrix to evaluation/confusion_matrix.png

Usage:
    python nlp/fine_tune.py
    python nlp/fine_tune.py --data data/labeled_headlines.csv --epochs 5
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Label encoding
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "models/finbert-indian-markets"


class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch dataset for headline sentiment classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    """Trainer subclass that uses class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_dataset(
    gold_path: str = "data/labeled_headlines.csv",
    silver_path: str = "data/silver_labels.csv",
) -> pd.DataFrame:
    """
    Load and combine gold + silver labeled data.
    Gold labels take priority over silver labels for the same headline.
    """
    frames = []

    if os.path.exists(gold_path):
        gold = pd.read_csv(gold_path, encoding="utf-8")
        gold["source"] = "gold"
        frames.append(gold)
        logger.info(f"Loaded {len(gold)} gold labels from {gold_path}")

    if os.path.exists(silver_path):
        silver = pd.read_csv(silver_path, encoding="utf-8")
        silver["source"] = "silver"
        frames.append(silver)
        logger.info(f"Loaded {len(silver)} silver labels from {silver_path}")

    if not frames:
        raise FileNotFoundError(
            f"No label files found at {gold_path} or {silver_path}"
        )

    df = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep gold over silver
    df = df.sort_values("source", ascending=True)  # gold before silver
    df = df.drop_duplicates(subset=["headline"], keep="first")
    df = df[df["label"].isin(LABEL2ID.keys())].reset_index(drop=True)

    logger.info(f"Combined dataset: {len(df)} rows")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")
    return df


def compute_metrics(eval_pred):
    """Compute classification metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy":    float((preds == labels).mean()),
    }


def fine_tune():
    """Run the full fine-tuning pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    # Load data
    df = load_dataset()
    labels = df["label"].map(LABEL2ID).values
    texts = df["headline"].tolist()

    # Stratified split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=128)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=128)

    # Convert to tensors
    def _to_tensors(enc):
        return {k: torch.tensor(v) for k, v in enc.items()}

    train_ds = SentimentDataset(_to_tensors(train_enc), torch.tensor(y_train))
    val_ds   = SentimentDataset(_to_tensors(val_enc),   torch.tensor(y_val))
    test_ds  = SentimentDataset(_to_tensors(test_enc),  torch.tensor(y_test))

    # Compute class weights for balanced loss
    class_weights_np = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    logger.info(f"Class weights: {dict(zip(ID2LABEL.values(), class_weights_np.round(3)))}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir                  = os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs            = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = 20,
        seed                        = 42,
        report_to                   = "none",
    )

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning with WeightedTrainer ...")
    trainer.train()

    # Save best model
    best_model_path = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set ...")
    test_preds_output = trainer.predict(test_ds)
    test_preds  = np.argmax(test_preds_output.predictions, axis=-1)
    test_labels = y_test

    report = classification_report(
        test_labels, test_preds,
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4,
    )
    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[ID2LABEL[i] for i in range(3)],
            yticklabels=[ID2LABEL[i] for i in range(3)],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — Test Set")
        plt.tight_layout()
        cm_path = os.path.join("evaluation", "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot")

    # Persist metrics and learning curves
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report)

    log_history = trainer.state.log_history
    curves_path = os.path.join("evaluation", "training_curves.json")
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Training curves saved to {curves_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fine_tune()
