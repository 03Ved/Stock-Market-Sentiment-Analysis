"""
Generate evaluation plots from training results.

Reads JSON files in the evaluation/ directory and saves PNG images to
evaluation/plots/ for display in the dashboard.

Supported input files:
  - evaluation/confusion_matrix.json
  - evaluation/training_curves.json
  - evaluation/model_comparison.json

Run with:
  PYTHONPATH=. python -m evaluation.generate_plots
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def generate_all_plots():
    """Generate all evaluation plots. Fails gracefully if data is missing."""
    os.makedirs("evaluation/plots", exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend — safe in server environments
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        logger.error(f"matplotlib or numpy not installed: {exc}")
        return

    # ------------------------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------------------------
    cm_path = "evaluation/confusion_matrix.json"
    if os.path.exists(cm_path):
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                cm_data = json.load(f)
            matrix = np.array(cm_data["matrix"])
            labels = cm_data.get("labels", ["positive", "negative", "neutral"])

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_yticklabels(labels, fontsize=12)
            ax.set_xlabel("Predicted", fontsize=13)
            ax.set_ylabel("Actual", fontsize=13)
            ax.set_title("Confusion Matrix — Fine-tuned FinBERT", fontsize=14)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(
                        j, i, str(matrix[i, j]),
                        ha="center", va="center", fontsize=16,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    )

            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig("evaluation/plots/confusion_matrix.png", dpi=150)
            plt.close(fig)
            logger.info("Saved confusion_matrix.png")
        except Exception as exc:
            logger.error(f"Failed to generate confusion matrix plot: {exc}")
    else:
        logger.info(f"Skipping confusion matrix — {cm_path} not found")

    # ------------------------------------------------------------------
    # 2. Training Curves
    # ------------------------------------------------------------------
    tc_path = "evaluation/training_curves.json"
    if os.path.exists(tc_path):
        try:
            with open(tc_path, "r", encoding="utf-8") as f:
                history = json.load(f)

            train_loss = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
            eval_loss  = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
            eval_f1    = [(h["epoch"], h["eval_f1_macro"]) for h in history if "eval_f1_macro" in h]

            if train_loss or eval_loss:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                if train_loss:
                    ax1.plot(
                        [x[0] for x in train_loss], [x[1] for x in train_loss],
                        "b-o", label="Train Loss", markersize=4,
                    )
                if eval_loss:
                    ax1.plot(
                        [x[0] for x in eval_loss], [x[1] for x in eval_loss],
                        "r-o", label="Val Loss", markersize=4,
                    )
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_title("Training & Validation Loss")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                if eval_f1:
                    ax2.plot(
                        [x[0] for x in eval_f1], [x[1] for x in eval_f1],
                        "g-o", markersize=5,
                    )
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("F1 Macro")
                ax2.set_title("Validation F1 Score")
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)

                fig.suptitle("FinBERT Fine-tuning — Learning Curves", fontsize=14)
                fig.tight_layout()
                fig.savefig("evaluation/plots/training_curves.png", dpi=150)
                plt.close(fig)
                logger.info("Saved training_curves.png")
            else:
                logger.info("training_curves.json has no usable loss entries — skipping")
        except Exception as exc:
            logger.error(f"Failed to generate training curves plot: {exc}")
    else:
        logger.info(f"Skipping training curves — {tc_path} not found")

    # ------------------------------------------------------------------
    # 3. Model Comparison Bar Chart
    # ------------------------------------------------------------------
    mc_path = "evaluation/model_comparison.json"
    if os.path.exists(mc_path):
        try:
            with open(mc_path, "r", encoding="utf-8") as f:
                comparison = json.load(f)

            models     = list(comparison.keys())
            f1_scores  = [comparison[m]["f1_macro"] for m in models]
            accuracies = [comparison[m]["accuracy"] for m in models]

            x     = np.arange(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width / 2, f1_scores,  width, label="F1 Macro",  color="#10B981")
            bars2 = ax.bar(x + width / 2, accuracies, width, label="Accuracy",  color="#6366F1")

            ax.set_ylabel("Score")
            ax.set_title("Model Comparison — Test Set Performance")
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=10)
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, axis="y", alpha=0.3)

            for bar in bars1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                )
            for bar in bars2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                )

            fig.tight_layout()
            fig.savefig("evaluation/plots/model_comparison.png", dpi=150)
            plt.close(fig)
            logger.info("Saved model_comparison.png")
        except Exception as exc:
            logger.error(f"Failed to generate model comparison plot: {exc}")
    else:
        logger.info(f"Skipping model comparison — {mc_path} not found")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_plots()
    print("Plot generation complete. Check evaluation/plots/")
