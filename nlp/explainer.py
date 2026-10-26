"""
Token-level sentiment attribution using Integrated Gradients.

This module explains FinBERT predictions by computing how much each input
token contributes to the predicted sentiment class. Uses the Integrated
Gradients method (Sundararajan et al., 2017).
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Label mapping matching FinBERT's output order
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentExplainer:
    """Explains FinBERT sentiment predictions via Integrated Gradients."""

    def __init__(
        self,
        model: Optional[AutoModelForSequenceClassification] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: str = "ProsusAI/finbert",
    ):
        """
        Initialize the explainer.

        Args:
            model: Pre-loaded model. If None, loads from model_name.
            tokenizer: Pre-loaded tokenizer. If None, loads from model_name.
            model_name: HuggingFace model identifier. Used only if model/tokenizer are None.
        """
        if model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        self.model.eval()

    def explain(self, text: str, n_steps: int = 50) -> list[dict]:
        """
        Compute token-level attribution for a sentiment prediction.

        Args:
            text: Input text (headline or headline + body).
            n_steps: Number of interpolation steps for Integrated Gradients.
                     Higher = more accurate but slower.

        Returns:
            List of dicts with keys: "token" (str), "attribution" (float).
            Special tokens [CLS], [SEP], [PAD] are excluded.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Baseline: all padding tokens
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        # Get embedding layer
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        embedding_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            input_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_ids)

        # Run Integrated Gradients
        delta = input_embeds - baseline_embeds
        attributions = torch.zeros_like(input_embeds)

        # Explain one fixed target: the prediction for the actual input.
        with torch.no_grad():
            pred_class = self.model(**inputs).logits.argmax(-1).item()

        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            interpolated = (baseline_embeds + alpha * delta).detach().requires_grad_(True)

            outputs = self.model(
                inputs_embeds=interpolated, attention_mask=attention_mask
            )
            score = outputs.logits[0, pred_class]
            gradient, = torch.autograd.grad(score, interpolated)
            attributions += gradient.detach()

        # Scale attributions by the input-baseline delta
        attributions = (attributions / n_steps) * delta
        # Sum over embedding dimension to get per-token scalar
        token_attributions = attributions.sum(dim=-1).squeeze().detach().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        result = []
        skip_tokens = {"[CLS]", "[SEP]", "[PAD]"}
        for tok, attr in zip(tokens, token_attributions):
            if tok in skip_tokens:
                continue
            result.append({"token": tok, "attribution": round(float(attr), 6)})

        result = self.merge_wordpiece_tokens(result)
        return result

    @staticmethod
    def merge_wordpiece_tokens(token_attributions: list[dict]) -> list[dict]:
        """
        Merge WordPiece subword tokens (## prefixed) back into whole words.
        Sum their attributions.

        Example:
            [{"token": "Rel", "attribution": 0.3}, {"token": "##iance", "attribution": 0.2}]
            becomes:
            [{"token": "Reliance", "attribution": 0.5}]
        """
        if not token_attributions:
            return []

        merged = []
        current_word = token_attributions[0]["token"]
        current_attr = token_attributions[0]["attribution"]

        for item in token_attributions[1:]:
            tok = item["token"]
            attr = item["attribution"]
            if tok.startswith("##"):
                # Continuation of the current word
                current_word += tok[2:]
                current_attr += attr
            else:
                # New word — save the previous one
                merged.append({"token": current_word, "attribution": round(current_attr, 6)})
                current_word = tok
                current_attr = attr

        # Don't forget the last word
        merged.append({"token": current_word, "attribution": round(current_attr, 6)})

        return merged


# Module-level singleton (lazy-loaded)
_explainer: Optional[SentimentExplainer] = None


def get_explainer() -> SentimentExplainer:
    """Get or create the singleton SentimentExplainer."""
    global _explainer
    if _explainer is None:
        from nlp import sentiment
        sentiment._load_model()
        _explainer = SentimentExplainer(model=sentiment._model, tokenizer=sentiment._tokenizer)
    return _explainer


def explain_text(text: str, n_steps: int = 50) -> list[dict]:
    """
    Convenience function: explain a single text.

    Args:
        text: The headline to explain.
        n_steps: Integrated Gradients precision.

    Returns:
        List of {"token": str, "attribution": float} dicts.
    """
    return get_explainer().explain(text, n_steps=n_steps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_headline = "Reliance Industries reports record quarterly profit"
    print(f"Explaining: {test_headline}")
    attrs = explain_text(test_headline, n_steps=30)
    for a in attrs:
        direction = "+" if a["attribution"] > 0 else "-" if a["attribution"] < 0 else " "
        print(f"  {direction} {a['token']:20s}  {a['attribution']:+.6f}")
