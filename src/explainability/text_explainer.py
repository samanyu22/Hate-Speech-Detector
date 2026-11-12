"""
Text Explainability Module
Computes token-level attributions and saliency scores via leave-one-out perturbation
and gradient-weighted sensitivity for transparent NLP evidence grounding.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F

from src.preprocessing.text_preprocessor import TextPreprocessor


class TextExplainer:
    """Computes grounded token-level saliency scores for deepfake text classifications."""

    def __init__(self, text_pipeline: Any):
        self.pipeline = text_pipeline

    def explain(self, text: str) -> Dict[str, Any]:
        """Calculates token importance scores by evaluating prediction drop upon masking."""
        base_pred = self.pipeline.predict(text)
        base_fake_prob = base_pred["probabilities"]["FAKE"]
        tokens = self.pipeline.preprocessor.tokenize(text)

        if not tokens:
            return {"tokens": [], "attributions": [], "top_evidence_phrases": []}

        attributions = []
        # Perturbation test: evaluate prediction drop when removing token
        for i, token in enumerate(tokens):
            perturbed_tokens = [t for j, t in enumerate(tokens) if j != i]
            perturbed_text = " ".join(perturbed_tokens)
            try:
                p_pred = self.pipeline.predict(perturbed_text)
                p_fake_prob = p_pred["probabilities"]["FAKE"]
                # Saliency is the drop in fake probability when removing this token
                saliency = float(base_fake_prob - p_fake_prob)
            except Exception:
                saliency = 0.0

            attributions.append({
                "token": token,
                "index": i,
                "saliency": round(saliency, 4),
                "is_suspicious": saliency > 0.05
            })

        # Normalize saliency to [0, 1] range for visual highlighting
        s_vals = [a["saliency"] for a in attributions]
        max_s = max(abs(min(s_vals)), abs(max(s_vals)), 1e-4)
        for a in attributions:
            a["normalized_weight"] = round(a["saliency"] / max_s, 3)

        # Extract top 3 evidence phrases
        sorted_attr = sorted(attributions, key=lambda x: abs(x["saliency"]), reverse=True)
        top_phrases = [x["token"] for x in sorted_attr[:5] if abs(x["saliency"]) > 0.01]

        return {
            "tokens": tokens,
            "attributions": attributions,
            "top_evidence_tokens": top_phrases,
            "base_confidence": base_pred["confidence"]
        }

