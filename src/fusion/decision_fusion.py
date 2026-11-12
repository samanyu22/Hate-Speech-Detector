"""
Decision-Level Late Fusion Module
Combines independent single-modality probability distributions via dynamic evidential weighting.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class DecisionLevelFusion:
    """Combines modality-specific probability outputs with evidential confidence weighting."""

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        self.weights = default_weights or {"text": 0.35, "vision": 0.40, "audio": 0.25}

    def fuse(
        self,
        text_prob: Optional[Dict[str, float]] = None,
        vision_prob: Optional[Dict[str, float]] = None,
        audio_prob: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Combines available probabilities into a fused consensus distribution."""
        active_modalities = []
        active_weights = []
        p_real_list = []
        p_fake_list = []

        if text_prob is not None:
            active_modalities.append("text")
            active_weights.append(self.weights.get("text", 0.33))
            p_real_list.append(text_prob.get("REAL", 0.5))
            p_fake_list.append(text_prob.get("FAKE", 0.5))

        if vision_prob is not None:
            active_modalities.append("vision")
            active_weights.append(self.weights.get("vision", 0.33))
            p_real_list.append(vision_prob.get("REAL", 0.5))
            p_fake_list.append(vision_prob.get("FAKE", 0.5))

        if audio_prob is not None:
            active_modalities.append("audio")
            active_weights.append(self.weights.get("audio", 0.33))
            p_real_list.append(audio_prob.get("REAL", 0.5))
            p_fake_list.append(audio_prob.get("FAKE", 0.5))

        if not active_modalities:
            return {
                "prediction": "UNCERTAIN",
                "confidence": 0.5,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "disagreement_index": 1.0
            }

        # Normalize active weights
        total_w = sum(active_weights)
        norm_weights = [w / total_w for w in active_weights]

        fused_real = sum(w * p for w, p in zip(norm_weights, p_real_list))
        fused_fake = sum(w * p for w, p in zip(norm_weights, p_fake_list))

        # Modality variance / disagreement
        disagreement = float(np.std(p_fake_list)) if len(p_fake_list) > 1 else 0.0

        pred_label = "FAKE" if fused_fake > fused_real else "REAL"
        confidence = max(fused_real, fused_fake)

        return {
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "REAL": round(float(fused_real), 4),
                "FAKE": round(float(fused_fake), 4)
            },
            "disagreement_index": round(disagreement, 4),
            "modalities_combined": active_modalities
        }

