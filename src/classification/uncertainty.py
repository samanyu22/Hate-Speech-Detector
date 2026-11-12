"""
Uncertainty Estimation and Abstention Engine
Computes calibrated probabilities, Shannon entropy, cross-modal contradiction penalties,
and governs UNCERTAIN classification decisions.
"""

import math
from typing import Dict, List, Any, Optional


class UncertaintyEngine:
    """Manages confidence calibration, entropy computation, and abstention policies."""

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        entropy_threshold: float = 0.65,
        temperature: float = 1.0
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature

    def compute_normalized_entropy(self, p_real: float, p_fake: float) -> float:
        """Calculates normalized Shannon binary entropy in range [0, 1]."""
        eps = 1e-9
        p0 = max(min(p_real, 1.0 - eps), eps)
        p1 = max(min(p_fake, 1.0 - eps), eps)
        h = -(p0 * math.log2(p0) + p1 * math.log2(p1))
        return float(min(max(h, 0.0), 1.0))

    def evaluate(
        self,
        p_real: float,
        p_fake: float,
        cross_modal_mismatch: bool = False,
        missing_modalities: Optional[List[str]] = None,
        modality_disagreement: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates whether a sample should be declared REAL, FAKE, or UNCERTAIN.
        """
        missing_modalities = missing_modalities or []
        entropy = self.compute_normalized_entropy(p_real, p_fake)
        raw_confidence = max(p_real, p_fake)
        tentative_label = "FAKE" if p_fake > p_real else "REAL"

        uncertainty_factors = []
        is_uncertain = False

        # 1. Low Confidence Abstention
        if raw_confidence < self.confidence_threshold:
            is_uncertain = True
            uncertainty_factors.append(
                f"Prediction confidence ({raw_confidence:.2f}) falls below minimum reliability threshold ({self.confidence_threshold:.2f})."
            )

        # 2. High Entropy Abstention
        if entropy > self.entropy_threshold:
            is_uncertain = True
            uncertainty_factors.append(
                f"Normalized prediction entropy ({entropy:.2f}) indicates severe model ambiguity (threshold: {self.entropy_threshold:.2f})."
            )

        # 3. Cross-modal contradiction with baseline disagreement
        if cross_modal_mismatch and modality_disagreement > 0.25:
            is_uncertain = True
            uncertainty_factors.append(
                f"Severe cross-modal contradiction detected with substantial single-modality disagreement ({modality_disagreement:.2f}). Evidence is conflicting."
            )

        # 4. Critical Missing Modality penalty
        if len(missing_modalities) >= 2:
            is_uncertain = True
            uncertainty_factors.append(
                f"Multiple modalities missing ({', '.join(missing_modalities)}); insufficient multi-source corroboration."
            )

        final_label = "UNCERTAIN" if is_uncertain else tentative_label

        return {
            "final_label": final_label,
            "is_uncertain": is_uncertain,
            "tentative_label": tentative_label,
            "confidence": round(raw_confidence, 4),
            "normalized_entropy": round(entropy, 4),
            "uncertainty_score": round(entropy * (1.0 - raw_confidence + 0.5), 4),
            "uncertainty_reasons": uncertainty_factors
        }

