"""
Explanation Quality Evaluation Module
Quantifies explanation faithfulness, stability, and localization relevance
via perturbation tests on actual samples.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class ExplanationQualityEvaluator:
    """Evaluates explanation faithfulness and attribution relevance."""

    def evaluate_faithfulness(
        self,
        base_probability: float,
        perturbed_probability: float
    ) -> Dict[str, Any]:
        """
        Calculates Faithfulness Metric:
        Drop in model confidence when top-attributed features are masked.
        Faithfulness = (P_base - P_perturbed) / P_base
        """
        prob_drop = base_probability - perturbed_probability
        faithfulness = max(0.0, prob_drop / max(base_probability, 1e-4))
        
        rating = "HIGH" if faithfulness > 0.40 else ("MODERATE" if faithfulness > 0.15 else "LOW")
        
        return {
            "base_probability": round(base_probability, 4),
            "perturbed_probability": round(perturbed_probability, 4),
            "probability_drop": round(prob_drop, 4),
            "faithfulness_score": round(min(faithfulness, 1.0), 4),
            "faithfulness_rating": rating
        }

    def evaluate_batch_explanations(self, explanation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates explanation quality metrics across an evaluation sample set."""
        if not explanation_results:
            return {"mean_faithfulness": 0.0, "total_samples_evaluated": 0}

        drops = [res.get("probability_drop", 0.0) for res in explanation_results]
        scores = [res.get("faithfulness_score", 0.0) for res in explanation_results]

        return {
            "total_samples_evaluated": len(explanation_results),
            "mean_faithfulness_score": round(float(np.mean(scores)), 4),
            "mean_probability_drop": round(float(np.mean(drops)), 4),
            "faithfulness_std": round(float(np.std(scores)), 4),
            "high_faithfulness_percentage": round(float(sum(1 for s in scores if s > 0.35) / len(scores) * 100), 1)
        }

