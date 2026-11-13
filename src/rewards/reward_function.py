from typing import Dict, Any, Optional
from backend.core.config import settings

class MultiObjectiveRewardCalculator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        cfg_weights = settings.reward_weights
        self.weights = weights or cfg_weights
        self.alpha = self.weights.get("alpha", 0.35)
        self.beta = self.weights.get("beta", 0.25)
        self.gamma = self.weights.get("gamma", 0.15)
        self.delta = self.weights.get("delta", 0.25)
        self.lambda_err = self.weights.get("lambda_err", 0.50)

    def calculate_reward(
        self,
        predicted_label: int,
        confidence: float,
        is_correct: Optional[bool] = None,
        suggested_label: Optional[int] = None,
        faithfulness_score: float = 0.5,
        explanation_rating: Optional[int] = None,
        context_used: bool = False,
        is_biased_false_positive: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates multi-objective reward vector:
        R = alpha * R_cls + beta * R_exp + gamma * R_ctx + delta * R_fb - lambda_err * R_err
        """
        # 1. R_classification: alignment with human/ground-truth assessment
        if is_correct is not None:
            r_cls = 1.0 if is_correct else 0.0
        elif suggested_label is not None:
            r_cls = 1.0 if (predicted_label == suggested_label) else 0.0
        else:
            # Baseline proxy: high confidence prediction
            r_cls = min(1.0, max(0.0, confidence))

        # 2. R_explanation: faithfulness drop ratio and attribution groundedness
        r_exp = min(1.0, max(0.0, faithfulness_score))

        # 3. R_context: reward for appropriate context disambiguation
        r_ctx = 0.85 if context_used else 0.50

        # 4. R_feedback: human rating (1 to 5 scale -> normalized to [0, 1])
        if explanation_rating is not None:
            r_fb = min(1.0, max(0.0, (explanation_rating - 1.0) / 4.0))
        elif is_correct is not None:
            r_fb = 1.0 if is_correct else 0.20
        else:
            r_fb = 0.50

        # 5. R_error: severe penalty for high-confidence erroneous or biased predictions
        if is_correct is False or (suggested_label is not None and suggested_label != predicted_label):
            # Error penalty proportional to overconfidence
            r_err = float(confidence)
        elif is_biased_false_positive:
            r_err = 1.0
        else:
            r_err = 0.0

        # Compute total composite reward
        raw_total = (
            self.alpha * r_cls +
            self.beta * r_exp +
            self.gamma * r_ctx +
            self.delta * r_fb -
            self.lambda_err * r_err
        )
        total_reward = round(float(raw_total), 4)

        return {
            "r_classification": round(float(r_cls), 4),
            "r_explanation": round(float(r_exp), 4),
            "r_context": round(float(r_ctx), 4),
            "r_feedback": round(float(r_fb), 4),
            "r_error": round(float(r_err), 4),
            "total_reward": total_reward,
            "formula": f"R = {self.alpha}*R_cls + {self.beta}*R_exp + {self.gamma}*R_ctx + {self.delta}*R_fb - {self.lambda_err}*R_err",
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
                "lambda_err": self.lambda_err
            }
        }

