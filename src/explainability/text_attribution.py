from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from src.preprocessing.text.normalizer import TextNormalizer

class TextAttributionEngine:
    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        self.normalizer = normalizer or TextNormalizer()

    def compute_perturbation_attribution(self, text: str, predict_fn) -> Dict[str, Any]:
        """
        Leave-One-Out Perturbation Attribution (Model-Agnostic LIME/Shapley surrogate).
        Measures the change in hateful class probability when each token is removed.
        Delta > 0 means the token increases hateful probability (positive evidence for hate).
        Delta < 0 means the token decreases hateful probability (negative evidence for hate / pushes non-hate).
        """
        words = self.normalizer.tokenize_words(self.normalizer.normalize(text))
        if not words:
            return {"token_attributions": [], "faithfulness_score": 1.0}

        base_res = predict_fn(text)
        base_prob = base_res.get("raw_hateful_probability", base_res.get("confidence", 0.5))
        base_logits = base_res.get("logits")

        raw_pert = []
        for i in range(len(words)):
            # Leave word i out
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = " ".join(perturbed_words) if perturbed_words else "[EMPTY]"
            
            p_res = predict_fn(perturbed_text)
            p_prob = p_res.get("raw_hateful_probability", p_res.get("confidence", 0.5))
            p_logits = p_res.get("logits")

            dp = float(base_prob - p_prob)
            dl = float(base_logits[1] - p_logits[1]) if (base_logits and p_logits) else 0.0
            raw_pert.append((words[i], dp, dl))

        # Handle Softmax Saturation: when probabilities are at extreme certainty (0.0 or 1.0),
        # use normalized logit sensitivity so attributions do not vanish into flat zeroes.
        max_prob_delta = max((abs(r[1]) for r in raw_pert), default=0.0)
        use_logit_attribution = (max_prob_delta < 0.0005 and base_logits is not None)

        if use_logit_attribution:
            max_dl = max((abs(r[2]) for r in raw_pert), default=1.0)
            norm_factor = max(1.0, max_dl)
            attributions = []
            for w, dp, dl in raw_pert:
                score = round(dl / norm_factor * 0.5, 4)
                pol = "positive_evidence" if score > 0.01 else ("negative_evidence" if score < -0.01 else "neutral")
                attributions.append({
                    "token": w,
                    "score": score,
                    "abs_score": abs(score),
                    "polarity": pol
                })
        else:
            attributions = []
            for w, dp, dl in raw_pert:
                pol = "positive_evidence" if dp > 0.001 else ("negative_evidence" if dp < -0.001 else "neutral")
                attributions.append({
                    "token": w,
                    "score": round(dp, 4),
                    "abs_score": round(abs(dp), 4),
                    "polarity": pol
                })

        # Calculate Faithfulness Validation (AOPC)
        sorted_indices = np.argsort([a["abs_score"] for a in attributions])[::-1]
        k = max(1, int(len(words) * 0.25))
        top_k_indices = set(sorted_indices[:k])
        ablated_words = [words[i] for i in range(len(words)) if i not in top_k_indices]
        ablated_text = " ".join(ablated_words) if ablated_words else "[EMPTY]"
        ablated_res = predict_fn(ablated_text)
        ablated_prob = ablated_res.get("raw_hateful_probability", 0.5)

        if use_logit_attribution and base_logits and ablated_res.get("logits"):
            ablated_logits = ablated_res["logits"]
            logit_shift = abs(base_logits[1] - ablated_logits[1])
            faithfulness = min(1.0, max(0.2, logit_shift / (abs(base_logits[1]) + 1e-4)))
        else:
            prob_drop = abs(base_prob - ablated_prob)
            faithfulness = min(1.0, max(0.0, prob_drop / max(0.01, base_prob)))

        return {
            "token_attributions": attributions,
            "faithfulness_score": round(float(faithfulness), 4),
            "top_k_removed_count": k,
            "baseline_prob": round(float(base_prob), 4),
            "ablated_prob": round(float(ablated_prob), 4)
        }

