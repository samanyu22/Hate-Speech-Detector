"""
Experiment 3: Generalization Across Manipulation Types and Unseen Modality Combinations
Evaluates how well the trained multimodal model transfers to unseen manipulation styles
and assesses performance degradation under distribution shift.
"""

from typing import Dict, List, Any
from src.data.dataset_manager import DatasetManager
from src.classification.detector import MultimodalVerificationSystem
from src.evaluation.metrics import MetricsCalculator


class Experiment3Runner:
    """Executes Experiment 3."""

    def __init__(self, detector: MultimodalVerificationSystem, dataset_manager: DatasetManager):
        self.detector = detector
        self.dm = dataset_manager

    def run(self, dataset_id: str = "benchmark_300") -> Dict[str, Any]:
        """Evaluates in-domain vs out-of-domain manipulation generalization."""
        records = self.dm.load_records(dataset_id)
        
        # Partition into In-Domain and Out-Of-Domain
        # In-domain: authentic, face_manipulation, misleading_caption
        # Out-of-domain (unseen/zero-shot): semantic_mismatch, synthetic_speech
        in_domain_records = [r for r in records if r.manipulation_type in ["authentic", "face_manipulation", "misleading_caption"] and r.split == "test"]
        out_domain_records = [r for r in records if r.manipulation_type in ["semantic_mismatch", "synthetic_speech", "ai_generated_image"] and r.split == "test"]

        # If test split is small for out_domain, draw from val/test
        if len(out_domain_records) < 10:
            out_domain_records = [r for r in records if r.manipulation_type in ["semantic_mismatch", "synthetic_speech", "ai_generated_image"]]

        # 1. In-domain evaluation
        y_true_in = [s.label for s in in_domain_records]
        in_preds, in_probs = [], []
        for s in in_domain_records:
            res = self.detector.verify_sample(
                text=s.text, image_source=s.image_path, audio_source=s.audio_path, fusion_strategy="cross_attention"
            )
            in_preds.append(res["final_prediction"])
            in_probs.append(res["probabilities"]["FAKE"])
        in_metrics = MetricsCalculator.calculate_metrics(y_true_in, in_preds, in_probs)

        # 2. Out-of-domain evaluation
        y_true_out = [s.label for s in out_domain_records]
        out_preds, out_probs = [], []
        for s in out_domain_records:
            res = self.detector.verify_sample(
                text=s.text, image_source=s.image_path, audio_source=s.audio_path, fusion_strategy="cross_attention"
            )
            out_preds.append(res["final_prediction"])
            out_probs.append(res["probabilities"]["FAKE"])
        out_metrics = MetricsCalculator.calculate_metrics(y_true_out, out_preds, out_probs)

        # Performance delta
        acc_drop = round((in_metrics["accuracy"] - out_metrics["accuracy"]) * 100, 2)
        f1_drop = round((in_metrics["f1_score"] - out_metrics["f1_score"]) * 100, 2)

        chart = MetricsCalculator.render_comparison_bar_b64(
            models=["In-Domain (Familiar)", "Out-Of-Domain (Unseen)"],
            accuracies=[in_metrics["accuracy"], out_metrics["accuracy"]],
            f1_scores=[in_metrics["f1_score"], out_metrics["f1_score"]],
            title="Experiment 3: Generalization Across Manipulation Domains"
        )

        return {
            "experiment_id": "exp3_generalization",
            "title": "Generalization Across Unseen Manipulation Types",
            "in_domain": {
                "categories": ["authentic", "face_manipulation", "misleading_caption"],
                "sample_count": len(in_domain_records),
                "metrics": in_metrics
            },
            "out_of_domain": {
                "categories": ["semantic_mismatch", "synthetic_speech", "ai_generated_image"],
                "sample_count": len(out_domain_records),
                "metrics": out_metrics
            },
            "generalization_gap": {
                "accuracy_drop_pct": acc_drop,
                "f1_drop_pct": f1_drop,
                "robustness_assessment": "Resilient" if acc_drop < 15.0 else "Vulnerable to Domain Shift"
            },
            "chart": chart
        }

