"""
Experiment 4: Modality Ablation Study
Systematically removes one or more modalities to measure their individual and joint
contributions to multimodal deepfake verification.
"""

from typing import Dict, List, Any
from src.data.dataset_manager import DatasetManager
from src.classification.detector import MultimodalVerificationSystem
from src.evaluation.metrics import MetricsCalculator


class Experiment4Runner:
    """Executes Experiment 4."""

    def __init__(self, detector: MultimodalVerificationSystem, dataset_manager: DatasetManager):
        self.detector = detector
        self.dm = dataset_manager

    def run(self, dataset_id: str = "benchmark_300") -> Dict[str, Any]:
        """Runs systematic modality ablation tests."""
        _, _, test = self.dm.get_splits(dataset_id)
        y_true = [s.label for s in test]

        configurations = [
            {"id": "all_three", "name": "Text + Vision + Audio", "use_t": True, "use_v": True, "use_a": True},
            {"id": "text_vision", "name": "Text + Vision (No Audio)", "use_t": True, "use_v": True, "use_a": False},
            {"id": "text_audio", "name": "Text + Audio (No Vision)", "use_t": True, "use_v": False, "use_a": True},
            {"id": "vision_audio", "name": "Vision + Audio (No Text)", "use_t": False, "use_v": True, "use_a": True},
            {"id": "text_only", "name": "Text Only", "use_t": True, "use_v": False, "use_a": False},
            {"id": "vision_only", "name": "Vision Only", "use_t": False, "use_v": True, "use_a": False},
            {"id": "audio_only", "name": "Audio Only", "use_t": False, "use_v": False, "use_a": True}
        ]

        ablation_results = {}
        names = []
        accuracies = []
        f1_scores = []

        for cfg in configurations:
            preds, probs = [], []
            for s in test:
                t = s.text if cfg["use_t"] else None
                v = s.image_path if cfg["use_v"] else None
                a = s.audio_path if cfg["use_a"] else None

                res = self.detector.verify_sample(
                    text=t, image_source=v, audio_source=a, fusion_strategy="cross_attention"
                )
                preds.append(res["final_prediction"])
                probs.append(res["probabilities"]["FAKE"])

            m = MetricsCalculator.calculate_metrics(y_true, preds, probs)
            ablation_results[cfg["id"]] = {
                "name": cfg["name"],
                "metrics": m
            }
            names.append(cfg["name"])
            accuracies.append(m["accuracy"])
            f1_scores.append(m["f1_score"])

        # Marginal contributions compared to Full (all_three)
        full_f1 = ablation_results["all_three"]["metrics"]["f1_score"]
        contributions = {
            "text_contribution_delta_f1": round(full_f1 - ablation_results["vision_audio"]["metrics"]["f1_score"], 4),
            "vision_contribution_delta_f1": round(full_f1 - ablation_results["text_audio"]["metrics"]["f1_score"], 4),
            "audio_contribution_delta_f1": round(full_f1 - ablation_results["text_vision"]["metrics"]["f1_score"], 4)
        }

        chart = MetricsCalculator.render_comparison_bar_b64(
            models=names,
            accuracies=accuracies,
            f1_scores=f1_scores,
            title="Experiment 4: Modality Ablation Analysis"
        )

        return {
            "experiment_id": "exp4_ablation",
            "title": "Modality Ablation Study",
            "sample_count": len(test),
            "ablation_table": ablation_results,
            "marginal_contributions": contributions,
            "chart": chart
        }

