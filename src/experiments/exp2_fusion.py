"""
Experiment 2: Multimodal Fusion Strategy Comparison
Compares Feature-Level, Cross-Attention, and Decision-Level Fusion architectures
under identical test conditions.
"""

import time
from typing import Dict, List, Any
from src.data.dataset_manager import DatasetManager
from src.classification.detector import MultimodalVerificationSystem
from src.evaluation.metrics import MetricsCalculator


class Experiment2Runner:
    """Executes Experiment 2."""

    def __init__(self, detector: MultimodalVerificationSystem, dataset_manager: DatasetManager):
        self.detector = detector
        self.dm = dataset_manager

    def run(self, dataset_id: str = "benchmark_300") -> Dict[str, Any]:
        """Trains and compares all 3 multimodal fusion strategies."""
        train, val, test = self.dm.get_splits(dataset_id)
        
        # Ensure multimodal models are trained
        self.detector.fit_multimodal(train, val, fusion_strategy="feature_level", epochs=3)
        self.detector.fit_multimodal(train, val, fusion_strategy="cross_attention", epochs=3)

        y_true = [s.label for s in test]

        # 1. Feature-Level Fusion Evaluation
        start_t = time.time()
        fl_preds, fl_probs = [], []
        for s in test:
            res = self.detector.verify_sample(
                text=s.text, image_source=s.image_path, audio_source=s.audio_path, fusion_strategy="feature_level"
            )
            fl_preds.append(res["final_prediction"])
            fl_probs.append(res["probabilities"]["FAKE"])
        fl_latency_ms = (time.time() - start_t) / len(test) * 1000.0
        fl_metrics = MetricsCalculator.calculate_metrics(y_true, fl_preds, fl_probs)

        # 2. Cross-Attention Fusion Evaluation
        start_t = time.time()
        ca_preds, ca_probs = [], []
        for s in test:
            res = self.detector.verify_sample(
                text=s.text, image_source=s.image_path, audio_source=s.audio_path, fusion_strategy="cross_attention"
            )
            ca_preds.append(res["final_prediction"])
            ca_probs.append(res["probabilities"]["FAKE"])
        ca_latency_ms = (time.time() - start_t) / len(test) * 1000.0
        ca_metrics = MetricsCalculator.calculate_metrics(y_true, ca_preds, ca_probs)

        # 3. Decision-Level Late Fusion Evaluation
        start_t = time.time()
        dl_preds, dl_probs = [], []
        for s in test:
            res = self.detector.verify_sample(
                text=s.text, image_source=s.image_path, audio_source=s.audio_path, fusion_strategy="decision_level"
            )
            dl_preds.append(res["final_prediction"])
            dl_probs.append(res["probabilities"]["FAKE"])
        dl_latency_ms = (time.time() - start_t) / len(test) * 1000.0
        dl_metrics = MetricsCalculator.calculate_metrics(y_true, dl_preds, dl_probs)

        # Comparison Chart
        models = ["Feature-Level", "Cross-Attention", "Decision-Level"]
        accs = [fl_metrics["accuracy"], ca_metrics["accuracy"], dl_metrics["accuracy"]]
        f1s = [fl_metrics["f1_score"], ca_metrics["f1_score"], dl_metrics["f1_score"]]

        chart = MetricsCalculator.render_comparison_bar_b64(
            models=models,
            accuracies=accs,
            f1_scores=f1s,
            title="Experiment 2: Fusion Architectures Comparison"
        )

        return {
            "experiment_id": "exp2_fusion_comparison",
            "title": "Fusion Strategy Comparison (Feature vs Cross-Attention vs Decision)",
            "test_sample_count": len(test),
            "strategies": {
                "feature_level": {
                    "metrics": fl_metrics,
                    "avg_latency_ms": round(fl_latency_ms, 2),
                    "strengths": "Fast computation, straightforward tensor concatenation with bilinear interaction.",
                    "limitations": "Does not model asymmetric directed attention between modalities."
                },
                "cross_attention": {
                    "metrics": ca_metrics,
                    "avg_latency_ms": round(ca_latency_ms, 2),
                    "strengths": "Bidirectional attention maps allow textual queries to directly probe visual & acoustic evidence.",
                    "limitations": "Slightly higher parameter count and computational footprint."
                },
                "decision_level": {
                    "metrics": dl_metrics,
                    "avg_latency_ms": round(dl_latency_ms, 2),
                    "strengths": "Highly modular; can incorporate off-the-shelf black-box modality classifiers without retraining.",
                    "limitations": "Lacks early and mid-level feature cross-pollination."
                }
            },
            "comparison_chart": chart
        }

