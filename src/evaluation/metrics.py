"""
Evaluation and Metrics Module
Calculates standard classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix)
and renders dynamic chart visualizations via Matplotlib encoded as base64 images.
"""

import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for server execution
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
)


class MetricsCalculator:
    """Calculates all official classification metrics and generates evaluation charts."""

    @staticmethod
    def calculate_metrics(
        y_true: List[str],  # List of "REAL" or "FAKE"
        y_pred: List[str],  # List of "REAL", "FAKE", or "UNCERTAIN"
        y_prob_fake: List[float]  # Probability of FAKE class
    ) -> Dict[str, Any]:
        """Calculates comprehensive metrics from genuine predictions."""
        total = len(y_true)
        if total == 0:
            return {}

        # Track abstention
        uncertain_count = sum(1 for p in y_pred if p == "UNCERTAIN")
        abstention_rate = uncertain_count / total

        # Map labels to binary (REAL: 0, FAKE: 1)
        # For evaluation of decisive predictions, treat UNCERTAIN as incorrect or evaluate filtered set
        y_true_bin = [1 if y == "FAKE" else 0 for y in y_true]
        
        # Binary prediction (if uncertain, default to 1-y_true to penalize abstention in strict mode, or 0.5)
        y_pred_bin = []
        for p in y_pred:
            if p == "FAKE":
                y_pred_bin.append(1)
            elif p == "REAL":
                y_pred_bin.append(0)
            else:
                y_pred_bin.append(1)  # Strict penalty

        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(y_true_bin, y_prob_fake)
        except Exception:
            roc_auc = 0.5

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true_bin, y_prob_fake)
            pr_auc = auc(r_curve, p_curve)
        except Exception:
            pr_auc = 0.5

        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

        # Class-wise metrics
        real_prec = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
        real_rec = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
        real_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)

        return {
            "total_samples": total,
            "abstention_count": uncertain_count,
            "abstention_rate": round(abstention_rate, 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
            "class_metrics": {
                "REAL": {"precision": round(float(real_prec), 4), "recall": round(float(real_rec), 4), "f1": round(float(real_f1), 4)},
                "FAKE": {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4)}
            }
        }

    @staticmethod
    def render_confusion_matrix_b64(cm: List[List[int]], title: str = "Confusion Matrix") -> str:
        """Renders VoiceBox-style high-contrast confusion matrix heatmap as base64 PNG."""
        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=120)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        
        ax.set_title(title.upper(), fontsize=11, fontweight="bold", pad=12, color="#0A0A0A")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["REAL", "FAKE"], fontweight="bold", color="#0A0A0A")
        ax.set_yticklabels(["REAL", "FAKE"], fontweight="bold", color="#0A0A0A")
        ax.set_xlabel("PREDICTED CLASS", fontweight="bold", fontsize=9, labelpad=8, color="#525252")
        ax.set_ylabel("TRUE CLASS", fontweight="bold", fontsize=9, labelpad=8, color="#525252")

        # Text in squares
        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, format(cm_arr[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "#0A0A0A",
                    fontweight="bold", fontsize=14
                )

        # Thick 2px border around plot
        for spine in ax.spines.values():
            spine.set_edgecolor("#0A0A0A")
            spine.set_linewidth(2)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def render_comparison_bar_b64(
        models: List[str],
        accuracies: List[float],
        f1_scores: List[float],
        title: str = "Model Performance Comparison"
    ) -> str:
        """Renders high-contrast editorial bar comparison chart as base64 PNG."""
        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        x = np.arange(len(models))
        width = 0.35

        rects1 = ax.bar(x - width/2, [a * 100 for a in accuracies], width, label="Accuracy (%)", color="#0A0A0A", edgecolor="#0A0A0A", linewidth=1.5)
        rects2 = ax.bar(x + width/2, [f * 100 for f in f1_scores], width, label="F1-Score (%)", color="#EF4444", edgecolor="#EF4444", linewidth=1.5)

        ax.set_title(title.upper(), fontsize=11, fontweight="bold", pad=12, color="#0A0A0A")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight="bold", fontsize=9, color="#0A0A0A", rotation=15)
        ax.set_ylim(0, 105)
        ax.set_ylabel("PERCENTAGE (%)", fontweight="bold", fontsize=9, color="#525252")
        ax.legend(frameon=True, edgecolor="#0A0A0A")

        # Thick 2px border around plot
        for spine in ax.spines.values():
            spine.set_edgecolor("#0A0A0A")
            spine.set_linewidth(2)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

