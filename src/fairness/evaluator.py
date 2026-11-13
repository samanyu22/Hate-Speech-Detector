from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from src.data.adapter import DatasetRecord

class BiasAndFairnessEvaluator:
    @staticmethod
    def evaluate_fairness(records: List[DatasetRecord], predict_fn) -> Dict[str, Any]:
        """
        Calculates disaggregated performance metrics and fairness disparities across linguistic subgroups.
        """
        groups: Dict[str, List[DatasetRecord]] = {
            "english": [r for r in records if r.language == "en"],
            "hindi_native": [r for r in records if r.language == "hi"],
            "hinglish_code_mixed": [r for r in records if r.language == "hi-en"]
        }

        group_metrics = {}
        fpr_by_group = {}
        fnr_by_group = {}

        for g_name, g_records in groups.items():
            if not g_records:
                continue

            labels = [r.label for r in g_records]
            preds = [predict_fn(r.text)["label"] for r in g_records]

            acc = accuracy_score(labels, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            fpr = fp / max(1, fp + tn)
            fnr = fn / max(1, fn + tp)

            fpr_by_group[g_name] = round(float(fpr), 4)
            fnr_by_group[g_name] = round(float(fnr), 4)

            group_metrics[g_name] = {
                "sample_count": len(g_records),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "fpr": round(float(fpr), 4),
                "fnr": round(float(fnr), 4),
                "confusion_matrix": cm.tolist()
            }

        # Calculate Disparities (Equalized Odds Gap)
        fpr_vals = list(fpr_by_group.values())
        fnr_vals = list(fnr_by_group.values())

        max_fpr_disparity = round(float(max(fpr_vals) - min(fpr_vals)), 4) if fpr_vals else 0.0
        max_fnr_disparity = round(float(max(fnr_vals) - min(fnr_vals)), 4) if fnr_vals else 0.0
        equalized_odds_gap = max(max_fpr_disparity, max_fnr_disparity)

        return {
            "group_metrics": group_metrics,
            "fpr_by_group": fpr_by_group,
            "fnr_by_group": fnr_by_group,
            "max_fpr_disparity": max_fpr_disparity,
            "max_fnr_disparity": max_fnr_disparity,
            "equalized_odds_gap": equalized_odds_gap,
            "is_fair_under_threshold": equalized_odds_gap < 0.15
        }

