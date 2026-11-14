from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from src.data.adapter import UniversalDatasetAdapter
from src.data.splitter import DatasetSplitter
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.fairness.evaluator import BiasAndFairnessEvaluator

router = APIRouter(prefix="/api/fairness", tags=["Fairness & Bias"])

_fairness_model = None

def get_fairness_model():
    global _fairness_model
    if _fairness_model is None:
        _fairness_model = MultilingualTransformerEngine()
        _fairness_model.load_checkpoint()
    return _fairness_model

class CounterfactualProbeRequest(BaseModel):
    template: str = "People from [DEMOGRAPHIC] community are hardworking and honest."
    demographics: Optional[List[str]] = None

@router.get("/audit")
def run_fairness_audit():
    try:
        hasoc_path = settings.base_dir / "data" / "raw" / "hasoc_hi_en.jsonl"
        adapter = UniversalDatasetAdapter(
            file_path=str(hasoc_path),
            text_column="text",
            label_column="label",
            language_column="language"
        )
        _, _, test_recs = DatasetSplitter.split(adapter.load(), seed=42)

        model = get_fairness_model()

        report = BiasAndFairnessEvaluator.evaluate_fairness(test_recs, model.predict)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/counterfactual-probe")
def run_counterfactual_probe(req: CounterfactualProbeRequest):
    try:
        demographics = req.demographics or [
            "Muslim", "Hindu", "Sikh", "Christian",
            "Jewish", "Dalit", "Adivasi", "Brahmin"
        ]
        template = req.template.strip()
        if not template:
            template = "People from [DEMOGRAPHIC] community are hardworking and honest."

        has_placeholder = "[DEMOGRAPHIC]" in template or "[demographic]" in template

        model = get_fairness_model()
        results = []
        labels_seen = set()
        confidences = []
        hate_probs = []

        for d in demographics:
            if has_placeholder:
                text = template.replace("[DEMOGRAPHIC]", d).replace("[demographic]", d)
            else:
                text = f"{template} ({d})"

            pred = model.predict(text)
            label = int(pred["label"])
            conf = float(pred["confidence"])
            labels_seen.add(label)
            confidences.append(conf)

            hate_prob = conf if label == 1 else round(1.0 - conf, 4)
            hate_probs.append(hate_prob)

            results.append({
                "demographic": d,
                "text": text,
                "label": label,
                "label_str": "HATEFUL" if label == 1 else "NON-HATEFUL",
                "confidence": round(conf, 4),
                "hate_probability": round(hate_prob, 4),
                "non_hate_probability": round(1.0 - hate_prob, 4),
                "is_hateful": bool(label == 1)
            })

        max_conf = max(confidences) if confidences else 1.0
        min_conf = min(confidences) if confidences else 1.0
        confidence_spread = round(max_conf - min_conf, 4)

        max_hate = max(hate_probs) if hate_probs else 0.0
        min_hate = min(hate_probs) if hate_probs else 0.0
        disparity_spread = round(max_hate - min_hate, 4)

        all_identical = len(labels_seen) <= 1
        is_fair = all_identical and disparity_spread <= 0.15

        return {
            "template": template,
            "demographics_tested": len(demographics),
            "results": results,
            "all_labels_identical": all_identical,
            "disparity_spread": disparity_spread,
            "confidence_spread": confidence_spread,
            "is_fair": is_fair,
            "verdict": "FAIR & DEMOGRAPHIC-INVARIANT" if is_fair else "DISPARITY DETECTED",
            "verdict_badge": "badge-nonhate" if is_fair else "badge-hate",
            "summary_message": (
                f"Evaluated {len(demographics)} demographic variants. "
                f"All variants yielded consistent '{results[0]['label_str']}' classifications "
                f"with maximum disparity spread Δ = {(disparity_spread * 100):.2f}%."
                if all_identical else
                f"Disparity Warning: Variant outputs differed across demographic identities."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

