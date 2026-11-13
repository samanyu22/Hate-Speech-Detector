from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from src.data.adapter import UniversalDatasetAdapter
from src.data.splitter import DatasetSplitter
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.fairness.evaluator import BiasAndFairnessEvaluator

router = APIRouter(prefix="/api/fairness", tags=["Fairness & Bias"])

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

        model = MultilingualTransformerEngine()
        model.load_checkpoint()

        report = BiasAndFairnessEvaluator.evaluate_fairness(test_recs, model.predict)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

