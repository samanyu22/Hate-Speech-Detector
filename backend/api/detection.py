import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from backend.schemas.detection import DetectionRequest, DetectionResponse, ExplanationResponse, TokenAttribution, VisualAttribution
from backend.core.config import settings
from backend.database.db import get_connection

from src.language.detector import MultilingualLanguageDetector
from src.preprocessing.text.normalizer import TextNormalizer
from src.preprocessing.context.context_processor import ContextProcessor
from src.preprocessing.image.image_preprocessor import ImagePreprocessor
from src.multimodal.ocr import OCRService
from src.classification.classical_baseline import ClassicalBaselineClassifier
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.models.llm_reasoner import LLMReasoningEngine
from src.explainability.text_attribution import TextAttributionEngine
from src.explainability.visual_attribution import VisualAttributionEngine
from src.explainability.synthesizer import ExplanationSynthesizer

router = APIRouter(prefix="/api/detection", tags=["Detection"])

# Singleton model instances
_normalizer = TextNormalizer()
_context_proc = ContextProcessor()
_image_proc = ImagePreprocessor()
_llm_reasoner = LLMReasoningEngine()
_xai_engine = TextAttributionEngine(_normalizer)

_classical_model: Optional[ClassicalBaselineClassifier] = None
_transformer_model: Optional[MultilingualTransformerEngine] = None

def get_models(force_reload: bool = False):
    global _classical_model, _transformer_model
    if _classical_model is None or force_reload:
        _classical_model = ClassicalBaselineClassifier()
        if _classical_model.checkpoint_path.exists():
            _classical_model.load()
    if _transformer_model is None or force_reload:
        _transformer_model = MultilingualTransformerEngine()
        if _transformer_model.checkpoint_path.exists():
            _transformer_model.load_checkpoint()
    return _classical_model, _transformer_model

@router.post("/detect", response_model=DetectionResponse)
def detect_content(req: DetectionRequest):
    classical_clf, transformer_clf = get_models()
    if not req.text and not req.image_path and not req.image_base64:
        raise HTTPException(status_code=400, detail="Must provide at least text or image content.")

    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    modalities_used = []

    # 1. Modality: Text Preprocessing & Language Detection
    clean_text = _normalizer.normalize(req.text)
    lang_info = MultilingualLanguageDetector.detect(clean_text)
    lang = lang_info["language"]
    if clean_text:
        modalities_used.append("text")

    # 2. Modality: Context Processing
    ctx_info = _context_proc.process(req.context)
    if ctx_info["has_context"]:
        modalities_used.append("context")

    # 3. Modality: Visual & OCR
    ocr_result = {"extracted_text": "", "bounding_boxes": [], "confidence": 0.0}
    if req.image_path or req.image_base64:
        try:
            img_source = req.image_path or req.image_base64
            img_props = _image_proc.inspect_and_preprocess(img_source)
            ocr_result = OCRService.extract_text(img_props["pil_image"])
            modalities_used.append("image")
            if ocr_result["extracted_text"]:
                modalities_used.append("ocr")
        except Exception as e:
            print(f"Image processing note: {e}")

    # Combine text and OCR text if present
    full_text = clean_text
    if ocr_result["extracted_text"]:
        full_text = f"{clean_text} [OCR: {ocr_result['extracted_text']}]".strip()

    # 4. Model Selection & Inference
    active_model_id = req.model_id or settings.app_config.get("active_model", {}).get("id", "multilingual-transformer-v1")
    model_version = "1.0"

    if active_model_id == "classical-tfidf-svm" and classical_clf and classical_clf.is_trained:
        pred_res = classical_clf.predict(full_text)
        pred_label = pred_res["label"]
        confidence = pred_res["confidence"]
        active_model_name = "TF-IDF + Linear Classifier"
        predict_fn = classical_clf.predict
    elif active_model_id == "llm-structured-reasoner-v1":
        # Structured LLM analysis
        trans_prob = transformer_clf.predict(full_text) if (transformer_clf and transformer_clf.is_trained) else None
        llm_out = _llm_reasoner.analyze(
            text=clean_text,
            language=lang,
            ocr_text=ocr_result["extracted_text"],
            context=ctx_info["formatted_context"],
            transformer_probs={"hateful": trans_prob["raw_hateful_probability"]} if trans_prob else None
        )
        pred_label = llm_out.predicted_label
        confidence = llm_out.calibrated_confidence
        active_model_name = "LLM Structured Analysis Engine"
        predict_fn = transformer_clf.predict if (transformer_clf and transformer_clf.is_trained) else classical_clf.predict
    else:
        # Default: Multilingual Transformer
        if transformer_clf and transformer_clf.is_trained:
            pred_res = transformer_clf.predict(full_text)
            pred_label = pred_res["label"]
            confidence = pred_res["confidence"]
            predict_fn = transformer_clf.predict
        elif classical_clf and classical_clf.is_trained:
            pred_res = classical_clf.predict(full_text)
            pred_label = pred_res["label"]
            confidence = pred_res["confidence"]
            predict_fn = classical_clf.predict
        else:
            raise HTTPException(status_code=500, detail="No trained model available. Train models in Model Lab.")
        active_model_name = "Multilingual Code-Mixed Transformer"

    # 5. Explainability Computation
    explanation_obj = None
    if req.explain:
        attr_data = _xai_engine.compute_perturbation_attribution(clean_text, predict_fn)
        token_attrs = [
            TokenAttribution(token=a["token"], score=a["score"], polarity=a["polarity"])
            for a in attr_data["token_attributions"]
        ]
        visual_attrs = [
            VisualAttribution(box=v["box"], region_label=v["region_label"], attribution_score=v["attribution_score"])
            for v in VisualAttributionEngine.map_ocr_attributions(ocr_result["bounding_boxes"], attr_data["token_attributions"])
        ]
        synth = ExplanationSynthesizer.synthesize(
            label=pred_label,
            confidence=confidence,
            language=lang,
            token_attributions=attr_data["token_attributions"],
            faithfulness_score=attr_data["faithfulness_score"],
            ocr_text=ocr_result["extracted_text"],
            context_data=ctx_info
        )

        explanation_obj = ExplanationResponse(
            token_attributions=token_attrs,
            visual_attributions=visual_attrs,
            ocr_evidence=ocr_result["extracted_text"] or None,
            contextual_evidence=ctx_info if ctx_info["has_context"] else None,
            natural_language_explanation=synth["natural_language_explanation"],
            faithfulness_score=synth["faithfulness_score"],
            uncertainty_notes="High contextual sensitivity." if confidence < 0.70 else "Grounded in observed token attributions."
        )

    # 6. Persistent DB Record
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions 
        (id, input_text, image_path, context_data, predicted_label, confidence, model_id, model_version, language_detected, modalities_used, explanation_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pred_id,
        clean_text,
        req.image_path,
        json.dumps(ctx_info),
        pred_label,
        confidence,
        active_model_id,
        model_version,
        lang,
        json.dumps(modalities_used),
        json.dumps(explanation_obj.model_dump() if explanation_obj else {}),
        now
    ))
    conn.commit()
    conn.close()

    return DetectionResponse(
        prediction_id=pred_id,
        predicted_label=pred_label,
        label_name="HATEFUL" if pred_label == 1 else "NON-HATEFUL",
        confidence=confidence,
        model_id=active_model_id,
        model_version=model_version,
        language_detected=lang,
        modalities_used=modalities_used,
        explanation=explanation_obj,
        created_at=now
    )
