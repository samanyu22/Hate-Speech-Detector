import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from fastapi import APIRouter, HTTPException, Depends
import torch
import torch.nn.functional as F

from backend.schemas.detection import DetectionRequest, DetectionResponse, ExplanationResponse, TokenAttribution, VisualAttribution
from backend.core.config import settings
from backend.database.db import get_connection

from src.language.detector import MultilingualLanguageDetector
from src.preprocessing.text.normalizer import TextNormalizer
from src.preprocessing.context.context_processor import ContextProcessor
from src.preprocessing.image.image_preprocessor import ImagePreprocessor
from src.multimodal.ocr import OCRService
from src.multimodal.fusion import MultimodalDetector
from src.classification.classical_baseline import ClassicalBaselineClassifier
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.models.llm_reasoner import LLMReasoningEngine
from src.explainability.text_attribution import TextAttributionEngine
from src.explainability.visual_attribution import VisualAttributionEngine
from src.explainability.synthesizer import ExplanationSynthesizer

router = APIRouter(prefix="/api/detection", tags=["Detection"])

# Singleton model & processor instances
_normalizer = TextNormalizer()
_context_proc = ContextProcessor()
_image_proc = ImagePreprocessor()
_llm_reasoner = LLMReasoningEngine()
_xai_engine = TextAttributionEngine(_normalizer)
_multimodal_detector = MultimodalDetector(dim=128, text_in_dim=256)

_classical_model: Optional[ClassicalBaselineClassifier] = None
_transformer_model: Optional[MultilingualTransformerEngine] = None
_current_loaded_checkpoint: Optional[str] = None

def get_active_model_registry_info() -> Dict[str, Any]:
    """Retrieve active model info dynamically from SQLite registry."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM model_registry WHERE is_active = 1 LIMIT 1").fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "id": settings.app_config.get("active_model", {}).get("id", "multilingual-transformer-v1"),
        "checkpoint_path": str(settings.base_dir / "models" / "checkpoints" / "multilingual_transformer.pt"),
        "version": "1.0"
    }

def get_models(force_reload: bool = False) -> Tuple[Optional[ClassicalBaselineClassifier], Optional[MultilingualTransformerEngine]]:
    global _classical_model, _transformer_model, _current_loaded_checkpoint
    
    active_info = get_active_model_registry_info()
    target_ckpt = active_info.get("checkpoint_path") or str(settings.base_dir / "models" / "checkpoints" / "multilingual_transformer.pt")

    if _classical_model is None or force_reload:
        _classical_model = ClassicalBaselineClassifier()
        if _classical_model.checkpoint_path.exists():
            _classical_model.load()

    need_reload = (
        _transformer_model is None or
        force_reload or
        _current_loaded_checkpoint != target_ckpt
    )

    if need_reload:
        _transformer_model = MultilingualTransformerEngine(checkpoint_path=target_ckpt)
        if Path(target_ckpt).exists():
            _transformer_model.load_checkpoint(Path(target_ckpt))
            _current_loaded_checkpoint = target_ckpt
        elif _transformer_model.checkpoint_path.exists():
            _transformer_model.load_checkpoint()
            _current_loaded_checkpoint = str(_transformer_model.checkpoint_path)

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
    clean_text = _normalizer.normalize(req.text or "")
    lang_info = MultilingualLanguageDetector.detect(clean_text)
    lang = lang_info["language"]
    if clean_text:
        modalities_used.append("text")

    # 2. Modality: Context Processing
    ctx_info = _context_proc.process(req.context)
    if ctx_info["has_context"]:
        modalities_used.append("context")

    # 3. Modality: Visual, OCR & Image Feature Tensor
    ocr_result = {"extracted_text": "", "bounding_boxes": [], "confidence": 0.0}
    img_tensor = None
    saved_image_path = req.image_path

    if req.image_path or req.image_base64:
        try:
            img_source = req.image_path or req.image_base64
            img_props = _image_proc.inspect_and_preprocess(img_source)
            ocr_result = OCRService.extract_text(img_props["pil_image"])
            modalities_used.append("image")
            if ocr_result["extracted_text"]:
                modalities_used.append("ocr")

            # Convert processed array [224, 224, 3] to PyTorch tensor [1, 3, 224, 224]
            arr = img_props["processed_array"]
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

            # Persist incoming base64 image to data/uploads for persistence & auditing
            if req.image_base64 and not req.image_path:
                uploads_dir = settings.base_dir / "data" / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                saved_file = uploads_dir / f"{pred_id}.png"
                img_props["pil_image"].save(saved_file)
                saved_image_path = str(saved_file)
        except Exception as e:
            print(f"Image processing note: {e}")

    # Combine text and OCR text if present
    full_text = clean_text
    if ocr_result["extracted_text"]:
        full_text = f"{clean_text} [OCR: {ocr_result['extracted_text']}]".strip()

    # 4. Model Selection & Inference
    active_reg = get_active_model_registry_info()
    active_model_id = req.model_id or active_reg.get("id", "multilingual-transformer-v1")
    model_version = active_reg.get("version", "1.0")

    if active_model_id == "classical-tfidf-svm" and classical_clf and classical_clf.is_trained:
        pred_res = classical_clf.predict(full_text)
        pred_label = pred_res["label"]
        confidence = pred_res["confidence"]
        predict_fn = classical_clf.predict
    elif active_model_id == "llm-structured-reasoner-v1":
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
        predict_fn = transformer_clf.predict if (transformer_clf and transformer_clf.is_trained) else classical_clf.predict
    else:
        # Default: Multilingual Attention Transformer with Multimodal Fusion if visual present
        if transformer_clf and transformer_clf.is_trained:
            pred_res = transformer_clf.predict(full_text)
            
            # Incorporate Cross-Modal CNN Feature Fusion when image tensor is available
            if img_tensor is not None:
                try:
                    text_feat = transformer_clf.encode_text(full_text)
                    with torch.no_grad():
                        mm_logits = _multimodal_detector(text_feat, img_tensor)
                        mm_probs = F.softmax(mm_logits, dim=-1).cpu().numpy()[0]
                    p_mm_hate = float(mm_probs[1])
                    # Calibrate text probability with visual CNN representation (70% text / 30% visual cross-attention)
                    fused_prob = round(0.70 * pred_res["raw_hateful_probability"] + 0.30 * p_mm_hate, 4)
                    pred_label = 1 if fused_prob >= 0.50 else 0
                    confidence = fused_prob if pred_label == 1 else round(1.0 - fused_prob, 4)
                    modalities_used.append("visual_cnn_fusion")
                except Exception as e:
                    print(f"Multimodal cross-attention fallback: {e}")
                    pred_label = pred_res["label"]
                    confidence = pred_res["confidence"]
            else:
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
        saved_image_path,
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
