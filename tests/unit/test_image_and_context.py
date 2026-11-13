import pytest
from PIL import Image
import numpy as np
from src.preprocessing.context.context_processor import ContextProcessor
from src.preprocessing.image.image_preprocessor import ImagePreprocessor
from src.multimodal.ocr import OCRService

def test_context_processor_formatting():
    proc = ContextProcessor({"mask_demographic_attributes": True})
    ctx_data = {
        "thread_history": ["First message in thread", "Second message in debate"],
        "platform": "twitter",
        "post_type": "reply"
    }
    res = proc.process(ctx_data)
    assert res["has_context"] is True
    assert res["thread_length"] == 2
    assert "First message in thread" in res["formatted_context"]

def test_context_demographic_masking():
    proc = ContextProcessor({"mask_demographic_attributes": True})
    ctx_data = {"thread_history": ["Discussion about Hindu and Muslim festivals"]}
    res = proc.process(ctx_data)
    assert "[DEMOGRAPHIC_NEUTRAL]" in res["formatted_context"]
    assert "Hindu" not in res["formatted_context"]

def test_image_preprocessor_pil():
    prep = ImagePreprocessor(target_size=(128, 128))
    img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    res = prep.inspect_and_preprocess(img)
    assert res["original_resolution"] == [300, 200]
    assert res["target_resolution"] == [128, 128]
    assert 0.0 <= res["mean_brightness"] <= 1.0
    assert isinstance(res["processed_array"], np.ndarray)

def test_ocr_service_clean_call():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    res = OCRService.extract_text(img)
    assert "extracted_text" in res
    assert "bounding_boxes" in res
    assert isinstance(res["extracted_text"], str)
