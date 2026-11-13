import io
import base64
import torch
from PIL import Image
from src.preprocessing.text.normalizer import TextNormalizer
from src.preprocessing.image.image_preprocessor import ImagePreprocessor
from src.multimodal.fusion import MultimodalDetector
from backend.schemas.detection import DetectionRequest
from src.language.detector import MultilingualLanguageDetector

def test_devanagari_token_unification():
    norm = TextNormalizer()
    tokens = norm.tokenize_words("नमस्ते दोस्तों ये गद्दार हैं!")
    assert "नमस्ते" in tokens
    assert "गद्दार" in tokens
    assert "दोस्तों" in tokens
    # Ensure virama and matras are not loose punctuation tokens
    assert "्" not in tokens
    assert "ा" not in tokens

def test_image_preprocessor_raw_base64_support():
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    prep = ImagePreprocessor()
    loaded = prep.load_image(raw_b64)
    assert loaded.size == (64, 64)

def test_multimodal_cross_attention_integration():
    detector = MultimodalDetector(dim=128, text_in_dim=256, fusion_type="cross_attention")
    detector.eval()
    t_feat = torch.randn(1, 256)
    img_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = detector(t_feat, img_tensor)
    assert logits.shape == (1, 2)

def test_detection_request_image_only_support():
    req = DetectionRequest(image_path="data/raw/images/meme_counter.png")
    assert req.text == ""
    assert req.image_path == "data/raw/images/meme_counter.png"

def test_hinglish_phonetic_elongation_matching():
    res = MultilingualLanguageDetector.detect("bhaaaai ye deshdrohi log hai")
    assert res["is_code_mixed"] is True
    assert res["language"] == "hi-en"

