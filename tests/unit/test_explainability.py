import pytest
from src.explainability.text_attribution import TextAttributionEngine
from src.preprocessing.text.normalizer import TextNormalizer
from src.models.transformer_classifier import MultilingualTransformerEngine

@pytest.fixture(scope="module")
def xai_setup():
    norm = TextNormalizer()
    engine = MultilingualTransformerEngine()
    if engine.checkpoint_path.exists():
        engine.load_checkpoint()
    xai = TextAttributionEngine(norm)
    return xai, engine

def test_perturbation_attribution_tokens(xai_setup):
    xai, engine = xai_setup
    text = "These vermin immigrants are destroying our country."
    res = xai.compute_perturbation_attribution(text, engine.predict)
    assert "token_attributions" in res
    assert len(res["token_attributions"]) > 0
    tokens = [t["token"] for t in res["token_attributions"]]
    assert "vermin" in tokens or "immigrants" in tokens

def test_faithfulness_score_bounds(xai_setup):
    xai, engine = xai_setup
    text = "French people are one of the best."
    res = xai.compute_perturbation_attribution(text, engine.predict)
    faithfulness = res.get("faithfulness_score", 0.0)
    assert 0.0 <= faithfulness <= 1.0

def test_empty_string_attribution(xai_setup):
    xai, engine = xai_setup
    res = xai.compute_perturbation_attribution("   ", engine.predict)
    assert res["token_attributions"] == []
    assert res["faithfulness_score"] == 1.0

def test_saturation_fallback_non_zero_attributions(xai_setup):
    """When a model is 100% confident, attributions should still have non-zero sensitivity in logit space."""
    xai, engine = xai_setup
    text = "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech."
    res = xai.compute_perturbation_attribution(text, engine.predict)
    non_zero_scores = [t for t in res["token_attributions"] if abs(t["score"]) > 0.0001]
    assert len(non_zero_scores) > 0, "Logit sensitivity fallback should produce non-zero scores"

