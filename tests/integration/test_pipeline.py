import pytest
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.explainability.text_attribution import TextAttributionEngine
from src.explainability.synthesizer import ExplanationSynthesizer

def test_full_detection_and_attribution_pipeline():
    engine = MultilingualTransformerEngine()
    engine.load_checkpoint()

    test_text = "These vermin parasites must be exterminated."
    pred = engine.predict(test_text)
    assert pred["label"] in [0, 1]
    assert 0.0 <= pred["confidence"] <= 1.0

    xai = TextAttributionEngine()
    attr = xai.compute_perturbation_attribution(test_text, engine.predict)
    assert len(attr["token_attributions"]) > 0
    assert 0.0 <= attr["faithfulness_score"] <= 1.0

    synth = ExplanationSynthesizer.synthesize(
        label=pred["label"],
        confidence=pred["confidence"],
        language="en",
        token_attributions=attr["token_attributions"],
        faithfulness_score=attr["faithfulness_score"]
    )
    assert "Classification:" in synth["natural_language_explanation"]
    assert "Faithfulness:" in synth["natural_language_explanation"]

