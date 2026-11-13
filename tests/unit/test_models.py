import pytest
from src.classification.classical_baseline import ClassicalBaselineClassifier
from src.models.transformer_classifier import MultilingualTransformerEngine
from src.models.llm_reasoner import LLMReasoningEngine

def test_classical_baseline_predict_structure():
    clf = ClassicalBaselineClassifier()
    if clf.checkpoint_path.exists():
        clf.load()
    res = clf.predict("Bhai kal ka cricket match dekh ke maza aa gaya!")
    assert "label" in res
    assert "label_name" in res
    assert "confidence" in res
    assert "raw_hateful_probability" in res
    assert res["label"] in [0, 1]
    assert 0.0 <= res["confidence"] <= 1.0

def test_multilingual_transformer_predict_structure():
    engine = MultilingualTransformerEngine()
    if engine.checkpoint_path.exists():
        engine.load_checkpoint()
    res = engine.predict("French cuisine is admired worldwide.")
    assert "label" in res
    assert "confidence" in res
    assert "logits" in res
    assert len(res["logits"]) == 2
    assert "content_coverage" in res

def test_llm_reasoner_structured_output():
    llm = LLMReasoningEngine()
    res = llm.analyze("We stand united against hate and violence.", language="en")
    assert res.predicted_label in [0, 1]
    assert res.label_name in ["NON-HATEFUL", "HATEFUL"]
    assert 0.0 <= res.calibrated_confidence <= 1.0
    assert len(res.reasoning_steps) > 0

def test_llm_reasoner_sarcasm_counter_detection():
    llm = LLMReasoningEngine()
    text = "Oh sure, blame every single problem in the world on immigrants, brilliant logic!"
    res = llm.analyze(text, language="en")
    assert res.predicted_label == 0
    assert res.label_name == "NON-HATEFUL"

