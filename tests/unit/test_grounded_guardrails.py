import pytest
from src.models.transformer_classifier import MultilingualTransformerEngine

@pytest.fixture(scope="module")
def transformer_engine():
    engine = MultilingualTransformerEngine()
    if engine.checkpoint_path.exists():
        engine.load_checkpoint()
    return engine

def test_general_english_monologue_not_flagged(transformer_engine):
    """Test that open-domain non-hateful psychological/casual monologues are not falsely flagged."""
    text = "People fake a lot of human interactions but I feel like I fake them all. And I fake them very well"
    res = transformer_engine.predict(text)
    assert res["label"] == 0, f"Expected NON-HATEFUL, got {res['label_name']}"
    assert res["label_name"] == "NON-HATEFUL"

def test_neat_monster_playful_phrase(transformer_engine):
    """Test that playful self-descriptions without hate targets are classified as NON-HATEFUL."""
    text = "I'm a very neat monster."
    res = transformer_engine.predict(text)
    assert res["label"] == 0
    assert res["label_name"] == "NON-HATEFUL"

def test_counterspeech_quotation_critique(transformer_engine):
    """Test that academic quotation of hate speech to condemn it is recognized as NON-HATEFUL."""
    text = "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech."
    res = transformer_engine.predict(text)
    assert res["label"] == 0
    assert res["label_name"] == "NON-HATEFUL"

def test_hate_endorsement_detected(transformer_engine):
    """Test that explicitly endorsing or calling hate speech acceptable is classified as HATEFUL."""
    text = "Someone wrote 'these immigrants are vermin' which is completely acceptable racist hate speech."
    res = transformer_engine.predict(text)
    assert res["label"] == 1
    assert res["label_name"] == "HATEFUL"

def test_xenophobic_sensory_insult_detected(transformer_engine):
    """Test that sensory/hygiene slurs targeting a protected nationality are classified as HATEFUL."""
    text = "italian people stinks"
    res = transformer_engine.predict(text)
    assert res["label"] == 1
    assert res["label_name"] == "HATEFUL"

def test_cultural_appreciation_not_flagged(transformer_engine):
    """Test that cultural appreciation and national praise are classified as NON-HATEFUL."""
    text = "french people are the best"
    res = transformer_engine.predict(text)
    assert res["label"] == 0
    assert res["label_name"] == "NON-HATEFUL"

