import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_api_code_mixed_hinglish_flow(client):
    """Test full API detection pipeline for Code-Mixed Hinglish."""
    payload = {
        "text": "Bhai kal ka cricket match dekh ke maza aa gaya, kya mast sixes maare!",
        "model_id": "multilingual-transformer-v1",
        "explain": True
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] == 0
    assert data["label_name"] == "NON-HATEFUL"
    assert data["explanation"] is not None

def test_api_devanagari_hate_flow(client):
    """Test full API detection pipeline for Devanagari Hindi hate speech."""
    payload = {
        "text": "इन देशद्रोहियों को सरेआम गोली मार देनी चाहिए, ये देश के गद्दार हैं।",
        "model_id": "multilingual-transformer-v1",
        "explain": True
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] == 1
    assert data["label_name"] == "HATEFUL"

def test_api_empty_payload_validation(client):
    """API should return 400 when neither text nor image is provided."""
    payload = {
        "text": "",
        "image_path": None,
        "image_base64": None
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 400
    assert "Must provide at least text or image" in res.json()["detail"]

def test_api_classical_model_choice(client):
    """Test API using classical TF-IDF model specifically."""
    payload = {
        "text": "Scientific progress requires international collaboration.",
        "model_id": "classical-tfidf-svm",
        "explain": False
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] == 0
    assert data["label_name"] == "NON-HATEFUL"

def test_api_llm_reasoner_choice(client):
    """Test API using LLM Structured Analysis Reasoner specifically."""
    payload = {
        "text": "Oh sure, blame every single problem in the world on immigrants, brilliant logic!",
        "model_id": "llm-structured-reasoner-v1",
        "explain": True
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] == 0
    assert data["label_name"] == "NON-HATEFUL"

def test_api_feedback_persistence_roundtrip(client):
    """Test submitting feedback and verifying that reward optimization updates."""
    # 1. Run detection
    detect_res = client.post("/api/detection/detect", json={"text": "Community clean-up keeps parks beautiful."})
    pred_id = detect_res.json()["prediction_id"]

    # 2. Submit feedback
    fb_payload = {
        "prediction_id": pred_id,
        "is_correct": True,
        "suggested_label": None,
        "explanation_rating": 5,
        "comment": "Accurate civil classification"
    }
    fb_res = client.post("/api/feedback/", json=fb_payload)
    assert fb_res.status_code == 200
    fb_data = fb_res.json()
    assert fb_data["status"] == "accepted"

