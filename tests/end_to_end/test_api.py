import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

def test_detection_api_pipeline():
    payload = {
        "text": "Bhai kal ka cricket match mast tha!",
        "explain": True
    }
    res = client.post("/api/detection/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_label"] in [0, 1]
    assert data["label_name"] in ["HATEFUL", "NON-HATEFUL"]
    assert "explanation" in data
    assert data["language_detected"] in ["hi-en", "en", "hi"]

def test_datasets_listing_endpoint():
    res = client.get("/api/datasets/")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) >= 3

def test_models_listing_endpoint():
    res = client.get("/api/models/")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 1

def test_feedback_and_reward_submission():
    # First generate prediction
    pred_res = client.post("/api/detection/detect", json={"text": "Test statement for audit.", "explain": False})
    assert pred_res.status_code == 200
    pred_id = pred_res.json()["prediction_id"]

    # Submit feedback
    fb_res = client.post("/api/feedback/", json={
        "prediction_id": pred_id,
        "is_correct": True,
        "explanation_rating": 5,
        "comment": "Accurate prediction verification."
    })
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] == "accepted"

