# STAGE 5 (Deployment) — DEPLOYMENT TEST

import importlib

import pytest
from fastapi.testclient import TestClient


deploy = importlib.import_module("05_deploy")

client = TestClient(deploy.app)

def test_health_check_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_source"].startswith(("mlflow:", "pickle:"))


# Prediction
def test_predict_phishing_like_email():
    payload = {
        "text": "URGENT: your account is suspended. "
                "Verify your password now: http://secure-login-update.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in ("phishing email", "safe email")
    assert body["label"] in (0, 1)
    assert body["confidence_type"] in ("probability", "decision_margin", "unavailable")

    if body["confidence_type"] == "probability":
        assert body["confidence"] is not None
        assert 0.0 <= body["confidence"] <= 1.0
    elif body["confidence_type"] == "decision_margin":
        assert body["decision_score"] is not None
        assert body["decision_score"] >= 0.0
        assert body["confidence"] is None  


def test_predict_safe_like_email():
    payload = {
        "text": "Hi team, attached are the meeting notes from Tuesday. "
                "Let me know if I missed anything. Best, Anna"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["label"] in (0, 1)


# Validation
def test_empty_text_is_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400  


def test_missing_text_field_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422  


def test_wrong_type_is_rejected():
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422  # text must be a string


# loggingevery prediction
def test_prediction_is_logged(tmp_path, monkeypatch):
    
    log_file = tmp_path / "predictions.log"
    monkeypatch.setattr(deploy, "LOG_FILE", str(log_file))

    client.post("/predict", json={"text": "win a free prize, click here"})

    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1  


if __name__ == "__main__":
    
    import sys
    sys.exit(pytest.main([__file__, "-v"]))