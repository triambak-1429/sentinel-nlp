from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_analyze_pii_detection():
    with TestClient(app) as client:
        payload = {"prompt": "User admin email is admin@company.com with key sk-live-9988776655"}
        response = client.post("/analyze", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["contains_pii"] is True
        assert data["risk_score"] > 0.0
        assert "latency_ms" in data

def test_empty_prompt_validation():
    with TestClient(app) as client:
        response = client.post("/analyze", json={"prompt": "   "})
        assert response.status_code == 400