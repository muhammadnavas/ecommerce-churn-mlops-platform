"""
Unit tests for FastAPI Churn Prediction service.
"""

from fastapi.testclient import TestClient
from src.api.main import app, load_artifacts

client = TestClient(app)

def test_api_health():
    load_artifacts()
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] in ["HEALTHY", "DEGRADED"]

def test_predict_endpoint():
    load_artifacts()
    payload = {
        "customer_id": "TEST_CUST_101",
        "recency_days": 15,
        "frequency": 6,
        "monetary": 850.0,
        "avg_order_value": 141.67,
        "total_items": 120,
        "cancellation_rate": 0.0,
        "distinct_products": 18,
        "active_months": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert data["customer_id"] == "TEST_CUST_101"
