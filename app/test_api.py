import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to python path to import app and src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pre-test check: Ensure we have a model trained so FastAPI doesn't crash on startup!
def ensure_model_is_registered():
    """
    Checks if a model is registered. If not, programmatically generates
    a quick dummy dataset, trains a pipeline, and registers it.
    This guarantees that pytest will pass even on clean CI/CD servers!
    """
    from src.pipeline_builder import REGISTRY_DIR, METADATA_PATH
    import json
    
    metadata_exists = os.path.exists(METADATA_PATH)
    
    if not metadata_exists:
        print("\n[Test Suite Setup] No model registered. Programmatically training a baseline model for testing...")
        
        # 1. Generate small dataset
        from data.generate_data import generate_churn_dataset
        df = generate_churn_dataset(num_samples=100)
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        
        # 2. Build and fit pipeline
        from sklearn.linear_model import LogisticRegression
        from src.pipeline_builder import create_production_pipeline, save_model_to_registry
        
        model = LogisticRegression(random_state=42)
        pipeline = create_production_pipeline(model)
        pipeline.fit(X, y)
        
        # 3. Save to registry
        metrics = {"Accuracy": 0.8, "Precision": 0.8, "Recall": 0.8, "F1-Score": 0.8, "ROC-AUC": 0.8}
        save_model_to_registry(
            pipeline, 
            model_name="Test Suite Baseline", 
            hyperparameters={"C": 1.0}, 
            test_metrics=metrics,
            description="Automatic baseline trained by Pytest."
        )
        print("[Test Suite Setup] Dummy model registered successfully.\n")

# Run registration check before importing app so it loads successfully
ensure_model_is_registered()

# Now we can safely import app
from app.main import app

@pytest.fixture(scope="module")
def client():
    """
    Fixture that yields a TestClient using a context manager.
    This guarantees that the FastAPI lifespan startup and shutdown
    handlers are executed properly before and after tests run!
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Test 1: Verify the health check endpoint returns 200 OK and loaded model details.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_predict_success(client):
    """
    Test 2: Verify prediction succeeds with a fully valid input payload.
    """
    payload = {
        "Age": 30,
        "Gender": "Male",
        "Subscription_Length": 12,
        "Monthly_Charges": 85.5,
        "Usage": 220,
        "Complaints": 0,
        "Payment_Method": "Credit Card"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["will_churn"] in ["YES", "NO"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "model_version" in data
    assert "model_type" in data

def test_predict_imputation_success(client):
    """
    Test 3: Verify prediction succeeds even when Age and Monthly_Charges are missing.
    This validates that our sklearn SimpleImputer works perfectly in production!
    """
    payload = {
        "Age": None,             # Missing
        "Gender": "Female",
        "Subscription_Length": 6,
        "Monthly_Charges": None,  # Missing
        "Usage": 50,
        "Complaints": 2,
        "Payment_Method": "Mailed Check"
    }
    response = client.post("/predict", json=payload)
    # The pipeline should handle the missing fields, impute them, and predict without throwing error
    assert response.status_code == 200
    
    data = response.json()
    assert data["will_churn"] in ["YES", "NO"]
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_predict_validation_error(client):
    """
    Test 4: Verify Pydantic validation rejects bad types (e.g. Complaints must be int, not string)
    and returns a standard 422 error.
    """
    payload = {
        "Age": 28,
        "Gender": "Female",
        "Subscription_Length": 12,
        "Monthly_Charges": 75.0,
        "Usage": 100,
        "Complaints": "three",  # Invalid type: should be integer!
        "Payment_Method": "Bank Transfer"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    
    # Assert the response body indicates a validation error at 'Complaints'
    data = response.json()
    assert "detail" in data
    error_detail = data["detail"][0]
    assert "Complaints" in error_detail["loc"]

