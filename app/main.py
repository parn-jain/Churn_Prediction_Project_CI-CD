import os
import sys
from typing import Optional
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add project root to python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline_builder import load_model_from_registry

# Global variable to hold our loaded ML Pipeline in RAM
model_pipeline = None
model_metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the model once on startup into RAM so predictions are blazing fast.
    If the model registry is missing, it dynamically runs on-the-fly training!
    """
    global model_pipeline, model_metadata
    
    # On-the-Fly Training Safeguard
    from src.pipeline_builder import METADATA_PATH
    if not os.path.exists(METADATA_PATH):
        print("[FastAPI Startup] Model registry not found! Running pipeline orchestrator on-the-fly...")
        try:
            import run_interactive
            run_interactive.run_pipeline_orchestrator()
            print("[FastAPI Startup] On-the-fly model training and registration complete!")
        except Exception as train_error:
            print(f"[FastAPI Startup ERROR] On-the-fly training failed: {str(train_error)}")
            
    print("[FastAPI Startup] Loading production ML model from registry...")
    try:
        model_pipeline, model_metadata = load_model_from_registry()
        print(f"[FastAPI Startup] Successfully loaded {model_metadata['model_name']} (v{model_metadata['version']})")
    except Exception as e:
        print(f"[FastAPI Startup ERROR] Failed to load model: {str(e)}")
        print("Backend will start, but predictions will fail until a model is registered.")
    yield
    print("[FastAPI Shutdown] Cleaning up resources...")

# Initialize FastAPI app with custom metadata
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A microservice to predict whether a customer will churn based on subscription usage features.",
    version="1.0.0",
    lifespan=lifespan
)

# 1. Define input schema using Pydantic
class CustomerFeatures(BaseModel):
    Age: Optional[float] = Field(None, description="Age of the customer in years (Optional, will be imputed if missing)", example=34.0)
    Gender: str = Field(..., description="Gender: 'Male' or 'Female'", example="Female")
    Subscription_Length: float = Field(..., description="Subscription duration in months", example=12.0)
    Monthly_Charges: Optional[float] = Field(None, description="Current monthly bill (Optional, will be imputed)", example=85.0)
    Usage: float = Field(..., description="Usage units (e.g. GB data used per month)", example=150.0)
    Complaints: int = Field(..., description="Number of customer service complaints registered (0-5)", example=1)
    Payment_Method: str = Field(..., description="Payment method: 'Credit Card', 'Bank Transfer', or 'Mailed Check'", example="Credit Card")

# 2. Define output schema
class PredictionResponse(BaseModel):
    will_churn: str = Field(..., description="Binary prediction: 'YES' (will churn) or 'NO' (will stay)")
    churn_probability: float = Field(..., description="Float probability score between 0 and 1")
    model_version: int = Field(..., description="The registered model version used for this prediction")
    model_type: str = Field(..., description="Name of the model algorithm used")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serves the beautiful, premium Customer Churn Dashboard web interface.
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        return HTMLResponse("<h1>Templates folder not found!</h1>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/health")
def read_health():
    """
    Health check API endpoint.
    """
    if model_pipeline is None:
        return {
            "status": "warning",
            "message": "API is running, but no production ML model has been registered yet."
        }
    return {
        "status": "healthy",
        "loaded_model": model_metadata["model_name"],
        "model_version": model_metadata["version"],
        "production_since": model_metadata["timestamp"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    """
    Endpoint to receive customer features and return a churn prediction.
    """
    global model_pipeline, model_metadata
    
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model is not loaded/registered yet."
        )
        
    try:
        # Convert incoming validated Pydantic model into a dictionary
        input_data = customer.model_dump()
        
        # Scikit-Learn pipelines expect a 2D-like shape (DataFrame or 2D array) with matching columns!
        # Wrap dictionary in a list to create a single-row DataFrame
        df_row = pd.DataFrame([input_data])
        
        # Run prediction probability through the pipeline
        # predict_proba returns a 2D array: [ [prob_class_0, prob_class_1] ]
        prob_churn = float(model_pipeline.predict_proba(df_row)[0][1])
        
        # Run hard class prediction (0 or 1)
        pred_class = int(model_pipeline.predict(df_row)[0])
        will_churn_str = "YES" if pred_class == 1 else "NO"
        
        return PredictionResponse(
            will_churn=will_churn_str,
            churn_probability=prob_churn,
            model_version=model_metadata["version"],
            model_type=model_metadata["model_name"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing model pipeline prediction: {str(e)}"
        )
