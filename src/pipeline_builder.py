import os
import json
import pickle
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Compute the project root directory dynamically relative to this source file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_DIR = os.path.join(PROJECT_ROOT, "models", "registry")
METADATA_PATH = os.path.join(REGISTRY_DIR, "metadata.json")

def create_production_pipeline(model_instance):
    """
    CONCEPT 9: SCIKIT-LEARN PIPELINE OBJECT
    Binds Imputation, Scaling, Encoding, and the Model into a single executable object.
    Notice that we build Imputation directly inside the pipeline now! This means our
    production API can receive incomplete raw data and clean it automatically.
    """
    # 1. Define preprocessors for Numerical columns
    # We impute missing numerical values with the median, then scale
    numerical_cols = ["Age", "Subscription_Length", "Monthly_Charges", "Usage", "Complaints"]
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # 2. Define preprocessors for Categorical columns
    # We encode nominal categories using OneHotEncoder with drop='first'
    categorical_cols = ["Gender", "Payment_Method"]
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(drop="first", sparse_output=False))
    ])
    
    # 3. Combine both preprocessors into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    # 4. Bind the preprocessor and the Model into a final Pipeline
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model_instance)
    ])
    
    return full_pipeline

def save_model_to_registry(pipeline, model_name, hyperparameters, test_metrics, description=""):
    """
    CONCEPT 10: MODEL VERSIONING & REGISTRY
    Saves the pipeline pickle file and logs its metadata to a centralized JSON registry.
    """
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    
    # 1. Read existing metadata or initialize new
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {"versions": {}, "latest_version": 0, "production_version": 0}
    else:
        metadata = {"versions": {}, "latest_version": 0, "production_version": 0}
        
    # 2. Determine new version number
    new_version = metadata.get("latest_version", 0) + 1
    model_filename = f"pipeline_v{new_version}.pkl"
    model_filepath = os.path.join(REGISTRY_DIR, model_filename)
    
    # 3. Serialize and save the full pipeline object
    print(f"\n[Registry] Saving model pipeline pickle to {model_filepath}...")
    with open(model_filepath, "wb") as f:
        pickle.dump(pipeline, f)
        
    # 4. Prepare metadata log
    version_entry = {
        "version": new_version,
        "filename": model_filename,
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "metrics": test_metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "status": "candidate"
    }
    
    # First model registered automatically becomes production-ready
    if metadata.get("production_version", 0) == 0:
        metadata["production_version"] = new_version
        version_entry["status"] = "production"
        
    metadata["versions"][str(new_version)] = version_entry
    metadata["latest_version"] = new_version
    
    # 5. Write updated metadata to JSON
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[Registry] Successfully registered v{new_version} ({model_name}) in metadata.json!")
    print(f"[Registry] Production version is currently set to: v{metadata['production_version']}")
    return new_version

def load_model_from_registry(version=None):
    """
    Loads a specific pipeline model version (or the production version if None) from the registry.
    """
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Model Registry metadata.json not found! Train a model first.")
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    if version is None:
        version = metadata.get("production_version", 1)
        
    version_key = str(version)
    if version_key not in metadata["versions"]:
        raise ValueError(f"Version {version} not found in model registry.")
        
    model_filename = metadata["versions"][version_key]["filename"]
    model_filepath = os.path.join(REGISTRY_DIR, model_filename)
    
    print(f"[Registry] Loading model pipeline {model_filename} (Version {version})...")
    with open(model_filepath, "rb") as f:
        pipeline = pickle.load(f)
        
    return pipeline, metadata["versions"][version_key]

def set_production_version(version):
    """
    Updates the production pointer in metadata.json to designate a specific version.
    """
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Model Registry metadata.json not found!")
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    version_key = str(version)
    if version_key not in metadata["versions"]:
        raise ValueError(f"Version {version} not found in model registry.")
        
    # Update statuses
    for v_str in metadata["versions"]:
        if v_str == version_key:
            metadata["versions"][v_str]["status"] = "production"
        elif metadata["versions"][v_str]["status"] == "production":
            metadata["versions"][v_str]["status"] = "candidate"
            
    metadata["production_version"] = int(version)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[Registry] Production pointer successfully moved to version v{version}!")

if __name__ == "__main__":
    # Test pipeline builder structure (dummy test)
    clf = RandomForestClassifier(max_depth=3, random_state=42)
    pipeline = create_production_pipeline(clf)
    print("Pre-fit Production Pipeline structure:")
    print(pipeline)
