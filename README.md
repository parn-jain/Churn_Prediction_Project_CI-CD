---
title: Customer Churn Analytics Dashboard
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Customer Churn Prediction: End-to-End ML Lifecycle & Study Guide

Welcome to the **Customer Churn Prediction End-to-End ML Project**! This project serves as a comprehensive, hands-on learning lab designed to revise and master the entire Machine Learning Lifecycle and beginner MLOps practices.

Every core concept—from raw exploratory data analysis (`EDA`) to a containerized `FastAPI` service with an automated `CI/CD` pipeline—is fully implemented, explained, and tested.

---

## 🏗️ Project Architecture & Data Flow

Below is the architectural flow of how our customer data travels from generation all the way to cloud-ready production container predictions:

```text
       [1. RAW DATA GENERATOR] (Injects NaNs, anomalies, 8.4% Churn rate)
                  │
                  ▼
       [2. DATA CLEANING & EDA] (Calculates MCAR/MAR; fits Median Imputer; caps outliers)
                  │
                  ▼
       [3. FEATURE ENGINEERING] (One-Hot Encodes categories; StandardScaler for numeric)
                  │
                  ▼
       [4. MODEL EXPLORATION] (Trains Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
                  │
                  ▼
       [5. EVALUATION METRICS] (Builds Confusion Matrix; computes Precision, Recall, F1, ROC-AUC)
                  │
                  ▼
       [6. GRID SEARCH CV & TUNING] (Runs K-Fold Cross-Validation & GridSearchCV on Random Forest)
                  │
                  ▼
       [7. PRODUCTION PIEPELINE] (Assembles unified sklearn.pipeline.Pipeline object)
                  │
                  ▼
       [8. MODEL VERSION REGISTRY] (Saves version pkl; updates metadata.json with candidate vs prod pointer)
                  │
                  ▼
       [9. FASTAPI SERVING] (Loads active production model; serving POST /predict; type-safety validation)
                  │
                  ▼
       [10. DOCKER PACKAGING] (Builds portable, clean container layer)
                  │
                  ▼
       [11. GITHUB ACTIONS CI/CD] (Triggers automated tests, training validations, and Docker builds on push)
```

---

## 🚀 Quick Start Guide

This project is built inside `D:\churn-prediction-project` inside an isolated Python virtual environment (`venv`). 

### 1. Set Up and Run the Interactive CLI Study Assistant
We have built a gorgeous terminal-based tutor `run_interactive.py` that walks you through every stage, prints mathematical explanations, and executes code live.

Open your PowerShell terminal and run:
```powershell
# Navigate to the project folder
cd D:\churn-prediction-project

# Activate the virtual environment
.\venv\Scripts\activate

# Launch the Interactive CLI Study Assistant!
python run_interactive.py
```

### 2. Start the FastAPI Production Server
To start the web server locally and query predictions:
```powershell
# Start Uvicorn hosting our FastAPI app
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **Predictive Web Dashboard UI**: Open your web browser and navigate to: `http://127.0.0.1:8000/` (Serves our gorgeous interactive analytics interface!)
* **Interactive API Swagger Docs**: Navigate to: `http://127.0.0.1:8000/docs` (Test the REST endpoints directly in your browser!)
* **JSON API Health Status Check**: Navigate to: `http://127.0.0.1:8000/health` (Exposes the currently loaded active production model metadata)

### 3. Run the Automated Test Suite
To execute our comprehensive unit tests verifying API validation, lifespans, and production imputation:
```powershell
pytest app/test_api.py -v
```

---

## 🎓 The 10 Core ML Concepts: Master Cheat Sheet

### Concept 1: Missing Values & Cleaning (EDA)
* **MCAR (Missing Completely at Random)**: Missingness is purely accidental (e.g. sensor battery failure).
* **MAR (Missing at Random)**: Missingness is related to another column (e.g., older customers not reporting salary).
* **MNAR (Missing Not at Random)**: Missingness is related to the value itself (e.g., extremely wealthy people refusing to answer wealth surveys).
* **Mean vs. Median Imputation**:
  * **Mean** is highly sensitive to outliers. 
  * **Median** (middle value) is **robust to outliers**. 
  * *Example*: `[30, 40, 50, 45, 999]` $\to$ Mean is **232.8**, Median is **45.0**. If we impute a normal customer's missing charges with the Mean, we assign a huge, skewed charge. **Always prefer Median Imputation for skewed or outlier-prone numerical data!**

### Concept 2: Feature Encoding
* **Label / Ordinal Encoding**: Maps categories to integers (0, 1, 2...). Use strictly for categories with an inherent rank or order (e.g., `Basic` = 0, `Standard` = 1, `Premium` = 2).
* **One-Hot Encoding (OHE)**: Creates separate binary columns ($0$ or $1$) for each unique category. Use for nominal (unordered) categories (e.g., `Gender`, `Payment_Method`).
* **Dummy Variable Trap**: Multicollinearity where columns are highly correlated. If `Gender_Male = 1`, then `Gender_Female` must be `0`. Resolve this by dropping the first column (`drop='first'`), reducing image/feature redundancy.

### Concept 3: Feature Scaling
* **StandardScaler (Z-Score Standardization)**:
  $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
  Rescales features to have **Mean ($\mu$) = 0** and **Standard Deviation ($\sigma$) = 1**.
* **MinMaxScaler (Normalization)**:
  $$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
  Compresses all data strictly between **0 and 1**.
* **Who needs scaling?**
  * **Requires**: Distance-based models (**KNN**, **SVM**) where magnitude dominates distance; Gradient models (**Logistic Regression**, **Neural Networks**) to converge faster.
  * **Does NOT Require**: Tree-based models (**Decision Trees**, **Random Forest**) because they split features one-by-one and don't compare magnitudes across features.

### Concept 4: Train-Test Split & Generalization
* **Generalization**: A model's ability to perform well on new, unseen customers.
* **Underfitting**: Model is too simple. Performs poorly on both train and test data.
* **Overfitting**: Model is too complex, memorizing training noise. Performs exceptionally on train, but fails on test.
* **Stratified Split**: Keeps target class distributions identical in both splits (e.g., both get exactly 8.4% churners). Crucial for imbalanced datasets.

### Concept 5: The 5 Classification Algorithms
1. **Logistic Regression**: Linear scoring passed through the **Sigmoid Function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   Outputs probability $0.0$ to $1.0$. Predicts class 1 if $\ge 0.5$.
2. **Decision Tree**: Asks splits that maximize **Information Gain** (reduction in **Gini Impurity** or **Entropy**).
   $$\text{Gini} = 1 - \sum P(x)^2 \quad\quad \text{Entropy} = -\sum P(x) \log_2 P(x)$$
3. **Random Forest**: **Bagging Ensemble**. Trains 100+ separate trees on random bootstrap rows and random features, taking the majority vote. Drastically reduces overfitting.
4. **SVM (Support Vector Machine)**: Draws a hyperplane boundary that maximizes the **Margin** to the closest points (**Support Vectors**). Projects non-linear data into higher dimensions using the **Kernel Trick**.
5. **KNN (K-Nearest Neighbors)**: **Lazy Learner**. Stores training data. For new points, computes Euclidean/Manhattan distance, finds the $K$ nearest neighbors, and outputs the majority class.

### Concept 6: Comprehensive Evaluation Metrics
For imbalanced data, predicting "nobody churns" gets **91.6% accuracy** but caught **0% of churners**. Never rely on accuracy alone!

* **Confusion Matrix**:

| | Predicted: NO | Predicted: YES |
|---|---|---|
| **Actual: NO** | **True Negative (TN)** | **False Positive (FP)** (Type I error) |
| **Actual: YES** | **False Negative (FN)** (Type II error) | **True Positive (TP)** |

* **Precision**: "When model says YES, how often correct?"
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity)**: "Out of all actual YES cases, how many did we catch?"
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1-Score**: Harmonic mean of Precision and Recall. The gold standard for imbalanced classification:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **ROC-AUC**: plots True Positive Rate vs. False Positive Rate across all thresholds. Represents how well the model ranks risk.

### Concept 7: K-Fold Cross-Validation
Divides training data into $K$ equal folds. Trains $K$ separate times, validating on a different fold each time. Eliminates "lucky splits" and measures metric stability (variance).

### Concept 8: Hyperparameter Tuning
* **Model Parameters**: Learned *during* training (e.g. decision splits, coefficient weights).
* **Hyperparameters**: Set *before* training to control behavior (e.g., `n_estimators`, `max_depth`).
* **GridSearchCV**: Systematically fits models across every single cell in a grid of hyperparameters using Cross-Validation to find the absolute best recipe.

### Concept 9: Scikit-Learn Pipelines
Combines preprocessing (Median Imputer, One-Hot Encoder, StandardScaler) and the classifier into a **single binary `.pkl` object**.
* **Prevents Data Leakage** by fitting scaling parameters strictly on training splits.
* **Production Simplicity**: Backend simply loads *one* pipeline file, inputs raw JSON, and receives predictions.

### Concept 10: Model Versioning & Registry
A database/metadata registry that tracks every trained model version, logging its exact hyperparameters, F1 performance, timestamp, and deployment status pointer (candidate vs. production). Enables seamless, instant rollbacks.

---

## 🐳 MLOps: Dockerization & CI/CD

### Docker Containerization
Our `Dockerfile` builds a portable Linux layer containing:
* **Base**: `python:3.10-slim`
* **Libraries**: Installs exact library locks from `requirements.txt` with `--no-cache-dir` to keep image sizes extremely small.
* **FastAPI**: Runs a production Uvicorn server exposing port `8000`.
* **Portability**: Run `docker build -t churn-api .` and `docker run -p 8000:8000 churn-api` to run this microservice identically on any OS or cloud provider.

### GitHub Actions CI/CD
Our automated pipeline (`.github/workflows/ml-pipeline.yml`) runs on every push:
1. Sets up virtual machines and Python caches.
2. Installs requirements.
3. Generates data, cleans it, and runs **automated Pytest unit tests**.
4. Tests model training and hyperparameter search compilation.
5. Performs a **mock Docker build** to guarantee package deployments will never fail.
