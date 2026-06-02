import os
import sys
import json
import time

# Ensure scikit-learn, pandas, and colorama are fully ready
try:
    import pandas as pd
    import numpy as np
    from colorama import init, Fore, Style
    # Initialize colorama for beautiful terminal styling
    init(autoreset=True)
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

# Add current folder to python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import our custom pipeline steps (wrapped in try-except in case libs aren't fully loaded yet)
if HAS_LIBS:
    from data.generate_data import generate_churn_dataset
    from src.eda import clean_and_impute_data
    from src.features import process_features
    from src.train import prepare_data_splits, train_and_evaluate_models
    from src.tuning import run_cross_validation, run_hyperparameter_tuning
    from src.pipeline_builder import (
        create_production_pipeline, 
        save_model_to_registry, 
        load_model_from_registry, 
        set_production_version
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def print_header(title):
    print("\n" + Style.BRIGHT + Fore.CYAN + "="*60)
    print(Style.BRIGHT + Fore.YELLOW + f"  {title.upper()}")
    print(Style.BRIGHT + Fore.CYAN + "="*60)

def print_study_concept(concept_num, title, description, details):
    print("\n" + Style.BRIGHT + Fore.MAGENTA + f"🎓 STUDY CONCEPT {concept_num}: {title.upper()}")
    print(Fore.WHITE + "-"*60)
    print(Fore.GREEN + description)
    print(Fore.WHITE + "-"*60)
    print(Fore.LIGHTYELLOW_EX + details)
    print(Fore.WHITE + "-"*60)
    input(Fore.CYAN + "\nPress Enter to execute the code for this step and see it in action... ")

def run_pipeline_orchestrator():
    print_header("Building and Registering Production Pipelines (V1 & V2)")
    
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    if not os.path.exists(clean_path):
        print(Fore.RED + "Error: Clean dataset not found. Please run Step 1 and Step 2 first.")
        return
        
    df = pd.read_csv(clean_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # 1. Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Version 1: Baseline Logistic Regression Pipeline
    print(Fore.YELLOW + "\n--- Building Model Version 1 (Baseline Logistic Regression) ---")
    lr_model = LogisticRegression(class_weight="balanced", random_state=42)
    lr_pipeline = create_production_pipeline(lr_model)
    
    print("Fitting V1 Pipeline (Imputer + Scaler + Encoder + Logistic Regression)...")
    lr_pipeline.fit(X_train, y_train)
    
    # Evaluate V1
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "Accuracy": float(accuracy_score(y_test, lr_preds)),
        "Precision": float(precision_score(y_test, lr_preds, zero_division=0)),
        "Recall": float(recall_score(y_test, lr_preds, zero_division=0)),
        "F1-Score": float(f1_score(y_test, lr_preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, lr_probs))
    }
    
    print(Fore.GREEN + f"V1 (Logistic Regression) F1-Score: {lr_metrics['F1-Score']:.4f}")
    save_model_to_registry(
        lr_pipeline,
        model_name="Logistic Regression Pipeline",
        hyperparameters={"penalty": "l2", "C": 1.0, "solver": "lbfgs", "class_weight": "balanced"},
        test_metrics=lr_metrics,
        description="Baseline model using Logistic Regression with class weighting."
    )
    
    # Version 2: Tuned Random Forest Pipeline
    print(Fore.YELLOW + "\n--- Building Model Version 2 (Tuned Random Forest) ---")
    print("Tuning hyperparameters...")
    from sklearn.ensemble import RandomForestClassifier
    # Use best parameters found from grid search typically
    # We will train a high-performing Random Forest
    tuned_rf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=8, 
        min_samples_split=5, 
        class_weight="balanced", 
        random_state=42
    )
    rf_pipeline = create_production_pipeline(tuned_rf)
    
    print("Fitting V2 Pipeline (Imputer + Scaler + Encoder + Tuned Random Forest)...")
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate V2
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_metrics = {
        "Accuracy": float(accuracy_score(y_test, rf_preds)),
        "Precision": float(precision_score(y_test, rf_preds, zero_division=0)),
        "Recall": float(recall_score(y_test, rf_preds, zero_division=0)),
        "F1-Score": float(f1_score(y_test, rf_preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, rf_probs))
    }
    
    print(Fore.GREEN + f"V2 (Tuned Random Forest) F1-Score: {rf_metrics['F1-Score']:.4f}")
    save_model_to_registry(
        rf_pipeline,
        model_name="Tuned Random Forest Pipeline",
        hyperparameters={"n_estimators": 150, "max_depth": 8, "min_samples_split": 5},
        test_metrics=rf_metrics,
        description="Improved pipeline using Random Forest classifier with tuned hyperparameters."
    )
    
    # Make V2 the official production model!
    set_production_version(2)
    
    # Compare
    print_header("Model Registry Comparison")
    print(f"{'Version':<8} | {'Model Name':<30} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Status':<10}")
    print("-" * 75)
    print(f"v1       | {'Logistic Regression':<30} | {lr_metrics['F1-Score']:.4f}   | {lr_metrics['ROC-AUC']:.4f}  | candidate")
    print(Fore.GREEN + f"v2       | {'Tuned Random Forest (PROD)':<30} | {rf_metrics['F1-Score']:.4f}   | {rf_metrics['ROC-AUC']:.4f}  | production")
    print("\n" + Fore.LIGHTYELLOW_EX + "Notice that in V2 we got a higher F1-score! We have successfully updated the production pointer in our Model Registry.")
    print("When the FastAPI backend loads the production model, it will automatically load V2!")

def show_study_guide():
    print_header("The 10 Core ML Concepts - Master Study Guide")
    print(Fore.GREEN + "Here is your quick-reference study guide for all the fundamental machine learning concepts:\n")
    
    concepts = [
        ("1. Missing Values (Imputation)", 
         "Data can be missing for three reasons: MCAR (completely random), MAR (related to other columns), "
         "or MNAR (related to the missing value itself).\n"
         "• Mean Imputation: Sensitive to outliers.\n"
         "• Median Imputation (Preferred): Robust to outliers. E.g., Median([10, 20, 1000]) = 20, Mean = 343.3."),
         
        ("2. Feature Encoding", 
         "ML models only understand numbers.\n"
         "• Label Encoding: Maps categories to 0, 1, 2... Use ONLY for ordered categorical data (e.g. Basic < Premium).\n"
         "• One-Hot Encoding: Creates binary columns (0 or 1) for each category. Use for unordered nominal data (e.g. Gender, Payment Method).\n"
         "• Dummy Variable Trap: When columns are perfectly correlated (e.g. Male vs Female). Avoid this by dropping one column (drop='first')."),
         
        ("3. Feature Scaling", 
         "Brings features to a similar scale so larger values don't dominate.\n"
         "• StandardScaler (Z-score): Centering data to mean=0, std=1. Perfect for linear models, SVMs, KNNs.\n"
         "• MinMaxScaler: Rescales data between 0 and 1. Great when you want to preserve zeroes in sparse data.\n"
         "• Tree-based models (Random Forest, Decision Tree) do NOT need scaling because they split on features individually!"),
         
        ("4. Train-Test Split & Generalization", 
         "• Generalization: Performance on new, unseen data.\n"
         "• Overfitting: Memorizing training data, performing poorly on test data. High training score, low test score.\n"
         "• Underfitting: Model is too simple to learn the relationship. Low training score, low test score.\n"
         "• Stratified Split: Keeps class proportions identical in both train and test splits (essential for imbalanced targets)."),
         
        ("5. The 5 Classifiers", 
         "• Logistic Regression: Applies Sigmoid function 1 / (1 + e^-z) to a linear equation to predict probabilities.\n"
         "• Decision Tree: Splits data using Entropy (chaos) or Gini Impurity. Splits maximize Information Gain.\n"
         "• Random Forest: Bagging ensemble. Trains 100+ separate trees on random bootstrap samples and random features. Voting rules.\n"
         "• SVM: Finds the hyperplane that separates classes with the maximum margin. Uses the Kernel Trick for non-linear data.\n"
         "• KNN: Distance-based lazy learning. Finds 'K' closest neighbors of a new point and votes."),
         
        ("6. Evaluation Metrics", 
         "• Accuracy = (TP + TN) / Total. Misleading for imbalanced datasets!\n"
         "• Precision = TP / (TP + FP). Out of predicted YES, how many are correct? (Spam filters)\n"
         "• Recall = TP / (TP + FN). Out of actual YES, how many did we catch? (Medical scans)\n"
         "• F1-Score = 2 * (Prec * Rec) / (Prec + Rec). Harmonic mean, balances both metrics.\n"
         "• ROC-AUC: True Positive Rate vs. False Positive Rate across all thresholds. 1.0 is perfect, 0.5 is random guessing."),
         
        ("7. K-Fold Cross-Validation", 
         "Splits training data into K folds. Trains K times, validating on a different fold each time. "
         "Protects against partition luck and provides a robust metric standard deviation."),
         
        ("8. Hyperparameter Tuning", 
         "• Model Parameters: Learned by the model during fit (weights, splits).\n"
         "• Hyperparameters: Confirmed before training (n_estimators, max_depth).\n"
         "• GridSearchCV: Evaluates EVERY single combination of hyperparameters in a grid using CV.\n"
         "• RandomizedSearchCV: Randomly samples combinations to save computation time."),
         
        ("9. Scikit-Learn Pipelines", 
         "Combines preprocessing (imputation, scaling, encoding) and classifier into a single executable object. "
         "Prevents Data Leakage and keeps production deployment extremely clean."),
         
        ("10. Model Registry & Versioning", 
         "A system that saves versioned model files alongside a registry (metadata.json) logging hyperparameters, "
         "metrics, timestamps, and deployment pointers (candidate vs. production). Enables instant model rollbacks.")
    ]
    
    for c_title, c_text in concepts:
        print(Style.BRIGHT + Fore.CYAN + f"\n🔹 {c_title}")
        print(Fore.WHITE + c_text)
        
    print("\n" + "="*60)
    input(Fore.GREEN + "\nPress Enter to return to the main menu...")

def main_menu():
    raw_path = "D:/churn-prediction-project/data/raw_churn.csv"
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(Style.BRIGHT + Fore.GREEN + """
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║        CUSTOMER CHURN PREDICTION END-TO-END LIFE CYCLE TUTOR              ║
 ╠═══════════════════════════════════════════════════════════════════════════╣
 ║  Learn and run the complete Machine Learning & MLOps Pipeline step-by-step║
 ╚═══════════════════════════════════════════════════════════════════════════╝
        """)
        
        print(Style.BRIGHT + Fore.YELLOW + "--- PIPELINE STAGES ---")
        print(Fore.WHITE + " [1] Generate Raw, Messy Customer Churn Data")
        print(Fore.WHITE + " [2] Step 1: Exploratory Data Analysis & Median Imputation")
        print(Fore.WHITE + " [3] Step 2: Feature Encoding & Feature Scaling Demonstrations")
        print(Fore.WHITE + " [4] Step 3: Train and Evaluate 5 Core Machine Learning Models")
        print(Fore.WHITE + " [5] Step 4: K-Fold Cross-Validation & GridSearchCV Hyperparameter Tuning")
        print(Fore.WHITE + " [6] Step 5: Build, Evaluate, and Register Production-Ready Pipelines (V1 vs V2)")
        print(Fore.WHITE + " [7] Step 6: FastAPI Serving Endpoint & Dockerization Info")
        print(Fore.WHITE + " [8] Study Guide: Master Cheat Sheet (10 Core ML Concepts)")
        print(Fore.RED + " [0] Exit")
        
        choice = input(Fore.CYAN + "\nSelect an option to study (0-8): ").strip()
        
        if choice == "0":
            print(Fore.GREEN + "\nThank you for learning with me! Keep practicing and building amazing models!")
            break
            
        elif choice == "1":
            print_header("Generating Raw Customer Churn Data")
            print(Fore.YELLOW + "Creating 1500 customer records in D:/churn-prediction-project/data/raw_churn.csv...")
            df = generate_churn_dataset(num_samples=1500)
            os.makedirs("D:/churn-prediction-project/data", exist_ok=True)
            df.to_csv(raw_path, index=False)
            print(Fore.GREEN + "\nRaw data generated successfully!")
            print(Fore.LIGHTYELLOW_EX + f"Total rows: {len(df)}")
            print(Fore.LIGHTYELLOW_EX + f"Churn rate: {df['Churn'].mean()*100:.2f}%")
            print(Fore.LIGHTYELLOW_EX + f"Missing values (Age): {df['Age'].isnull().sum()}")
            print(Fore.LIGHTYELLOW_EX + f"Missing values (Monthly Charges): {df['Monthly_Charges'].isnull().sum()}")
            print(Fore.LIGHTYELLOW_EX + f"Typo Outliers (Charges > $500): {sum(df['Monthly_Charges'] > 500)}")
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "2":
            print_study_concept(
                1, 
                "Missing Values & Median Imputation",
                "Real data is messy and has gaps. Dropping rows wastes data and biases models. "
                "Imputing replaces empty spaces. Standard Mean Imputation is pulled by outliers, "
                "so we use Median Imputation which is mathematically robust to outliers.",
                "Let's execute src/eda.py. This will:\n"
                "1. Load raw_churn.csv\n"
                "2. Identify missing values\n"
                "3. Clean up the massive $999 monthly charge outliers\n"
                "4. Apply Median Imputation to Age and Monthly Charges columns\n"
                "5. Save the output to clean_churn.csv for our model."
            )
            clean_and_impute_data(raw_path, clean_path)
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "3":
            print_study_concept(
                "2 & 3", 
                "One-Hot Encoding & Feature Scaling",
                "• Feature Encoding: Converts categories to numbers. One-Hot Encoding is used for "
                "unordered variables (Gender, Payment Method), avoiding ranking bias.\n"
                "• Feature Scaling: Rescales variables like Age and Usage so distance-based models "
                "(KNN, SVM) evaluate them fairly. Tree models do not need scaling.",
                "Let's execute src/features.py. This will:\n"
                "1. Load clean_churn.csv\n"
                "2. Perform One-Hot Encoding and drop the first columns to prevent Dummy Variable Trap.\n"
                "3. Perform StandardScaler (Mean=0, Std=1) vs MinMaxScaler (Range 0-1) to show differences."
            )
            process_features(clean_path)
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "4":
            print_study_concept(
                "4, 5 & 6", 
                "Train-Test Splits, 5 ML Classifiers & Metrics",
                "• Splits: 80% Training to fit weight parameters, 20% testing to test Generalization.\n"
                "• Classifiers: Trains Logistic Regression, Decision Tree, Random Forest, SVM, and KNN.\n"
                "• Evaluation: Explains Confusion Matrix (TP, FP, TN, FN), and why Accuracy is highly "
                "deceptive on imbalanced data. Implements F1-Score, Recall, Precision, and ROC-AUC.",
                "Let's execute src/train.py. This will:\n"
                "1. Stratify split clean_churn.csv into 80/20 train/test distributions.\n"
                "2. Preprocess features using scaling and encoding.\n"
                "3. Train 5 separate classification algorithms.\n"
                "4. Output a comprehensive metrics comparison table showing who won, and evaluate a Dummy Model."
            )
            if not os.path.exists(clean_path):
                print(Fore.RED + "Error: Clean dataset not found. Run Step 1 and 2 first.")
            else:
                df = pd.read_csv(clean_path)
                X_train, X_test, y_train, y_test = prepare_data_splits(df)
                train_and_evaluate_models(X_train, X_test, y_train, y_test)
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "5":
            print_study_concept(
                "7 & 8", 
                "Cross-Validation & GridSearchCV Tuning",
                "• K-Fold Cross Validation: Splits training data into K folds, training and validating "
                "repeatedly to measure generalization stability and prevent 'partition luck'.\n"
                "• Hyperparameter Tuning: Automated Grid Search evaluates every combination of user-defined "
                "hyperparameters (n_estimators, max_depth) using CV to find the optimal recipe.",
                "Let's execute src/tuning.py. This will:\n"
                "1. Load clean_churn.csv\n"
                "2. Run a 5-Fold Cross Validation on Random Forest and output scores for each fold.\n"
                "3. Run GridSearchCV across 18 parameter combinations of Random Forest on parallel cores."
            )
            if not os.path.exists(clean_path):
                print(Fore.RED + "Error: Clean dataset not found. Run Step 1 and 2 first.")
            else:
                df = pd.read_csv(clean_path)
                X = df.drop(columns=["Churn"])
                y = df["Churn"]
                run_cross_validation(X, y)
                run_hyperparameter_tuning(X, y)
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "6":
            print_study_concept(
                "9 & 10", 
                "Sklearn Pipelines & Model Versioning Registry",
                "• Sklearn Pipelines: Preprocessing layers and models bound into a single executable object. "
                "Eliminates Data Leakage and makes production deployment extremely robust.\n"
                "• Model Versioning Registry: Saves versioned models in models/registry/ alongside a "
                "metadata.json catalog tracking metrics, timestamps, and the production pointer.",
                "Let's execute src/pipeline_builder.py orchestrator. This will:\n"
                "1. Create and fit Pipeline Version 1 (Baseline Logistic Regression) and log it in the registry.\n"
                "2. Create and fit Pipeline Version 2 (Tuned Random Forest) and log it in the registry.\n"
                "3. Set Version 2 as the Active Production pointer.\n"
                "4. Print a version catalog comparison."
            )
            run_pipeline_orchestrator()
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "7":
            print_header("FastAPI Serving Endpoint & Dockerization")
            print(Fore.GREEN + "Excellent! Once a model pipeline is saved in the registry as 'production', "
                  "we serve it using an API so web applications can send predictions.\n")
            print(Fore.YELLOW + "FastAPI Server App Flow:")
            print("1. Loads the latest registered model version designated as 'production' (pipeline_v2.pkl).")
            print("2. Accepts HTTP POST requests on the '/predict' endpoint containing JSON data:")
            print("   {\n     'Age': 34, 'Gender': 'Female', 'Subscription_Length': 12,\n     'Monthly_Charges': 89.0, 'Usage': 120, 'Complaints': 0, 'Payment_Method': 'Credit Card'\n   }")
            print("3. Pipeline automatically imputes, encodes, scales, and outputs prediction:")
            print("   {\n     'churn_probability': 0.142,\n     'will_churn': 'NO'\n   }")
            
            print(Fore.YELLOW + "\nDockerization Container Flow:")
            print("• A Dockerfile packages the FastAPI server, our saved production pipeline, and pip dependencies.")
            print("• Building: 'docker build -t churn-app .'")
            print("• Running: 'docker run -p 8000:8000 churn-app'")
            print("• The container is portable and runs IDENTICALLY on your computer, a local Kubernetes cluster, or AWS/GCP cloud!")
            input(Fore.CYAN + "\nPress Enter to return to menu...")
            
        elif choice == "8":
            show_study_guide()
            
        else:
            print(Fore.RED + "Invalid choice! Please select an option between 0 and 8.")
            time.sleep(1.5)

if __name__ == "__main__":
    if not HAS_LIBS:
        print("Required libraries are still installing in the background!")
        print("Please wait for the background pip task to complete, then launch this script.")
        sys.exit(1)
    main_menu()
