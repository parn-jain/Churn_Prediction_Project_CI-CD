import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import the 5 essential ML Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def prepare_data_splits(df):
    """
    CONCEPT 4: TRAIN-TEST SPLIT
    Splits features and target, then divides them into Train (80%) and Test (20%) sets.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Stratify=y ensures train and test splits have the EXACT SAME ratio of Churners (imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n" + "="*50)
    print("CONCEPT 4: TRAIN-TEST SPLIT & CLASS SPLIT")
    print("="*50)
    print(f"Total dataset size:  {len(df)} rows")
    print(f"Training split size: {len(X_train)} rows (80%)")
    print(f"Testing split size:  {len(X_test)} rows (20%)")
    
    # Show Churn distribution in Train and Test to prove stratify worked
    print(f"\nTarget Class Distribution:")
    print(f"- Train set Churn rate: {y_train.mean()*100:.2f}%")
    print(f"- Test set Churn rate:  {y_test.mean()*100:.2f}%")
    print("="*50)
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X_train):
    """
    Creates a ColumnTransformer to handle both preprocessing steps:
    1. Scaling numerical features
    2. Encoding categorical features
    """
    numerical_cols = ["Age", "Subscription_Length", "Monthly_Charges", "Usage", "Complaints"]
    categorical_cols = ["Gender", "Payment_Method"]
    
    # Preprocessor using ColumnTransformer
    # We use drop='first' on OneHotEncoder to avoid multicollinearity (Dummy Variable Trap!)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    CONCEPT 5 & 6: TRAINS 5 DIFFERENT CLASSIFIERS & COMPARISON METRICS
    """
    # Create and fit the preprocessor on training data
    # (In a real pipeline, we bind this with the model. Here we fit it manually for illustration)
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Define our 5 classifiers
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("CONCEPT 5: TRAINING THE 5 CORE CLASSIFIERS")
    print("="*50)
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Train model
        model.fit(X_train_proc, y_train)
        
        # Predict classes (YES/NO)
        y_pred = model.predict(X_test_proc)
        # Predict probability scores (used for ROC-AUC)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        
        # CONCEPT 6: EVALUATION METRICS
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }
        
    print("\n" + "="*50)
    print("CONCEPT 6: EVALUATION METRICS SUMMARY")
    print("="*50)
    
    # Print results in a neat text table
    print(f"{'Classifier':<30} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1-Score':<8} | {'AUC':<5}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:.3f} | {metrics['Precision']:.3f} | "
              f"{metrics['Recall']:.3f} | {metrics['F1-Score']:.3f} | {metrics['ROC-AUC']:.3f}")
        
    print("\n--- Educational Insight on Class Imbalance ---")
    # Make a Dummy Model that predicts "Everyone Stays" (Churn = 0)
    dummy_preds = np.zeros(len(y_test))
    dummy_acc = accuracy_score(y_test, dummy_preds)
    dummy_f1 = f1_score(y_test, dummy_preds, zero_division=0)
    dummy_rec = recall_score(y_test, dummy_preds, zero_division=0)
    
    print("Let's look at a DUMMY model that simply predicts NO CUSTOMER CHURNS:")
    print(f"- Dummy Accuracy: {dummy_acc*100:.2f}% (Sounds amazing!)")
    print(f"- Dummy Recall:   {dummy_rec*100:.2f}% (Caught 0 actual churners!)")
    print(f"- Dummy F1-Score: {dummy_f1:.3f} (Tells the true story: completely useless!)")
    print("This is why F1-Score and Recall are crucial metrics in real-world ML applications.")
    print("="*50)
    
    return results

if __name__ == "__main__":
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    if not os.path.exists(clean_path):
        print(f"Error: {clean_path} not found. Run eda.py first.")
    else:
        df = pd.read_csv(clean_path)
        X_train, X_test, y_train, y_test = prepare_data_splits(df)
        train_and_evaluate_models(X_train, X_test, y_train, y_test)
