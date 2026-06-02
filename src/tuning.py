import os
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_features_transformer():
    """
    Builds the ColumnTransformer for preprocessing.
    """
    numerical_cols = ["Age", "Subscription_Length", "Monthly_Charges", "Usage", "Complaints"]
    categorical_cols = ["Gender", "Payment_Method"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor

def run_cross_validation(X, y):
    """
    CONCEPT 7: K-FOLD CROSS-VALIDATION
    Evaluates a default Random Forest classifier using 5-Fold Cross Validation.
    """
    print("\n" + "="*50)
    print("CONCEPT 7: 5-FOLD CROSS-VALIDATION")
    print("="*50)
    
    # 1. Transform features first (for simple illustration of CV)
    preprocessor = build_features_transformer()
    X_processed = preprocessor.fit_transform(X)
    
    # 2. Define stratified K-Fold to maintain churn class proportions across folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # 4. Run cross-validation scores using F1-score as target metric
    print("Running 5-Fold Cross Validation on Random Forest (calculating F1-score for each fold)...")
    scores = cross_val_score(rf, X_processed, y, cv=cv, scoring="f1")
    
    for idx, score in enumerate(scores):
        print(f"- Fold {idx+1}: F1-Score = {score:.4f}")
        
    print(f"\nAverage Cross-Validated F1-Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f} variance)")
    print("\n--- Educational Insight ---")
    print("Instead of relying on one train/test partition, K-Fold CV tests the model on 5 distinct subsets.")
    print("This gives us a standard deviation (variance). A high standard deviation means the model "
          "is highly sensitive to which data it trains on (unstable). A low standard deviation means stable generalization.")
    print("="*50)

def run_hyperparameter_tuning(X, y):
    """
    CONCEPT 8: HYPERPARAMETER TUNING (GridSearchCV)
    Exhaustively searches for the optimal hyperparameters for Random Forest.
    """
    print("\n" + "="*50)
    print("CONCEPT 8: HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*50)
    
    preprocessor = build_features_transformer()
    X_processed = preprocessor.fit_transform(X)
    
    # Define our hyperparameter search space (grid)
    # We want to tune:
    # - n_estimators (number of decision trees in forest)
    # - max_depth (maximum depth of each tree, controls overfitting)
    # - criterion (function to measure splitting impurity: 'gini' vs 'entropy')
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 8],
        "criterion": ["gini", "entropy"]
    }
    
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-Fold to speed up training
    
    # Initialize Grid Search
    # Refit=True ensures the best parameter combination is trained on the entire dataset at the end!
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,  # Use all available CPU cores!
        verbose=1
    )
    
    print("Starting GridSearchCV across 18 parameter combinations (3-Fold CV = 54 total models)...")
    grid_search.fit(X_processed, y)
    
    print("\nGrid Search Complete!")
    print(f"Best Hyperparameters:    {grid_search.best_params_}")
    print(f"Best CV F1-Score achieved: {grid_search.best_score_:.4f}")
    
    # Return the best, fully trained model
    return grid_search.best_estimator_, grid_search.best_params_

if __name__ == "__main__":
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    if not os.path.exists(clean_path):
        print(f"Error: {clean_path} not found. Run eda.py first.")
    else:
        df = pd.read_csv(clean_path)
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        
        run_cross_validation(X, y)
        run_hyperparameter_tuning(X, y)
