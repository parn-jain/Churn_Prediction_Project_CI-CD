import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

def demonstrate_encoding(df):
    """
    Demonstrates how One-Hot Encoding works and prints before-and-after states.
    """
    print("\n" + "="*50)
    print("CONCEPT 2: FEATURE ENCODING DEMONSTRATION")
    print("="*50)
    
    sample_df = df[["Gender", "Payment_Method"]].head(5).copy()
    print("Original Categorical Columns:")
    print(sample_df)
    
    # Initialize OneHotEncoder
    # sparse_output=False returns a numpy array instead of a sparse matrix
    # drop="first" drops the first category to avoid multicollinearity (Dummy Variable Trap!)
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    
    # Fit and transform
    encoded_features = encoder.fit_transform(sample_df)
    encoded_cols = encoder.get_feature_names_out(["Gender", "Payment_Method"])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_cols)
    
    print("\nOne-Hot Encoded Columns (with drop='first' applied):")
    print(encoded_df)
    print("\n--- Educational Insight ---")
    print("Notice that Gender has only ONE column: 'Gender_Male'. If it is 1, the person is Male. "
          "If it is 0, the person is Female. The Female column was dropped because it was 100% redundant.")
    print("Similarly, Payment_Method (3 categories) is represented by TWO columns. "
          "If both 'Payment_Method_Credit Card' and 'Payment_Method_Mailed Check' are 0, "
          "it automatically implies the payment method was 'Bank Transfer' (the dropped category).")
    print("This drops redundancy and saves our model from the Dummy Variable Trap!")
    print("="*50)

def demonstrate_scaling(df):
    """
    Demonstrates StandardScaler vs MinMaxScaler and shows why it matters.
    """
    print("\n" + "="*50)
    print("CONCEPT 3: FEATURE SCALING DEMONSTRATION")
    print("="*50)
    
    sample_df = df[["Age", "Monthly_Charges", "Usage"]].head(5).copy()
    print("Original Features (Note the massive difference in scale!):")
    print(sample_df)
    
    # 1. StandardScaler (Mean=0, Std=1)
    std_scaler = StandardScaler()
    std_scaled = std_scaler.fit_transform(sample_df)
    std_df = pd.DataFrame(std_scaled, columns=["Age_Std", "Charges_Std", "Usage_Std"])
    
    # 2. MinMaxScaler (Range [0, 1])
    minmax_scaler = MinMaxScaler()
    minmax_scaled = minmax_scaler.fit_transform(sample_df)
    minmax_df = pd.DataFrame(minmax_scaled, columns=["Age_MinMax", "Charges_MinMax", "Usage_MinMax"])
    
    print("\nStandardScaler Output (Centered around 0, Std=1):")
    print(std_df)
    print(f"Mean of Age_Std: {std_df['Age_Std'].mean():.1f} (Almost 0)")
    print(f"Std of Age_Std:  {std_df['Age_Std'].std():.1f} (Exactly 1)")
    
    print("\nMinMaxScaler Output (Strictly between 0 and 1):")
    print(minmax_df)
    print("\n--- Educational Insight ---")
    print("Look at the scales! Before scaling, a change in 'Usage' (e.g. from 10 to 500) "
          "was mathematically 100 times larger than a change in 'Age' (e.g. from 20 to 30).")
    print("After StandardScaler, both features now reside in a similar range (typically -3 to +3), "
          "allowing models like KNN and SVM to evaluate them on equal terms!")
    print("="*50)

def process_features(input_csv_path):
    """
    Utility to run the demonstrations on our dataset.
    """
    df = pd.read_csv(input_csv_path)
    demonstrate_encoding(df)
    demonstrate_scaling(df)

if __name__ == "__main__":
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    import os
    if not os.path.exists(clean_path):
        print(f"Error: {clean_path} not found. Please run eda.py first.")
    else:
        process_features(clean_path)
