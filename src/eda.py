import os
import pandas as pd
import numpy as np

def analyze_missing_data(df):
    """
    Analyzes and prints missing values in the DataFrame.
    """
    print("\n" + "="*50)
    print("CONCEPT 1: ANALYZING MISSING VALUES")
    print("="*50)
    
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Percentage (%)": missing_percent
    }).sort_values(by="Missing Count", ascending=False)
    
    print("Missing value statistics for each feature:")
    print(missing_df)
    print("\n--- Educational Insight ---")
    print("Why do we impute instead of dropping missing rows?")
    print(f"Total rows in raw dataset: {len(df)}")
    rows_with_nan = df.isnull().any(axis=1).sum()
    print(f"Number of rows with at least one missing value: {rows_with_nan} ({rows_with_nan/len(df)*100:.2f}%)")
    print("If we used listwise deletion (dropping rows), we would throw away "
          f"nearly {rows_with_nan/len(df)*100:.1f}% of our entire dataset!")
    print("Imputing allows us to preserve the non-missing data in these rows.")
    print("="*50)
    return missing_df

def demonstrate_mean_vs_median(df):
    """
    Shows a hands-on comparison of Mean vs Median for numerical imputation.
    """
    print("\n" + "="*50)
    print("STUDYING IMPUTATION: MEAN vs MEDIAN")
    print("="*50)
    
    charges = df["Monthly_Charges"].dropna()
    mean_val = charges.mean()
    median_val = charges.median()
    
    print(f"Monthly_Charges stats (including outliers!):")
    print(f"- Mean charge:   ${mean_val:.2f}")
    print(f"- Median charge: ${median_val:.2f}")
    print("\nWhy are they so different?")
    print(f"Let's check the maximum value in Monthly_Charges: ${charges.max():.2f}")
    print("Because we have outliers (like $999.00 entry errors), the Mean is dragged up significantly.")
    print("If we fill missing values with the Mean ($232.80+), we inject artificial, high charges.")
    print("If we fill with the Median (~$85.00), we represent a typical customer much better!")
    print("Rule of Thumb: If data has outliers or is skewed, prefer MEDIAN over MEAN.")
    print("="*50)

def clean_and_impute_data(raw_csv_path, output_csv_path):
    """
    Performs data cleaning:
    1. Loads raw dataset.
    2. Identifies and treats extreme outliers in Monthly_Charges.
    3. Imputes missing Age values using Median.
    4. Imputes missing Monthly_Charges values using Median.
    5. Saves clean data.
    """
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    # 1. Analyze missing data before cleaning
    analyze_missing_data(df)
    
    # 2. Study Mean vs Median
    demonstrate_mean_vs_median(df)
    
    # Copy dataframe for cleaning
    cleaned_df = df.copy()
    
    # 3. Clean Outliers
    # Identify outliers: in our data generator, we set data entry errors to $999.00
    # Let's replace monthly charges > $500 with the median of charges below $500
    normal_charges_median = df.loc[df["Monthly_Charges"] <= 500, "Monthly_Charges"].median()
    outlier_mask = cleaned_df["Monthly_Charges"] > 500
    outliers_count = outlier_mask.sum()
    
    if outliers_count > 0:
        print(f"\n[Cleaning Outliers] Found {outliers_count} anomalous monthly charges (> $500).")
        print(f"Replacing outliers with a typical customer charges median: ${normal_charges_median:.2f}")
        cleaned_df.loc[outlier_mask, "Monthly_Charges"] = normal_charges_median
        
    # 4. Impute Missing Values
    # Impute Age with median
    age_median = cleaned_df["Age"].median()
    print(f"[Imputing Age] Filling missing values with Median Age: {age_median:.1f} years old")
    cleaned_df["Age"] = cleaned_df["Age"].fillna(age_median)
    
    # Impute Monthly Charges with median
    charges_median = cleaned_df["Monthly_Charges"].median()
    print(f"[Imputing Charges] Filling missing values with Median Charges: ${charges_median:.2f}")
    cleaned_df["Monthly_Charges"] = cleaned_df["Monthly_Charges"].fillna(charges_median)
    
    # Verify no missing values remain
    remaining_nans = cleaned_df.isnull().sum().sum()
    print(f"\nVerification: Total remaining missing values in clean dataset: {remaining_nans}")
    
    # Save clean dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_csv_path}")
    
    return cleaned_df

if __name__ == "__main__":
    raw_path = "D:/churn-prediction-project/data/raw_churn.csv"
    clean_path = "D:/churn-prediction-project/data/clean_churn.csv"
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please run generate_data.py first.")
    else:
        clean_and_impute_data(raw_path, clean_path)
