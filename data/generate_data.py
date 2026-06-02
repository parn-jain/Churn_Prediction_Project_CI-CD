import os
import numpy as np
import pandas as pd

def generate_churn_dataset(num_samples=1500, random_seed=42):
    """
    Generates a realistic, imbalanced customer churn dataset with:
    - Missing values (to practice Imputation)
    - Categorical variables (to practice Feature Encoding)
    - Outliers (to practice robust scaling/cleaning)
    - Strong logical correlations mixed with random noise.
    """
    np.random.seed(random_seed)
    
    # 1. Generate core features
    age = np.random.randint(18, 76, size=num_samples)
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.48, 0.52])
    subscription_length = np.random.randint(1, 25, size=num_samples)  # in months
    monthly_charges = np.random.uniform(20.0, 150.0, size=num_samples)
    usage = np.random.randint(5, 500, size=num_samples)  # usage in units (e.g. GBs)
    
    # Complaints: heavily skewed, most people have 0, some have 1, few have 2+
    complaints = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.70, 0.18, 0.08, 0.03, 0.01])
    
    payment_method = np.random.choice(
        ["Credit Card", "Bank Transfer", "Mailed Check"], 
        size=num_samples, 
        p=[0.50, 0.35, 0.15]
    )
    
    # 2. Define Churn Logic (Creating the Ground Truth with correlation + noise)
    # Churn probability calculation:
    # - Complaints increase churn probability significantly
    # - High monthly charges increase churn
    # - Long subscription length decreases churn (customer loyalty)
    # - High usage decreases churn (highly engaged customer)
    # - Older age slightly increases churn
    
    # Calculate a log-odds score (similar to a Logistic Regression formula!)
    # log_odds = intercept + beta1*x1 + beta2*x2 ...
    log_odds = (
        -2.5 
        + 0.8 * complaints 
        + 0.015 * (monthly_charges - 70) 
        - 0.05 * subscription_length 
        - 0.003 * (usage - 200) 
        + 0.01 * (age - 45)
    )
    
    # Convert log-odds to probability using the Sigmoid Function: p = 1 / (1 + e^-x)
    # This is the exact math behind Logistic Regression! We will study it in Step 5.
    churn_prob = 1 / (1 + np.exp(-log_odds))
    
    # Sample binary Churn target (1 = YES, 0 = NO) based on the probabilities
    churn = np.random.binomial(1, churn_prob)
    
    # 3. Create initial DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Subscription_Length": subscription_length,
        "Monthly_Charges": monthly_charges,
        "Usage": usage,
        "Complaints": complaints,
        "Payment_Method": payment_method,
        "Churn": churn
    })
    
    # 4. Inject Messiness!
    # Let's inject 5% missing values (NaNs) in Age and Monthly_Charges to simulate real-world gaps
    age_missing_idx = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    monthly_missing_idx = np.random.choice(num_samples, size=int(num_samples * 0.04), replace=False)
    
    df.loc[age_missing_idx, "Age"] = np.nan
    df.loc[monthly_missing_idx, "Monthly_Charges"] = np.nan
    
    # Let's inject a few extreme outliers in Monthly_Charges to demonstrate robust scaling
    # E.g., data entry errors where monthly bills are $999.00
    outlier_idx = np.random.choice(num_samples, size=5, replace=False)
    df.loc[outlier_idx, "Monthly_Charges"] = 999.00
    
    return df

if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs("D:/churn-prediction-project/data", exist_ok=True)
    
    print("Generating raw, messy churn dataset...")
    df = generate_churn_dataset(num_samples=1500)
    
    # Save to CSV
    output_path = "D:/churn-prediction-project/data/raw_churn.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated and saved successfully to: {output_path}")
    print("\nDataset Summary Statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Churn Rate (Class Imbalance): {df['Churn'].mean() * 100:.2f}% Churn vs {(1 - df['Churn'].mean()) * 100:.2f}% Retained")
    print(f"- Missing values in Age: {df['Age'].isnull().sum()}")
    print(f"- Missing values in Monthly Charges: {df['Monthly_Charges'].isnull().sum()}")
    print(f"- Outliers (Monthly Charges > $500): {sum(df['Monthly_Charges'] > 500)}")
