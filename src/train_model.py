import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # 1. Load your new Parquet dataset
    df = pd.read_parquet("data/geo_statistical_summary.parquet")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Clean/prepare columns
    df["hmda_denial_rate"] = pd.to_numeric(df["hmda_denial_rate"], errors="coerce").fillna(0.15)
    
    # 2. Synthesize or map realistic applicant training features from county stats
    # (If you don't have individual application records, we simulate training samples around county means)
    np.random.seed(42)
    n_samples = 5000
    
    # Draw county contexts
    sampled_counties = df.sample(n=n_samples, replace=True)
    
    loan_amounts = np.random.normal(250, 80, n_samples).clip(50, 800)
    incomes = np.random.normal(80, 30, n_samples).clip(20, 300)
    minority_pcts = np.random.uniform(5, 85, n_samples)
    
    # Probability of denial increases with Loan-to-Income and local County Denial Rate
    lti = loan_amounts / incomes
    county_denial = sampled_counties["hmda_denial_rate"].values
    
    denial_logits = -2.0 + (0.4 * lti) + (0.015 * minority_pcts) + (3.0 * county_denial)
    denial_probs = 1 / (1 + np.exp(-denial_logits))
    target_denied = (np.random.rand(n_samples) < denial_probs).astype(int)
    
    X = pd.DataFrame({
        'loan_amount': loan_amounts,
        'income': incomes,
        'tract_minority_population_percent': minority_pcts
    })
    y = target_denied
    
    # 3. Fit Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X, y)
    
    # 4. Save model artifact
    joblib.dump(rf_model, "src/mortgage_model.pkl")
    print("Successfully trained and updated src/mortgage_model.pkl!")

if __name__ == "__main__":
    train_and_save_model()