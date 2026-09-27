import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Made with Gemini

def train_and_save_model():
    print("Loading Parquet dataset for synthetic feature baseline generation...")
    # 1. Load Parquet dataset and normalize column names
    df = pd.read_parquet("data/geo_statistical_summary.parquet")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Clean/prepare key numeric columns with safe defaults
    numeric_defaults = {
        "hmda_denial_rate": 0.15,
        "median_dti": 38.0,
        "median_cltv": 80.0,
        "mean_interest_rate": 6.5,
        "mean_rate_spread": 0.3,
        "median_income": 75.0,
        "median_loan_amount": 250.0
    }
    
    for col, default_val in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)
        else:
            df[col] = default_val

    # 2. Synthesize applicant training samples conditioned on county statistics
    np.random.seed(42)
    n_samples = 10000  # Expanded sample size for robust 7-feature training
    
    # Draw county contexts
    sampled_counties = df.sample(n=n_samples, replace=True).reset_index(drop=True)
    
    # Feature distributions conditioned on regional averages
    incomes = np.random.normal(sampled_counties["median_income"], 25, n_samples).clip(20, 400)
    loan_amounts = np.random.normal(sampled_counties["median_loan_amount"], 75, n_samples).clip(30, 900)
    dtis = np.random.normal(sampled_counties["median_dti"], 8, n_samples).clip(15, 65)
    cltvs = np.random.normal(sampled_counties["median_cltv"], 10, n_samples).clip(40, 110)
    interest_rates = np.random.normal(sampled_counties["mean_interest_rate"], 0.75, n_samples).clip(3.0, 12.0)
    rate_spreads = np.random.exponential(sampled_counties["mean_rate_spread"] + 0.1, n_samples).clip(0.0, 4.0)
    minority_pcts = np.random.uniform(5, 85, n_samples)
    
    # Realistic Underwriting Denial Probability Logic
    # DTI > 43% and CLTV > 80% heavily drive denial probabilities
    dti_penalty = np.where(dtis > 43, (dtis - 43) * 0.12, 0)
    cltv_penalty = np.where(cltvs > 80, (cltvs - 80) * 0.05, 0)
    lti = loan_amounts / incomes
    county_denial = sampled_counties["hmda_denial_rate"].values
    
    # Logit formula incorporating financial risk controls & demographic variance
    denial_logits = (
        -3.5 
        + (0.35 * lti)
        + dti_penalty
        + cltv_penalty
        + (0.8 * rate_spreads)
        + (0.012 * minority_pcts)
        + (2.5 * county_denial)
    )
    
    denial_probs = 1 / (1 + np.exp(-denial_logits))
    target_denied = (np.random.rand(n_samples) < denial_probs).astype(int)
    
    # 3. Construct 7-Feature Matrix matching app.py schema exactly
    X = pd.DataFrame({
        'loan_amount': loan_amounts,
        'income': incomes,
        'dti': dtis,
        'cltv': cltvs,
        'interest_rate': interest_rates,
        'rate_spread': rate_spreads,
        'tract_minority_population_percent': minority_pcts
    })
    y = target_denied
    
    print(f"Training Random Forest on {n_samples:,} records with {X.shape[1]} features...")
    print(f"Synthesized Target Denial Rate: {y.mean() * 100:.1f}%")
    
    # 4. Fit Random Forest Model
    rf_model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=8, 
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    
    # Print Feature Importances for verification
    fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nModel Feature Importances:")
    print(fi.map("{:.3f}".format))
    
    # 5. Save model artifact
    joblib.dump(rf_model, "src/mortgage_model.pkl")
    print("\nSuccessfully trained and updated src/mortgage_model.pkl!")

if __name__ == "__main__":
    train_and_save_model()