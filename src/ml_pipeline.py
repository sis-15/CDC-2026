import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Made with Gemini

def train_mortgage_model(hmda_csv_path):
    df = pd.read_csv(hmda_csv_path, low_memory=False)
    
    # filter for approval (1) vs denial (3)
    df = df[df['action_taken'].isin([1, 3])].copy()
    df['target'] = (df['action_taken'] == 3).astype(int)  # 1 = Denied, 0 = Approved
    
    # feature selection
    features = ['loan_amount', 'income', 'tract_minority_population_percent']
    df = df[features + ['target']].dropna()
    
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # save model weights
    joblib.dump(model, "src/mortgage_model.pkl")
    print("Model trained and saved to src/mortgage_model.pkl")

if __name__ == "__main__":
    train_mortgage_model("data/2025_combined_mlar.csv")