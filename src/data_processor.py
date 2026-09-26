import pandas as pd
import numpy as np

def process_hmda_lar(filepath):
    # load hmda csv
    df = pd.read_csv(filepath, low_memory=False)
    
    # action_taken == 3 means loan denied
    df['is_denied'] = (df['action_taken'] == 3).astype(int)
    
    # group by county fips
    summary = df.groupby('county_code').agg(
        total_applications=('action_taken', 'count'),
        total_denials=('is_denied', 'sum'),
        denial_rate=('is_denied', 'mean'),
        median_income=('income', 'median'),
        median_loan_amount=('loan_amount', 'median')
    ).reset_index()
    
    # compute loan-to-income ratio
    summary['loan_to_income_ratio'] = summary['median_loan_amount'] / (summary['median_income'] + 1e-5)
    
    return summary

if __name__ == "__main__":
    df_hmda = process_hmda_lar("data/2025_combined_mlar.csv")
    df_hmda.to_csv("data/processed_hmda_county.csv", index=False)