import os
import pandas as pd
from src.data_processor import process_hmda_lar
from src.stats_engine import calculate_spatial_entropy, calculate_jsd
from src.ml_pipeline import train_mortgage_model

def run_pipeline_test():
    print("--- 1. TESTING DATA PROCESSOR ---")
    df_processed = process_hmda_lar("data/mock_2025_combined_mlar.csv")
    assert not df_processed.empty, "Data processor returned empty DataFrame"
    assert "denial_rate" in df_processed.columns, "Missing denial_rate column"
    print("Data processor test passed!")

    print("\n--- 2. TESTING STATS ENGINE ---")
    # test spatial entropy
    test_probs = [0.1, 0.4, 0.5]
    se_score = calculate_spatial_entropy(test_probs)
    assert se_score > 0, "Spatial entropy calculation failed"
    
    # test JSD
    local_dist = [10, 20, 30, 40]
    national_dist = [25, 25, 25, 25]
    jsd_score = calculate_jsd(local_dist, national_dist)
    assert 0 <= jsd_score <= 1, "JSD score out of bounds [0, 1]"
    print(f"Stats engine test passed! SE: {se_score:.2f}, JSD: {jsd_score:.2f}")

    print("\n--- 3. TESTING ML PIPELINE ---")
    train_mortgage_model("data/mock_2025_combined_mlar.csv")
    assert os.path.exists("src/mortgage_model.pkl"), "ML model pkl file was not created"
    print("ML pipeline test passed!")

    print("\nALL BACKEND PIPELINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline_test()