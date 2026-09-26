import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# 1. create mock 2025 HMDA modified LAR
np.random.seed(42)
n_hmda = 1000

fips_list = ["37063", "37183", "37119", "37081"]  # NC county FIPS codes

mock_hmda = pd.DataFrame({
    "county_code": np.random.choice(fips_list, n_hmda),
    "action_taken": np.random.choice([1, 1, 1, 3], n_hmda),  # 1 = Approved, 3 = Denied
    "income": np.random.randint(30, 200, n_hmda),
    "loan_amount": np.random.randint(100, 800, n_hmda),
    "tract_minority_population_percent": np.random.uniform(5, 65, n_hmda)
})

# ensure county_code has leading zero string format
mock_hmda["county_code"] = mock_hmda["county_code"].astype(str).str.zfill(5)
mock_hmda.to_csv("data/mock_2025_combined_mlar.csv", index=False)

# 2. create mock 2026 CFPB complaints
n_cfpb = 200
complaint_issues = ["Loan modification", "Closing costs", "Trouble payment", "Redlining/Discrimination"]

mock_cfpb = pd.DataFrame({
    "County FIPS": np.random.choice(fips_list, n_cfpb),
    "Issue": np.random.choice(complaint_issues, n_cfpb),
    "Company": np.random.choice(["Wells Fargo", "Bank of America", "Rocket Mortgage"], n_cfpb),
    "complaint_what_happened": [
        "The bank charged an unexpected fee during closing without disclosure."
    ] * n_cfpb
})

mock_cfpb["County FIPS"] = mock_cfpb["County FIPS"].astype(str).str.zfill(5)
mock_cfpb.to_csv("data/mock_2026_cfpb_complaints.csv", index=False)

print("Mock files created in data/ folder!")