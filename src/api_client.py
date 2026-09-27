import requests
import pandas as pd

# Made with Gemini

def fetch_cfpb_complaints(state="NC", product="Mortgage", size=500):
    """
    Fetch live complaint data directly from official CFPB API endpoint
    """
    url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    
    params = {
        "state": state,
        "product": product,
        "size": size,
        "sort": "created_date_desc"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # parse search results
        hits = data.get("hits", {}).get("hits", [])
        complaints = [hit["_source"] for hit in hits]
        
        df = pd.DataFrame(complaints)
        
        # standardize columns
        if not df.empty:
            df["zip_code"] = df.get("zip_code", None)
            df["issue"] = df.get("issue", None)
            df["company"] = df.get("company", None)
            df["complaint_what_happened"] = df.get("complaint_what_happened", "")
            
        return df
    
    except Exception as e:
        print(f"CFPB API Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df_live = fetch_cfpb_complaints(state="NC", size=10)
    print(f"Fetched {len(df_live)} live complaints from CFPB API!")