# Consumer Protection and Financial Equity Engine
## Generated with Gemini

An interactive dashboard powered by Streamlit, Folium, and Machine Learning to analyze structural disparities, regional mortgage equity (HMDA), and CFPB consumer complaint trends.

---

## Key Features

- Interactive Spatial Heatmaps: Visualize market disparity metrics like JSD Score, Spatial Entropy, HMDA Denial Rates, and Complaint Volumes across counties using CARTO tiles and Folium.
- Risk and Approval Estimator: Interactive Machine Learning inference (Random Forest) predicting mortgage application approval probabilities based on loan amount, applicant income, and tract demographics.
- Consumer Actionability Engine (NLP): Real-time classification of complaint narratives to predict issue categories, historical monetary relief rates, and generate auto-filled CFPB dispute letters.
- Dual Data Modes: Switch seamlessly between offline local datasets and live queries to the official CFPB API.

---

## Project Structure

```text
├── app.py                      # Main Streamlit application entry point
├── src/
│   ├── api_client.py           # Live CFPB API fetcher module
│   ├── map_engine.py           # Folium / CARTO map rendering engine
│   ├── nlp_engine.py           # NLP complaint classifier & dispute letter generator
│   └── mortgage_model.pkl      # Pre-trained Random Forest model
├── data/                       # Local offline dataset backups
├── requirements.txt            # Python dependency requirements
└── README.md                   # Project documentation