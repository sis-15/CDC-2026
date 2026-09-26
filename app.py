import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px

# Import custom modularized engines
from src.api_client import fetch_cfpb_complaints
from src.map_engine import build_carto_map
from src.nlp_engine import classify_complaint_narrative, generate_dispute_letter

# ==========================================
# 1. Page Config (MUST BE FIRST STREAMLIT CALL)
# ==========================================
st.set_page_config(
    page_title="CFPB & HMDA Disparity Tracker",
    page_icon="🗺️",
    layout="wide"
)

# ==========================================
# 2. Key & Data Loading
# ==========================================
# carto_key = st.secrets.get("CARTOMAPS_API_KEY", "")

@st.cache_data
def load_data():
    return pd.read_csv("data/mock_geo_statistical_summary.csv")

df = load_data()

@st.cache_data(ttl=3600)
def load_live_cfpb_data(state="NC"):
    df_live = fetch_cfpb_complaints(state=state, product="Mortgage", size=1000)
    if df_live.empty:
        df_live = pd.read_csv("data/mock_2026_cfpb_complaints.csv")
    return df_live

# ==========================================
# 3. App Header & Sidebar Controls
# ==========================================
st.title("🛡️ Consumer Protection & Financial Equity Engine")
st.markdown("""
*Analyzing structural disparities, regional complaint divergence, and mortgage equity using CFPB & HMDA data.*
""")

st.sidebar.header("Filter & Controls")

# Heatmap metric selector
selected_metric = st.sidebar.selectbox(
    "Select Heatmap Metric",
    ["JSD_Score", "Spatial_Entropy", "HMDA_Denial_Rate", "Total_Complaints"],
    help="JSD: Market Disparity | Spatial Entropy: Inequality Chaos | HMDA: Denial Rate"
)

# Data source mode toggle
data_source = st.sidebar.radio("Data Source Mode", ["Local Offline Dataset", "Live CFPB API"])

if data_source == "Live CFPB API":
    selected_state = st.sidebar.selectbox("Select State for API Query", ["NC", "GA", "FL", "VA", "SC"])
    with st.spinner("Querying live CFPB API..."):
        df_cfpb = load_live_cfpb_data(state=selected_state)
    st.sidebar.success(f"Connected! {len(df_cfpb)} live complaints fetched.")
else:
    df_cfpb = pd.read_csv("data/mock_2026_cfpb_complaints.csv")

# CARTO Key status indicator
# if not carto_key:
#     st.sidebar.error("⚠️ CARTO Key Missing in Secrets!")
# else:
#     st.sidebar.success("🔑 CARTO Key Loaded")

# ==========================================
# 4. Tabs & Layout
# ==========================================
tab_map, tab_predict, tab_nlp = st.tabs([
    "🗺️ Interactive Heatmap & Analytics", 
    "📈 Risk & Approval Predictor", 
    "📝 Consumer Actionability Engine (NLP)"
])

# ------------------------------------------
# TAB 1: MAP & ANALYTICS
# ------------------------------------------
with tab_map:
    st.subheader(f"Geographic Overview: {selected_metric.replace('_', ' ')}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Build map using modularized map_engine
        m = build_carto_map(df, selected_metric)
        st_folium(m, height=500, use_container_width=True, returned_objects=[])

    with col2:
        st.write("### Regional Summary")
        st.dataframe(df[["County", "JSD_Score", "Spatial_Entropy", "HMDA_Denial_Rate"]], use_container_width=True)
        
        fig = px.bar(df, x="County", y=selected_metric, color=selected_metric, title=f"{selected_metric} by County")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: PREDICTIVE MODEL & EXPLAINABILITY
# ------------------------------------------
with tab_predict:
    st.subheader("Predictive Risk & Approval Estimator")
    st.caption("Live inference and model explainability powered by Random Forest trained on HMDA applicant data.")
    
    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        st.markdown("### Applicant & Tract Profile")
        income = st.number_input("Applicant Annual Income ($k)", value=75, step=5, help="Annual household income in thousands")
        loan_amount = st.number_input("Requested Loan Amount ($k)", value=250, step=10, help="Total mortgage loan requested in thousands")
        minority_pct = st.slider("Tract Minority Population %", 0.0, 100.0, 25.0, help="Percentage of minority population in census tract")
        
        # Calculate derived feature for better context
        lti_ratio = loan_amount / income if income > 0 else 0
        st.caption(f"**Calculated Loan-to-Income (LTI) Ratio:** `{lti_ratio:.2f}x`")
        
        run_model = st.button("Run Risk & Approval Model", use_container_width=True)

    with col_viz:
        st.markdown("### Model Assessment")
        
        if run_model:
            try:
                model = joblib.load("src/mortgage_model.pkl")
                input_data = pd.DataFrame(
                    [[loan_amount, income, minority_pct]], 
                    columns=['loan_amount', 'income', 'tract_minority_population_percent']
                )
                
                denial_prob = model.predict_proba(input_data)[0][1]
                approval_prob = 1.0 - denial_prob
                
                st.success("Model Inference Complete")
                
                # Metric display
                m1, m2 = st.columns(2)
                m1.metric(label="Estimated Approval Probability", value=f"{approval_prob * 100:.1f}%")
                m2.metric(label="Estimated Denial Risk", value=f"{denial_prob * 100:.1f}%")
                
                st.progress(approval_prob)
                
                if denial_prob > 0.5:
                    st.warning("High Denial Risk Flagged: Loan-to-income ratio or regional tract characteristics indicate elevated risk.")
                else:
                    st.info("Favorable Approval Outlook: Applicant income and loan profile fall within typical approval ranges.")
                    
            except Exception as e:
                st.error(f"Error executing model inference: {e}")
        else:
            st.info("Adjust the parameters on the left and click 'Run Risk & Approval Model' to inspect predictions.")

    st.markdown("---")
    
    # Feature Importance Visualization
    st.subheader("Model Feature Importance & Explainability")
    st.markdown("""
    *Understanding driver impact: How different financial and regional variables contribute to approval decisioning across the dataset.*
    """)
    
    try:
        model = joblib.load("src/mortgage_model.pkl")
        
        # Extract feature importances from trained Random Forest
        feature_names = ['Loan Amount', 'Applicant Income', 'Tract Minority %']
        importances = model.feature_importances_
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Random Forest Feature Importance Weights',
            labels={'Importance': 'Relative Importance Weight', 'Feature': 'Model Feature'},
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_fi.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_fi, use_container_width=True)
        
    except Exception as e:
        st.warning("Feature importance plot offline: Ensure `src/mortgage_model.pkl` is present in your repository.")

# ------------------------------------------
# TAB 3: NLP ASSISTANT
# ------------------------------------------
with tab_nlp:
    st.subheader("Consumer Incident Classifier & Resolution Assistant")
    
    selected_county = st.selectbox("Select your county:", df["County"].unique())
    user_narrative = st.text_area(
        "Describe your financial issue / complaint:",
        placeholder="e.g., The bank charged me an unexpected fee during closing and failed to disclose..."
    )
    
    if st.button("Analyze Narrative & Generate Action Plan"):
        if user_narrative:
            st.info("Classifying issue via NLP Pipeline...")
            
            # Use modularized nlp_engine functions
            category, relief_rate = classify_complaint_narrative(user_narrative)
            dispute_letter = generate_dispute_letter(selected_county, user_narrative, category)
            
            st.write("### Analysis Results")
            st.write(f"**Predicted Sub-issue:** {category}")
            st.write(f"**Historical Relief Rate:** {relief_rate * 100:.0f}% of similar complaints resulted in monetary relief.")
            
            st.subheader("Generated CFPB Dispute Letter Template")
            st.code(dispute_letter, language="markdown")
        else:
            st.warning("Please enter a description of your issue.")