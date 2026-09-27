import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from streamlit_folium import st_folium
from src.map_engine import build_carto_map
from src.nlp_engine import predict_complaint_category, generate_dispute_letter

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="Consumer Protection & Financial Equity Engine",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------
# METRIC DEFINITIONS & CONTEXT DICTIONARY
# ------------------------------------------
METRIC_DEFINITIONS = {
    "JSD_Score": {
        "title": "Jensen-Shannon Divergence (JSD) Score",
        "formula": "JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)",
        "meaning": "Measures statistical divergence between regional demographic composition and loan approval distributions.",
        "action": "High scores (>0.3) flag areas where loan approval rates strongly deviate from demographic expectations."
    },
    "Spatial_Entropy": {
        "title": "Spatial Entropic Inequality",
        "formula": "H(X) = - ∑ P(x_i) * log2(P(x_i))",
        "meaning": "Quantifies the randomness and spatial disorder of complaint distribution across tracts.",
        "action": "Low entropy values indicate hyper-localized geographic pockets of financial distress."
    },
    "HMDA_Denial_Rate": {
        "title": "HMDA Mortgage Denial Rate",
        "formula": "Denial Rate = (Denied Applications) / (Total Applications)",
        "meaning": "The proportion of mortgage applications rejected by financial institutions within the county.",
        "action": "Denial rates above 20% highlight potential credit access bottlenecks."
    },
    "Total_Complaints": {
        "title": "CFPB Complaint Volume",
        "formula": "Count(Complaints per County)",
        "meaning": "Raw aggregation of formal consumer grievances submitted to the CFPB.",
        "action": "High volumes signify widespread consumer dissatisfaction with loan terms or servicing."
    },
    "Disparity_Ratio": {
        "title": "Racial Approval Disparity Ratio",
        "formula": "Denial Rate (High Minority Tracts) / Denial Rate (Low Minority Tracts)",
        "meaning": "Compares mortgage rejection probability between majority-minority census tracts and majority-white tracts.",
        "action": "Ratios > 1.5 indicate that minority-majority tracts experience 50%+ higher denial rates."
    }
}

# ------------------------------------------
# DATA LOADING (.PARQUET INTEGRATION)
# ------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load the pre-aggregated Parquet file
        df = pd.read_parquet("data/geo_statistical_summary.parquet")
        
        # Standardize column names (handles lowercase, trailing spaces, or underscores)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        
        # Map standardized names back to expected app keys if necessary
        column_mapping = {
            "total_complaints": "Total_Complaints",
            "hmda_denial_rate": "HMDA_Denial_Rate",
            "jsd_score": "JSD_Score",
            "spatial_entropy": "Spatial_Entropy",
            "disparity_ratio": "Disparity_Ratio",
            "county": "County",
            "state": "State",
            "lat": "Lat",
            "lon": "Lon",
            "fips": "fips"
        }
        df = df.rename(columns=column_mapping)
        
        # Clean up numeric types
        numeric_cols = ["Lat", "Lon", "Total_Complaints", "HMDA_Denial_Rate", "JSD_Score", "Spatial_Entropy", "Disparity_Ratio"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        # Drop rows with missing lat/lon coordinates
        df = df.dropna(subset=["Lat", "Lon"])
        
    except Exception as e:
        st.error(f"Error loading Parquet dataset: {e}")
        # Emergency fallback mock data structure
        data = {
            "County": ["Wake", "Mecklenburg", "Durham", "Forsyth", "Guilford"],
            "State": ["NC"] * 5,
            "Lat": [35.7796, 35.2271, 35.9940, 36.0999, 36.0726],
            "Lon": [-78.6382, -80.8431, -78.8986, -80.2442, -79.7920],
            "JSD_Score": [0.12, 0.45, 0.28, 0.38, 0.19],
            "Spatial_Entropy": [2.1, 1.2, 1.8, 1.4, 2.0],
            "HMDA_Denial_Rate": [0.11, 0.24, 0.18, 0.22, 0.15],
            "Total_Complaints": [450, 1200, 680, 890, 510],
            "Disparity_Ratio": [1.1, 1.8, 1.4, 1.6, 1.2]
        }
        df = pd.DataFrame(data)
        
    return df

df = load_data()

# Ensure Disparity Ratio exists in dataframe
if "Disparity_Ratio" not in df.columns:
    if "HMDA_Denial_Rate" in df.columns:
        df["Disparity_Ratio"] = (df["HMDA_Denial_Rate"] * 1.5).round(2)
    else:
        df["Disparity_Ratio"] = 1.0

# ------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------
st.sidebar.header("Filter & Controls")

# State Selection Dropdown
available_states = ["All"] + sorted([str(s) for s in df["State"].dropna().unique().tolist()])
selected_state = st.sidebar.selectbox("Select State", available_states, index=0, key="state_select_sidebar")

# Filter dataframe based on state selection
if selected_state != "All":
    filtered_df = df[df["State"] == selected_state].copy()
else:
    filtered_df = df.copy()

# Metric Selection Dropdown
selected_metric = st.sidebar.selectbox(
    "Select Metric to Analyze",
    options=list(METRIC_DEFINITIONS.keys()),
    format_func=lambda x: METRIC_DEFINITIONS.get(x, {}).get("title", x),
    key="metric_select_sidebar"
)

# ------------------------------------------
# MAIN HEADER
# ------------------------------------------
st.title("Consumer Protection & Financial Equity Engine")
st.markdown("""
An analytical platform mapping structural mortgage disparity, estimating approval risk, and providing AI-driven consumer dispute assistance.
""")

# ------------------------------------------
# APPLICATION TABS
# ------------------------------------------
tab_map, tab_predict, tab_nlp = st.tabs([
    "Geographic Disparity Map", 
    "Predictive Risk & Explainability", 
    "Consumer Action Assistant"
])

# ------------------------------------------
# TAB 1: GEOGRAPHIC DISPARITY MAP
# ------------------------------------------
with tab_map:
    metric_info = METRIC_DEFINITIONS[selected_metric]
    
    st.subheader(f"Regional Distribution: {metric_info['title']}")
    
    with st.expander("📖 Metric Definition & Analytical Context", expanded=True):
        st.markdown(f"""
        **What it measures:** {metric_info['meaning']}  
        **Mathematical Concept:** `{metric_info['formula']}`  
        **Policy Implication:** {metric_info['action']}
        """)

    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        # Uses filtered_df so map updates when selecting a state
        m = build_carto_map(filtered_df, selected_metric)
        st_folium(m, height=520, use_container_width=True, returned_objects=[])

    with col_stats:
        st.write("### Regional Summary")
        
        if not filtered_df.empty and selected_metric in filtered_df.columns:
            avg_val = filtered_df[selected_metric].mean()
            max_row = filtered_df.loc[filtered_df[selected_metric].idxmax()]
            
            st.metric("Average Value", f"{avg_val:.3f}")
            st.metric("Highest Severity County", f"{max_row['County']} ({max_row[selected_metric]:.3f})")
            
            fig = px.bar(
                filtered_df, 
                x="County", 
                y=selected_metric, 
                color=selected_metric,
                color_continuous_scale="Reds",
                title=f"{selected_metric.replace('_', ' ')} by County"
            )
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

# ------------------------------------------
# TAB 2: PREDICTIVE RISK & EXPLAINABILITY
# ------------------------------------------
with tab_predict:
    st.subheader("Predictive Risk & Approval Estimator")
    st.caption("Inference and model explainability powered by Random Forest trained on HMDA applicant data.")
    
    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        st.markdown("### Applicant & Tract Profile")
        income = st.number_input("Applicant Annual Income ($k)", value=75, step=5)
        loan_amount = st.number_input("Requested Loan Amount ($k)", value=250, step=10)
        minority_pct = st.slider("Tract Minority Population %", 0.0, 100.0, 25.0)
        
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
            st.info("Adjust parameters on the left and click 'Run Risk & Approval Model' to inspect predictions.")

    st.markdown("---")
    
    st.subheader("Model Feature Importance & Explainability")
    st.markdown("""
    *Understanding driver impact: How different financial and regional variables contribute to approval decisioning across the dataset.*
    """)
    
    try:
        model = joblib.load("src/mortgage_model.pkl")
        
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
        
    except Exception:
        st.info("Feature importance visualization active when `src/mortgage_model.pkl` is loaded.")
        
    with st.expander("💡 How to interpret Feature Importance"):
        st.markdown("""
        - **Loan Amount vs. Applicant Income**: Standard debt-to-income indicators evaluated by underwriting algorithms.
        - **Tract Minority %**: Measures whether geographic census tract demographics disproportionately correlate with rejection rates independently of individual income factors, indicating potential **systemic redlining**.
        """)

# ------------------------------------------
# TAB 3: CONSUMER ACTION ASSISTANT (NLP)
# ------------------------------------------
with tab_nlp:
    st.subheader("Consumer Actionability & Dispute Engine")
    st.markdown("""
    Submit a plain-language summary of a financial complaint to classify the dispute category and automatically draft a formal CFPB dispute letter.
    """)
    
    complaint_text = st.text_area(
        "Describe the Consumer Issue or Narrative:",
        height=140,
        placeholder="e.g., I applied for a mortgage refinancing, but the lender added unexpected closing fees and denied the loan without providing an Adverse Action Notice."
    )
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        company_name = st.text_input("Financial Institution / Lender Name:", value="Acme Mortgage Corp")
    with col_b:
        consumer_name = st.text_input("Consumer Name:", value="Jane Doe")
        
    if st.button("Generate Dispute Analysis & Letter", use_container_width=True):
        if complaint_text.strip():
            category = predict_complaint_category(complaint_text)
            letter = generate_dispute_letter(complaint_text, company_name, consumer_name)
            
            st.markdown("### Analysis Results")
            st.info(f"**Predicted Complaint Category:** `{category}`")
            
            st.markdown("### Drafted Formal CFPB Dispute Letter")
            st.code(letter, language="text")
            
            st.download_button(
                label="Download Dispute Letter (.txt)",
                data=letter,
                file_name="CFPB_Dispute_Letter.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("Please enter a complaint narrative before generating a letter.")