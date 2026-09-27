import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

from src.map_engine import build_carto_map
from src.nlp_engine import predict_complaint_category, generate_dispute_letter

# Made with Gemini

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
    "Disparity_Ratio": {
        "title": "Racial Approval Disparity Ratio",
        "formula": "Denial Rate (High Minority Tracts) / Denial Rate (Low Minority Tracts)",
        "meaning": "Compares mortgage rejection probability between majority-minority census tracts and majority-white tracts.",
        "action": "Ratios > 1.5 indicate that minority-majority tracts experience 50%+ higher denial rates.",
        "fmt": ":.2f"
    },
    "HMDA_Denial_Rate": {
        "title": "HMDA Mortgage Denial Rate",
        "formula": "Denial Rate = (Denied Applications) / (Total Applications)",
        "meaning": "The proportion of mortgage applications rejected by financial institutions within the county.",
        "action": "Denial rates above 20% highlight potential credit access bottlenecks.",
        "fmt": ":.1%"
    },
    "JSD_Score": {
        "title": "Jensen-Shannon Divergence (JSD) Score",
        "formula": "JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)",
        "meaning": "Measures statistical divergence between regional demographic composition and loan approval distributions.",
        "action": "High scores (>0.3) flag areas where loan approval rates strongly deviate from demographic expectations.",
        "fmt": ":.3f"
    },
    "Spatial_Entropy": {
        "title": "Spatial Entropic Inequality",
        "formula": "H(X) = - ∑ P(x_i) * log2(P(x_i))",
        "meaning": "Quantifies the randomness and spatial disorder of complaint distribution across tracts.",
        "action": "Low entropy values indicate hyper-localized geographic pockets of financial distress.",
        "fmt": ":.3f"
    },
    "Median_DTI": {
        "title": "Median Debt-to-Income (DTI) Ratio",
        "formula": "Median(Total Monthly Debt Obligations / Gross Monthly Income)",
        "meaning": "Key underwriting metric measuring applicant debt burden across the county.",
        "action": "DTI > 43% generally represents elevated underwriting risk under Qualified Mortgage standards.",
        "fmt": ":.1f"
    },
    "Median_CLTV": {
        "title": "Median Combined Loan-to-Value (CLTV) Ratio",
        "formula": "Median(Total Secured Loan Balances / Appraised Property Value)",
        "meaning": "Measures total borrowing relative to property value, indicating equity buffer.",
        "action": "CLTV > 80% usually requires private mortgage insurance (PMI) and indicates higher leverage.",
        "fmt": ":.1f"
    },
    "Mean_Interest_Rate": {
        "title": "Mean Note Interest Rate",
        "formula": "Average Note Interest Rate across originated loans",
        "meaning": "Baseline borrowing cost charged by lenders in the geographic area.",
        "action": "Higher average rates indicate overall tighter credit pricing or subprime concentration.",
        "fmt": ":.2f%"
    },
    "Mean_Rate_Spread": {
        "title": "Mean Rate Spread",
        "formula": "Average (APOR Difference above threshold)",
        "meaning": "Difference between APR and Average Prime Offer Rate for higher-priced mortgage loans.",
        "action": "Elevated rate spreads flag areas with higher concentration of subprime or predatory loan pricing.",
        "fmt": ":.2f%"
    },
    "Median_Income": {
        "title": "Median Applicant Income",
        "formula": "Median Annual Applicant Income ($k)",
        "meaning": "General economic capacity and earnings baseline for mortgage applicants in the area.",
        "action": "Lower median incomes require lower loan caps to maintain sustainable DTIs.",
        "fmt": ":$,.0f"
    },
    "Median_Loan_Amount": {
        "title": "Median Requested Loan Amount",
        "formula": "Median Loan Amount ($k)",
        "meaning": "Standard scale of mortgage credit extended per transaction in the county.",
        "action": "Evaluated alongside income to measure borrowing leverage.",
        "fmt": ":$,.0f"
    },
    "Total_Complaints": {
        "title": "CFPB Complaint Volume",
        "formula": "Count(Complaints per County)",
        "meaning": "Raw aggregation of formal consumer grievances submitted to the CFPB.",
        "action": "High volumes signify widespread consumer dissatisfaction with loan terms or servicing.",
        "fmt": ":,.0f"
    }
}

# ------------------------------------------
# CACHED MODEL & DATA LOADERS
# ------------------------------------------
@st.cache_resource
def load_mortgage_model():
    """Cache model in memory to prevent repeatedly reading from disk."""
    try:
        return joblib.load("src/mortgage_model.pkl")
    except Exception as e:
        st.warning(f"Mortgage model could not be loaded from disk (`src/mortgage_model.pkl`): {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("data/geo_statistical_summary.parquet")
        
        # Standardize spaces and case for inspection
        clean_cols = {c: c.strip().lower() for c in df.columns}
        rename_dict = {}

        # Pattern matchers for key dataset columns
        for orig_col, lower_col in clean_cols.items():
            if "fips" in lower_col:
                rename_dict[orig_col] = "FIPS"
            elif "county" in lower_col:
                rename_dict[orig_col] = "County"
            elif "state" in lower_col:
                rename_dict[orig_col] = "State"
            elif lower_col in ["lat", "latitude"]:
                rename_dict[orig_col] = "Lat"
            elif lower_col in ["lon", "long", "longitude"]:
                rename_dict[orig_col] = "Lon"
            elif "complaint" in lower_col:
                rename_dict[orig_col] = "Total_Complaints"
            elif "denial" in lower_col:
                rename_dict[orig_col] = "HMDA_Denial_Rate"
            elif "disparity" in lower_col:
                rename_dict[orig_col] = "Disparity_Ratio"
            elif "dti" in lower_col:  # Catches median_dti, debt_to_income, media_dti, etc.
                rename_dict[orig_col] = "Median_DTI"
            elif "cltv" in lower_col or "ltv" in lower_col:
                rename_dict[orig_col] = "Median_CLTV"
            elif "income" in lower_col:
                rename_dict[orig_col] = "Median_Income"
            elif "loan" in lower_col:
                rename_dict[orig_col] = "Median_Loan_Amount"
            elif "spread" in lower_col:
                rename_dict[orig_col] = "Mean_Rate_Spread"
            elif "interest" in lower_col or "rate" in lower_col:
                rename_dict[orig_col] = "Mean_Interest_Rate"
            elif "jsd" in lower_col:
                rename_dict[orig_col] = "JSD_Score"
            elif "entropy" in lower_col or lower_col == "se":
                rename_dict[orig_col] = "Spatial_Entropy"

        df = df.rename(columns=rename_dict)

    except Exception as e:
        st.error(f"Error loading Parquet: {e}")
        # Fallback mock dataset
        data = {
            "FIPS": ["37183", "37119", "37063", "37067", "37081"],
            "County": ["Wake", "Mecklenburg", "Durham", "Forsyth", "Guilford"],
            "State": ["NC"] * 5,
            "Lat": [35.7796, 35.2271, 35.9940, 36.0999, 36.0726],
            "Lon": [-78.6382, -80.8431, -78.8986, -80.2442, -79.7920],
            "Total_Complaints": [450, 1200, 680, 890, 510],
            "HMDA_Denial_Rate": [0.11, 0.24, 0.18, 0.22, 0.15],
            "Disparity_Ratio": [1.1, 1.8, 1.4, 1.6, 1.2],
            "Median_Loan_Amount": [310.0, 280.0, 260.0, 210.0, 225.0],
            "Median_DTI": [36.0, 42.0, 38.5, 41.0, 37.0],
            "Median_Income": [80.5, 69.2, 71.0, 58.4, 61.2],
            "Median_CLTV": [78.0, 85.0, 82.0, 88.0, 80.0],
            "Mean_Interest_Rate": [6.25, 6.85, 6.50, 6.95, 6.40],
            "Mean_Rate_Spread": [0.35, 1.15, 0.65, 0.95, 0.45],
            "JSD_Score": [0.12, 0.45, 0.28, 0.38, 0.19],
            "Spatial_Entropy": [2.1, 1.2, 1.8, 1.4, 2.0]
        }
        df = pd.DataFrame(data)

    # ------------------------------------------
    # GUARANTEED POST-PROCESSING & DEFAULTS
    # ------------------------------------------
    if "Median_DTI" not in df.columns:
        df["Median_DTI"] = 36.0

    if "Disparity_Ratio" not in df.columns:
        df["Disparity_Ratio"] = (df["HMDA_Denial_Rate"] * 1.5).round(2) if "HMDA_Denial_Rate" in df.columns else 1.0

    df["Disparity_Ratio"] = df["Disparity_Ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Clean numeric types
    numeric_cols = [
        "Lat", "Lon", "Total_Complaints", "HMDA_Denial_Rate", "Disparity_Ratio",
        "Median_Loan_Amount", "Median_DTI", "Median_Income", "Median_CLTV",
        "Mean_Interest_Rate", "Mean_Rate_Spread", "JSD_Score", "Spatial_Entropy"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                errors="coerce"
            )

    # Fix positive US longitudes
    if "Lon" in df.columns:
        df["Lon"] = df["Lon"].apply(lambda x: -abs(x) if pd.notnull(x) and x > 0 else x)

    df = df.dropna(subset=["Lat", "Lon"])

    return df

df = load_data()

# Load machine learning model once
mortgage_model = load_mortgage_model()

# ------------------------------------------
# SIDEBAR FILTERS & CONTEXT
# ------------------------------------------
st.sidebar.header("Filter & Controls")

available_states = ["All"] + sorted([str(s) for s in df["State"].dropna().unique().tolist()])
selected_state = st.sidebar.selectbox("Select State", available_states, index=0, key="state_select_sidebar")

if selected_state != "All":
    filtered_df = df[df["State"] == selected_state].copy()
else:
    filtered_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### Metric Reference Guide")
selected_guide_metric = st.sidebar.selectbox(
    "View Metric Definition",
    options=list(METRIC_DEFINITIONS.keys()),
    format_func=lambda x: METRIC_DEFINITIONS.get(x, {}).get("title", x),
    key="metric_guide_sidebar"
)

metric_info = METRIC_DEFINITIONS[selected_guide_metric]
st.sidebar.caption(f"**Formula:** `{metric_info['formula']}`")
st.sidebar.caption(f"**Meaning:** {metric_info['meaning']}")
st.sidebar.caption(f"**Action:** {metric_info['action']}")

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

# ==========================================
# TAB 1: GEOSPATIAL & METRIC ANALYSIS
# ==========================================
with tab_map:
    st.markdown("### Select Analysis Metric")
    
    # Categorized Metric Selector
    metric_categories = {
        "Fair Lending & Disparity": {
            "Disparity Ratio": "Disparity_Ratio",
            "HMDA Denial Rate": "HMDA_Denial_Rate",
            "JSD Divergence Score": "JSD_Score",
            "Spatial Entropy": "Spatial_Entropy"
        },
        "Underwriting & Risk Controls": {
            "Median DTI Ratio": "Median_DTI",
            "Median CLTV Ratio": "Median_CLTV",
            "Mean Interest Rate": "Mean_Interest_Rate",
            "Mean Rate Spread": "Mean_Rate_Spread",
            "Median Applicant Income": "Median_Income",
            "Median Loan Amount": "Median_Loan_Amount"
        },
        "Activity & Volume": {
            "CFPB Complaint Volume": "Total_Complaints"
        }
    }
    
    col_cat, col_met = st.columns([1, 2])
    with col_cat:
        selected_category = st.selectbox(
            "Metric Category:", 
            options=list(metric_categories.keys()),
            key="metric_category_select"
        )
    with col_met:
        selected_metric_label = st.selectbox(
            "Choose Metric to Visualize:",
            options=list(metric_categories[selected_category].keys()),
            key="metric_label_select"
        )
        
    selected_metric_col = metric_categories[selected_category][selected_metric_label]

    st.markdown("---")

    # Regional Summary Banner
    st.markdown("### Regional Summary")
    
    avg_disparity = filtered_df['Disparity_Ratio'].mean() if 'Disparity_Ratio' in filtered_df.columns else 0
    avg_denial = filtered_df['HMDA_Denial_Rate'].mean() if 'HMDA_Denial_Rate' in filtered_df.columns else 0
    avg_dti = filtered_df['Median_DTI'].mean() if 'Median_DTI' in filtered_df.columns else 0
    total_counties = len(filtered_df)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Counties / Tracts", f"{total_counties:,}")
    with m_col2:
        st.metric("Avg Disparity Ratio", f"{avg_disparity:.2f}x")
    with m_col3:
        st.metric("Avg HMDA Denial Rate", f"{avg_denial * 100:.1f}%")
    with m_col4:
        st.metric("Avg County Median DTI", f"{avg_dti:.1f}%")

    st.markdown("---")

    # Map Rendering
    st.markdown("### Geographic Risk & Disparity Map")

    # Force US Center Fallbacks
    if not filtered_df.empty and filtered_df["Lat"].mean() > 0 and filtered_df["Lon"].mean() < 0:
        center_lat = filtered_df["Lat"].mean()
        center_lon = filtered_df["Lon"].mean()
    else:
        center_lat = 37.8  # Default US Lat
        center_lon = -96.0 # Default US Lon

    zoom_lvl = 6 if selected_state != "All" else 4

    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=zoom_lvl, 
        tiles="OpenStreetMap"
    )

    Fullscreen(position="topright", title="Expand map", title_cancel="Exit fullscreen").add_to(m)

    if selected_metric_col in filtered_df.columns:
        col_median = filtered_df[selected_metric_col].median()
    else:
        col_median = 0
    
    for _, row in filtered_df.iterrows():
        val = row.get(selected_metric_col, 0)
        
        # Color scale rules
        if selected_metric_col == "Disparity_Ratio":
            color = "red" if val >= 1.5 else "blue"
        elif selected_metric_col == "HMDA_Denial_Rate":
            color = "red" if val >= 0.20 else "blue"
        elif selected_metric_col == "Median_DTI":
            color = "red" if val >= 43.0 else "blue"
        elif selected_metric_col == "Median_CLTV":
            color = "red" if val >= 80.0 else "blue"
        elif selected_metric_col == "Mean_Rate_Spread":
            color = "red" if val >= 1.5 else "blue"
        else:
            color = "red" if val > col_median else "blue"

        county_name = row.get("County", "Unknown County")
        state_name = row.get("State", "")
        fips = row.get("FIPS", "N/A")
        
        # Smart formatting for tooltips
        dti_raw = row.get('Median_DTI')
        inc_raw = row.get('Median_Income')
        loan_raw = row.get('Median_Loan_Amount')
        disp_raw = row.get('Disparity_Ratio')

        dti_str = f"{dti_raw:.1f}%" if pd.notnull(dti_raw) and dti_raw > 0 else "N/A"
        
        # Handle $k vs raw$ auto-scaling
        if pd.notnull(inc_raw) and inc_raw > 0:
            inc_str = f"${inc_raw:,.0f}k" if inc_raw < 1000 else f"${inc_raw:,.0f}"
        else:
            inc_str = "N/A"

        if pd.notnull(loan_raw) and loan_raw > 0:
            loan_str = f"${loan_raw:,.0f}k" if loan_raw < 1000 else f"${loan_raw:,.0f}"
        else:
            loan_str = "N/A"

        disp_str = f"{disp_raw:.2f}x" if pd.notnull(disp_raw) and disp_raw > 0 else "N/A"

        popup_html = f"""
        <b>{county_name}, {state_name}</b> (FIPS: {fips})<br>
        <b>{selected_metric_label}:</b> {val}<br>
        <hr style="margin: 4px 0;">
        <b>Median Income:</b> {inc_str}<br>
        <b>Median Loan:</b> {loan_str}<br>
        <b>Median DTI:</b> {dti_str}<br>
        <b>Disparity Ratio:</b> {disp_str}
        """

        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_html,
            tooltip=f"{county_name}, {state_name} - {selected_metric_label}: {val}"
        ).add_to(m)

    st_folium(m, width="100%", height=550)

    st.markdown("---")

    # Visualizations
    st.markdown("### Analytical Breakdown & Distribution")

    if selected_metric_col in filtered_df.columns:
        fig_dist = px.histogram(
            filtered_df, 
            x=selected_metric_col, 
            nbins=40,
            title=f"Distribution of {selected_metric_label}",
            color_discrete_sequence=['#1f77b4'],
            height=380
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        if 'County' in filtered_df.columns:
            top_df = filtered_df.nlargest(15, selected_metric_col)
            fig_rank = px.bar(
                top_df, 
                x=selected_metric_col, 
                y='County', 
                orientation='h',
                title=f"Top 15 Counties by {selected_metric_label}",
                color=selected_metric_col,
                color_continuous_scale="Reds",
                height=480
            )
            fig_rank.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.warning(f"Column '{selected_metric_col}' is not present in the dataset.")

# ==========================================
# TAB 2: PREDICTIVE RISK & EXPLAINABILITY
# ==========================================
with tab_predict:
    st.subheader("Predictive Risk & Approval Estimator")
    st.caption("Inference and model explainability powered by Random Forest incorporating core HMDA underwriting risk controls.")
    
    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        st.markdown("### Applicant Underwriting Profile")
        
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            income = st.number_input("Annual Income ($k)", value=75, step=5, key="input_inc")
            loan_amount = st.number_input("Loan Amount ($k)", value=250, step=10, key="input_loan")
            dti = st.number_input("Debt-to-Income (DTI) %", value=36.0, step=1.0, key="input_dti")
        with c_in2:
            cltv = st.number_input("Combined Loan-to-Value (CLTV) %", value=80.0, step=1.0, key="input_cltv")
            interest_rate = st.number_input("Note Interest Rate %", value=6.50, step=0.125, key="input_rate")
            rate_spread = st.number_input("Rate Spread %", value=0.25, step=0.1, key="input_spread")
            
        lti_ratio = loan_amount / income if income > 0 else 0
        st.caption(f"**Calculated Loan-to-Income (LTI) Ratio:** `{lti_ratio:.2f}x` | **DTI:** `{dti:.1f}%` | **CLTV:** `{cltv:.1f}%`")
        
        run_model = st.button("Run Risk & Approval Model", use_container_width=True)

    with col_viz:
        st.markdown("### Model Assessment")
        
        if run_model:
            if mortgage_model is not None:
                # 1. Try predicting with 6 underwriting features
                try:
                    input_data_6 = pd.DataFrame([[
                        loan_amount, income, dti, cltv, interest_rate, rate_spread
                    ]], columns=[
                        'loan_amount', 'income', 'dti', 'cltv', 'interest_rate', 'rate_spread'
                    ])
                    denial_prob = mortgage_model.predict_proba(input_data_6)[0][1]
                    approval_prob = 1.0 - denial_prob
                    
                    st.success("Model Inference Complete")
                    m1, m2 = st.columns(2)
                    m1.metric(label="Estimated Approval Probability", value=f"{approval_prob * 100:.1f}%")
                    m2.metric(label="Estimated Denial Risk", value=f"{denial_prob * 100:.1f}%")
                    st.progress(approval_prob)

                except Exception:
                    # 2. Fallback to 2 basic features (loan_amount, income)
                    try:
                        input_data_2 = pd.DataFrame(
                            [[loan_amount, income]], 
                            columns=['loan_amount', 'income']
                        )
                        denial_prob = mortgage_model.predict_proba(input_data_2)[0][1]
                        approval_prob = 1.0 - denial_prob
                        
                        st.success("Model Inference Complete (Baseline Features)")
                        m1, m2 = st.columns(2)
                        m1.metric(label="Estimated Approval Probability", value=f"{approval_prob * 100:.1f}%")
                        m2.metric(label="Estimated Denial Risk", value=f"{denial_prob * 100:.1f}%")
                        st.progress(approval_prob)
                    except Exception as ex_fallback:
                        st.error(f"Error executing model inference: {ex_fallback}")
            else:
                st.error("Model file (`src/mortgage_model.pkl`) not available. Check your file path.")
        else:
            st.info("Adjust parameters on the left and click 'Run Risk & Approval Model' to inspect predictions.")

    st.markdown("---")
    
    st.subheader("Model Feature Importance & Explainability")
    st.markdown("""
    *Understanding driver impact: Evaluating how debt burden (DTI), equity leverage (CLTV), and pricing (Interest Rate/Spread) contribute to approval decisions.*
    """)
    
    if mortgage_model is not None and hasattr(mortgage_model, "feature_importances_"):
        n_features = len(mortgage_model.feature_importances_)
        if n_features == 6:
            feature_names = ['Loan Amount', 'Applicant Income', 'DTI Ratio', 'CLTV Ratio', 'Interest Rate', 'Rate Spread']
        elif n_features == 2:
            feature_names = ['Loan Amount', 'Applicant Income']
        else:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
            
        importances = mortgage_model.feature_importances_
        
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
        fig_fi.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance visualization is active when `src/mortgage_model.pkl` is loaded and trained.")
        
    with st.expander("💡 How to interpret Underwriting Feature Importance"):
        st.markdown("""
        - **DTI & CLTV Ratios**: Standard objective financial risk measures evaluated under Qualified Mortgage rules. High DTI or CLTV heavily drives loan rejections.
        - **Interest Rate & Rate Spread**: Capture credit pricing terms and subprime loan tiering.
        """)

# ==========================================
# TAB 3: CONSUMER ACTION ASSISTANT (NLP)
# ==========================================
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
            with st.spinner("Analyzing complaint and generating document..."):
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