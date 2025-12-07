import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from causal_pipeline import CausalIntelligenceEngine
from data_loader import DataLoader
import logging

# --- Page Configuration ---
st.set_page_config(
    page_title="Causal Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🧠 Causal Intelligence Platform")
st.markdown("### Production-Grade Causal Inference & Uplift Modeling")
st.markdown("---")

# --- Sidebar: Data & Settings ---
with st.sidebar:
    st.header("1. Data Configuration")
    data_source = st.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV"])
    
    df = None
    
    if data_source == "Generate Synthetic Data":
        n_samples = st.slider("Number of Samples", 1000, 50000, 10000)
        if st.button("Generate Data"):
            loader = DataLoader(n_samples=n_samples)
            df = loader.generate_data()
            st.session_state['data'] = df
            st.success(f"Generated {n_samples} records.")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df

    # Check if data exists in session state
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.divider()
        st.header("2. Variable Selection")
        
        all_cols = df.columns.tolist()
        treatment = st.selectbox("Treatment (Intervention)", all_cols, index=all_cols.index("used_new_feature") if "used_new_feature" in all_cols else 0)
        outcome = st.selectbox("Outcome (Target)", all_cols, index=all_cols.index("total_spend") if "total_spend" in all_cols else 1)
        confounders = st.multiselect("Confounders (Biasing Factors)", all_cols, default=["account_age", "is_power_user"] if "account_age" in all_cols else [])
        
        segment_col = st.selectbox("Segment for Uplift (Optional)", [None] + all_cols, index=all_cols.index("is_power_user") + 1 if "is_power_user" in all_cols else 0)
        
        run_analysis = st.button("🚀 Run Causal Analysis")
    else:
        run_analysis = False

# --- Main Dashboard Area ---
if 'data' in st.session_state:
    # Preview Data
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(df.head())

    if run_analysis:
        try:
            with st.spinner("Building Causal Graph & Matching Users..."):
                # Initialize Engine
                engine = CausalIntelligenceEngine(
                    data=df,
                    treatment_col=treatment,
                    outcome_col=outcome,
                    confounders=confounders
                )
                
                # Run Pipeline
                engine.create_causal_graph()
                engine.identify_effect()
                ate = engine.estimate_effect()
                
                # Run Validation
                refute_rcc, refute_placebo = engine.validate_robustness()
                
                # --- RESULTS SECTION ---
                st.markdown("### 🎯 Executive Summary")
                
                col1, col2, col3 = st.columns(3)
                
                # Naive Estimate (Simple Difference)
                naive_diff = df[df[treatment]==1][outcome].mean() - df[df[treatment]==0][outcome].mean()
                
                with col1:
                    st.metric(label="Naive Estimate (Biased)", value=f"${naive_diff:.2f}", delta="Raw Correlation")
                
                with col2:
                    st.metric(label="Causal Estimate (True ROI)", value=f"${ate:.2f}", delta_color="normal", help="Adjusted for confounders")
                    
                with col3:
                    bias_correction = naive_diff - ate
                    st.metric(label="Bias Corrected", value=f"${bias_correction:.2f}", delta="- Overestimation Removed", delta_color="inverse")

                # --- TABS for Details ---
                tab1, tab2, tab3 = st.tabs(["📈 Uplift Analysis", "🔍 Robustness Checks", "🕸️ Causal Graph"])
                
                with tab1:
                    if segment_col:
                        st.subheader(f"Who responds best? (Segment: {segment_col})")
                        uplift_results = engine.estimate_heterogeneous_effect(segment_col)
                        
                        # Prepare data for plotting
                        seg_names = list(uplift_results.keys())
                        seg_values = list(uplift_results.values())
                        uplift_df = pd.DataFrame({"Segment": seg_names, "Incremental Value ($)": seg_values})
                        
                        # Plotly Bar Chart
                        fig = px.bar(uplift_df, x="Segment", y="Incremental Value ($)", color="Segment", 
                                     title="Heterogeneous Treatment Effect", text_auto='.2f')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Insight Generation
                        if len(seg_values) >= 2:
                            diff = seg_values[1] - seg_values[0] # Assumes binary for simplicity
                            best_segment = seg_names[1] if diff > 0 else seg_names[0]
                            st.info(f"💡 **Strategy Insight:** The '{best_segment}' segment generates significantly higher ROI. Prioritize rollout here.")
                    else:
                        st.warning("Select a 'Segment' in the sidebar to view Uplift Analysis.")

                with tab2:
                    st.subheader("Model Validation")
                    st.write("To trust these numbers, we stressed-tested the model assumptions:")
                    
                    check_1, check_2 = st.columns(2)
                    with check_1:
                        st.markdown("**Test 1: Placebo Treatment**")
                        st.caption("If we fake the treatment, does effect go to zero?")
                        st.code(refute_placebo)
                    
                    with check_2:
                        st.markdown("**Test 2: Random Common Cause**")
                        st.caption("If we add random noise, does the estimate stay stable?")
                        st.code(refute_rcc)

                with tab3:
                    st.subheader("Assumed Causal DAG")
                    try:
                        st.image("notebooks/plots/causal_graph.png", caption="Causal Directed Acyclic Graph")
                    except:
                        st.warning("Graph image not found. Ensure Graphviz is installed.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    # Landing Page State
    st.info("👈 Please generate or upload data in the sidebar to begin.")
    st.markdown("""
    ### Why use this platform?
    1. **Beyond Correlation:** Standard charts lie. This tool finds *causality*.
    2. **Selection Bias Correction:** Handles 'Power User' bias automatically.
    3. **Uplift Modeling:** Tells you exactly *who* to target.
    """)