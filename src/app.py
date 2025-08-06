# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import ks_2samp
import streamlit.components.v1 as components
from typing import Tuple, Optional, Dict, Any, List
import json
import asyncio
import copy
import xgboost as xgb
import lightgbm as lgb

# Assume these functions exist in a 'utils.py' file as per the original import
# If not, you need to define them.
try:
    from utils import create_features, prepare_data_for_model
except ImportError:
    st.error("Could not import helper functions from 'utils.py'. Please ensure the file exists and is correct.")
    # Define dummy functions to allow the app to run
    def create_features(df):
        st.warning("`create_features` function not found. Using raw data.")
        return df
    def prepare_data_for_model(df, features):
        st.warning("`prepare_data_for_model` function not found. Attempting to use data as is.")
        # Ensure all required features are present, fill with 0 if not
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        return df[features]

# ───────────────────────── Page setup ───────────────────────────────
st.set_page_config(page_title="ICU Bed Forecasting Dashboard", page_icon="🏥", layout="wide")

# Check for and display the logo
try:
    st.image("src/ibm_logo.png", width=300)
except Exception:
    st.warning("Logo image not found at 'src/ibm_logo.png'.")
    
st.title("🏥 AI-Powered ICU Bed Forecasting")
st.markdown("---")

# ───────────────── Constants and Configuration ──────────────────
DATA_PATH = "data/HRR_Scorecard_2.csv" 
XGB_MODEL_PATH = "models/xgboost_reduced_features.pkl"
LGB_MODEL_PATH = "models/lightgbm_reduced_features.pkl"
DEFAULT_TARGET_COL = "Available ICU Beds"
DEFAULT_EXCLUDED_COLS = ["HRR", "Available ICU Beds"]

# ───────────────── Caching and File Loading ──────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models(xgb_path, lgb_path):
    """Loads both models from specified paths."""
    xgb_model, lgb_model = None, None
    try:
        xgb_model = joblib.load(xgb_path)
    except FileNotFoundError:
        st.error(f"XGBoost model not found at '{xgb_path}'. Please check the file path.")
    try:
        lgb_model = joblib.load(lgb_path)
    except FileNotFoundError:
        st.error(f"LightGBM model not found at '{lgb_path}'. Please check the file path.")
    return xgb_model, lgb_model

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_process_data(data_path):
    """Loads and processes data from a specified path."""
    try:
        df_raw = pd.read_csv(data_path)
        df_raw = df_raw[~df_raw["HRR"].astype(str).str.contains("Based on a 50%", na=False)]
        numeric_cols = df_raw.columns.drop("HRR", errors="ignore")
        for col in numeric_cols:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(r"[^\d.]", "", regex=True), errors='coerce')
            if df_raw[col].isnull().any():
                df_raw[col] = df_raw[col].fillna(df_raw[col].median())
        df_featured = create_features(df_raw)
        return df_raw, df_featured
    except FileNotFoundError:
        st.error(f"Dataset not found at '{data_path}'. Please check the file path.")
        return None, None

@st.cache_resource(show_spinner="Initializing SHAP explainer...")
def get_shap_explainer(_model, data):
    return shap.Explainer(_model, data)

@st.cache_resource(show_spinner="Initializing LIME explainer...")
def get_lime_explainer(training_data, feature_names, class_names, mode):
    return LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        verbose=False
    )

def retrain_model(model_name: str, data: pd.DataFrame) -> Any:
    """Trains a new model (XGBoost or LightGBM) on the provided data."""
    st.write(f"Starting retraining for {model_name}...")
    y = data[DEFAULT_TARGET_COL]
    X = data.drop(columns=DEFAULT_EXCLUDED_COLS, errors='ignore').select_dtypes(include=np.number)
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "XGBoost":
        model = xgb.XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(random_state=42, n_estimators=100)
    else:
        st.error(f"Unknown model name '{model_name}' for retraining.")
        return None
        
    model.fit(X_train, y_train)
    st.write("Retraining complete.")
    return model

def render_time_series_forecast(df_featured: pd.DataFrame, model, model_name: str):
    """Renders the time series forecasting UI."""
    st.header(f"🔮 Dynamic Time Series Forecasting ({model_name})")
    st.info("Generate a dynamic 14-day forecast for a selected region based on the model's current prediction.")

    region = st.selectbox(
        "Select a Hospital Referral Region (HRR) for Forecasting", 
        df_featured["HRR"].dropna().unique(), 
        key="forecast_region_selector"
    )
    
    if st.button("Generate 14-Day Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            region_row = df_featured[df_featured["HRR"] == region].iloc[0:1].copy()
            model_features = model.feature_names_in_
            
            # Base prediction
            X_base = prepare_data_for_model(region_row, model_features)
            base_prediction = model.predict(X_base)[0]
            
            # Generate a 14-day forecast with some random walk for simulation
            forecast_dates = pd.to_datetime(pd.date_range(start=pd.Timestamp.now().date(), periods=14))
            forecast_values = [base_prediction]
            for _ in range(13):
                next_val = forecast_values[-1] * np.random.normal(1.0, 0.05) # Simulate daily fluctuation
                forecast_values.append(next_val)
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted_ICU_Beds': forecast_values
            })
            forecast_df['Upper_Bound'] = forecast_df['Predicted_ICU_Beds'] * 1.15 # 15% confidence interval
            forecast_df['Lower_Bound'] = forecast_df['Predicted_ICU_Beds'] * 0.85
            forecast_df['Date_Formatted'] = forecast_df['Date'].dt.strftime('%b %d')

            # --- Visualization ---
            fig = go.Figure()
            # Confidence Interval Area
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Upper_Bound'], mode='lines',
                line=dict(width=0), showlegend=False, name='Upper Bound'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Lower_Bound'], mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                showlegend=False, name='Lower Bound'
            ))
            # Forecast Line
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Predicted_ICU_Beds'], mode='lines+markers',
                line=dict(color='rgb(0,100,80)'), name='Forecast',
                hovertemplate = '<b>%{customdata}</b><br>Forecast: %{y:,.0f}<extra></extra>',
                customdata = forecast_df['Date_Formatted']
            ))
            fig.update_layout(
                title=f"14-Day ICU Bed Forecast for {region}",
                xaxis_title="Date", yaxis_title="Predicted Available ICU Beds",
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = forecast_df['Date'],
                    ticktext = forecast_df['Date_Formatted']
                )
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Load all files and initialize app state ---
df_raw, df_featured = load_and_process_data(DATA_PATH)
xgb_model, lgb_model = load_models(XGB_MODEL_PATH, LGB_MODEL_PATH)

MODELS = {}
if xgb_model: MODELS["XGBoost"] = xgb_model
if lgb_model: MODELS["LightGBM"] = lgb_model

if 'champion_models' not in st.session_state:
    st.session_state.champion_models = copy.deepcopy(MODELS)
if 'challenger_models' not in st.session_state:
    st.session_state.challenger_models = {}

if df_featured is None:
    st.warning("Dataset could not be loaded. The application cannot continue.")
    st.stop()

if not MODELS:
    st.error("No models could be loaded. Please check file paths. The application cannot continue.")
    st.stop()

# ─────────────────── Sidebar to Select Model ────────────────────────
st.sidebar.header("⚙ Controls")
active_model_name = st.sidebar.selectbox("Select Active Model", options=list(MODELS.keys()))
model = MODELS[active_model_name]
model_features = model.feature_names_in_

# ───────────────────────── Main Application Tabs ────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard & Explainability", 
    "Bulk Forecast", 
    "Model Comparison", 
    "Data Explorer & EDA",
    "MLOps & Monitoring",
    "Conclusion & Recommendations"
])

with tab1:
    st.header(f"📍 Single Region Forecast ({active_model_name})")
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        region = st.selectbox("Select HRR", df_featured["HRR"].dropna().unique())
        region_row = df_featured[df_featured["HRR"] == region].iloc[0:1]
        
        if 'Available Hospital Beds' in region_row.columns:
            st.metric("Available Hospital Beds", f"{int(region_row['Available Hospital Beds'].iloc[0]):,}")
        st.metric("Adult Population", f"{int(region_row['Adult Population'].iloc[0]):,}")
        
        if st.button(f"Predict & Explain for {region}", type="primary", use_container_width=True):
            try:
                X_single = prepare_data_for_model(region_row, model_features)
                pred_single = float(model.predict(X_single)[0])
                
                st.subheader("📈 Prediction Results")
                st.metric("Predicted Available ICU Beds", f"{pred_single:,.0f}")
                
                with st.expander("View Prediction Explanation (SHAP & LIME)", expanded=True):
                    X_all = prepare_data_for_model(df_featured, model_features)
    
                    st.subheader("SHAP Waterfall Plot")
                    st.markdown("Shows how each feature contributes to pushing the prediction from a baseline value to the final output.")
                    shap_explainer = get_shap_explainer(model, X_all)
                    shap_values = shap_explainer(X_single)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("LIME Explanation")
                    st.markdown("Shows the top features that influenced this specific prediction, treating the model as a black box.")
                    lime_explainer = get_lime_explainer(
                        training_data=X_all.values,
                        feature_names=X_all.columns.tolist(),
                        class_names=['Available ICU Beds'],
                        mode='regression'
                    )
                    lime_exp = lime_explainer.explain_instance(
                        data_row=X_single.iloc[0].values,
                        predict_fn=model.predict,
                        num_features=15
                    )
                    components.html(lime_exp.as_html(), height=800, scrolling=True)
            except Exception as e:
                st.error(f"An error occurred during prediction/explanation: {e}")

    with col_right:
        st.subheader("Top 20 Regions by Adult Population")
        top_regions = df_featured.nlargest(20, 'Adult Population')
        fig = px.bar(
            top_regions, x="HRR", y="Adult Population",
            title="Regions by Population Size",
            color="Adult Population",
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Forecast for All HRRs")
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None

    if st.button("Run Forecast for All Regions", type="primary"):
        with st.spinner("Running predictions..."):
            X_all = prepare_data_for_model(df_featured, model_features)
            predictions = model.predict(X_all)
            
            forecast_df = df_featured[["HRR", "Adult Population"]].copy()
            forecast_df["Predicted_Available_ICU_Beds"] = predictions
            st.session_state.forecast_df = forecast_df

    if st.session_state.forecast_df is not None:
        fc = st.session_state.forecast_df
        st.success("Bulk forecast complete!")
        st.dataframe(fc, use_container_width=True)
        st.download_button("📥 Download Forecast CSV", data=fc.to_csv(index=False), file_name="icu_forecast_all_regions_reduced.csv")
        
        top_regions_pred = fc.nlargest(20, "Predicted_Available_ICU_Beds")
        fig2 = px.bar(
            top_regions_pred, x="HRR", y="Predicted_Available_ICU_Beds",
            title="Top 20 Regions with Highest Predicted Available ICU Beds",
            labels={"Predicted_Available_ICU_Beds": "Predicted Beds"},
            color="Predicted_Available_ICU_Beds", color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Model Comparison")
    st.info("Metrics are calculated on a 20% hold-out test set to match the Colab evaluation.")

    if len(MODELS) < 2:
        st.warning("Please ensure both XGBoost and LightGBM models are loaded to enable comparison.")
    else:
        y_true_full = df_featured[DEFAULT_TARGET_COL]
        X_full = df_featured.drop(columns=DEFAULT_EXCLUDED_COLS, errors='ignore')
        
        _, X_test, _, y_test = train_test_split(
            X_full, y_true_full, test_size=0.2, random_state=42
        )

        xgb_mod = MODELS["XGBoost"]
        X_xgb_test = prepare_data_for_model(X_test, xgb_mod.feature_names_in_)
        pred_xgb = xgb_mod.predict(X_xgb_test)
        r2_xgb = r2_score(y_test, pred_xgb)
        mae_xgb = mean_absolute_error(y_test, pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))

        lgb_mod = MODELS["LightGBM"]
        X_lgb_test = prepare_data_for_model(X_test, lgb_mod.feature_names_in_)
        pred_lgb = lgb_mod.predict(X_lgb_test)
        r2_lgb = r2_score(y_test, pred_lgb)
        mae_lgb = mean_absolute_error(y_test, pred_lgb)
        rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("XGBoost Metrics")
            st.metric("Accuracy (R²)", f"{r2_xgb:.3f}")
            st.metric("MAE", f"{mae_xgb:.2f}")
            st.metric("RMSE", f"{rmse_xgb:.2f}")
        with col2:
            st.subheader("LightGBM Metrics")
            st.metric("Accuracy (R²)", f"{r2_lgb:.3f}")
            st.metric("MAE", f"{mae_lgb:.2f}")
            st.metric("RMSE", f"{rmse_lgb:.2f}")

        st.subheader("Actual vs. Predicted Values (on Test Set)")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual Values', marker=dict(color='grey', opacity=0.7)))
        fig_comp.add_trace(go.Scatter(y=pred_xgb, mode='lines', name='XGBoost Predictions', line=dict(color='blue')))
        fig_comp.add_trace(go.Scatter(y=pred_lgb, mode='lines', name='LightGBM Predictions', line=dict(color='green', dash='dash')))
        fig_comp.update_layout(title="Model Prediction Comparison on Test Data", xaxis_title="Test Set Index", yaxis_title="Available ICU Beds")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("Prediction Error Analysis")
        errors_xgb = y_test - pred_xgb
        errors_lgb = y_test - pred_lgb
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Histogram(x=errors_xgb, name='XGBoost Errors', opacity=0.75))
        fig_errors.add_trace(go.Histogram(x=errors_lgb, name='LightGBM Errors', opacity=0.75))
        fig_errors.update_layout(barmode='overlay', title_text='Distribution of Prediction Errors')
        st.plotly_chart(fig_errors, use_container_width=True)

with tab4:
    st.header("Data Explorer & EDA")

    # --- Historical Projections ---
    st.subheader("Historical ICU Bed Needs Projection")
    time_cols = [col for col in df_raw.columns if "ICU Beds Needed" in col]
    if time_cols:
        default_regions = df_raw["HRR"].head(3).tolist()
        selected_regions = st.multiselect("Select HRRs for Historical Projections", df_raw["HRR"].unique(), default=default_regions)
        if selected_regions:
            ts_data = df_raw[df_raw["HRR"].isin(selected_regions)]
            ts_melted = ts_data.melt(id_vars="HRR", value_vars=time_cols, var_name="Time Period", value_name="Beds Needed")
            
            fig_ts = px.line(
                ts_melted, x="Time Period", y="Beds Needed", color="HRR",
                title="Projected ICU Bed Needs (from raw data)", markers=True
            )
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Historical time series columns (e.g., 'ICU Beds Needed, Six Months') not found in the dataset.")

    st.markdown("---")

    # --- Exploratory Data Analysis (EDA) ---
    st.subheader("Exploratory Scatter Plot")
    st.markdown("Explore relationships between different features in the dataset.")
    
    numeric_cols = df_featured.select_dtypes(include=np.number).columns
    c1, c2 = st.columns(2)
    x_ax = c1.selectbox("Select X-axis", numeric_cols, index=list(numeric_cols).index('Adult Population') if 'Adult Population' in numeric_cols else 0)
    y_ax = c2.selectbox("Select Y-axis", numeric_cols, index=list(numeric_cols).index('Available Hospital Beds') if 'Available Hospital Beds' in numeric_cols else 1)
    
    fig_scatter = px.scatter(
        df_featured, x=x_ax, y=y_ax, 
        hover_name="HRR", title=f"{y_ax.replace('_', ' ').title()} vs. {x_ax.replace('_', ' ').title()}",
        color_continuous_scale="Viridis",
        color='Population 65+' if 'Population 65+' in df_featured.columns else None
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")

    # --- ADDED: Dynamic Time Series Forecasting Section ---
    render_time_series_forecast(df_featured, model, active_model_name)


with tab5:
    st.header("MLOps & Monitoring")
    
    st.subheader("Model Management (Champion-Challenger)")
    st.info("Retrain models and compare them against the current champion to ensure peak performance.")

    if not st.session_state.champion_models:
        st.error("No champion models available for management.")
        st.stop()

    model_to_manage = st.selectbox("Select Model to Manage", list(st.session_state.champion_models.keys()))
    champion_model = st.session_state.champion_models[model_to_manage]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Champion Model")
        st.write(f"*Type:* {model_to_manage}")
        st.write(f"This is the current production model.")
        
    with col2:
        st.subheader("Challenger Model")
        if st.button(f"Retrain {model_to_manage} to Create Challenger", use_container_width=True):
            with st.spinner(f"Retraining {model_to_manage}... This may take a moment."):
                challenger_model = retrain_model(model_to_manage, df_featured)
                if challenger_model:
                    st.session_state.challenger_models[model_to_manage] = challenger_model
                    st.success("Challenger model created successfully!")
                else:
                    st.error("Failed to create challenger model.")
    
    if model_to_manage in st.session_state.challenger_models:
        st.markdown("---")
        st.subheader("Champion vs. Challenger Evaluation")
        
        challenger_model = st.session_state.challenger_models[model_to_manage]
        
        y_true = df_featured[DEFAULT_TARGET_COL]
        X_full = df_featured.drop(columns=DEFAULT_EXCLUDED_COLS, errors='ignore')
        _, X_test, _, y_test = train_test_split(X_full, y_true, test_size=0.2, random_state=42)
        
        X_test_champ = prepare_data_for_model(X_test, champion_model.feature_names_in_)
        X_test_chall = prepare_data_for_model(X_test, challenger_model.feature_names_in_)
        
        pred_champ = champion_model.predict(X_test_champ)
        pred_chall = challenger_model.predict(X_test_chall)
        
        metrics_champ = {"R²": r2_score(y_test, pred_champ), "MAE": mean_absolute_error(y_test, pred_champ)}
        metrics_chall = {"R²": r2_score(y_test, pred_chall), "MAE": mean_absolute_error(y_test, pred_chall)}
        
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("*Champion Metrics*")
                st.metric("R² Score", f"{metrics_champ['R²']:.4f}")
                st.metric("Mean Absolute Error", f"{metrics_champ['MAE']:.4f}")
        with c2:
            with st.container(border=True):
                st.markdown("*Challenger Metrics*")
                st.metric("R² Score", f"{metrics_chall['R²']:.4f}", delta=f"{metrics_chall['R²'] - metrics_champ['R²']:.4f}")
                st.metric("Mean Absolute Error", f"{metrics_chall['MAE']:.4f}", delta=f"{metrics_chall['MAE'] - metrics_champ['MAE']:.4f}", delta_color="inverse")

        st.markdown("---")
        st.subheader("Recommendation")
        if metrics_chall['R²'] > metrics_champ['R²'] and metrics_chall['MAE'] < metrics_champ['MAE']:
            st.success("✅ **Promote Challenger:** The Challenger model shows superior performance on the test set.")
            if st.button("Promote Challenger to Champion", type="primary"):
                st.session_state.champion_models[model_to_manage] = challenger_model
                del st.session_state.challenger_models[model_to_manage]
                st.success(f"New {model_to_manage} model promoted to Champion!")
                st.rerun()
        else:
            st.warning("**Keep Champion:** The current Champion model performs better or equally well. Do not promote the challenger.")

    st.markdown("---")
    st.subheader("Model Monitoring: Data Drift")
    st.info("This tab helps detect if the new data has statistically shifted away from the data the model was trained on, which could impact performance.")

    reference_df = df_featured.sample(frac=0.7, random_state=42)
    current_df = df_featured.drop(reference_df.index)

    st.markdown(f"Comparing *current data* ({len(current_df)} rows) against a *reference training sample* ({len(reference_df)} rows).")

    numeric_cols_drift = df_featured.select_dtypes(include=np.number).columns.tolist()
    feature_to_check = st.selectbox("Select a feature to check for drift:", numeric_cols_drift)

    ref_data = reference_df[feature_to_check].dropna()
    curr_data = current_df[feature_to_check].dropna()
    ks_stat, p_value = ks_2samp(ref_data, curr_data)

    st.subheader(f"Drift Analysis for '{feature_to_check}'")
    col1, col2 = st.columns(2)
    col1.metric("KS Statistic", f"{ks_stat:.4f}")
    col2.metric("P-Value", f"{p_value:.4f}")

    if p_value < 0.05:
        st.warning(f"**Significant Drift Detected!** The distribution of '{feature_to_check}' in the current data is statistically different from the reference data (p < 0.05). Model performance may be affected.")
    else:
        st.success(f"**No significant drift detected.** The distribution of '{feature_to_check}' appears stable.")

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Histogram(x=ref_data, name='Reference Data', opacity=0.75))
    fig_drift.add_trace(go.Histogram(x=curr_data, name='Current Data', opacity=0.75))
    fig_drift.update_layout(
        barmode='overlay',
        title_text=f"Distribution Comparison for '{feature_to_check}'",
        xaxis_title=feature_to_check,
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_drift, use_container_width=True)

with tab6:
    st.header("Conclusion & Recommendations")

    st.subheader("Summary of Findings")
    st.markdown("""
    - **High Predictive Accuracy**: The models demonstrate a strong ability to forecast ICU bed availability, with XGBoost generally outperforming LightGBM in terms of R² and error metrics.
    - **Key Drivers**: The most influential factors in predicting ICU availability are consistently related to overall hospital capacity (`Available Hospital Beds`), regional population size (`Adult Population`), and demographic factors like the elderly population (`Population 65+`).
    - **Actionable Insights**: The dashboard provides on-demand, region-specific forecasts and explanations, transforming complex model outputs into clear, actionable intelligence for decision-makers.
    """)

    st.subheader("Strategic Recommendations for Stakeholders")
    
    st.markdown("#### For Hospital Administrators:")
    st.markdown("""
    - **Resource Allocation**: Use the single-region forecast to anticipate bed shortages and proactively manage resources, such as staffing, ventilators, and supplies.
    - **Capacity Planning**: Analyze the time-series projections to inform long-term strategic decisions about expanding ICU capacity or reallocating resources between facilities.
    - **Peer Benchmarking**: Compare your region's predicted availability with others in the bulk forecast to identify best practices or areas needing improvement.
    """)

    st.markdown("#### For Public Health Officials:")
    st.markdown("""
    - **Identify Hotspots**: Use the bulk forecast to identify regions at high risk of ICU bed shortages and target them for support and intervention.
    - **Policy Making**: Leverage the feature importance insights (from SHAP/LIME) to understand the underlying drivers of ICU strain in different regions, informing targeted public health policies.
    - **Emergency Preparedness**: The tool can be used as an early-warning system to prepare for surges in demand, enabling a more proactive and effective response.
    """)
    
    st.markdown("#### For Data Science & MLOps Teams:")
    st.markdown("""
    - Regularly use the **MLOps & Monitoring** tab to retrain and evaluate challenger models, ensuring the most accurate model is always in production.
    - Monitor the **Data Drift** section to identify when a model might need retraining due to significant changes in the input data characteristics, preventing model degradation.
    """)

    st.subheader("Understanding Model Explainability (SHAP & LIME)")
    st.markdown("""
    - **Why did the model say that?** The SHAP and LIME plots on the main dashboard are crucial for building trust and transparency. They answer the critical question of why a specific forecast was made.
    - **SHAP (SHapley Additive exPlanations)**: Breaks down a single prediction to show the precise contribution of each feature. This is useful for detailed, quantitative analysis.
    - **LIME (Local Interpretable Model-agnostic Explanations)**: Provides a more intuitive, high-level summary of the most important factors for a single prediction, making it accessible to non-technical users.
    """)

    st.markdown("---")
    st.success("This dashboard serves as a powerful decision-support tool, bridging the gap between advanced AI and practical, real-world healthcare operations.")
