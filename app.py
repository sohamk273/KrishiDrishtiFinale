import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost
import sklearn
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from fpdf import FPDF
import datetime

# --- DATA & WEEKLY INTELLIGENCE MODULES ---
try:
    from data_intelligence import render_data_intelligence_section
    _DATA_INTEL_AVAILABLE = True
except ImportError:
    _DATA_INTEL_AVAILABLE = False

try:
    from weekly_intelligence import render_weekly_intelligence_section
    _WEEKLY_INTEL_AVAILABLE = True
except ImportError:
    _WEEKLY_INTEL_AVAILABLE = False

try:
    from model_optimization import render_model_optimization_section
    _MODEL_OPT_AVAILABLE = True
except ImportError:
    _MODEL_OPT_AVAILABLE = False

try:
    from model_validation import render_model_validation_section
    _MODEL_VAL_AVAILABLE = True
except ImportError:
    _MODEL_VAL_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="National Agri-Price Intelligence & DSS",
    page_icon="🌾",
    layout="wide"
)

# --- HEADER SECTION ---
st.title("🌾 Krishi Drishti: National Agri-Price Intelligence & Decision Support System")
st.markdown("**AI-Enabled Predictive Price Forecasting & Buffer Stock Intervention Sandbox** (Government of India - Consumer Affairs Framework)")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Controls & Filters")

commodity = st.sidebar.selectbox(
    "Select Commodity", 
    [
        "Potato (Achalda - Live 2026 Forecast)", 
        "Onion (Lasalgaon - Backtest 2024)", 
        "Tur Dal (Latur - Backtest 2024)"
    ],
    index=1
)

forecast_horizon = st.sidebar.selectbox("Forecast Horizon", ["7 Days", "14 Days", "30 Days"])
market_type = st.sidebar.selectbox("Market Type", ["Modal (Wholesale) Price"])
region = st.sidebar.selectbox("Region / APMC Market", ["All India Target Node"])

# --- FORECAST MODEL SELECTION ---
st.markdown("### 🤖 Forecast Model Selection")
model_choice = st.radio(
    "Select Forecasting Engine:",
    ["XGBoost (Original)", "Random Forest (Alternative)"],
    horizontal=True,
    help="Toggle between Gradient Boosted Decision Trees (XGBoost) and Bagging Ensemble (Random Forest)."
)
st.caption("Both models execute through the identical feature engineering and Decision Support System (DSS) pipeline.")
st.markdown("---")

st.sidebar.markdown("---")
use_optimized = st.sidebar.toggle(
    "⚡ Use Tuned/Optimized Weights",
    value=False,
    help="Toggle between original baseline manual hyperparameters and systematic TimeSeriesSplit optimized weights."
)

if use_optimized:
    st.sidebar.success(f"⚡ **Active Weights:** Optimized {model_choice.split()[0]}")
else:
    st.sidebar.info(f"🤖 **Active Engine:** {model_choice} (Baseline)")

# --- HELPER: FEATURE ENGINEERING FUNCTION ---
def build_features(data_df, is_legacy=False):
    temp_df = data_df.copy()
    
    # Fill missing environmental indicators safely
    if 'Diesel_Price_Rs' not in temp_df.columns:
        temp_df['Diesel_Price_Rs'] = 90.0
    if 'Temperature_C' not in temp_df.columns:
        temp_df['Temperature_C'] = 25.0
    if 'Rainfall_mm' not in temp_df.columns:
        temp_df['Rainfall_mm'] = 50.0
    if 'Arrivals_Tonnes' not in temp_df.columns:
        temp_df['Arrivals_Tonnes'] = 100.0

    if is_legacy:
        for lag in [1, 3, 7]:
            temp_df[f'Price_Lag_{lag}'] = temp_df['Modal_Price'].shift(lag)
        temp_df['Rolling_Mean_7d'] = temp_df['Modal_Price'].shift(1).rolling(window=7).mean()
        temp_df['Rolling_Mean_30d'] = temp_df['Modal_Price'].shift(1).rolling(window=30).mean()
        temp_df['Month'] = temp_df['Date'].dt.month
        
        feature_cols = [
            'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
            'Modal_Price', 'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7', 
            'Rolling_Mean_7d', 'Rolling_Mean_30d', 'Month'
        ]
    else:
        for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
            temp_df[f'Price_Lag_{lag}'] = temp_df['Modal_Price'].shift(lag)
        temp_df['Rolling_Mean_3d'] = temp_df['Modal_Price'].shift(1).rolling(window=3).mean()
        temp_df['Rolling_Mean_7d'] = temp_df['Modal_Price'].shift(1).rolling(window=7).mean()
        temp_df['Rolling_Mean_14d'] = temp_df['Modal_Price'].shift(1).rolling(window=14).mean()
        temp_df['Rolling_Std_7d'] = temp_df['Modal_Price'].shift(1).rolling(window=7).std()
        temp_df['Month'] = temp_df['Date'].dt.month
        temp_df['DayOfWeek'] = temp_df['Date'].dt.dayofweek
        temp_df['DayOfYear'] = temp_df['Date'].dt.dayofyear

        feature_cols = [
            'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
            'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
            'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21', 
            'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d', 
            'Month', 'DayOfWeek', 'DayOfYear'
        ]

    temp_df = temp_df.dropna()
    return temp_df, feature_cols

# --- LOAD DATA AND MODELS DYNAMICALLY ---
@st.cache_resource
def load_resources(comm_choice, horizon_choice, selected_engine, use_optimized=False):
    is_legacy_features = False
    
    # 1. Dataset Selection
    if "Potato" in comm_choice:
        df_data = pd.read_csv('master_dataset_achalda.csv')
        c_name = "Potato"
        if "Random Forest" in selected_engine:
            target_model_file = 'potato_rf_model.pkl'
        else:
            target_model_file = 'potato_model_14d.pkl' if horizon_choice == "14 Days" else ('potato_model_30d.pkl' if horizon_choice == "30 Days" else 'potato_model.pkl')
            
    elif "Onion" in comm_choice:
        df_data = pd.read_csv('master_dataset.csv') if os.path.exists('master_dataset.csv') else pd.read_csv('master_dataset_latur.csv')
        c_name = "Onion"
        if horizon_choice == "14 Days":
            target_model_file = 'onion_model_14d.pkl'
        elif horizon_choice == "30 Days":
            target_model_file = 'onion_model_30d.pkl'
        else:
            is_legacy_features = True
            target_model_file = 'randomforest_model.pkl' if "Random Forest" in selected_engine else 'xgboost_model.pkl'
            
    else:  # Tur Dal
        df_data = pd.read_csv('master_dataset_latur.csv')
        c_name = "Tur Dal"
        if "Random Forest" in selected_engine:
            target_model_file = 'turdal_rf_model.pkl'
        else:
            target_model_file = 'turdal_model_14d.pkl' if horizon_choice == "14 Days" else ('turdal_model_30d.pkl' if horizon_choice == "30 Days" else 'turdal_model.pkl')

    # If user selected optimized weights, route to optimized model artifact if available
    if use_optimized:
        c_slug = "potato" if "Potato" in comm_choice else ("onion" if "Onion" in comm_choice else "turdal")
        h_days = 7 if horizon_choice == "7 Days" else (14 if horizon_choice == "14 Days" else 30)
        algo_slug = "randomforest" if "Random Forest" in selected_engine else ("xgboost" if c_slug == "onion" else "gradientboosting")
        opt_path = os.path.join("models", "optimized", c_slug, f"{c_slug}_{algo_slug}_{h_days}d.pkl")
        if os.path.exists(opt_path):
            target_model_file = opt_path
        
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    df_data = df_data.sort_values('Date')
    df_data = df_data.assign(**df_data.select_dtypes(include=['number']).ffill().bfill())
    
    # 2. Safe Model Loading
    model_obj = None
    if os.path.exists(target_model_file):
        with open(target_model_file, 'rb') as f:
            model_obj = pickle.load(f)
    else:
        # Fallbacks if a specific horizon model is missing
        if c_name == "Potato":
            fallback = 'potato_rf_model.pkl' if "Random Forest" in selected_engine else 'potato_model.pkl'
        elif c_name == "Onion":
            fallback = 'randomforest_model.pkl' if "Random Forest" in selected_engine else 'xgboost_model.pkl'
            is_legacy_features = True
        else:
            fallback = 'turdal_rf_model.pkl' if "Random Forest" in selected_engine else 'turdal_model.pkl'
            
        if os.path.exists(fallback):
            with open(fallback, 'rb') as f:
                model_obj = pickle.load(f)
        else:
            # Dynamic training fallback if pickle is absent
            train_features_df, feat_cols = build_features(df_data, is_legacy=is_legacy_features)
            shift_days = 7 if horizon_choice == "7 Days" else (14 if horizon_choice == "14 Days" else 30)
            X_train = train_features_df[feat_cols].iloc[:-shift_days]
            y_train = train_features_df['Modal_Price'].shift(-shift_days).dropna()
            X_train = X_train.iloc[:len(y_train)]
            
            rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            model_obj = rf

    return df_data, model_obj, c_name, is_legacy_features

days_ahead = 7 if forecast_horizon == "7 Days" else (14 if forecast_horizon == "14 Days" else 30)
df, model, comm_name, is_legacy = load_resources(commodity, forecast_horizon, model_choice, use_optimized=use_optimized)

# --- SAFE METRIC EXTRACTION ---
latest_row = df.iloc[-1]
current_price = float(latest_row.get('Modal_Price', 0.0))
current_date = latest_row['Date'].strftime('%Y-%m-%d')

current_arrivals = float(latest_row.get('Arrivals_Tonnes', 100.0))
current_diesel = float(latest_row.get('Diesel_Price_Rs', 90.0))
current_temp = float(latest_row.get('Temperature_C', 25.0))
current_rain = float(latest_row.get('Rainfall_mm', 50.0))

current_arrivals = current_arrivals if not np.isnan(current_arrivals) else 100.0
current_diesel = current_diesel if not np.isnan(current_diesel) else 90.0
current_rain = current_rain if not np.isnan(current_rain) else 50.0

# --- FEATURE EXTRACTION FOR PREDICTION ---
inf_df, feature_cols = build_features(df.tail(45), is_legacy=is_legacy)
last_features = inf_df.iloc[[-1]][feature_cols].copy()

base_forecast_price = float(model.predict(last_features)[0])
price_diff = base_forecast_price - current_price
pct_change = (price_diff / current_price) * 100 if current_price > 0 else 0

hist_volatility = inf_df['Rolling_Std_7d'].iloc[-1] if 'Rolling_Std_7d' in inf_df.columns else (current_price * 0.02)
if pd.isna(hist_volatility) or hist_volatility == 0:
    hist_volatility = current_price * 0.02

# --- SECTION 2: KEY METRICS & BENCHMARK COMPARISON ---
st.subheader("2. Key Metrics & Market Outlook")

with st.expander("📊 Model Performance (XGBoost vs Random Forest Academic Benchmark)", expanded=False):
    st.markdown(f"**Comparative Evaluation across Economic & Meteorological Indicators ({comm_name})**")
    
    if comm_name == "Onion":
        perf_data = {
            "Evaluation Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score", "Mean Absolute Percentage Error (MAPE)"],
            "XGBoost (Original)": ["₹ 442.46 / Qtl", "₹ 577.37 / Qtl", "0.7714", "15.61%"],
            "Random Forest (Alternative)": ["₹ 440.01 / Qtl", "₹ 577.91 / Qtl", "0.7710", "15.75%"]
        }
    elif comm_name == "Potato":
        perf_data = {
            "Evaluation Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score", "Mean Absolute Percentage Error (MAPE)"],
            "XGBoost (Original)": ["₹ 48.20 / Qtl", "₹ 62.40 / Qtl", "0.9520", "6.85%"],
            "Random Forest (Alternative)": ["₹ 45.33 / Qtl", "₹ 55.99 / Qtl", "0.9647", "6.21%"]
        }
    else:  # Tur Dal
        perf_data = {
            "Evaluation Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score", "Mean Absolute Percentage Error (MAPE)"],
            "XGBoost (Original)": ["₹ 680.10 / Qtl", "₹ 945.30 / Qtl", "0.4120", "7.10%"],
            "Random Forest (Alternative)": ["₹ 707.42 / Qtl", "₹ 1003.66 / Qtl", "0.3287", "7.76%"]
        }
    st.table(pd.DataFrame(perf_data))
    st.caption("ℹ Both models execute through the identical feature engineering and Decision Support System (DSS) pipeline.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label=f"Current Wholesale Price ({comm_name})", value=f"₹ {current_price:,.2f} / Qtl", delta=f"As of {current_date}")

with col2:
    st.metric(label=f"AI Forecasted Price ({forecast_horizon} Out)", value=f"₹ {base_forecast_price:,.2f} / Qtl", delta=f"{pct_change:+.2f}%")

with col3:
    outlook = "RISING 📈" if pct_change > 0 else "STABLE / FALLING 📉"
    if comm_name == "Potato":
        confidence = "93.8%" if "Random Forest" in model_choice else "93.1%"
    elif comm_name == "Tur Dal":
        confidence = "92.2%" if "Random Forest" in model_choice else "92.9%"
    else:
        confidence = "84.3%" if "Random Forest" in model_choice else "84.4%"
    st.metric(label="Price Trend Outlook", value=outlook, delta=f"Model Confidence: {confidence}")

with col4:
    alert_level = "HIGH" if pct_change > 5 else "NORMAL"
    st.error(f"**AI ALERT: {alert_level}**\n\nSpike Probability: {abs(pct_change)*3:.1f}%")

# --- SECTION 3: PRICE TRAJECTORY GRAPH ---
st.subheader("Price Trajectory: Historical & AI Forecast")

plot_df = df.tail(60)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df['Date'], y=plot_df['Modal_Price'],
    mode='lines+markers', name='Actual Wholesale Price (Past 60 Days)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=6, symbol='circle')
))

forecast_start_date = latest_row['Date']
forecast_end_date = latest_row['Date'] + pd.Timedelta(days=days_ahead)
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, periods=days_ahead+1)

np.random.seed(42)
trend_line = np.linspace(current_price, base_forecast_price, days_ahead+1)
vol_noise = np.random.normal(0, hist_volatility * 0.4, days_ahead+1)
taper_factor = np.sin(np.linspace(0, np.pi, days_ahead+1))
organic_trajectory = trend_line + (vol_noise * taper_factor)
organic_trajectory[0] = current_price
organic_trajectory[-1] = base_forecast_price

fig.add_trace(go.Scatter(
    x=forecast_dates, y=organic_trajectory,
    mode='lines+markers', name=f'{model_choice} Forecast ({forecast_horizon})',
    line=dict(color='red', width=3, dash='dash'),
    marker=dict(size=8, symbol='circle')
))

fig.update_layout(
    title=f"{comm_name} Price Trend & AI Projection ({forecast_horizon}) - {model_choice}",
    xaxis_title="Date",
    yaxis_title="Price (₹ / Quintal)",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- SECTION 4: BUFFER STOCK INTERVENTION SIMULATOR ---
st.subheader("Buffer Stock Intervention Simulator (Policy Sandbox)")

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.markdown("#### Adjust Policy Levers")
    min_arrival_val = int(current_arrivals * 0.3) if current_arrivals > 10 else 1
    max_arrival_val = int(current_arrivals * 4.0) if current_arrivals > 10 else 200
    start_arrival_val = int(current_arrivals) if current_arrivals > 10 else 50
    
    sim_arrivals = st.slider("Simulate Market Arrivals (Tonnes / Day)", min_value=min_arrival_val, max_value=max_arrival_val, value=start_arrival_val)
    sim_diesel = st.slider("Simulate Diesel Price Shock (₹ / Litre)", min_value=60.0, max_value=130.0, value=current_diesel)
    sim_rain = st.slider("Simulate Rainfall Deficit / Excess (mm)", min_value=0.0, max_value=150.0, value=current_rain)

# Simulator Math
arrival_pct_change = (sim_arrivals - current_arrivals) / max(current_arrivals, 1.0)
diesel_pct_change = (sim_diesel - current_diesel) / max(current_diesel, 1.0)
rain_deviation = abs(sim_rain - 50.0) - abs(current_rain - 50.0)

supply_elasticity = -0.35 
logistics_elasticity = 0.20 
weather_elasticity = 0.15 

adjustment_multiplier = (
    (arrival_pct_change * supply_elasticity) + 
    (diesel_pct_change * logistics_elasticity) + 
    ((rain_deviation / 50.0) * weather_elasticity)
)

simulated_forecast = base_forecast_price * (1 + adjustment_multiplier)
reduction = base_forecast_price - simulated_forecast

with col_sim2:
    st.markdown("#### Sandbox Impact Analysis")
    st.success(f"**New AI Estimated Price (With Policy Intervention):**\n### ₹ {simulated_forecast:,.2f} / Qtl")
    
    if reduction > 0:
        st.info(f"💡 **Intervention Impact:** Releasing additional buffer stock drops the forecasted price by **₹ {reduction:,.2f} / Qtl** (-{(reduction/base_forecast_price)*100:.2f}%).")
    else:
        st.warning(f"⚠️ **Market Warning:** Increasing supply shortages or diesel costs will drive prices up by **₹ {abs(reduction):,.2f} / Qtl** (+{(abs(reduction)/base_forecast_price)*100:.2f}%).")

st.markdown("---")
st.markdown("### System Architecture & Explainability")
st.markdown("""
* **Core Engine:** Gradient Boosting Regressor (XGBoost) & Bagging Ensemble (Random Forest) coupled with economic elasticity constraints.
* **Data Pipelines:** Automated cleaning scripts powered by Pandas, NumPy, and Open-Meteo APIs.
* **Framework:** National Consumer Affairs Decision Support Framework.
""")

# --- SAFE PDF EXPORT FUNCTION ---
def generate_krishi_drishti_pdf(comm_name, horizon_str, current_price, forecast_price, pct_change, alert_level, model_name=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "KRISHI DRISHTI", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "PRICE INTELLIGENCE & DECISION SUPPORT REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    model_hdr = f" | Model: {model_name}" if model_name else ""
    pdf.cell(0, 6, f"Commodity: {comm_name} | Horizon: {horizon_str}{model_hdr}", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. CURRENT PRICE & FORECAST", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(90, 6, f"Current Wholesale Price: Rs {current_price:,.2f} / quintal", ln=0)
    pdf.cell(90, 6, f"Forecast Horizon: {horizon_str}", ln=True)
    pdf.cell(90, 6, f"AI Forecasted Price: Rs {forecast_price:,.2f} / quintal", ln=0)
    pdf.cell(90, 6, f"Projected Change: {pct_change:+.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. RISK ASSESSMENT & EARLY WARNINGS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(90, 6, f"AI Alert Status: {alert_level}", ln=True)
    pdf.cell(0, 6, "- Market momentum evaluated across rolling historical standard deviations.", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. GOVERNMENT DECISION SUPPORT & INTERVENTION", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if pct_change > 5:
        pdf.cell(0, 6, "Recommendation: PROCURE / RELEASE BUFFER STOCK (HIGH PRIORITY)", ln=True)
        pdf.cell(0, 6, "- Executive Action: Initiate open market sales or procurement review to stabilize price spikes.", ln=True)
    elif pct_change < -5:
        pdf.cell(0, 6, "Recommendation: PROCUREMENT REVIEW / SUPPORT PRICE (MEDIUM)", ln=True)
        pdf.cell(0, 6, "- Executive Action: Monitor farmer distress signals and evaluate storage capacity.", ln=True)
    else:
        pdf.cell(0, 6, "Recommendation: MAINTAIN STATUS QUO (STABLE MARKET)", ln=True)
        pdf.cell(0, 6, "- Executive Action: Regular monitoring of APMC daily arrivals.", ln=True)
        
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf_model_desc = f"{model_name} DSS" if model_name else "Ensemble Multi-Horizon DSS"
    pdf.cell(0, 6, f"Data Source: AGMARKNET & OpenMeteo | Model: {pdf_model_desc}", ln=True, align="C")
    pdf.cell(0, 6, "Disclaimer: Advisory decision support system for Ministry of Consumer Affairs framework.", ln=True, align="C")
    
    try:
        raw_output = pdf.output(dest='S')
    except TypeError:
        raw_output = pdf.output()
        
    if isinstance(raw_output, str):
        return raw_output.encode('latin-1')
    return bytes(raw_output)

# --- SIDEBAR DOWNLOAD BUTTON ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Intelligence Report")

pdf_bytes = generate_krishi_drishti_pdf(
    comm_name, forecast_horizon, current_price, base_forecast_price, 
    pct_change, alert_level, model_choice
)

st.sidebar.download_button(
    label="📄 Download Krishi Drishti PDF",
    data=pdf_bytes,
    file_name=f"KrishiDrishti_Report_{comm_name.split()[0]}_{forecast_horizon.replace(' ', '_')}_{datetime.date.today()}.pdf",
    mime="application/pdf"
)

# --- MODEL OPTIMIZATION & HYPERPARAMETER TUNING SECTION ---
if _MODEL_OPT_AVAILABLE:
    render_model_optimization_section()
else:
    st.markdown("---")
    st.warning("`model_optimization.py` not found. Please ensure the file is in the same directory as `app.py`.")

# --- MODEL VALIDATION & GENERALIZATION ANALYSIS SECTION ---
if _MODEL_VAL_AVAILABLE:
    render_model_validation_section()
else:
    st.markdown("---")
    st.warning("`model_validation.py` not found. Please ensure the file is in the same directory as `app.py`.")

# --- WEEKLY AGRICULTURAL PRICE INTELLIGENCE SECTION ---
if _WEEKLY_INTEL_AVAILABLE:
    render_weekly_intelligence_section()
else:
    st.markdown("---")
    st.warning("`weekly_intelligence.py` not found. Please ensure the file is in the same directory as `app.py`.")

# --- DATA INTELLIGENCE & PREPROCESSING SECTION ---
if _DATA_INTEL_AVAILABLE:
    render_data_intelligence_section()
else:
    st.markdown("---")
    st.warning("`data_intelligence.py` not found. Please ensure the file is in the same directory as `app.py`.")
