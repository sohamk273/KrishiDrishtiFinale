# 🌾 National Agri-Price Intelligence & Decision Support System

## 📌 Project Overview
The **National Agri-Price Intelligence & Decision Support System (DSS)** is an advanced AI-enabled predictive forecasting and policy simulation framework developed under the Consumer Affairs governance model. Designed for agricultural policymakers, procurement managers, and market analysts, the system addresses critical food inflation challenges by providing multi-horizon price intelligence, early volatility warnings, and interactive policy intervention sandboxes for agricultural staples including pulses (Tur Dal), onions, and potatoes.

---

## 🚀 Key Features & Capabilities

* **Multi-Horizon Price Forecasting:** Leverages robust Gradient Boosting Regressor models to generate precise wholesale price projections across 7-day, 14-day, and 30-day forecasting horizons.
* **Early Warning & Spike Probability:** Automatically evaluates market momentum and rolling historical standard deviations to flag abnormal volatility risks and trigger high-inflation early warnings.
* **Explainable Macro Drivers:** Integrates key economic and environmental drivers—such as wholesale market daily arrivals, diesel transportation costs, and weather shocks (rainfall/temperature)—ensuring complete transparency in price predictions.
* **Interactive Price Trajectory Visualization:** Utilizes dynamic Plotly charts that display historical actuals alongside organic daily forecast paths featuring realistic market fluctuations.
* **Policy Intervention Sandbox (Buffer Stock Simulator):** Empowers government stakeholders to simulate real-world policy levers (adjusting buffer stock release volumes, fuel price shocks, and weather stress) to evaluate their direct impact on mitigating food price inflation.
* **Executive PDF Intelligence Reports:** Generates and exports downloadable professional decision support reports (*Krishi Drishti*) complete with metric breakdowns, risk assessments, and executive recommendations.

---

## 🛠️ System Architecture & Technology Stack
* **Frontend / Dashboard:** Streamlit UI framework with custom sidebar filters and interactive Plotly data visualizations.
* **Machine Learning Core:** Scikit-Learn & XGBoost multi-horizon regression models trained on APMC market data (AGMARKNET) and meteorological feeds (Open-Meteo).
* **Data Pipelines:** Automated preprocessing pipelines utilizing Pandas and NumPy for lag feature engineering, rolling means, and historical volatility tracking.
* **Reporting Engine:** FPDF-powered automated PDF report generation framework.

---

## 💻 Setup & Installation Guide for Beginners

Follow these simple steps to set up and run the National Agri-Price Intelligence & Decision Support System on your local machine:

### Prerequisites
Make sure you have **Python 3.9+** installed on your system.

### Step 1: Clone or Download the Repository
Navigate to your project directory in your terminal or command prompt where all your project files (`app.py`, model `.pkl` files, and datasets) are stored.

### Step 2: Install Required Dependencies
Run the following command in your terminal to install the necessary Python libraries:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly fpdf2
```

### Step 3: Verify Your Files
Ensure your project folder contains the following files in the same directory:
* `app.py` (The main Streamlit application script)
* Model files (`potato_model.pkl`, `potato_model_14d.pkl`, `potato_model_30d.pkl`, `xgboost_model.pkl`, `onion_model_14d.pkl`, `onion_model_30d.pkl`, `turdal_model.pkl`, `turdal_model_14d.pkl`, `turdal_model_30d.pkl`)
* Dataset files (`master_dataset.csv`, `master_dataset_achalda.csv`, `master_dataset_latur.csv`)

### Step 4: Run the Streamlit Application
Execute the following command in your terminal:
```bash
streamlit run app.py
```

### Step 5: Access the Dashboard
Once the app starts, Streamlit will automatically open a local web page in your browser (usually at `http://localhost:8501`). If it doesn't open automatically, copy and paste that URL into your browser address bar.

---

*Author: Aryan Bhuimbar*
