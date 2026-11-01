"""
model_optimization.py
=====================
KrishiDrishti – Model Optimization & Hyperparameter Tuning Dashboard Module

Provides interactive visualizations and analytical comparisons for:
  - TimeSeriesSplit chronological cross-validation results.
  - Original vs. Optimized performance (MAE, RMSE, MAPE, R²).
  - Hyperparameter search landscape & trial distribution.
  - Hyperparameter sensitivity/importance analysis.
  - Cross-commodity & multi-horizon benchmarking.
  - Export of optimization metadata and trials CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "optimized")
SUMMARY_FILE = os.path.join(MODELS_DIR, "master_optimization_summary.json")

PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0E1117"

COMMODITY_CONFIG = {
    "Onion (Lasalgaon)": {
        "key": "onion",
        "color": "#FF6B35",
        "color_opt": "#10B981",
        "icon": "🧅",
        "market": "Lasalgaon APMC, Maharashtra",
        "test_period": "2024 Holdout Year",
    },
    "Potato (Achalda)": {
        "key": "potato",
        "color": "#8B5CF6",
        "color_opt": "#06B6D4",
        "icon": "🥔",
        "market": "Achalda APMC, Uttar Pradesh",
        "test_period": "2026 Holdout Test Set",
    },
    "Tur Dal (Latur)": {
        "key": "turdal",
        "color": "#F59E0B",
        "color_opt": "#10B981",
        "icon": "🫘",
        "market": "Latur APMC, Maharashtra",
        "test_period": "2024 Holdout Year",
    },
}


def _layout(**kwargs):
    base = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    base.update(kwargs)
    return base


def load_optimization_data():
    candidates = [
        SUMMARY_FILE,
        os.path.join("models", "optimized", "master_optimization_summary.json"),
        os.path.join(os.getcwd(), "models", "optimized", "master_optimization_summary.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_trials_csv(comm_key: str, algo: str, horizon: int):
    filename = f"search_results_{algo.lower()}_{horizon}d.csv"
    candidates = [
        os.path.join(MODELS_DIR, comm_key, filename),
        os.path.join("models", "optimized", comm_key, filename),
        os.path.join(os.getcwd(), "models", "optimized", comm_key, filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART A: MODEL PERFORMANCE COMPARISON (ORIGINAL VS OPTIMIZED)
# ─────────────────────────────────────────────────────────────────────────────
def chart_performance_comparison(entry: dict, cfg: dict) -> go.Figure:
    orig_test = entry["original"]["holdout_metrics"]
    opt_test = entry["optimized"]["holdout_metrics"]
    orig_cv = entry["original"]["cv_metrics"]
    opt_cv = entry["optimized"]["cv_metrics"]

    categories = [
        "Holdout Test MAE (₹)",
        "Holdout Test RMSE (₹)",
        "Holdout Test MAPE (%)",
        "CV Mean MAE (₹)",
        "CV Mean RMSE (₹)",
    ]

    orig_values = [
        orig_test["test_mae"],
        orig_test["test_rmse"],
        orig_test["test_mape"],
        orig_cv["cv_mae_mean"],
        orig_cv["cv_rmse_mean"],
    ]

    opt_values = [
        opt_test["test_mae"],
        opt_test["test_rmse"],
        opt_test["test_mape"],
        opt_cv["cv_mae_mean"],
        opt_cv["cv_rmse_mean"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=orig_values,
        name="Original Model (Manual Hyperparams)",
        marker_color="#EF4444",
        opacity=0.85,
        text=[f"{v:.1f}" for v in orig_values],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=opt_values,
        name="Optimized Model (Systematic Search)",
        marker_color="#10B981",
        opacity=0.9,
        text=[f"{v:.1f}" for v in opt_values],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title=f"Original vs. Optimized Evaluation Metrics – {entry['commodity']} ({entry['horizon_days']}-Day Horizon)",
        yaxis_title="Metric Value",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART B: HYPERPARAMETER SEARCH LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────
def chart_search_landscape(trials_df: pd.DataFrame, algo: str) -> go.Figure:
    fig = go.Figure()

    # Color scale based on MAE
    fig.add_trace(go.Scatter(
        x=trials_df.index + 1,
        y=trials_df["mean_test_mae"],
        mode="markers+lines",
        marker=dict(
            size=11,
            color=trials_df["mean_test_mae"],
            colorscale="Viridis_r",
            showscale=True,
            colorbar=dict(title="CV MAE (₹)"),
            line=dict(width=1, color="#FFFFFF")
        ),
        line=dict(color="rgba(150, 150, 150, 0.3)", dash="dot"),
        text=[f"Trial #{i+1}<br>CV MAE: ₹{r['mean_test_mae']:.2f}<br>Rank: {int(r['rank'])}"
              for i, r in trials_df.iterrows()],
        hoverinfo="text",
        name="Trial Configurations"
    ))

    # Highlight best trial
    best_row = trials_df.loc[trials_df["rank"] == 1].iloc[0]
    best_idx = trials_df[trials_df["rank"] == 1].index[0] + 1
    fig.add_trace(go.Scatter(
        x=[best_idx],
        y=[best_row["mean_test_mae"]],
        mode="markers",
        marker=dict(size=18, color="#F59E0B", symbol="star", line=dict(width=2, color="#FFFFFF")),
        name=f"Optimal Configuration (₹{best_row['mean_test_mae']:.2f})",
        hoverinfo="text",
        text=[f"⭐ Best Trial (#{best_idx})<br>Optimal CV MAE: ₹{best_row['mean_test_mae']:.2f}"]
    ))

    fig.update_layout(
        title=f"Hyperparameter Optimization Search Landscape ({algo})",
        xaxis_title="Search Iteration / Trial Sequence",
        yaxis_title="Cross-Validation MAE (₹ / Quintal)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART C: HYPERPARAMETER SENSITIVITY & IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def chart_param_sensitivity(trials_df: pd.DataFrame) -> go.Figure:
    param_cols = [c for c in trials_df.columns if c not in ["rank", "mean_test_mae", "std_test_score"]]
    numeric_params = []
    correlations = []

    for col in param_cols:
        # Check if numeric or can be encoded
        series = trials_df[col]
        if pd.api.types.is_numeric_dtype(series):
            std = series.std()
            if std is not None and not np.isnan(std) and std > 0:
                corr = series.corr(trials_df["mean_test_mae"])
                if not np.isnan(corr):
                    numeric_params.append(col)
                    correlations.append(abs(corr))

    if not numeric_params:
        # Fallback to variance across parameters
        return go.Figure()

    # Sort descending
    sorted_pairs = sorted(zip(numeric_params, correlations), key=lambda x: x[1], reverse=True)
    p_names = [p[0] for p in sorted_pairs]
    p_corrs = [p[1] for p in sorted_pairs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=p_corrs,
        y=p_names,
        orientation="h",
        marker=dict(
            color=p_corrs,
            colorscale="Teal",
            showscale=False
        ),
        text=[f"{c:.2f}" for c in p_corrs],
        textposition="auto"
    ))

    fig.update_layout(
        title="Hyperparameter Sensitivity Index (Correlation with Cross-Validation MAE)",
        xaxis_title="Normalized Sensitivity (|Correlation Coefficient| with Validation Loss)",
        yaxis=dict(autorange="reversed"),
        height=400,
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART D: COMMODITY & HORIZON CROSS-COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def chart_cross_commodity(master_data: dict, horizon: int) -> go.Figure:
    comm_names = []
    orig_maes = []
    opt_maes = []
    improvements = []

    h_key = f"{horizon}d"

    for c_key, c_cfg in COMMODITY_CONFIG.items():
        key = c_cfg["key"]
        if key in master_data and h_key in master_data[key]:
            # Use primary algorithm
            algos = list(master_data[key][h_key].keys())
            primary = algos[0] if algos else None
            if primary:
                entry = master_data[key][h_key][primary]
                comm_names.append(entry["commodity"].split()[0])
                orig_maes.append(entry["original"]["holdout_metrics"]["test_mae"])
                opt_maes.append(entry["optimized"]["holdout_metrics"]["test_mae"])
                improvements.append(entry["improvements"]["holdout_mae_pct"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comm_names,
        y=orig_maes,
        name="Original Holdout MAE",
        marker_color="#EF4444",
        text=[f"₹{v:.1f}" for v in orig_maes],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=comm_names,
        y=opt_maes,
        name="Optimized Holdout MAE",
        marker_color="#10B981",
        text=[f"₹{v:.1f} ({imp:+.1f}%)" for v, imp in zip(opt_maes, improvements)],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=f"Cross-Commodity Holdout Error Comparison ({horizon}-Day Forecasting Horizon)",
        yaxis_title="Holdout Test MAE (₹ / Quintal)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def render_model_optimization_section():
    """
    Renders the complete Model Optimization & Hyperparameter Tuning section.
    Called inside app.py.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0b1e36, #162a45, #1d3557);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 20px;
            border-left: 5px solid #06B6D4;
        ">
            <h2 style="color: #38BDF8; margin: 0 0 8px 0; font-family: Inter, sans-serif;">
                🎯 Model Optimization & Hyperparameter Tuning
            </h2>
            <p style="color: #CBD5E1; margin: 0; font-size: 0.96rem;">
                Scientific comparison between baseline models and systematic TimeSeriesSplit optimization 
                using RandomizedSearchCV and refined GridSearchCV. Time-series cross-validation ensures 
                strict chronological ordering with zero future-data leakage.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    master_data = load_optimization_data()

    if not master_data:
        st.warning(
            "⚠️ Optimization artifacts not found in `models/optimized/`. "
            "Please run `python tune_models.py` to initiate the hyperparameter search pipeline."
        )
        return

    # ── Selectors ────────────────────────────────────────────────────────────
    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 2])

    with sel_col1:
        commodity_label = st.selectbox(
            "Commodity",
            list(COMMODITY_CONFIG.keys()),
            key="opt_commodity_select"
        )

    cfg = COMMODITY_CONFIG[commodity_label]
    comm_key = cfg["key"]

    with sel_col2:
        horizon_choice = st.selectbox(
            "Forecasting Horizon",
            ["7 Days", "14 Days", "30 Days"],
            key="opt_horizon_select"
        )
        horizon_days = int(horizon_choice.split()[0])
        h_key = f"{horizon_days}d"

    # Available algorithms for this commodity
    available_algos = list(master_data.get(comm_key, {}).get(h_key, {}).keys())
    if not available_algos:
        st.error(f"No optimization data for {commodity_label} ({horizon_choice}).")
        return

    with sel_col3:
        algo_choice = st.selectbox(
            "Algorithm Engine",
            available_algos,
            key="opt_algo_select"
        )

    entry = master_data[comm_key][h_key][algo_choice]
    orig_test = entry["original"]["holdout_metrics"]
    opt_test = entry["optimized"]["holdout_metrics"]
    orig_cv = entry["original"]["cv_metrics"]
    opt_cv = entry["optimized"]["cv_metrics"]
    imps = entry["improvements"]

    # ── KPI Metrics Strip ────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)

    delta_color = "normal" if imps["holdout_mae_pct"] >= 0 else "inverse"
    m1.metric(
        "🎯 Holdout Test MAE",
        f"₹{opt_test['test_mae']:.2f}",
        delta=f"{imps['holdout_mae_pct']:+.2f}% vs Original",
        delta_color=delta_color
    )
    m2.metric(
        "📉 Baseline Test MAE",
        f"₹{orig_test['test_mae']:.2f}",
        delta="Manual Weights",
        delta_color="off"
    )
    m3.metric(
        "📊 Holdout Test RMSE",
        f"₹{opt_test['test_rmse']:.2f}",
        delta=f"{imps['holdout_rmse_pct']:+.2f}%",
        delta_color="normal"
    )
    m4.metric(
        "📈 Holdout MAPE",
        f"{opt_test['test_mape']:.2f}%",
        delta=f"{imps['holdout_mape_pct']:+.2f}%",
        delta_color="normal"
    )
    m5.metric(
        "🔄 TimeSeries CV MAE",
        f"₹{opt_cv['cv_mae_mean']:.2f}",
        delta=f"{imps['cv_mae_pct']:+.2f}% (3 Folds)",
        delta_color="normal"
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    tab_comp, tab_search, tab_sens, tab_cross, tab_params, tab_export = st.tabs([
        "📊 Model Comparison",
        "🔍 Search Landscape",
        "⚖️ Parameter Sensitivity",
        "🌐 Multi-Commodity Benchmarks",
        "📋 Parameters & Folds",
        "💾 Artifacts & Downloads"
    ])

    trials_df = load_trials_csv(comm_key, algo_choice, horizon_days)

    with tab_comp:
        st.plotly_chart(chart_performance_comparison(entry, cfg), use_container_width=True)
        st.caption(
            f"Evaluated on {cfg['test_period']} (unseen test holdout dataset) and chronological TimeSeriesSplit folds. "
            f"Features used: {entry['features_count']} indicators (past lags & moving averages only)."
        )

    with tab_search:
        if trials_df is not None and not trials_df.empty:
            st.plotly_chart(chart_search_landscape(trials_df, algo_choice), use_container_width=True)
            st.caption(
                "Each point represents an explored hyperparameter combination evaluated across 3 chronological "
                "TimeSeriesSplit validation splits. The golden star marks the lowest validation MAE configuration."
            )
        else:
            st.info("Trial search history CSV is unavailable for this selection.")

    with tab_sens:
        if trials_df is not None and not trials_df.empty:
            fig_sens = chart_param_sensitivity(trials_df)
            if len(fig_sens.data) > 0:
                st.plotly_chart(fig_sens, use_container_width=True)
                st.caption(
                    "Shows the relative sensitivity of validation loss (CV MAE) to changes in each hyperparameter. "
                    "Parameters with higher indices have the greatest impact on forecasting accuracy."
                )
            else:
                st.info("Parameter variance was focused on fixed fine-tuning dimensions.")
        else:
            st.info("Parameter sensitivity data unavailable.")

    with tab_cross:
        st.plotly_chart(chart_cross_commodity(master_data, horizon_days), use_container_width=True)
        st.caption(
            f"Cross-commodity comparison of original vs. optimized holdout MAE for {horizon_days}-day horizon. "
            "Evaluates generalization consistency across diverse market dynamics (Onion, Potato, Tur Dal)."
        )

    with tab_params:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### ⚙️ Baseline Hyperparameters")
            st.json(entry["original"]["hyperparameters"])
        with col_p2:
            st.markdown("#### 🚀 Optimized Hyperparameters")
            st.json(entry["optimized"]["hyperparameters"])

        st.markdown("#### 🔄 TimeSeriesSplit Fold Validation Details")
        opt_folds = entry["optimized"]["cv_metrics"]["folds"]
        orig_folds = entry["original"]["cv_metrics"]["folds"]
        folds_table = []
        for i in range(len(opt_folds)):
            folds_table.append({
                "Fold #": f"Split {i + 1}",
                "Training Samples": opt_folds[i]["train_size"],
                "Validation Samples": opt_folds[i]["val_size"],
                "Original Fold MAE (₹)": f"{orig_folds[i]['mae']:.2f}",
                "Optimized Fold MAE (₹)": f"{opt_folds[i]['mae']:.2f}",
                "Fold MAE Improvement (%)": f"{((orig_folds[i]['mae'] - opt_folds[i]['mae']) / orig_folds[i]['mae']) * 100:+.2f}%",
                "Optimized Fold RMSE (₹)": f"{opt_folds[i]['rmse']:.2f}",
                "Optimized Fold MAPE (%)": f"{opt_folds[i]['mape']:.2f}%",
            })
        st.dataframe(pd.DataFrame(folds_table), use_container_width=True, hide_index=True)

    with tab_export:
        st.markdown("#### 📦 Saved Model Artifacts & Reproducibility Files")
        st.markdown(f"- **Trained Model Artifact**: `{entry['model_artifact']}`")
        st.markdown(f"- **Search Trials CSV**: `{entry['trials_csv']}`")
        st.markdown(f"- **Timestamp**: `{entry['timestamp']}`")

        down_col1, down_col2 = st.columns(2)
        with down_col1:
            json_bytes = json.dumps(entry, indent=2).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download {comm_key.upper()} {horizon_days}d Summary (JSON)",
                data=json_bytes,
                file_name=f"optimization_{comm_key}_{algo_choice.lower()}_{horizon_days}d.json",
                mime="application/json",
                key="opt_down_json"
            )
        with down_col2:
            if trials_df is not None:
                csv_bytes = trials_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download Search Trials (CSV)",
                    data=csv_bytes,
                    file_name=f"search_trials_{comm_key}_{algo_choice.lower()}_{horizon_days}d.csv",
                    mime="text/csv",
                    key="opt_down_csv"
                )
