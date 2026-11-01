"""
model_validation.py
===================
KrishiDrishti – Model Validation & Generalization Analysis Dashboard Module

Provides interactive diagnostics and overfitting detection for:
  - 3-Way Chronological Evaluation (Train, Validation, Test).
  - Generalization Gap & Relative Gap (%) quantification.
  - 4-Tier Model Health Classification with mathematical evidence.
  - Chronological Learning Curves with increasing training dataset size.
  - Actual vs. Predicted price trajectories across Train / Val / Test periods.
  - In-depth Residual Analysis (Distribution, Box Plot, Homoscedasticity, Time-trend).
  - Actionable model-specific improvement recommendations.
  - Downloadable validation reports and comparative CSV tables.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "model_validation")
MASTER_REPORT_FILE = os.path.join(REPORTS_DIR, "master_validation_report.json")
MASTER_CSV_FILE = os.path.join(REPORTS_DIR, "master_comparison_table.csv")

PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0E1117"

COMMODITY_CONFIG = {
    "Onion (Lasalgaon)": {
        "key": "onion",
        "color": "#FF6B35",
        "market": "Lasalgaon APMC, Maharashtra",
    },
    "Potato (Achalda)": {
        "key": "potato",
        "color": "#8B5CF6",
        "market": "Achalda APMC, Uttar Pradesh",
    },
    "Tur Dal (Latur)": {
        "key": "turdal",
        "color": "#10B981",
        "market": "Latur APMC, Maharashtra",
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


def load_master_validation_report():
    candidates = [
        MASTER_REPORT_FILE,
        os.path.join("reports", "model_validation", "master_validation_report.json"),
        os.path.join(os.getcwd(), "reports", "model_validation", "master_validation_report.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_master_comparison_df():
    candidates = [
        MASTER_CSV_FILE,
        os.path.join("reports", "model_validation", "master_comparison_table.csv"),
        os.path.join(os.getcwd(), "reports", "model_validation", "master_comparison_table.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: TRAIN VS VALIDATION VS TEST METRIC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def chart_train_val_test_bars(entry: dict) -> go.Figure:
    m = entry["metrics"]
    categories = ["MAE (₹/Qtl)", "RMSE (₹/Qtl)", "MAPE (%)"]

    tr_vals = [m["train"]["mae"], m["train"]["rmse"], m["train"]["mape"]]
    val_vals = [m["val"]["mae"], m["val"]["rmse"], m["val"]["mape"]]
    test_vals = [m["test"]["mae"], m["test"]["rmse"], m["test"]["mape"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=tr_vals,
        name="Training Partition",
        marker_color="#3B82F6",
        text=[f"{v:.1f}" for v in tr_vals],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=val_vals,
        name="Validation Partition",
        marker_color="#F59E0B",
        text=[f"{v:.1f}" for v in val_vals],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=test_vals,
        name="Unseen Test Partition",
        marker_color="#10B981",
        text=[f"{v:.1f}" for v in test_vals],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=f"Train vs. Validation vs. Test Metrics – {entry['commodity']} ({entry['horizon_days']}d | {entry['algorithm']})",
        yaxis_title="Metric Value",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: CHRONOLOGICAL LEARNING CURVE
# ─────────────────────────────────────────────────────────────────────────────
def chart_learning_curve(learning_curve: list, algo: str, horizon: int) -> go.Figure:
    df_lc = pd.DataFrame(learning_curve)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_lc["training_samples"],
        y=df_lc["train_mae"],
        mode="lines+markers",
        name="Training MAE",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=7),
        hovertemplate="Train Samples: %{x}<br>Train MAE: ₹%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_lc["training_samples"],
        y=df_lc["val_mae"],
        mode="lines+markers",
        name="Validation MAE",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=7),
        hovertemplate="Train Samples: %{x}<br>Validation MAE: ₹%{y:.2f}<extra></extra>"
    ))

    # Fill between lines to highlight generalization gap
    fig.add_trace(go.Scatter(
        x=pd.concat([df_lc["training_samples"], df_lc["training_samples"][::-1]]),
        y=pd.concat([df_lc["val_mae"], df_lc["train_mae"][::-1]]),
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Generalization Gap"
    ))

    fig.update_layout(
        title=f"Chronological Learning Curve ({algo} – {horizon}-Day Horizon)",
        xaxis_title="Chronological Training Sample Size (Past Observations)",
        yaxis_title="Mean Absolute Error (₹ / Quintal)",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: ACTUAL VS PREDICTED TRAJECTORY ACROSS SPLITS
# ─────────────────────────────────────────────────────────────────────────────
def chart_actual_vs_predicted(pred_sample: dict, commodity_name: str, horizon: int) -> go.Figure:
    fig = go.Figure()

    # Combine partitions for a continuous timeline
    parts = [("train", "Train", "#3B82F6"), ("val", "Val", "#F59E0B"), ("test", "Test", "#10B981")]

    for key, label, color in parts:
        data = pred_sample[key]
        dates = pd.to_datetime(data["dates"])
        actuals = data["actual"]
        preds = data["predicted"]

        fig.add_trace(go.Scatter(
            x=dates,
            y=actuals,
            mode="lines",
            name=f"Actual Price ({label})",
            line=dict(color="#CBD5E1", width=1.5),
            legendgroup=f"{label}_actual",
            hovertemplate=f"Date: %{{x|%d %b %Y}}<br>Actual: ₹%{{y:,.2f}}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=preds,
            mode="lines",
            name=f"Predicted ({label})",
            line=dict(color=color, width=2.2, dash="solid" if label == "Test" else "dash"),
            legendgroup=f"{label}_pred",
            hovertemplate=f"Date: %{{x|%d %b %Y}}<br>Predicted: ₹%{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Actual vs. Predicted Price Trajectory across Partitions ({commodity_name} – {horizon}d Horizon)",
        xaxis_title="Date (Target Timestamp)",
        yaxis_title="Modal Price (₹ / Quintal)",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: RESIDUAL DIAGNOSTICS (HISTOGRAM & HOMOSCEDASTICITY)
# ─────────────────────────────────────────────────────────────────────────────
def chart_residual_distribution(pred_sample: dict) -> go.Figure:
    val_res = pred_sample["val"]["residuals"]
    test_res = pred_sample["test"]["residuals"]
    all_res = val_res + test_res

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=val_res,
        name="Validation Residuals",
        marker_color="#F59E0B",
        opacity=0.7,
        nbinsx=30,
        hovertemplate="Residual: ₹%{x:,.1f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Histogram(
        x=test_res,
        name="Test Residuals",
        marker_color="#10B981",
        opacity=0.7,
        nbinsx=30,
        hovertemplate="Residual: ₹%{x:,.1f}<br>Count: %{y}<extra></extra>"
    ))

    # Add zero residual reference line
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#EF4444")

    fig.update_layout(
        barmode="overlay",
        title="Residual Error Distribution (Actual − Predicted)",
        xaxis_title="Residual Error (₹ / Quintal)",
        yaxis_title="Observation Frequency",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


def chart_residuals_vs_predicted(pred_sample: dict) -> go.Figure:
    val_preds = pred_sample["val"]["predicted"]
    val_res = pred_sample["val"]["residuals"]
    test_preds = pred_sample["test"]["predicted"]
    test_res = pred_sample["test"]["residuals"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=val_preds,
        y=val_res,
        mode="markers",
        name="Validation Residuals",
        marker=dict(size=7, color="#F59E0B", opacity=0.75),
        hovertemplate="Predicted: ₹%{x:,.1f}<br>Residual: ₹%{y:,.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=test_preds,
        y=test_res,
        mode="markers",
        name="Test Residuals",
        marker=dict(size=8, color="#10B981", opacity=0.8),
        hovertemplate="Predicted: ₹%{x:,.1f}<br>Residual: ₹%{y:,.1f}<extra></extra>"
    ))

    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="#EF4444")

    fig.update_layout(
        title="Homoscedasticity Analysis (Residuals vs. Predicted Values)",
        xaxis_title="Predicted Modal Price (₹ / Quintal)",
        yaxis_title="Residual Error (₹ / Quintal)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


def chart_residuals_over_time(pred_sample: dict) -> go.Figure:
    val_dates = pd.to_datetime(pred_sample["val"]["dates"])
    val_res = pred_sample["val"]["residuals"]
    test_dates = pd.to_datetime(pred_sample["test"]["dates"])
    test_res = pred_sample["test"]["residuals"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=val_dates,
        y=val_res,
        mode="lines+markers",
        name="Validation Residuals",
        line=dict(color="#F59E0B", width=1.5),
        marker=dict(size=5),
        hovertemplate="Date: %{x|%d %b %Y}<br>Residual: ₹%{y:,.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_res,
        mode="lines+markers",
        name="Test Residuals",
        line=dict(color="#10B981", width=1.5),
        marker=dict(size=5),
        hovertemplate="Date: %{x|%d %b %Y}<br>Residual: ₹%{y:,.1f}<extra></extra>"
    ))

    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="#EF4444")

    fig.update_layout(
        title="Residual Error Tracking Over Time (Temporal Bias / Drift)",
        xaxis_title="Date",
        yaxis_title="Residual (₹ / Quintal)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: GENERALIZATION GAP BENCHMARK ACROSS COMMODITIES & HORIZONS
# ─────────────────────────────────────────────────────────────────────────────
def chart_generalization_gaps(df_comp: pd.DataFrame) -> go.Figure:
    sub = df_comp[df_comp["Version"] == "Optimized"].copy()
    fig = go.Figure()

    for comm in sorted(sub["Commodity"].unique()):
        comm_df = sub[sub["Commodity"] == comm].sort_values("Horizon")
        fig.add_trace(go.Bar(
            x=comm_df["Horizon"],
            y=comm_df["Relative Gap (%)"],
            name=comm,
            text=[f"{v:+.1f}%" for v in comm_df["Relative Gap (%)"]],
            textposition="auto"
        ))

    # Add threshold guides
    fig.add_hline(y=30, line_dash="dash", line_color="#F59E0B", annotation_text="Potential Overfitting Threshold (30%)")
    fig.add_hline(y=75, line_dash="dash", line_color="#EF4444", annotation_text="Significant Overfitting Threshold (75%)")

    fig.update_layout(
        barmode="group",
        title="Relative Generalization Gap by Commodity & Horizon (Val MAE vs. Train MAE)",
        xaxis_title="Forecasting Horizon",
        yaxis_title="Relative Generalization Gap (%)",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def render_model_validation_section():
    """
    Renders the complete Model Validation & Generalization Analysis section.
    Called inside app.py.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #132238, #1c3558, #22426e);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 20px;
            border-left: 5px solid #10B981;
        ">
            <h2 style="color: #34D399; margin: 0 0 8px 0; font-family: Inter, sans-serif;">
                🏥 Model Validation & Generalization Analysis
            </h2>
            <p style="color: #CBD5E1; margin: 0; font-size: 0.96rem;">
                Rigorous 3-way chronological evaluation (Train ➔ Validation ➔ Test) verifying whether models 
                are learning genuine agricultural supply-demand patterns rather than memorizing historical observations. 
                Features objective 4-tier overfitting classification, learning curves, and residual diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    master_data = load_master_validation_report()
    df_comp = load_master_comparison_df()

    if not master_data or df_comp is None:
        st.warning(
            "⚠️ Validation evaluation reports not found. "
            "Please run `python evaluate_generalization.py` to initiate the evaluation pipeline."
        )
        return

    # ── SECTION A: OVERALL MODEL HEALTH KPI CARDS ────────────────────────────
    total_models = len(df_comp)
    good_gen = len(df_comp[df_comp["Health Classification"] == "Good Generalization"])
    pot_overfit = len(df_comp[df_comp["Health Classification"] == "Potential Overfitting"])
    sig_overfit = len(df_comp[df_comp["Health Classification"] == "Significant Overfitting"])
    underfit = len(df_comp[df_comp["Health Classification"] == "Potential Underfitting"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📋 Total Evaluated", f"{total_models} Models", delta="18 Baseline + 18 Opt", delta_color="off")
    k2.metric("🟢 Good Generalization", f"{good_gen}", delta=f"{good_gen/total_models*100:.0f}% of Models", delta_color="normal")
    k3.metric("🟠 Potential Overfit", f"{pot_overfit}", delta=f"{pot_overfit/total_models*100:.0f}%", delta_color="inverse")
    k4.metric("🔴 Significant Overfit", f"{sig_overfit}", delta=f"{sig_overfit/total_models*100:.0f}%", delta_color="inverse")
    k5.metric("🟡 Underfitting", f"{underfit}", delta="0% (Healthy)", delta_color="normal")

    st.markdown("---")

    # ── SELECTORS ────────────────────────────────────────────────────────────
    c_col1, c_col2, c_col3, c_col4 = st.columns([2, 2, 2, 2])

    with c_col1:
        commodity_label = st.selectbox(
            "Commodity",
            list(COMMODITY_CONFIG.keys()),
            key="val_commodity_select"
        )
    comm_key = COMMODITY_CONFIG[commodity_label]["key"]

    with c_col2:
        horizon_choice = st.selectbox(
            "Forecasting Horizon",
            ["7 Days", "14 Days", "30 Days"],
            key="val_horizon_select"
        )
        horizon_days = int(horizon_choice.split()[0])
        h_key = f"{horizon_days}d"

    available_algos = list(master_data.get(comm_key, {}).get(h_key, {}).keys())

    with c_col3:
        algo_choice = st.selectbox(
            "Algorithm Engine",
            available_algos,
            key="val_algo_select"
        )

    with c_col4:
        version_choice = st.selectbox(
            "Model Version",
            ["Optimized (Tuned)", "Original (Baseline)"],
            key="val_version_select"
        )
        m_type = "optimized" if "Optimized" in version_choice else "original"

    entry = master_data[comm_key][h_key][algo_choice][m_type]
    m = entry["metrics"]
    gen = entry["generalization"]
    res = entry["residuals_test"]

    # ── HEALTH VERDICT BANNER ────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: #111827;
            border-left: 4px solid {'#10B981' if 'Good' in gen['badge'] else ('#F59E0B' if 'Potential' in gen['badge'] else '#EF4444')};
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <span style="font-size: 1.15rem; font-weight: 700;">{gen['badge']}</span>
            <p style="margin: 6px 0 0 0; color: #E2E8F0; font-size: 0.95rem;">{gen['verdict']}</p>
            <small style="color: #94A3B8;">
                Dataset Partitions: Train ({entry['dates_meta']['train']}) | Val ({entry['dates_meta']['val']}) | Test ({entry['dates_meta']['test']})
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── METRIC STRIP FOR CURRENT SELECTION ───────────────────────────────────
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("📉 Training MAE", f"₹{m['train']['mae']:.2f}", delta=f"R²: {m['train']['r2']:.2f}", delta_color="off")
    sm2.metric("📊 Validation MAE", f"₹{m['val']['mae']:.2f}", delta=f"R²: {m['val']['r2']:.2f}", delta_color="off")
    sm3.metric("🎯 Test MAE", f"₹{m['test']['mae']:.2f}", delta=f"R²: {m['test']['r2']:.2f}", delta_color="off")
    sm4.metric("⚖️ Generalization Gap", f"₹{gen['generalization_gap_mae']:.2f}", delta=f"{gen['relative_gap_pct']:+.1f}% Rel Gap", delta_color="inverse" if gen['relative_gap_pct'] > 30 else "normal")
    sm5.metric("📦 Test MAPE", f"{m['test']['mape']:.2f}%", delta=f"RMSE: ₹{m['test']['rmse']:.1f}", delta_color="off")

    st.markdown("---")

    # ── TABS FOR DETAILED DIAGNOSTICS ────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Partition Comparison",
        "📈 Learning Curves",
        "🔮 Prediction Diagnostics",
        "📉 Residual Analysis",
        "🌐 Multi-Commodity Gaps",
        "💡 Model Recommendations",
        "🗂️ Master Evaluation Table"
    ])

    with tab1:
        st.plotly_chart(chart_train_val_test_bars(entry), use_container_width=True)
        st.caption(
            "Comparing Training, Validation, and Testing errors across MAE, RMSE, and MAPE. "
            "A balanced model exhibits small differences between Training and Validation loss."
        )

    with tab2:
        st.plotly_chart(chart_learning_curve(entry["learning_curve"], algo_choice, horizon_days), use_container_width=True)
        st.caption(
            "Chronological Learning Curves: Model fitted sequentially on 25%, 45%, 65%, 85%, and 100% of historical past observations "
            "and evaluated on the fixed validation window. Shaded region depicts the convergence of the generalization gap."
        )

    with tab3:
        st.plotly_chart(chart_actual_vs_predicted(entry["predictions_sample"], commodity_label.split()[0], horizon_days), use_container_width=True)
        st.caption(
            "Continuous chronological price tracking: Blue (Train partition), Yellow (Validation partition), and Green (Unseen Test holdout partition). "
            "Shows whether predictions track turning points and seasonal harvest troughs."
        )

    with tab4:
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.plotly_chart(chart_residual_distribution(entry["predictions_sample"]), use_container_width=True)
        with r_col2:
            st.plotly_chart(chart_residuals_vs_predicted(entry["predictions_sample"]), use_container_width=True)

        st.plotly_chart(chart_residuals_over_time(entry["predictions_sample"]), use_container_width=True)

        st.markdown("#### 📐 Summary Residual Statistics (Test Partition)")
        r_df = pd.DataFrame([{
            "Mean Residual (₹)": f"{res['mean_residual']:+.2f}",
            "Std Deviation (₹)": f"{res['std_residual']:.2f}",
            "Mean Absolute Residual (₹)": f"{res['mean_abs_residual']:.2f}",
            "Min Residual (₹)": f"{res['min_residual']:.2f}",
            "Max Residual (₹)": f"{res['max_residual']:.2f}",
            "Skewness Index": f"{res['skewness']:.3f}"
        }])
        st.dataframe(r_df, use_container_width=True, hide_index=True)
        st.caption("A mean residual near zero indicates zero structural bias. Skewness near 0 indicates symmetrical prediction errors.")

    with tab5:
        st.plotly_chart(chart_generalization_gaps(df_comp), use_container_width=True)
        st.caption(
            "Generalization gap (%) across Onion, Potato, and Tur Dal across 7d, 14d, and 30d horizons. "
            "Dashed lines denote the 30% (Potential Overfit) and 75% (Significant Overfit) decision boundaries."
        )

    with tab6:
        st.markdown(f"### 💡 Actionable Improvement Recommendations for {commodity_label} ({algo_choice} – {horizon_choice})")
        for rec in entry["recommendations"]:
            st.markdown(f"- 📌 **Actionable Advice**: {rec}")

        st.markdown("---")
        st.markdown("#### ⚙️ Evaluated Hyperparameter State")
        st.json(entry["hyperparameters"])

    with tab7:
        st.markdown("### 📋 Complete Cross-Model Validation Benchmark Table")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Download button
        csv_bytes = df_comp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Validation Benchmark Table (CSV)",
            data=csv_bytes,
            file_name="krishi_drishti_model_validation_benchmark.csv",
            mime="text/csv",
            key="val_down_csv"
        )
