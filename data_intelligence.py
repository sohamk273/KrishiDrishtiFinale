"""
data_intelligence.py
====================
KrishiDrishti – Data Intelligence & Preprocessing Dashboard Module

Academically demonstrates data quality, EDA, outlier analysis,
feature engineering and preprocessing methodology for:
  - Onion  (Lasalgaon) : master_dataset.csv
  - Potato (Achalda)   : master_dataset_achalda.csv
  - Tur Dal (Latur)    : master_dataset_latur.csv

Read-only access. Does NOT modify any existing datasets or models.
Author  : KrishiDrishti Team
Version : 1.0.0
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COMMODITY_CONFIG = {
    "Onion (Lasalgaon)": {
        "file": "master_dataset.csv",
        "color": "#FF6B35",
        "light_color": "rgba(255,107,53,0.15)",
        "market": "Lasalgaon APMC, Maharashtra",
        "unit": "Rs/Quintal",
        "icon": "🧅",
        "description": "Largest onion trading hub in Asia; price benchmark for national markets.",
    },
    "Potato (Achalda)": {
        "file": "master_dataset_achalda.csv",
        "color": "#8B5CF6",
        "light_color": "rgba(139,92,246,0.15)",
        "market": "Achalda APMC, Uttar Pradesh",
        "unit": "Rs/Quintal",
        "icon": "🥔",
        "description": "Key UP potato market; live 2026 forecasting dataset.",
    },
    "Tur Dal (Latur)": {
        "file": "master_dataset_latur.csv",
        "color": "#10B981",
        "light_color": "rgba(16,185,129,0.15)",
        "market": "Latur APMC, Maharashtra",
        "unit": "Rs/Quintal",
        "icon": "🫘",
        "description": "Primary pigeon pea / tur dal wholesale market in Marathwada region.",
    },
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0E1117"


def _make_layout(**kwargs):
    """Return common plotly layout kwargs."""
    base = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif"),
    )
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_commodity_data(file_path: str) -> pd.DataFrame:
    """Load and minimally type-cast a cleaned commodity CSV."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the exact feature engineering from training scripts.
    Returns a NEW DataFrame; originals are not modified.
    """
    out = df.copy()
    # Lag features – matches all three *_train_extended.py scripts
    for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
        out[f"Price_Lag_{lag}"] = out["Modal_Price"].shift(lag)
    # Rolling statistics
    out["Rolling_Mean_3d"]  = out["Modal_Price"].shift(1).rolling(window=3).mean()
    out["Rolling_Mean_7d"]  = out["Modal_Price"].shift(1).rolling(window=7).mean()
    out["Rolling_Mean_14d"] = out["Modal_Price"].shift(1).rolling(window=14).mean()
    out["Rolling_Mean_30d"] = out["Modal_Price"].shift(1).rolling(window=30).mean()
    out["Rolling_Std_7d"]   = out["Modal_Price"].shift(1).rolling(window=7).std()
    # Calendar features
    out["Month"]     = out["Date"].dt.month
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["DayOfYear"] = out["Date"].dt.dayofyear
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – DATA QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def render_data_quality(df: pd.DataFrame, cfg: dict):
    st.markdown("### 📋 Dataset Quality Metrics")
    color      = cfg["color"]
    num_cols   = df.select_dtypes(include="number").columns.tolist()
    cat_cols   = [c for c in df.select_dtypes(exclude="number").columns
                  if c != "Date"]
    miss       = df.isnull().sum()
    total_miss = int(miss.sum())
    mem_kb     = df.memory_usage(deep=True).sum() / 1024
    completeness = 100 - (total_miss / (len(df) * df.shape[1]) * 100)

    # ── Row 1 ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Records",   f"{len(df):,}")
    c2.metric("📊 Total Columns",   f"{df.shape[1]}")
    c3.metric("🔢 Numerical Cols",  f"{len(num_cols)}")
    c4.metric("🗂️ Categorical Cols", f"{len(cat_cols)}")

    # ── Row 2 ────────────────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    date_min = df["Date"].min().strftime("%d %b %Y")
    date_max = df["Date"].max().strftime("%d %b %Y")
    c5.metric("📅 Date Range",      f"{date_min} → {date_max}")
    c6.metric("⚠️ Total Missing",   f"{total_miss}")
    c7.metric("🔁 Duplicates",      f"{int(df.duplicated().sum())}")
    c8.metric("💾 Memory Usage",    f"{mem_kb:.1f} KB")

    # ── Row 3 ────────────────────────────────────────────────────────────────
    c9, c10 = st.columns(2)
    c9.metric("📆 Unique Dates",   f"{df['Date'].nunique():,}")
    c10.metric("✅ Data Completeness", f"{completeness:.4f}%")

    # ── Column-wise table ────────────────────────────────────────────────────
    st.markdown("#### Column-wise Missing Value Summary")
    mv_df = pd.DataFrame({
        "Column":         df.columns.tolist(),
        "Data Type":      [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": [int(df[c].notna().sum()) for c in df.columns],
        "Missing Count":  [int(miss[c]) for c in df.columns],
        "Missing (%)":    [round(miss[c] / len(df) * 100, 4) for c in df.columns],
        "Unique Values":  [int(df[c].nunique()) for c in df.columns],
    })
    st.dataframe(mv_df, use_container_width=True, hide_index=True)

    if total_miss == 0:
        st.success(
            "✅ **All columns are 100% complete.** "
            "Zero missing values detected in the cleaned dataset."
        )
    else:
        st.warning(
            f"⚠️ {total_miss} missing value(s) detected. "
            "These are handled via forward-fill in the app's data-loading step."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – MISSING VALUE VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_missing_value_viz(df: pd.DataFrame, cfg: dict):
    st.markdown("### 🔍 Missing Value Visualizations")
    color    = cfg["color"]
    cols     = df.columns.tolist()
    miss_sum = df.isnull().sum()
    miss_pct = (miss_sum / len(df) * 100).round(4)
    completeness = 100 - miss_pct

    tab1, tab2, tab3, tab4 = st.tabs([
        "🟥 Missing Heatmap",
        "📊 Missing Count",
        "📈 Missing Percentage",
        "✅ Data Completeness",
    ])

    # ── Tab 1: Heatmap (200-row sample) ─────────────────────────────────────
    with tab1:
        sample = df.sample(min(200, len(df)), random_state=42).sort_index()
        z_mat  = sample.isnull().values.astype(int).T
        fig = go.Figure(go.Heatmap(
            z=z_mat,
            x=[str(i) for i in range(z_mat.shape[1])],
            y=cols,
            colorscale=[[0, "#1A2942"], [1, "#FF4136"]],
            showscale=True,
            colorbar=dict(
                title="Status",
                tickvals=[0, 1],
                ticktext=["Present", "Missing"],
            ),
        ))
        fig.update_layout(
            title="Missing Value Heatmap (200-row sample)",
            height=340,
            xaxis_title="Row Index (sampled)",
            yaxis_title="Column",
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)
        if miss_sum.sum() == 0:
            st.success("Heatmap shows all green (no missing values). Dataset is fully complete.")

    # ── Tab 2: Count bar ─────────────────────────────────────────────────────
    with tab2:
        bar_colors = [color if v > 0 else "#22C55E" for v in miss_sum.values]
        fig = go.Figure(go.Bar(
            x=cols,
            y=miss_sum.values,
            marker_color=bar_colors,
            text=miss_sum.values,
            textposition="outside",
        ))
        fig.update_layout(
            title="Missing Value Count per Column",
            height=380,
            xaxis_title="Column",
            yaxis_title="Missing Count",
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Percentage ────────────────────────────────────────────────────
    with tab3:
        fig = go.Figure(go.Bar(
            x=cols,
            y=miss_pct.values,
            marker_color=[color if v > 0 else "#22C55E" for v in miss_pct.values],
            text=[f"{v:.4f}%" for v in miss_pct.values],
            textposition="outside",
        ))
        fig.add_hline(
            y=5, line_dash="dash", line_color="orange",
            annotation_text="5% threshold",
            annotation_position="top left",
        )
        ymax = max(float(miss_pct.max()) + 5, 15)
        fig.update_layout(
            title="Missing Value Percentage per Column",
            height=380,
            xaxis_title="Column",
            yaxis_title="Missing %",
            yaxis_range=[0, ymax],
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Completeness ──────────────────────────────────────────────────
    with tab4:
        fig = go.Figure(go.Bar(
            x=cols,
            y=completeness.values,
            marker_color=["#22C55E" if v == 100 else color for v in completeness.values],
            text=[f"{v:.2f}%" for v in completeness.values],
            textposition="inside",
        ))
        fig.add_hline(
            y=95, line_dash="dash", line_color="orange",
            annotation_text="95% completeness threshold",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title="Column-wise Data Completeness (%)",
            height=380,
            xaxis_title="Column",
            yaxis_title="Completeness %",
            yaxis_range=[0, 105],
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def render_eda(df: pd.DataFrame, cfg: dict, name: str):
    st.markdown("### 📊 Exploratory Data Analysis")
    color = cfg["color"]
    light = cfg["light_color"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Distribution",
        "📦 Box Plots",
        "💰 Price Comparison",
        "🚛 Arrivals",
        "📅 Historical Trends",
        "🌡️ Correlation & Seasonal",
    ])

    price_cols = [c for c in ["Min_Price", "Modal_Price", "Max_Price"] if c in df.columns]

    # ── Tab 1: Histograms ────────────────────────────────────────────────────
    with tab1:
        fig = make_subplots(
            rows=1, cols=len(price_cols),
            subplot_titles=[c.replace("_", " ") for c in price_cols],
        )
        palette = ["#10B981", color, "#EF4444"]
        for i, (col, clr) in enumerate(zip(price_cols, palette), 1):
            fig.add_trace(go.Histogram(
                x=df[col], nbinsx=50, name=col,
                marker_color=clr, opacity=0.8,
            ), row=1, col=i)
        fig.update_layout(
            title=f"{name} – Price Distribution Histograms",
            height=400,
            showlegend=False,
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[price_cols].describe().round(2), use_container_width=True)

    # ── Tab 2: Box plots ─────────────────────────────────────────────────────
    with tab2:
        fig = go.Figure()
        palette = ["#10B981", color, "#EF4444"]
        for col, clr in zip(price_cols, palette):
            fig.add_trace(go.Box(
                y=df[col], name=col.replace("_", " "),
                marker_color=clr, boxpoints="outliers",
            ))
        fig.update_layout(
            title=f"{name} – Price Distribution Box Plots",
            height=460,
            yaxis_title="Rs / Quintal",
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Outlier dots beyond whiskers represent genuine price-spike events. "
            "They are retained in the model training data as valid market signals."
        )

    # ── Tab 3: Min / Modal / Max comparison ──────────────────────────────────
    with tab3:
        plot_df = df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["Min_Price"],
            mode="lines", name="Min Price",
            line=dict(color="#10B981", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["Modal_Price"],
            mode="lines+markers", name="Modal Price",
            line=dict(color=color, width=2.5),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["Max_Price"],
            mode="lines", name="Max Price",
            line=dict(color="#EF4444", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            title=f"{name} – Min / Modal / Max Price Comparison (Last 90 Days)",
            height=420,
            xaxis_title="Date",
            yaxis_title="Rs / Quintal",
            hovermode="x unified",
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Arrivals ───────────────────────────────────────────────────────
    with tab4:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Arrival Qty Distribution", "Daily Arrivals Over Time"],
        )
        fig.add_trace(go.Histogram(
            x=df["Arrivals_Tonnes"], nbinsx=60,
            marker_color="#8B5CF6", opacity=0.85, name="Arrivals",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Arrivals_Tonnes"],
            mode="lines", name="Arrivals (Tonnes)",
            line=dict(color="#8B5CF6", width=1.5),
        ), row=1, col=2)
        fig.update_layout(
            title=f"{name} – Market Arrivals Analysis",
            height=400,
            showlegend=False,
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        arr_stats = df["Arrivals_Tonnes"].describe().round(2)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Arrivals",   f"{arr_stats['mean']:.1f} T")
        c2.metric("Median Arrivals", f"{arr_stats['50%']:.1f} T")
        c3.metric("Max Arrivals",    f"{arr_stats['max']:.1f} T")
        c4.metric("Std Dev",         f"{arr_stats['std']:.1f} T")

    # ── Tab 5: Historical Trends ──────────────────────────────────────────────
    with tab5:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Modal_Price"],
            mode="lines", name="Modal Price",
            line=dict(color=color, width=1.8),
            fill="tozeroy", fillcolor=light,
        ))
        roll30 = df["Modal_Price"].rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=roll30,
            mode="lines", name="30-Day Rolling Mean",
            line=dict(color="#FBBF24", width=2, dash="dash"),
        ))
        fig.update_layout(
            title=f"{name} – Complete Historical Price Trend",
            height=450,
            xaxis_title="Date",
            yaxis_title="Rs / Quintal",
            hovermode="x unified",
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monthly price trends by year
        df_m = df.copy()
        df_m["Month_num"] = df_m["Date"].dt.month
        df_m["Year"]      = df_m["Date"].dt.year
        monthly = df_m.groupby(["Year", "Month_num"])["Modal_Price"].mean().reset_index()
        fig2 = go.Figure()
        year_colors = ["#60A5FA","#34D399","#F59E0B","#F87171","#A78BFA","#FB923C"]
        for idx, yr in enumerate(sorted(monthly["Year"].unique())):
            sub = monthly[monthly["Year"] == yr]
            fig2.add_trace(go.Scatter(
                x=[MONTH_NAMES[m-1] for m in sub["Month_num"]],
                y=sub["Modal_Price"],
                mode="lines+markers",
                name=str(yr),
                line=dict(width=2, color=year_colors[idx % len(year_colors)]),
                marker=dict(size=7),
            ))
        fig2.update_layout(
            title=f"{name} – Monthly Price Trends by Year",
            height=420,
            xaxis_title="Month",
            yaxis_title="Avg Rs / Quintal",
            **_make_layout()
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 6: Correlation + Seasonal ────────────────────────────────────────
    with tab6:
        # Correlation heatmap
        num_df = df.select_dtypes(include="number")
        corr   = num_df.corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        ))
        fig.update_layout(
            title=f"{name} – Numerical Feature Correlation Heatmap",
            height=460,
            **_make_layout()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Seasonal pattern
        df_s = df.copy()
        df_s["MonthNum"] = df_s["Date"].dt.month
        seasonal = df_s.groupby("MonthNum")["Modal_Price"].agg(["mean", "std"]).reset_index()
        seasonal["MonthName"] = [MONTH_NAMES[m-1] for m in seasonal["MonthNum"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=seasonal["MonthName"],
            y=seasonal["mean"] + seasonal["std"],
            mode="lines", line=dict(width=0),
            showlegend=False,
            fillcolor=light,
            fill="tozeroy",
            name="Upper Band",
        ))
        fig3.add_trace(go.Scatter(
            x=seasonal["MonthName"],
            y=seasonal["mean"] - seasonal["std"],
            mode="lines", line=dict(width=0),
            showlegend=False,
            fillcolor=light,
            fill="tonexty",
            name="Lower Band",
        ))
        fig3.add_trace(go.Scatter(
            x=seasonal["MonthName"],
            y=seasonal["mean"],
            mode="lines+markers",
            name="Avg Modal Price",
            line=dict(color=color, width=3),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig3.update_layout(
            title=f"{name} – Seasonal Price Patterns (Avg +/- Std by Month)",
            height=420,
            xaxis_title="Month",
            yaxis_title="Rs / Quintal",
            **_make_layout()
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def _iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _zscore_mask(series: pd.Series, threshold: float = 3.0):
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold


def render_outlier_analysis(df: pd.DataFrame, cfg: dict, name: str):
    st.markdown("### 🎯 Outlier Analysis")
    color = cfg["color"]

    # ── Metrics for Modal_Price and Arrivals ─────────────────────────────────
    c1, c2 = st.columns(2)
    for col_name, container in [("Modal_Price", c1), ("Arrivals_Tonnes", c2)]:
        if col_name not in df.columns:
            continue
        s        = df[col_name].dropna()
        lo, hi   = _iqr_bounds(s)
        iqr_mask = (s < lo) | (s > hi)
        z_mask   = _zscore_mask(s)
        n_iqr    = int(iqr_mask.sum())
        n_z      = int(z_mask.sum())
        label    = "Rs" if "Price" in col_name else "T"
        with container:
            st.markdown(f"#### {col_name.replace('_', ' ')}")
            m1, m2 = st.columns(2)
            m1.metric("IQR Outliers",   f"{n_iqr}  ({n_iqr/len(s)*100:.1f}%)")
            m2.metric("Z-score > 3 SD", f"{n_z}  ({n_z/len(s)*100:.1f}%)")
            m3, m4 = st.columns(2)
            m3.metric("IQR Lower Fence", f"{label} {lo:,.1f}")
            m4.metric("IQR Upper Fence", f"{label} {hi:,.1f}")

    # ── Box plot with fences ──────────────────────────────────────────────────
    lo, hi = _iqr_bounds(df["Modal_Price"].dropna())
    fig = go.Figure(go.Box(
        y=df["Modal_Price"],
        name="Modal Price",
        marker_color=color,
        boxpoints="outliers",
    ))
    fig.add_hline(y=hi, line_dash="dash", line_color="orange",
                  annotation_text=f"IQR Upper Fence: Rs {hi:,.0f}",
                  annotation_position="top right")
    fig.add_hline(y=lo, line_dash="dash", line_color="cyan",
                  annotation_text=f"IQR Lower Fence: Rs {lo:,.0f}",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"{name} Modal Price – IQR Outlier Detection",
        height=450,
        yaxis_title="Rs / Quintal",
        **_make_layout()
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Outlier timeline ──────────────────────────────────────────────────────
    prices   = df["Modal_Price"].bfill()
    lo2, hi2 = _iqr_bounds(prices)
    iqr_flag = (prices < lo2) | (prices > hi2)
    df_flag  = df.copy()
    df_flag["IQR_Outlier"] = iqr_flag.values

    outlier_rows = df_flag[df_flag["IQR_Outlier"]]
    normal_rows  = df_flag[~df_flag["IQR_Outlier"]]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Date"], y=df["Modal_Price"],
        mode="lines", name="Modal Price",
        line=dict(color=color, width=1.5),
    ))
    if len(outlier_rows) > 0:
        fig2.add_trace(go.Scatter(
            x=outlier_rows["Date"],
            y=outlier_rows["Modal_Price"],
            mode="markers",
            name="Potential Anomaly (IQR)",
            marker=dict(color="orange", size=10, symbol="triangle-up",
                        line=dict(color="white", width=1.5)),
        ))
    fig2.update_layout(
        title=f"{name} – Price Anomalies Timeline (IQR Method)",
        height=420,
        xaxis_title="Date",
        yaxis_title="Rs / Quintal",
        hovermode="x unified",
        **_make_layout()
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "**Agricultural Price Spike Policy**: Flagged observations represent potential market "
        "events (harvest shocks, supply disruptions, policy interventions). "
        "They are **retained** in the training dataset as genuine signals and classified as "
        "anomalies requiring agronomic investigation — NOT automatically removed."
    )

    if len(outlier_rows) > 0:
        with st.expander(f"View {len(outlier_rows)} flagged outlier records (IQR method)"):
            st.dataframe(
                outlier_rows[["Date", "Modal_Price", "Arrivals_Tonnes"]].reset_index(drop=True),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FEATURE ENGINEERING VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def render_feature_engineering(df_feat: pd.DataFrame, cfg: dict, name: str):
    st.markdown("### ⚙️ Feature Engineering Visualization")
    color = cfg["color"]
    clean  = df_feat.dropna().copy()
    recent = clean.tail(90)

    st.markdown(
        "The features below are engineered from raw `Modal_Price` using temporal lags "
        "and rolling statistics — exactly replicating what the trained forecasting models "
        "use at inference time (see `onion_train_extended.py`, `potato_train_extended.py`, "
        "`tur_trained_extended.py`)."
    )

    # ── Lag features ──────────────────────────────────────────────────────────
    st.markdown("#### Price Lag Features vs. Modal Price")
    lag_cols = [f for f in ["Price_Lag_1","Price_Lag_3","Price_Lag_7","Price_Lag_14","Price_Lag_21"]
                if f in recent.columns]
    lag_colors = ["#60A5FA","#34D399","#A78BFA","#F59E0B","#F87171"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Modal_Price"],
        mode="lines", name="Modal Price (t)",
        line=dict(color=color, width=3),
    ))
    for lc, clr in zip(lag_cols, lag_colors):
        days = lc.split("_")[-1]
        fig.add_trace(go.Scatter(
            x=recent["Date"], y=recent[lc],
            mode="lines", name=f"Lag {days}",
            line=dict(color=clr, width=1.5, dash="dot"),
            opacity=0.85,
        ))
    fig.update_layout(
        title=f"{name} – Modal Price and Lag Features (Last 90 Days)",
        height=430,
        xaxis_title="Date",
        yaxis_title="Rs / Quintal",
        hovermode="x unified",
        **_make_layout()
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling means ─────────────────────────────────────────────────────────
    st.markdown("#### Rolling Mean Features vs. Modal Price")
    roll_cols   = [f for f in ["Rolling_Mean_3d","Rolling_Mean_7d",
                                "Rolling_Mean_14d","Rolling_Mean_30d"]
                   if f in recent.columns]
    roll_colors = ["#60A5FA","#34D399","#A78BFA","#FBBF24"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Modal_Price"],
        mode="lines", name="Modal Price",
        line=dict(color=color, width=2.5),
    ))
    for rc, clr in zip(roll_cols, roll_colors):
        label = rc.replace("Rolling_Mean_","").replace("d","-day Mean")
        fig2.add_trace(go.Scatter(
            x=recent["Date"], y=recent[rc],
            mode="lines", name=label,
            line=dict(color=clr, width=2, dash="dash"),
        ))
    fig2.update_layout(
        title=f"{name} – Rolling Mean Features (Last 90 Days)",
        height=430,
        xaxis_title="Date",
        yaxis_title="Rs / Quintal",
        hovermode="x unified",
        **_make_layout()
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Rolling Std (volatility) ──────────────────────────────────────────────
    if "Rolling_Std_7d" in recent.columns:
        st.markdown("#### 7-Day Rolling Volatility (Standard Deviation)")
        fig3 = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Modal Price", "7-Day Rolling Std Dev (Price Volatility)"],
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        fig3.add_trace(go.Scatter(
            x=recent["Date"], y=recent["Modal_Price"],
            mode="lines", name="Modal Price",
            line=dict(color=color, width=2),
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=recent["Date"], y=recent["Rolling_Std_7d"],
            mode="lines", name="Rolling Std 7d",
            line=dict(color="#EF4444", width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        ), row=2, col=1)
        fig3.update_layout(
            title=f"{name} – Price Volatility (7-Day Rolling Std Dev)",
            height=500,
            hovermode="x unified",
            **_make_layout()
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Calendar features ─────────────────────────────────────────────────────
    st.markdown("#### Calendar Feature Influence on Modal Price")
    month_avg = clean.groupby("Month")["Modal_Price"].mean()
    dow_avg   = clean.groupby("DayOfWeek")["Modal_Price"].mean()
    doy_samp  = clean[["DayOfYear","Modal_Price"]].sample(
        min(500, len(clean)), random_state=42)

    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Avg Price by Month", "Avg Price by Day of Week", "Price vs Day of Year"],
    )
    fig4.add_trace(go.Bar(
        x=[MONTH_NAMES[m-1] for m in month_avg.index],
        y=month_avg.values,
        marker_color=color, name="By Month",
    ), row=1, col=1)
    fig4.add_trace(go.Bar(
        x=[DOW_NAMES[d] for d in dow_avg.index],
        y=dow_avg.values,
        marker_color="#8B5CF6", name="By DoW",
    ), row=1, col=2)
    fig4.add_trace(go.Scatter(
        x=doy_samp["DayOfYear"], y=doy_samp["Modal_Price"],
        mode="markers",
        marker=dict(color=color, size=4, opacity=0.6),
        name="Price vs DoY",
    ), row=1, col=3)
    fig4.update_layout(
        title=f"{name} – Calendar Features and Modal Price",
        height=400,
        showlegend=False,
        **_make_layout()
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Feature summary table ────────────────────────────────────────────────
    st.markdown("#### Engineered Feature Summary Table")
    RAW_COLS = {"Min_Price","Max_Price","Modal_Price","Arrivals_Tonnes",
                "Temperature_C","Rainfall_mm","Diesel_Price_Rs","Date"}
    rows = []
    for feat in df_feat.columns:
        if feat in RAW_COLS:
            ftype = "Raw Input"
        elif "Lag" in feat:
            ftype = "Temporal Lag"
        elif "Rolling_Mean" in feat:
            ftype = "Rolling Mean"
        elif "Rolling_Std" in feat:
            ftype = "Rolling Volatility"
        elif feat in ["Month","DayOfWeek","DayOfYear"]:
            ftype = "Calendar Feature"
        else:
            ftype = "Other"
        is_num = pd.api.types.is_numeric_dtype(df_feat[feat])
        mean_val = round(float(df_feat[feat].mean()), 2) if is_num else "—"
        std_val  = round(float(df_feat[feat].std()), 2)  if is_num else "—"
        rows.append({
            "Feature":    feat,
            "Type":       ftype,
            "Non-Null":   int(df_feat[feat].notna().sum()),
            "Mean":       mean_val,
            "Std":        std_val,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – PREPROCESSING METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def render_preprocessing_methodology(df: pd.DataFrame, cfg: dict, name: str):
    st.markdown("### 🔬 Preprocessing Methodology Documentation")

    st.markdown(
        "The following documents the **actual, verifiable preprocessing pipeline** "
        "applied to the KrishiDrishti datasets, traced from source scripts and "
        "the current state of the cleaned datasets."
    )

    # ── Workflow diagram ─────────────────────────────────────────────────────
    steps = [
        "Raw AGMARKNET\nCSV Export",
        "Market Filter\n(APMC Target)",
        "Date Parsing\n& Validation",
        "Daily\nAggregation",
        "Weather API\nMerge",
        "Forward-Fill\nWeather Gaps",
        "Duplicate\nRemoval",
        "Outlier\nInspection",
        "Feature\nEngineering",
        "Cleaned\nMaster Dataset",
        "ML Model\nTraining",
    ]
    n    = len(steps)
    xpos = list(range(n))
    ypos = [0] * n

    fig = go.Figure()
    step_colors = {9: "#22C55E", 8: cfg["color"]}
    for i, (x, label) in enumerate(zip(xpos, steps)):
        clr = step_colors.get(i, "#1E3A5F")
        fig.add_trace(go.Scatter(
            x=[x], y=[0],
            mode="markers+text",
            marker=dict(size=58, color=clr, symbol="square",
                        line=dict(color="white", width=2)),
            text=[label],
            textposition="middle center",
            textfont=dict(size=8.5, color="white", family="Inter"),
            hoverinfo="text",
            hovertext=label.replace("\n", " "),
            showlegend=False,
        ))
        if i < n - 1:
            fig.add_annotation(
                x=x + 0.52, y=0, ax=x + 0.08, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.6,
                arrowcolor="white", arrowwidth=2,
            )
    fig.update_layout(
        title="KrishiDrishti – Data Preprocessing and Feature Engineering Workflow",
        height=230,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.6, n - 0.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.9, 0.9]),
        margin=dict(l=20, r=20, t=60, b=10),
        **_make_layout()
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Step-by-step table ───────────────────────────────────────────────────
    with st.expander("Step-by-Step Preprocessing Documentation", expanded=True):
        st.markdown("""
| Step | Operation | Source Script | Evidence / Verification |
|------|-----------|---------------|------------------------|
| 1 | **Raw AGMARKNET CSV Export** | Manual download from agmarknet.gov.in | `lasalgaon_onions.csv` (raw file retained) |
| 2 | **Market Filter** | `1_clean_agmarknet.py` L57 | `str.contains('LASALGAON')` filter applied |
| 3 | **Date Parsing & Validation** | `1_clean_agmarknet.py` L41–44 | `pd.to_datetime(..., errors='coerce')` + `dropna(subset=['Date'])` |
| 4 | **Daily Aggregation** | `2b_final_cleanup.py` L18 | `groupby('Date').agg(...)` – one row per day; Arrivals summed, prices averaged |
| 5 | **Weather Feature Merge** | `2_fetch_weather.py` | OpenMeteo API → `Temperature_C`, `Rainfall_mm` appended |
| 6 | **Forward-Fill Weather Gaps** | `2b_final_cleanup.py` L22–23 | `ffill()` applied to 3 trailing weather rows at end of 2024 |
| 7 | **Duplicate Handling** | `2b_final_cleanup.py` L18 | Daily aggregation inherently resolves per-variety duplicate rows |
| 8 | **Outlier Inspection** | `data_intelligence.py` (this module) | IQR + Z-score flagging; price spikes retained as genuine market events |
| 9 | **Feature Engineering** | `*_train_extended.py` scripts | Lag 1–21 + Rolling Mean 3/7/14d + Rolling Std 7d + Month/DoW/DoY |
| 10 | **Cleaned Master Dataset** | `2b_final_cleanup.py` L26 | `master_dataset.csv` → 0 duplicates, 0 missing values (Onion/TurDal); 1–2 (Potato trailing) |
| 11 | **ML Model Training** | `onion/potato/tur_train_extended.py` | Time-series train/test split → GradientBoostingRegressor (300 trees, lr=0.015) |
        """)

    # ── Current data quality callout ────────────────────────────────────────
    st.markdown("#### Current Dataset State (Post-Preprocessing)")
    miss_total = int(df.isnull().sum().sum())
    dup_total  = int(df.duplicated().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Missing Values",
        f"{miss_total}",
        delta="Fully cleaned" if miss_total == 0 else f"{miss_total/(len(df)*df.shape[1])*100:.4f}%",
    )
    c2.metric(
        "Duplicate Records",
        f"{dup_total}",
        delta="None detected" if dup_total == 0 else "Review needed",
    )
    c3.metric(
        "Total Records",
        f"{len(df):,}",
        delta=f"{df['Date'].min().year} – {df['Date'].max().year}",
    )

    st.info(
        "Note: Raw intermediate outputs (pre-cleaning CSVs) are not available in the "
        "repository. The before/after comparison is inferred from documented operations "
        "in the cleaning scripts. Current cleaned dataset state is directly verified."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT – called from app.py
# ─────────────────────────────────────────────────────────────────────────────
def render_data_intelligence_section():
    """
    Render the complete Data Intelligence and Preprocessing dashboard section.
    Import and call this function from app.py.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 18px;
        ">
            <h2 style="color: #FBBF24; margin: 0 0 8px 0; font-family: Inter, sans-serif;">
                Data Intelligence &amp; Preprocessing
            </h2>
            <p style="color: #CBD5E1; margin: 0; font-size: 0.97rem;">
                Academically demonstrating data quality, EDA, outlier inspection,
                feature engineering, and preprocessing methodology of the KrishiDrishti pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Commodity selector ───────────────────────────────────────────────────
    st.markdown("#### Select Commodity for Analysis")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        selected = st.selectbox(
            "Commodity",
            list(COMMODITY_CONFIG.keys()),
            key="di_commodity_select",
            label_visibility="collapsed",
        )
    cfg = COMMODITY_CONFIG[selected]
    st.markdown(
        f"**{cfg['icon']} {selected}** &nbsp;|&nbsp;"
        f" Market: `{cfg['market']}` &nbsp;|&nbsp; Unit: `{cfg['unit']}`"
    )
    st.caption(cfg["description"])

    # ── Load data ────────────────────────────────────────────────────────────
    file_path = cfg["file"]
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found: `{file_path}`. "
                 "Please ensure you are running Streamlit from the project directory.")
        return

    with st.spinner(f"Loading {selected} dataset…"):
        df      = load_commodity_data(file_path)
        df_feat = engineer_features(df)

    # ── Sub-section nav ──────────────────────────────────────────────────────
    section = st.radio(
        "Analysis Section",
        [
            "📋 Data Quality Metrics",
            "🔍 Missing Value Analysis",
            "📊 Exploratory Data Analysis",
            "🎯 Outlier Analysis",
            "⚙️ Feature Engineering",
            "🔬 Preprocessing Methodology",
        ],
        horizontal=True,
        key="di_section_select",
    )
    st.markdown("---")

    if section == "📋 Data Quality Metrics":
        render_data_quality(df, cfg)
    elif section == "🔍 Missing Value Analysis":
        render_missing_value_viz(df, cfg)
    elif section == "📊 Exploratory Data Analysis":
        render_eda(df, cfg, selected)
    elif section == "🎯 Outlier Analysis":
        render_outlier_analysis(df, cfg, selected)
    elif section == "⚙️ Feature Engineering":
        render_feature_engineering(df_feat, cfg, selected)
    elif section == "🔬 Preprocessing Methodology":
        render_preprocessing_methodology(df, cfg, selected)
