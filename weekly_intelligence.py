"""
weekly_intelligence.py
======================
KrishiDrishti – Weekly Agricultural Price Intelligence Module

Provides weekly-aggregated analysis (2021-2025) for:
  - Onion  (Lasalgaon) : data/weekly_aggregated/weekly_onion_lasalgaon.csv
  - Potato (Achalda)   : data/weekly_aggregated/weekly_potato_achalda.csv
  - Tur Dal (Latur)    : data/weekly_aggregated/weekly_turdal_latur.csv

Aggregation uses ISO calendar weeks; no synthetic data is generated.
Original daily datasets are untouched.
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WEEKLY_DIR = os.path.join("data", "weekly_aggregated")

COMMODITY_MAP = {
    "Onion (Lasalgaon)": {
        "weekly_csv":  os.path.join(WEEKLY_DIR, "weekly_onion_lasalgaon.csv"),
        "daily_csv":   "master_dataset.csv",
        "color":       "#FF6B35",
        "colors_year": ["#FF6B35","#FF9F70","#FFD4B8","#CC4400","#FF3300"],
        "market":      "Lasalgaon APMC, Maharashtra",
        "icon":        "🧅",
    },
    "Potato (Achalda)": {
        "weekly_csv":  os.path.join(WEEKLY_DIR, "weekly_potato_achalda.csv"),
        "daily_csv":   "master_dataset_achalda.csv",
        "color":       "#8B5CF6",
        "colors_year": ["#8B5CF6","#A78BFA","#C4B5FD","#6D28D9","#4C1D95"],
        "market":      "Achalda APMC, Uttar Pradesh",
        "icon":        "🥔",
    },
    "Tur Dal (Latur)": {
        "weekly_csv":  os.path.join(WEEKLY_DIR, "weekly_turdal_latur.csv"),
        "daily_csv":   "master_dataset_latur.csv",
        "color":       "#10B981",
        "colors_year": ["#10B981","#34D399","#6EE7B7","#059669","#047857"],
        "market":      "Latur APMC, Maharashtra",
        "icon":        "🫘",
    },
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0E1117"

TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]


def _layout(**kw):
    base = dict(template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family="Inter, sans-serif"))
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_weekly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Week_Start_Date"] = pd.to_datetime(df["Week_Start_Date"])
    df["Week_End_Date"]   = pd.to_datetime(df["Week_End_Date"])
    df["Year"]    = df["Year"].astype(int)
    df["ISO_Week"] = df["ISO_Week"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _filter_target_years(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose ISO Year falls in TARGET_YEARS."""
    return df[df["Year"].isin(TARGET_YEARS)].copy()


def _daily_in_range(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["Date"] >= "2021-01-01") & (df["Date"] <= "2025-12-31")
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# CHART A – Weekly Modal Price Trend
# ─────────────────────────────────────────────────────────────────────────────
def chart_weekly_trend(wdf: pd.DataFrame, cfg: dict, name: str) -> go.Figure:
    df = _filter_target_years(wdf).sort_values("Week_Start_Date")
    color = cfg["color"]
    light = color + "28"   # 16% alpha hex suffix

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Week_Start_Date"],
        y=df["Weekly_Mean_Modal_Price"],
        mode="lines+markers",
        name="Weekly Mean Price",
        line=dict(color=color, width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor=light,
        customdata=df[["ISO_Week","Observation_Count","Year"]],
        hovertemplate=(
            "<b>Week %{customdata[0]}, %{customdata[2]}</b><br>"
            "Week start : %{x|%d %b %Y}<br>"
            "Mean price : ₹%{y:,.2f}/Qtl<br>"
            "Obs count  : %{customdata[1]}<extra></extra>"
        ),
    ))
    # 4-week rolling average overlay
    roll = df["Weekly_Mean_Modal_Price"].rolling(4, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df["Week_Start_Date"], y=roll,
        mode="lines", name="4-Week Rolling Avg",
        line=dict(color="#FBBF24", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"{name} – Weekly Modal Price Trend (2021–2025)",
        xaxis_title="Week Start Date",
        yaxis_title="₹ / Quintal",
        hovermode="x unified",
        height=460,
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART B – Year-wise Weekly Comparison
# ─────────────────────────────────────────────────────────────────────────────
def chart_yearwise_weekly(wdf: pd.DataFrame, cfg: dict, name: str,
                           selected_years: list) -> go.Figure:
    df     = _filter_target_years(wdf)
    colors = cfg["colors_year"]
    fig = go.Figure()
    for i, yr in enumerate(sorted(selected_years)):
        sub = df[df["Year"] == yr].sort_values("ISO_Week")
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ISO_Week"],
            y=sub["Weekly_Mean_Modal_Price"],
            mode="lines+markers",
            name=str(yr),
            line=dict(color=colors[i % len(colors)], width=2.2),
            marker=dict(size=5),
            customdata=sub[["Observation_Count","Week_Start_Date"]],
            hovertemplate=(
                f"<b>{yr} – ISO Week %{{x}}</b><br>"
                "Mean price : ₹%{y:,.2f}/Qtl<br>"
                "Obs count  : %{customdata[0]}<br>"
                "Week start : %{customdata[1]|%d %b}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=f"{name} – Year-wise ISO Week Price Comparison",
        xaxis_title="ISO Week Number",
        yaxis_title="₹ / Quintal",
        hovermode="x unified",
        height=460,
        xaxis=dict(tickmode="linear", tick0=1, dtick=4),
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART C – Monthly Average Price Comparison
# ─────────────────────────────────────────────────────────────────────────────
def chart_monthly_comparison(daily_df: pd.DataFrame, cfg: dict, name: str,
                              selected_years: list) -> go.Figure:
    df = _daily_in_range(daily_df).copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df = df[df["Year"].isin(selected_years)]

    monthly = (
        df.groupby(["Year","Month"])["Modal_Price"]
        .agg(["mean","count"])
        .reset_index()
    )
    colors = cfg["colors_year"]
    fig = go.Figure()
    for i, yr in enumerate(sorted(selected_years)):
        sub = monthly[monthly["Year"] == yr].sort_values("Month")
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=[MONTH_NAMES[m-1] for m in sub["Month"]],
            y=sub["mean"],
            mode="lines+markers",
            name=str(yr),
            line=dict(color=colors[i % len(colors)], width=2.2),
            marker=dict(size=8, symbol="circle"),
            customdata=sub["count"],
            hovertemplate=(
                f"<b>{yr} – %{{x}}</b><br>"
                "Avg price : ₹%{y:,.2f}/Qtl<br>"
                "Daily obs : %{customdata}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=f"{name} – Monthly Average Price Comparison (2021–2025)",
        xaxis_title="Month",
        yaxis_title="Avg ₹ / Quintal",
        height=440,
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART D – Weekly Price Distribution Box Plots
# ─────────────────────────────────────────────────────────────────────────────
def chart_weekly_distribution(wdf: pd.DataFrame, cfg: dict, name: str,
                               selected_years: list) -> go.Figure:
    df     = _filter_target_years(wdf)
    df     = df[df["Year"].isin(selected_years)]
    colors = cfg["colors_year"]
    fig = go.Figure()
    for i, yr in enumerate(sorted(selected_years)):
        sub = df[df["Year"] == yr]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Box(
            y=sub["Weekly_Mean_Modal_Price"],
            name=str(yr),
            marker_color=colors[i % len(colors)],
            boxpoints="all",
            jitter=0.35,
            pointpos=-1.6,
            customdata=sub["ISO_Week"],
            hovertemplate=(
                f"<b>{yr}</b><br>"
                "Weekly mean : ₹%{y:,.2f}/Qtl<br>"
                "ISO Week    : %{customdata}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=f"{name} – Year-wise Weekly Price Distribution",
        yaxis_title="Weekly Mean ₹ / Quintal",
        height=460,
        **_layout()
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def build_stats_table(wdf: pd.DataFrame, selected_years: list) -> pd.DataFrame:
    df = _filter_target_years(wdf)
    df = df[df["Year"].isin(selected_years)]
    rows = []
    for yr in sorted(df["Year"].unique()):
        sub = df[df["Year"] == yr]["Weekly_Mean_Modal_Price"]
        rows.append({
            "Year":        int(yr),
            "Weeks":       len(sub),
            "Min (₹)":     round(sub.min(), 2),
            "Max (₹)":     round(sub.max(), 2),
            "Mean (₹)":    round(sub.mean(), 2),
            "Median (₹)":  round(sub.median(), 2),
            "Std Dev (₹)": round(sub.std(), 2),
            "CV (%)":      round(sub.std() / sub.mean() * 100, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def render_weekly_intelligence_section():
    """
    Render the Weekly Agricultural Price Intelligence section.
    Call from app.py after the Data Intelligence section.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d1b2a, #1b2838, #1e3a5f);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 18px;
        ">
            <h2 style="color: #60A5FA; margin: 0 0 8px 0; font-family: Inter, sans-serif;">
                📅 Weekly Agricultural Price Intelligence
            </h2>
            <p style="color: #CBD5E1; margin: 0; font-size: 0.97rem;">
                ISO-week aggregated Modal Price analysis (2021–2025) &nbsp;|&nbsp;
                Aggregation from actual daily observations only — no synthetic data generated.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guard: check CSVs exist ──────────────────────────────────────────────
    if not os.path.isdir(WEEKLY_DIR):
        st.error(
            f"Weekly aggregated data not found at `{WEEKLY_DIR}/`. "
            "Please run `generate_weekly_aggregates.py` first."
        )
        return

    # ── Selectors ────────────────────────────────────────────────────────────
    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 3])

    with sel_col1:
        commodity = st.selectbox(
            "Commodity",
            list(COMMODITY_MAP.keys()),
            key="wi_commodity",
        )

    cfg        = COMMODITY_MAP[commodity]
    weekly_df  = load_weekly(cfg["weekly_csv"])
    daily_df   = load_daily(cfg["daily_csv"])

    available_years = sorted(
        [y for y in weekly_df["Year"].unique() if y in TARGET_YEARS]
    )

    with sel_col2:
        selected_years = st.multiselect(
            "Years to compare",
            options=available_years,
            default=available_years,
            key="wi_years",
        )

    if not selected_years:
        st.warning("Please select at least one year.")
        return

    with sel_col3:
        st.markdown(
            f"**{cfg['icon']} {commodity}** &nbsp;|&nbsp; `{cfg['market']}`"
        )

    # ── Available data note ──────────────────────────────────────────────────
    data_years_str = ", ".join(str(y) for y in available_years)
    if set(available_years) != {2021,2022,2023,2024,2025}:
        st.info(
            f"ℹ️ Available data years for **{commodity}**: {data_years_str}. "
            "The source dataset does not extend beyond its recorded date range."
        )

    st.markdown("---")

    # ── Summary metrics strip ────────────────────────────────────────────────
    filtered = weekly_df[weekly_df["Year"].isin(selected_years)]
    total_weeks = len(filtered)
    total_obs   = int(filtered["Observation_Count"].sum())
    overall_mean = filtered["Weekly_Mean_Modal_Price"].mean()
    overall_min  = filtered["Weekly_Mean_Modal_Price"].min()
    overall_max  = filtered["Weekly_Mean_Modal_Price"].max()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📅 Total Weeks",       f"{total_weeks}")
    m2.metric("📊 Total Daily Obs",   f"{total_obs:,}")
    m3.metric("💰 Overall Mean",      f"₹{overall_mean:,.0f}")
    m4.metric("📉 Weekly Min",        f"₹{overall_min:,.0f}")
    m5.metric("📈 Weekly Max",        f"₹{overall_max:,.0f}")

    st.markdown("---")

    # ── Chart tabs ───────────────────────────────────────────────────────────
    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
        "📈 Weekly Trend",
        "📊 Year-wise Comparison",
        "📅 Monthly Comparison",
        "📦 Price Distribution",
        "🗂️ Raw Weekly Data",
    ])

    with tab_a:
        st.plotly_chart(
            chart_weekly_trend(weekly_df, cfg, commodity),
            use_container_width=True,
        )
        st.caption(
            "X-axis: ISO Week_Start_Date. Y-axis: arithmetic mean of daily Modal_Price "
            "within that ISO week. Dashed yellow: 4-week rolling average."
        )

    with tab_b:
        fig_b = chart_yearwise_weekly(weekly_df, cfg, commodity, selected_years)
        st.plotly_chart(fig_b, use_container_width=True)
        st.caption(
            "X-axis: ISO week number (1–53). Each line is a separate calendar year. "
            "Weeks with partial observations are included with actual available count."
        )

    with tab_c:
        fig_c = chart_monthly_comparison(daily_df, cfg, commodity, selected_years)
        st.plotly_chart(fig_c, use_container_width=True)
        st.caption(
            "Computed directly from original daily observations grouped by calendar month. "
            "Shows multi-year seasonal price patterns."
        )

    with tab_d:
        fig_d = chart_weekly_distribution(weekly_df, cfg, commodity, selected_years)
        st.plotly_chart(fig_d, use_container_width=True)

        # Stats table
        st.markdown("#### Year-wise Weekly Price Statistics")
        stats_df = build_stats_table(weekly_df, selected_years)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.caption(
            "CV = Coefficient of Variation (Std Dev / Mean × 100). "
            "Higher CV indicates greater intra-year price volatility."
        )

    with tab_e:
        st.markdown("#### Weekly Aggregated Dataset")
        show_df = _filter_target_years(weekly_df)
        show_df = show_df[show_df["Year"].isin(selected_years)].reset_index(drop=True)

        # Validation notice
        total_src_obs = int(show_df["Observation_Count"].sum())
        daily_count   = len(_daily_in_range(daily_df)[
            _daily_in_range(daily_df)["Date"].dt.year.isin(selected_years)
        ])
        match_flag = "✅" if abs(total_src_obs - daily_count) <= 7 else "⚠️"
        st.success(
            f"{match_flag} **Validation**: {total_src_obs:,} daily observations "
            f"aggregated into {len(show_df)} ISO weeks  "
            f"(daily dataset has {daily_count:,} rows for selected years — "
            f"small differences are due to ISO week year-boundary overlap)."
        )

        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Download button
        csv_bytes = show_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {commodity.split()[0]} Weekly CSV",
            data=csv_bytes,
            file_name=f"weekly_{commodity.split()[0].lower()}_{'-'.join(str(y) for y in sorted(selected_years))}.csv",
            mime="text/csv",
            key="wi_download",
        )

        # Methodology note
        with st.expander("Aggregation Methodology"):
            st.markdown("""
**Formula:**
```
Weekly_Mean_Modal_Price = Σ(Modal_Price_daily) / Observation_Count
```

**Steps:**
1. Load cleaned daily dataset (`master_dataset*.csv`) — **no modification**.
2. Filter to **2021-01-01 → 2025-12-31**.
3. Extract ISO calendar `year` and `week` via `pandas.Series.dt.isocalendar()`.
4. `groupby(['ISO_Year', 'ISO_Week'])` — guarantees year isolation (Week 1 of 2022 ≠ Week 1 of 2023).
5. Aggregate: `mean` of `Modal_Price`, `count` of valid daily rows.
6. Derive `Week_Start_Date` = `min(Date)` within group, `Week_End_Date` = `max(Date)`.
7. **Partial weeks** (start/end of year or dataset) — mean computed from available observations only; count reflects actual available rows.
8. **No missing weeks filled** — only ISO weeks with at least one real daily observation appear.
            """)
