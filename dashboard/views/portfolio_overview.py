"""
Portfolio Overview Page
Author: Saurabh Chavan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import style_chart, COLORS, info_box, warning_box, section_header

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

@st.cache_data
def load_ecl_summary():
    p = MODEL_DIR / "ecl_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_mc_metrics():
    p = MODEL_DIR / "mc_risk_metrics.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_portfolio_totals():
    p = MODEL_DIR / "dashboard_portfolio_totals.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_fico_summary():
    p = MODEL_DIR / "dashboard_fico_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_ltv_summary():
    p = MODEL_DIR / "dashboard_ltv_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_vintage_summary():
    p = MODEL_DIR / "dashboard_vintage_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_ecl_loan_level():
    """Try parquet first (local), fall back to small CSVs (deployed)."""
    p = MODEL_DIR / "ecl_loan_level.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def render():
    st.title("Portfolio Overview")
    info_box(
        "<strong>What is this dashboard?</strong> This is an end-to-end "
        "CECL (Current Expected Credit Loss) credit risk modeling framework "
        "applied to 3.8 million Fannie Mae mortgage loans. CECL (ASC 326) "
        "requires banks to estimate <strong>lifetime expected losses</strong> "
        "from day one of every loan, considering multiple economic scenarios."
    )

    ecl_summary = load_ecl_summary()
    mc_metrics = load_mc_metrics()
    totals = load_portfolio_totals()

    # --- KPI Row 1 ---
    section_header("Portfolio Summary")
    k1, k2, k3, k4 = st.columns(4)
    if totals is not None:
        t = totals.iloc[0]
        k1.metric("Total Loans", f"{t['total_loans']:,.0f}")
        k2.metric("Outstanding Balance", f"${t['total_balance']/1e9:,.1f}B")
        k3.metric("Default Rate", f"{t['default_rate']*100:.1f}%")
        k4.metric("Mean FICO", f"{t['mean_fico']:.0f}")
    else:
        k1.metric("Total Loans", "3,804,801")
        k2.metric("Outstanding Balance", "$693.8B")
        k3.metric("Default Rate", "14.4%")
        k4.metric("Mean FICO", "722")

    # --- KPI Row 2: ECL ---
    section_header("Expected Credit Loss (CECL)")
    e1, e2, e3, e4 = st.columns(4)
    if ecl_summary is not None:
        base = ecl_summary[ecl_summary["scenario"] == "Baseline"].iloc[0]
        adv = ecl_summary[ecl_summary["scenario"] == "Severely Adverse"].iloc[0]
        wt = ecl_summary[ecl_summary["scenario"].str.contains("Weighted", na=False)]
        wt = wt.iloc[0] if len(wt) > 0 else base
        e1.metric("Baseline ECL", f"${base['total_ecl']/1e6:,.0f}M", f"{base['portfolio_ecl_rate']*100:.2f}%")
        e2.metric("Severely Adverse ECL", f"${adv['total_ecl']/1e6:,.0f}M", f"{adv['portfolio_ecl_rate']*100:.2f}%", delta_color="inverse")
        e3.metric("Weighted ECL (60/40)", f"${wt['total_ecl']/1e6:,.0f}M", f"{wt['portfolio_ecl_rate']*100:.2f}%")
    else:
        e1.metric("Baseline ECL", "$115,562M", "16.66%")
        e2.metric("Severely Adverse ECL", "$221,865M", "31.98%", delta_color="inverse")
        e3.metric("Weighted ECL (60/40)", "$158,083M", "22.78%")
    if mc_metrics is not None:
        mc = mc_metrics.iloc[0]
        e4.metric("VaR 99.9% (MC)", f"${mc['var_999']/1e6:,.0f}M", f"{mc['var_999']/mc['total_balance']*100:.2f}%", delta_color="inverse")
    else:
        e4.metric("VaR 99.9% (MC)", "$99,208M", "14.30%", delta_color="inverse")

    info_box(
        "<strong>How to read:</strong> Baseline ECL = expected loss under normal conditions. "
        "Severely Adverse = Fed worst-case (unemployment 10%, HPI -33%). "
        "Weighted ECL (60/40) = reserve on financial statements. "
        "VaR 99.9% = loss exceeded only 0.1% of the time across 10,000 Monte Carlo simulations."
    )

    # --- Portfolio Composition ---
    section_header("Portfolio Composition")

    # Use parquet if available (local), otherwise small CSVs (deployed)
    loan_df = load_ecl_loan_level()
    fico_csv = load_fico_summary()
    ltv_csv = load_ltv_summary()
    vintage_csv = load_vintage_summary()

    has_data = loan_df is not None or fico_csv is not None

    if has_data:
        tab_fico, tab_ltv, tab_vintage, tab_ecl = st.tabs(["By FICO Score", "By LTV Ratio", "By Vintage Year", "ECL by Segment"])

        with tab_fico:
            col_c, col_t = st.columns([2, 1])
            with col_c:
                if loan_df is not None:
                    data = loan_df.groupby("fico_bucket", observed=True).agg(
                        count=("default_flag", "count"), defaults=("default_flag", "sum")).reset_index()
                    data["default_rate"] = data["defaults"] / data["count"]
                elif fico_csv is not None:
                    data = fico_csv
                else:
                    data = None

                if data is not None:
                    bucket_col = "fico_bucket" if "fico_bucket" in data.columns else data.columns[0]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=data[bucket_col], y=data["count"], name="Loan Count", marker_color=COLORS["primary"]))
                    fig.add_trace(go.Scatter(x=data[bucket_col], y=data["default_rate"]*100, name="Default Rate (%)",
                        mode="lines+markers", marker=dict(color=COLORS["danger"], size=10),
                        line=dict(color=COLORS["danger"], width=2), yaxis="y2"))
                    fig.update_layout(title="Loan Distribution and Default Rate by FICO Score",
                        xaxis_title="FICO Bucket", yaxis=dict(title="Loan Count"),
                        yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False),
                        legend=dict(x=0.01, y=0.99), bargap=0.3)
                    style_chart(fig, 420)
                    st.plotly_chart(fig, use_container_width=True)
            with col_t:
                info_box("<strong>FICO Score</strong> is the single strongest predictor of default. "
                         "Borrowers below 620 (subprime) default at rates 3-5x higher than those above 740.")

        with tab_ltv:
            col_c, col_t = st.columns([2, 1])
            with col_c:
                if loan_df is not None:
                    data = loan_df.groupby("ltv_bucket", observed=True).agg(
                        count=("default_flag", "count"), defaults=("default_flag", "sum")).reset_index()
                    data["default_rate"] = data["defaults"] / data["count"]
                elif ltv_csv is not None:
                    data = ltv_csv
                else:
                    data = None

                if data is not None:
                    bucket_col = "ltv_bucket" if "ltv_bucket" in data.columns else data.columns[0]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=data[bucket_col], y=data["count"], name="Loan Count", marker_color=COLORS["success"]))
                    fig.add_trace(go.Scatter(x=data[bucket_col], y=data["default_rate"]*100, name="Default Rate (%)",
                        mode="lines+markers", marker=dict(color=COLORS["danger"], size=10),
                        line=dict(color=COLORS["danger"], width=2), yaxis="y2"))
                    fig.update_layout(title="Loan Distribution and Default Rate by LTV",
                        xaxis_title="LTV Bucket", yaxis=dict(title="Loan Count"),
                        yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False),
                        legend=dict(x=0.01, y=0.99), bargap=0.3)
                    style_chart(fig, 420)
                    st.plotly_chart(fig, use_container_width=True)
            with col_t:
                info_box("<strong>Loan-to-Value (LTV)</strong> measures borrower leverage. "
                         "Above 90% LTV, a small price decline puts the borrower underwater.")

        with tab_vintage:
            col_c, col_t = st.columns([2, 1])
            with col_c:
                if loan_df is not None:
                    data = loan_df.groupby("origination_year").agg(
                        count=("default_flag", "count"), defaults=("default_flag", "sum"),
                        balance=("current_balance", "sum")).reset_index()
                    data = data[data["origination_year"] >= 2004]
                    data["default_rate"] = data["defaults"] / data["count"]
                elif vintage_csv is not None:
                    data = vintage_csv
                else:
                    data = None

                if data is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=data["origination_year"].astype(str), y=data["balance"]/1e9,
                        name="Balance ($B)", marker_color=COLORS["purple"]))
                    fig.add_trace(go.Scatter(x=data["origination_year"].astype(str), y=data["default_rate"]*100,
                        name="Default Rate (%)", mode="lines+markers",
                        marker=dict(color=COLORS["danger"], size=10), line=dict(color=COLORS["danger"], width=2), yaxis="y2"))
                    fig.update_layout(title="Portfolio by Origination Year",
                        xaxis_title="Year", yaxis=dict(title="Balance ($B)"),
                        yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False), bargap=0.3)
                    style_chart(fig, 420)
                    st.plotly_chart(fig, use_container_width=True)
            with col_t:
                info_box("<strong>2007 vintages</strong> have the highest default rate (17.4%), "
                         "originated at peak bubble with loosened standards.")

        with tab_ecl:
            col_c, col_t = st.columns([2, 1])
            with col_c:
                if loan_df is not None and "ecl_dollars" in loan_df.columns:
                    data = loan_df.groupby("fico_bucket", observed=True).agg(
                        ecl=("ecl_dollars", "sum"), balance=("current_balance", "sum")).reset_index()
                    data["ecl_rate"] = data["ecl"] / data["balance"]
                elif fico_csv is not None and "ecl_rate" in fico_csv.columns:
                    data = fico_csv
                else:
                    data = None

                if data is not None:
                    bucket_col = "fico_bucket" if "fico_bucket" in data.columns else data.columns[0]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=data[bucket_col], y=data["ecl_rate"]*100,
                        marker_color=[COLORS["danger"], COLORS["warning"], "#eab308", COLORS["success"], COLORS["primary"]],
                        text=[f"{r*100:.1f}%" for r in data["ecl_rate"]], textposition="outside"))
                    fig.update_layout(title="CECL ECL Rate by FICO Bucket", xaxis_title="FICO Bucket",
                        yaxis_title="ECL Rate (%)", showlegend=False, bargap=0.3)
                    style_chart(fig, 420)
                    st.plotly_chart(fig, use_container_width=True)
            with col_t:
                info_box("Subprime borrowers (FICO < 620) have ECL rates roughly 2x higher than super-prime (740+).")

    # --- Production Context ---
    section_header("How This Framework Operates in Production")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Scoring Pipeline")
        info_box("<strong>Daily:</strong> New loans scored through PD + LGD models for day-one ECL reserve.<br>"
                 "<strong>Monthly:</strong> Full portfolio re-scored with updated FRED macro data.")
    with p2:
        st.markdown("#### Governance Cycle")
        info_box("<strong>Quarterly:</strong> PSI computed, calibration refreshed, stress tests updated.<br>"
                 "<strong>Annually:</strong> Full re-development. Independent validation by challenge function.")