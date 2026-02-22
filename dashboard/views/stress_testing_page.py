"""
Stress Testing Page
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
def load_stress_summary():
    p = MODEL_DIR / "stress_test_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_quarterly_path():
    p = MODEL_DIR / "stress_severely_adverse_quarterly.csv"
    return pd.read_csv(p) if p.exists() else None


def render():
    st.title("Stress Testing: Federal Reserve Scenarios")
    info_box("<strong>Stress testing:</strong> The Fed publishes hypothetical disaster scenarios annually. "
             "Banks must demonstrate survival. 2025 Severely Adverse: unemployment 10%, HPI -33%, GDP -7.8%.")

    section_header("Scenario Comparison")
    sd = load_stress_summary()
    k1, k2, k3, k4 = st.columns(4)
    if sd is not None:
        b = sd[sd["scenario"]=="Baseline"].iloc[0]
        a = sd[sd["scenario"]=="Severely Adverse"].iloc[0]
        k1.metric("Baseline 13Q Loss", f"${b['total_loss']/1e6:,.0f}M", f"{b['loss_rate']*100:.2f}%")
        k2.metric("Adverse 13Q Loss", f"${a['total_loss']/1e6:,.0f}M", f"{a['loss_rate']*100:.2f}%", delta_color="inverse")
        k3.metric("Stress Increment", f"+${(a['total_loss']-b['total_loss'])/1e6:,.0f}M")
        k4.metric("Loss Multiplier", f"{a['total_loss']/b['total_loss']:.1f}x")
    else:
        k1.metric("Baseline 13Q Loss", "$99,615M", "14.36%")
        k2.metric("Adverse 13Q Loss", "$311,945M", "44.96%", delta_color="inverse")
        k3.metric("Stress Increment", "+$212,330M")
        k4.metric("Loss Multiplier", "3.1x")

    tab_path, tab_custom, tab_meth = st.tabs(["Quarterly Loss Path", "Custom Scenario", "Methodology"])

    with tab_path:
        qp = load_quarterly_path()
        if qp is not None:
            cc, ce = st.columns([2, 1])
            with cc:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=qp["quarter"], y=qp["cumulative_el"]/1e6, name="Cumulative Loss ($M)",
                    mode="lines+markers", line=dict(color=COLORS["danger"], width=3), fill="tozeroy",
                    fillcolor="rgba(220,38,38,0.08)"))
                fig.add_trace(go.Bar(x=qp["quarter"], y=qp["quarterly_el"]/1e6, name="Quarterly Loss ($M)",
                    marker_color=COLORS["primary"], opacity=0.6))
                fig.update_layout(title="Severely Adverse: Quarterly and Cumulative Loss", xaxis_title="Quarter",
                    yaxis_title="Loss ($M)", bargap=0.3)
                style_chart(fig, 480)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=qp["quarter"], y=qp["pd_multiplier"], name="PD Mult",
                    mode="lines+markers", line=dict(color=COLORS["primary"], width=2)))
                fig2.add_trace(go.Scatter(x=qp["quarter"], y=qp["lgd_multiplier"], name="LGD Mult",
                    mode="lines+markers", line=dict(color=COLORS["warning"], width=2)))
                fig2.add_hline(y=1.0, line_dash="dash", line_color=COLORS["gray"], annotation_text="Baseline")
                fig2.update_layout(title="Stress Multipliers Over Horizon", xaxis_title="Quarter", yaxis_title="Multiplier")
                style_chart(fig2, 350)
                st.plotly_chart(fig2, use_container_width=True)
            with ce:
                info_box("Blue bars = quarterly loss. Red line = cumulative total. "
                         "Losses accelerate as unemployment peaks ~2025Q3-Q4.")
                info_box("<strong>Peak:</strong> 2025Q4, PD ~2.5x, LGD ~1.5x baseline.")
        else:
            st.info("Quarterly path not found. Run `run_stress_test.py`.")

    with tab_custom:
        st.markdown("### Custom Stress Scenario")
        info_box("Drag sliders to see how losses change under different economic assumptions.")
        c1, c2, c3 = st.columns(3)
        with c1:
            ur = st.slider("Unemployment (%)", 3.0, 15.0, 4.3, 0.1)
        with c2:
            hpi = st.slider("HPI Change (%)", -40.0, 20.0, 0.0, 1.0)
        with c3:
            gdp = st.slider("GDP Growth (%)", -10.0, 5.0, 2.0, 0.5)

        pd_m = 1.0 + 0.25*max(ur-4.3, 0) + (abs(gdp)*0.05 if gdp < 0 else 0)
        lgd_m = 1.0 + 0.015*max(-hpi, 0)
        base_rate = 0.1256*0.2887
        stressed_rate = base_rate*pd_m*lgd_m
        total_bal = 693.8e9

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("PD Multiplier", f"{pd_m:.2f}x")
        r2.metric("LGD Multiplier", f"{lgd_m:.2f}x")
        r3.metric("Annual EL", f"${stressed_rate*total_bal/1e6:,.0f}M",
                   f"{(stressed_rate-base_rate)*total_bal/1e6:+,.0f}M vs baseline")
        r4.metric("Loss Rate", f"{stressed_rate*100:.2f}%")

        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=stressed_rate*100,
            delta={"reference": base_rate*100, "suffix": "pp"},
            title={"text": "Portfolio Loss Rate (%)"},
            gauge={"axis": {"range": [0, 20]}, "bar": {"color": COLORS["primary"]},
                   "steps": [{"range":[0,5],"color":"#f0fdf4"},{"range":[5,10],"color":"#fefce8"},{"range":[10,20],"color":"#fef2f2"}],
                   "threshold": {"line":{"color":COLORS["danger"],"width":3},"thickness":0.8,"value":base_rate*100}}))
        fig.update_layout(height=300, paper_bgcolor="#ffffff", font=dict(color="#1e293b"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_meth:
        m1, m2 = st.columns(2)
        with m1:
            info_box("<strong>Why not direct macro substitution?</strong> PD/LGD models use origination-time features. "
                     "A loan originated at peak HPI is riskier, not safer. Substituting stressed HPI would invert the signal.")
        with m2:
            info_box("<strong>Scalar overlay:</strong> Score baseline once, then multiply by stress factors. "
                     "PD: +25% per +1% unemployment. LGD: +1.5% per -1% HPI. Calibrated to 2008 crisis.")
        warning_box("Limitation: assumes linear multipliers. Real relationships may accelerate at extremes.")