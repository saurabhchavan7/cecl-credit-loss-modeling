"""
PD Model Page
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
def load_validation_summary():
    p = MODEL_DIR / "validation_summary.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None

@st.cache_data
def load_lr_coefficients():
    p = MODEL_DIR / "lr_coefficients.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_iv_summary():
    p = MODEL_DIR / "iv_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_calibration():
    p = MODEL_DIR / "lr_calibration_validation.csv"
    return pd.read_csv(p) if p.exists() else None


def render():
    st.title("PD Model: Probability of Default")
    info_box("<strong>What does the PD model do?</strong> It estimates the probability that a borrower "
             "will default within 12 months. This is the first component of <strong>ECL = PD x LGD x EAD</strong>.")

    section_header("Model Performance Summary")
    vs = load_validation_summary()

    k1, k2, k3, k4 = st.columns(4)
    if vs is not None:
        lr, xgb = vs["logistic_regression"], vs["xgboost"]
        k1.metric("LR Validation AUC", f"{lr['val_auc']:.4f}")
        k2.metric("XGBoost Val AUC", f"{xgb['val_auc']:.4f}", f"+{xgb['val_auc']-lr['val_auc']:.4f}")
        k3.metric("LR Validation KS", f"{lr['val_ks']:.4f}")
        k4.metric("PSI (Train vs Val)", f"{lr['psi_val']:.4f}",
                   "Needs monitoring" if lr['psi_val'] > 0.25 else "Stable",
                   delta_color="inverse" if lr['psi_val'] > 0.25 else "normal")
    else:
        k1.metric("LR Validation AUC", "0.6777")
        k2.metric("XGBoost Val AUC", "0.7529", "+0.0752")
        k3.metric("LR Validation KS", "0.2651")
        k4.metric("PSI (Train vs Val)", "0.2916", "Needs monitoring", delta_color="inverse")

    tab_cmp, tab_cal, tab_iv, tab_coef, tab_why = st.tabs(["Model Comparison", "Calibration", "Information Value", "Coefficients", "Why Logistic Regression?"])

    # --- Model Comparison ---
    with tab_cmp:
        cc, ce = st.columns([2, 1])
        with cc:
            if vs is not None:
                labels = ["Train AUC", "Val AUC", "Test AUC", "Train KS", "Val KS", "Test KS"]
                lr_v = [lr["train_auc"], lr["val_auc"], lr["test_auc"], lr["train_ks"], lr["val_ks"], lr["test_ks"]]
                xg_v = [xgb["train_auc"], xgb["val_auc"], xgb["test_auc"], xgb["train_ks"], xgb["val_ks"], xgb["test_ks"]]
            else:
                labels = ["Train AUC", "Val AUC", "Test AUC", "Train KS", "Val KS", "Test KS"]
                lr_v = [0.7048, 0.6777, 0.6838, 0.3142, 0.2651, 0.2651]
                xg_v = [0.7579, 0.7529, 0.7606, 0.3893, 0.3853, 0.3853]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=lr_v, name="Logistic Regression", marker_color=COLORS["primary"],
                text=[f"{v:.4f}" for v in lr_v], textposition="outside"))
            fig.add_trace(go.Bar(x=labels, y=xg_v, name="XGBoost Challenger", marker_color=COLORS["success"],
                text=[f"{v:.4f}" for v in xg_v], textposition="outside"))
            fig.add_hline(y=0.75, line_dash="dash", line_color=COLORS["warning"], annotation_text="AUC Target: 0.75")
            fig.update_layout(title="Primary vs Challenger Performance", barmode="group", bargap=0.2)
            style_chart(fig, 480)
            st.plotly_chart(fig, use_container_width=True)
        with ce:
            info_box("<strong>AUC-ROC</strong> measures rank-ordering ability (>0.75 is good). "
                     "<strong>KS</strong> measures max separation (>0.30 is good). "
                     "XGBoost gains +7.5 AUC points over LR, the cost of interpretability.")
            warning_box("<strong>PSI = 0.29:</strong> Score distribution shifted between 2005 training and 2006 validation. "
                        "Reflects genuine deterioration in origination quality during the housing bubble.")

    # --- Calibration ---
    with tab_cal:
        cc, ce = st.columns([2, 1])
        with cc:
            cal = load_calibration()
            if cal is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(1, len(cal)+1)), y=cal["avg_predicted_pd"]*100,
                    name="Predicted PD (%)", marker_color=COLORS["primary"], opacity=0.7))
                fig.add_trace(go.Scatter(x=list(range(1, len(cal)+1)), y=cal["actual_default_rate"]*100,
                    name="Actual Default Rate (%)", mode="lines+markers",
                    marker=dict(color=COLORS["danger"], size=10), line=dict(width=2.5)))
                fig.update_layout(title="Calibration: Predicted vs Actual by Decile",
                    xaxis_title="Risk Decile (1=Lowest Risk)", yaxis_title="Rate (%)", bargap=0.3)
                style_chart(fig, 480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Calibration data not found. Run `run_pd_model.py`.")
        with ce:
            info_box("<strong>Why calibration matters more than AUC:</strong> If the model says 5% PD but actual is 8%, "
                     "the bank is under-reserved by 3%. Across $694B, that is $21B missing. "
                     "LR was trained <strong>without class reweighting</strong> to preserve calibration.")

    # --- IV ---
    with tab_iv:
        cc, ce = st.columns([2, 1])
        with cc:
            iv = load_iv_summary()
            if iv is not None:
                iv = iv.head(15)
                colors = [COLORS["danger"] if v>0.5 else COLORS["success"] if v>0.3 else COLORS["primary"] if v>0.1
                          else COLORS["warning"] if v>0.02 else COLORS["gray"] for v in iv["iv"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=iv["iv"], y=iv["feature"], orientation="h", marker_color=colors,
                    text=[f"{v:.4f}" for v in iv["iv"]], textposition="outside"))
                fig.add_vline(x=0.02, line_dash="dash", line_color=COLORS["gray"])
                fig.add_vline(x=0.30, line_dash="dash", line_color=COLORS["success"])
                fig.add_vline(x=0.50, line_dash="dash", line_color=COLORS["danger"])
                fig.update_layout(title="Information Value: Feature Predictive Power",
                    xaxis_title="IV", yaxis=dict(autorange="reversed"))
                style_chart(fig, 500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("IV summary not found.")
        with ce:
            info_box("<strong>IV thresholds:</strong> Gray <0.02 (drop), Amber 0.02-0.10 (weak), "
                     "Blue 0.10-0.30 (medium), Green 0.30-0.50 (strong), Red >0.50 (suspicious). "
                     "`borrower_credit_score` IV>0.50 was investigated and confirmed legitimate.")

    # --- Coefficients ---
    with tab_coef:
        cc, ce = st.columns([2, 1])
        with cc:
            coef = load_lr_coefficients()
            if coef is not None:
                coef["feature_clean"] = coef["feature"].str.replace("_woe", "")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=coef["coefficient"], y=coef["feature_clean"], orientation="h",
                    marker_color=[COLORS["primary"] if c>0 else COLORS["danger"] for c in coef["coefficient"]],
                    text=[f"{v:+.4f}" for v in coef["coefficient"]], textposition="outside"))
                fig.add_vline(x=0, line_color=COLORS["gray"])
                fig.update_layout(title="Logistic Regression Coefficients", xaxis_title="Coefficient",
                    yaxis=dict(autorange="reversed"), showlegend=False)
                style_chart(fig, 450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coefficient data not found.")
        with ce:
            info_box("Positive coefficients mean higher WoE (lower risk) reduces default probability. "
                     "All should be positive since WoE ensures monotonic relationships.")

    # --- Why LR ---
    with tab_why:
        r1, r2 = st.columns(2)
        with r1:
            info_box("<strong>1. Interpretability:</strong> Coefficients directly answer 'why is this loan risky?'")
            info_box("<strong>2. Monotonicity:</strong> WoE guarantees higher FICO = lower risk, always.")
        with r2:
            info_box("<strong>3. Stability:</strong> Coefficients stable across time and resampling.")
            info_box("<strong>4. Regulatory acceptance:</strong> OCC/Fed/FDIC have decades of comfort with LR.")
        warning_box("<strong>Tradeoff:</strong> LR sacrifices ~7.5 AUC points vs XGBoost. This is the explicit cost of interpretability.")