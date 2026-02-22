"""
Loan Scorer Page
Author: Saurabh Chavan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import style_chart, COLORS, info_box, warning_box, section_header

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

@st.cache_resource
def load_models():
    try:
        m = {}
        m["pd_model"] = joblib.load(MODEL_DIR / "pd_logistic_regression.pkl")
        m["woe_results"] = joblib.load(MODEL_DIR / "woe_results.pkl")
        m["lgd_model"] = joblib.load(MODEL_DIR / "lgd_ols.pkl")
        with open(MODEL_DIR / "selected_features.txt") as f:
            m["pd_features"] = [l.strip() for l in f if l.strip()]
        with open(MODEL_DIR / "lgd_features.txt") as f:
            m["lgd_features"] = [l.strip() for l in f if l.strip()]
        return m
    except Exception:
        return None

def assign_rating(pd_val):
    thresholds = [(0.01,1,"Minimal"),(0.03,2,"Low"),(0.05,3,"Low-Mod"),(0.08,4,"Moderate"),
                  (0.12,5,"Mod-High"),(0.18,6,"High"),(0.25,7,"Very High"),(0.35,8,"Substandard"),
                  (0.50,9,"Doubtful")]
    for t, r, l in thresholds:
        if pd_val < t:
            return r, l
    return 10, "Loss"


def render():
    st.title("Loan Scorer: Real-Time Risk Assessment")
    info_box("<strong>Production simulation:</strong> Enter borrower/loan details to score through the PD and LGD models, "
             "producing a risk rating and day-one ECL reserve.")

    section_header("Borrower and Loan Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Borrower**")
        fico = st.slider("FICO Score", 500, 850, 720)
        dti = st.slider("DTI (%)", 10, 65, 35)
        coborrower = st.selectbox("Co-borrower?", ["No","Yes"], key="ls_cb")
    with c2:
        st.markdown("**Loan**")
        loan_amt = st.number_input("Loan Amount ($)", 50000, 2000000, 250000, 10000)
        prop_val = st.number_input("Property Value ($)", 50000, 3000000, 320000, 10000)
        rate = st.slider("Interest Rate (%)", 2.0, 9.0, 6.5, 0.125)
        term = st.selectbox("Term (months)", [360, 180, 240], key="ls_term")
    with c3:
        st.markdown("**Property & Macro**")
        occ = st.selectbox("Occupancy", ["Primary","Investment","Second Home"], key="ls_occ")
        purpose = st.selectbox("Purpose", ["Purchase","Cash-Out Refi","Rate/Term Refi"], key="ls_purp")
        has_mi = st.selectbox("Mortgage Insurance?", ["No","Yes"], key="ls_mi")
        unemp = st.slider("Unemployment (%)", 3.0, 12.0, 4.3, 0.1)

    ltv = min(loan_amt / prop_val * 100, 200) if prop_val > 0 else 100

    if st.button("Score This Loan", type="primary", use_container_width=True):
        section_header("Scoring Results")

        models = load_models()
        used_models = False

        if models is not None:
            try:
                from pd_model import apply_woe_transformation
                loan_data = pd.DataFrame([{
                    "borrower_credit_score": fico, "dti": dti,
                    "has_coborrower": 1 if coborrower=="Yes" else 0,
                    "original_ltv": ltv, "original_cltv": ltv,
                    "has_mortgage_insurance": 1 if has_mi=="Yes" else 0,
                    "original_interest_rate": rate, "original_loan_term": term,
                    "is_cashout_refi": 1 if purpose=="Cash-Out Refi" else 0,
                    "fico_x_unemployment": fico*unemp,
                    "unemployment_rate": unemp, "hpi_national": 180.0,
                    "original_upb": loan_amt, "loan_age_at_default": 48.0,
                    "was_modified": 0,
                    "is_investment_property": 1 if occ=="Investment" else 0,
                    "is_condo": 0,
                }])
                X_woe = apply_woe_transformation(loan_data, models["woe_results"], models["pd_features"])
                pd_score = models["pd_model"].predict_proba(X_woe)[0, 1]
                lgd_cols = models["lgd_features"]
                X_lgd = pd.DataFrame(columns=lgd_cols)
                for col in lgd_cols:
                    X_lgd[col] = [loan_data[col].iloc[0] if col in loan_data.columns else 0.0]
                lgd_score = float(np.clip(models["lgd_model"].predict(X_lgd)[0], 0.0, 1.0))
                used_models = True
            except Exception:
                used_models = False

        if not used_models:
            base_pd = 0.12
            pd_score = np.clip(base_pd + (720-fico)*0.002 + (ltv-75)*0.001 + (dti-35)*0.001
                + (unemp-4.3)*0.02 + (0.04 if occ=="Investment" else 0)
                + (0.02 if purpose=="Cash-Out Refi" else 0), 0.005, 0.95)
            lgd_score = np.clip(0.35 + (ltv-75)*0.003 + (-0.12 if has_mi=="Yes" else 0), 0.05, 0.95)

        lifetime = min(term/12.0, 5.0)
        ecl = pd_score * lgd_score * loan_amt * lifetime
        rating, label = assign_rating(pd_score)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("12-Month PD", f"{pd_score*100:.2f}%")
        r2.metric("LGD", f"{lgd_score*100:.1f}%")
        r3.metric("Lifetime ECL", f"${ecl:,.0f}")
        r4.metric("Risk Rating", f"{rating}/10 - {label}")

        if not used_models:
            warning_box("Models not loaded from disk. Using simplified formulas for demo. "
                        "In production, trained .pkl models are loaded from models/ directory.")

        gc, gb = st.columns(2)
        with gc:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=rating,
                title={"text": f"Risk Rating: {label}"},
                gauge={"axis":{"range":[1,10],"tickvals":list(range(1,11))}, "bar":{"color":COLORS["primary"]},
                       "steps":[{"range":[1,3],"color":"#f0fdf4"},{"range":[3,5],"color":"#fefce8"},
                                {"range":[5,7],"color":"#fff7ed"},{"range":[7,10],"color":"#fef2f2"}]}))
            fig.update_layout(height=300, paper_bgcolor="#ffffff", font=dict(color="#1e293b"))
            st.plotly_chart(fig, use_container_width=True)
        with gb:
            summary = pd.DataFrame({
                "Attribute": ["FICO","LTV","DTI","Loan Amount","Rate","PD","LGD","ECL","ECL %","Rating"],
                "Value": [f"{fico}",f"{ltv:.1f}%",f"{dti}%",f"${loan_amt:,}",f"{rate}%",
                    f"{pd_score*100:.2f}%",f"{lgd_score*100:.1f}%",f"${ecl:,.0f}",
                    f"{ecl/loan_amt*100:.2f}%",f"{rating}/10 ({label})"],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        info_box("In production, this score is stored in the loan origination system, "
                 "ECL is booked to the allowance, and the rating determines regulatory capital treatment.")