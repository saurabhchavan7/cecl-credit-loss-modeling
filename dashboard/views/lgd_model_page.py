"""
LGD Model Page
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
def load_lgd_validation():
    p = MODEL_DIR / "lgd_validation_summary.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None

@st.cache_data
def load_lgd_coefficients():
    p = MODEL_DIR / "lgd_ols_coefficients.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_ecl_loan_level():
    p = MODEL_DIR / "ecl_loan_level.parquet"
    return pd.read_parquet(p) if p.exists() else None


def render():
    st.title("LGD Model: Loss Given Default")
    info_box("<strong>LGD</strong> measures the fraction of exposure lost when a borrower defaults. "
             "An LGD of 0.40 = bank loses 40 cents per dollar. Second component of <strong>ECL = PD x LGD x EAD</strong>.")

    section_header("LGD Model Performance")
    vd = load_lgd_validation()
    k1, k2, k3, k4 = st.columns(4)
    if vd is not None:
        k1.metric("OLS Val R-sq", f"{vd.loc['val_r2','ols']:.4f}")
        k2.metric("XGB Val R-sq", f"{vd.loc['val_r2','xgboost']:.4f}", f"+{vd.loc['val_r2','xgboost']-vd.loc['val_r2','ols']:.4f}")
        k3.metric("OLS Val RMSE", f"{vd.loc['val_rmse','ols']:.4f}")
    else:
        k1.metric("OLS Val R-sq", "0.1508")
        k2.metric("XGB Val R-sq", "0.2465", "+0.0957")
        k3.metric("OLS Val RMSE", "0.2800")
    k4.metric("Training Population", "278,220 defaults")

    info_box("<strong>Is R-sq of 0.15 acceptable?</strong> Yes. LGD has high inherent variance from "
             "property-specific factors, foreclosure timelines, and market timing. Literature reports 0.10-0.25. "
             "What matters more: OLS calibration ratio = 1.05 (5% conservative bias, acceptable).")

    tab_cmp, tab_coef, tab_seg, tab_meth = st.tabs(["Model Comparison", "OLS Coefficients", "Segment Analysis", "Methodology"])

    with tab_cmp:
        cc, ce = st.columns([2, 1])
        with cc:
            if vd is not None:
                labels = ["Train R2", "Val R2", "Test R2", "Train RMSE", "Val RMSE", "Test RMSE"]
                ols = [vd.loc[m, "ols"] for m in ["train_r2","val_r2","test_r2","train_rmse","val_rmse","test_rmse"]]
                xgb = [vd.loc[m, "xgboost"] for m in ["train_r2","val_r2","test_r2","train_rmse","val_rmse","test_rmse"]]
            else:
                labels = ["Train R2", "Val R2", "Test R2", "Train RMSE", "Val RMSE", "Test RMSE"]
                ols = [0.1702, 0.1508, 0.1574, 0.27, 0.28, 0.28]
                xgb = [0.2985, 0.2465, 0.2401, 0.25, 0.26, 0.27]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=ols, name="OLS", marker_color=COLORS["primary"],
                text=[f"{v:.4f}" for v in ols], textposition="outside"))
            fig.add_trace(go.Bar(x=labels, y=xgb, name="XGBoost", marker_color=COLORS["success"],
                text=[f"{v:.4f}" for v in xgb], textposition="outside"))
            fig.update_layout(title="LGD: OLS vs XGBoost", barmode="group", bargap=0.2)
            style_chart(fig, 440)
            st.plotly_chart(fig, use_container_width=True)
        with ce:
            info_box("XGBoost captures ~10 more R-sq points through non-linear splits. "
                     "RMSE of 0.28 = predictions off ~28pp for individual loans, but portfolio-level mean is well-calibrated.")

    with tab_coef:
        cc, ce = st.columns([2, 1])
        with cc:
            coef = load_lgd_coefficients()
            if coef is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=coef["coefficient"], y=coef["feature"], orientation="h",
                    marker_color=[COLORS["danger"] if c > 0 else COLORS["success"] for c in coef["coefficient"]],
                    text=[f"{v:+.6f}" for v in coef["coefficient"]], textposition="outside"))
                fig.add_vline(x=0, line_color=COLORS["gray"])
                fig.update_layout(title="OLS LGD Coefficients", xaxis_title="Coefficient",
                    yaxis=dict(autorange="reversed"), showlegend=False)
                style_chart(fig, 480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("LGD coefficients not found.")
        with ce:
            info_box("<strong>Red (positive):</strong> increases LGD (more loss). Higher LTV = less equity buffer.<br>"
                     "<strong>Green (negative):</strong> decreases LGD. Mortgage insurance reduces loss by ~15pp.")

    with tab_seg:
        seg_col = st.selectbox("Segment by:", ["fico_bucket", "ltv_bucket"], key="lgd_seg")

        # Try parquet first (local), then small CSV fallback (deployed)
        data = None
        loan_df = load_ecl_loan_level()

        if loan_df is not None and "lgd_predicted" in loan_df.columns:
            data = (
                loan_df[loan_df["default_flag"] == 1]
                .groupby(seg_col, observed=True)
                .agg(mean_lgd=("lgd_predicted", "mean"), count=("lgd_predicted", "count"))
                .reset_index()
            )
        else:
            csv_path = MODEL_DIR / f"dashboard_lgd_by_{seg_col}.csv"
            if csv_path.exists():
                data = pd.read_csv(csv_path)

        if data is not None and len(data) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data[seg_col], y=data["mean_lgd"] * 100,
                marker_color=COLORS["primary"],
                text=[f"{v*100:.1f}%" for v in data["mean_lgd"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Predicted LGD by {seg_col}",
                xaxis_title=seg_col,
                yaxis_title="Mean LGD (%)",
                showlegend=False,
                bargap=0.3,
            )
            style_chart(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

            info_box("Higher-risk segments (low FICO, high LTV) have higher predicted LGD, "
                     "confirming the model produces economically sensible predictions.")
        else:
            st.info("Segment data not found. Run `generate_dashboard_data.py` to create summary CSVs.")

    with tab_meth:
        m1, m2 = st.columns(2)
        with m1:
            info_box("<strong>LGD Formula:</strong><br>Total Loss = EAD - Recovery + Costs<br>LGD = Total Loss / EAD<br><br>"
                     "EAD from second-to-last observation (Fannie Mae sets UPB=0 in final record).")
        with m2:
            info_box("<strong>Why OLS over Beta regression?</strong> Observed LGD goes up to 1.5 (costs exceeding balance). "
                     "Beta regression requires (0,1). OLS with clipping is the pragmatic industry approach.")
        warning_box("<strong>Data note:</strong> net_sale_proceeds='C' (confidential) treated as NaN, not zero. "
                    "Treating as zero would artificially inflate LGD.")