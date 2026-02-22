"""
Generate small summary CSVs from the large parquet file for dashboard deployment.

Run this ONCE locally. It reads ecl_loan_level.parquet and produces small CSV
files that the dashboard can use on Streamlit Cloud where the parquet is too
large to push to GitHub.

Author: Saurabh Chavan
"""

import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"


def main():
    print("Loading ecl_loan_level.parquet...")
    df = pd.read_parquet(MODEL_DIR / "ecl_loan_level.parquet")
    print(f"  Loaded {len(df):,} loans")

    # --- 1. FICO bucket summary ---
    fico = df.groupby("fico_bucket", observed=True).agg(
        count=("default_flag", "count"),
        defaults=("default_flag", "sum"),
        balance=("current_balance", "sum"),
        ecl=("ecl_dollars", "sum"),
        avg_pd=("pd_12m", "mean"),
        avg_lgd=("lgd_predicted", "mean"),
    ).reset_index()
    fico["default_rate"] = fico["defaults"] / fico["count"]
    fico["ecl_rate"] = fico["ecl"] / fico["balance"]
    fico.to_csv(MODEL_DIR / "dashboard_fico_summary.csv", index=False)
    print(f"  Saved dashboard_fico_summary.csv ({len(fico)} rows)")

    # --- 2. LTV bucket summary ---
    ltv = df.groupby("ltv_bucket", observed=True).agg(
        count=("default_flag", "count"),
        defaults=("default_flag", "sum"),
        balance=("current_balance", "sum"),
        ecl=("ecl_dollars", "sum"),
        avg_pd=("pd_12m", "mean"),
        avg_lgd=("lgd_predicted", "mean"),
    ).reset_index()
    ltv["default_rate"] = ltv["defaults"] / ltv["count"]
    ltv["ecl_rate"] = ltv["ecl"] / ltv["balance"]
    ltv.to_csv(MODEL_DIR / "dashboard_ltv_summary.csv", index=False)
    print(f"  Saved dashboard_ltv_summary.csv ({len(ltv)} rows)")

    # --- 3. Vintage year summary ---
    vintage = df.groupby("origination_year").agg(
        count=("default_flag", "count"),
        defaults=("default_flag", "sum"),
        balance=("current_balance", "sum"),
        ecl=("ecl_dollars", "sum"),
        avg_pd=("pd_12m", "mean"),
    ).reset_index()
    vintage = vintage[vintage["origination_year"] >= 2004]
    vintage["default_rate"] = vintage["defaults"] / vintage["count"]
    vintage["ecl_rate"] = vintage["ecl"] / vintage["balance"]
    vintage.to_csv(MODEL_DIR / "dashboard_vintage_summary.csv", index=False)
    print(f"  Saved dashboard_vintage_summary.csv ({len(vintage)} rows)")

    # --- 4. LGD segment summary (defaulted loans only) ---
    defaulted = df[df["default_flag"] == 1]
    if "lgd_predicted" in defaulted.columns:
        for seg_col in ["fico_bucket", "ltv_bucket"]:
            seg = defaulted.groupby(seg_col, observed=True).agg(
                count=("lgd_predicted", "count"),
                mean_lgd=("lgd_predicted", "mean"),
            ).reset_index()
            seg.to_csv(MODEL_DIR / f"dashboard_lgd_by_{seg_col}.csv", index=False)
            print(f"  Saved dashboard_lgd_by_{seg_col}.csv")

    # --- 5. Portfolio totals ---
    totals = {
        "total_loans": len(df),
        "total_balance": df["current_balance"].sum(),
        "default_rate": df["default_flag"].mean(),
        "mean_fico": df["borrower_credit_score"].mean(),
        "mean_pd": df["pd_12m"].mean(),
        "mean_lgd": df["lgd_predicted"].mean() if "lgd_predicted" in df.columns else 0,
    }
    pd.DataFrame([totals]).to_csv(MODEL_DIR / "dashboard_portfolio_totals.csv", index=False)
    print(f"  Saved dashboard_portfolio_totals.csv")

    print("\nDone. These small CSVs replace ecl_loan_level.parquet for deployment.")
    print("Total size of new CSVs: < 10KB")


if __name__ == "__main__":
    main()