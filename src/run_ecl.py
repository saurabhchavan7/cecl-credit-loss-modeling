"""
Full ECL Calculation Pipeline (Phases 5 + 6)
=============================================

Scores the full portfolio with trained PD and LGD models, then
computes lifetime CECL Expected Credit Loss.

Steps:
1. Load portfolio and trained models
2. Score PD for every loan (logistic regression)
3. Score LGD for every loan (OLS model)
4. Compute ECL under baseline scenario
5. Compute ECL under stressed scenario (PD * stress multiplier)
6. Compute scenario-weighted ECL
7. Segment-level ECL breakdown
8. Save results

Author: Saurabh Chavan
"""

import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pd_model import apply_woe_transformation
from lgd_model import LGD_FEATURES
from ecl_engine import compute_portfolio_ecl, compute_scenario_weighted_ecl

COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MODEL_DIR = project_root / "models"


def main():
    t_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - FULL ECL CALCULATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load portfolio and models
    # ------------------------------------------------------------------
    print("\nStep 1: Loading portfolio and trained models...")

    df = pd.read_parquet(COMBINED_PATH)
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"
    print(f"  Portfolio: {len(df):,} loans")

    # Load PD model and WoE mappings
    lr_model = joblib.load(MODEL_DIR / "pd_logistic_regression.pkl")
    woe_results = joblib.load(MODEL_DIR / "woe_results.pkl")
    print(f"  Loaded PD logistic regression model")

    # Load LGD model
    lgd_model = joblib.load(MODEL_DIR / "lgd_ols.pkl")
    print(f"  Loaded LGD OLS model")

    # Load feature lists
    with open(MODEL_DIR / "selected_features.txt") as f:
        pd_features = [line.strip() for line in f if line.strip()]
    with open(MODEL_DIR / "lgd_features.txt") as f:
        lgd_features = [line.strip() for line in f if line.strip()]

    print(f"  PD features: {len(pd_features)}")
    print(f"  LGD features: {len(lgd_features)}")

    # ------------------------------------------------------------------
    # Step 2: Score PD for every loan
    # ------------------------------------------------------------------
    print("\nStep 2: Scoring PD for all loans...")

    X_woe = apply_woe_transformation(df, woe_results, pd_features)
    pd_predictions = lr_model.predict_proba(X_woe)[:, 1]

    print(f"  PD predictions: min={pd_predictions.min():.4f}, "
          f"mean={pd_predictions.mean():.4f}, max={pd_predictions.max():.4f}")

    del X_woe
    gc.collect()

    # ------------------------------------------------------------------
    # Step 3: Score LGD for every loan
    # ------------------------------------------------------------------
    print("\nStep 3: Scoring LGD for all loans...")

    # Prepare LGD features. For non-defaulted loans, loan_age_at_default
    # and was_modified are not available. We use median values from
    # the training population of defaulted loans.
    X_lgd = df[lgd_features].copy()

    # Fill missing values with reasonable defaults
    lgd_fill_values = {
        "loan_age_at_default": 48.0,   # ~4 years, typical default timing
        "was_modified": 0.0,           # assume not modified
    }
    for col in lgd_features:
        if col in lgd_fill_values:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill_values[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())

    lgd_predictions = lgd_model.predict(X_lgd)
    lgd_predictions = np.clip(lgd_predictions, 0.0, 1.0)

    print(f"  LGD predictions: min={lgd_predictions.min():.4f}, "
          f"mean={lgd_predictions.mean():.4f}, max={lgd_predictions.max():.4f}")

    del X_lgd
    gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Compute baseline ECL
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Computing baseline ECL")
    print("=" * 70)

    baseline_results, baseline_summary = compute_portfolio_ecl(
        df, pd_predictions, lgd_predictions, "Baseline"
    )

    # ------------------------------------------------------------------
    # Step 5: Compute stressed ECL
    # ------------------------------------------------------------------
    # Stress scenario: multiply PD by a stress factor to simulate
    # economic deterioration. In a full implementation, we would use
    # the Fed's macro scenario paths to re-score the PD model under
    # stressed macro conditions. Here we apply a multiplier approach
    # which is a common simplified methodology.
    #
    # Stress multiplier of 2.5x represents a severe recession:
    # - 12% baseline PD becomes 30% stressed PD
    # - Consistent with 2008-crisis default rate increases
    # LGD stress: multiply by 1.3 (foreclosure costs rise in recession)
    print("\n" + "=" * 70)
    print("Step 5: Computing severely adverse ECL")
    print("=" * 70)

    pd_stressed = np.clip(pd_predictions * 2.5, 0.0, 0.99)
    lgd_stressed = np.clip(lgd_predictions * 1.3, 0.0, 1.0)

    stressed_results, stressed_summary = compute_portfolio_ecl(
        df, pd_stressed, lgd_stressed, "Severely Adverse"
    )

    # ------------------------------------------------------------------
    # Step 6: Scenario-weighted ECL
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 6: Scenario-weighted ECL")
    print("=" * 70)

    weighted = compute_scenario_weighted_ecl(
        [baseline_summary, stressed_summary],
        {"Baseline": 0.60, "Severely Adverse": 0.40},
    )

    # ------------------------------------------------------------------
    # Step 7: Segment-level ECL breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 7: ECL by segment")
    print("=" * 70)

    # Add ECL results back to portfolio
    df["pd_12m"] = pd_predictions
    df["lgd_predicted"] = lgd_predictions
    df["ecl_dollars"] = baseline_results["ecl_dollars"].values
    df["current_balance"] = baseline_results["current_balance"].values

    # By FICO bucket
    print("\n  ECL by FICO Bucket (Baseline):")
    print(f"  {'FICO':<12s} {'Loans':>10s} {'Balance ($B)':>14s} {'ECL ($M)':>12s} {'ECL Rate':>10s} {'Avg PD':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*12} {'-'*10} {'-'*8}")
    for bucket in ["<620", "620-660", "660-700", "700-740", "740+"]:
        mask = df["fico_bucket"] == bucket
        if mask.sum() > 0:
            seg = df[mask]
            bal = seg["current_balance"].sum()
            ecl = seg["ecl_dollars"].sum()
            rate = ecl / bal if bal > 0 else 0
            avg_pd = seg["pd_12m"].mean()
            print(f"  {bucket:<12s} {mask.sum():>10,} ${bal/1e9:>13.2f} ${ecl/1e6:>11.1f} {rate*100:>9.2f}% {avg_pd*100:>7.2f}%")

    # By LTV bucket
    print("\n  ECL by LTV Bucket (Baseline):")
    print(f"  {'LTV':<12s} {'Loans':>10s} {'Balance ($B)':>14s} {'ECL ($M)':>12s} {'ECL Rate':>10s} {'Avg LGD':>9s}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*12} {'-'*10} {'-'*9}")
    for bucket in ["<60", "60-70", "70-80", "80-90", "90+"]:
        mask = df["ltv_bucket"] == bucket
        if mask.sum() > 0:
            seg = df[mask]
            bal = seg["current_balance"].sum()
            ecl = seg["ecl_dollars"].sum()
            rate = ecl / bal if bal > 0 else 0
            avg_lgd = seg["lgd_predicted"].mean()
            print(f"  {bucket:<12s} {mask.sum():>10,} ${bal/1e9:>13.2f} ${ecl/1e6:>11.1f} {rate*100:>9.2f}% {avg_lgd*100:>8.2f}%")

    # By vintage year
    print("\n  ECL by Origination Year (Baseline):")
    print(f"  {'Year':<8s} {'Loans':>10s} {'Balance ($B)':>14s} {'ECL ($M)':>12s} {'ECL Rate':>10s}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*12} {'-'*10}")
    for year in sorted(df["origination_year"].unique()):
        if year < 2005:
            continue
        mask = df["origination_year"] == year
        seg = df[mask]
        bal = seg["current_balance"].sum()
        ecl = seg["ecl_dollars"].sum()
        rate = ecl / bal if bal > 0 else 0
        print(f"  {int(year):<8d} {mask.sum():>10,} ${bal/1e9:>13.2f} ${ecl/1e6:>11.1f} {rate*100:>9.2f}%")

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 8: Saving results")
    print(f"{'='*70}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save portfolio-level summary
    ecl_summary = pd.DataFrame([baseline_summary, stressed_summary])
    ecl_summary.loc[len(ecl_summary)] = {
        "scenario": "Weighted (60/40)",
        "n_loans": baseline_summary["n_loans"],
        "total_balance": weighted["total_balance"],
        "total_ecl": weighted["weighted_ecl"],
        "portfolio_ecl_rate": weighted["weighted_ecl_rate"],
        "mean_pd": baseline_summary["mean_pd"],
        "mean_lgd": baseline_summary["mean_lgd"],
        "mean_loan_ecl": np.nan,
        "median_loan_ecl": np.nan,
    }
    ecl_summary.to_csv(MODEL_DIR / "ecl_summary.csv", index=False)

    # Save loan-level baseline ECL (for dashboard)
    ecl_output_cols = [
        "origination_year", "fico_bucket", "ltv_bucket",
        "original_upb", "original_interest_rate", "original_loan_term",
        "borrower_credit_score", "original_ltv", "dti",
        "default_flag", "pd_12m", "lgd_predicted",
        "ecl_dollars", "current_balance",
    ]
    available_cols = [c for c in ecl_output_cols if c in df.columns]
    df[available_cols].to_parquet(MODEL_DIR / "ecl_loan_level.parquet")

    print(f"  Saved ECL summary to {MODEL_DIR / 'ecl_summary.csv'}")
    print(f"  Saved loan-level ECL to {MODEL_DIR / 'ecl_loan_level.parquet'}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ECL CALCULATION COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")

    # Final headline numbers
    print(f"\n  HEADLINE RESULTS:")
    print(f"  Portfolio: {baseline_summary['n_loans']:,} loans, "
          f"${baseline_summary['total_balance']/1e9:.1f}B outstanding")
    print(f"  Baseline ECL:          ${baseline_summary['total_ecl']/1e6:,.0f}M "
          f"({baseline_summary['portfolio_ecl_rate']*100:.2f}%)")
    print(f"  Severely Adverse ECL:  ${stressed_summary['total_ecl']/1e6:,.0f}M "
          f"({stressed_summary['portfolio_ecl_rate']*100:.2f}%)")
    print(f"  Weighted ECL (60/40):  ${weighted['weighted_ecl']/1e6:,.0f}M "
          f"({weighted['weighted_ecl_rate']*100:.2f}%)")


if __name__ == "__main__":
    main()