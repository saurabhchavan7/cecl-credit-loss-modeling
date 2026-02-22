"""
Full Stress Testing Pipeline (Phase 7)
=======================================

Runs Federal Reserve 2025 stress test scenarios on the portfolio.

Approach: Macro-sensitivity overlay method.
Rather than substituting scenario macro values into origination-time features
(which produces inverted results because origination-time HPI/unemployment
have opposite economic meaning to point-in-time values), we use a scalar
approach:

1. Score the portfolio once under current (baseline) conditions to get
   baseline PD and LGD for each loan.
2. For each stressed quarter, compute stress MULTIPLIERS based on how
   much macro variables deviate from baseline:
   - PD multiplier: driven by unemployment change (higher unemployment
     = more defaults)
   - LGD multiplier: driven by HPI change (lower home prices = higher
     loss severity)
3. Apply multipliers to baseline PD and LGD to get stressed estimates.

This is the standard "scalar" stress testing approach used at large banks
when the underlying models use origination-time features.

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

from stress_testing import (
    load_fed_scenarios,
    compute_segment_stress_results,
)
from pd_model import apply_woe_transformation

COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
SCENARIOS_DIR = project_root / "data" / "scenarios"
MODEL_DIR = project_root / "models"


def compute_stress_multipliers(scenario_df, baseline_scenario_df):
    """
    Compute PD and LGD stress multipliers for each quarter based on
    macro variable deviations from baseline.

    PD multiplier logic:
      Each +1% unemployment above baseline increases PD by ~25%.
      This is calibrated to the empirical relationship observed during
      2008 crisis where unemployment rose ~5% and default rates roughly
      doubled.

    LGD multiplier logic:
      Each -10% HPI decline from baseline increases LGD by ~15%.
      Calibrated to the 2008 experience where ~33% HPI decline
      roughly doubled loss severity.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        Stressed scenario quarterly path.
    baseline_scenario_df : pd.DataFrame
        Baseline scenario quarterly path.

    Returns
    -------
    pd.DataFrame
        Quarterly multipliers with columns: quarter, pd_multiplier,
        lgd_multiplier, unemployment_rate, house_price_index.
    """
    results = []

    for idx in range(len(scenario_df)):
        quarter = scenario_df["quarter"].iloc[idx]

        # Get baseline values for this quarter
        if idx < len(baseline_scenario_df):
            base_ur = baseline_scenario_df["unemployment_rate"].iloc[idx]
            base_hpi = baseline_scenario_df["house_price_index"].iloc[idx]
        else:
            base_ur = baseline_scenario_df["unemployment_rate"].iloc[-1]
            base_hpi = baseline_scenario_df["house_price_index"].iloc[-1]

        stress_ur = scenario_df["unemployment_rate"].iloc[idx]
        stress_hpi = scenario_df["house_price_index"].iloc[idx]

        # PD multiplier: +25% per +1% unemployment above baseline
        ur_delta = stress_ur - base_ur
        pd_multiplier = 1.0 + 0.25 * max(ur_delta, 0)
        # Also add a small increase for negative GDP growth
        if "real_gdp_growth" in scenario_df.columns:
            gdp = scenario_df["real_gdp_growth"].iloc[idx]
            if gdp < 0:
                pd_multiplier += abs(gdp) * 0.05  # +5% PD per -1% GDP

        # LGD multiplier: +15% per -10% HPI decline from baseline
        hpi_change_pct = (stress_hpi - base_hpi) / base_hpi * 100
        lgd_multiplier = 1.0 + 0.15 * max(-hpi_change_pct / 10.0, 0)

        results.append({
            "quarter": quarter,
            "unemployment_rate": stress_ur,
            "house_price_index": stress_hpi,
            "ur_delta": ur_delta,
            "hpi_change_pct": hpi_change_pct,
            "pd_multiplier": pd_multiplier,
            "lgd_multiplier": lgd_multiplier,
        })

    return pd.DataFrame(results)


def main():
    t_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - STRESS TESTING (Phase 7)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load data and models
    # ------------------------------------------------------------------
    print("\nStep 1: Loading portfolio, models, and scenarios...")

    df = pd.read_parquet(COMBINED_PATH)
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"
    total_balance = df["original_upb"].sum()
    print(f"  Portfolio: {len(df):,} loans, ${total_balance/1e9:.1f}B")

    pd_model = joblib.load(MODEL_DIR / "pd_logistic_regression.pkl")
    woe_results = joblib.load(MODEL_DIR / "woe_results.pkl")
    lgd_model = joblib.load(MODEL_DIR / "lgd_ols.pkl")

    with open(MODEL_DIR / "selected_features.txt") as f:
        pd_features = [line.strip() for line in f if line.strip()]
    with open(MODEL_DIR / "lgd_features.txt") as f:
        lgd_features = [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------
    # Step 2: Load Fed scenarios
    # ------------------------------------------------------------------
    print("\nStep 2: Loading Federal Reserve 2025 scenarios...")
    scenarios = load_fed_scenarios(SCENARIOS_DIR)

    if "Baseline" not in scenarios or "Severely Adverse" not in scenarios:
        print("  ERROR: Need both Baseline and Severely Adverse scenarios.")
        sys.exit(1)

    baseline_scen = scenarios["Baseline"]
    adverse_scen = scenarios["Severely Adverse"]

    print(f"\n  Severely Adverse scenario highlights:")
    print(f"    Unemployment: {adverse_scen['unemployment_rate'].iloc[0]:.1f}% "
          f"-> peak {adverse_scen['unemployment_rate'].max():.1f}%")
    hpi_start = adverse_scen["house_price_index"].iloc[0]
    hpi_trough = adverse_scen["house_price_index"].min()
    print(f"    HPI: {hpi_start:.0f} -> trough {hpi_trough:.0f} "
          f"({(hpi_trough/baseline_scen['house_price_index'].iloc[0]-1)*100:+.1f}% from baseline)")

    # ------------------------------------------------------------------
    # Step 3: Score baseline PD and LGD
    # ------------------------------------------------------------------
    print("\nStep 3: Scoring baseline PD and LGD...")

    X_woe = apply_woe_transformation(df, woe_results, pd_features)
    pd_baseline = pd_model.predict_proba(X_woe)[:, 1]
    del X_woe
    gc.collect()

    X_lgd = df[lgd_features].copy()
    lgd_fill = {"loan_age_at_default": 48.0, "was_modified": 0.0}
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())
    lgd_baseline = lgd_model.predict(X_lgd)
    lgd_baseline = np.clip(lgd_baseline, 0.0, 1.0)
    del X_lgd
    gc.collect()

    print(f"  Baseline PD:  mean={pd_baseline.mean()*100:.2f}%, "
          f"min={pd_baseline.min()*100:.2f}%, max={pd_baseline.max()*100:.2f}%")
    print(f"  Baseline LGD: mean={lgd_baseline.mean()*100:.2f}%, "
          f"min={lgd_baseline.min()*100:.2f}%, max={lgd_baseline.max()*100:.2f}%")

    # ------------------------------------------------------------------
    # Step 4: Compute stress multipliers
    # ------------------------------------------------------------------
    print("\nStep 4: Computing stress multipliers...")

    multipliers = compute_stress_multipliers(adverse_scen, baseline_scen)

    print(f"\n  Quarterly Stress Multipliers:")
    print(f"  {'Quarter':<10s} {'Unemp':>7s} {'UR Delta':>9s} "
          f"{'HPI':>7s} {'HPI Chg':>8s} {'PD Mult':>8s} {'LGD Mult':>9s}")
    print(f"  {'-'*10} {'-'*7} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")
    for _, row in multipliers.iterrows():
        print(f"  {row['quarter']:<10s} {row['unemployment_rate']:>6.1f}% "
              f"{row['ur_delta']:>+8.1f}% {row['house_price_index']:>7.0f} "
              f"{row['hpi_change_pct']:>+7.1f}% {row['pd_multiplier']:>7.2f}x "
              f"{row['lgd_multiplier']:>8.2f}x")

    # ------------------------------------------------------------------
    # Step 5: Run quarterly stress path
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 5: Quarterly loss path under each scenario")
    print(f"{'='*70}")

    upb = df["original_upb"].values

    # Baseline path (no multipliers, constant)
    baseline_el_annual = pd_baseline * lgd_baseline * upb
    baseline_quarterly_el = baseline_el_annual.sum() / 4.0

    print(f"\n  Baseline: constant quarterly loss = ${baseline_quarterly_el/1e6:,.0f}M")
    print(f"  Baseline 13-quarter total = ${baseline_quarterly_el*13/1e6:,.0f}M "
          f"({baseline_quarterly_el*13/total_balance*100:.2f}%)")

    # Severely adverse path
    print(f"\n  Severely Adverse Quarterly Path:")
    print(f"  {'Quarter':<10s} {'PD Mult':>8s} {'LGD Mult':>9s} {'Mean PD':>8s} "
          f"{'Mean LGD':>9s} {'Q Loss($M)':>12s} {'Cum Loss($M)':>14s}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*12} {'-'*14}")

    cum_loss = 0.0
    quarterly_results = []

    for _, mrow in multipliers.iterrows():
        pd_stressed = np.clip(pd_baseline * mrow["pd_multiplier"], 0.0, 0.99)
        lgd_stressed = np.clip(lgd_baseline * mrow["lgd_multiplier"], 0.0, 1.0)

        quarterly_el = (pd_stressed * lgd_stressed * upb).sum() / 4.0
        cum_loss += quarterly_el

        mean_pd = pd_stressed.mean()
        mean_lgd = lgd_stressed.mean()

        print(f"  {mrow['quarter']:<10s} {mrow['pd_multiplier']:>7.2f}x "
              f"{mrow['lgd_multiplier']:>8.2f}x {mean_pd*100:>7.2f}% "
              f"{mean_lgd*100:>8.2f}% ${quarterly_el/1e6:>11,.0f} "
              f"${cum_loss/1e6:>13,.0f}")

        quarterly_results.append({
            "quarter": mrow["quarter"],
            "pd_multiplier": mrow["pd_multiplier"],
            "lgd_multiplier": mrow["lgd_multiplier"],
            "mean_pd": mean_pd,
            "mean_lgd": mean_lgd,
            "quarterly_el": quarterly_el,
            "cumulative_el": cum_loss,
        })

    adverse_total = cum_loss
    adverse_rate = adverse_total / total_balance

    # ------------------------------------------------------------------
    # Step 6: Segment-level stress analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 6: Segment-level stress analysis (peak stress quarter)")
    print(f"{'='*70}")

    # Use peak stress quarter multipliers
    peak_idx = multipliers["pd_multiplier"].idxmax()
    peak_pd_mult = multipliers.loc[peak_idx, "pd_multiplier"]
    peak_lgd_mult = multipliers.loc[peak_idx, "lgd_multiplier"]
    peak_quarter = multipliers.loc[peak_idx, "quarter"]

    print(f"  Peak stress quarter: {peak_quarter}")
    print(f"  PD multiplier: {peak_pd_mult:.2f}x, LGD multiplier: {peak_lgd_mult:.2f}x")

    pd_peak = np.clip(pd_baseline * peak_pd_mult, 0.0, 0.99)
    lgd_peak = np.clip(lgd_baseline * peak_lgd_mult, 0.0, 1.0)

    print(f"  Baseline PD: {pd_baseline.mean()*100:.2f}% -> "
          f"Stressed PD: {pd_peak.mean()*100:.2f}%")
    print(f"  Baseline LGD: {lgd_baseline.mean()*100:.2f}% -> "
          f"Stressed LGD: {lgd_peak.mean()*100:.2f}%")

    compute_segment_stress_results(
        df, pd_baseline, lgd_baseline, pd_peak, lgd_peak,
        "fico_bucket", "FICO Bucket",
    )
    compute_segment_stress_results(
        df, pd_baseline, lgd_baseline, pd_peak, lgd_peak,
        "ltv_bucket", "LTV Bucket",
    )

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 7: Saving stress test results")
    print(f"{'='*70}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(quarterly_results).to_csv(
        MODEL_DIR / "stress_severely_adverse_quarterly.csv", index=False
    )

    summary = pd.DataFrame([
        {
            "scenario": "Baseline",
            "total_loss": baseline_quarterly_el * 13,
            "loss_rate": baseline_quarterly_el * 13 / total_balance,
            "total_balance": total_balance,
        },
        {
            "scenario": "Severely Adverse",
            "total_loss": adverse_total,
            "loss_rate": adverse_rate,
            "total_balance": total_balance,
        },
    ])
    summary.to_csv(MODEL_DIR / "stress_test_summary.csv", index=False)
    print(f"  Saved results to {MODEL_DIR}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    baseline_total = baseline_quarterly_el * 13
    stress_increment = adverse_total - baseline_total

    print(f"\n{'='*70}")
    print("STRESS TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Portfolio:        {len(df):,} loans, ${total_balance/1e9:.1f}B outstanding")
    print(f"  Baseline Loss:    ${baseline_total/1e6:>10,.0f}M ({baseline_total/total_balance*100:.2f}%)")
    print(f"  Adverse Loss:     ${adverse_total/1e6:>10,.0f}M ({adverse_rate*100:.2f}%)")
    print(f"  Stress Increment: ${stress_increment/1e6:>+10,.0f}M")
    print(f"\n  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()