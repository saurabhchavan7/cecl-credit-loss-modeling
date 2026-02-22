"""
Full Monte Carlo Simulation Pipeline (Phase 8)
================================================

Generates 10,000 random economic scenarios, computes portfolio losses
under each, and derives VaR, Expected Shortfall, and sensitivity analysis.

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
from monte_carlo import (
    compute_historical_macro_stats,
    run_monte_carlo,
    compute_risk_metrics,
    sensitivity_analysis,
)

COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MACRO_PATH = project_root / "data" / "processed" / "macro" / "fred_macro_monthly.csv"
MODEL_DIR = project_root / "models"
N_SIMULATIONS = 10_000


def main():
    t_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - MONTE CARLO SIMULATION (Phase 8)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load portfolio and score baseline PD/LGD
    # ------------------------------------------------------------------
    print("\nStep 1: Loading portfolio and scoring baseline...")

    df = pd.read_parquet(COMBINED_PATH)
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"
    total_balance = df["original_upb"].sum()
    portfolio_upb = df["original_upb"].values.astype(float)
    print(f"  Portfolio: {len(df):,} loans, ${total_balance/1e9:.1f}B")

    # Score baseline PD
    pd_model = joblib.load(MODEL_DIR / "pd_logistic_regression.pkl")
    woe_results = joblib.load(MODEL_DIR / "woe_results.pkl")
    with open(MODEL_DIR / "selected_features.txt") as f:
        pd_features = [line.strip() for line in f if line.strip()]

    X_woe = apply_woe_transformation(df, woe_results, pd_features)
    pd_baseline = pd_model.predict_proba(X_woe)[:, 1]
    del X_woe
    gc.collect()

    # Score baseline LGD
    lgd_model = joblib.load(MODEL_DIR / "lgd_ols.pkl")
    with open(MODEL_DIR / "lgd_features.txt") as f:
        lgd_features = [line.strip() for line in f if line.strip()]

    X_lgd = df[lgd_features].copy()
    lgd_fill = {"loan_age_at_default": 48.0, "was_modified": 0.0}
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())
    lgd_baseline = lgd_model.predict(X_lgd)
    lgd_baseline = np.clip(lgd_baseline, 0.0, 1.0)
    del X_lgd, df
    gc.collect()

    baseline_el = (pd_baseline * lgd_baseline * portfolio_upb).sum()
    print(f"  Baseline PD: {pd_baseline.mean()*100:.2f}%")
    print(f"  Baseline LGD: {lgd_baseline.mean()*100:.2f}%")
    print(f"  Baseline Annual EL: ${baseline_el/1e6:,.0f}M "
          f"({baseline_el/total_balance*100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 2: Compute historical macro statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 2: Historical macro statistics")
    print(f"{'='*70}")

    # Check both possible macro cache paths
    macro_path = MACRO_PATH
    if not Path(macro_path).exists():
        alt_path = project_root / "data" / "macro" / "fred_macro_monthly.csv"
        if alt_path.exists():
            macro_path = alt_path
        else:
            print(f"  ERROR: Macro data not found at {MACRO_PATH} or {alt_path}")
            sys.exit(1)

    macro_stats = compute_historical_macro_stats(macro_path)

    # ------------------------------------------------------------------
    # Step 3: Run Monte Carlo simulation
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Step 3: Running {N_SIMULATIONS:,} Monte Carlo simulations")
    print(f"{'='*70}")

    losses, scenarios = run_monte_carlo(
        portfolio_upb, pd_baseline, lgd_baseline,
        macro_stats, n_simulations=N_SIMULATIONS,
    )

    # ------------------------------------------------------------------
    # Step 4: Compute risk metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 4: Risk metrics from loss distribution")
    print(f"{'='*70}")

    metrics = compute_risk_metrics(losses, total_balance)

    # ------------------------------------------------------------------
    # Step 5: Sensitivity analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 5: Sensitivity analysis (tornado chart data)")
    print(f"{'='*70}")

    sensitivity_df = sensitivity_analysis(
        portfolio_upb, pd_baseline, lgd_baseline, macro_stats
    )

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 6: Saving results")
    print(f"{'='*70}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save loss distribution
    loss_dist = pd.DataFrame({
        "simulation_id": range(N_SIMULATIONS),
        "portfolio_loss": losses,
        "loss_rate": losses / total_balance,
    })
    loss_dist.to_csv(MODEL_DIR / "mc_loss_distribution.csv", index=False)

    # Save scenario details
    scenarios.to_csv(MODEL_DIR / "mc_scenarios.csv", index=False)

    # Save risk metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df["total_balance"] = total_balance
    metrics_df.to_csv(MODEL_DIR / "mc_risk_metrics.csv", index=False)

    # Save sensitivity
    sensitivity_df.to_csv(MODEL_DIR / "mc_sensitivity.csv", index=False)

    print(f"  Saved all Monte Carlo results to {MODEL_DIR}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print("MONTE CARLO SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"  Portfolio:             {len(portfolio_upb):,} loans, ${total_balance/1e9:.1f}B")
    print(f"  Simulations:           {N_SIMULATIONS:,}")
    print(f"  Expected Loss (mean):  ${metrics['expected_loss']/1e6:>10,.0f}M "
          f"({metrics['expected_loss']/total_balance*100:.2f}%)")
    print(f"  VaR 99%:               ${metrics['var_99']/1e6:>10,.0f}M "
          f"({metrics['var_99']/total_balance*100:.2f}%)")
    print(f"  VaR 99.9%:             ${metrics['var_999']/1e6:>10,.0f}M "
          f"({metrics['var_999']/total_balance*100:.2f}%)")
    print(f"  Expected Shortfall:    ${metrics['es_99']/1e6:>10,.0f}M "
          f"({metrics['es_99']/total_balance*100:.2f}%)")
    print(f"\n  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()