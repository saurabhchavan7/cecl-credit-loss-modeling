"""
Stress Testing Module (Phase 7)
================================

Implements the Federal Reserve stress testing framework using the 2025
DFAST/CCAR scenarios. Instead of applying simple multipliers, this module
re-scores PD and LGD models under the Fed's quarterly macro paths.

The Fed publishes two scenarios each year:
  1. Baseline: most likely economic path
  2. Severely Adverse: deep recession (unemployment 10%, HPI -33%, GDP -7.8%)

For each scenario, we:
  1. Read the quarterly macro path (13 quarters: 2025Q1 to 2028Q1)
  2. For each quarter, substitute the scenario's macro variables into our
     PD and LGD models
  3. Score every loan's PD and LGD under those macro conditions
  4. Compute quarterly losses and cumulative ECL
  5. Compare baseline vs stressed losses by segment

This directly addresses the JD requirement: "Develop credit risk models
for stress testing" and "Partner with business analyst team to generate
business insights."

Author: Saurabh Chavan
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")


def load_fed_scenarios(scenarios_dir):
    """
    Load Federal Reserve 2025 stress test scenarios.

    The scenario CSVs contain quarterly macro variable paths over 13
    quarters (2025Q1 through 2028Q1).

    Parameters
    ----------
    scenarios_dir : str or Path
        Directory containing baseline_2025.csv and severely_adverse_2025.csv.

    Returns
    -------
    dict
        Scenario name -> pd.DataFrame with quarterly macro variables.
    """
    scenarios_dir = Path(scenarios_dir)
    scenarios = {}

    for name, filename in [
        ("Baseline", "baseline_2025.csv"),
        ("Severely Adverse", "severely_adverse_2025.csv"),
    ]:
        path = scenarios_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            scenarios[name] = df
            print(f"  Loaded {name}: {len(df)} quarters, {df.shape[1]} variables")
            print(f"    Quarters: {df['quarter'].iloc[0]} to {df['quarter'].iloc[-1]}")
        else:
            print(f"  WARNING: {path} not found")

    return scenarios


def map_scenario_to_model_features(scenario_row):
    """
    Map Fed scenario macro variables to the feature names used in our
    PD and LGD models.

    The Fed scenario CSVs use different column names than our FRED-based
    features. This function creates the mapping.

    Parameters
    ----------
    scenario_row : pd.Series
        One quarter's macro variables from the Fed scenario.

    Returns
    -------
    dict
        Feature name -> value, compatible with our model features.
    """
    mapping = {}

    # Map scenario columns to our model feature names.
    # The exact column names depend on how the scenarios were saved in Phase 0.
    col_map = {
        "unemployment_rate": "unemployment_rate",
        "house_price_index": "hpi_national",
        "treasury_rate_3m": "fed_funds_rate",
        "real_gdp_growth": "gdp_growth_pct",
        "mortgage_rate_30y": "mortgage_rate_30y",
        "treasury_yield_10y": "treasury_10y",
        "bbb_corporate_yield": "baa_spread",
    }

    for scenario_col, model_col in col_map.items():
        if scenario_col in scenario_row.index:
            mapping[model_col] = scenario_row[scenario_col]

    return mapping


def score_portfolio_under_scenario(
    portfolio_df,
    pd_model,
    woe_results,
    pd_features,
    lgd_model,
    lgd_features,
    macro_overrides,
):
    """
    Score the entire portfolio's PD and LGD under a specific macro scenario.

    Replaces the macro features in the portfolio with the scenario's
    macro values, then re-scores PD and LGD.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Full loan-level dataset.
    pd_model : fitted model
        PD logistic regression model.
    woe_results : dict
        WoE transformation mappings.
    pd_features : list
        PD model feature names.
    lgd_model : fitted model
        LGD OLS model.
    lgd_features : list
        LGD model feature names.
    macro_overrides : dict
        Feature name -> scenario value to substitute.

    Returns
    -------
    np.array
        PD predictions under the scenario.
    np.array
        LGD predictions under the scenario.
    """
    from pd_model import apply_woe_transformation

    # Create a copy of relevant columns with macro overrides
    df_scenario = portfolio_df.copy()
    for feat, val in macro_overrides.items():
        if feat in df_scenario.columns:
            df_scenario[feat] = val

    # Also update interaction features that depend on macro variables
    if "fico_x_unemployment" in df_scenario.columns and "unemployment_rate" in macro_overrides:
        df_scenario["fico_x_unemployment"] = (
            df_scenario["borrower_credit_score"] * macro_overrides["unemployment_rate"]
        )

    # Score PD
    X_woe = apply_woe_transformation(df_scenario, woe_results, pd_features)
    pd_preds = pd_model.predict_proba(X_woe)[:, 1]

    # Score LGD
    X_lgd = df_scenario[lgd_features].copy()
    lgd_fill = {
        "loan_age_at_default": 48.0,
        "was_modified": 0.0,
    }
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())

    lgd_preds = lgd_model.predict(X_lgd)
    lgd_preds = np.clip(lgd_preds, 0.0, 1.0)

    return pd_preds, lgd_preds


def run_stress_test(
    portfolio_df,
    scenarios,
    pd_model,
    woe_results,
    pd_features,
    lgd_model,
    lgd_features,
):
    """
    Run the full stress test across all Fed scenarios.

    For each scenario and each quarter, scores the portfolio and computes
    quarterly and cumulative losses.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Full loan-level dataset.
    scenarios : dict
        Scenario name -> DataFrame of quarterly macro paths.
    pd_model, woe_results, pd_features : PD model components.
    lgd_model, lgd_features : LGD model components.

    Returns
    -------
    dict
        Scenario name -> dict with quarterly results and summary.
    """
    results = {}
    total_balance = portfolio_df["original_upb"].sum()

    for scenario_name, scenario_df in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Stress Test: {scenario_name}")
        print(f"{'='*70}")

        quarterly_results = []
        t_start = time.time()

        for idx, row in scenario_df.iterrows():
            quarter = row["quarter"]
            macro_overrides = map_scenario_to_model_features(row)

            if not macro_overrides:
                print(f"  WARNING: No macro mappings for {quarter}, skipping")
                continue

            # Score portfolio under this quarter's macro conditions
            pd_preds, lgd_preds = score_portfolio_under_scenario(
                portfolio_df, pd_model, woe_results, pd_features,
                lgd_model, lgd_features, macro_overrides,
            )

            # Quarterly expected loss = mean(PD * LGD) * total_balance / 4
            # (dividing by 4 to convert annual PD to quarterly)
            quarterly_el = (pd_preds * lgd_preds).mean() * total_balance / 4.0

            # Store results
            qr = {
                "quarter": quarter,
                "mean_pd": pd_preds.mean(),
                "mean_lgd": lgd_preds.mean(),
                "quarterly_el": quarterly_el,
            }
            qr.update(macro_overrides)
            quarterly_results.append(qr)

        elapsed = time.time() - t_start

        # Build quarterly results table
        qr_df = pd.DataFrame(quarterly_results)
        qr_df["cumulative_el"] = qr_df["quarterly_el"].cumsum()

        # Total losses over the scenario horizon
        total_loss = qr_df["quarterly_el"].sum()
        loss_rate = total_loss / total_balance if total_balance > 0 else 0

        # Print quarterly path
        print(f"\n  Quarterly Loss Path:")
        print(f"  {'Quarter':<10s} {'Unemp':>8s} {'HPI':>8s} {'Mean PD':>8s} "
              f"{'Mean LGD':>9s} {'Q Loss($M)':>12s} {'Cum Loss($M)':>14s}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*12} {'-'*14}")

        for _, qr in qr_df.iterrows():
            ur = qr.get("unemployment_rate", np.nan)
            hpi = qr.get("hpi_national", np.nan)
            print(f"  {qr['quarter']:<10s} {ur:>7.1f}% {hpi:>8.1f} "
                  f"{qr['mean_pd']*100:>7.2f}% {qr['mean_lgd']*100:>8.2f}% "
                  f"${qr['quarterly_el']/1e6:>11,.0f} ${qr['cumulative_el']/1e6:>13,.0f}")

        print(f"\n  Scenario Summary ({scenario_name}):")
        print(f"    Total Balance:      ${total_balance/1e9:,.1f}B")
        print(f"    Total Loss:         ${total_loss/1e6:,.0f}M")
        print(f"    Loss Rate:          {loss_rate*100:.2f}%")
        print(f"    Peak Quarterly PD:  {qr_df['mean_pd'].max()*100:.2f}%")
        print(f"    Computation Time:   {elapsed:.1f} seconds")

        results[scenario_name] = {
            "quarterly": qr_df,
            "total_loss": total_loss,
            "loss_rate": loss_rate,
            "total_balance": total_balance,
        }

    return results


def compute_segment_stress_results(
    portfolio_df,
    pd_baseline, lgd_baseline,
    pd_stressed, lgd_stressed,
    segment_col, segment_name,
):
    """
    Compare baseline vs stressed losses by portfolio segment.

    This identifies WHERE risk concentrates under stress -- the key
    insight that drives business decisions like "tighten lending
    standards for subprime high-LTV segment."

    Parameters
    ----------
    portfolio_df : pd.DataFrame
    pd_baseline, lgd_baseline : baseline PD/LGD arrays
    pd_stressed, lgd_stressed : stressed PD/LGD arrays
    segment_col : str
        Column name for segmentation.
    segment_name : str
        Display name.

    Returns
    -------
    pd.DataFrame
        Segment-level stress comparison.
    """
    df = pd.DataFrame({
        "segment": np.asarray(portfolio_df[segment_col]),
        "balance": portfolio_df["original_upb"].values,
        "el_baseline": pd_baseline * lgd_baseline * portfolio_df["original_upb"].values,
        "el_stressed": pd_stressed * lgd_stressed * portfolio_df["original_upb"].values,
    })

    seg = df.groupby("segment").agg(
        n_loans=("balance", "count"),
        total_balance=("balance", "sum"),
        baseline_el=("el_baseline", "sum"),
        stressed_el=("el_stressed", "sum"),
    ).reset_index()

    seg["baseline_rate"] = seg["baseline_el"] / seg["total_balance"]
    seg["stressed_rate"] = seg["stressed_el"] / seg["total_balance"]
    seg["stress_increment"] = seg["stressed_rate"] - seg["baseline_rate"]
    seg["stress_multiplier"] = seg["stressed_rate"] / seg["baseline_rate"]

    print(f"\n  Stress Results by {segment_name}:")
    print(f"  {'Segment':<12s} {'Loans':>10s} {'Bal($B)':>10s} "
          f"{'Base Rate':>10s} {'Stress Rate':>12s} {'Increment':>10s} {'Mult':>6s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")

    for _, row in seg.iterrows():
        print(f"  {str(row['segment']):<12s} {row['n_loans']:>10,} "
              f"${row['total_balance']/1e9:>9.1f} "
              f"{row['baseline_rate']*100:>9.2f}% "
              f"{row['stressed_rate']*100:>11.2f}% "
              f"{row['stress_increment']*100:>+9.2f}% "
              f"{row['stress_multiplier']:>5.1f}x")

    return seg