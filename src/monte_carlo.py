"""
Monte Carlo Simulation Module (Phase 8)
========================================

Instead of testing just 2 scenarios (baseline, adverse), Monte Carlo
generates thousands of random possible economic futures. Each future has
different unemployment, GDP, and HPI paths drawn from realistic
distributions. We run our loss models under each future, producing a
DISTRIBUTION of possible losses.

From this distribution:
  Expected Loss (mean) = most likely loss = what we reserve (CECL)
  VaR 99.9% = loss exceeded only 0.1% of the time = capital buffer
  Expected Shortfall (ES) = average loss in the worst 1% = tail risk

Key technical details:
  - Macro variables are correlated (when unemployment is high, HPI is low).
    We use the Cholesky decomposition of the historical correlation matrix
    to generate correlated random draws.
  - We use the same scalar/multiplier approach as stress testing: baseline
    PD and LGD are multiplied by scenario-specific factors derived from
    the random macro draws.

Author: Saurabh Chavan
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def compute_historical_macro_stats(macro_csv_path):
    """
    Compute historical means, standard deviations, and correlations
    for key macro variables from the FRED data.

    These statistics define the distribution from which we draw random
    macro scenarios. Using historical data ensures our simulated futures
    are realistic and properly correlated.

    Parameters
    ----------
    macro_csv_path : str or Path
        Path to the FRED macro monthly CSV from Phase 2.

    Returns
    -------
    dict with keys:
        means: dict of variable -> historical mean
        stds: dict of variable -> historical std
        correlation_matrix: np.array
        variable_names: list of variable names
    """
    macro_df = pd.read_csv(macro_csv_path, index_col=0, parse_dates=True)

    # Focus on the variables that drive our stress multipliers
    variables = {
        "unemployment_rate": "unemployment_rate",
        "hpi_change": None,  # Computed below
        "gdp_growth": None,  # Computed below
    }

    # Compute derived variables
    if "hpi_national" in macro_df.columns:
        macro_df["hpi_change_annual"] = macro_df["hpi_national"].pct_change(12) * 100
    if "gdp" in macro_df.columns:
        macro_df["gdp_growth_annual"] = macro_df["gdp"].pct_change(4) * 100

    # Select the three key variables
    analysis_cols = []
    col_names = []

    if "unemployment_rate" in macro_df.columns:
        analysis_cols.append("unemployment_rate")
        col_names.append("unemployment_rate")
    if "hpi_change_annual" in macro_df.columns:
        analysis_cols.append("hpi_change_annual")
        col_names.append("hpi_change_annual")
    if "gdp_growth_annual" in macro_df.columns:
        analysis_cols.append("gdp_growth_annual")
        col_names.append("gdp_growth_annual")

    subset = macro_df[analysis_cols].dropna()

    means = {col: subset[col].mean() for col in analysis_cols}
    stds = {col: subset[col].std() for col in analysis_cols}
    corr_matrix = subset.corr().values

    print(f"  Historical macro statistics (from {len(subset)} monthly observations):")
    for col in analysis_cols:
        print(f"    {col}: mean={means[col]:.2f}, std={stds[col]:.2f}")

    print(f"\n  Correlation matrix:")
    print(f"  {'':>25s}", end="")
    for c in col_names:
        print(f" {c[:12]:>12s}", end="")
    print()
    for i, c1 in enumerate(col_names):
        print(f"  {c1:>25s}", end="")
        for j in range(len(col_names)):
            print(f" {corr_matrix[i, j]:>12.3f}", end="")
        print()

    return {
        "means": means,
        "stds": stds,
        "correlation_matrix": corr_matrix,
        "variable_names": col_names,
    }


def generate_correlated_scenarios(n_simulations, macro_stats, random_seed=42):
    """
    Generate correlated random macro scenarios using Cholesky decomposition.

    The Cholesky decomposition transforms independent standard normal
    random variables into correlated random variables that match the
    historical correlation structure. This ensures that when we draw
    a scenario with high unemployment, it also has low HPI growth and
    low GDP growth (as observed historically).

    Parameters
    ----------
    n_simulations : int
        Number of random scenarios to generate.
    macro_stats : dict
        Output from compute_historical_macro_stats.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape (n_simulations, n_variables) with correlated macro draws.
    """
    rng = np.random.RandomState(random_seed)
    n_vars = len(macro_stats["variable_names"])

    # Cholesky decomposition of the correlation matrix.
    # L is a lower-triangular matrix such that L * L^T = correlation_matrix.
    # Multiplying independent standard normals by L produces correlated normals.
    corr = macro_stats["correlation_matrix"]

    # Ensure the correlation matrix is positive definite (numerical stability)
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < 0:
        # Add small diagonal to make positive definite
        corr += np.eye(n_vars) * (abs(np.min(eigvals)) + 0.01)

    L = np.linalg.cholesky(corr)

    # Generate independent standard normal draws
    Z = rng.standard_normal((n_simulations, n_vars))

    # Transform to correlated draws
    correlated_Z = Z @ L.T

    # Scale to historical mean and std
    scenarios = pd.DataFrame(
        columns=macro_stats["variable_names"],
        index=range(n_simulations),
    )

    for i, col in enumerate(macro_stats["variable_names"]):
        mean = macro_stats["means"][col]
        std = macro_stats["stds"][col]
        scenarios[col] = mean + std * correlated_Z[:, i]

    # Apply realistic bounds
    if "unemployment_rate" in scenarios.columns:
        scenarios["unemployment_rate"] = scenarios["unemployment_rate"].clip(2.0, 15.0)
    if "hpi_change_annual" in scenarios.columns:
        scenarios["hpi_change_annual"] = scenarios["hpi_change_annual"].clip(-40.0, 30.0)
    if "gdp_growth_annual" in scenarios.columns:
        scenarios["gdp_growth_annual"] = scenarios["gdp_growth_annual"].clip(-15.0, 15.0)

    return scenarios


def compute_scenario_multipliers(scenarios_df, baseline_ur=4.3, baseline_hpi_change=0.0):
    """
    Convert macro draws into PD and LGD multipliers.

    Uses the same calibration as the stress testing module:
    - PD: +25% per +1% unemployment above baseline, +5% per -1% GDP
    - LGD: +15% per -10% HPI decline

    Parameters
    ----------
    scenarios_df : pd.DataFrame
        Correlated macro scenarios.
    baseline_ur : float
        Baseline unemployment rate.
    baseline_hpi_change : float
        Baseline annual HPI change (%).

    Returns
    -------
    pd.DataFrame with pd_multiplier and lgd_multiplier columns added.
    """
    df = scenarios_df.copy()

    # PD multiplier
    df["pd_multiplier"] = 1.0
    if "unemployment_rate" in df.columns:
        ur_delta = (df["unemployment_rate"] - baseline_ur).clip(lower=0)
        df["pd_multiplier"] += 0.25 * ur_delta
    if "gdp_growth_annual" in df.columns:
        gdp_neg = (-df["gdp_growth_annual"]).clip(lower=0)
        df["pd_multiplier"] += 0.05 * gdp_neg

    # LGD multiplier
    df["lgd_multiplier"] = 1.0
    if "hpi_change_annual" in df.columns:
        hpi_decline = (-df["hpi_change_annual"] + baseline_hpi_change).clip(lower=0)
        df["lgd_multiplier"] += 0.015 * hpi_decline  # +1.5% per -1% HPI

    # Floor multipliers at 1.0 (no improvement over baseline in loss models)
    # and cap at reasonable maximums
    df["pd_multiplier"] = df["pd_multiplier"].clip(0.5, 5.0)
    df["lgd_multiplier"] = df["lgd_multiplier"].clip(0.5, 3.0)

    return df


def run_monte_carlo(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    macro_stats,
    n_simulations=10_000,
    random_seed=42,
):
    """
    Run Monte Carlo simulation on the portfolio.

    For each of n_simulations random macro scenarios:
    1. Draw correlated macro variables
    2. Convert to PD and LGD multipliers
    3. Compute portfolio loss = sum(PD_stressed * LGD_stressed * UPB) annualized
    4. Store the total loss

    Result: distribution of n_simulations portfolio loss values.

    Parameters
    ----------
    portfolio_upb : np.array
        Original UPB for each loan.
    pd_baseline : np.array
        Baseline PD for each loan.
    lgd_baseline : np.array
        Baseline LGD for each loan.
    macro_stats : dict
        Historical macro statistics.
    n_simulations : int
        Number of Monte Carlo scenarios.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.array
        Array of portfolio losses, one per simulation.
    pd.DataFrame
        Scenario details with multipliers and losses.
    """
    print(f"\n  Running {n_simulations:,} Monte Carlo simulations...")
    t0 = time.time()

    # Pre-compute baseline expected loss per loan (annual)
    baseline_el_per_loan = pd_baseline * lgd_baseline * portfolio_upb
    total_balance = portfolio_upb.sum()

    # Generate correlated scenarios
    scenarios = generate_correlated_scenarios(n_simulations, macro_stats, random_seed)

    # Compute multipliers
    scenarios = compute_scenario_multipliers(scenarios)

    # Compute portfolio loss for each scenario
    losses = np.zeros(n_simulations)

    pd_mult = scenarios["pd_multiplier"].values
    lgd_mult = scenarios["lgd_multiplier"].values

    for i in range(n_simulations):
        # Apply multipliers to baseline expected loss
        # This is equivalent to: sum(PD*mult_pd * LGD*mult_lgd * UPB)
        # = mult_pd * mult_lgd * sum(PD * LGD * UPB)
        # = mult_pd * mult_lgd * baseline_portfolio_el
        losses[i] = pd_mult[i] * lgd_mult[i] * baseline_el_per_loan.sum()

        if (i + 1) % 2000 == 0:
            print(f"    Completed {i+1:,}/{n_simulations:,} simulations")

    scenarios["portfolio_loss"] = losses
    scenarios["loss_rate"] = losses / total_balance

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f} seconds")

    return losses, scenarios


def compute_risk_metrics(losses, total_balance):
    """
    Compute key risk metrics from the loss distribution.

    Parameters
    ----------
    losses : np.array
        Portfolio losses from Monte Carlo simulation.
    total_balance : float
        Total portfolio outstanding balance.

    Returns
    -------
    dict of risk metrics.
    """
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)

    expected_loss = np.mean(losses)
    std_loss = np.std(losses)
    var_99 = np.percentile(losses, 99)
    var_999 = np.percentile(losses, 99.9)
    es_99 = np.mean(losses[losses >= var_99])

    # Percentiles for the full distribution
    p5 = np.percentile(losses, 5)
    p25 = np.percentile(losses, 25)
    p50 = np.percentile(losses, 50)
    p75 = np.percentile(losses, 75)
    p95 = np.percentile(losses, 95)

    metrics = {
        "expected_loss": expected_loss,
        "std_loss": std_loss,
        "var_99": var_99,
        "var_999": var_999,
        "es_99": es_99,
        "p5": p5,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "min_loss": sorted_losses[0],
        "max_loss": sorted_losses[-1],
    }

    print(f"\n  Risk Metrics:")
    print(f"    Expected Loss (mean):   ${expected_loss/1e6:>10,.0f}M "
          f"({expected_loss/total_balance*100:.2f}%)")
    print(f"    Std Deviation:          ${std_loss/1e6:>10,.0f}M")
    print(f"    VaR 99%:                ${var_99/1e6:>10,.0f}M "
          f"({var_99/total_balance*100:.2f}%)")
    print(f"    VaR 99.9%:              ${var_999/1e6:>10,.0f}M "
          f"({var_999/total_balance*100:.2f}%)")
    print(f"    Expected Shortfall 99%: ${es_99/1e6:>10,.0f}M "
          f"({es_99/total_balance*100:.2f}%)")
    print(f"\n  Loss Distribution:")
    print(f"    5th percentile:  ${p5/1e6:>10,.0f}M ({p5/total_balance*100:.2f}%)")
    print(f"    25th percentile: ${p25/1e6:>10,.0f}M ({p25/total_balance*100:.2f}%)")
    print(f"    Median:          ${p50/1e6:>10,.0f}M ({p50/total_balance*100:.2f}%)")
    print(f"    75th percentile: ${p75/1e6:>10,.0f}M ({p75/total_balance*100:.2f}%)")
    print(f"    95th percentile: ${p95/1e6:>10,.0f}M ({p95/total_balance*100:.2f}%)")
    print(f"    Maximum:         ${sorted_losses[-1]/1e6:>10,.0f}M "
          f"({sorted_losses[-1]/total_balance*100:.2f}%)")

    return metrics


def sensitivity_analysis(portfolio_upb, pd_baseline, lgd_baseline,
                         macro_stats, n_simulations=5000):
    """
    One-at-a-time sensitivity analysis.

    Holds all variables at baseline except one, which is shocked.
    Shows which macro variable drives the most portfolio risk.

    Parameters
    ----------
    portfolio_upb, pd_baseline, lgd_baseline : portfolio data
    macro_stats : historical statistics
    n_simulations : int

    Returns
    -------
    pd.DataFrame
        Sensitivity results.
    """
    baseline_el = (pd_baseline * lgd_baseline * portfolio_upb).sum()
    total_balance = portfolio_upb.sum()

    variables_to_shock = {
        "unemployment_rate": [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
        "hpi_change_annual": [10.0, 5.0, 0.0, -5.0, -10.0, -20.0, -30.0],
        "gdp_growth_annual": [4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -8.0],
    }

    print(f"\n  Sensitivity Analysis (one variable at a time):")
    results = []

    for var_name, shock_values in variables_to_shock.items():
        print(f"\n  {var_name}:")
        print(f"  {'Value':>10s} {'PD Mult':>8s} {'LGD Mult':>9s} "
              f"{'Loss ($M)':>12s} {'Loss Rate':>10s} {'vs Base':>10s}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*12} {'-'*10} {'-'*10}")

        for val in shock_values:
            # Create a single-row scenario
            scenario_row = {v: macro_stats["means"][v]
                           for v in macro_stats["variable_names"]
                           if v in macro_stats["means"]}
            scenario_row[var_name] = val

            scenario_df = pd.DataFrame([scenario_row])
            scenario_df = compute_scenario_multipliers(scenario_df)

            pm = scenario_df["pd_multiplier"].iloc[0]
            lm = scenario_df["lgd_multiplier"].iloc[0]
            loss = pm * lm * baseline_el
            loss_rate = loss / total_balance
            vs_base = loss / baseline_el

            results.append({
                "variable": var_name,
                "value": val,
                "pd_multiplier": pm,
                "lgd_multiplier": lm,
                "loss": loss,
                "loss_rate": loss_rate,
            })

            print(f"  {val:>10.1f} {pm:>7.2f}x {lm:>8.2f}x "
                  f"${loss/1e6:>11,.0f} {loss_rate*100:>9.2f}% {vs_base:>9.2f}x")

    return pd.DataFrame(results)