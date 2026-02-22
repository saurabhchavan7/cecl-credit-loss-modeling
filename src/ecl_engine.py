"""
EAD Model & ECL Calculation Engine (Phases 5 + 6)
===================================================

Phase 5: EAD (Exposure at Default)
For fixed-rate fully-amortizing mortgages, EAD at any future month t is
the scheduled remaining balance given the original loan terms:

    Balance(t) = P * [(1+r)^n - (1+r)^t] / [(1+r)^n - 1]

Where:
    P = original principal balance
    r = monthly interest rate (annual rate / 12)
    n = total number of payments (loan term in months)
    t = number of payments already made (loan age)

Phase 6: ECL (Expected Credit Loss under CECL)
CECL requires LIFETIME expected loss from day one of every loan:

    ECL = SUM over t=1 to remaining_term:
        Marginal_PD(t) * LGD(t) * EAD(t) * Discount_Factor(t)

Where:
    Marginal_PD(t) = probability of defaulting in exactly month t
    LGD(t) = expected loss severity if default occurs at month t
    EAD(t) = expected exposure if default occurs at month t
    Discount_Factor(t) = 1/(1+r)^t (time value of money)

Scenario weighting for CECL:
    ECL_final = w_base * ECL_baseline + w_adverse * ECL_adverse

Author: Saurabh Chavan
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# EAD: Amortization Schedule
# ---------------------------------------------------------------------------

def compute_scheduled_balance(original_upb, annual_rate, loan_term_months, month_t):
    """
    Compute the scheduled remaining balance at month t for a fixed-rate
    fully-amortizing mortgage.

    This is the standard amortization formula used by every mortgage
    servicer. The balance declines slowly at first (most of the payment
    goes to interest) and accelerates later (more goes to principal).

    Parameters
    ----------
    original_upb : float or np.array
        Original unpaid principal balance (loan amount at origination).
    annual_rate : float or np.array
        Annual interest rate (e.g., 0.065 for 6.5%).
    loan_term_months : int or np.array
        Total loan term in months (typically 360 for 30-year).
    month_t : int or np.array
        Number of payments already made.

    Returns
    -------
    float or np.array
        Scheduled remaining balance at month t.
    """
    # Monthly interest rate
    r = annual_rate / 12.0

    # Handle edge case: zero interest rate (rare but possible)
    # For zero-rate loans, balance declines linearly
    if np.isscalar(r):
        if r == 0:
            return original_upb * (1.0 - month_t / loan_term_months)
    else:
        # Vectorized: handle zero rates
        r = np.where(r == 0, 1e-10, r)

    n = loan_term_months

    # Amortization formula:
    # Balance(t) = P * [(1+r)^n - (1+r)^t] / [(1+r)^n - 1]
    factor_n = (1 + r) ** n
    factor_t = (1 + r) ** month_t

    balance = original_upb * (factor_n - factor_t) / (factor_n - 1)

    # Balance cannot be negative (fully paid off)
    balance = np.maximum(balance, 0.0)

    return balance


def compute_ead_schedule(original_upb, annual_rate, loan_term_months,
                         current_age, projection_months):
    """
    Compute EAD for each future month over a projection horizon.

    This produces an array of scheduled balances from the current age
    forward, representing the exposure the bank would face if the
    borrower defaulted at each future point.

    Parameters
    ----------
    original_upb : float or np.array
        Original loan balance.
    annual_rate : float or np.array
        Annual interest rate.
    loan_term_months : int or np.array
        Total loan term in months.
    current_age : int or np.array
        Current loan age in months.
    projection_months : int
        Number of months to project forward.

    Returns
    -------
    np.array
        Shape (projection_months,) or (n_loans, projection_months).
        Scheduled remaining balance at each future month.
    """
    future_months = np.arange(1, projection_months + 1)

    if np.isscalar(original_upb):
        # Single loan
        ages = current_age + future_months
        # Cap at loan term (balance is zero after maturity)
        ages = np.minimum(ages, loan_term_months)
        ead = compute_scheduled_balance(original_upb, annual_rate,
                                        loan_term_months, ages)
    else:
        # Vectorized for portfolio (n_loans,)
        # Result shape: (n_loans, projection_months)
        n_loans = len(original_upb)
        ead = np.zeros((n_loans, projection_months))

        for t_idx, t in enumerate(future_months):
            ages = current_age + t
            # Cap at loan term
            ages = np.minimum(ages, loan_term_months)
            ead[:, t_idx] = compute_scheduled_balance(
                original_upb, annual_rate, loan_term_months, ages
            )

    return ead


# ---------------------------------------------------------------------------
# PD Term Structure: Extend 12-month PD to Lifetime
# ---------------------------------------------------------------------------

def build_pd_term_structure(pd_12m, remaining_months, seasoning_curve=None):
    """
    Extend a 12-month PD estimate to a lifetime (monthly) PD curve.

    Banks need lifetime PD for CECL, but PD models typically estimate
    12-month default probability. We extend using a seasoning curve
    that captures the empirical pattern: mortgage defaults peak around
    months 36-60 after origination, then decline.

    The approach:
    1. Start with the 12-month PD from the PD model
    2. Distribute it across 12 months using the seasoning curve shape
    3. For months beyond 12, scale the curve based on the overall PD level

    Parameters
    ----------
    pd_12m : float or np.array
        12-month probability of default from the PD model.
    remaining_months : int or np.array
        Remaining months to maturity for each loan.
    seasoning_curve : np.array, optional
        Monthly hazard rate multipliers. If None, uses empirical
        mortgage default pattern.

    Returns
    -------
    np.array
        Shape (max_remaining_months,) or (n_loans, max_remaining_months).
        Monthly marginal PD at each future month.
    """
    if seasoning_curve is None:
        # Empirical mortgage seasoning curve (relative hazard rates).
        # Defaults ramp up over the first 3-5 years, peak around year 4,
        # then gradually decline as surviving borrowers prove stable.
        # This shape is well-documented in mortgage credit risk literature.
        #
        # Normalized so the first 12 months sum to approximately the
        # annual PD level.
        seasoning_multipliers = np.array([
            # Year 1 (months 1-12): ramp-up phase
            0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
            0.9, 0.95, 1.0, 1.0, 1.0, 1.0,
            # Year 2 (months 13-24): continuing ramp
            1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
            1.30, 1.30, 1.30, 1.30, 1.25, 1.20,
            # Year 3 (months 25-36): approaching peak
            1.15, 1.10, 1.10, 1.10, 1.10, 1.10,
            1.10, 1.10, 1.05, 1.05, 1.05, 1.00,
            # Year 4 (months 37-48): peak default period
            1.00, 1.00, 0.95, 0.95, 0.90, 0.90,
            0.85, 0.85, 0.80, 0.80, 0.75, 0.75,
            # Year 5+ (months 49-60): decline
            0.70, 0.65, 0.60, 0.55, 0.50, 0.50,
            0.45, 0.45, 0.40, 0.40, 0.35, 0.35,
        ])
        # Extend to 360 months (30 years) with gradual decay
        if len(seasoning_multipliers) < 360:
            last_val = seasoning_multipliers[-1]
            decay_months = 360 - len(seasoning_multipliers)
            # Exponential decay: defaults become increasingly rare
            # for seasoned loans
            decay = last_val * np.exp(
                -0.005 * np.arange(1, decay_months + 1)
            )
            decay = np.maximum(decay, 0.05)  # Floor at 5% of base rate
            seasoning_multipliers = np.concatenate([seasoning_multipliers, decay])

        seasoning_curve = seasoning_multipliers

    if np.isscalar(pd_12m):
        # Single loan
        max_months = int(remaining_months)
        if max_months <= 0:
            return np.array([0.0])

        # Scale seasoning curve so first 12 months sum to pd_12m
        curve_12m = seasoning_curve[:12]
        scale_factor = pd_12m / curve_12m.sum() if curve_12m.sum() > 0 else 0

        # Monthly hazard rate = scale_factor * seasoning_multiplier
        n_months = min(max_months, len(seasoning_curve))
        monthly_hazard = scale_factor * seasoning_curve[:n_months]

        # Cap monthly hazard at reasonable maximum
        monthly_hazard = np.minimum(monthly_hazard, 0.05)

        # Convert hazard rates to marginal PD using survival function.
        # Survival(t) = product of (1 - hazard(s)) for s=1 to t-1
        # Marginal_PD(t) = hazard(t) * Survival(t)
        # This ensures: a loan that defaulted in month 5 cannot default
        # again in month 6. Total cumulative PD never exceeds 100%.
        survival = np.ones(n_months)
        marginal_pd = np.zeros(n_months)
        for t in range(n_months):
            if t == 0:
                survival[t] = 1.0
            else:
                survival[t] = survival[t - 1] * (1.0 - monthly_hazard[t - 1])
            marginal_pd[t] = monthly_hazard[t] * survival[t]

        return marginal_pd
    else:
        # Vectorized for portfolio
        n_loans = len(pd_12m)
        max_remaining = int(np.max(remaining_months))
        max_months = min(max_remaining, len(seasoning_curve))

        # Scale factor per loan
        curve_12m_sum = seasoning_curve[:12].sum()
        scale_factors = pd_12m / curve_12m_sum if curve_12m_sum > 0 else np.zeros(n_loans)

        # Build monthly hazard matrix: (n_loans, max_months)
        monthly_hazard = np.outer(scale_factors, seasoning_curve[:max_months])
        monthly_hazard = np.minimum(monthly_hazard, 0.05)

        # Zero out months beyond each loan's remaining term
        for i in range(n_loans):
            rem = int(remaining_months[i])
            if rem < max_months:
                monthly_hazard[i, rem:] = 0.0

        # Apply survival function to convert hazard to marginal PD
        survival = np.ones((n_loans, max_months))
        marginal_pd = np.zeros((n_loans, max_months))
        for t in range(max_months):
            if t > 0:
                survival[:, t] = survival[:, t - 1] * (1.0 - monthly_hazard[:, t - 1])
            marginal_pd[:, t] = monthly_hazard[:, t] * survival[:, t]

        return marginal_pd


# ---------------------------------------------------------------------------
# ECL Calculation
# ---------------------------------------------------------------------------

def compute_loan_ecl(original_upb, annual_rate, loan_term_months,
                     current_age, pd_12m, lgd, discount_rate=None):
    """
    Compute lifetime Expected Credit Loss for a single loan.

    ECL = SUM over t: Marginal_PD(t) * LGD * EAD(t) * DF(t)

    Parameters
    ----------
    original_upb : float
        Original loan balance.
    annual_rate : float
        Annual interest rate.
    loan_term_months : int
        Total loan term in months.
    current_age : int
        Current loan age in months.
    pd_12m : float
        12-month probability of default.
    lgd : float
        Loss given default (assumed constant over life for simplicity).
    discount_rate : float, optional
        Annual discount rate. If None, uses the loan's contract rate
        (CECL allows the effective interest rate).

    Returns
    -------
    float
        Lifetime ECL in dollars.
    """
    remaining = max(loan_term_months - current_age, 0)
    if remaining == 0 or pd_12m <= 0 or lgd <= 0:
        return 0.0

    if discount_rate is None:
        discount_rate = annual_rate

    # Monthly discount rate
    monthly_dr = discount_rate / 12.0

    # Build PD term structure
    monthly_pd = build_pd_term_structure(pd_12m, remaining)
    n_months = len(monthly_pd)

    # Build EAD schedule
    future_ages = current_age + np.arange(1, n_months + 1)
    future_ages = np.minimum(future_ages, loan_term_months)
    ead_schedule = compute_scheduled_balance(
        original_upb, annual_rate, loan_term_months, future_ages
    )

    # Discount factors
    discount_factors = 1.0 / (1.0 + monthly_dr) ** np.arange(1, n_months + 1)

    # ECL = sum of: marginal_pd * lgd * ead * discount_factor
    ecl = np.sum(monthly_pd * lgd * ead_schedule * discount_factors)

    return ecl


def compute_portfolio_ecl(loans_df, pd_predictions, lgd_predictions,
                          scenario_label="Baseline"):
    """
    Compute ECL for an entire portfolio of loans.

    This is the core CECL calculation. For each loan, we project
    lifetime losses using the PD model output, LGD estimate, and
    amortization-based EAD.

    Parameters
    ----------
    loans_df : pd.DataFrame
        Loan-level data with columns: original_upb, original_interest_rate,
        original_loan_term, loan_age (current age in months).
    pd_predictions : np.array
        12-month PD for each loan from the PD model.
    lgd_predictions : np.array
        LGD estimate for each loan from the LGD model.
    scenario_label : str
        Label for the economic scenario.

    Returns
    -------
    pd.DataFrame
        Loan-level ECL results.
    dict
        Portfolio-level summary.
    """
    print(f"\n  Computing ECL for scenario: {scenario_label}")
    print(f"  Loans: {len(loans_df):,}")

    t0 = time.time()
    n_loans = len(loans_df)

    # Extract loan characteristics as numpy arrays for speed
    upb = loans_df["original_upb"].values.astype(float)
    rate = loans_df["original_interest_rate"].values.astype(float) / 100.0
    term = loans_df["original_loan_term"].values.astype(float)

    # Current age: use median seasoning if loan_age not available.
    # For a static portfolio analysis, we assume loans are at origination
    # (age=0) since we're computing day-one CECL reserves.
    if "loan_age" in loans_df.columns:
        age = loans_df["loan_age"].values.astype(float)
    else:
        age = np.zeros(n_loans)

    pd_arr = np.asarray(pd_predictions, dtype=float)
    lgd_arr = np.asarray(lgd_predictions, dtype=float)

    # Handle NaN and invalid values in loan characteristics.
    # Even a single NaN will poison the entire portfolio sum.
    upb = np.nan_to_num(upb, nan=0.0)
    rate = np.nan_to_num(rate, nan=0.06)       # Default to 6% if missing
    term = np.nan_to_num(term, nan=360.0)       # Default to 30-year if missing
    age = np.nan_to_num(age, nan=0.0)
    pd_arr = np.nan_to_num(pd_arr, nan=0.0)
    lgd_arr = np.nan_to_num(lgd_arr, nan=0.0)

    # Clip inputs to valid ranges
    pd_arr = np.clip(pd_arr, 0.0, 1.0)
    lgd_arr = np.clip(lgd_arr, 0.0, 1.5)
    rate = np.clip(rate, 0.001, 0.20)  # Rate between 0.1% and 20%
    term = np.clip(term, 12, 480)      # Term between 1 and 40 years
    age = np.clip(age, 0, 480)

    # Compute ECL for each loan
    ecl_values = np.zeros(n_loans)
    current_balance = np.zeros(n_loans)

    for i in range(n_loans):
        ecl_values[i] = compute_loan_ecl(
            original_upb=upb[i],
            annual_rate=rate[i],
            loan_term_months=int(term[i]),
            current_age=int(age[i]),
            pd_12m=pd_arr[i],
            lgd=lgd_arr[i],
        )
        # Current balance for ECL rate calculation
        current_balance[i] = compute_scheduled_balance(
            upb[i], rate[i], int(term[i]), int(age[i])
        )

    elapsed = time.time() - t0

    # Build loan-level results
    results = pd.DataFrame({
        "original_upb": upb,
        "current_balance": current_balance,
        "pd_12m": pd_arr,
        "lgd": lgd_arr,
        "ecl_dollars": ecl_values,
    })
    results["ecl_rate"] = np.where(
        results["current_balance"] > 0,
        results["ecl_dollars"] / results["current_balance"],
        0.0,
    )

    # Portfolio summary
    total_balance = current_balance.sum()
    total_ecl = ecl_values.sum()
    portfolio_ecl_rate = total_ecl / total_balance if total_balance > 0 else 0

    summary = {
        "scenario": scenario_label,
        "n_loans": n_loans,
        "total_balance": total_balance,
        "total_ecl": total_ecl,
        "portfolio_ecl_rate": portfolio_ecl_rate,
        "mean_pd": pd_arr.mean(),
        "mean_lgd": lgd_arr.mean(),
        "mean_loan_ecl": ecl_values.mean(),
        "median_loan_ecl": np.median(ecl_values),
    }

    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"  Portfolio Summary ({scenario_label}):")
    print(f"    Total Outstanding Balance: ${total_balance/1e9:,.2f}B")
    print(f"    Total ECL:                 ${total_ecl/1e6:,.1f}M")
    print(f"    Portfolio ECL Rate:         {portfolio_ecl_rate*100:.2f}%")
    print(f"    Mean PD (12-month):         {pd_arr.mean()*100:.2f}%")
    print(f"    Mean LGD:                   {lgd_arr.mean()*100:.2f}%")

    return results, summary


def compute_scenario_weighted_ecl(summaries, weights):
    """
    Compute scenario-weighted ECL per CECL requirements.

    CECL requires banks to consider multiple economic scenarios and
    weight them by their probability of occurrence.

    Typical weights:
        Baseline: 50% (most likely outcome)
        Adverse: 30% (downside risk)
        Optimistic: 20% (upside possibility)

    Parameters
    ----------
    summaries : list of dict
        Portfolio summaries from compute_portfolio_ecl for each scenario.
    weights : dict
        Scenario name -> probability weight. Must sum to 1.0.

    Returns
    -------
    dict
        Weighted ECL summary.
    """
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"  WARNING: Scenario weights sum to {total_weight:.2f}, not 1.0")

    weighted_ecl = 0.0
    print(f"\n  Scenario-Weighted ECL:")
    print(f"  {'Scenario':<20s} {'Weight':>8s} {'ECL ($M)':>12s} {'ECL Rate':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*10}")

    for s in summaries:
        scenario = s["scenario"]
        weight = weights.get(scenario, 0.0)
        ecl = s["total_ecl"]
        rate = s["portfolio_ecl_rate"]
        weighted_ecl += weight * ecl
        print(f"  {scenario:<20s} {weight:>8.0%} ${ecl/1e6:>11,.1f} {rate*100:>9.2f}%")

    total_balance = summaries[0]["total_balance"]
    weighted_rate = weighted_ecl / total_balance if total_balance > 0 else 0

    print(f"  {'-'*52}")
    print(f"  {'Weighted ECL':<20s} {'':>8s} ${weighted_ecl/1e6:>11,.1f} {weighted_rate*100:>9.2f}%")

    return {
        "weighted_ecl": weighted_ecl,
        "weighted_ecl_rate": weighted_rate,
        "total_balance": total_balance,
    }