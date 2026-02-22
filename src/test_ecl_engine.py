"""
Test EAD & ECL Engine
======================

Validates amortization, PD term structure, and ECL calculations
on known examples before running on the full portfolio.

Author: Saurabh Chavan
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ecl_engine import (
    compute_scheduled_balance,
    compute_ead_schedule,
    build_pd_term_structure,
    compute_loan_ecl,
    compute_portfolio_ecl,
    compute_scenario_weighted_ecl,
)

passed = 0
failed = 0


def check(condition, test_name, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {test_name}")
    else:
        failed += 1
        print(f"  FAIL: {test_name}")
    if detail:
        print(f"        {detail}")


# ===========================================================================
# TEST 1: Amortization schedule correctness
# ===========================================================================
def test_amortization():
    print("\n" + "=" * 70)
    print("TEST 1: Amortization schedule correctness")
    print("=" * 70)

    # Known example: $200,000 loan, 6% rate, 30-year term
    # At month 0: balance = $200,000
    # At month 360: balance = $0
    upb = 200_000.0
    rate = 0.06
    term = 360

    # Balance at origination (month 0)
    bal_0 = compute_scheduled_balance(upb, rate, term, 0)
    check(
        abs(bal_0 - upb) < 1.0,
        "Balance at month 0 equals original UPB",
        f"Balance: ${bal_0:,.2f}, Expected: ${upb:,.2f}",
    )

    # Balance at maturity (month 360)
    bal_360 = compute_scheduled_balance(upb, rate, term, 360)
    check(
        abs(bal_360) < 1.0,
        "Balance at maturity is ~$0",
        f"Balance: ${bal_360:,.2f}",
    )

    # Balance at month 180 (halfway through a 30-year mortgage)
    # After 15 years of a 30-year 6% loan, approximately $142K remains
    # (most early payments go to interest, not principal)
    bal_180 = compute_scheduled_balance(upb, rate, term, 180)
    check(
        130_000 < bal_180 < 155_000,
        "Balance at month 180 is in plausible range ($130K-$155K)",
        f"Balance: ${bal_180:,.2f}",
    )

    # Balance should monotonically decrease
    balances = [compute_scheduled_balance(upb, rate, term, t) for t in range(0, 361, 12)]
    is_decreasing = all(balances[i] >= balances[i+1] for i in range(len(balances)-1))
    check(is_decreasing, "Balance monotonically decreases over time")

    # Vectorized version should match scalar
    upb_arr = np.array([200_000, 300_000, 150_000])
    rate_arr = np.array([0.06, 0.05, 0.07])
    term_arr = np.array([360, 360, 180])
    ages = np.array([12, 24, 6])

    bal_vec = compute_scheduled_balance(upb_arr, rate_arr, term_arr, ages)
    bal_scalar_0 = compute_scheduled_balance(200_000, 0.06, 360, 12)
    check(
        abs(bal_vec[0] - bal_scalar_0) < 0.01,
        "Vectorized matches scalar computation",
        f"Vec: ${bal_vec[0]:,.2f}, Scalar: ${bal_scalar_0:,.2f}",
    )


# ===========================================================================
# TEST 2: EAD schedule
# ===========================================================================
def test_ead_schedule():
    print("\n" + "=" * 70)
    print("TEST 2: EAD schedule projection")
    print("=" * 70)

    ead = compute_ead_schedule(200_000, 0.06, 360, 0, 12)

    check(len(ead) == 12, "EAD schedule has 12 months", f"Got {len(ead)}")
    check(ead[0] < 200_000, "First month EAD < original UPB")
    check(ead[-1] < ead[0], "EAD decreases over projection horizon")
    check(all(ead > 0), "All EAD values are positive")

    # Vectorized
    upb_arr = np.array([200_000, 300_000])
    rate_arr = np.array([0.06, 0.05])
    term_arr = np.array([360, 360])
    age_arr = np.array([0, 12])

    ead_vec = compute_ead_schedule(upb_arr, rate_arr, term_arr, age_arr, 12)
    check(ead_vec.shape == (2, 12), "Vectorized EAD shape is (2, 12)",
          f"Got {ead_vec.shape}")
    check(ead_vec[1, 0] < 300_000, "Second loan EAD starts below original UPB")


# ===========================================================================
# TEST 3: PD term structure
# ===========================================================================
def test_pd_term_structure():
    print("\n" + "=" * 70)
    print("TEST 3: PD term structure")
    print("=" * 70)

    # 12-month PD of 10%
    monthly_pd = build_pd_term_structure(0.10, 360)

    check(len(monthly_pd) > 0, "PD term structure is non-empty")
    check(len(monthly_pd) <= 360, "PD term structure length <= 360")
    check(all(monthly_pd >= 0), "All monthly PD values are non-negative")
    check(all(monthly_pd <= 0.05), "All monthly PD values capped at 5%")

    # First 12 months should sum approximately to the annual PD
    sum_12m = monthly_pd[:12].sum()
    check(
        0.05 < sum_12m < 0.20,
        "First 12 months PD sum is in reasonable range",
        f"Sum: {sum_12m:.4f} (target ~0.10)",
    )

    # PD should eventually decline for long remaining terms
    if len(monthly_pd) > 60:
        check(
            monthly_pd[59] < monthly_pd[20],
            "PD at month 60 < PD at month 20 (seasoning decline)",
            f"Month 20: {monthly_pd[20]:.6f}, Month 60: {monthly_pd[59]:.6f}",
        )

    # Zero PD should produce all-zero term structure
    zero_pd = build_pd_term_structure(0.0, 360)
    check(all(zero_pd == 0), "Zero PD produces all-zero term structure")

    # Vectorized
    pd_arr = np.array([0.05, 0.10, 0.20])
    rem_arr = np.array([300, 360, 180])
    monthly_pd_vec = build_pd_term_structure(pd_arr, rem_arr)

    check(monthly_pd_vec.shape[0] == 3, "Vectorized PD has 3 rows")
    # Higher PD should produce higher monthly PDs
    check(
        monthly_pd_vec[2, 0] > monthly_pd_vec[0, 0],
        "Higher annual PD -> higher monthly PD",
    )


# ===========================================================================
# TEST 4: Single loan ECL
# ===========================================================================
def test_single_loan_ecl():
    print("\n" + "=" * 70)
    print("TEST 4: Single loan ECL calculation")
    print("=" * 70)

    # Example: $200K loan, 6% rate, 30-year, brand new, 10% PD, 40% LGD
    ecl = compute_loan_ecl(
        original_upb=200_000,
        annual_rate=0.06,
        loan_term_months=360,
        current_age=0,
        pd_12m=0.10,
        lgd=0.40,
    )

    # ECL should be positive
    check(ecl > 0, "ECL is positive", f"ECL: ${ecl:,.2f}")

    # ECL should be less than the full loan amount
    check(ecl < 200_000, "ECL < original UPB", f"ECL: ${ecl:,.2f}")

    # Rough reasonableness: ECL ~ PD * LGD * balance * some duration factor
    # For a 30-year loan with 10% PD and 40% LGD, lifetime ECL should be
    # meaningful but not enormous. Expect $5K-$50K range.
    check(
        1_000 < ecl < 80_000,
        "ECL in reasonable range ($1K-$80K)",
        f"ECL: ${ecl:,.2f}",
    )

    # Higher PD should produce higher ECL
    ecl_high_pd = compute_loan_ecl(200_000, 0.06, 360, 0, 0.20, 0.40)
    check(
        ecl_high_pd > ecl,
        "Higher PD -> higher ECL",
        f"ECL at 10% PD: ${ecl:,.2f}, ECL at 20% PD: ${ecl_high_pd:,.2f}",
    )

    # Higher LGD should produce higher ECL
    ecl_high_lgd = compute_loan_ecl(200_000, 0.06, 360, 0, 0.10, 0.60)
    check(
        ecl_high_lgd > ecl,
        "Higher LGD -> higher ECL",
        f"ECL at 40% LGD: ${ecl:,.2f}, ECL at 60% LGD: ${ecl_high_lgd:,.2f}",
    )

    # Mature loan (age 300, only 60 months left) should have lower ECL
    ecl_mature = compute_loan_ecl(200_000, 0.06, 360, 300, 0.10, 0.40)
    check(
        ecl_mature < ecl,
        "Mature loan (60 months left) has lower ECL than new loan",
        f"New: ${ecl:,.2f}, Mature: ${ecl_mature:,.2f}",
    )

    # Zero PD loan should have zero ECL
    ecl_zero = compute_loan_ecl(200_000, 0.06, 360, 0, 0.0, 0.40)
    check(ecl_zero == 0.0, "Zero PD -> zero ECL")


# ===========================================================================
# TEST 5: Portfolio ECL on synthetic data
# ===========================================================================
def test_portfolio_ecl():
    print("\n" + "=" * 70)
    print("TEST 5: Portfolio ECL on synthetic portfolio")
    print("=" * 70)

    # Create a small synthetic portfolio
    np.random.seed(42)
    n = 1000
    synthetic = pd.DataFrame({
        "original_upb": np.random.uniform(100_000, 400_000, n),
        "original_interest_rate": np.random.uniform(4.0, 8.0, n),
        "original_loan_term": np.full(n, 360),
    })

    pd_preds = np.random.uniform(0.02, 0.15, n)
    lgd_preds = np.random.uniform(0.20, 0.50, n)

    results, summary = compute_portfolio_ecl(
        synthetic, pd_preds, lgd_preds, "Test Baseline"
    )

    check(len(results) == n, "Results have correct row count")
    check(summary["total_ecl"] > 0, "Total ECL is positive")
    check(summary["total_balance"] > 0, "Total balance is positive")
    check(
        0.001 < summary["portfolio_ecl_rate"] < 0.20,
        "Portfolio ECL rate in plausible range (0.1%-20%)",
        f"ECL rate: {summary['portfolio_ecl_rate']*100:.2f}%",
    )
    check(
        all(results["ecl_dollars"] >= 0),
        "All loan-level ECLs are non-negative",
    )

    # Scenario weighting
    results2, summary2 = compute_portfolio_ecl(
        synthetic, pd_preds * 2.0, lgd_preds * 1.3, "Test Adverse"
    )

    check(
        summary2["total_ecl"] > summary["total_ecl"],
        "Adverse scenario has higher ECL than baseline",
    )

    weighted = compute_scenario_weighted_ecl(
        [summary, summary2],
        {"Test Baseline": 0.60, "Test Adverse": 0.40},
    )

    check(
        weighted["weighted_ecl"] > summary["total_ecl"],
        "Weighted ECL > baseline ECL (adverse pulls it up)",
    )
    check(
        weighted["weighted_ecl"] < summary2["total_ecl"],
        "Weighted ECL < adverse ECL (baseline pulls it down)",
    )


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - EAD & ECL ENGINE TEST SUITE")
    print("=" * 70)
    t_start = time.time()

    test_amortization()
    test_ead_schedule()
    test_pd_term_structure()
    test_single_loan_ecl()
    test_portfolio_ecl()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Time:     {elapsed:.1f} seconds")

    if failed == 0:
        print("\n  ALL TESTS PASSED.")
        print("  Safe to run ECL on the full portfolio.")
    else:
        print(f"\n  {failed} TEST(S) FAILED.")

    sys.exit(0 if failed == 0 else 1)