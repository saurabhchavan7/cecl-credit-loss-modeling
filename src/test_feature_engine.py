"""
Test Feature Engineering on a Single Quarter (2006Q1)
=====================================================

Before running the full pipeline on 265 million rows across 12 quarters,
we validate every step on ONE quarter (2006Q1, ~18 million rows, 253K loans).

This script tests:
1. Column availability in parquet files
2. Static feature extraction (groupby.first)
3. Default flag computation (90+ DPD and zero balance codes)
4. LGD/EAD calculation (second-to-last observation approach)
5. Derived feature creation (bins, flags, interactions)
6. FRED macro data fetch and merge
7. Output shape and value range sanity checks

Run this BEFORE running the full pipeline.

Author: Saurabh Chavan
"""

import os
import sys
import gc
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so we can import feature_engine
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from feature_engine import (
    ALL_NEEDED_COLS,
    STATIC_COLS,
    LOSS_COLS,
    FICO_BINS,
    FICO_LABELS,
    LTV_BINS,
    LTV_LABELS,
    DTI_BINS,
    DTI_LABELS,
    LGD_FLOOR,
    LGD_CAP,
    DEFAULT_ZERO_BALANCE_CODES,
    extract_loan_level_features,
    create_derived_features,
    fetch_fred_macro_data,
    merge_macro_features,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Test on 2006Q1 because we already profiled it in data_quality_check.py.
# Known values from Phase 1 to validate against:
#   Rows: 18,159,238
#   Unique loans: 253,043
#   Default rate: ~14.19%
#   Mean FICO: 722
#   Mean LGD (from test_lgd_debug.py): 0.414
TEST_QUARTER_FILE = project_root / "data" / "processed" / "quarterly" / "2006Q1.parquet"
TEST_QUARTER_LABEL = "2006Q1"

# Expected values from Phase 1 (used for validation, with tolerance)
EXPECTED_UNIQUE_LOANS = 253_043
EXPECTED_DEFAULT_RATE_LOW = 0.10   # at least 10% (crisis vintage)
EXPECTED_DEFAULT_RATE_HIGH = 0.20  # at most 20%
EXPECTED_MEAN_FICO_LOW = 690
EXPECTED_MEAN_FICO_HIGH = 760
# LGD range: After properly handling "C" (confidential) net_sale_proceeds
# values, the mean LGD for loans with valid recovery data should align
# with the 0.414 we saw in test_lgd_debug.py. We use a wider range to
# account for population differences across the full quarter.
EXPECTED_MEAN_LGD_LOW = 0.25
EXPECTED_MEAN_LGD_HIGH = 0.55

# Counters
passed = 0
failed = 0
warnings_count = 0


def check(condition, test_name, detail=""):
    """Report pass/fail for a single test."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {test_name}")
    else:
        failed += 1
        print(f"  FAIL: {test_name}")
    if detail:
        print(f"        {detail}")


def warn(message):
    """Report a non-fatal warning."""
    global warnings_count
    warnings_count += 1
    print(f"  WARN: {message}")


# ===========================================================================
# TEST 1: Parquet file exists and has required columns
# ===========================================================================
def test_parquet_columns():
    print("\n" + "=" * 70)
    print("TEST 1: Parquet file exists and has required columns")
    print("=" * 70)

    check(TEST_QUARTER_FILE.exists(), "Parquet file exists",
          str(TEST_QUARTER_FILE))

    if not TEST_QUARTER_FILE.exists():
        print("  Cannot continue without parquet file. Exiting.")
        sys.exit(1)

    # Read just the schema (no data loaded)
    pf_schema = pd.read_parquet(TEST_QUARTER_FILE, columns=None).columns.tolist()

    missing_cols = [c for c in ALL_NEEDED_COLS if c not in pf_schema]
    check(
        len(missing_cols) == 0,
        "All required columns present in parquet",
        f"Missing: {missing_cols}" if missing_cols else f"{len(ALL_NEEDED_COLS)} columns confirmed",
    )

    # If columns are missing, show what IS available for debugging
    if missing_cols:
        print(f"  Available columns ({len(pf_schema)}): {pf_schema[:20]}...")
        sys.exit(1)


# ===========================================================================
# TEST 2: Static feature extraction on a small sample
# ===========================================================================
def test_static_extraction_sample():
    print("\n" + "=" * 70)
    print("TEST 2: Static feature extraction on 500K-row sample")
    print("=" * 70)

    # Load a small chunk to test logic without loading the full quarter
    df_sample = pd.read_parquet(TEST_QUARTER_FILE, columns=ALL_NEEDED_COLS)
    # Take only loans that appear in the first 500K rows
    df_sample = df_sample.head(500_000)
    sample_loans = df_sample["loan_id"].nunique()
    print(f"  Sample: {len(df_sample):,} rows, {sample_loans:,} unique loans")

    # Extract static features
    df_sample.sort_values(["loan_id", "monthly_reporting_period"], inplace=True)
    static = df_sample.groupby("loan_id")[STATIC_COLS].first()

    check(
        len(static) == sample_loans,
        "Static extraction produces one row per loan",
        f"Expected {sample_loans}, got {len(static)}",
    )

    # Validate FICO range
    fico = pd.to_numeric(static["borrower_credit_score"], errors="coerce")
    valid_fico = fico.dropna()
    if len(valid_fico) > 0:
        check(
            valid_fico.min() >= 300 and valid_fico.max() <= 850,
            "FICO scores in valid range [300, 850]",
            f"Min: {valid_fico.min()}, Max: {valid_fico.max()}",
        )
    else:
        warn("No valid FICO scores in sample")

    # Validate LTV range
    ltv = pd.to_numeric(static["original_ltv"], errors="coerce")
    valid_ltv = ltv.dropna()
    if len(valid_ltv) > 0:
        check(
            valid_ltv.min() >= 1 and valid_ltv.max() <= 200,
            "LTV values in plausible range [1, 200]",
            f"Min: {valid_ltv.min()}, Max: {valid_ltv.max()}",
        )

    # Validate channel values
    channels = static["channel"].dropna().unique()
    check(
        all(c in {"R", "C", "B", "T", "9"} for c in channels),
        "Channel values are valid",
        f"Found: {sorted(channels)}",
    )

    # Validate origination_date format (MM/YYYY)
    sample_dates = static["origination_date"].dropna().head(5).tolist()
    print(f"  Sample origination dates: {sample_dates}")
    parsed = pd.to_datetime(static["origination_date"], format="%m/%Y", errors="coerce")
    pct_parsed = parsed.notna().mean()
    check(
        pct_parsed > 0.95,
        "Origination date parses as MM/YYYY",
        f"{pct_parsed:.1%} parsed successfully",
    )

    del df_sample, static
    gc.collect()


# ===========================================================================
# TEST 3: Full single-quarter extraction
# ===========================================================================
def test_full_quarter_extraction():
    print("\n" + "=" * 70)
    print("TEST 3: Full single-quarter loan-level extraction (2006Q1)")
    print("=" * 70)
    print("  This will load the full 2006Q1 parquet (~18M rows).")
    print("  Expected time: 1-5 minutes depending on machine.")

    t_start = time.time()
    loan_df = extract_loan_level_features(TEST_QUARTER_FILE, TEST_QUARTER_LABEL)
    elapsed = time.time() - t_start

    print(f"\n  Extraction completed in {elapsed:.1f} seconds")

    # --- Validate loan count ---
    n_loans = len(loan_df)
    tolerance = 0.02  # allow 2% deviation from expected
    check(
        abs(n_loans - EXPECTED_UNIQUE_LOANS) / EXPECTED_UNIQUE_LOANS < tolerance,
        "Loan count matches expected",
        f"Got {n_loans:,}, expected ~{EXPECTED_UNIQUE_LOANS:,} "
        f"(diff: {abs(n_loans - EXPECTED_UNIQUE_LOANS):,})",
    )

    # --- Validate default rate ---
    default_rate = loan_df["default_flag"].mean()
    check(
        EXPECTED_DEFAULT_RATE_LOW <= default_rate <= EXPECTED_DEFAULT_RATE_HIGH,
        f"Default rate in expected range [{EXPECTED_DEFAULT_RATE_LOW}, {EXPECTED_DEFAULT_RATE_HIGH}]",
        f"Got {default_rate:.4f}",
    )

    # --- Validate mean FICO ---
    mean_fico = loan_df["borrower_credit_score"].mean()
    check(
        EXPECTED_MEAN_FICO_LOW <= mean_fico <= EXPECTED_MEAN_FICO_HIGH,
        f"Mean FICO in expected range [{EXPECTED_MEAN_FICO_LOW}, {EXPECTED_MEAN_FICO_HIGH}]",
        f"Got {mean_fico:.1f}",
    )

    # --- Validate LGD for defaulted loans ---
    defaulted = loan_df[loan_df["default_flag"] == 1]
    n_defaulted = len(defaulted)
    check(n_defaulted > 0, "Defaulted loans exist", f"Count: {n_defaulted:,}")

    if n_defaulted > 0:
        lgd_valid = defaulted["lgd"].dropna()
        if len(lgd_valid) > 0:
            mean_lgd = lgd_valid.mean()
            check(
                EXPECTED_MEAN_LGD_LOW <= mean_lgd <= EXPECTED_MEAN_LGD_HIGH,
                f"Mean LGD in expected range [{EXPECTED_MEAN_LGD_LOW}, {EXPECTED_MEAN_LGD_HIGH}]",
                f"Got {mean_lgd:.4f}",
            )
            check(
                lgd_valid.min() >= LGD_FLOOR,
                f"LGD floor applied (>= {LGD_FLOOR})",
                f"Min LGD: {lgd_valid.min():.4f}",
            )
            check(
                lgd_valid.max() <= LGD_CAP,
                f"LGD cap applied (<= {LGD_CAP})",
                f"Max LGD: {lgd_valid.max():.4f}",
            )

        ead_valid = defaulted["ead"].dropna()
        if len(ead_valid) > 0:
            # Some defaulted loans may have zero EAD if the second-to-last
            # observation also had zero UPB (unusual termination). These
            # are excluded from LGD calculation but remain in the dataset.
            n_zero_ead = (ead_valid == 0).sum()
            n_positive_ead = (ead_valid > 0).sum()
            print(f"\n  EAD diagnostic:")
            print(f"    Total defaulted with EAD data: {len(ead_valid):,}")
            print(f"    Positive EAD: {n_positive_ead:,}")
            print(f"    Zero EAD: {n_zero_ead:,}")

            ead_positive = ead_valid[ead_valid > 0]
            check(
                n_positive_ead > n_defaulted * 0.95,
                "At least 95% of defaulted loans have positive EAD",
                f"{n_positive_ead:,} / {len(ead_valid):,} "
                f"({100*n_positive_ead/len(ead_valid):.1f}%)",
            )
            if len(ead_positive) > 0:
                check(
                    10_000 < ead_positive.mean() < 1_000_000,
                    "Mean EAD (positive only) in plausible range [$10K, $1M]",
                    f"Mean EAD: ${ead_positive.mean():,.0f}",
                )

        # LGD diagnostic: check how many loans had missing recovery data
        # (net_sale_proceeds = "C" in raw data)
        lgd_valid = defaulted["lgd"].dropna()
        n_lgd_valid = len(lgd_valid)
        n_lgd_null = defaulted["lgd"].isna().sum() - (len(defaulted) - n_defaulted)
        print(f"\n  LGD diagnostic:")
        print(f"    Defaulted loans: {n_defaulted:,}")
        print(f"    Valid LGD: {n_lgd_valid:,}")
        print(f"    Null LGD (zero EAD or missing recovery): "
              f"{defaulted['lgd'].isna().sum() - (len(loan_df) - n_defaulted):,}")

    # --- Validate derived features ---
    print("\n  Checking derived features...")

    check("fico_bucket" in loan_df.columns, "fico_bucket created")
    check("ltv_bucket" in loan_df.columns, "ltv_bucket created")
    check("dti_bucket" in loan_df.columns, "dti_bucket created")
    check("has_coborrower" in loan_df.columns, "has_coborrower created")
    check("has_mortgage_insurance" in loan_df.columns, "has_mortgage_insurance created")
    check("is_investment_property" in loan_df.columns, "is_investment_property created")
    check("is_cashout_refi" in loan_df.columns, "is_cashout_refi created")
    check("fico_x_ltv" in loan_df.columns, "fico_x_ltv interaction created")
    check("origination_year" in loan_df.columns, "origination_year created")
    check("origination_quarter" in loan_df.columns, "origination_quarter label set")

    # Validate binary flags are actually binary
    binary_flags = [
        "has_coborrower", "has_mortgage_insurance", "is_first_time_buyer",
        "is_investment_property", "is_second_home", "is_cashout_refi",
        "is_refi_nocashout", "channel_broker", "channel_correspondent",
        "channel_retail", "is_condo", "is_manufactured_housing",
        "is_multi_unit", "dti_missing", "fico_missing",
    ]
    for flag in binary_flags:
        if flag in loan_df.columns:
            unique_vals = set(loan_df[flag].dropna().unique())
            check(
                unique_vals.issubset({0, 1}),
                f"{flag} is binary (0/1)",
                f"Unique values: {unique_vals}",
            )

    # Validate FICO buckets match expected labels
    if "fico_bucket" in loan_df.columns:
        actual_buckets = set(loan_df["fico_bucket"].dropna().unique())
        expected_buckets = set(FICO_LABELS)
        check(
            actual_buckets.issubset(expected_buckets),
            "FICO bucket labels are correct",
            f"Found: {sorted(actual_buckets)}",
        )

    # --- Cross-validation: default rate by FICO bucket should be monotonic ---
    # Lower FICO = higher default rate. This is a fundamental credit risk
    # relationship. If this fails, something is wrong with either the default
    # flag or the FICO bucketing.
    print("\n  Cross-validating default rate by FICO bucket (should be monotonic)...")
    if "fico_bucket" in loan_df.columns:
        fico_default = (
            loan_df.groupby("fico_bucket", observed=True)["default_flag"]
            .mean()
            .reindex(FICO_LABELS)
        )
        print(f"  Default rates by FICO bucket:")
        for bucket, rate in fico_default.items():
            if pd.notna(rate):
                print(f"    {bucket:>10s}: {rate:.4f}")

        # Check monotonicity (lower FICO bucket should have higher default rate)
        rates = fico_default.dropna().values
        if len(rates) >= 2:
            is_monotonic = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
            check(
                is_monotonic,
                "Default rate is monotonically decreasing with higher FICO",
                "This confirms the default flag and FICO buckets are correct",
            )
            if not is_monotonic:
                warn("Non-monotonic default rate by FICO. Review default definition.")

    # --- Print column inventory ---
    print(f"\n  Output columns ({len(loan_df.columns)}):")
    for i, col in enumerate(loan_df.columns):
        dtype = loan_df[col].dtype
        n_null = loan_df[col].isna().sum()
        pct_null = 100 * n_null / len(loan_df)
        print(f"    {i+1:>3d}. {col:<35s} {str(dtype):<12s} "
              f"null: {n_null:>8,} ({pct_null:.1f}%)")

    # Free memory
    del loan_df
    gc.collect()

    return True


# ===========================================================================
# TEST 4: FRED macro data fetch
# ===========================================================================
def test_fred_macro():
    print("\n" + "=" * 70)
    print("TEST 4: FRED macro data fetch")
    print("=" * 70)

    # Load FRED API key
    try:
        import dotenv
        dotenv.load_dotenv(project_root / ".env")
    except ImportError:
        warn("python-dotenv not installed. Trying os.environ directly.")

    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        warn("FRED_API_KEY not found. Skipping macro data test.")
        warn("Set FRED_API_KEY in .env or environment to test macro features.")
        return

    check(len(fred_key) > 10, "FRED API key loaded", f"Key length: {len(fred_key)}")

    # Fetch macro data
    try:
        macro_df = fetch_fred_macro_data(fred_key)

        check(macro_df is not None, "Macro data fetched successfully")
        check(len(macro_df) > 200, "Sufficient macro observations",
              f"Got {len(macro_df)} monthly rows")

        # Validate expected columns exist
        expected_macro_cols = [
            "unemployment_rate", "fed_funds_rate", "mortgage_rate_30y",
            "hpi_national", "treasury_10y", "baa_spread",
            "unemployment_change_12m", "hpi_change_12m_pct",
        ]
        for col in expected_macro_cols:
            check(col in macro_df.columns, f"Macro column exists: {col}")

        # Validate unemployment rate range (should be 3-15% for US history)
        ur = macro_df["unemployment_rate"].dropna()
        if len(ur) > 0:
            check(
                2.0 < ur.min() < 5.0 and 8.0 < ur.max() < 16.0,
                "Unemployment rate in historical range",
                f"Min: {ur.min():.1f}%, Max: {ur.max():.1f}%",
            )

        # Validate date range covers our loan origination period (2005-2007)
        check(
            macro_df.index.min().year <= 2005,
            "Macro data starts before 2005",
            f"Starts: {macro_df.index.min()}",
        )
        check(
            macro_df.index.max().year >= 2010,
            "Macro data extends to at least 2010",
            f"Ends: {macro_df.index.max()}",
        )

        # Save as cache for the full pipeline
        cache_path = project_root / "data" / "processed" / "macro" / "fred_macro_monthly.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        macro_df.to_csv(cache_path)
        print(f"\n  Saved macro cache to {cache_path}")
        print(f"  The full pipeline will reuse this cached file.")

    except Exception as e:
        check(False, "Macro data fetch", f"Error: {e}")
        traceback.print_exc()


# ===========================================================================
# TEST 5: Macro merge on sample data
# ===========================================================================
def test_macro_merge_sample():
    print("\n" + "=" * 70)
    print("TEST 5: Macro merge on synthetic sample")
    print("=" * 70)

    # Check if cached macro data exists from Test 4
    cache_path = project_root / "data" / "processed" / "macro" / "fred_macro_monthly.csv"
    if not cache_path.exists():
        warn("No cached macro data. Run Test 4 first. Skipping merge test.")
        return

    macro_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # Create a small synthetic loan dataset to test the merge logic
    synthetic_loans = pd.DataFrame({
        "loan_id": ["LOAN_A", "LOAN_B", "LOAN_C", "LOAN_D"],
        "origination_date": ["01/2005", "06/2006", "03/2007", "12/2007"],
        "borrower_credit_score": [720, 650, 780, 600],
        "default_flag": [0, 1, 0, 1],
    })
    synthetic_loans.index = synthetic_loans["loan_id"]

    merged = merge_macro_features(synthetic_loans, macro_df)

    check(
        len(merged) == len(synthetic_loans),
        "Merge preserves loan count",
        f"Input: {len(synthetic_loans)}, Output: {len(merged)}",
    )

    check(
        "unemployment_rate" in merged.columns,
        "unemployment_rate merged into loan data",
    )

    check(
        "fico_x_unemployment" in merged.columns,
        "fico_x_unemployment interaction created",
    )

    # Validate that each loan got the correct month's unemployment
    n_matched = merged["unemployment_rate"].notna().sum()
    check(
        n_matched == len(merged),
        "All synthetic loans matched to macro data",
        f"Matched: {n_matched} / {len(merged)}",
    )

    # Print the merged result for visual inspection
    print("\n  Merged sample (verify macro values make sense):")
    display_cols = [
        "origination_date", "borrower_credit_score",
        "unemployment_rate", "mortgage_rate_30y", "hpi_national",
        "fico_x_unemployment",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    for _, row in merged.iterrows():
        parts = [f"{c}={row[c]}" for c in display_cols]
        print(f"    {', '.join(parts)}")

    del merged
    gc.collect()


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - FEATURE ENGINEERING TEST SUITE")
    print("=" * 70)
    print(f"Test quarter: {TEST_QUARTER_LABEL}")
    print(f"Parquet path: {TEST_QUARTER_FILE}")
    print(f"Project root: {project_root}")

    t_start = time.time()

    # Run tests in order. Each builds on the previous.
    test_parquet_columns()       # Quick: validates data is accessible
    test_static_extraction_sample()  # Quick: tests logic on 500K rows
    test_full_quarter_extraction()   # Slow: full 18M row extraction
    test_fred_macro()            # Medium: depends on internet + API
    test_macro_merge_sample()    # Quick: tests merge logic

    elapsed = time.time() - t_start

    # --- Final Report ---
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Warnings: {warnings_count}")
    print(f"  Time:     {elapsed:.1f} seconds")

    if failed == 0:
        print("\n  ALL TESTS PASSED.")
        print("  Safe to run the full pipeline: python src/feature_engine.py")
    else:
        print(f"\n  {failed} TEST(S) FAILED.")
        print("  Fix the issues above before running the full pipeline.")

    sys.exit(0 if failed == 0 else 1)