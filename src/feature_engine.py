"""
Feature Engineering Module for CECL Credit Risk Modeling Project
================================================================

This module converts loan-month performance data (265 million rows across
12 quarterly parquet files) into a single loan-level dataset (~3.8 million rows)
suitable for PD, LGD, and EAD modeling.

The transformation follows this logic:
- Each loan has ONE row in the output dataset
- Static origination features come from the FIRST observation per loan
- Default flag comes from scanning ALL observations per loan
- LGD comes from the SECOND-TO-LAST and LAST observations (verified in test_lgd_debug.py)
- Macroeconomic features are merged by origination date from FRED

Memory constraint: Machine has 8-16 GB RAM and cannot load all 12 quarters
simultaneously. This module processes ONE quarter at a time.

Author: Saurabh Chavan
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default definition: loan is considered defaulted if it ever reaches any of
# these conditions. This follows industry standard (Basel/CECL).
# Delinquency status >= 3 means 90+ days past due.
# Zero balance codes indicating credit loss events:
#   02 = Third Party Sale
#   03 = Short Sale
#   06 = Repurchased (due to serious delinquency)
#   09 = REO Disposition (bank took ownership and sold)
#   15 = Non-Performing Note Sale
DEFAULT_ZERO_BALANCE_CODES = {"02", "03", "06", "09", "15"}

# FICO score bins: industry standard segmentation used by regulators and
# internal risk rating systems. Lower FICO = higher probability of default.
FICO_BINS = [0, 620, 660, 700, 740, 850]
FICO_LABELS = ["<620", "620-660", "660-700", "700-740", "740+"]

# LTV bins: Loan-to-Value ratio segments. Higher LTV means the borrower has
# less equity (skin in the game), increasing default risk. LTV > 80% typically
# requires private mortgage insurance (PMI).
LTV_BINS = [0, 60, 70, 80, 90, 200]
LTV_LABELS = ["<60", "60-70", "70-80", "80-90", "90+"]

# DTI bins: Debt-to-Income ratio segments. Higher DTI means a larger share of
# the borrower's income goes to debt payments, reducing their capacity to
# absorb financial shocks (job loss, rate increase, medical bills).
DTI_BINS = [0, 20, 30, 40, 50, 100]
DTI_LABELS = ["<20", "20-30", "30-40", "40-50", "50+"]

# LGD floor and cap: LGD below 0 means recovery exceeded the outstanding
# balance (rare, usually due to mortgage insurance proceeds). LGD above 1.5
# means costs exceeded 150% of the balance (extreme outlier, usually data
# issues or very long foreclosure timelines). Capping prevents these outliers
# from distorting model training.
LGD_FLOOR = 0.0
LGD_CAP = 1.5

# Columns needed from the quarterly parquet files.
# Identifiers and time
ID_COLS = ["loan_id", "monthly_reporting_period"]

# Static origination features (do not change over the life of the loan)
STATIC_COLS = [
    "channel",                    # R=Retail, C=Correspondent, B=Broker
    "original_interest_rate",     # Contract rate at origination
    "original_upb",               # Original unpaid principal balance (loan amount)
    "original_loan_term",         # Term in months (typically 360 for 30-year)
    "origination_date",           # When the loan was originated
    "first_payment_date",         # When first payment was due
    "original_ltv",               # Loan-to-Value at origination
    "original_cltv",              # Combined LTV (includes second liens)
    "number_of_borrowers",        # 1 or 2
    "dti",                        # Debt-to-Income ratio
    "borrower_credit_score",      # FICO score at origination
    "coborrower_credit_score",    # Co-borrower FICO (missing if single borrower)
    "first_time_home_buyer",      # Y/N flag
    "loan_purpose",               # P=Purchase, C=Cash-out Refi, N=No Cash-out Refi
    "property_type",              # SF=Single Family, CO=Condo, etc.
    "number_of_units",            # 1-4
    "occupancy_status",           # P=Primary, I=Investment, S=Second Home
    "property_state",             # Two-letter state code
    "msa",                        # Metropolitan Statistical Area code
    "zip_code_short",             # First 3 digits of ZIP
    "mortgage_insurance_pct",     # MI percentage (missing if no MI)
    "amortization_type",          # FRM=Fixed Rate Mortgage, ARM=Adjustable
]

# Performance columns needed for default and LGD calculation
PERFORMANCE_COLS = [
    "current_actual_upb",              # Current outstanding balance
    "current_loan_delinquency_status", # Months delinquent (0,1,2,3,...)
    "zero_balance_code",               # Why the loan left the pool
    "loan_age",                        # Months since origination
    "modification_flag",               # Y if loan was modified
]

# Loss fields: populated only after default and property disposition.
# These are ~99.9% missing across the dataset because most loans do not default.
LOSS_COLS = [
    "foreclosure_costs",
    "property_preservation_costs",
    "asset_recovery_costs",
    "misc_holding_expenses",
    "holding_taxes",
    "net_sale_proceeds",
    "credit_enhancement_proceeds",
    "repurchase_make_whole_proceeds",
    "other_foreclosure_proceeds",
]

ALL_NEEDED_COLS = ID_COLS + STATIC_COLS + PERFORMANCE_COLS + LOSS_COLS


def extract_loan_level_features(quarter_path, quarter_label):
    """
    Extract loan-level features from a single quarterly parquet file.

    This is the core function of Phase 2. It takes a parquet file containing
    loan-month observations (e.g., 18 million rows for 2006Q1) and produces
    a loan-level dataset (e.g., 253K rows for 2006Q1) with:
      - Static origination features (from first observation)
      - Default flag (from scanning all observations)
      - LGD and EAD (from second-to-last and last observations)
      - Derived features (bins, flags, interactions)

    Parameters
    ----------
    quarter_path : str or Path
        Path to the quarterly parquet file.
    quarter_label : str
        Label for logging (e.g., "2006Q1").

    Returns
    -------
    pd.DataFrame
        Loan-level dataset with one row per loan.
    """
    print(f"\n{'='*70}")
    print(f"Processing {quarter_label}: {quarter_path}")
    print(f"{'='*70}")
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load parquet and select only needed columns
    # ------------------------------------------------------------------
    # Parquet format supports column-level reads, so we only load the
    # columns we need. This is critical for memory management.
    print(f"  Loading parquet file...")
    df = pd.read_parquet(quarter_path, columns=ALL_NEEDED_COLS)
    n_rows = len(df)
    n_loans = df["loan_id"].nunique()
    print(f"  Loaded {n_rows:,} rows, {n_loans:,} unique loans")

    # ------------------------------------------------------------------
    # Step 2: Extract static origination features
    # ------------------------------------------------------------------
    # Static features come from the FIRST observation per loan, which
    # represents the loan's characteristics at origination. We sort by
    # reporting period to ensure we get the earliest observation.
    print(f"  Extracting static origination features...")
    df.sort_values(["loan_id", "monthly_reporting_period"], inplace=True)
    static = df.groupby("loan_id")[STATIC_COLS].first()
    print(f"  Static features extracted: {static.shape[0]:,} loans")

    # ------------------------------------------------------------------
    # Step 3: Compute default flag
    # ------------------------------------------------------------------
    # A loan is flagged as defaulted if AT ANY POINT during its life it:
    #   (a) Reaches 90+ days past due (delinquency status >= 3), OR
    #   (b) Has a zero balance code indicating a credit loss event
    #
    # This is a LIFETIME default flag, not a point-in-time indicator.
    # It answers: "Did this loan EVER default?"
    print(f"  Computing default flags...")

    # (a) Check maximum delinquency status across all observations.
    # Convert to numeric first; non-numeric values (like 'XX' for unknown)
    # become NaN, which we fill with 0 (assume current if unknown).
    dq_status = pd.to_numeric(
        df["current_loan_delinquency_status"], errors="coerce"
    )
    max_dq = dq_status.groupby(df["loan_id"]).max()
    ever_90dpd = (max_dq >= 3).astype(int)

    # (b) Check if loan ever had a default-related zero balance code.
    # Zero balance code is a string field. We check if any observation
    # for a loan has a code in our default set.
    def has_default_zb_code(series):
        """Check if any value in the series is a default zero balance code."""
        return int(series.isin(DEFAULT_ZERO_BALANCE_CODES).any())

    zb_default = (
        df.groupby("loan_id")["zero_balance_code"]
        .apply(has_default_zb_code)
    )

    # Combine: default if EITHER condition is met
    default_flag = ((ever_90dpd == 1) | (zb_default == 1)).astype(int)
    default_flag.name = "default_flag"

    # Also store max delinquency for potential use as a feature
    max_dq.name = "max_delinquency_status"

    print(f"  Default rate: {default_flag.mean():.4f} "
          f"({default_flag.sum():,} defaults out of {len(default_flag):,} loans)")

    # ------------------------------------------------------------------
    # Step 4: Compute LGD and EAD for defaulted loans
    # ------------------------------------------------------------------
    # LGD (Loss Given Default) measures how much the bank loses when a
    # borrower defaults, expressed as a fraction of the exposure.
    #
    # Key insight verified in test_lgd_debug.py:
    # Fannie Mae sets current_actual_upb to ZERO in the FINAL observation
    # when a loan is removed from the pool. Therefore:
    #   - EAD (Exposure at Default) = UPB from SECOND-TO-LAST observation
    #   - Loss fields = from the LAST observation (final disposition record)
    #
    # We must set loan_id as the DataFrame index BEFORE groupby, then use
    # .nth(-2) for second-to-last and .last() for the final observation.
    print(f"  Computing LGD and EAD for defaulted loans...")

    defaulted_loan_ids = default_flag[default_flag == 1].index
    n_defaulted = len(defaulted_loan_ids)

    # Initialize empty LGD summary in case no valid data
    empty_lgd_cols = [
        "ead", "total_costs", "total_recovery", "total_loss",
        "lgd_raw", "lgd", "loan_age_at_default", "was_modified",
    ]

    if n_defaulted > 0:
        # Filter to only defaulted loans to save memory
        df_defaulted = df[df["loan_id"].isin(defaulted_loan_ids)].copy()
        df_defaulted.sort_values(
            ["loan_id", "monthly_reporting_period"], inplace=True
        )

        # Set loan_id as index for correct groupby behavior with .nth()
        df_defaulted.set_index("loan_id", inplace=True)

        # Filter out loans with fewer than 2 observations. These loans
        # have no second-to-last row, so we cannot compute EAD.
        obs_counts = df_defaulted.groupby(level="loan_id").size()
        loans_with_enough_obs = obs_counts[obs_counts >= 2].index
        n_skipped_few_obs = n_defaulted - len(loans_with_enough_obs)
        if n_skipped_few_obs > 0:
            print(f"    Skipping {n_skipped_few_obs:,} defaulted loans "
                  f"with < 2 observations")
        df_defaulted = df_defaulted.loc[
            df_defaulted.index.isin(loans_with_enough_obs)
        ]

        if len(loans_with_enough_obs) == 0:
            print(f"  No defaulted loans with >= 2 observations in {quarter_label}")
            lgd_summary = pd.DataFrame(columns=empty_lgd_cols)
        else:
            # EAD: second-to-last observation's UPB
            # This is the balance just before the loan was removed from the pool
            ead_series = (
                df_defaulted.groupby(level="loan_id")["current_actual_upb"]
                .nth(-2)
            )

            # Loss fields: from the last observation (disposition record)
            loss_fields = LOSS_COLS + ["current_actual_upb"]
            last_obs = df_defaulted.groupby(level="loan_id")[loss_fields].last()

            # Loan age at default: from the last observation
            age_at_default = (
                df_defaulted.groupby(level="loan_id")["loan_age"].last()
            )
            age_at_default.name = "loan_age_at_default"

            # Modification flag: did the loan get modified before defaulting?
            was_modified = (
                df_defaulted.groupby(level="loan_id")["modification_flag"]
                .apply(lambda s: int((s == "Y").any()))
            )
            was_modified.name = "was_modified"

            # Build LGD dataset
            lgd_df = pd.DataFrame(index=ead_series.index)
            lgd_df["ead"] = pd.to_numeric(ead_series, errors="coerce")

            # Sum all cost fields.
            # These represent the total expenses the bank incurred during
            # the foreclosure and property disposition process.
            cost_cols = [
                "foreclosure_costs",
                "property_preservation_costs",
                "asset_recovery_costs",
                "misc_holding_expenses",
                "holding_taxes",
            ]
            for col in cost_cols:
                lgd_df[col] = pd.to_numeric(
                    last_obs[col], errors="coerce"
                ).fillna(0.0)
            lgd_df["total_costs"] = lgd_df[cost_cols].sum(axis=1)

            # Sum all recovery/proceeds fields.
            # These represent what the bank recovered from selling the property
            # and any insurance or credit enhancement proceeds.
            #
            # IMPORTANT: Fannie Mae stores net_sale_proceeds as "C" (string)
            # when the amount is confidential or credit-enhanced. When we
            # coerce to numeric, "C" becomes NaN. For these loans we cannot
            # compute a reliable LGD, so we mark them as NaN rather than
            # assuming zero recovery (which would artificially inflate LGD).
            recovery_cols = [
                "net_sale_proceeds",
                "credit_enhancement_proceeds",
                "repurchase_make_whole_proceeds",
                "other_foreclosure_proceeds",
            ]
            for col in recovery_cols:
                lgd_df[col] = pd.to_numeric(last_obs[col], errors="coerce")

            # Track which loans have valid net_sale_proceeds (numeric, not "C").
            # If net_sale_proceeds is NaN (was "C" or truly missing), the LGD
            # calculation is unreliable.
            has_valid_recovery = lgd_df["net_sale_proceeds"].notna()

            # Fill remaining NaN recovery fields with 0 only for loans that
            # have valid net_sale_proceeds (the primary recovery source).
            for col in recovery_cols:
                lgd_df.loc[has_valid_recovery, col] = (
                    lgd_df.loc[has_valid_recovery, col].fillna(0.0)
                )

            lgd_df["total_recovery"] = lgd_df[recovery_cols].sum(axis=1)
            # Loans without valid recovery data get NaN total_recovery
            lgd_df.loc[~has_valid_recovery, "total_recovery"] = np.nan

            # LGD calculation:
            # Total Loss = EAD - Total Recovery + Total Costs
            # LGD = Total Loss / EAD
            #
            # Interpretation:
            #   LGD = 0.40 means the bank lost 40% of the outstanding balance
            #   LGD = 0.00 means the bank recovered everything (rare)
            #   LGD > 1.00 means costs exceeded the balance (extreme cases)
            #   LGD < 0.00 means recovery exceeded balance (insurance payout)
            lgd_df["total_loss"] = (
                lgd_df["ead"] - lgd_df["total_recovery"] + lgd_df["total_costs"]
            )

            # Only compute LGD for loans with positive EAD and valid recovery
            # data. Zero EAD means the second-to-last observation also had
            # zero balance (unusual termination). Missing recovery means
            # net_sale_proceeds was "C" (confidential).
            valid_lgd_mask = (lgd_df["ead"] > 0) & has_valid_recovery
            lgd_df["lgd_raw"] = np.nan
            lgd_df.loc[valid_lgd_mask, "lgd_raw"] = (
                lgd_df.loc[valid_lgd_mask, "total_loss"]
                / lgd_df.loc[valid_lgd_mask, "ead"]
            )

            # Cap LGD to remove outliers
            # Floor at 0: negative LGD (recovery > balance) is set to 0
            # Cap at 1.5: extreme cost overruns are capped to prevent distortion
            lgd_df["lgd"] = lgd_df["lgd_raw"].clip(lower=LGD_FLOOR, upper=LGD_CAP)

            # Add age at default and modification flag
            lgd_df["loan_age_at_default"] = age_at_default
            lgd_df["was_modified"] = was_modified

            # Keep only the summary columns we need
            lgd_output_cols = [
                "ead", "total_costs", "total_recovery", "total_loss",
                "lgd_raw", "lgd", "loan_age_at_default", "was_modified",
            ]
            lgd_summary = lgd_df[lgd_output_cols]

            # Print diagnostics
            n_valid_lgd = lgd_df["lgd"].notna().sum()
            n_zero_ead = (lgd_df["ead"] == 0).sum()
            n_missing_recovery = (~has_valid_recovery).sum()

            print(f"  LGD computed for {len(lgd_summary):,} defaulted loans")
            print(f"    Valid LGD values: {n_valid_lgd:,}")
            print(f"    Zero EAD (excluded from LGD): {n_zero_ead:,}")
            print(f"    Missing recovery data (excluded from LGD): {n_missing_recovery:,}")
            if n_valid_lgd > 0:
                print(f"    Mean LGD (raw): {lgd_df['lgd_raw'].dropna().mean():.4f}")
                print(f"    Mean LGD (capped): {lgd_df['lgd'].dropna().mean():.4f}")
                print(f"    Median LGD (capped): {lgd_df['lgd'].dropna().median():.4f}")
            print(f"    Mean EAD: ${lgd_df['ead'].mean():,.0f}")

        # Free memory from the defaulted loan subset
        del df_defaulted
        gc.collect()
    else:
        # No defaults in this quarter (unlikely but handle gracefully)
        lgd_summary = pd.DataFrame(columns=empty_lgd_cols)
        print(f"  No defaults found in {quarter_label}")

    # ------------------------------------------------------------------
    # Step 5: Free the main dataframe from memory
    # ------------------------------------------------------------------
    del df
    gc.collect()

    # ------------------------------------------------------------------
    # Step 6: Combine static features, default flag, and LGD into one
    # loan-level dataset
    # ------------------------------------------------------------------
    print(f"  Assembling loan-level dataset...")
    loan_level = static.copy()
    loan_level["default_flag"] = default_flag
    loan_level["max_delinquency_status"] = max_dq

    # Left-join LGD data (only defaulted loans have LGD)
    loan_level = loan_level.join(lgd_summary, how="left")

    # ------------------------------------------------------------------
    # Step 7: Create derived features
    # ------------------------------------------------------------------
    print(f"  Creating derived features...")
    loan_level = create_derived_features(loan_level)

    # Add quarter label for tracking origination vintage
    loan_level["origination_quarter"] = quarter_label

    elapsed = time.time() - t_start
    print(f"  Completed {quarter_label} in {elapsed:.1f} seconds")
    print(f"  Output shape: {loan_level.shape}")

    return loan_level


def create_derived_features(df):
    """
    Create derived features from raw origination data.

    These features capture risk dimensions that the raw data does not
    directly express. Each feature is motivated by credit risk theory
    and industry practice.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-level dataset with static origination features.

    Returns
    -------
    pd.DataFrame
        Dataset with additional derived feature columns.
    """
    # ------------------------------------------------------------------
    # Binary flags derived from missingness patterns
    # ------------------------------------------------------------------
    # has_coborrower: Loans with two borrowers have dual income streams,
    # providing a buffer if one borrower loses their job. The co-borrower
    # credit score field is missing (~54%) when there is no co-borrower.
    df["has_coborrower"] = df["coborrower_credit_score"].notna().astype(int)

    # has_mortgage_insurance: MI protects the lender if the borrower
    # defaults. Loans with MI typically have higher LTV (>80%) but the
    # insurance reduces LGD. MI percentage is missing (~86%) when there
    # is no MI on the loan.
    df["has_mortgage_insurance"] = df["mortgage_insurance_pct"].notna().astype(int)

    # ------------------------------------------------------------------
    # Categorical bins for continuous risk drivers
    # ------------------------------------------------------------------
    # FICO bins: Industry standard segmentation. The <620 segment is
    # commonly called "subprime" and carries significantly higher default
    # risk. Each bin represents a meaningfully different risk tier.
    df["fico_bucket"] = pd.cut(
        df["borrower_credit_score"],
        bins=FICO_BINS,
        labels=FICO_LABELS,
        include_lowest=True,
    )

    # LTV bins: LTV is the ratio of loan amount to property value.
    # At 80% LTV, the borrower has 20% equity. At 90%+ LTV, the borrower
    # is highly leveraged and a small property value decline puts them
    # underwater (owing more than the property is worth).
    df["ltv_bucket"] = pd.cut(
        df["original_ltv"],
        bins=LTV_BINS,
        labels=LTV_LABELS,
        include_lowest=True,
    )

    # DTI bins: DTI measures payment burden. A borrower with 45% DTI
    # spends nearly half their gross income on debt payments, leaving
    # little room for unexpected expenses.
    df["dti_bucket"] = pd.cut(
        df["dti"],
        bins=DTI_BINS,
        labels=DTI_LABELS,
        include_lowest=True,
    )

    # ------------------------------------------------------------------
    # Binary categorical features
    # ------------------------------------------------------------------
    # is_first_time_buyer: First-time buyers lack homeownership experience
    # and may have less financial cushion, slightly increasing default risk.
    df["is_first_time_buyer"] = (df["first_time_home_buyer"] == "Y").astype(int)

    # is_investment_property: Investment properties have historically
    # higher default rates because borrowers are more willing to walk
    # away from an investment property than their primary residence
    # (strategic default). This was a major driver of losses in 2008.
    df["is_investment_property"] = (df["occupancy_status"] == "I").astype(int)

    # is_second_home: Second homes fall between primary and investment
    # in terms of default risk.
    df["is_second_home"] = (df["occupancy_status"] == "S").astype(int)

    # is_cashout_refi: Cash-out refinances extract equity from the home,
    # increasing the loan balance relative to property value. Borrowers
    # who repeatedly extract equity tend to have weaker financial profiles
    # and higher default rates than purchase borrowers.
    df["is_cashout_refi"] = (df["loan_purpose"] == "C").astype(int)

    # is_refi_nocashout: Standard rate/term refinance. Lower risk than
    # cash-out but different risk profile than purchase.
    df["is_refi_nocashout"] = (df["loan_purpose"] == "N").astype(int)

    # ------------------------------------------------------------------
    # Channel indicators
    # ------------------------------------------------------------------
    # Broker-originated loans (channel = B) historically had higher default
    # rates due to less rigorous underwriting and incentive misalignment
    # (broker gets paid at origination, bears no default risk).
    df["channel_broker"] = (df["channel"] == "B").astype(int)
    df["channel_correspondent"] = (df["channel"] == "C").astype(int)
    df["channel_retail"] = (df["channel"] == "R").astype(int)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    # fico_x_ltv: Captures the combined effect of borrower quality and
    # leverage. A low-FICO, high-LTV borrower is the riskiest combination:
    # weak credit AND little equity. This interaction is a standard feature
    # in credit risk scorecards.
    df["fico_x_ltv"] = df["borrower_credit_score"] * df["original_ltv"]

    # fico_x_dti: Low FICO + high DTI indicates a borrower who is both
    # credit-impaired and heavily leveraged on income. Double stress.
    df["fico_x_dti"] = df["borrower_credit_score"] * df["dti"]

    # ------------------------------------------------------------------
    # Vintage features
    # ------------------------------------------------------------------
    # Extract origination year and month from the origination date.
    # Vintage is a critical risk factor: loans originated during credit
    # booms (2005-2007) had systematically weaker underwriting.
    orig_date = pd.to_datetime(df["origination_date"], format="%m/%Y", errors="coerce")
    df["origination_year"] = orig_date.dt.year
    df["origination_month"] = orig_date.dt.month

    # ------------------------------------------------------------------
    # Property type flags
    # ------------------------------------------------------------------
    # Condos and manufactured housing have different risk profiles than
    # single-family homes due to market liquidity and collateral value
    # stability differences.
    df["is_condo"] = (df["property_type"] == "CO").astype(int)
    df["is_manufactured_housing"] = (df["property_type"] == "MH").astype(int)

    # ------------------------------------------------------------------
    # Multi-unit flag
    # ------------------------------------------------------------------
    # 2-4 unit properties (duplexes, triplexes, fourplexes) may have
    # rental income supporting the mortgage, but also carry different
    # risk characteristics than single-unit properties.
    df["is_multi_unit"] = (df["number_of_units"] > 1).astype(int)

    # ------------------------------------------------------------------
    # Missing value indicators
    # ------------------------------------------------------------------
    # In credit risk modeling, the REASON a value is missing can itself
    # be predictive. For example, missing DTI may indicate the lender
    # did not verify income (stated income loan), which was a risk
    # factor during the 2005-2007 period.
    df["dti_missing"] = df["dti"].isna().astype(int)
    df["fico_missing"] = df["borrower_credit_score"].isna().astype(int)

    return df


def fetch_fred_macro_data(fred_api_key, start_date="2000-01-01", end_date="2025-12-31"):
    """
    Fetch macroeconomic time series from the Federal Reserve Economic Data
    (FRED) API and return a monthly DataFrame.

    These macro variables serve two purposes in credit risk modeling:
    1. As features: economic conditions at origination influence default risk
       (e.g., loans originated during low unemployment have different risk
       profiles than those originated during recessions)
    2. For stress testing: macro scenarios drive PD and LGD projections

    Parameters
    ----------
    fred_api_key : str
        FRED API key (from .env file).
    start_date : str
        Start date for data fetch (YYYY-MM-DD).
    end_date : str
        End date for data fetch (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Monthly macro data indexed by date with columns for each series.
    """
    from fredapi import Fred

    fred = Fred(api_key=fred_api_key)

    # Series to fetch and their descriptions.
    # Each series is chosen because it is a known driver of credit risk.
    series_map = {
        "UNRATE": "unemployment_rate",
        # Unemployment is the single strongest macro predictor of mortgage
        # default. Job loss is the primary trigger for missed payments.

        "FEDFUNDS": "fed_funds_rate",
        # The federal funds rate drives short-term borrowing costs. When
        # the Fed raises rates, adjustable-rate mortgages become more
        # expensive and the economy tends to slow.

        "CPIAUCSL": "cpi_index",
        # Consumer Price Index. Inflation erodes purchasing power, making
        # it harder for borrowers to keep up with payments. Also used to
        # adjust nominal values.

        "MORTGAGE30US": "mortgage_rate_30y",
        # The 30-year fixed mortgage rate determines refinancing incentives.
        # When market rates are below the loan rate, borrowers refinance
        # (prepay). When above, borrowers are "locked in."

        "CSUSHPINSA": "hpi_national",
        # Case-Shiller National Home Price Index. Housing prices directly
        # drive LTV (and therefore LGD). Falling prices create negative
        # equity, which triggers strategic defaults.

        "GS10": "treasury_10y",
        # 10-Year Treasury yield. A benchmark for long-term interest rates
        # and overall economic outlook. Inversion (short > long) signals
        # recession risk.

        "BAA10Y": "baa_spread",
        # Moody's BAA corporate bond spread over 10-year Treasury.
        # Measures credit market stress. Wider spreads indicate higher
        # perceived default risk across the economy.

        "GDP": "gdp",
        # Gross Domestic Product. Overall economic output. GDP contraction
        # (recession) is associated with rising defaults across all segments.
    }

    print("\nFetching macroeconomic data from FRED...")
    macro_frames = {}

    for series_id, col_name in series_map.items():
        try:
            print(f"  Fetching {series_id} ({col_name})...")
            data = fred.get_series(
                series_id, observation_start=start_date, observation_end=end_date
            )
            macro_frames[col_name] = data
            print(f"    Retrieved {len(data)} observations")
        except Exception as e:
            print(f"    WARNING: Failed to fetch {series_id}: {e}")

    # Combine all series into a single DataFrame.
    # Different series have different frequencies (monthly, quarterly, daily).
    # We resample everything to monthly frequency using forward fill.
    macro_df = pd.DataFrame(macro_frames)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.resample("MS").last()  # Month-Start frequency, take last value
    macro_df = macro_df.ffill()  # Forward fill for any gaps

    # ------------------------------------------------------------------
    # Derived macro features
    # ------------------------------------------------------------------

    # 12-month change in unemployment rate.
    # A RISING unemployment rate is a stronger default predictor than the
    # level itself. A 2% increase signals economic deterioration.
    macro_df["unemployment_change_12m"] = (
        macro_df["unemployment_rate"] - macro_df["unemployment_rate"].shift(12)
    )

    # 12-month percentage change in house prices.
    # Falling home prices reduce borrower equity and increase LGD.
    # A 10% decline means a borrower with 10% equity is now underwater.
    macro_df["hpi_change_12m_pct"] = (
        macro_df["hpi_national"].pct_change(periods=12) * 100
    )

    # GDP growth (quarterly data resampled to monthly, so compute
    # quarter-over-quarter change where available).
    macro_df["gdp_growth_pct"] = macro_df["gdp"].pct_change(periods=3) * 100

    print(f"  Macro data shape: {macro_df.shape}")
    print(f"  Date range: {macro_df.index.min()} to {macro_df.index.max()}")

    return macro_df


def merge_macro_features(loan_df, macro_df):
    """
    Merge macroeconomic data at the time of loan origination.

    The economic environment when a loan is originated affects its risk
    profile. Loans originated during loose monetary policy (low rates,
    low unemployment) may have been underwritten with weaker standards,
    and the macro conditions at origination serve as proxy variables for
    the credit environment.

    Parameters
    ----------
    loan_df : pd.DataFrame
        Loan-level dataset with origination_date column.
    macro_df : pd.DataFrame
        Monthly macro data from FRED.

    Returns
    -------
    pd.DataFrame
        Loan-level dataset with macro features added.
    """
    print("\nMerging macroeconomic features at origination...")

    # Parse origination date to datetime for merge
    loan_df["orig_date_parsed"] = pd.to_datetime(
        loan_df["origination_date"], format="%m/%Y", errors="coerce"
    )

    # Create a merge key: first day of the origination month
    loan_df["macro_merge_key"] = loan_df["orig_date_parsed"].dt.to_period("M").dt.to_timestamp()

    # Prepare macro data with the same merge key
    macro_merge = macro_df.copy()
    macro_merge.index.name = "macro_merge_key"
    macro_merge = macro_merge.reset_index()

    # Merge on the month of origination
    merged = loan_df.merge(macro_merge, on="macro_merge_key", how="left")

    # Create interaction: fico_x_unemployment
    # Weak borrowers (low FICO) are disproportionately affected by rising
    # unemployment. This interaction captures that non-linear relationship.
    if "borrower_credit_score" in merged.columns and "unemployment_rate" in merged.columns:
        merged["fico_x_unemployment"] = (
            merged["borrower_credit_score"] * merged["unemployment_rate"]
        )

    # Drop temporary merge columns
    merged.drop(columns=["orig_date_parsed", "macro_merge_key"], inplace=True, errors="ignore")

    n_macro_matched = merged["unemployment_rate"].notna().sum()
    print(f"  Macro data matched for {n_macro_matched:,} / {len(merged):,} loans "
          f"({100*n_macro_matched/len(merged):.1f}%)")

    return merged


def run_feature_engineering(
    processed_dir,
    output_dir,
    fred_api_key=None,
    macro_cache_path=None,
):
    """
    Run the full feature engineering pipeline across all quarterly parquets.

    Processing strategy (memory-constrained):
    1. Fetch/load macro data once
    2. Process each quarter independently: load parquet, extract loan-level
       features, save as small parquet
    3. After all quarters, combine the loan-level parquets (3.8M rows fits
       in memory, unlike the 265M loan-month rows)
    4. Apply train/validation/test split by origination vintage

    Parameters
    ----------
    processed_dir : str or Path
        Directory containing quarterly parquet files.
    output_dir : str or Path
        Directory to save loan-level output files.
    fred_api_key : str, optional
        FRED API key. If None, macro features are skipped.
    macro_cache_path : str or Path, optional
        Path to cached macro CSV. If exists, skip FRED fetch.

    Returns
    -------
    pd.DataFrame
        Combined loan-level dataset with all features and split labels.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Fetch or load macro data
    # ------------------------------------------------------------------
    macro_dir = output_dir.parent / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    if macro_cache_path is None:
        macro_cache_path = macro_dir / "fred_macro_monthly.csv"

    macro_cache_path = Path(macro_cache_path)

    if macro_cache_path.exists():
        print(f"Loading cached macro data from {macro_cache_path}")
        macro_df = pd.read_csv(macro_cache_path, index_col=0, parse_dates=True)
        print(f"  Loaded macro data: {macro_df.shape}")
    elif fred_api_key:
        macro_df = fetch_fred_macro_data(fred_api_key)
        macro_df.to_csv(macro_cache_path)
        print(f"  Saved macro data to {macro_cache_path}")
    else:
        print("WARNING: No FRED API key and no cached macro data.")
        print("  Proceeding without macroeconomic features.")
        macro_df = None

    # ------------------------------------------------------------------
    # Step 2: Discover quarterly parquet files
    # ------------------------------------------------------------------
    quarterly_dir = processed_dir / "quarterly"
    parquet_files = sorted(quarterly_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {quarterly_dir}. "
            f"Run data_pipeline.py first."
        )

    print(f"\nFound {len(parquet_files)} quarterly parquet files:")
    for f in parquet_files:
        print(f"  {f.name}")

    # ------------------------------------------------------------------
    # Step 3: Process each quarter independently
    # ------------------------------------------------------------------
    quarter_output_dir = output_dir / "loan_level_quarterly"
    quarter_output_dir.mkdir(parents=True, exist_ok=True)

    for pf in parquet_files:
        # Extract quarter label from filename (e.g., "2006Q1.parquet" -> "2006Q1")
        quarter_label = pf.stem

        output_path = quarter_output_dir / f"{quarter_label}_loan_level.parquet"

        # Skip if already processed (allows resuming after interruption)
        if output_path.exists():
            print(f"\n  Skipping {quarter_label} (already processed: {output_path})")
            continue

        # Extract loan-level features for this quarter
        loan_df = extract_loan_level_features(pf, quarter_label)

        # Merge macro features if available
        if macro_df is not None:
            loan_df = merge_macro_features(loan_df, macro_df)

        # Save loan-level parquet for this quarter
        loan_df.to_parquet(output_path)
        print(f"  Saved loan-level data to {output_path}")
        print(f"  Shape: {loan_df.shape}")

        # Free memory
        del loan_df
        gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Combine all loan-level quarterly files
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Combining all loan-level quarterly files...")
    print(f"{'='*70}")

    quarter_files = sorted(quarter_output_dir.glob("*_loan_level.parquet"))
    combined_parts = []

    for qf in quarter_files:
        part = pd.read_parquet(qf)
        combined_parts.append(part)
        print(f"  Loaded {qf.name}: {len(part):,} loans")

    combined = pd.concat(combined_parts, axis=0)
    print(f"\nCombined dataset: {len(combined):,} loans")
    print(f"  Defaults: {combined['default_flag'].sum():,} "
          f"({combined['default_flag'].mean():.4f})")

    # Free parts from memory
    del combined_parts
    gc.collect()

    # ------------------------------------------------------------------
    # Step 5: Train/Validation/Test split by vintage year
    # ------------------------------------------------------------------
    # Time-based split is MANDATORY in credit risk modeling.
    # We train on earlier vintages and validate/test on later ones.
    # This simulates production: models are always trained on past data
    # and used to predict future outcomes.
    #
    # Split strategy:
    #   Train:      2005 vintages (loans originated in 2005)
    #   Validation: 2006 vintages (out-of-time validation)
    #   Test:       2007 vintages (out-of-time test, these loans lived
    #               through the worst of the 2008 crisis)
    print("\nApplying time-based train/validation/test split...")

    combined["data_split"] = "unknown"
    combined.loc[combined["origination_year"] == 2005, "data_split"] = "train"
    combined.loc[combined["origination_year"] == 2006, "data_split"] = "validation"
    combined.loc[combined["origination_year"] == 2007, "data_split"] = "test"

    for split in ["train", "validation", "test", "unknown"]:
        n = (combined["data_split"] == split).sum()
        if n > 0:
            dr = combined.loc[
                combined["data_split"] == split, "default_flag"
            ].mean()
            print(f"  {split:>12s}: {n:>10,} loans, default rate: {dr:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Save combined dataset
    # ------------------------------------------------------------------
    combined_path = output_dir / "loan_level_combined.parquet"
    combined.to_parquet(combined_path)
    print(f"\nSaved combined loan-level dataset to {combined_path}")
    print(f"  Shape: {combined.shape}")
    print(f"  Columns: {list(combined.columns)}")

    elapsed_total = time.time() - t_pipeline_start
    print(f"\nTotal feature engineering time: {elapsed_total:.1f} seconds")

    # ------------------------------------------------------------------
    # Step 7: Print summary statistics
    # ------------------------------------------------------------------
    print_summary_statistics(combined)

    return combined


def print_summary_statistics(df):
    """
    Print comprehensive summary statistics for the loan-level dataset.

    This serves as a sanity check and provides key numbers for the
    model documentation (Phase 9).
    """
    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING SUMMARY STATISTICS")
    print(f"{'='*70}")

    print(f"\nDataset size: {len(df):,} loans")
    print(f"Columns: {df.shape[1]}")

    # Default rates by key segments
    print("\n--- Default Rates by Segment ---")

    # By FICO bucket
    print("\nBy FICO Bucket:")
    if "fico_bucket" in df.columns:
        fico_stats = df.groupby("fico_bucket", observed=True)["default_flag"].agg(["count", "mean"])
        fico_stats.columns = ["count", "default_rate"]
        for idx, row in fico_stats.iterrows():
            print(f"  {idx:>10s}: {row['count']:>10,} loans, "
                  f"default rate: {row['default_rate']:.4f}")

    # By LTV bucket
    print("\nBy LTV Bucket:")
    if "ltv_bucket" in df.columns:
        ltv_stats = df.groupby("ltv_bucket", observed=True)["default_flag"].agg(["count", "mean"])
        ltv_stats.columns = ["count", "default_rate"]
        for idx, row in ltv_stats.iterrows():
            print(f"  {idx:>10s}: {row['count']:>10,} loans, "
                  f"default rate: {row['default_rate']:.4f}")

    # By origination year
    print("\nBy Origination Year:")
    if "origination_year" in df.columns:
        year_stats = df.groupby("origination_year")["default_flag"].agg(["count", "mean"])
        year_stats.columns = ["count", "default_rate"]
        for idx, row in year_stats.iterrows():
            print(f"  {int(idx):>10d}: {row['count']:>10,} loans, "
                  f"default rate: {row['default_rate']:.4f}")

    # LGD summary (defaulted loans only)
    defaulted = df[df["default_flag"] == 1]
    if len(defaulted) > 0 and "lgd" in defaulted.columns:
        print(f"\n--- LGD Summary (Defaulted Loans Only) ---")
        print(f"  Count with valid LGD: {defaulted['lgd'].notna().sum():,}")
        print(f"  Count with null LGD: {defaulted['lgd'].isna().sum():,}")
        lgd_valid = defaulted["lgd"].dropna()
        if len(lgd_valid) > 0:
            print(f"  Mean LGD: {lgd_valid.mean():.4f}")
            print(f"  Median LGD: {lgd_valid.median():.4f}")
            print(f"  Std LGD: {lgd_valid.std():.4f}")
        ead_valid = defaulted["ead"].dropna()
        if len(ead_valid) > 0:
            print(f"  Mean EAD: ${ead_valid.mean():,.0f}")

    # Macro feature coverage
    if "unemployment_rate" in df.columns:
        print(f"\n--- Macro Feature Coverage ---")
        print(f"  Unemployment rate: "
              f"{df['unemployment_rate'].notna().sum():,} / {len(df):,} "
              f"({100*df['unemployment_rate'].notna().mean():.1f}%)")

    # Missing value summary for key modeling features
    print(f"\n--- Missing Values (Key Features) ---")
    key_features = [
        "borrower_credit_score", "original_ltv", "dti",
        "original_interest_rate", "original_upb",
    ]
    for feat in key_features:
        if feat in df.columns:
            n_missing = df[feat].isna().sum()
            pct = 100 * n_missing / len(df)
            print(f"  {feat:>30s}: {n_missing:>10,} missing ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Entry point for standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import dotenv

    # Load environment variables (FRED API key)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        dotenv.load_dotenv(env_path)

    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("WARNING: FRED_API_KEY not found in environment.")
        print("  Macro features will be skipped.")

    # Paths relative to project root
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "processed"

    combined_df = run_feature_engineering(
        processed_dir=processed_dir,
        output_dir=output_dir,
        fred_api_key=fred_key,
    )

    print("\nDone. Feature engineering complete.")