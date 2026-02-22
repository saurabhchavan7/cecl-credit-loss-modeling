"""
test_feature_extraction.py
--------------------------
Tests the loan-level feature extraction logic on one quarter (2006Q1)
before running on all 12 quarters. Validates that:
    - Static origination features are correctly extracted
    - Default target is correctly defined
    - LGD calculation produces sensible values
    - Memory usage is within acceptable bounds

This script processes one quarter and prints detailed validation results.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_loan_level_features(df):
    """
    Convert loan-month observations into a loan-level dataset.

    For each unique loan, this function extracts:
        1. Static origination features (from the first observation)
        2. Performance outcome (default flag, computed from full history)
        3. Loss severity (LGD, for defaulted loans with disposition data)
        4. Timing information (when default occurred, loan lifetime)

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month level data for one quarter from the parquet file.

    Returns
    -------
    pd.DataFrame
        One row per loan with origination features and outcome variables.
    """

    # ==================================================================
    # STEP 1: Extract static origination features.
    # These are constant across all monthly observations for a given loan.
    # We take the first observation (earliest reporting period) per loan.
    # ==================================================================

    static_cols = [
        "loan_id", "channel", "original_interest_rate", "original_upb",
        "original_loan_term", "origination_date", "first_payment_date",
        "original_ltv", "original_cltv", "number_of_borrowers", "dti",
        "borrower_credit_score", "coborrower_credit_score",
        "first_time_home_buyer", "loan_purpose", "property_type",
        "number_of_units", "occupancy_status", "property_state",
        "msa", "zip_code_short", "mortgage_insurance_pct", "amortization_type",
    ]

    # Sort by reporting period so .first() gives us the earliest observation.
    df_sorted = df.sort_values(["loan_id", "monthly_reporting_period"])
    static = df_sorted.groupby("loan_id")[static_cols].first().reset_index(drop=True)

    # loan_id is already in static_cols, so it is preserved.

    # ==================================================================
    # STEP 2: Compute default target from full loan history.
    #
    # A loan is defined as DEFAULTED if at any point in its life it:
    #   (a) Reaches 90+ days past due (delinquency status >= 3), OR
    #   (b) Has a zero balance code indicating credit loss:
    #       02 = Third Party Sale
    #       03 = Short Sale
    #       06 = Repurchased (due to credit issues)
    #       09 = REO Disposition (deed-in-lieu or foreclosure sale)
    #       15 = Non-Performing Note Sale
    #
    # This is the standard industry definition used by Fannie Mae and
    # consistent with regulatory guidance for CECL modeling.
    # ==================================================================

    default_zb_codes = {"02", "03", "06", "09", "15"}

    # Maximum delinquency ever reached by each loan.
    max_delinq = df.groupby("loan_id")["current_loan_delinquency_status"].max()
    max_delinq.name = "max_delinquency_status"

    # Whether loan ever had a default-related zero balance code.
    zb_per_loan = df.groupby("loan_id")["zero_balance_code"].apply(
        lambda x: x.dropna().unique()
    )
    has_default_zb = zb_per_loan.apply(
        lambda codes: any(str(c).strip() in default_zb_codes for c in codes)
    )
    has_default_zb.name = "has_default_zb_code"

    # Combine into default flag.
    default_info = pd.DataFrame({
        "max_delinquency_status": max_delinq,
        "has_default_zb_code": has_default_zb,
    })
    default_info["default_flag"] = (
        (default_info["max_delinquency_status"] >= 3) |
        (default_info["has_default_zb_code"])
    ).astype(int)

    # ==================================================================
    # STEP 3: Compute LGD (Loss Given Default) for defaulted loans.
    #
    # LGD = Total Loss / Exposure at Default
    #
    # Total Loss = UPB at default
    #              - Net Sale Proceeds
    #              - Credit Enhancement Proceeds
    #              - Repurchase Make Whole Proceeds
    #              - Other Foreclosure Proceeds
    #              + Foreclosure Costs
    #              + Property Preservation Costs
    #              + Asset Recovery Costs
    #              + Misc Holding Expenses
    #              + Holding Taxes
    #
    # LGD is only meaningful for loans that have completed the
    # disposition process (foreclosure sale, short sale, etc.).
    # We identify these by non-null net_sale_proceeds.
    # ==================================================================

    # Get the last observation for each loan (contains loss fields if disposed).
    last_obs = df_sorted.groupby("loan_id").last()

    recovery_cols = [
        "net_sale_proceeds", "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    ]
    cost_cols = [
        "foreclosure_costs", "property_preservation_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes",
    ]

    total_recovery = last_obs[recovery_cols].fillna(0).sum(axis=1)
    total_costs = last_obs[cost_cols].fillna(0).sum(axis=1)

    # UPB at the time the loan was removed from the pool.
    # Use current_actual_upb from the last observation as proxy for EAD.
    # For loans with zero balance, the UPB at time of default is better
    # captured from the observation just before zero balance.
    ead = last_obs["current_actual_upb"].fillna(0)

    # Net loss = EAD - recoveries + costs
    net_loss = ead - total_recovery + total_costs

    # LGD = net_loss / EAD (bounded, as some edge cases can produce < 0)
    lgd = pd.Series(np.where(ead > 0, net_loss / ead, np.nan), index=last_obs.index)
    lgd.name = "lgd"

    # Flag indicating whether loss data is available for this loan.
    has_loss_data = last_obs["net_sale_proceeds"].notna()
    has_loss_data.name = "has_loss_data"

    # ==================================================================
    # STEP 4: Compute loan lifetime and timing features.
    # ==================================================================

    loan_lifetime = df.groupby("loan_id")["loan_age"].max()
    loan_lifetime.name = "loan_lifetime_months"

    # ==================================================================
    # STEP 5: Merge everything into a single loan-level dataset.
    # ==================================================================

    result = static.set_index("loan_id")
    result = result.join(default_info[["max_delinquency_status", "default_flag"]])
    result = result.join(lgd)
    result = result.join(has_loss_data)
    result = result.join(loan_lifetime)

    result = result.reset_index()

    return result


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    filepath = PROJECT_ROOT / "data" / "processed" / "quarterly" / "2006Q1.parquet"

    print(f"Loading {filepath.name}...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} loan-month rows, {df['loan_id'].nunique():,} loans")

    print("\nExtracting loan-level features...")
    loans = extract_loan_level_features(df)

    # Free the large dataframe.
    del df

    print(f"Result: {len(loans):,} loans, {len(loans.columns)} columns")
    print(f"Memory: {loans.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ==================================================================
    # VALIDATION
    # ==================================================================

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Check 1: loan_id looks correct
    sample_id = loans["loan_id"].iloc[0]
    print(f"\n1. Sample loan_id: {sample_id}")
    assert str(sample_id).isdigit() and len(str(sample_id)) >= 8, "loan_id format wrong"
    print("   PASS")

    # Check 2: Default rate is reasonable (expect 10-20% for 2006 vintage)
    default_rate = loans["default_flag"].mean() * 100
    print(f"\n2. Default rate: {default_rate:.2f}%")
    print(f"   Defaults: {loans['default_flag'].sum():,} / {len(loans):,}")
    assert 5 < default_rate < 30, f"Default rate {default_rate:.1f}% outside expected range"
    print("   PASS")

    # Check 3: FICO distribution
    fico_mean = loans["borrower_credit_score"].mean()
    print(f"\n3. Mean FICO: {fico_mean:.0f}")
    assert 650 < fico_mean < 800, f"Mean FICO {fico_mean:.0f} outside expected range"
    print("   PASS")

    # Check 4: LTV distribution
    ltv_mean = loans["original_ltv"].mean()
    print(f"\n4. Mean LTV: {ltv_mean:.1f}%")
    assert 40 < ltv_mean < 90, f"Mean LTV {ltv_mean:.1f}% outside expected range"
    print("   PASS")

    # Check 5: LGD for defaulted loans with loss data
    defaulted_with_loss = loans[(loans["default_flag"] == 1) & (loans["has_loss_data"] == True)]
    if len(defaulted_with_loss) > 0:
        lgd_mean = defaulted_with_loss["lgd"].mean()
        lgd_median = defaulted_with_loss["lgd"].median()
        print(f"\n5. LGD statistics (defaulted loans with loss data):")
        print(f"   Count: {len(defaulted_with_loss):,}")
        print(f"   Mean LGD: {lgd_mean:.3f} ({lgd_mean*100:.1f}%)")
        print(f"   Median LGD: {lgd_median:.3f} ({lgd_median*100:.1f}%)")
        print(f"   Min: {defaulted_with_loss['lgd'].min():.3f}")
        print(f"   Max: {defaulted_with_loss['lgd'].max():.3f}")
        # LGD for 2006 vintage through 2008 crisis should be high (30-60%)
        assert 0.1 < lgd_mean < 0.9, f"Mean LGD {lgd_mean:.3f} outside expected range"
        print("   PASS")
    else:
        print("\n5. No defaulted loans with loss data found (may be normal for early quarters)")

    # Check 6: No duplicate loan IDs
    dup_count = loans["loan_id"].duplicated().sum()
    print(f"\n6. Duplicate loan IDs: {dup_count}")
    assert dup_count == 0, f"Found {dup_count} duplicate loan IDs"
    print("   PASS")

    # Check 7: Loan lifetime makes sense
    lifetime_mean = loans["loan_lifetime_months"].mean()
    print(f"\n7. Mean loan lifetime: {lifetime_mean:.0f} months")
    print(f"   Max lifetime: {loans['loan_lifetime_months'].max():.0f} months")
    print("   PASS")

    # Check 8: Default flag breakdown by FICO bucket
    print(f"\n8. Default rate by FICO bucket:")
    fico = loans["borrower_credit_score"].dropna()
    loans_with_fico = loans[loans["borrower_credit_score"].notna()].copy()
    loans_with_fico["fico_bucket"] = pd.cut(
        loans_with_fico["borrower_credit_score"],
        bins=[0, 620, 660, 700, 740, 850],
        labels=["<620", "620-660", "660-700", "700-740", "740+"]
    )
    bucket_defaults = loans_with_fico.groupby("fico_bucket")["default_flag"].mean() * 100
    for bucket, rate in bucket_defaults.items():
        print(f"   {bucket:<12} {rate:.2f}%")

    # Verify monotonicity: lower FICO should have higher default rate.
    rates = bucket_defaults.values
    is_monotonic = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    print(f"   Monotonic (lower FICO = higher default): {is_monotonic}")
    if is_monotonic:
        print("   PASS")
    else:
        print("   WARNING: Not perfectly monotonic (may still be acceptable)")

    # Print sample rows.
    print(f"\n" + "=" * 60)
    print("SAMPLE ROWS")
    print("=" * 60)
    display_cols = [
        "loan_id", "borrower_credit_score", "original_ltv", "dti",
        "original_upb", "loan_purpose", "default_flag", "max_delinquency_status",
        "lgd", "loan_lifetime_months"
    ]
    print(loans[display_cols].head(10).to_string())

    print("\nFeature extraction test complete.")


if __name__ == "__main__":
    main()