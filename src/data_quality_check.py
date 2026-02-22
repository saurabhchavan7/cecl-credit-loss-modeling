"""
data_quality_check.py
---------------------
Quick data profiling on one quarter to understand data quality before
feature engineering. This runs on a single parquet file (not all 12)
to stay within memory limits.

Outputs:
    - Missing value rates for all columns
    - Value distributions for key modeling variables
    - Default rate calculation
    - Basic statistics for numeric fields

Run after data_pipeline.py has completed.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    QUARTERLY_DIR = PROJECT_ROOT / "data" / "processed" / "quarterly"

    # Use 2006Q1 as representative sample: mid-vintage, moderate size.
    filepath = QUARTERLY_DIR / "2006Q1.parquet"
    print(f"Loading {filepath.name} for data quality profiling...")
    df = pd.read_parquet(filepath)

    print(f"Rows: {len(df):,}")
    print(f"Unique loans: {df['loan_id'].nunique():,}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------------
    # 1. MISSING VALUES
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. MISSING VALUE RATES")
    print("=" * 60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    }).sort_values("missing_pct", ascending=False)

    for col, row in missing_report.iterrows():
        if row["missing_pct"] > 0:
            print(f"  {col:<45} {row['missing_pct']:>7.2f}%")

    # ------------------------------------------------------------------
    # 2. KEY VARIABLE DISTRIBUTIONS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. KEY VARIABLE DISTRIBUTIONS (origination-level, first obs per loan)")
    print("=" * 60)

    # Get one row per loan (first observation) for origination-level stats.
    first_obs = df.sort_values("monthly_reporting_period").groupby("loan_id").first()

    print(f"\nBorrower Credit Score (FICO):")
    fico = first_obs["borrower_credit_score"].dropna()
    print(f"  Count: {len(fico):,}  |  Mean: {fico.mean():.0f}  |  "
          f"Median: {fico.median():.0f}  |  Min: {fico.min():.0f}  |  Max: {fico.max():.0f}")
    print(f"  Distribution:")
    bins = [0, 620, 660, 700, 740, 850]
    labels = ["<620 (Subprime)", "620-660", "660-700", "700-740", "740+ (Prime)"]
    fico_bins = pd.cut(fico, bins=bins, labels=labels)
    for label, count in fico_bins.value_counts().sort_index().items():
        print(f"    {label:<20} {count:>10,}  ({count/len(fico)*100:.1f}%)")

    print(f"\nOriginal LTV:")
    ltv = first_obs["original_ltv"].dropna()
    print(f"  Count: {len(ltv):,}  |  Mean: {ltv.mean():.1f}  |  "
          f"Median: {ltv.median():.0f}  |  Min: {ltv.min():.0f}  |  Max: {ltv.max():.0f}")
    print(f"  Distribution:")
    ltv_bins = pd.cut(ltv, bins=[0, 60, 70, 80, 90, 200], labels=["<60", "60-70", "70-80", "80-90", "90+"])
    for label, count in ltv_bins.value_counts().sort_index().items():
        print(f"    {label:<20} {count:>10,}  ({count/len(ltv)*100:.1f}%)")

    print(f"\nDTI:")
    dti = first_obs["dti"].dropna()
    print(f"  Count: {len(dti):,}  |  Mean: {dti.mean():.1f}  |  "
          f"Median: {dti.median():.0f}  |  Min: {dti.min():.0f}  |  Max: {dti.max():.0f}")

    print(f"\nOriginal UPB ($):")
    upb = first_obs["original_upb"].dropna()
    print(f"  Count: {len(upb):,}  |  Mean: ${upb.mean():,.0f}  |  "
          f"Median: ${upb.median():,.0f}  |  Min: ${upb.min():,.0f}  |  Max: ${upb.max():,.0f}")

    print(f"\nLoan Purpose:")
    for val, count in first_obs["loan_purpose"].value_counts().items():
        label = {"P": "Purchase", "R": "Refinance", "C": "Cash-out Refi", "U": "Refi-Not Specified"}.get(val, val)
        print(f"    {label:<25} {count:>10,}  ({count/len(first_obs)*100:.1f}%)")

    print(f"\nProperty Type:")
    for val, count in first_obs["property_type"].value_counts().items():
        label = {"SF": "Single Family", "CO": "Condo", "PU": "PUD", "MH": "Manufactured", "CP": "Co-op"}.get(val, val)
        print(f"    {label:<25} {count:>10,}  ({count/len(first_obs)*100:.1f}%)")

    print(f"\nOccupancy Status:")
    for val, count in first_obs["occupancy_status"].value_counts().items():
        label = {"P": "Primary Residence", "S": "Second Home", "I": "Investor"}.get(val, val)
        print(f"    {label:<25} {count:>10,}  ({count/len(first_obs)*100:.1f}%)")

    print(f"\nChannel:")
    for val, count in first_obs["channel"].value_counts().items():
        label = {"R": "Retail", "C": "Correspondent", "B": "Broker"}.get(val, val)
        print(f"    {label:<25} {count:>10,}  ({count/len(first_obs)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 3. DEFAULT ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. DEFAULT ANALYSIS")
    print("=" * 60)

    # A loan is considered defaulted if it ever reaches 90+ days past due
    # or has a zero balance code indicating credit loss (03, 06, 09).
    default_codes = {"02", "03", "06", "09", "15"}

    # Check delinquency-based defaults.
    max_delinq = df.groupby("loan_id")["current_loan_delinquency_status"].max()
    delinq_defaults = (max_delinq >= 3).sum()

    # Check zero-balance-code-based defaults.
    zb_defaults = df[df["zero_balance_code"].isin(default_codes)]["loan_id"].nunique()

    # Combined: loan defaulted if EITHER condition is met.
    delinq_default_loans = set(max_delinq[max_delinq >= 3].index)
    zb_default_loans = set(df[df["zero_balance_code"].isin(default_codes)]["loan_id"].unique())
    all_default_loans = delinq_default_loans | zb_default_loans

    total_loans = df["loan_id"].nunique()
    default_rate = len(all_default_loans) / total_loans * 100

    print(f"  Total unique loans in quarter: {total_loans:,}")
    print(f"  Loans with 90+ DPD:            {delinq_defaults:,}")
    print(f"  Loans with default ZB code:    {zb_defaults:,}")
    print(f"  Combined unique defaults:      {len(all_default_loans):,}")
    print(f"  Default rate:                  {default_rate:.2f}%")

    # Zero balance code distribution.
    print(f"\nZero Balance Code Distribution (loan-level):")
    zb_per_loan = df.groupby("loan_id")["zero_balance_code"].last()
    zb_labels = {
        "01": "Prepaid/Matured",
        "02": "Third Party Sale",
        "03": "Short Sale",
        "06": "Repurchased",
        "09": "REO Disposition",
        "15": "Non-Perf Note Sale",
        "16": "Reperf Note Sale",
        "96": "Removal (non-credit)",
        "nan": "Still Active",
    }
    for val, count in zb_per_loan.value_counts().head(10).items():
        label = zb_labels.get(str(val), str(val))
        print(f"    {label:<25} {count:>10,}  ({count/total_loans*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. DELINQUENCY STATUS DISTRIBUTION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. DELINQUENCY STATUS DISTRIBUTION (all observations)")
    print("=" * 60)

    delinq = df["current_loan_delinquency_status"].value_counts().sort_index()
    delinq_labels = {
        0: "Current",
        1: "30 DPD",
        2: "60 DPD",
        3: "90 DPD",
    }
    for val, count in delinq.head(10).items():
        label = delinq_labels.get(int(val), f"{int(val)*30} DPD") if pd.notna(val) else "Unknown"
        print(f"    {label:<20} {count:>12,}  ({count/len(df)*100:.2f}%)")

    print("\nData quality check complete.")


if __name__ == "__main__":
    main()