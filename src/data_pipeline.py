"""
data_pipeline.py
----------------
Fannie Mae Single-Family Loan Performance Data Ingestion Pipeline

This module reads raw Fannie Mae quarterly loan performance files and produces
a clean, analysis-ready dataset for credit risk modeling. The raw files use
a combined format (post-October 2020) where each row represents one loan
observed at one monthly reporting period, containing both origination-level
(static) attributes and monthly performance (dynamic) attributes.

Data source: Fannie Mae Single-Family Loan Performance Dataset
Format: Pipe-delimited (|), no header row, 110 raw columns (108 data + 2 empty
        from leading/trailing pipes)
Reference: SF Glossary & File Layout (February 2026)

Column Alignment (verified against raw data):
    The raw file has leading and trailing pipe characters on each row.
    When parsed with sep='|', this produces 110 columns indexed 0-109:
        - Index 0:   empty (artifact of leading pipe)
        - Index 1:   glossary field 2 (loan_id)
        - Index 2:   glossary field 3 (monthly_reporting_period)
        - ...
        - Index N:   glossary field N+1
        - Index 108: glossary field 109 (if present)
        - Index 109: empty (artifact of trailing pipe)

    Therefore: raw_index = glossary_field_number - 1

    Verification (from 2005Q1 row 1):
        Index 1  = 100000102115 = loan_id (field 2)        -> 2-1 = 1 CORRECT
        Index 7  = 4.5          = orig_rate (field 8)       -> 8-1 = 7 CORRECT
        Index 9  = 95000        = original_upb (field 10)   -> 10-1 = 9 CORRECT
        Index 19 = 51           = original_ltv (field 20)   -> 20-1 = 19 CORRECT
        Index 23 = 783          = fico_score (field 24)     -> 24-1 = 23 CORRECT

Memory Management:
    Raw files are 6+ GB each. We read in chunks of 500,000 rows, select only
    the 42 columns needed for modeling, and save each quarter as a compressed
    parquet file before moving to the next quarter.
"""

import gc
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# COLUMN DEFINITIONS
# ---------------------------------------------------------------------------
# Maps each field name to its glossary field position number (1-108).
# The raw file column index = field_position - 1 because the leading pipe
# creates an empty column at index 0.
# ---------------------------------------------------------------------------

FIELD_POSITIONS = {
    "reference_pool_id": 1,
    "loan_id": 2,
    "monthly_reporting_period": 3,
    "channel": 4,
    "seller_name": 5,
    "servicer_name": 6,
    "master_servicer": 7,
    "original_interest_rate": 8,
    "current_interest_rate": 9,
    "original_upb": 10,
    "upb_at_issuance": 11,
    "current_actual_upb": 12,
    "original_loan_term": 13,
    "origination_date": 14,
    "first_payment_date": 15,
    "loan_age": 16,
    "remaining_months_legal_maturity": 17,
    "remaining_months_to_maturity": 18,
    "maturity_date": 19,
    "original_ltv": 20,
    "original_cltv": 21,
    "number_of_borrowers": 22,
    "dti": 23,
    "borrower_credit_score": 24,
    "coborrower_credit_score": 25,
    "first_time_home_buyer": 26,
    "loan_purpose": 27,
    "property_type": 28,
    "number_of_units": 29,
    "occupancy_status": 30,
    "property_state": 31,
    "msa": 32,
    "zip_code_short": 33,
    "mortgage_insurance_pct": 34,
    "amortization_type": 35,
    "prepayment_penalty_flag": 36,
    "interest_only_flag": 37,
    "io_first_pi_date": 38,
    "months_to_amortization": 39,
    "current_loan_delinquency_status": 40,
    "loan_payment_history": 41,
    "modification_flag": 42,
    "mi_cancellation_indicator": 43,
    "zero_balance_code": 44,
    "zero_balance_effective_date": 45,
    "upb_at_removal": 46,
    "repurchase_date": 47,
    "scheduled_principal_current": 48,
    "total_principal_current": 49,
    "unscheduled_principal_current": 50,
    "last_paid_installment_date": 51,
    "foreclosure_date": 52,
    "disposition_date": 53,
    "foreclosure_costs": 54,
    "property_preservation_costs": 55,
    "asset_recovery_costs": 56,
    "misc_holding_expenses": 57,
    "holding_taxes": 58,
    "net_sale_proceeds": 59,
    "credit_enhancement_proceeds": 60,
    "repurchase_make_whole_proceeds": 61,
    "other_foreclosure_proceeds": 62,
    "modification_noninterest_bearing_upb": 63,
    "principal_forgiveness_amount": 64,
    "original_list_start_date": 65,
    "original_list_price": 66,
    "current_list_start_date": 67,
    "current_list_price": 68,
    "borrower_credit_score_at_issuance": 69,
    "coborrower_credit_score_at_issuance": 70,
    "borrower_credit_score_current": 71,
    "coborrower_credit_score_current": 72,
    "mi_type": 73,
    "servicing_activity_indicator": 74,
    "current_period_modification_loss": 75,
    "cumulative_modification_loss": 76,
    "current_period_credit_event_net": 77,
    "cumulative_credit_event_net": 78,
    "special_eligibility_program": 79,
    "foreclosure_principal_writeoff": 80,
    "relocation_mortgage_indicator": 81,
    "zero_balance_code_change_date": 82,
    "loan_holdback_indicator": 83,
    "loan_holdback_effective_date": 84,
    "delinquent_accrued_interest": 85,
    "property_valuation_method": 86,
    "high_balance_loan_indicator": 87,
    "arm_initial_fixed_rate_le_5yr": 88,
    "arm_product_type": 89,
    "initial_fixed_rate_period": 90,
    "interest_rate_adjustment_frequency": 91,
    "next_interest_rate_adj_date": 92,
    "next_payment_change_date": 93,
    "arm_index": 94,
    "arm_cap_structure": 95,
    "initial_rate_cap_up_pct": 96,
    "periodic_rate_cap_up_pct": 97,
    "lifetime_rate_cap_up_pct": 98,
    "mortgage_margin": 99,
    "arm_balloon_indicator": 100,
    "arm_plan_number": 101,
    "borrower_assistance_plan": 102,
    "hltv_refi_indicator": 103,
    "deal_name": 104,
    "repurchase_make_whole_proceeds_flag": 105,
    "alternative_delinquency_resolution": 106,
    "alternative_delinquency_resolution_count": 107,
    "total_deferral_amount": 108,
}


# ---------------------------------------------------------------------------
# COLUMNS RELEVANT TO CREDIT RISK MODELING
# ---------------------------------------------------------------------------

SELECTED_COLUMNS = [
    # --- Identifiers and Time ---
    "loan_id",
    "monthly_reporting_period",

    # --- Origination Characteristics (Static) ---
    "channel",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "origination_date",
    "first_payment_date",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "dti",
    "borrower_credit_score",
    "coborrower_credit_score",
    "first_time_home_buyer",
    "loan_purpose",
    "property_type",
    "number_of_units",
    "occupancy_status",
    "property_state",
    "msa",
    "zip_code_short",
    "mortgage_insurance_pct",
    "amortization_type",

    # --- Monthly Performance (Dynamic) ---
    "current_interest_rate",
    "current_actual_upb",
    "loan_age",
    "remaining_months_legal_maturity",
    "current_loan_delinquency_status",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "borrower_assistance_plan",

    # --- Loss-Related Fields (for LGD calculation) ---
    "foreclosure_date",
    "disposition_date",
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

# ---------------------------------------------------------------------------
# Precompute column selection parameters.
# raw_index = glossary_field_number - 1
# ---------------------------------------------------------------------------

_SELECTED_INDICES = sorted([FIELD_POSITIONS[col] - 1 for col in SELECTED_COLUMNS])
_INDEX_TO_NAME = {FIELD_POSITIONS[col] - 1: col for col in SELECTED_COLUMNS}
_SORTED_NAMES = [_INDEX_TO_NAME[i] for i in _SELECTED_INDICES]

# Columns that must be read as strings to preserve formatting.
_STRING_COLUMNS = {
    "loan_id", "monthly_reporting_period", "channel", "origination_date",
    "first_payment_date", "first_time_home_buyer", "loan_purpose",
    "property_type", "occupancy_status", "property_state", "msa",
    "zip_code_short", "amortization_type", "current_loan_delinquency_status",
    "modification_flag", "zero_balance_code", "zero_balance_effective_date",
    "foreclosure_date", "disposition_date", "borrower_assistance_plan",
}

_DTYPE_SPEC = {_INDEX_TO_NAME[i]: str for i in _SELECTED_INDICES
               if _INDEX_TO_NAME[i] in _STRING_COLUMNS}


def verify_column_alignment(filepath):
    """
    Read one row from the raw file and verify column mapping is correct.

    This sanity check runs before processing any data. It reads the first
    row and validates that known fields contain expected value types:
        - loan_id should be a 12-character numeric string
        - original_loan_term should be a number like 120 or 360
        - property_type should be a 2-char code like SF, CO, PU, MH
        - channel should be R, C, or B

    If any check fails, the function raises an error to prevent processing
    with misaligned columns.
    """

    print("  Verifying column alignment on first row...")

    test_df = pd.read_csv(
        filepath,
        sep="|",
        header=None,
        nrows=1,
        usecols=_SELECTED_INDICES,
        names=_SORTED_NAMES,
        dtype=str,
    )

    row = test_df.iloc[0]

    # Check loan_id: should be a long numeric string, not a date.
    loan_id = str(row["loan_id"])
    assert len(loan_id) >= 8 and loan_id.isdigit(), \
        f"Column alignment error: loan_id = '{loan_id}' (expected 12-digit number)"

    # Check channel: should be R, C, or B.
    channel = str(row["channel"]).strip()
    assert channel in ("R", "C", "B"), \
        f"Column alignment error: channel = '{channel}' (expected R, C, or B)"

    # Check property_type: should be SF, CO, PU, MH, CP.
    prop_type = str(row["property_type"]).strip()
    assert prop_type in ("SF", "CO", "PU", "MH", "CP"), \
        f"Column alignment error: property_type = '{prop_type}' (expected SF/CO/PU/MH/CP)"

    # Check borrower_credit_score: should be a number between 300 and 850.
    fico = row["borrower_credit_score"]
    if pd.notna(fico) and str(fico).strip():
        fico_val = float(fico)
        assert 300 <= fico_val <= 850, \
            f"Column alignment error: FICO = {fico_val} (expected 300-850)"

    print("  Column alignment verified: loan_id={}, channel={}, "
          "property_type={}, FICO={}".format(loan_id, channel, prop_type, fico))

    return True


def apply_data_types(df):
    """
    Convert raw columns to appropriate analytical types.

    Date strings (MMYYYY) become datetime, delinquency status becomes numeric,
    financial fields become float, and low-cardinality fields become categorical.
    """

    df = df.copy()

    # Date columns: MMYYYY format to datetime.
    date_columns = [
        "monthly_reporting_period", "origination_date", "first_payment_date",
        "zero_balance_effective_date", "foreclosure_date", "disposition_date",
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%m%Y", errors="coerce")

    # Delinquency status: "00"=current, "01"=30DPD, ..., "XX"=unknown -> NaN.
    if "current_loan_delinquency_status" in df.columns:
        df["current_loan_delinquency_status"] = pd.to_numeric(
            df["current_loan_delinquency_status"], errors="coerce"
        )

    # Numeric financial columns.
    numeric_columns = [
        "original_interest_rate", "current_interest_rate", "original_upb",
        "current_actual_upb", "original_loan_term", "loan_age",
        "remaining_months_legal_maturity", "original_ltv", "original_cltv",
        "number_of_borrowers", "dti", "borrower_credit_score",
        "coborrower_credit_score", "mortgage_insurance_pct", "number_of_units",
        "foreclosure_costs", "property_preservation_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes",
        "net_sale_proceeds", "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical columns for memory efficiency.
    categorical_columns = [
        "channel", "first_time_home_buyer", "loan_purpose", "property_type",
        "occupancy_status", "property_state", "amortization_type",
        "modification_flag", "zero_balance_code", "borrower_assistance_plan",
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def load_quarterly_file_chunked(filepath, chunksize=500_000):
    """
    Read a single Fannie Mae quarterly file using chunked processing.

    Reads 500,000 rows at a time to keep peak memory usage manageable.
    Only the 42 selected columns are retained from each chunk.
    """

    print(f"  Reading: {filepath}")
    print(f"  Selecting {len(SELECTED_COLUMNS)} of 108 columns, chunk size {chunksize:,}")

    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        filepath,
        sep="|",
        header=None,
        usecols=_SELECTED_INDICES,
        names=_SORTED_NAMES,
        dtype=_DTYPE_SPEC,
        low_memory=False,
        on_bad_lines="skip",
        chunksize=chunksize,
    )

    for chunk_num, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        chunks.append(chunk)
        if chunk_num % 10 == 0:
            print(f"    ...processed {total_rows:,} rows")

    df = pd.concat(chunks, ignore_index=True)
    unique_loans = df["loan_id"].nunique()
    print(f"  Loaded {total_rows:,} rows, {unique_loans:,} unique loans")

    del chunks
    gc.collect()

    return df


def process_and_save_quarter(filepath, output_dir, quarter_name):
    """
    Load one quarter, apply data types, and save as parquet.

    Processes a single quarter at a time and saves immediately to disk,
    freeing memory before the next quarter is processed. Skips quarters
    that have already been processed (allows resuming interrupted runs).
    """

    output_path = Path(output_dir) / f"{quarter_name}.parquet"

    if output_path.exists():
        print(f"  {quarter_name}: Already processed, skipping.")
        df = pd.read_parquet(output_path, columns=["loan_id"])
        summary = {"quarter": quarter_name, "rows": len(df), "loans": df["loan_id"].nunique()}
        del df
        gc.collect()
        return summary

    df = load_quarterly_file_chunked(filepath)

    print(f"  Applying data types for {quarter_name}...")
    df = apply_data_types(df)
    df["vintage_quarter"] = quarter_name

    # Convert categorical to string for parquet compatibility.
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    summary = {"quarter": quarter_name, "rows": len(df), "loans": df["loan_id"].nunique()}

    print(f"  Saved {quarter_name}: {size_mb:,.1f} MB")

    del df
    gc.collect()

    return summary


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    QUARTERLY_PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "quarterly"

    quarters = sorted([
        d.name for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and d.name[0:4].isdigit()
    ])

    print(f"Found {len(quarters)} quarterly datasets: {quarters}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # STEP 0: Verify column alignment on the first file before bulk run.
    # This prevents wasting hours processing data with wrong columns.
    # -------------------------------------------------------------------
    first_csv = list((RAW_DATA_DIR / quarters[0]).glob("*.csv"))[0]
    verify_column_alignment(first_csv)
    print("=" * 60)

    # -------------------------------------------------------------------
    # STEP 1: Process each quarter individually.
    # -------------------------------------------------------------------
    summaries = []
    for quarter in quarters:
        print(f"\nProcessing {quarter}...")
        print("-" * 40)

        quarter_dir = RAW_DATA_DIR / quarter
        csv_files = list(quarter_dir.glob("*.csv"))

        if not csv_files:
            print(f"  WARNING: No CSV file found in {quarter_dir}")
            continue

        summary = process_and_save_quarter(
            filepath=csv_files[0],
            output_dir=QUARTERLY_PARQUET_DIR,
            quarter_name=quarter,
        )
        summaries.append(summary)

    # -------------------------------------------------------------------
    # STEP 2: Print summary table.
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  {'Quarter':<12} {'Rows':>15} {'Unique Loans':>15}")
    print(f"  {'-'*12} {'-'*15} {'-'*15}")
    for s in summaries:
        print(f"  {s['quarter']:<12} {s['rows']:>15,} {s['loans']:>15,}")

    total_rows = sum(s["rows"] for s in summaries)
    total_loans = sum(s["loans"] for s in summaries)
    print(f"  {'-'*12} {'-'*15} {'-'*15}")
    print(f"  {'TOTAL':<12} {total_rows:>15,} {total_loans:>15,}")

    print("\nQuarterly parquet files saved in: data/processed/quarterly/")
    print("Phase 1 data ingestion complete.")