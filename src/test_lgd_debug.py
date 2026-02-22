"""
test_lgd_debug.py
-----------------
Step-by-step debug of LGD calculation. Traces every intermediate step
using a single known defaulted loan before applying to the full dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    filepath = PROJECT_ROOT / "data" / "processed" / "quarterly" / "2006Q1.parquet"

    print("Loading data...")
    df = pd.read_parquet(filepath)
    df = df.sort_values(["loan_id", "monthly_reporting_period"])

    # ---------------------------------------------------------------
    # STEP 1: Find one defaulted loan to trace through manually.
    # ---------------------------------------------------------------
    print("\n--- STEP 1: Find a defaulted loan ---")

    reo_rows = df[df["zero_balance_code"] == "09"]
    print(f"Rows with ZB code 09 (REO): {len(reo_rows):,}")

    if len(reo_rows) == 0:
        print("No REO loans found. Checking what ZB codes exist...")
        print(df["zero_balance_code"].value_counts())
        return

    sample_loan_id = reo_rows["loan_id"].iloc[0]
    print(f"Sample defaulted loan_id: {sample_loan_id}")

    # ---------------------------------------------------------------
    # STEP 2: Look at this loan's full history.
    # ---------------------------------------------------------------
    print(f"\n--- STEP 2: Full history of loan {sample_loan_id} ---")

    loan_history = df[df["loan_id"] == sample_loan_id].copy()
    print(f"Total observations: {len(loan_history)}")

    cols_to_show = [
        "monthly_reporting_period", "current_actual_upb",
        "current_loan_delinquency_status", "zero_balance_code",
        "net_sale_proceeds", "foreclosure_costs"
    ]
    print("\nLast 5 observations:")
    print(loan_history[cols_to_show].tail(5).to_string())

    # ---------------------------------------------------------------
    # STEP 3: Get the second-to-last row for this loan.
    # ---------------------------------------------------------------
    print(f"\n--- STEP 3: EAD from second-to-last observation ---")

    second_to_last_row = loan_history.iloc[-2]
    last_row = loan_history.iloc[-1]

    ead = second_to_last_row["current_actual_upb"]
    print(f"Second-to-last UPB (EAD): ${ead:,.2f}")
    print(f"Last UPB (should be 0):   ${last_row['current_actual_upb']:,.2f}")
    print(f"Last ZB code:             {last_row['zero_balance_code']}")
    print(f"Net sale proceeds:        ${last_row['net_sale_proceeds']}")
    print(f"Foreclosure costs:        ${last_row['foreclosure_costs']}")

    # ---------------------------------------------------------------
    # STEP 4: Calculate LGD for this one loan.
    # ---------------------------------------------------------------
    print(f"\n--- STEP 4: LGD calculation for this loan ---")

    recovery_cols = [
        "net_sale_proceeds", "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    ]
    cost_cols = [
        "foreclosure_costs", "property_preservation_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes",
    ]

    total_recovery = sum(
        0 if pd.isna(last_row[c]) else last_row[c] for c in recovery_cols
    )
    total_costs = sum(
        0 if pd.isna(last_row[c]) else last_row[c] for c in cost_cols
    )
    net_loss = ead - total_recovery + total_costs
    lgd = net_loss / ead if ead > 0 else float("nan")

    print(f"Total recovery: ${total_recovery:,.2f}")
    print(f"Total costs:    ${total_costs:,.2f}")
    print(f"Net loss:       ${net_loss:,.2f}")
    print(f"EAD:            ${ead:,.2f}")
    print(f"LGD:            {lgd:.3f} ({lgd*100:.1f}%)")

    # ---------------------------------------------------------------
    # STEP 5: Now apply to all defaulted loans using correct approach.
    # The key insight: groupby().nth(-2) returns rows with the ORIGINAL
    # dataframe index, not loan_id as index. We need to set loan_id
    # as index first OR use a different approach.
    # ---------------------------------------------------------------
    print(f"\n--- STEP 5: Check groupby index behavior ---")

    # Method A: groupby with loan_id as index
    df_indexed = df.set_index("loan_id")
    grouped = df_indexed.groupby(level="loan_id")

    last_obs = grouped.last()
    second_last_obs = grouped.nth(-2)

    print(f"last_obs index type: {type(last_obs.index)}")
    print(f"last_obs index name: {last_obs.index.name}")
    print(f"second_last_obs index type: {type(second_last_obs.index)}")
    print(f"second_last_obs index name: {second_last_obs.index.name}")
    print(f"last_obs shape: {last_obs.shape}")
    print(f"second_last_obs shape: {second_last_obs.shape}")

    # Check if our sample loan is in second_last_obs.
    print(f"\nSample loan in last_obs: {sample_loan_id in last_obs.index}")
    print(f"Sample loan in second_last_obs: {sample_loan_id in second_last_obs.index}")

    if sample_loan_id in second_last_obs.index:
        print(f"Second-to-last UPB from grouped: "
              f"${second_last_obs.loc[sample_loan_id, 'current_actual_upb']:,.2f}")

    # ---------------------------------------------------------------
    # STEP 6: Full LGD calculation for all defaulted loans.
    # ---------------------------------------------------------------
    print(f"\n--- STEP 6: Full LGD calculation ---")

    default_zb_codes = ["02", "03", "06", "09", "15"]
    defaulted_mask = last_obs["zero_balance_code"].isin(default_zb_codes)
    defaulted_ids = defaulted_mask[defaulted_mask].index.tolist()
    print(f"Defaulted loans: {len(defaulted_ids):,}")

    # Get EAD from second-to-last observation.
    ead_all = second_last_obs.loc[
        second_last_obs.index.isin(defaulted_ids), "current_actual_upb"
    ]
    print(f"EAD values retrieved: {len(ead_all):,}")
    print(f"EAD non-null: {ead_all.notna().sum():,}")
    print(f"EAD > 0: {(ead_all > 0).sum():,}")

    if len(ead_all) == 0:
        print("ERROR: No EAD values found. Index mismatch likely.")
        return

    # Get loss fields from last observation.
    loss_data = last_obs.loc[defaulted_ids]

    total_recovery = loss_data[recovery_cols].fillna(0).sum(axis=1)
    total_costs = loss_data[cost_cols].fillna(0).sum(axis=1)

    # Align EAD to loss data index.
    ead_aligned = ead_all.reindex(loss_data.index)

    net_loss = ead_aligned - total_recovery + total_costs
    lgd_all = net_loss / ead_aligned
    lgd_valid = lgd_all[(ead_aligned > 0) & lgd_all.notna()]

    print(f"\nLGD RESULTS:")
    print(f"  Loans with valid LGD: {len(lgd_valid):,}")
    print(f"  Mean LGD:   {lgd_valid.mean():.3f} ({lgd_valid.mean()*100:.1f}%)")
    print(f"  Median LGD: {lgd_valid.median():.3f} ({lgd_valid.median()*100:.1f}%)")
    print(f"  Min LGD:    {lgd_valid.min():.3f}")
    print(f"  Max LGD:    {lgd_valid.max():.3f}")

    if 0.1 < lgd_valid.mean() < 0.9:
        print("  VALIDATION: PASS")
    else:
        print("  VALIDATION: NEEDS REVIEW")


if __name__ == "__main__":
    main()