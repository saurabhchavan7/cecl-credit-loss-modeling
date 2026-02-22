"""
test_lgd.py
-----------
Tests the LGD calculation logic using the second-to-last observation
for EAD (Exposure at Default), since Fannie Mae sets UPB to zero in
the final observation when a loan is removed from the pool.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    filepath = PROJECT_ROOT / "data" / "processed" / "quarterly" / "2006Q1.parquet"

    print(f"Loading {filepath.name}...")
    df = pd.read_parquet(filepath)
    df = df.sort_values(["loan_id", "monthly_reporting_period"])

    # Get the last observation per loan (contains loss fields and ZB code).
    last_obs = df.groupby("loan_id").last()

    # Identify defaulted loans with disposition (Short Sale or REO).
    default_zb_codes = ["02", "03", "06", "09", "15"]
    defaulted_mask = last_obs["zero_balance_code"].isin(default_zb_codes)
    defaulted_ids = defaulted_mask[defaulted_mask].index.tolist()
    print(f"Defaulted loans with ZB code: {len(defaulted_ids):,}")

    # Get the second-to-last observation per loan for EAD.
    # This is the last observation BEFORE the balance was zeroed out.
    second_last_obs = df.groupby("loan_id").nth(-2)

    # Filter to defaulted loans only.
    ead_series = second_last_obs.loc[
        second_last_obs.index.isin(defaulted_ids), "current_actual_upb"
    ]
    print(f"Loans with EAD data: {len(ead_series):,}")
    print(f"EAD stats: mean=${ead_series.mean():,.0f}, median=${ead_series.median():,.0f}")

    # Get loss fields from the last observation for defaulted loans.
    loss_data = last_obs.loc[defaulted_ids].copy()

    recovery_cols = [
        "net_sale_proceeds", "credit_enhancement_proceeds",
        "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
    ]
    cost_cols = [
        "foreclosure_costs", "property_preservation_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes",
    ]

    total_recovery = loss_data[recovery_cols].fillna(0).sum(axis=1)
    total_costs = loss_data[cost_cols].fillna(0).sum(axis=1)

    # Align EAD with loss data (both indexed by loan_id).
    ead = ead_series.reindex(loss_data.index)

    # Calculate net loss and LGD.
    net_loss = ead - total_recovery + total_costs
    lgd = net_loss / ead

    # Filter to loans where EAD > 0 and LGD is not NaN.
    valid = (ead > 0) & lgd.notna()
    lgd_valid = lgd[valid]

    print(f"\nLGD RESULTS:")
    print(f"  Loans with valid LGD: {len(lgd_valid):,}")
    print(f"  Mean LGD:   {lgd_valid.mean():.3f} ({lgd_valid.mean()*100:.1f}%)")
    print(f"  Median LGD: {lgd_valid.median():.3f} ({lgd_valid.median()*100:.1f}%)")
    print(f"  Std LGD:    {lgd_valid.std():.3f}")
    print(f"  Min LGD:    {lgd_valid.min():.3f}")
    print(f"  Max LGD:    {lgd_valid.max():.3f}")

    # Distribution of LGD.
    print(f"\n  LGD Distribution:")
    bins = [-999, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 999]
    labels = ["<0 (gain)", "0-20%", "20-40%", "40-60%", "60-80%", "80-100%", ">100%"]
    lgd_bins = pd.cut(lgd_valid, bins=bins, labels=labels)
    for label, count in lgd_bins.value_counts().sort_index().items():
        print(f"    {label:<15} {count:>8,}  ({count/len(lgd_valid)*100:.1f}%)")

    # Sanity check: mean LGD for 2006 vintage through crisis should be 30-60%.
    mean_lgd = lgd_valid.mean()
    if 0.1 < mean_lgd < 0.9:
        print(f"\n  LGD validation: PASS (mean {mean_lgd:.1%} is within expected range)")
    else:
        print(f"\n  LGD validation: WARNING (mean {mean_lgd:.1%} outside expected 10-90% range)")


if __name__ == "__main__":
    main()