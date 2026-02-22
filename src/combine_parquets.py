"""
combine_parquets.py
-------------------
Combines the 12 quarterly parquet files into a single dataset.

This is a standalone script separated from the main pipeline because
the combine step requires loading all quarters into memory simultaneously.
With quarterly parquets totaling ~460 MB, this is feasible even on machines
that could not handle the original 72 GB of raw CSV data.

Run this after data_pipeline.py has finished processing all quarters.
"""

import gc
import pandas as pd
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    QUARTERLY_DIR = PROJECT_ROOT / "data" / "processed" / "quarterly"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "loan_performance.parquet"

    files = sorted(QUARTERLY_DIR.glob("*.parquet"))
    print(f"Found {len(files)} quarterly parquet files")
    print(f"Total size: {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
    print("-" * 60)

    frames = []
    total_rows = 0

    for f in files:
        df = pd.read_parquet(f)
        row_count = len(df)
        loan_count = df["loan_id"].nunique()
        total_rows += row_count
        print(f"  {f.name:<20} {row_count:>12,} rows    {loan_count:>10,} loans")
        frames.append(df)

    print("-" * 60)
    print(f"Concatenating {total_rows:,} total rows...")

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Save combined dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nSaved to {OUTPUT_PATH} ({size_mb:,.1f} MB)")

    # Print final summary.
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows (loan-month observations): {len(combined):,}")
    print(f"Unique loans: {combined['loan_id'].nunique():,}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Memory usage: {combined.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    print(f"\nSample (first 5 rows):")
    print(combined[["loan_id", "monthly_reporting_period", "borrower_credit_score",
                     "original_ltv", "original_upb", "current_loan_delinquency_status"]].head().to_string())

    print("\nCombine complete.")


if __name__ == "__main__":
    main()