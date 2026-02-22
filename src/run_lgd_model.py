"""
Full LGD Model Training Pipeline
==================================

Trains and validates LGD models on the full 278K defaulted-loan dataset.

Author: Saurabh Chavan
"""

import gc
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from lgd_model import (
    prepare_lgd_dataset,
    train_lgd_ols,
    train_lgd_xgboost,
    validate_lgd_model,
    compute_lgd_by_segment,
    macro_sensitivity_check,
    LGD_FEATURES,
)

COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MODEL_DIR = project_root / "models"


def main():
    t_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - FULL LGD MODEL PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load and prepare data
    # ------------------------------------------------------------------
    print("\nStep 1: Loading data...")
    df = pd.read_parquet(COMBINED_PATH)
    data, targets, features = prepare_lgd_dataset(df)
    del df
    gc.collect()

    X_train = data["train"]
    X_val = data["val"]
    X_test = data["test"]
    y_train = targets["y_train"]
    y_val = targets["y_val"]
    y_test = targets["y_test"]

    # ------------------------------------------------------------------
    # Step 2: Train OLS model (primary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: Training OLS LGD model on full training data")
    print("=" * 70)

    ols_model = train_lgd_ols(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 3: Train XGBoost model (challenger)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Training XGBoost LGD model on full training data")
    print("=" * 70)

    xgb_model = train_lgd_xgboost(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 4: Full validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Full validation")
    print("=" * 70)

    print("\n--- OLS Model ---")
    ols_train_m, ols_train_p = validate_lgd_model(ols_model, X_train, y_train, "Train")
    ols_val_m, ols_val_p = validate_lgd_model(ols_model, X_val, y_val, "Validation")
    ols_test_m, ols_test_p = validate_lgd_model(ols_model, X_test, y_test, "Test")

    print("\n--- XGBoost Model ---")
    xgb_train_m, xgb_train_p = validate_lgd_model(xgb_model, X_train, y_train, "Train")
    xgb_val_m, xgb_val_p = validate_lgd_model(xgb_model, X_val, y_val, "Validation")
    xgb_test_m, xgb_test_p = validate_lgd_model(xgb_model, X_test, y_test, "Test")

    # ------------------------------------------------------------------
    # Step 5: Model comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LGD MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<25s} {'OLS':>12s} {'XGBoost':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Train R2':<25s} {ols_train_m['r2']:>12.4f} {xgb_train_m['r2']:>12.4f}")
    print(f"  {'Validation R2':<25s} {ols_val_m['r2']:>12.4f} {xgb_val_m['r2']:>12.4f}")
    print(f"  {'Test R2':<25s} {ols_test_m['r2']:>12.4f} {xgb_test_m['r2']:>12.4f}")
    print(f"  {'Train RMSE':<25s} {ols_train_m['rmse']:>12.4f} {xgb_train_m['rmse']:>12.4f}")
    print(f"  {'Validation RMSE':<25s} {ols_val_m['rmse']:>12.4f} {xgb_val_m['rmse']:>12.4f}")
    print(f"  {'Test RMSE':<25s} {ols_test_m['rmse']:>12.4f} {xgb_test_m['rmse']:>12.4f}")
    print(f"  {'Train Calibration':<25s} {ols_train_m['mean_predicted']/ols_train_m['mean_actual']:>12.4f} {xgb_train_m['mean_predicted']/xgb_train_m['mean_actual']:>12.4f}")
    print(f"  {'Val Calibration':<25s} {ols_val_m['mean_predicted']/ols_val_m['mean_actual']:>12.4f} {xgb_val_m['mean_predicted']/xgb_val_m['mean_actual']:>12.4f}")
    print(f"  {'Test Calibration':<25s} {ols_test_m['mean_predicted']/ols_test_m['mean_actual']:>12.4f} {xgb_test_m['mean_predicted']/xgb_test_m['mean_actual']:>12.4f}")

    # ------------------------------------------------------------------
    # Step 6: Segment analysis (OLS model on validation set)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 6: Segment analysis")
    print(f"{'='*70}")

    # LTV segments
    ltv_buckets = pd.cut(
        X_val["original_ltv"].values,
        bins=[0, 60, 70, 80, 90, 200],
        labels=["<60", "60-70", "70-80", "80-90", "90+"],
    )
    compute_lgd_by_segment(y_val.values, ols_val_p, ltv_buckets, "LTV Bucket (OLS)")

    # FICO segments
    fico_buckets = pd.cut(
        X_val["borrower_credit_score"].values,
        bins=[0, 620, 660, 700, 740, 850],
        labels=["<620", "620-660", "660-700", "700-740", "740+"],
    )
    compute_lgd_by_segment(y_val.values, ols_val_p, fico_buckets, "FICO Bucket (OLS)")

    # MI segments
    compute_lgd_by_segment(
        y_val.values, ols_val_p,
        X_val["has_mortgage_insurance"].values, "Has MI (OLS)"
    )

    # ------------------------------------------------------------------
    # Step 7: Macro sensitivity (OLS model)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 7: Macro sensitivity analysis")
    print(f"{'='*70}")

    X_baseline = X_val.head(2000).copy()

    macro_sensitivity_check(
        ols_model, X_baseline,
        "unemployment_rate", [4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        features,
    )

    macro_sensitivity_check(
        ols_model, X_baseline,
        "hpi_national", [120, 140, 160, 180, 200, 220],
        features,
    )

    # ------------------------------------------------------------------
    # Step 8: Save models and artifacts
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 8: Saving models and artifacts")
    print(f"{'='*70}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ols_model, MODEL_DIR / "lgd_ols.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "lgd_xgboost.pkl")
    print(f"  Saved LGD models to {MODEL_DIR}")

    # Save OLS coefficients
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": ols_model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(MODEL_DIR / "lgd_ols_coefficients.csv", index=False)

    # Save XGBoost importance
    import xgboost as xgb
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "lgd_xgb_feature_importance.csv", index=False)

    # Save validation summary
    summary = {
        "ols": {
            "train_r2": ols_train_m["r2"],
            "val_r2": ols_val_m["r2"],
            "test_r2": ols_test_m["r2"],
            "train_rmse": ols_train_m["rmse"],
            "val_rmse": ols_val_m["rmse"],
            "test_rmse": ols_test_m["rmse"],
            "train_mae": ols_train_m["mae"],
            "val_mae": ols_val_m["mae"],
            "test_mae": ols_test_m["mae"],
        },
        "xgboost": {
            "train_r2": xgb_train_m["r2"],
            "val_r2": xgb_val_m["r2"],
            "test_r2": xgb_test_m["r2"],
            "train_rmse": xgb_train_m["rmse"],
            "val_rmse": xgb_val_m["rmse"],
            "test_rmse": xgb_test_m["rmse"],
            "train_mae": xgb_train_m["mae"],
            "val_mae": xgb_val_m["mae"],
            "test_mae": xgb_test_m["mae"],
        },
    }
    pd.DataFrame(summary).to_csv(MODEL_DIR / "lgd_validation_summary.csv")

    # Save feature list
    with open(MODEL_DIR / "lgd_features.txt", "w") as f:
        for feat in features:
            f.write(feat + "\n")

    print(f"  Saved all artifacts")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"LGD MODEL PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()