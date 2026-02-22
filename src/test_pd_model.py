"""
Test PD Model Pipeline - Step by Step Validation
=================================================

Tests the entire PD modeling pipeline on a SAMPLE of the data before
running on the full 3.8 million loan dataset.

Test strategy:
1. Load combined dataset and verify structure
2. Handle the "unknown" split (pre-2005 loans -> merge into train)
3. Test WoE/IV on a 50K sample
4. Validate IV values make economic sense
5. Test WoE transformation consistency
6. Test logistic regression on sample
7. Test XGBoost on sample
8. Test validation metrics
9. Test calibration table
10. Test PSI computation

Run this BEFORE running the full PD model pipeline.

Author: Saurabh Chavan
"""

import os
import sys
import gc
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pd_model import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_CANDIDATE_FEATURES,
    calculate_woe_iv_for_feature,
    calculate_woe_iv_all_features,
    apply_woe_transformation,
    train_logistic_regression,
    train_xgboost_challenger,
    validate_model,
    compute_calibration_table,
    compute_psi,
    prepare_features_for_xgboost,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
SAMPLE_SIZE = 50_000  # Sample size for quick testing
TARGET = "default_flag"

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
# TEST 1: Load data and verify structure
# ===========================================================================
def test_load_data():
    print("\n" + "=" * 70)
    print("TEST 1: Load combined dataset and verify structure")
    print("=" * 70)

    check(COMBINED_PATH.exists(), "Combined parquet exists", str(COMBINED_PATH))

    df = pd.read_parquet(COMBINED_PATH)
    print(f"  Dataset shape: {df.shape}")

    check(TARGET in df.columns, "Target column exists")
    check("data_split" in df.columns, "data_split column exists")

    # Verify splits
    split_counts = df["data_split"].value_counts()
    print(f"\n  Split distribution:")
    for split, count in split_counts.items():
        dr = df.loc[df["data_split"] == split, TARGET].mean()
        print(f"    {split:>12s}: {count:>10,} loans, default rate: {dr:.4f}")

    # Verify candidate features exist in the dataset
    missing_features = [f for f in ALL_CANDIDATE_FEATURES if f not in df.columns]
    check(
        len(missing_features) == 0,
        "All candidate features exist in dataset",
        f"Missing: {missing_features}" if missing_features else f"{len(ALL_CANDIDATE_FEATURES)} features confirmed",
    )

    return df


# ===========================================================================
# TEST 2: Handle unknown split and create samples
# ===========================================================================
def test_prepare_splits(df):
    print("\n" + "=" * 70)
    print("TEST 2: Prepare train/val/test splits and sample")
    print("=" * 70)

    # Move "unknown" (pre-2005) loans into training set.
    # These are older vintages that were active in 2005Q1 performance data.
    # They belong in training since they precede our validation period.
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"

    train = df[df["data_split"] == "train"]
    val = df[df["data_split"] == "validation"]
    test = df[df["data_split"] == "test"]

    print(f"  After merging unknown into train:")
    print(f"    Train:      {len(train):>10,} loans, DR: {train[TARGET].mean():.4f}")
    print(f"    Validation: {len(val):>10,} loans, DR: {val[TARGET].mean():.4f}")
    print(f"    Test:       {len(test):>10,} loans, DR: {test[TARGET].mean():.4f}")

    check(len(train) > 1_000_000, "Train set has > 1M loans")
    check(len(val) > 500_000, "Validation set has > 500K loans")
    check(len(test) > 500_000, "Test set has > 500K loans")

    # Create stratified sample for quick testing
    # Stratified to maintain the same default rate
    print(f"\n  Creating stratified sample of {SAMPLE_SIZE:,} for testing...")

    sample_train = train.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), int(SAMPLE_SIZE * len(x) / len(train))),
            random_state=42,
        )
    )
    sample_val = val.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), int(SAMPLE_SIZE * 0.5 * len(x) / len(val))),
            random_state=42,
        )
    )
    sample_test = test.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), int(SAMPLE_SIZE * 0.5 * len(x) / len(test))),
            random_state=42,
        )
    )

    print(f"    Sample train: {len(sample_train):,} loans, DR: {sample_train[TARGET].mean():.4f}")
    print(f"    Sample val:   {len(sample_val):,} loans, DR: {sample_val[TARGET].mean():.4f}")
    print(f"    Sample test:  {len(sample_test):,} loans, DR: {sample_test[TARGET].mean():.4f}")

    check(
        abs(sample_train[TARGET].mean() - train[TARGET].mean()) < 0.02,
        "Sample train default rate matches full train",
    )

    return sample_train, sample_val, sample_test


# ===========================================================================
# TEST 3: WoE/IV on single feature
# ===========================================================================
def test_woe_single_feature(sample_train):
    print("\n" + "=" * 70)
    print("TEST 3: WoE/IV calculation on single feature (borrower_credit_score)")
    print("=" * 70)

    result = calculate_woe_iv_for_feature(
        sample_train, "borrower_credit_score", TARGET, n_bins=10
    )

    check(result["iv"] > 0, "IV is positive", f"IV = {result['iv']:.4f}")
    check(
        result["iv"] > 0.10,
        "FICO has at least Medium IV (> 0.10)",
        f"IV = {result['iv']:.4f}",
    )
    check(len(result["woe_table"]) > 0, "WoE table is non-empty")
    check(len(result["woe_map"]) > 0, "WoE map is non-empty")

    # Print WoE table for visual inspection
    woe_table = result["woe_table"]
    print(f"\n  WoE Table for borrower_credit_score:")
    print(f"  {'Bin':<25s} {'Count':>8s} {'Events':>8s} {'Event Rate':>10s} {'WoE':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in woe_table.iterrows():
        print(f"  {str(row['bin']):<25s} {row['total']:>8,.0f} "
              f"{row['events']:>8,.0f} {row['event_rate']:>10.4f} {row['woe']:>10.4f}")

    # WoE should be monotonically increasing with FICO (higher FICO = more
    # non-defaults = positive WoE). This is the fundamental credit risk
    # relationship.
    woe_values = woe_table["woe"].values
    is_generally_increasing = woe_values[-1] > woe_values[0]
    check(
        is_generally_increasing,
        "WoE generally increases with FICO (higher FICO = lower risk)",
        f"First bin WoE: {woe_values[0]:.4f}, Last bin WoE: {woe_values[-1]:.4f}",
    )

    return result


# ===========================================================================
# TEST 4: WoE/IV for all features
# ===========================================================================
def test_woe_all_features(sample_train):
    print("\n" + "=" * 70)
    print("TEST 4: WoE/IV for all candidate features")
    print("=" * 70)

    iv_summary, woe_results = calculate_woe_iv_all_features(
        sample_train, ALL_CANDIDATE_FEATURES, TARGET
    )

    check(len(iv_summary) == len(ALL_CANDIDATE_FEATURES), "IV computed for all features")

    # Print IV summary
    print(f"\n  Information Value Summary:")
    print(f"  {'Feature':<35s} {'IV':>10s} {'Interpretation':<15s}")
    print(f"  {'-'*35} {'-'*10} {'-'*15}")
    for _, row in iv_summary.iterrows():
        print(f"  {row['feature']:<35s} {row['iv']:>10.4f} {row['interpretation']:<15s}")

    # Key economic intuition checks:
    # FICO should be a strong predictor (IV > 0.10)
    fico_iv = iv_summary.loc[
        iv_summary["feature"] == "borrower_credit_score", "iv"
    ].values[0]
    check(
        fico_iv > 0.10,
        "FICO is at least a Medium predictor",
        f"IV = {fico_iv:.4f}",
    )

    # LTV should be predictive (IV > 0.02)
    ltv_iv = iv_summary.loc[
        iv_summary["feature"] == "original_ltv", "iv"
    ].values[0]
    check(
        ltv_iv > 0.02,
        "LTV is at least a Weak predictor",
        f"IV = {ltv_iv:.4f}",
    )

    # Count useful features (IV > 0.02)
    n_useful = (iv_summary["iv"] > 0.02).sum()
    check(
        n_useful >= 5,
        "At least 5 features with IV > 0.02",
        f"Found {n_useful} useful features",
    )

    # Select features with IV > 0.02 for modeling
    selected = iv_summary.loc[iv_summary["iv"] > 0.02, "feature"].tolist()
    print(f"\n  Selected features (IV > 0.02): {len(selected)}")
    for f in selected:
        iv_val = iv_summary.loc[iv_summary["feature"] == f, "iv"].values[0]
        print(f"    {f}: IV = {iv_val:.4f}")

    return iv_summary, woe_results, selected


# ===========================================================================
# TEST 5: WoE transformation
# ===========================================================================
def test_woe_transformation(sample_train, sample_val, woe_results, selected):
    print("\n" + "=" * 70)
    print("TEST 5: WoE transformation")
    print("=" * 70)

    X_train_woe = apply_woe_transformation(sample_train, woe_results, selected)
    X_val_woe = apply_woe_transformation(sample_val, woe_results, selected)

    check(
        X_train_woe.shape[1] == len(selected),
        "WoE output has correct number of columns",
        f"Expected {len(selected)}, got {X_train_woe.shape[1]}",
    )

    check(
        len(X_train_woe) == len(sample_train),
        "WoE output has correct number of rows",
    )

    # Check no NaN in WoE features (should be filled with 0)
    n_nan = X_train_woe.isna().sum().sum()
    check(n_nan == 0, "No NaN values in WoE-transformed features", f"NaN count: {n_nan}")

    # WoE values should be bounded (typically between -5 and +5)
    max_abs_woe = X_train_woe.abs().max().max()
    check(
        max_abs_woe < 10,
        "WoE values are reasonably bounded",
        f"Max absolute WoE: {max_abs_woe:.4f}",
    )

    # Print sample WoE values
    print(f"\n  Sample WoE values (first 5 rows):")
    print(X_train_woe.head())

    return X_train_woe, X_val_woe


# ===========================================================================
# TEST 6: Logistic Regression training
# ===========================================================================
def test_logistic_regression(X_train_woe, y_train, X_val_woe, y_val):
    print("\n" + "=" * 70)
    print("TEST 6: Logistic Regression training and scoring")
    print("=" * 70)

    model = train_logistic_regression(X_train_woe, y_train)

    check(hasattr(model, "predict_proba"), "Model has predict_proba method")

    # Score training data
    pred_train = model.predict_proba(X_train_woe)[:, 1]
    pred_val = model.predict_proba(X_val_woe)[:, 1]

    # Predictions should be probabilities between 0 and 1
    check(
        pred_train.min() >= 0 and pred_train.max() <= 1,
        "Training predictions in [0, 1]",
        f"Min: {pred_train.min():.6f}, Max: {pred_train.max():.6f}",
    )

    # AUC should be reasonable (> 0.60 even on sample)
    from sklearn.metrics import roc_auc_score
    auc_train = roc_auc_score(y_train, pred_train)
    auc_val = roc_auc_score(y_val, pred_val)

    check(auc_train > 0.60, "Train AUC > 0.60", f"AUC = {auc_train:.4f}")
    check(auc_val > 0.55, "Validation AUC > 0.55", f"AUC = {auc_val:.4f}")

    # Check for overfitting: train AUC should not be much higher than val AUC
    auc_gap = auc_train - auc_val
    check(
        auc_gap < 0.10,
        "AUC gap (train - val) < 0.10 (no severe overfitting)",
        f"Gap = {auc_gap:.4f}",
    )

    return model


# ===========================================================================
# TEST 7: XGBoost training
# ===========================================================================
def test_xgboost(sample_train, y_train, sample_val, y_val, selected):
    print("\n" + "=" * 70)
    print("TEST 7: XGBoost challenger training and scoring")
    print("=" * 70)

    # Prepare numeric features for XGBoost
    X_train_xgb, encoders = prepare_features_for_xgboost(sample_train, selected)
    X_val_xgb, _ = prepare_features_for_xgboost(sample_val, selected)

    check(
        X_train_xgb.isna().sum().sum() == 0,
        "No NaN in XGBoost features",
    )

    model = train_xgboost_challenger(X_train_xgb, y_train)

    pred_train = model.predict_proba(X_train_xgb)[:, 1]
    pred_val = model.predict_proba(X_val_xgb)[:, 1]

    from sklearn.metrics import roc_auc_score
    auc_train = roc_auc_score(y_train, pred_train)
    auc_val = roc_auc_score(y_val, pred_val)

    check(auc_train > 0.60, "XGBoost train AUC > 0.60", f"AUC = {auc_train:.4f}")
    check(auc_val > 0.55, "XGBoost val AUC > 0.55", f"AUC = {auc_val:.4f}")

    return model


# ===========================================================================
# TEST 8: Validation metrics
# ===========================================================================
def test_validation_metrics(y_true, y_pred):
    print("\n" + "=" * 70)
    print("TEST 8: Validation metrics computation")
    print("=" * 70)

    metrics = validate_model(y_true, y_pred, "Sample Test")

    check("auc" in metrics, "AUC computed")
    check("gini" in metrics, "Gini computed")
    check("ks" in metrics, "KS computed")
    check("brier" in metrics, "Brier score computed")

    check(0 < metrics["auc"] < 1, "AUC in valid range (0, 1)")
    check(metrics["ks"] > 0, "KS is positive")
    check(metrics["brier"] < 0.5, "Brier score is reasonable (< 0.5)")


# ===========================================================================
# TEST 9: Calibration table
# ===========================================================================
def test_calibration(y_true, y_pred):
    print("\n" + "=" * 70)
    print("TEST 9: Calibration table")
    print("=" * 70)

    cal_table = compute_calibration_table(y_true, y_pred)

    check(len(cal_table) > 0, "Calibration table is non-empty")
    check("avg_predicted_pd" in cal_table.columns, "Has predicted PD column")
    check("actual_default_rate" in cal_table.columns, "Has actual default rate column")

    # Print calibration table
    print(f"\n  Calibration Table:")
    print(f"  {'Decile':<25s} {'Count':>8s} {'Predicted':>10s} {'Actual':>10s} {'Ratio':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in cal_table.iterrows():
        print(f"  {str(row['decile']):<25s} {row['count']:>8,.0f} "
              f"{row['avg_predicted_pd']:>10.4f} {row['actual_default_rate']:>10.4f} "
              f"{row['ratio']:>8.2f}")

    # Rank ordering: actual default rate should GENERALLY increase across
    # deciles. On small samples, minor inversions in adjacent deciles are
    # normal due to noise. We check two things:
    # (a) The top decile has a much higher default rate than the bottom decile
    # (b) Spearman correlation between predicted and actual is positive
    actual_rates = cal_table["actual_default_rate"].values
    predicted_rates = cal_table["avg_predicted_pd"].values

    # The top predicted-risk decile should have at least 3x the default
    # rate of the bottom decile. This confirms the model separates risk.
    top_bottom_ratio = actual_rates[-1] / max(actual_rates[0], 0.001)
    check(
        top_bottom_ratio > 3.0,
        "Top decile default rate > 3x bottom decile (strong rank ordering)",
        f"Top: {actual_rates[-1]:.4f}, Bottom: {actual_rates[0]:.4f}, Ratio: {top_bottom_ratio:.1f}x",
    )

    # Spearman rank correlation between predicted and actual
    from scipy.stats import spearmanr
    corr, pval = spearmanr(predicted_rates, actual_rates)
    check(
        corr > 0.80,
        "Spearman correlation > 0.80 between predicted and actual",
        f"Correlation: {corr:.4f}, p-value: {pval:.6f}",
    )


# ===========================================================================
# TEST 10: PSI
# ===========================================================================
def test_psi(pred_train, pred_val):
    print("\n" + "=" * 70)
    print("TEST 10: PSI computation")
    print("=" * 70)

    psi_val, psi_detail = compute_psi(pred_train, pred_val)

    check(psi_val >= 0, "PSI is non-negative", f"PSI = {psi_val:.4f}")
    check(np.isfinite(psi_val), "PSI is finite")

    status = "STABLE" if psi_val < 0.10 else ("MONITOR" if psi_val < 0.25 else "UNSTABLE")
    print(f"\n  PSI = {psi_val:.4f} [{status}]")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - PD MODEL TEST SUITE")
    print("=" * 70)
    t_start = time.time()

    # Test 1: Load data
    df = test_load_data()

    # Test 2: Prepare splits and samples
    sample_train, sample_val, sample_test = test_prepare_splits(df)

    # Free full dataset from memory (we work with samples now)
    del df
    gc.collect()

    y_train = sample_train[TARGET]
    y_val = sample_val[TARGET]
    y_test = sample_test[TARGET]

    # Test 3: WoE on single feature
    test_woe_single_feature(sample_train)

    # Test 4: WoE/IV for all features
    iv_summary, woe_results, selected = test_woe_all_features(sample_train)

    # Test 5: WoE transformation
    X_train_woe, X_val_woe = test_woe_transformation(
        sample_train, sample_val, woe_results, selected
    )

    # Test 6: Logistic Regression
    lr_model = test_logistic_regression(X_train_woe, y_train, X_val_woe, y_val)

    # Test 7: XGBoost
    xgb_model = test_xgboost(sample_train, y_train, sample_val, y_val, selected)

    # Test 8: Validation metrics
    pred_val_lr = lr_model.predict_proba(X_val_woe)[:, 1]
    test_validation_metrics(y_val, pred_val_lr)

    # Test 9: Calibration
    test_calibration(y_val, pred_val_lr)

    # Test 10: PSI
    pred_train_lr = lr_model.predict_proba(X_train_woe)[:, 1]
    test_psi(pred_train_lr, pred_val_lr)

    # --- Final Report ---
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Time:     {elapsed:.1f} seconds")

    if failed == 0:
        print("\n  ALL TESTS PASSED.")
        print("  Safe to run the full PD model pipeline on 3.8M loans.")
    else:
        print(f"\n  {failed} TEST(S) FAILED.")
        print("  Review the failures above before running the full pipeline.")

    sys.exit(0 if failed == 0 else 1)