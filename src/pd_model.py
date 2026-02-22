"""
PD (Probability of Default) Model Module
=========================================

This module implements the full PD modeling pipeline for CECL credit risk:
1. Weight of Evidence (WoE) transformation
2. Information Value (IV) calculation for variable selection
3. Logistic Regression (primary model, industry standard)
4. XGBoost (challenger model)
5. Comprehensive model validation (AUC, Gini, KS, calibration, PSI)

Why Logistic Regression for PD?
- Interpretability: regulators ask "WHY is this loan risky?" Coefficients
  directly answer this question.
- Monotonicity: with WoE-transformed inputs, relationships are guaranteed
  monotonic (higher FICO = lower risk, always).
- Stability: coefficients are stable over time, unlike tree-based models.
- Regulatory acceptance: OCC, Fed, FDIC have decades of comfort with
  logistic regression for credit risk.

Why XGBoost as challenger?
- Better discrimination (higher AUC) due to non-linear splits.
- Captures interactions automatically.
- Serves as a benchmark to evaluate the logistic regression's performance gap.
- Industry practice: regulators expect a challenger model comparison.

Author: Saurabh Chavan
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# WoE / IV Calculation
# ---------------------------------------------------------------------------

def calculate_woe_iv_for_feature(df, feature, target, n_bins=10, min_pct=0.05):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a
    single feature.

    WoE is the standard preprocessing technique in credit risk scorecards.
    It transforms each feature into a measure of its predictive power for
    separating defaults from non-defaults.

    For each bin of the feature:
        WoE = ln(% of Non-Defaults in bin / % of Defaults in bin)

    Interpretation:
        Positive WoE = more non-defaults than defaults = low risk
        Negative WoE = more defaults than non-defaults = high risk
        Magnitude = strength of the signal

    IV = SUM over bins of: (% Non-Defaults - % Defaults) * WoE
    IV interpretation:
        < 0.02  = not useful, drop it
        0.02-0.10 = weak predictor
        0.10-0.30 = medium predictor
        0.30-0.50 = strong predictor
        > 0.50  = suspiciously strong (check for data leakage)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with the feature and target columns.
    feature : str
        Name of the feature column.
    target : str
        Name of the binary target column (0/1).
    n_bins : int
        Number of bins for continuous features.
    min_pct : float
        Minimum percentage of observations per bin. Bins below this
        threshold are merged to avoid unstable WoE estimates.

    Returns
    -------
    dict with keys:
        'feature': feature name
        'iv': total Information Value
        'woe_table': DataFrame with bin-level WoE details
        'woe_map': dict mapping bin labels to WoE values
    """
    temp = df[[feature, target]].copy()
    temp = temp.dropna(subset=[feature])

    if len(temp) == 0:
        return {
            "feature": feature,
            "iv": 0.0,
            "woe_table": pd.DataFrame(),
            "woe_map": {},
        }

    # Determine if feature is categorical or continuous
    is_categorical = (
        temp[feature].dtype == "object"
        or temp[feature].dtype.name == "category"
        or temp[feature].nunique() <= 10
    )

    if is_categorical:
        # For categorical features, each unique value is a bin
        temp["bin"] = temp[feature].astype(str)
    else:
        # For continuous features, use quantile-based binning.
        # pd.qcut creates bins with approximately equal number of observations.
        # Duplicates='drop' handles features with many tied values.
        try:
            temp["bin"] = pd.qcut(temp[feature], q=n_bins, duplicates="drop")
        except ValueError:
            # If quantile binning fails (too few unique values), treat as categorical
            temp["bin"] = temp[feature].astype(str)

    # Count events (defaults) and non-events per bin
    grouped = temp.groupby("bin", observed=True)[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    # Merge small bins: bins with fewer than min_pct of total observations
    # produce unstable WoE estimates. In production, these would be merged
    # with adjacent bins. For simplicity, we flag them but keep them.
    total_obs = grouped["total"].sum()
    grouped["pct_of_total"] = grouped["total"] / total_obs

    # Calculate distribution of events and non-events across bins
    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    if total_events == 0 or total_non_events == 0:
        # Cannot compute WoE if there are no events or no non-events
        return {
            "feature": feature,
            "iv": 0.0,
            "woe_table": pd.DataFrame(),
            "woe_map": {},
        }

    grouped["dist_events"] = grouped["events"] / total_events
    grouped["dist_non_events"] = grouped["non_events"] / total_non_events

    # Apply Laplace smoothing to avoid division by zero and log(0).
    # Add 0.5 to both numerator and denominator. This is standard practice
    # in credit risk to handle bins with zero events or zero non-events.
    smoothing = 0.5
    grouped["dist_events_smooth"] = (
        (grouped["events"] + smoothing) / (total_events + smoothing * len(grouped))
    )
    grouped["dist_non_events_smooth"] = (
        (grouped["non_events"] + smoothing) / (total_non_events + smoothing * len(grouped))
    )

    # WoE calculation
    grouped["woe"] = np.log(
        grouped["dist_non_events_smooth"] / grouped["dist_events_smooth"]
    )

    # IV calculation per bin
    grouped["iv_bin"] = (
        (grouped["dist_non_events_smooth"] - grouped["dist_events_smooth"])
        * grouped["woe"]
    )

    total_iv = grouped["iv_bin"].sum()

    # Event rate per bin (for reporting)
    grouped["event_rate"] = grouped["events"] / grouped["total"]

    # Build WoE mapping
    woe_map = grouped["woe"].to_dict()

    # Build clean output table
    woe_table = grouped[[
        "total", "events", "non_events", "event_rate",
        "dist_events", "dist_non_events", "woe", "iv_bin",
    ]].copy()
    woe_table = woe_table.reset_index()

    return {
        "feature": feature,
        "iv": total_iv,
        "woe_table": woe_table,
        "woe_map": woe_map,
    }


def calculate_woe_iv_all_features(df, features, target, n_bins=10):
    """
    Calculate WoE and IV for all features in the list.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target.
    features : list of str
        Feature column names.
    target : str
        Binary target column name.
    n_bins : int
        Number of bins for continuous features.

    Returns
    -------
    iv_summary : pd.DataFrame
        Summary of IV for each feature, sorted descending.
    woe_results : dict
        Feature name -> WoE result dict (from calculate_woe_iv_for_feature).
    """
    print(f"\nCalculating WoE/IV for {len(features)} features...")
    results = {}
    iv_rows = []

    for i, feat in enumerate(features):
        result = calculate_woe_iv_for_feature(df, feat, target, n_bins=n_bins)
        results[feat] = result
        iv_rows.append({"feature": feat, "iv": result["iv"]})

        if (i + 1) % 10 == 0 or (i + 1) == len(features):
            print(f"  Processed {i+1}/{len(features)} features")

    iv_summary = pd.DataFrame(iv_rows)
    iv_summary = iv_summary.sort_values("iv", ascending=False).reset_index(drop=True)

    # Add IV interpretation
    def interpret_iv(iv_val):
        if iv_val < 0.02:
            return "Not useful"
        elif iv_val < 0.10:
            return "Weak"
        elif iv_val < 0.30:
            return "Medium"
        elif iv_val < 0.50:
            return "Strong"
        else:
            return "Suspicious"

    iv_summary["interpretation"] = iv_summary["iv"].apply(interpret_iv)

    return iv_summary, results


def apply_woe_transformation(df, woe_results, selected_features):
    """
    Apply WoE transformation to selected features.

    Replaces each feature's values with the corresponding bin's WoE value.
    After transformation, logistic regression sees clean, monotonic, linear
    inputs.

    CRITICAL: For continuous features, we must use the SAME bin edges that
    were computed during WoE calculation on the training data. If we re-bin
    new data using pd.qcut, the bin boundaries change (because the new data
    has a different distribution), which assigns loans to wrong WoE values
    and destroys model discrimination on validation/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to transform.
    woe_results : dict
        WoE results from calculate_woe_iv_all_features.
    selected_features : list of str
        Features to transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with WoE-transformed features (columns named <feature>_woe).
    """
    woe_df = pd.DataFrame(index=df.index)

    for feat in selected_features:
        result = woe_results[feat]
        woe_map = result["woe_map"]
        woe_table = result["woe_table"]

        if len(woe_map) == 0:
            woe_df[f"{feat}_woe"] = 0.0
            continue

        col_data = df[feat]

        is_categorical = (
            col_data.dtype == "object"
            or col_data.dtype.name == "category"
            or col_data.nunique() <= 10
        )

        if is_categorical:
            # Map string values to WoE
            mapped = col_data.astype(str).map(woe_map)
            mapped = mapped.astype(float).fillna(0.0)
            woe_df[f"{feat}_woe"] = mapped
        else:
            # For continuous features, extract the bin edges from the
            # woe_table that was computed on training data. The 'bin'
            # column contains pd.Interval objects from pd.qcut.
            # We extract the left/right edges to rebuild the exact
            # same bins for any new dataset.
            try:
                intervals = woe_table["bin"].values
                # Extract bin edges from the Interval objects
                edges = []
                for interval in intervals:
                    if hasattr(interval, "left"):
                        edges.append(interval.left)
                # Add the right edge of the last interval
                last_interval = intervals[-1]
                if hasattr(last_interval, "right"):
                    edges.append(last_interval.right)

                if len(edges) >= 2:
                    # Extend edges to cover any values outside the training range.
                    # New data may have values below the training min or above
                    # the training max. Setting -inf/+inf ensures every value
                    # falls into a bin.
                    edges[0] = -np.inf
                    edges[-1] = np.inf

                    # Use pd.cut with the FIXED training bin edges
                    temp_binned = pd.cut(col_data, bins=edges)
                    mapped = temp_binned.map(woe_map)
                    mapped = mapped.astype(float).fillna(0.0)
                    woe_df[f"{feat}_woe"] = mapped
                else:
                    # Fallback if edge extraction fails
                    woe_df[f"{feat}_woe"] = 0.0

            except (ValueError, KeyError, TypeError):
                # Fallback: use string-based mapping
                temp_binned = col_data.astype(str)
                mapped = temp_binned.map(woe_map)
                mapped = mapped.astype(float).fillna(0.0)
                woe_df[f"{feat}_woe"] = mapped

    return woe_df


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

# Candidate features for PD modeling.
# These are the features we will evaluate using IV.
# Excluded: identifiers, target variables, LGD/EAD fields (those are for
# the LGD/EAD models), and raw fields that have been replaced by derived
# features (e.g., origination_date replaced by origination_year/month).

CONTINUOUS_FEATURES = [
    "borrower_credit_score",
    "original_ltv",
    "original_cltv",
    "dti",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "number_of_borrowers",
    "fico_x_ltv",
    "fico_x_dti",
    "unemployment_rate",
    "fed_funds_rate",
    "mortgage_rate_30y",
    "hpi_national",
    "treasury_10y",
    "baa_spread",
    "unemployment_change_12m",
    "hpi_change_12m_pct",
    "gdp_growth_pct",
    "fico_x_unemployment",
]

CATEGORICAL_FEATURES = [
    "fico_bucket",
    "ltv_bucket",
    "dti_bucket",
    "is_first_time_buyer",
    "is_investment_property",
    "is_second_home",
    "is_cashout_refi",
    "is_refi_nocashout",
    "channel_broker",
    "channel_correspondent",
    "channel_retail",
    "has_coborrower",
    "has_mortgage_insurance",
    "is_condo",
    "is_manufactured_housing",
    "is_multi_unit",
    "dti_missing",
    "fico_missing",
]

ALL_CANDIDATE_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression PD model on WoE-transformed features.

    Uses L2 regularization (ridge) to prevent overfitting.
    class_weight='balanced' handles the imbalanced classes (defaults are
    the minority class at ~12% in training data).

    Parameters
    ----------
    X_train : pd.DataFrame
        WoE-transformed training features.
    y_train : pd.Series
        Binary target (0/1).

    Returns
    -------
    LogisticRegression
        Trained model.
    """
    print("\nTraining Logistic Regression PD model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Default rate: {y_train.mean():.4f}")

    t_start = time.time()

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    elapsed = time.time() - t_start
    print(f"  Training completed in {elapsed:.1f} seconds")

    # Print coefficients
    print(f"\n  Model coefficients:")
    print(f"  {'Feature':<40s} {'Coefficient':>12s}")
    print(f"  {'-'*40} {'-'*12}")
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<40s} {row['coefficient']:>12.4f}")

    print(f"  {'Intercept':<40s} {model.intercept_[0]:>12.4f}")

    return model


def train_xgboost_challenger(X_train, y_train):
    """
    Train an XGBoost challenger model on ORIGINAL (non-WoE) features.

    XGBoost handles non-linearity and interactions natively, so it does
    not need WoE transformation. It typically achieves higher AUC but
    is less interpretable.

    Parameters
    ----------
    X_train : pd.DataFrame
        Original (non-WoE) training features. Must be numeric.
    y_train : pd.Series
        Binary target (0/1).

    Returns
    -------
    xgb.XGBClassifier
        Trained model.
    """
    import xgboost as xgb

    print("\nTraining XGBoost challenger model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")

    # Calculate scale_pos_weight for class imbalance handling.
    # This is the ratio of non-defaults to defaults.
    n_default = y_train.sum()
    n_non_default = len(y_train) - n_default
    scale_pos_weight = n_non_default / n_default
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    t_start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,             # Keep shallow for interpretability
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    model.fit(X_train, y_train, verbose=False)

    elapsed = time.time() - t_start
    print(f"  Training completed in {elapsed:.1f} seconds")

    # Print feature importance (top 20)
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 20 feature importances:")
    print(f"  {'Feature':<40s} {'Importance':>12s}")
    print(f"  {'-'*40} {'-'*12}")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']:<40s} {row['importance']:>12.4f}")

    return model


# ---------------------------------------------------------------------------
# Model Validation
# ---------------------------------------------------------------------------

def validate_model(y_true, y_pred_proba, dataset_label=""):
    """
    Compute comprehensive validation metrics for a PD model.

    Metrics computed:
    - AUC-ROC: Overall ability to separate defaults from non-defaults.
      > 0.75 is good for credit risk.
    - Gini coefficient: = 2*AUC - 1. > 0.50 is good.
    - KS statistic: Maximum separation between cumulative default and
      non-default distributions. > 0.30 is good.
    - Brier score: Mean squared error of predicted probabilities.
      Lower is better. Measures calibration + discrimination jointly.

    Parameters
    ----------
    y_true : array-like
        Actual binary outcomes (0/1).
    y_pred_proba : array-like
        Predicted probabilities of default.
    dataset_label : str
        Label for printing (e.g., "Train", "Validation", "Test").

    Returns
    -------
    dict
        Dictionary of all metrics.
    """
    metrics = {}

    # AUC-ROC
    auc = roc_auc_score(y_true, y_pred_proba)
    metrics["auc"] = auc
    metrics["gini"] = 2 * auc - 1

    # KS Statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    metrics["ks"] = ks

    # Brier Score (calibration measure)
    brier = brier_score_loss(y_true, y_pred_proba)
    metrics["brier"] = brier

    # Print results
    label = f" ({dataset_label})" if dataset_label else ""
    print(f"\n  Validation Metrics{label}:")
    print(f"    AUC-ROC:      {auc:.4f}  {'GOOD' if auc > 0.75 else 'NEEDS IMPROVEMENT'}")
    print(f"    Gini:         {metrics['gini']:.4f}  {'GOOD' if metrics['gini'] > 0.50 else 'NEEDS IMPROVEMENT'}")
    print(f"    KS Statistic: {ks:.4f}  {'GOOD' if ks > 0.30 else 'NEEDS IMPROVEMENT'}")
    print(f"    Brier Score:  {brier:.4f}  (lower is better)")

    return metrics


def compute_calibration_table(y_true, y_pred_proba, n_bins=10):
    """
    Compute calibration (predicted vs actual) by decile.

    This is MORE IMPORTANT than discrimination in credit risk because
    the predicted probabilities directly determine loss reserves.
    If the model says 5% PD but actual is 8%, the bank is under-reserved.

    Parameters
    ----------
    y_true : array-like
        Actual binary outcomes.
    y_pred_proba : array-like
        Predicted probabilities.
    n_bins : int
        Number of bins (deciles = 10).

    Returns
    -------
    pd.DataFrame
        Calibration table with predicted PD, actual default rate, and counts.
    """
    temp = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred_proba,
    })

    # Create decile bins based on predicted probability
    try:
        temp["decile"] = pd.qcut(temp["predicted"], q=n_bins, duplicates="drop")
    except ValueError:
        temp["decile"] = pd.cut(temp["predicted"], bins=n_bins)

    cal_table = temp.groupby("decile", observed=True).agg(
        count=("actual", "count"),
        n_defaults=("actual", "sum"),
        avg_predicted_pd=("predicted", "mean"),
        actual_default_rate=("actual", "mean"),
    ).reset_index()

    cal_table["ratio"] = cal_table["actual_default_rate"] / cal_table["avg_predicted_pd"]

    return cal_table


def compute_psi(y_pred_train, y_pred_test, n_bins=10):
    """
    Compute Population Stability Index (PSI).

    PSI measures how much the score distribution has shifted between
    the training population and a new population. Large shifts indicate
    the model may need recalibration.

    PSI < 0.10: Stable (no action needed)
    PSI 0.10-0.25: Minor shift (monitor closely)
    PSI > 0.25: Significant shift (consider recalibration)

    Parameters
    ----------
    y_pred_train : array-like
        Predicted probabilities on training data.
    y_pred_test : array-like
        Predicted probabilities on test data.
    n_bins : int
        Number of bins for the comparison.

    Returns
    -------
    float
        PSI value.
    pd.DataFrame
        Bin-level PSI details.
    """
    # Convert to numpy arrays to avoid any pandas dtype issues
    train_arr = np.asarray(y_pred_train, dtype=float)
    test_arr = np.asarray(y_pred_test, dtype=float)

    # Create bin edges from training distribution using numpy percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(train_arr, percentiles)

    # Ensure unique bin edges (handles tied values)
    bin_edges = np.unique(bin_edges)

    # Extend edges to cover full range
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute histogram counts using numpy (no pandas dependency)
    train_counts = np.histogram(train_arr, bins=bin_edges)[0]
    test_counts = np.histogram(test_arr, bins=bin_edges)[0]

    # Convert to percentages
    train_pcts = train_counts / train_counts.sum()
    test_pcts = test_counts / test_counts.sum()

    # Replace zeros with small value to avoid division by zero and log(0)
    train_pcts = np.maximum(train_pcts, 0.0001)
    test_pcts = np.maximum(test_pcts, 0.0001)

    # PSI per bin
    psi_bins = (test_pcts - train_pcts) * np.log(test_pcts / train_pcts)
    psi_total = float(psi_bins.sum())

    # Build detail table
    bin_labels = [
        f"({bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]"
        for i in range(len(bin_edges) - 1)
    ]
    psi_table = pd.DataFrame({
        "bin": bin_labels,
        "train_pct": train_pcts,
        "test_pct": test_pcts,
        "psi_bin": psi_bins,
    })

    return psi_total, psi_table


def run_full_validation(
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_name="Model",
):
    """
    Run comprehensive validation across train, validation, and test sets.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must have predict_proba method.
    X_train, y_train : training data
    X_val, y_val : validation data (out-of-time: 2006 vintage)
    X_test, y_test : test data (out-of-time: 2007 vintage, crisis period)
    model_name : str
        Label for the model.

    Returns
    -------
    dict
        All validation results.
    """
    print(f"\n{'='*70}")
    print(f"FULL VALIDATION: {model_name}")
    print(f"{'='*70}")

    results = {"model_name": model_name}

    # Generate predictions
    pred_train = model.predict_proba(X_train)[:, 1]
    pred_val = model.predict_proba(X_val)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    # Discrimination metrics
    results["train_metrics"] = validate_model(y_train, pred_train, "Train")
    results["val_metrics"] = validate_model(y_val, pred_val, "Validation (2006)")
    results["test_metrics"] = validate_model(y_test, pred_test, "Test (2007)")

    # Calibration
    print(f"\n  Calibration Table (Validation Set):")
    cal_table = compute_calibration_table(y_val, pred_val)
    results["calibration_table_val"] = cal_table
    print(f"  {'Decile':<25s} {'Count':>8s} {'Predicted':>10s} {'Actual':>10s} {'Ratio':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in cal_table.iterrows():
        print(f"  {str(row['decile']):<25s} {row['count']:>8,.0f} "
              f"{row['avg_predicted_pd']:>10.4f} {row['actual_default_rate']:>10.4f} "
              f"{row['ratio']:>8.2f}")

    # PSI: Train vs Validation
    psi_val, psi_detail_val = compute_psi(pred_train, pred_val)
    results["psi_val"] = psi_val
    status = "STABLE" if psi_val < 0.10 else ("MONITOR" if psi_val < 0.25 else "UNSTABLE")
    print(f"\n  PSI (Train vs Validation): {psi_val:.4f} [{status}]")

    # PSI: Train vs Test
    psi_test, psi_detail_test = compute_psi(pred_train, pred_test)
    results["psi_test"] = psi_test
    status = "STABLE" if psi_test < 0.10 else ("MONITOR" if psi_test < 0.25 else "UNSTABLE")
    print(f"  PSI (Train vs Test):       {psi_test:.4f} [{status}]")

    # Prediction distribution summary
    print(f"\n  Prediction Distribution:")
    for label, preds in [("Train", pred_train), ("Validation", pred_val), ("Test", pred_test)]:
        print(f"    {label:>12s}: mean={preds.mean():.4f}, "
              f"median={np.median(preds):.4f}, "
              f"std={preds.std():.4f}, "
              f"min={preds.min():.4f}, max={preds.max():.4f}")

    results["pred_train"] = pred_train
    results["pred_val"] = pred_val
    results["pred_test"] = pred_test

    return results


def prepare_features_for_xgboost(df, features):
    """
    Prepare features for XGBoost by converting categoricals to numeric.

    XGBoost requires all-numeric input. Categorical features (fico_bucket,
    ltv_bucket, dti_bucket) are label-encoded.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features.
    features : list of str
        Feature column names.

    Returns
    -------
    pd.DataFrame
        All-numeric feature DataFrame.
    dict
        Label encoders for categorical columns (for scoring new data).
    """
    result = pd.DataFrame(index=df.index)
    encoders = {}

    for feat in features:
        if feat not in df.columns:
            continue

        col = df[feat]

        if col.dtype == "object" or col.dtype.name == "category":
            le = LabelEncoder()
            # Handle NaN by converting to string first
            encoded = le.fit_transform(col.astype(str).fillna("MISSING"))
            result[feat] = encoded
            encoders[feat] = le
        else:
            result[feat] = col.fillna(col.median())

    return result, encoders