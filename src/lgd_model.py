"""
LGD (Loss Given Default) Model Module
=======================================

This module models the severity of loss when a borrower defaults.
LGD answers: "If this loan defaults, what fraction of the outstanding
balance will the bank lose?"

LGD = (EAD - Recovery + Costs) / EAD

Key drivers of LGD for mortgages:
1. LTV at default: If the borrower owes more than the home is worth
   (underwater), the bank cannot recover the full balance from selling
   the property. Higher LTV = higher LGD.
2. Home price changes since origination: Falling home prices reduce
   the collateral value, increasing LGD.
3. Foreclosure costs: Legal fees, property maintenance, taxes during
   the foreclosure process add to losses.
4. Time to resolution: Longer foreclosure timelines (judicial states)
   mean more carrying costs and further price deterioration.
5. Economic conditions: Recessions depress property values AND extend
   time-to-sell, compounding losses.

Model choice: Two-stage approach
  Stage 1: OLS regression on LGD (simple, interpretable, standard in banking)
  Stage 2: Comparison with gradient-boosted regression (XGBoost) as challenger

Why not Beta regression?
  While LGD is theoretically bounded [0,1], our data has LGD values up to
  1.5 (costs exceeding balance). Beta regression requires values strictly
  in (0,1). OLS with capped predictions is the pragmatic industry approach
  and produces easily interpretable coefficients for model documentation.

Training population: ONLY defaulted loans with valid LGD values (278K loans).
Non-defaulted loans are excluded because they have no observed loss.

Author: Saurabh Chavan
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------

# Features for LGD modeling. These are selected based on economic intuition
# about what drives loss severity after default occurs.
LGD_FEATURES = [
    # Collateral risk (strongest LGD drivers)
    "original_ltv",
    # Higher LTV at origination means less equity buffer. If the borrower
    # put down only 5% and prices fell 10%, the bank is underwater by 5%
    # of the loan balance before even accounting for foreclosure costs.

    "original_cltv",
    # Combined LTV includes second liens. A borrower with 80% first lien
    # LTV and a 15% HELOC has 95% CLTV -- very little equity cushion.

    "has_mortgage_insurance",
    # MI reduces LGD because the insurer covers a portion of the loss.
    # Loans with MI may have 20-30% lower LGD than uninsured loans.

    # Borrower characteristics
    "borrower_credit_score",
    # Lower FICO borrowers tend to have higher LGD because they are more
    # likely to delay the foreclosure process, accumulating costs, and
    # their properties may be in worse condition.

    "dti",
    # Higher DTI borrowers have less capacity to cure a default, leading
    # to longer resolution times and higher costs.

    # Loan characteristics
    "original_interest_rate",
    # Higher rate loans were priced for risk. The rate itself does not
    # directly drive LGD, but serves as a proxy for unmeasured risk
    # factors the original underwriter identified.

    "original_upb",
    # Larger loans may have different LGD profiles due to property type
    # and market segment differences.

    "loan_age_at_default",
    # Loans that default early (within 2 years) have different LGD
    # profiles than seasoned loans. Early defaults often indicate fraud
    # or misrepresentation. Seasoned defaults after a rate reset or
    # economic shock may have more equity built up.

    "was_modified",
    # Modified loans went through a loss mitigation process. If the
    # modification failed and the loan still defaulted, the LGD may
    # be higher because the bank already made concessions.

    # Property characteristics
    "is_investment_property",
    # Investment properties may sell at different speeds and prices.
    # Banks may recover less because investors are more strategic
    # about walking away.

    "is_condo",
    # Condos may have different resale values and market liquidity
    # compared to single-family homes.

    # Macro conditions at origination
    "unemployment_rate",
    # Higher unemployment at time of default correlates with worse
    # recovery rates because fewer buyers are in the market.

    "hpi_national",
    # Home price index at origination. Combined with current HPI,
    # this determines equity position. Loans originated at peak HPI
    # (2006) had the worst LGD when prices fell 33%.
]


def prepare_lgd_dataset(df):
    """
    Prepare the LGD modeling dataset from the combined loan-level data.

    Filters to defaulted loans with valid LGD, applies the time-based
    train/validation/test split, and selects LGD features.

    Parameters
    ----------
    df : pd.DataFrame
        Combined loan-level dataset with default_flag, lgd, and data_split.

    Returns
    -------
    dict with keys: train, val, test (each a pd.DataFrame)
    dict with keys: y_train, y_val, y_test (each a pd.Series of LGD values)
    list of feature names actually available
    """
    print("\nPreparing LGD dataset...")

    # Filter to defaulted loans with valid LGD
    lgd_data = df[(df["default_flag"] == 1) & (df["lgd"].notna())].copy()
    print(f"  Total defaulted loans with valid LGD: {len(lgd_data):,}")

    # Show LGD distribution
    print(f"  LGD distribution:")
    print(f"    Mean:   {lgd_data['lgd'].mean():.4f}")
    print(f"    Median: {lgd_data['lgd'].median():.4f}")
    print(f"    Std:    {lgd_data['lgd'].std():.4f}")
    print(f"    Min:    {lgd_data['lgd'].min():.4f}")
    print(f"    Max:    {lgd_data['lgd'].max():.4f}")

    # Check which features are available
    available_features = [f for f in LGD_FEATURES if f in lgd_data.columns]
    missing_features = [f for f in LGD_FEATURES if f not in lgd_data.columns]
    if missing_features:
        print(f"  WARNING: Missing features: {missing_features}")
    print(f"  Available features: {len(available_features)}")

    # Split by vintage (same split as PD model)
    # Move unknown into train
    lgd_data.loc[lgd_data["data_split"] == "unknown", "data_split"] = "train"

    train = lgd_data[lgd_data["data_split"] == "train"]
    val = lgd_data[lgd_data["data_split"] == "validation"]
    test = lgd_data[lgd_data["data_split"] == "test"]

    print(f"\n  Split sizes:")
    print(f"    Train:      {len(train):>8,} loans, mean LGD: {train['lgd'].mean():.4f}")
    print(f"    Validation: {len(val):>8,} loans, mean LGD: {val['lgd'].mean():.4f}")
    print(f"    Test:       {len(test):>8,} loans, mean LGD: {test['lgd'].mean():.4f}")

    # Prepare feature matrices (fill NaN with median from training set)
    X_train = train[available_features].copy()
    X_val = val[available_features].copy()
    X_test = test[available_features].copy()

    # Fill missing values with training median (standard practice)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    y_train = train["lgd"]
    y_val = val["lgd"]
    y_test = test["lgd"]

    # Report missing value handling
    for feat in available_features:
        n_missing = train[feat].isna().sum()
        if n_missing > 0:
            pct = 100 * n_missing / len(train)
            print(f"    {feat}: {n_missing:,} missing ({pct:.1f}%) filled with median {train_medians[feat]:.4f}")

    data = {"train": X_train, "val": X_val, "test": X_test}
    targets = {"y_train": y_train, "y_val": y_val, "y_test": y_test}

    return data, targets, available_features


def train_lgd_ols(X_train, y_train):
    """
    Train an OLS linear regression LGD model.

    OLS is chosen because:
    1. Coefficients are directly interpretable (each unit change in LTV
       increases LGD by X percentage points)
    2. Standard in banking LGD modeling
    3. Easy to document and explain to model risk reviewers
    4. Predictions can be clipped to [0, 1.5] post-hoc

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        LGD target values.

    Returns
    -------
    LinearRegression
        Trained model.
    """
    print("\nTraining OLS LGD model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")

    t0 = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f} seconds")

    # Print coefficients with interpretation
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    print(f"\n  Coefficients:")
    print(f"  {'Feature':<30s} {'Coefficient':>12s} {'Interpretation'}")
    print(f"  {'-'*30} {'-'*12} {'-'*40}")
    for _, row in coef_df.iterrows():
        coef = row["coefficient"]
        feat = row["feature"]
        # Provide economic interpretation
        if feat == "original_ltv":
            interp = f"+1 LTV point -> {coef*1:+.4f} LGD"
        elif feat == "borrower_credit_score":
            interp = f"+10 FICO -> {coef*10:+.4f} LGD"
        elif feat == "loan_age_at_default":
            interp = f"+12 months age -> {coef*12:+.4f} LGD"
        elif feat == "unemployment_rate":
            interp = f"+1% unemployment -> {coef*1:+.4f} LGD"
        elif feat == "hpi_national":
            interp = f"+10 HPI points -> {coef*10:+.4f} LGD"
        else:
            interp = ""
        print(f"  {feat:<30s} {coef:>12.6f} {interp}")

    print(f"  {'Intercept':<30s} {model.intercept_:>12.6f}")

    return model


def train_lgd_xgboost(X_train, y_train):
    """
    Train an XGBoost gradient-boosted regression model for LGD.

    Serves as challenger model to benchmark OLS performance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        LGD target values.

    Returns
    -------
    xgb.XGBRegressor
        Trained model.
    """
    import xgboost as xgb

    print("\nTraining XGBoost LGD model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")

    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, verbose=False)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f} seconds")

    # Feature importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature Importances:")
    print(f"  {'Feature':<30s} {'Importance':>12s}")
    print(f"  {'-'*30} {'-'*12}")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<30s} {row['importance']:>12.4f}")

    return model


def validate_lgd_model(model, X, y_true, label="", clip_range=(0.0, 1.5)):
    """
    Validate an LGD model with standard regression metrics.

    Metrics:
    - R-squared: proportion of variance explained (higher is better)
    - RMSE: root mean squared error (lower is better)
    - MAE: mean absolute error (lower is better)
    - Mean predicted vs mean actual (calibration check)

    Parameters
    ----------
    model : fitted model with predict method
    X : pd.DataFrame
        Feature matrix.
    y_true : pd.Series
        Actual LGD values.
    label : str
        Label for printing.
    clip_range : tuple
        Min/max for prediction clipping.

    Returns
    -------
    dict of metrics, np.array of predictions
    """
    y_pred = model.predict(X)

    # Clip predictions to valid range
    y_pred = np.clip(y_pred, clip_range[0], clip_range[1])

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_pred = y_pred.mean()
    mean_actual = y_true.mean()

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_predicted": mean_pred,
        "mean_actual": mean_actual,
    }

    tag = f" ({label})" if label else ""
    print(f"\n  LGD Validation Metrics{tag}:")
    print(f"    R-squared:       {r2:.4f}")
    print(f"    RMSE:            {rmse:.4f}")
    print(f"    MAE:             {mae:.4f}")
    print(f"    Mean Predicted:  {mean_pred:.4f}")
    print(f"    Mean Actual:     {mean_actual:.4f}")
    print(f"    Calibration:     {mean_pred/mean_actual:.4f} (1.0 = perfect)")

    return metrics, y_pred


def compute_lgd_by_segment(y_true, y_pred, segment_values, segment_name):
    """
    Compute predicted vs actual LGD by a segmentation variable.

    Parameters
    ----------
    y_true : array-like
        Actual LGD values.
    y_pred : array-like
        Predicted LGD values.
    segment_values : array-like
        Segmentation variable. Must be same length as y_true and y_pred.
    segment_name : str
        Name for printing.

    Returns
    -------
    pd.DataFrame
        Segment-level comparison.
    """
    # Reset everything to plain numpy/lists to avoid index alignment issues
    actual = np.asarray(y_true).ravel()
    predicted = np.asarray(y_pred).ravel()
    segments = np.asarray(segment_values).ravel()

    assert len(actual) == len(predicted) == len(segments), (
        f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}, "
        f"segments={len(segments)}"
    )

    temp = pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "segment": segments,
    })

    seg_table = temp.groupby("segment").agg(
        count=("actual", "count"),
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
    ).reset_index()

    seg_table["ratio"] = seg_table["mean_predicted"] / seg_table["mean_actual"]

    print(f"\n  LGD by {segment_name}:")
    print(f"  {'Segment':<20s} {'Count':>8s} {'Actual':>10s} {'Predicted':>10s} {'Ratio':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in seg_table.iterrows():
        print(f"  {str(row['segment']):<20s} {row['count']:>8,.0f} "
              f"{row['mean_actual']:>10.4f} {row['mean_predicted']:>10.4f} "
              f"{row['ratio']:>8.2f}")

    return seg_table


def macro_sensitivity_check(model, X_baseline, feature_name, shock_values, feature_list):
    """
    Check LGD model sensitivity to a macroeconomic variable.

    Holds all features at their baseline values and varies one feature
    through a range of shock values. This demonstrates the model responds
    appropriately to economic conditions.

    For example: Does LGD increase when unemployment rises?
    If not, the model has a fundamental flaw.

    Parameters
    ----------
    model : fitted model
    X_baseline : pd.DataFrame
        Baseline feature values (typically training medians).
    feature_name : str
        The macro feature to shock.
    shock_values : list
        Values to test.
    feature_list : list
        Full feature list for the model.

    Returns
    -------
    pd.DataFrame
        Sensitivity table.
    """
    if feature_name not in feature_list:
        print(f"  {feature_name} not in model features, skipping sensitivity check")
        return pd.DataFrame()

    results = []
    baseline_median = X_baseline[feature_name].median()

    for shock_val in shock_values:
        X_shock = X_baseline.copy()
        X_shock[feature_name] = shock_val
        pred = model.predict(X_shock)
        pred = np.clip(pred, 0.0, 1.5)
        results.append({
            "shock_value": shock_val,
            "mean_lgd": pred.mean(),
        })

    sensitivity = pd.DataFrame(results)
    sensitivity["lgd_change"] = sensitivity["mean_lgd"] - sensitivity["mean_lgd"].iloc[0]

    print(f"\n  Sensitivity: {feature_name} (baseline median: {baseline_median:.2f})")
    print(f"  {'Value':>10s} {'Mean LGD':>10s} {'Change':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for _, row in sensitivity.iterrows():
        print(f"  {row['shock_value']:>10.2f} {row['mean_lgd']:>10.4f} {row['lgd_change']:>+10.4f}")

    return sensitivity