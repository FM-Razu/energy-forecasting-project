"""Feature engineering module for creating derived features from solar PV data."""

import os
import glob
import pandas as pd
import numpy as np


def load_cleaned_data(data_dir: str = "data") -> pd.DataFrame:
    """Load the cleaned CSV file."""
    csv_files = glob.glob(os.path.join(data_dir, "**", "*cleaned*.csv"), recursive=True)
    if not csv_files:
        # Fall back to first raw file if no cleaned data exists
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.read_csv(csv_files[0])


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from solar PV power generation data.

    Creates 14 new features across 3 categories:
    - Domain-specific: temporal features that capture solar production patterns
    - Statistical: rolling statistics to capture trends and volatility
    - Interactions: combinations of features that together predict better than alone

    Args:
        df: Input DataFrame with 'LocalTime' and 'Power(MW)' columns.

    Returns:
        DataFrame with original columns plus engineered features.
    """
    df = df.copy()

    # Parse datetime
    df["LocalTime"] = pd.to_datetime(df["LocalTime"], format="%m/%d/%y %H:%M")

    # ===========================================================
    # DOMAIN-SPECIFIC FEATURES
    # Solar power production is driven by time of day and season.
    # These capture the physical reality of solar generation.
    # ===========================================================

    # Hour of day (0-23) - solar panels only produce when sun is up (~6am-8pm)
    # Higher impact during midday peak sun hours
    df["hour"] = df["LocalTime"].dt.hour

    # Month (1-12) - seasonal sun angle changes affect panel efficiency
    # Winter months have shorter days and lower sun angle
    df["month"] = df["LocalTime"].dt.month

    # Day of year (1-366) - captures annual seasonal trend in solar irradiance
    # Used for modeling smooth seasonal variation
    df["day_of_year"] = df["LocalTime"].dt.dayofyear

    # Hour sine/cos encoding - captures cyclical nature of daily patterns
    # Without this, model treats 23:00 and 00:00 as maximally different
    # when they're actually consecutive
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Month sine/cos encoding - captures cyclical seasonal patterns
    # December and January are consecutive, not opposite ends of a spectrum
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Weekend flag - electricity demand and solar self-consumption patterns
    # differ on weekends vs weekdays in residential/commercial settings
    df["is_weekend"] = df["LocalTime"].dt.dayofweek.isin([5, 6]).astype(int)

    # ===========================================================
    # STATISTICAL FEATURES
    # Rolling statistics capture recent trends and volatility in power output.
    # ===========================================================

    # Rolling mean (12 periods = 1 hour at 5-min intervals)
    # Smooths short-term fluctuations to capture underlying production level
    df["rolling_power_mean_1h"] = df["Power(MW)"].rolling(window=12, min_periods=1).mean()

    # Rolling std (12 periods = 1 hour) - measures output volatility
    # High std may indicate passing clouds or equipment variability
    df["rolling_power_std_1h"] = df["Power(MW)"].rolling(window=12, min_periods=1).std()

    # Lag feature (6 periods = 30 min) - captures autocorrelation
    # Previous output helps predict current output due to weather persistence
    df["power_lag_30min"] = df["Power(MW)"].shift(6)

    # ===========================================================
    # INTERACTION FEATURES
    # Product/ratio combinations that together matter more than individually.
    # ===========================================================

    # Weekend x Hour interaction - weekend load patterns differ from weekday
    # e.g., residential solar self-consumption higher on weekends when
    # occupants are home during day
    df["weekend_hour_interaction"] = df["is_weekend"] * df["hour"]

    # Seasonal hour interaction - morning ramp-up is steeper in summer
    # Captures that solar production curve shape varies by season
    df["month_hour_interaction"] = df["month"] * df["hour"]

    # Night flag x rolling_std - identifies unstable night-time readings
    # (inverter artifacts when power should be zero)
    # Also flags unstable daytime readings (cloudy conditions)
    df["night_volatility_flag"] = (df["hour"] < 6).astype(int) * df["rolling_power_std_1h"]

    return df


def select_features(df: pd.DataFrame, variance_threshold: float = 0.01,
                    target_col: str = "Power(MW)") -> tuple:
    """Select features by removing highly correlated and low-variance features.

    Step 1 - Correlation filter: drop features with >0.95 correlation to another
             feature (keeps first in pair, drops the later one).
    Step 2 - Variance filter: drop features with normalized variance below threshold.

    Args:
        df: DataFrame with engineered features (including LocalTime and target_col).
        variance_threshold: Fraction below which normalized variance triggers removal.
        target_col: Name of target column to exclude from variance filtering.

    Returns:
        Tuple of (selected_feature_names, reduced_dataframe).
    """
    # Numeric columns only, exclude target from feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]
    dropped = {"correlation": [], "variance": []}

    # ---- Step 1: Correlation filter ----
    corr_matrix = df[feature_cols].corr().abs()

    # Upper triangle of correlation matrix (avoid diagonal and duplicates)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation > 0.95 to any other feature
    high_corr_features = []
    for col in upper.columns:
        if any(upper[col] > 0.95):
            high_corr_features.append(col)

    # Drop duplicates: keep first feature in highly correlated pairs
    to_drop_corr = set()
    for col in high_corr_features:
        # Find which other features it's highly correlated with
        highly_correlated_with = upper[col][upper[col] > 0.95].index.tolist()
        for other in highly_correlated_with:
            # Drop the one that appears later in the column list
            if numeric_cols.index(col) > numeric_cols.index(other):
                to_drop_corr.add(col)
                break
            else:
                to_drop_corr.add(other)

    dropped["correlation"] = sorted(to_drop_corr)

    # ---- Step 2: Variance filter ----
    # Normalize features to [0,1] range before comparing variance
    # This makes the threshold meaningful across features with different scales
    feature_df = df[feature_cols].copy()
    feature_min = feature_df.min()
    feature_max = feature_df.max()
    range_vals = feature_max - feature_min
    # Avoid division by zero for constant features
    range_vals = range_vals.replace(0, 1)
    normalized = (feature_df - feature_min) / range_vals

    variances = normalized.var()
    overall_variance = variances.mean()
    threshold = overall_variance * variance_threshold

    to_drop_var = variances[variances < threshold].index.tolist()
    dropped["variance"] = sorted(to_drop_var)

    # ---- Combine and report ----
    all_dropped = dropped["correlation"] + dropped["variance"]
    selected_cols = [c for c in feature_cols if c not in all_dropped]
    # Always include target column in output
    selected_cols = [target_col] + selected_cols

    # ---- Logging ----
    print("\n" + "-" * 40)
    print("FEATURE SELECTION RESULTS")
    print("-" * 40)

    if dropped["correlation"]:
        print(f"\nDropped ({len(dropped['correlation'])}) due to high correlation (>0.95):")
        for f in dropped["correlation"]:
            correlated_with = [
                c for c in feature_cols
                if c != f and abs(df[f].corr(df[c])) > 0.95
            ]
            print(f"  - {f}  (correlated with: {correlated_with})")
    else:
        print("\nNo features dropped for high correlation")

    if dropped["variance"]:
        print(f"\nDropped ({len(dropped['variance'])}) due to low normalized variance (< {variance_threshold:.0%} of mean normalized variance):")
        for f in dropped["variance"]:
            print(f"  - {f}  (normalized variance: {variances[f]:.6f}, threshold: {threshold:.6f})")
    else:
        print("\nNo features dropped for low variance")

    print(f"\nSelected features ({len(selected_cols)}): {selected_cols}")

    return selected_cols, df[selected_cols]


def main():
    """Load cleaned data, run feature engineering, and save result."""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # Load cleaned data
    try:
        df = load_cleaned_data("data")
        print(f"\nLoaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Engineer features
    print("\nEngineering features...")
    df_features = create_features(df)

    original_cols = ["LocalTime", "Power(MW)"]
    new_cols = [c for c in df_features.columns if c not in original_cols]

    print(f"\nOriginal columns: {len(original_cols)}")
    print(f"New features: {len(new_cols)}")
    print(f"New feature names: {new_cols}")

    # Print feature categories summary
    print("\n" + "-" * 40)
    print("FEATURE CATEGORIES:")
    print("-" * 40)
    print("Domain features (8): hour, month, day_of_year, hour_sin, hour_cos,")
    print("                      month_sin, month_cos, is_weekend")
    print("Statistical (4): rolling_power_mean_1h, rolling_power_std_1h,")
    print("                  power_lag_30min")
    print("Interactions (3): weekend_hour_interaction, month_hour_interaction,")
    print("                   night_volatility_flag")

    # Save engineered features
    save_path = "data/features.csv"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df_features.to_csv(save_path, index=False)
    print(f"\nSaved feature-engineered data to: {save_path}")

    # Feature selection
    selected_features, df_selected = select_features(df_features)

    # Save selected features
    save_path_selected = "data/features_selected.csv"
    df_features[selected_features].to_csv(save_path_selected, index=False)
    print(f"Saved selected features to: {save_path_selected}")

    # Print sample of engineered data
    print("\n" + "-" * 40)
    print("SAMPLE OF ENGINEERED DATA (first 5 rows):")
    print("-" * 40)
    print(df_features[["LocalTime", "Power(MW)", "hour", "month", "is_weekend",
                       "rolling_power_mean_1h", "weekend_hour_interaction"]].head())

    print("\n" + "=" * 60)
    print(f"Final shape: {df_features.shape[0]} rows, {df_features.shape[1]} columns")
    print(f"Selected features: {len(selected_features)} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
