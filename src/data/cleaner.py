"""Data cleaning module for preprocessing raw datasets."""

import os
import glob
import pandas as pd
import numpy as np

try:
    from .quality import check_data_quality
except ImportError:
    from quality import check_data_quality


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def find_csv_file(data_dir: str = "data") -> str:
    """Find the first CSV file in the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files[0]


def detect_time_series(df: pd.DataFrame) -> bool:
    """Detect if DataFrame appears to be time series data.

    Looks for datetime-like columns or time-related column names.
    """
    # Check for datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"])
    if len(datetime_cols) > 0:
        return True

    # Check for time-related column names
    time_keywords = ["time", "date", "datetime", "timestamp", "localtime", "utc"]
    for col in df.columns:
        if any(kw in col.lower() for kw in time_keywords):
            return True

    return False


def handle_nulls(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """Handle null values according to cleaning rules.

    Rules:
    - Drop rows where target is null (if target specified)
    - Drop columns with > 50% nulls
    - Forward-fill other columns if time series
    - Drop rows with remaining nulls if not time series

    Args:
        df: Input DataFrame.
        target_column: Name of target column.

    Returns:
        DataFrame with nulls handled.
    """
    df = df.copy()
    initial_rows = len(df)

    # Drop columns with > 50% nulls
    null_rates = df.isnull().sum() / len(df)
    cols_to_drop = null_rates[null_rates > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} columns with >50% nulls: {cols_to_drop}")

    # Drop rows where target is null
    if target_column and target_column in df.columns:
        target_nulls = df[target_column].isnull().sum()
        df = df.dropna(subset=[target_column])
        if target_nulls > 0:
            print(f"  Dropped {target_nulls} rows where target was null")

    # Check if time series
    is_ts = detect_time_series(df)

    if is_ts:
        # Forward-fill remaining nulls
        df = df.ffill()
        # Backfill any leading nulls that couldn't be forward-filled
        df = df.bfill()
        print("  Forward-filled (then back-filled) remaining nulls (time series)")
    else:
        # Drop rows with any remaining nulls
        null_rows_before = df.isnull().any(axis=1).sum()
        df = df.dropna()
        if null_rows_before > 0:
            print(f"  Dropped {null_rows_before} rows with remaining nulls")

    rows_after = len(df)
    print(f"  Rows: {initial_rows} -> {rows_after} ({initial_rows - rows_after} removed)")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact row duplicates, keeping first occurrence.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    initial_rows = len(df)
    df = df.drop_duplicates(keep="first")
    rows_after = len(df)

    if initial_rows != rows_after:
        print(f"  Removed {initial_rows - rows_after} duplicate rows")
        print(f"  Rows: {initial_rows} -> {rows_after}")

    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate dtypes.

    - Numeric columns: float64 or int64
    - Categorical/object columns: string

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with converted dtypes.
    """
    df = df.copy()

    for col in df.columns:
        col_dtype = df[col].dtype

        # Convert numeric columns
        if np.issubdtype(col_dtype, np.number):
            if np.issubdtype(col_dtype, np.floating):
                df[col] = df[col].astype(np.float64)
            elif np.issubdtype(col_dtype, np.integer):
                df[col] = df[col].astype(np.int64)
        # Convert object columns to string
        elif col_dtype == "object":
            df[col] = df[col].astype("string")

    print("  Converted numeric columns to float64/int64")
    print("  Converted object columns to string")

    return df


def clean_data(
    df: pd.DataFrame,
    target_column: str = None,
    save_path: str = "data/cleaned.csv",
    run_quality_check: bool = True,
) -> tuple:
    """Clean data and optionally run quality check.

    Args:
        df: Input DataFrame.
        target_column: Name of target column for null handling.
        save_path: Path to save cleaned CSV.
        run_quality_check: Whether to run quality gate after cleaning.

    Returns:
        Tuple of (cleaned_df, quality_result or None).
    """
    print(f"\nInitial shape: {df.shape}")

    # Step 1: Handle nulls
    print("\n1. Handling nulls...")
    df = handle_nulls(df, target_column)

    # Step 2: Remove duplicates
    print("\n2. Removing duplicates...")
    df = remove_duplicates(df)

    # Step 3: Convert dtypes
    print("\n3. Converting dtypes...")
    df = convert_dtypes(df)

    print(f"\nFinal shape: {df.shape}")

    # Save cleaned data
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved cleaned data to: {save_path}")

    # Run quality check if requested
    quality_result = None
    if run_quality_check:
        print("\n" + "=" * 50)
        print("Running quality gate on cleaned data...")
        print("=" * 50)
        quality_result = check_data_quality(df, target_column=target_column)

    return df, quality_result


def main():
    """Main function to load, clean, and validate data."""
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    # Find and load raw data
    try:
        csv_path = find_csv_file("data")
        # Skip already cleaned files
        if "cleaned" in csv_path.lower():
            alt_files = [f for f in glob.glob(os.path.join("data", "**", "*.csv"), recursive=True)
                        if "cleaned" not in f.lower()]
            if alt_files:
                csv_path = alt_files[0]
        print(f"\nLoading raw data: {csv_path}")
        df = load_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    before_rows = len(df)
    print(f"Before cleaning: {before_rows:,} rows")

    # Clean data
    cleaned_df, quality_result = clean_data(df)

    after_rows = len(cleaned_df)

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Rows before: {before_rows:,}")
    print(f"Rows after:  {after_rows:,}")
    print(f"Removed:     {before_rows - after_rows:,} ({(before_rows - after_rows) / before_rows * 100:.2f}%)")

    # Print quality result
    if quality_result:
        print("\n" + "=" * 60)
        print("QUALITY CHECK RESULT (on cleaned data)")
        print("=" * 60)
        print(f"Status: {'PASSED' if quality_result['success'] else 'FAILED'}")

        if quality_result["failures"]:
            print(f"\nCritical Issues ({len(quality_result['failures'])}):")
            for failure in quality_result["failures"]:
                print(f"  [!] {failure}")

        if quality_result["warnings"]:
            print(f"\nWarnings ({len(quality_result['warnings'])}):")
            for warning in quality_result["warnings"]:
                print(f"  [*] {warning}")

        print("\nStatistics:")
        stats = quality_result["statistics"]
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Total columns: {stats['total_columns']}")
        print(f"  Total nulls: {stats['total_nulls']:,}")
        print(f"  Numeric columns: {stats['numeric_columns']}")
        print(f"  Categorical columns: {stats['categorical_columns']}")
        print(f"  Duplicate rows: {stats['duplicate_rows']:,}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
