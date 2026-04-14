"""Data loading module for loading and summarizing CSV datasets."""

import os
import glob
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(file_path)


def print_shape(df: pd.DataFrame) -> None:
    """Print the shape of the DataFrame (rows, columns)."""
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")


def print_columns_info(df: pd.DataFrame) -> None:
    """Print column names and their data types."""
    print("\nColumn Names and Data Types:")
    print("-" * 40)
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for numeric columns."""
    print("\nSummary Statistics (Numeric Columns):")
    print("-" * 40)
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("  No numeric columns found.")
        return

    stats = numeric_df.describe()
    for col in numeric_df.columns:
        print(f"\n  {col}:")
        print(f"    Mean: {stats.loc['mean', col]:.4f}")
        print(f"    Std:  {stats.loc['std', col]:.4f}")
        print(f"    Min:  {stats.loc['min', col]:.4f}")
        print(f"    Max:  {stats.loc['max', col]:.4f}")


def print_missing_values(df: pd.DataFrame) -> None:
    """Print missing value counts and percentages for each column."""
    print("\nMissing Value Counts:")
    print("-" * 40)
    total_rows = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total_rows) * 100

    for col in df.columns:
        count = missing[col]
        pct = missing_pct[col]
        print(f"  {col}: {count} ({pct:.2f}%)")


def find_csv_file(data_dir: str = "data") -> str:
    """Find the first CSV file in the data directory.

    Args:
        data_dir: Path to the data directory.

    Returns:
        Path to the first CSV file found.

    Raises:
        FileNotFoundError: If no CSV file is found.
    """
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files[0]


def main():
    """Main function to run all data loading and summary operations."""
    # Find and load the CSV file
    try:
        csv_path = find_csv_file("data")
        print(f"Loading: {csv_path}\n")
        df = load_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print all summaries
    print_shape(df)
    print_columns_info(df)
    print_summary_statistics(df)
    print_missing_values(df)


if __name__ == "__main__":
    main()
