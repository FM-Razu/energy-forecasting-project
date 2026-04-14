"""Data quality validation module."""

import os
import glob
import pandas as pd
import numpy as np


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def find_csv_file(data_dir: str = "data") -> str:
    """Find the first CSV file in the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files[0]


def check_schema(df: pd.DataFrame, required_columns: list = None) -> dict:
    """Check 1: Schema validation - required columns exist and have correct dtypes.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names. If None, checks that df has at least 1 column.

    Returns:
        Dict with 'passed', 'errors', and 'warnings' keys.
    """
    result = {"passed": True, "errors": [], "warnings": []}

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            result["passed"] = False
            result["errors"].append(f"Missing required columns: {missing}")
    elif len(df.columns) == 0:
        result["passed"] = False
        result["errors"].append("DataFrame has no columns")

    # Check for object dtype columns that might need attention
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if len(object_cols) > len(df.columns) * 0.8:
        result["warnings"].append(
            f"High proportion of object columns ({len(object_cols)}/{len(df.columns)})"
        )

    return result


def check_row_count(df: pd.DataFrame, min_rows: int = 100, warn_rows: int = 1000) -> dict:
    """Check 2: Row count validation.

    Args:
        df: DataFrame to validate.
        min_rows: Minimum required rows (critical if below).
        warn_rows: Threshold for warning if below.

    Returns:
        Dict with 'passed', 'errors', and 'warnings' keys.
    """
    result = {"passed": True, "errors": [], "warnings": []}
    row_count = len(df)

    if row_count < min_rows:
        result["passed"] = False
        result["errors"].append(
            f"Row count ({row_count}) is below minimum ({min_rows})"
        )
    elif row_count < warn_rows:
        result["warnings"].append(
            f"Row count ({row_count}) is below recommended threshold ({warn_rows})"
        )

    return result


def check_null_rates(
    df: pd.DataFrame, critical_threshold: float = 0.5, warn_threshold: float = 0.2
) -> dict:
    """Check 3: Null rate validation.

    Args:
        df: DataFrame to validate.
        critical_threshold: Max null rate before critical error.
        warn_threshold: Max null rate before warning.

    Returns:
        Dict with 'passed', 'errors', and 'warnings' keys.
    """
    result = {"passed": True, "errors": [], "warnings": []}
    null_rates = df.isnull().sum() / len(df)

    for col, rate in null_rates.items():
        if rate > critical_threshold:
            result["passed"] = False
            result["errors"].append(
                f"Column '{col}' has {rate:.1%} nulls (>{critical_threshold:.0%})"
            )
        elif rate > warn_threshold:
            result["warnings"].append(
                f"Column '{col}' has {rate:.1%} nulls (>{warn_threshold:.0%})"
            )

    return result


def check_value_ranges(
    df: pd.DataFrame,
    negative_allowed: bool = False,
    max_ratio: float = 100.0,
) -> dict:
    """Check 4: Value ranges - numeric columns within sensible bounds.

    Args:
        df: DataFrame to validate.
        negative_allowed: Whether negative values are allowed in numeric columns.
        max_ratio: Max ratio of max/min for numeric columns (warn if exceeded).

    Returns:
        Dict with 'passed', 'errors', and 'warnings' keys.
    """
    result = {"passed": True, "errors": [], "warnings": []}
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        min_val = col_data.min()
        max_val = col_data.max()

        # Check for negative values
        if not negative_allowed and min_val < 0:
            result["passed"] = False
            result["errors"].append(
                f"Column '{col}' has negative values (min={min_val})"
            )

        # Check for extreme ratios (possible outliers or unit errors)
        if min_val > 0 and max_val / min_val > max_ratio:
            result["warnings"].append(
                f"Column '{col}' has high range ratio (max/min={max_val/min_val:.1f})"
            )

        # Check for extremely large values that might be errors
        if abs(max_val) > 1e10:
            result["warnings"].append(
                f"Column '{col}' has very large values (max={max_val})"
            )

    return result


def check_target_distribution(
    df: pd.DataFrame,
    target_column: str = None,
    min_class_ratio: float = 0.05,
) -> dict:
    """Check 5: Target distribution for classification problems.

    Args:
        df: DataFrame to validate.
        target_column: Name of target column. If None, tries to auto-detect.
        min_class_ratio: Minimum ratio each class should have.

    Returns:
        Dict with 'passed', 'errors', and 'warnings' keys.
    """
    result = {"passed": True, "errors": [], "warnings": []}

    # Auto-detect target column if not specified
    if target_column is None:
        # Look for columns with 'target', 'label', 'class', or 'y' in name
        candidates = [
            col
            for col in df.columns
            if any(k in col.lower() for k in ["target", "label", "class", "y"])
        ]
        if candidates:
            target_column = candidates[0]

    if target_column is None or target_column not in df.columns:
        result["warnings"].append("No target column specified for distribution check")
        return result

    target = df[target_column].dropna()
    if target.dtype in [np.float64, np.float32] and not target.apply(
        lambda x: x.is_integer()
    ).all():
        # Continuous variable, not classification
        result["warnings"].append(
            f"Target column '{target_column}' appears continuous, skipping classification check"
        )
        return result

    class_counts = target.value_counts()
    n_classes = len(class_counts)
    total = len(target)

    if n_classes < 2:
        result["passed"] = False
        result["errors"].append(
            f"Target column has only {n_classes} class (need 2+ for classification)"
        )
        return result

    for cls, count in class_counts.items():
        ratio = count / total
        if ratio < min_class_ratio:
            result["warnings"].append(
                f"Class '{cls}' has only {ratio:.1%} of data (<{min_class_ratio:.0%})"
            )

    return result


def check_data_quality(
    df: pd.DataFrame,
    required_columns: list = None,
    target_column: str = None,
) -> dict:
    """Run all 5 data quality checks and return comprehensive results.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        target_column: Name of target column for classification check.

    Returns:
        Dict with structure:
        {
            'success': bool (all checks passed),
            'failures': list of critical errors,
            'warnings': list of concerns,
            'statistics': dict with counts and metrics
        }
    """
    all_failures = []
    all_warnings = []

    # Run all 5 checks
    schema_result = check_schema(df, required_columns)
    row_result = check_row_count(df)
    null_result = check_null_rates(df)
    range_result = check_value_ranges(df)
    target_result = check_target_distribution(df, target_column)

    # Aggregate results
    for result in [schema_result, row_result, null_result, range_result, target_result]:
        all_failures.extend(result["errors"])
        all_warnings.extend(result["warnings"])

    # Calculate statistics
    null_counts = df.isnull().sum()
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls": int(df.isnull().sum().sum()),
        "nulls_by_column": {col: int(count) for col, count in null_counts.items()},
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return {
        "success": len(all_failures) == 0,
        "failures": all_failures,
        "warnings": all_warnings,
        "statistics": statistics,
    }


def main():
    """Main function to run data quality checks on a CSV file."""
    # Find and load the CSV file
    try:
        csv_path = find_csv_file("data")
        print(f"Loading: {csv_path}\n")
        df = load_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run quality checks
    result = check_data_quality(df)

    # Print results
    print("=" * 60)
    print("DATA QUALITY CHECK RESULTS")
    print("=" * 60)

    print(f"\nOverall Status: {'PASSED' if result['success'] else 'FAILED'}")

    if result["failures"]:
        print(f"\nCRITICAL ISSUES ({len(result['failures'])}):")
        for failure in result["failures"]:
            print(f"  [!] {failure}")

    if result["warnings"]:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for warning in result["warnings"]:
            print(f"  [*] {warning}")

    print("\nSTATISTICS:")
    stats = result["statistics"]
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total columns: {stats['total_columns']}")
    print(f"  Total nulls: {stats['total_nulls']:,}")
    print(f"  Numeric columns: {stats['numeric_columns']}")
    print(f"  Categorical columns: {stats['categorical_columns']}")
    print(f"  Duplicate rows: {stats['duplicate_rows']:,}")

    if stats["nulls_by_column"]:
        print("\n  Nulls by column:")
        for col, count in stats["nulls_by_column"].items():
            pct = count / stats["total_rows"] * 100
            print(f"    {col}: {count:,} ({pct:.2f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
