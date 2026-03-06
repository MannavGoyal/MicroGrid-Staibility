"""Example usage of the Data Validator module.

Run this from the backend directory:
    python -m examples.test_validator

Or run tests instead:
    python -m pytest tests/unit/test_validator.py -v
"""

import pandas as pd
import numpy as np

# Direct import to avoid relative import issues
import sys
sys.path.insert(0, 'src')

try:
    from data.validator import DataValidator
except ImportError:
    print("Note: Run this script from the backend directory:")
    print("  python -m examples.test_validator")
    print("\nOr run the unit tests instead:")
    print("  python -m pytest tests/unit/test_validator.py -v")
    sys.exit(1)


def example_clean_data():
    """Example with clean, valid data."""
    print("=" * 60)
    print("Example 1: Clean Data")
    print("=" * 60)
    
    validator = DataValidator(pv_capacity_kw=10.0)
    
    # Create clean data
    df = pd.DataFrame({
        'pv_power': [0.5, 1.0, 1.5, 2.0, 1.8, 1.2, 0.8],
        'irradiance': [0.1, 0.2, 0.3, 0.4, 0.35, 0.25, 0.15],
        'temperature': [25.0, 26.0, 27.0, 28.0, 27.5, 26.5, 25.5]
    })
    
    report = validator.validate_timeseries(df)
    
    print(f"\nValidation Result: {report}")
    print(f"Valid: {report.valid}")
    print(f"Critical Errors: {report.critical_errors}")
    print(f"Warnings: {report.warnings}")
    print(f"Info: {report.info}")
    print()


def example_missing_values():
    """Example with missing values."""
    print("=" * 60)
    print("Example 2: Data with Missing Values")
    print("=" * 60)
    
    validator = DataValidator()
    
    # Create data with missing values
    df = pd.DataFrame({
        'pv_power': [1.0, np.nan, 1.5, np.nan, 1.8],
        'irradiance': [0.2, 0.3, np.nan, 0.4, 0.35],
        'temperature': [25.0, 26.0, 27.0, 28.0, 27.5]
    })
    
    report = validator.validate_timeseries(df)
    
    print(f"\nValidation Result: {report}")
    print(f"Valid: {report.valid}")
    print(f"\nMissing Value Report:")
    print(f"  Total missing: {report.missing_value_report.missing_count}")
    print(f"  Percentage: {report.missing_value_report.missing_percentage:.2f}%")
    print(f"  Columns with missing: {report.missing_value_report.columns_with_missing}")
    print(f"  Max gap size: {report.missing_value_report.max_gap_size}")
    print()


def example_outliers():
    """Example with outliers."""
    print("=" * 60)
    print("Example 3: Data with Outliers")
    print("=" * 60)
    
    validator = DataValidator()
    
    # Create data with outliers
    normal_values = [10.0 + i * 0.1 for i in range(50)]
    df = pd.DataFrame({
        'pv_power': normal_values + [1000.0],  # 1000 is an outlier
        'irradiance': [0.2] * 50 + [0.25]
    })
    
    report = validator.validate_timeseries(df)
    
    print(f"\nValidation Result: {report}")
    print(f"Valid: {report.valid}")
    print(f"\nOutlier Report:")
    print(f"  Total outliers: {report.outlier_report.outlier_count}")
    print(f"  Percentage: {report.outlier_report.outlier_percentage:.2f}%")
    print(f"  Outliers by column: {list(report.outlier_report.outliers_by_column.keys())}")
    print()


def example_physical_constraints():
    """Example with physical constraint violations."""
    print("=" * 60)
    print("Example 4: Physical Constraint Violations")
    print("=" * 60)
    
    validator = DataValidator(pv_capacity_kw=10.0)
    
    # Create data with constraint violations
    df = pd.DataFrame({
        'pv_power': [1.0, 2.0, -0.5, 12.0, 5.0],  # Negative and exceeds capacity
        'irradiance': [0.5, 0.8, 1.5, 1.0, 0.9],  # Exceeds 1.2 kW/m²
        'temperature': [25.0, 26.0, 70.0, 28.0, -60.0]  # Exceeds max and below min
    })
    
    report = validator.validate_timeseries(df)
    
    print(f"\nValidation Result: {report}")
    print(f"Valid: {report.valid}")
    print(f"\nConstraint Report:")
    print(f"  Violations:")
    for violation in report.constraint_report.violations:
        print(f"    - {violation}")
    print()


def example_comprehensive():
    """Example with multiple issues."""
    print("=" * 60)
    print("Example 5: Comprehensive Validation (Multiple Issues)")
    print("=" * 60)
    
    validator = DataValidator(pv_capacity_kw=10.0)
    
    # Create data with multiple issues
    df = pd.DataFrame({
        'pv_power': [1.0, 2.0, np.nan, -0.5, 5.0, 12.0],
        'irradiance': [0.5, 0.8, 1.5, 1.0, 0.9, np.nan],
        'temperature': [25.0, 26.0, 27.0, 28.0, 70.0, 27.5]
    })
    
    report = validator.validate_timeseries(df)
    
    print(f"\nValidation Result: {report}")
    print(f"Valid: {report.valid}")
    print(f"\nCritical Errors: {len(report.critical_errors)}")
    for error in report.critical_errors:
        print(f"  - {error}")
    print(f"\nWarnings: {len(report.warnings)}")
    for warning in report.warnings:
        print(f"  - {warning}")
    print(f"\nInfo: {len(report.info)}")
    for info in report.info:
        print(f"  - {info}")
    print()


if __name__ == "__main__":
    example_clean_data()
    example_missing_values()
    example_outliers()
    example_physical_constraints()
    example_comprehensive()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
