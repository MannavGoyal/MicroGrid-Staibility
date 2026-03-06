"""Data Validator module for checking data quality and physical plausibility."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MissingValueReport:
    """Report on missing values in the dataset."""
    total_values: int
    missing_count: int
    missing_percentage: float
    columns_with_missing: Dict[str, int] = field(default_factory=dict)
    max_gap_size: int = 0
    
    def __repr__(self):
        return (
            f"MissingValueReport(missing={self.missing_count}/{self.total_values} "
            f"({self.missing_percentage:.2f}%), max_gap={self.max_gap_size})"
        )


@dataclass
class OutlierReport:
    """Report on statistical outliers in the dataset."""
    total_values: int
    outlier_count: int
    outlier_percentage: float
    outliers_by_column: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    
    def __repr__(self):
        return (
            f"OutlierReport(outliers={self.outlier_count}/{self.total_values} "
            f"({self.outlier_percentage:.2f}%))"
        )


@dataclass
class ConstraintReport:
    """Report on physical constraint violations."""
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def __repr__(self):
        return (
            f"ConstraintReport(violations={len(self.violations)}, "
            f"warnings={len(self.warnings)})"
        )


@dataclass
class ValidationReport:
    """Complete validation report for time-series data."""
    valid: bool
    critical_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    missing_value_report: Optional[MissingValueReport] = None
    outlier_report: Optional[OutlierReport] = None
    constraint_report: Optional[ConstraintReport] = None
    
    def __repr__(self):
        status = "VALID" if self.valid else "INVALID"
        return (
            f"ValidationReport({status}, "
            f"errors={len(self.critical_errors)}, "
            f"warnings={len(self.warnings)}, "
            f"info={len(self.info)})"
        )


class DataValidator:
    """Validator for time-series data quality and physical plausibility."""
    
    # Physical constraint ranges
    PV_POWER_MIN = 0.0  # kW
    PV_POWER_MAX = None  # Will be set based on array capacity if available
    IRRADIANCE_MIN = 0.0  # kW/m²
    IRRADIANCE_MAX = 1.2  # kW/m²
    TEMPERATURE_MIN = -50.0  # °C
    TEMPERATURE_MAX = 60.0  # °C
    
    # Validation thresholds
    MISSING_VALUE_CRITICAL_THRESHOLD = 0.10  # 10%
    OUTLIER_STD_THRESHOLD = 3.0  # 3 standard deviations
    
    def __init__(self, pv_capacity_kw: Optional[float] = None):
        """
        Initialize DataValidator.
        
        Args:
            pv_capacity_kw: PV array capacity in kW (optional, for PV power validation)
        """
        self.pv_capacity_kw = pv_capacity_kw
    
    def validate_timeseries(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate time-series data quality.
        
        Args:
            df: DataFrame with time-series data
            
        Returns:
            ValidationReport with comprehensive validation results
        """
        report = ValidationReport(valid=True)
        
        # Basic checks
        if df.empty:
            report.valid = False
            report.critical_errors.append("DataFrame is empty")
            return report
        
        if len(df.columns) == 0:
            report.valid = False
            report.critical_errors.append("DataFrame has no columns")
            return report
        
        # Check if DataFrame has only index but no data
        if df.shape[1] == 0:
            report.valid = False
            report.critical_errors.append("DataFrame has no columns")
            return report
        
        report.info.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Check for missing values
        missing_report = self.check_missing_values(df)
        report.missing_value_report = missing_report
        
        if missing_report.missing_percentage > self.MISSING_VALUE_CRITICAL_THRESHOLD * 100:
            report.valid = False
            report.critical_errors.append(
                f"Missing values exceed critical threshold: "
                f"{missing_report.missing_percentage:.2f}% > "
                f"{self.MISSING_VALUE_CRITICAL_THRESHOLD * 100}%"
            )
        elif missing_report.missing_count > 0:
            report.warnings.append(
                f"Dataset contains {missing_report.missing_count} missing values "
                f"({missing_report.missing_percentage:.2f}%)"
            )
        else:
            report.info.append("No missing values detected")
        
        # Check for outliers
        outlier_report = self.check_outliers(df)
        report.outlier_report = outlier_report
        
        if outlier_report.outlier_count > 0:
            report.warnings.append(
                f"Detected {outlier_report.outlier_count} outliers "
                f"({outlier_report.outlier_percentage:.2f}%) "
                f"beyond {self.OUTLIER_STD_THRESHOLD} standard deviations"
            )
        else:
            report.info.append("No statistical outliers detected")
        
        # Check physical constraints
        constraint_report = self.check_physical_constraints(df)
        report.constraint_report = constraint_report
        
        if constraint_report.has_violations:
            report.valid = False
            report.critical_errors.extend(constraint_report.violations)
        
        if constraint_report.has_warnings:
            report.warnings.extend(constraint_report.warnings)
        
        if not constraint_report.has_violations and not constraint_report.has_warnings:
            report.info.append("All physical constraints satisfied")
        
        return report
    
    def check_missing_values(self, df: pd.DataFrame) -> MissingValueReport:
        """
        Check for missing values and gaps in the dataset.
        
        Args:
            df: DataFrame to check
            
        Returns:
            MissingValueReport with missing value statistics
        """
        total_values = df.size
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / total_values * 100) if total_values > 0 else 0.0
        
        # Count missing values per column
        columns_with_missing = {}
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            if col_missing > 0:
                columns_with_missing[col] = int(col_missing)
        
        # Calculate maximum gap size (consecutive missing values)
        max_gap_size = 0
        for col in df.columns:
            if df[col].isnull().any():
                # Find consecutive missing values
                is_null = df[col].isnull()
                gaps = is_null.astype(int).groupby((~is_null).cumsum()).sum()
                if len(gaps) > 0:
                    max_gap_size = max(max_gap_size, int(gaps.max()))
        
        return MissingValueReport(
            total_values=int(total_values),
            missing_count=int(missing_count),
            missing_percentage=float(missing_percentage),
            columns_with_missing=columns_with_missing,
            max_gap_size=int(max_gap_size)
        )
    
    def check_outliers(self, df: pd.DataFrame) -> OutlierReport:
        """
        Detect statistical outliers using 3 standard deviations method.
        
        Args:
            df: DataFrame to check
            
        Returns:
            OutlierReport with outlier statistics
        """
        total_values = 0
        outlier_count = 0
        outliers_by_column = {}
        
        # Check numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip columns with all NaN
            if df[col].isnull().all():
                continue
            
            col_data = df[col].dropna()
            total_values += len(col_data)
            
            if len(col_data) < 3:  # Need at least 3 values for std
                continue
            
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:  # All values are the same
                continue
            
            # Find outliers beyond threshold standard deviations
            lower_bound = mean - self.OUTLIER_STD_THRESHOLD * std
            upper_bound = mean + self.OUTLIER_STD_THRESHOLD * std
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            col_outliers = col_data[outlier_mask]
            
            if len(col_outliers) > 0:
                outlier_count += len(col_outliers)
                # Store index and value of outliers (limit to first 10 per column)
                outliers_by_column[col] = [
                    (str(idx), float(val)) 
                    for idx, val in col_outliers.head(10).items()
                ]
        
        outlier_percentage = (outlier_count / total_values * 100) if total_values > 0 else 0.0
        
        return OutlierReport(
            total_values=int(total_values),
            outlier_count=int(outlier_count),
            outlier_percentage=float(outlier_percentage),
            outliers_by_column=outliers_by_column
        )
    
    def check_physical_constraints(self, df: pd.DataFrame) -> ConstraintReport:
        """
        Validate physical plausibility of values.
        
        Checks:
        - PV power: 0 ≤ pv_power ≤ array_capacity
        - Irradiance: 0 ≤ irradiance ≤ 1.2 kW/m²
        - Temperature: -50°C ≤ temperature ≤ 60°C
        
        Args:
            df: DataFrame to check
            
        Returns:
            ConstraintReport with violations and warnings
        """
        report = ConstraintReport()
        
        # Check PV power constraints
        pv_cols = [col for col in df.columns if 'pv' in col.lower() and 'power' in col.lower()]
        for col in pv_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_data = df[col].dropna()
            
            # Check for negative values
            negative_count = (col_data < self.PV_POWER_MIN).sum()
            if negative_count > 0:
                report.violations.append(
                    f"Column '{col}': {negative_count} negative PV power values detected "
                    f"(min: {col_data.min():.4f} kW)"
                )
            
            # Check against capacity if available
            if self.pv_capacity_kw is not None:
                exceeds_capacity = (col_data > self.pv_capacity_kw).sum()
                if exceeds_capacity > 0:
                    report.violations.append(
                        f"Column '{col}': {exceeds_capacity} values exceed PV capacity "
                        f"({self.pv_capacity_kw} kW), max: {col_data.max():.4f} kW"
                    )
        
        # Check irradiance constraints
        irradiance_cols = [col for col in df.columns if 'irradiance' in col.lower() or 'ghi' in col.lower()]
        for col in irradiance_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_data = df[col].dropna()
            
            # Check for negative values
            negative_count = (col_data < self.IRRADIANCE_MIN).sum()
            if negative_count > 0:
                report.violations.append(
                    f"Column '{col}': {negative_count} negative irradiance values detected "
                    f"(min: {col_data.min():.4f} kW/m²)"
                )
            
            # Check for values exceeding physical maximum
            exceeds_max = (col_data > self.IRRADIANCE_MAX).sum()
            if exceeds_max > 0:
                report.violations.append(
                    f"Column '{col}': {exceeds_max} irradiance values exceed physical maximum "
                    f"({self.IRRADIANCE_MAX} kW/m²), max: {col_data.max():.4f} kW/m²"
                )
        
        # Check temperature constraints
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        for col in temp_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_data = df[col].dropna()
            
            # Check for values below minimum
            below_min = (col_data < self.TEMPERATURE_MIN).sum()
            if below_min > 0:
                report.violations.append(
                    f"Column '{col}': {below_min} temperature values below reasonable minimum "
                    f"({self.TEMPERATURE_MIN}°C), min: {col_data.min():.2f}°C"
                )
            
            # Check for values above maximum
            above_max = (col_data > self.TEMPERATURE_MAX).sum()
            if above_max > 0:
                report.violations.append(
                    f"Column '{col}': {above_max} temperature values above reasonable maximum "
                    f"({self.TEMPERATURE_MAX}°C), max: {col_data.max():.2f}°C"
                )
        
        return report
