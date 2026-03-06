"""Unit tests for Data Validator module."""

import pytest
import pandas as pd
import numpy as np
from backend.src.data.validator import (
    DataValidator,
    ValidationReport,
    MissingValueReport,
    OutlierReport,
    ConstraintReport
)


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        validator = DataValidator()
        df = pd.DataFrame()
        
        report = validator.validate_timeseries(df)
        
        assert not report.valid
        assert len(report.critical_errors) > 0
        assert "empty" in report.critical_errors[0].lower()
    
    def test_validate_dataframe_no_columns(self):
        """Test validation of DataFrame with no columns."""
        validator = DataValidator()
        df = pd.DataFrame(index=range(10))
        
        report = validator.validate_timeseries(df)
        
        assert not report.valid
        # DataFrame with index but no columns is considered empty by pandas
        assert any("empty" in err.lower() or "no columns" in err.lower() 
                  for err in report.critical_errors)
    
    def test_validate_clean_data(self):
        """Test validation of clean data with no issues."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'pv_power': [0.5, 1.0, 1.5, 2.0, 1.8],
            'irradiance': [0.1, 0.2, 0.3, 0.4, 0.35],
            'temperature': [25.0, 26.0, 27.0, 28.0, 27.5]
        })
        
        report = validator.validate_timeseries(df)
        
        assert report.valid
        assert len(report.critical_errors) == 0
        assert report.missing_value_report.missing_count == 0
        assert report.outlier_report.outlier_count == 0
    
    def test_check_missing_values_none(self):
        """Test missing value check with no missing values."""
        validator = DataValidator()
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        report = validator.check_missing_values(df)
        
        assert report.missing_count == 0
        assert report.missing_percentage == 0.0
        assert len(report.columns_with_missing) == 0
        assert report.max_gap_size == 0
    
    def test_check_missing_values_some_missing(self):
        """Test missing value check with some missing values."""
        validator = DataValidator()
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan, 5],
            'col2': [10, 20, np.nan, 40, 50]
        })
        
        report = validator.check_missing_values(df)
        
        assert report.missing_count == 3
        assert report.missing_percentage == 30.0  # 3 out of 10 values
        assert 'col1' in report.columns_with_missing
        assert 'col2' in report.columns_with_missing
        assert report.columns_with_missing['col1'] == 2
        assert report.columns_with_missing['col2'] == 1
    
    def test_check_missing_values_gap_size(self):
        """Test missing value gap size calculation."""
        validator = DataValidator()
        df = pd.DataFrame({
            'col1': [1, np.nan, np.nan, np.nan, 5, 6, np.nan, 8]
        })
        
        report = validator.check_missing_values(df)
        
        assert report.max_gap_size == 3  # Three consecutive NaN values
    
    def test_check_outliers_none(self):
        """Test outlier detection with no outliers."""
        validator = DataValidator()
        # Normal distribution data
        np.random.seed(42)
        df = pd.DataFrame({
            'col1': np.random.normal(100, 10, 100)
        })
        
        report = validator.check_outliers(df)
        
        # With 100 samples from normal distribution, should have very few outliers
        assert report.outlier_percentage < 5.0
    
    def test_check_outliers_with_outliers(self):
        """Test outlier detection with clear outliers."""
        validator = DataValidator()
        # Create data with clear outliers: many normal values + one extreme outlier
        # With 50 normal values around 10, the std will be small, making 1000 a clear outlier
        normal_values = [10.0 + i * 0.1 for i in range(50)]  # 10.0 to 14.9
        df = pd.DataFrame({
            'col1': normal_values + [1000.0]  # 1000 is a clear outlier
        })
        
        report = validator.check_outliers(df)
        
        assert report.outlier_count > 0
        assert 'col1' in report.outliers_by_column
    
    def test_check_outliers_constant_values(self):
        """Test outlier detection with constant values (std=0)."""
        validator = DataValidator()
        df = pd.DataFrame({
            'col1': [5, 5, 5, 5, 5]
        })
        
        report = validator.check_outliers(df)
        
        # No outliers when all values are the same
        assert report.outlier_count == 0
    
    def test_check_outliers_skips_non_numeric(self):
        """Test that outlier detection skips non-numeric columns."""
        validator = DataValidator()
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e']
        })
        
        report = validator.check_outliers(df)
        
        # Should only check numeric column
        assert 'text' not in report.outliers_by_column
    
    def test_check_physical_constraints_pv_power_negative(self):
        """Test detection of negative PV power values."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'pv_power': [1.0, 2.0, -0.5, 3.0]
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('negative' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_pv_power_exceeds_capacity(self):
        """Test detection of PV power exceeding capacity."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'pv_power': [5.0, 8.0, 12.0, 7.0]  # 12.0 exceeds capacity
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('exceed' in v.lower() and 'capacity' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_pv_power_valid(self):
        """Test valid PV power values."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'pv_power': [0.0, 5.0, 8.0, 10.0]
        })
        
        report = validator.check_physical_constraints(df)
        
        # Should have no violations for PV power
        pv_violations = [v for v in report.violations if 'pv' in v.lower()]
        assert len(pv_violations) == 0
    
    def test_check_physical_constraints_irradiance_negative(self):
        """Test detection of negative irradiance values."""
        validator = DataValidator()
        df = pd.DataFrame({
            'irradiance': [0.5, 0.8, -0.1, 1.0]
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('negative' in v.lower() and 'irradiance' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_irradiance_exceeds_max(self):
        """Test detection of irradiance exceeding physical maximum."""
        validator = DataValidator()
        df = pd.DataFrame({
            'irradiance': [0.5, 0.8, 1.5, 1.0]  # 1.5 exceeds 1.2 kW/m²
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('exceed' in v.lower() and 'irradiance' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_irradiance_valid(self):
        """Test valid irradiance values."""
        validator = DataValidator()
        df = pd.DataFrame({
            'irradiance': [0.0, 0.5, 1.0, 1.2]
        })
        
        report = validator.check_physical_constraints(df)
        
        # Should have no violations for irradiance
        irr_violations = [v for v in report.violations if 'irradiance' in v.lower()]
        assert len(irr_violations) == 0
    
    def test_check_physical_constraints_temperature_below_min(self):
        """Test detection of temperature below minimum."""
        validator = DataValidator()
        df = pd.DataFrame({
            'temperature': [20.0, 25.0, -60.0, 30.0]  # -60 below -50°C
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('below' in v.lower() and 'temperature' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_temperature_above_max(self):
        """Test detection of temperature above maximum."""
        validator = DataValidator()
        df = pd.DataFrame({
            'temperature': [20.0, 25.0, 70.0, 30.0]  # 70 above 60°C
        })
        
        report = validator.check_physical_constraints(df)
        
        assert report.has_violations
        assert any('above' in v.lower() and 'temperature' in v.lower() for v in report.violations)
    
    def test_check_physical_constraints_temperature_valid(self):
        """Test valid temperature values."""
        validator = DataValidator()
        df = pd.DataFrame({
            'temperature': [-50.0, 0.0, 25.0, 60.0]
        })
        
        report = validator.check_physical_constraints(df)
        
        # Should have no violations for temperature
        temp_violations = [v for v in report.violations if 'temperature' in v.lower()]
        assert len(temp_violations) == 0
    
    def test_validate_timeseries_critical_missing_threshold(self):
        """Test that exceeding missing value threshold marks report as invalid."""
        validator = DataValidator()
        # Create DataFrame with >10% missing values
        data = [1.0] * 80 + [np.nan] * 20
        df = pd.DataFrame({'col1': data})
        
        report = validator.validate_timeseries(df)
        
        assert not report.valid
        assert any('missing' in err.lower() and 'threshold' in err.lower() 
                  for err in report.critical_errors)
    
    def test_validate_timeseries_warnings_for_minor_issues(self):
        """Test that minor issues generate warnings but don't invalidate data."""
        validator = DataValidator()
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],  # 5% missing
            'col2': [10, 20, 30, 40, 1000]  # Has outlier
        })
        
        report = validator.validate_timeseries(df)
        
        assert report.valid  # Should still be valid
        assert len(report.warnings) > 0  # But should have warnings
    
    def test_validate_timeseries_comprehensive(self):
        """Test comprehensive validation with multiple issue types."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'pv_power': [1.0, 2.0, np.nan, -0.5, 5.0],  # Missing + negative
            'irradiance': [0.5, 0.8, 1.5, 1.0, 0.9],  # Exceeds max
            'temperature': [25.0, 26.0, 27.0, 28.0, 70.0]  # Exceeds max
        })
        
        report = validator.validate_timeseries(df)
        
        assert not report.valid
        assert len(report.critical_errors) > 0
        assert report.missing_value_report is not None
        assert report.outlier_report is not None
        assert report.constraint_report is not None
    
    def test_column_name_case_insensitive(self):
        """Test that column name matching is case-insensitive."""
        validator = DataValidator(pv_capacity_kw=10.0)
        df = pd.DataFrame({
            'PV_Power': [1.0, 2.0, 3.0],
            'Irradiance': [0.5, 0.6, 0.7],
            'Temperature': [25.0, 26.0, 27.0]
        })
        
        report = validator.check_physical_constraints(df)
        
        # Should recognize columns despite different case
        assert not report.has_violations
    
    def test_validator_without_pv_capacity(self):
        """Test validator works without PV capacity specified."""
        validator = DataValidator()  # No pv_capacity_kw
        df = pd.DataFrame({
            'pv_power': [1.0, 2.0, 15.0, 3.0]  # Would exceed 10kW but no capacity set
        })
        
        report = validator.check_physical_constraints(df)
        
        # Should only check for negative values, not capacity
        capacity_violations = [v for v in report.violations if 'capacity' in v.lower()]
        assert len(capacity_violations) == 0


class TestReportDataClasses:
    """Test suite for report data classes."""
    
    def test_missing_value_report_repr(self):
        """Test MissingValueReport string representation."""
        report = MissingValueReport(
            total_values=100,
            missing_count=5,
            missing_percentage=5.0,
            max_gap_size=2
        )
        
        repr_str = repr(report)
        assert '5/100' in repr_str
        assert '5.00%' in repr_str
        assert 'max_gap=2' in repr_str
    
    def test_outlier_report_repr(self):
        """Test OutlierReport string representation."""
        report = OutlierReport(
            total_values=100,
            outlier_count=3,
            outlier_percentage=3.0
        )
        
        repr_str = repr(report)
        assert '3/100' in repr_str
        assert '3.00%' in repr_str
    
    def test_constraint_report_properties(self):
        """Test ConstraintReport properties."""
        report = ConstraintReport()
        assert not report.has_violations
        assert not report.has_warnings
        
        report.violations.append("Test violation")
        assert report.has_violations
        
        report.warnings.append("Test warning")
        assert report.has_warnings
    
    def test_validation_report_repr(self):
        """Test ValidationReport string representation."""
        report = ValidationReport(
            valid=False,
            critical_errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        repr_str = repr(report)
        assert 'INVALID' in repr_str
        assert 'errors=2' in repr_str
        assert 'warnings=1' in repr_str
