"""Parser module for configuration and data parsing."""

import json
import pandas as pd
from pathlib import Path
from typing import Union
from pydantic import ValidationError

from ..config.schemas import Configuration


class ConfigurationError(Exception):
    """Exception raised for configuration parsing errors."""
    pass


class DataFormatError(Exception):
    """Exception raised for data format parsing errors."""
    pass


class ValidationResult:
    """Result of configuration validation."""
    
    def __init__(self, valid: bool, errors: list = None):
        self.valid = valid
        self.errors = errors or []
    
    def __repr__(self):
        if self.valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={self.errors})"


class Parser:
    """Parser for configuration files and time-series data."""
    
    def parse_config(self, config_path: str) -> Configuration:
        """
        Parse JSON configuration file into Configuration object.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configuration object
            
        Raises:
            ConfigurationError: If file cannot be parsed or validation fails
        """
        try:
            path = Path(config_path)
            
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            # Try different encodings
            config_dict = None
            last_error = None
            
            for encoding in ['utf-8-sig', 'utf-8', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        config_dict = json.load(f)
                    break
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    last_error = e
                    continue
            
            if config_dict is None:
                if isinstance(last_error, json.JSONDecodeError):
                    raise ConfigurationError(
                        f"Invalid JSON in configuration file at line {last_error.lineno}, "
                        f"column {last_error.colno}: {last_error.msg}"
                    )
                else:
                    raise ConfigurationError(
                        f"Unable to decode configuration file with supported encodings: {config_path}"
                    )
            
            # Validate and parse using Pydantic
            try:
                config = Configuration(**config_dict)
                return config
            except ValidationError as e:
                # Format validation errors descriptively
                error_messages = []
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    msg = error['msg']
                    error_messages.append(f"Field '{field}': {msg}")
                
                raise ConfigurationError(
                    f"Configuration validation failed:\n" + "\n".join(error_messages)
                )
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file at line {e.lineno}, column {e.colno}: {e.msg}"
            )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error parsing configuration: {str(e)}")
    
    def parse_timeseries_data(
        self, 
        data_path: str, 
        format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Parse time-series data from CSV or other formats.
        
        Args:
            data_path: Path to data file
            format: Data format ('csv', 'excel', 'parquet')
            
        Returns:
            DataFrame with parsed time-series data
            
        Raises:
            DataFormatError: If file cannot be parsed
        """
        try:
            path = Path(data_path)
            
            if not path.exists():
                raise DataFormatError(f"Data file not found: {data_path}")
            
            # Parse based on format
            if format.lower() == 'csv':
                # Try different encodings and delimiters
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']:
                    for sep in [',', ';', '\t']:
                        try:
                            df = pd.read_csv(
                                path, 
                                encoding=encoding,
                                sep=sep,
                                parse_dates=True
                            )
                            # Check if we got meaningful data (more than just one column)
                            if len(df.columns) > 1:
                                return df
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                
                raise DataFormatError(
                    f"Unable to parse CSV file with supported encodings and delimiters: {data_path}"
                )
                
            elif format.lower() == 'excel':
                try:
                    df = pd.read_excel(path)
                    return df
                except Exception as e:
                    raise DataFormatError(f"Error parsing Excel file: {str(e)}")
                    
            elif format.lower() == 'parquet':
                try:
                    df = pd.read_parquet(path)
                    return df
                except Exception as e:
                    raise DataFormatError(f"Error parsing Parquet file: {str(e)}")
            else:
                raise DataFormatError(f"Unsupported format: {format}")
                
        except Exception as e:
            if isinstance(e, DataFormatError):
                raise
            raise DataFormatError(f"Error parsing time-series data: {str(e)}")
    
    def validate_config(self, config: Configuration) -> ValidationResult:
        """
        Validate configuration object against schema.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            ValidationResult indicating if configuration is valid
        """
        errors = []
        
        try:
            # Pydantic already validates on construction, but we can add
            # additional business logic validation here
            
            # Check if initial SOC is within battery capacity
            if config.microgrid_configuration.initial_soc_kwh is not None:
                if config.microgrid_configuration.initial_soc_kwh > config.microgrid_configuration.battery_capacity_kwh:
                    errors.append(
                        "initial_soc_kwh cannot exceed battery_capacity_kwh"
                    )
                if config.microgrid_configuration.initial_soc_kwh < 0:
                    errors.append(
                        "initial_soc_kwh cannot be negative"
                    )
            
            # Check diesel generator configuration
            if config.microgrid_configuration.has_diesel_generator:
                if config.microgrid_configuration.diesel_capacity_kw is None:
                    errors.append(
                        "diesel_capacity_kw must be specified when has_diesel_generator is True"
                    )
                elif config.microgrid_configuration.diesel_capacity_kw <= 0:
                    errors.append(
                        "diesel_capacity_kw must be positive"
                    )
            
            # Check battery power rating vs capacity
            if config.microgrid_configuration.battery_power_kw > config.microgrid_configuration.battery_capacity_kwh * 2:
                errors.append(
                    "battery_power_kw seems unrealistic (> 2C rate). "
                    "Consider reducing power rating or increasing capacity."
                )
            
            if errors:
                return ValidationResult(valid=False, errors=errors)
            
            return ValidationResult(valid=True)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(valid=False, errors=errors)
