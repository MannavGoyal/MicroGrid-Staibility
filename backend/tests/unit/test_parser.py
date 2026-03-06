"""Unit tests for Parser module."""

import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path

from backend.src.data.parser import Parser, ConfigurationError, DataFormatError, ValidationResult
from backend.src.config.schemas import Configuration, ForecastHorizon, ModelType, MicrogridMode


@pytest.fixture
def parser():
    """Create a Parser instance."""
    return Parser()


@pytest.fixture
def valid_config_dict():
    """Create a valid configuration dictionary."""
    return {
        "experiment_name": "test_experiment",
        "forecast_horizon": "15min",
        "model_configuration": {
            "model_type": "lstm",
            "hyperparameters": {"hidden_size": 64, "num_layers": 2},
            "sequence_length": 12
        },
        "microgrid_configuration": {
            "mode": "islanded",
            "pv_capacity_kw": 10.0,
            "battery_capacity_kwh": 5.0,
            "battery_power_kw": 3.0,
            "inverter_capacity_kw": 12.0,
            "initial_soc_kwh": 2.5,
            "has_diesel_generator": False
        },
        "training_configuration": {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        },
        "data_path": "data/test.csv",
        "output_dir": "results"
    }


class TestParseConfig:
    """Tests for parse_config method."""
    
    def test_parse_valid_config(self, parser, valid_config_dict, tmp_path):
        """Test parsing a valid configuration file."""
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config_dict, f)
        
        config = parser.parse_config(str(config_file))
        
        assert isinstance(config, Configuration)
        assert config.experiment_name == "test_experiment"
        assert config.forecast_horizon == ForecastHorizon.FIFTEEN_MIN
        assert config.model_configuration.model_type == ModelType.LSTM
        assert config.microgrid_configuration.mode == MicrogridMode.ISLANDED
        assert config.microgrid_configuration.pv_capacity_kw == 10.0
    
    def test_parse_config_file_not_found(self, parser):
        """Test parsing non-existent configuration file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            parser.parse_config("nonexistent.json")
    
    def test_parse_invalid_json(self, parser, tmp_path):
        """Test parsing invalid JSON."""
        config_file = tmp_path / "invalid.json"
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            parser.parse_config(str(config_file))
    
    def test_parse_config_missing_required_field(self, parser, valid_config_dict, tmp_path):
        """Test parsing configuration with missing required field."""
        del valid_config_dict["experiment_name"]
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config_dict, f)
        
        with pytest.raises(ConfigurationError, match="validation failed"):
            parser.parse_config(str(config_file))
    
    def test_parse_config_invalid_field_value(self, parser, valid_config_dict, tmp_path):
        """Test parsing configuration with invalid field value."""
        valid_config_dict["microgrid_configuration"]["pv_capacity_kw"] = -5.0
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config_dict, f)
        
        with pytest.raises(ConfigurationError, match="validation failed"):
            parser.parse_config(str(config_file))
    
    def test_parse_config_inverter_less_than_pv(self, parser, valid_config_dict, tmp_path):
        """Test validation error when inverter capacity < PV capacity."""
        valid_config_dict["microgrid_configuration"]["inverter_capacity_kw"] = 8.0
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(valid_config_dict, f)
        
        with pytest.raises(ConfigurationError, match="Inverter capacity must be >= PV capacity"):
            parser.parse_config(str(config_file))
    
    def test_parse_config_utf8_with_bom(self, parser, valid_config_dict, tmp_path):
        """Test parsing UTF-8 file with BOM."""
        config_file = tmp_path / "config_bom.json"
        with open(config_file, 'w', encoding='utf-8-sig') as f:
            json.dump(valid_config_dict, f)
        
        config = parser.parse_config(str(config_file))
        assert isinstance(config, Configuration)
        assert config.experiment_name == "test_experiment"


class TestParseTimeseriesData:
    """Tests for parse_timeseries_data method."""
    
    def test_parse_csv_data(self, parser, tmp_path):
        """Test parsing CSV time-series data."""
        csv_file = tmp_path / "data.csv"
        df_original = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'pv_power': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'irradiance': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'temperature': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        })
        df_original.to_csv(csv_file, index=False)
        
        df = parser.parse_timeseries_data(str(csv_file), format='csv')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'pv_power' in df.columns
        assert 'irradiance' in df.columns
        assert 'temperature' in df.columns
    
    def test_parse_csv_with_semicolon_delimiter(self, parser, tmp_path):
        """Test parsing CSV with semicolon delimiter."""
        csv_file = tmp_path / "data_semicolon.csv"
        with open(csv_file, 'w') as f:
            f.write("timestamp;pv_power;irradiance\n")
            f.write("2024-01-01 00:00:00;0.5;500\n")
            f.write("2024-01-01 00:05:00;0.6;600\n")
        
        df = parser.parse_timeseries_data(str(csv_file), format='csv')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert len(df.columns) == 3
    
    def test_parse_data_file_not_found(self, parser):
        """Test parsing non-existent data file."""
        with pytest.raises(DataFormatError, match="Data file not found"):
            parser.parse_timeseries_data("nonexistent.csv", format='csv')
    
    def test_parse_unsupported_format(self, parser, tmp_path):
        """Test parsing with unsupported format."""
        data_file = tmp_path / "data.xyz"
        data_file.touch()
        
        with pytest.raises(DataFormatError, match="Unsupported format"):
            parser.parse_timeseries_data(str(data_file), format='xyz')
    
    def test_parse_csv_latin1_encoding(self, parser, tmp_path):
        """Test parsing CSV with Latin-1 encoding."""
        csv_file = tmp_path / "data_latin1.csv"
        with open(csv_file, 'w', encoding='latin-1') as f:
            f.write("timestamp,pv_power,temperature\n")
            f.write("2024-01-01 00:00:00,0.5,25\n")
        
        df = parser.parse_timeseries_data(str(csv_file), format='csv')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestValidateConfig:
    """Tests for validate_config method."""
    
    def test_validate_valid_config(self, parser, valid_config_dict):
        """Test validating a valid configuration."""
        config = Configuration(**valid_config_dict)
        result = parser.validate_config(config)
        
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_validate_initial_soc_exceeds_capacity(self, parser, valid_config_dict):
        """Test validation error when initial SOC exceeds capacity."""
        valid_config_dict["microgrid_configuration"]["initial_soc_kwh"] = 10.0
        config = Configuration(**valid_config_dict)
        
        result = parser.validate_config(config)
        
        assert result.valid is False
        assert any("initial_soc_kwh cannot exceed battery_capacity_kwh" in err for err in result.errors)
    
    def test_validate_negative_initial_soc(self, parser, valid_config_dict):
        """Test validation error for negative initial SOC."""
        valid_config_dict["microgrid_configuration"]["initial_soc_kwh"] = -1.0
        config = Configuration(**valid_config_dict)
        
        result = parser.validate_config(config)
        
        assert result.valid is False
        assert any("initial_soc_kwh cannot be negative" in err for err in result.errors)
    
    def test_validate_diesel_generator_without_capacity(self, parser, valid_config_dict):
        """Test validation error when diesel generator enabled without capacity."""
        valid_config_dict["microgrid_configuration"]["has_diesel_generator"] = True
        valid_config_dict["microgrid_configuration"]["diesel_capacity_kw"] = None
        config = Configuration(**valid_config_dict)
        
        result = parser.validate_config(config)
        
        assert result.valid is False
        assert any("diesel_capacity_kw must be specified" in err for err in result.errors)
    
    def test_validate_unrealistic_battery_power(self, parser, valid_config_dict):
        """Test validation warning for unrealistic battery power rating."""
        valid_config_dict["microgrid_configuration"]["battery_power_kw"] = 15.0  # 3C rate
        config = Configuration(**valid_config_dict)
        
        result = parser.validate_config(config)
        
        assert result.valid is False
        assert any("battery_power_kw seems unrealistic" in err for err in result.errors)
    
    def test_validate_config_with_diesel_generator(self, parser, valid_config_dict):
        """Test validating configuration with diesel generator."""
        valid_config_dict["microgrid_configuration"]["has_diesel_generator"] = True
        valid_config_dict["microgrid_configuration"]["diesel_capacity_kw"] = 5.0
        config = Configuration(**valid_config_dict)
        
        result = parser.validate_config(config)
        
        assert result.valid is True
        assert len(result.errors) == 0
