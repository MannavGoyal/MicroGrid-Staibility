"""Unit tests for DataPipeline module."""

import pytest
import pandas as pd
import numpy as np
from backend.src.data.pipeline import DataPipeline, DataSplit, Scaler
from backend.src.config.schemas import (
    Configuration,
    ModelConfig,
    MicrogridConfig,
    TrainingConfig,
    ForecastHorizon,
    ModelType,
    MicrogridMode,
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Configuration(
        experiment_name="test_experiment",
        forecast_horizon=ForecastHorizon.FIFTEEN_MIN,
        model_configuration=ModelConfig(
            model_type=ModelType.LSTM,
            sequence_length=12,
            hyperparameters={}
        ),
        microgrid_configuration=MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        ),
        training_configuration=TrainingConfig(),
        data_path="test_data.csv",
        output_dir="test_results"
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'irradiance': np.random.uniform(0, 1.0, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'cloud_cover': np.random.uniform(0, 1, n_samples),
        'humidity': np.random.uniform(0.3, 0.9, n_samples),
        'wind_speed': np.random.uniform(0, 10, n_samples),
        'pv_power': np.random.uniform(0, 8, n_samples)
    })
    
    return df


@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample DataFrame with missing values."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'irradiance': np.random.uniform(0, 1.0, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'pv_power': np.random.uniform(0, 8, n_samples)
    })
    
    # Introduce missing values
    df.loc[10:12, 'irradiance'] = np.nan
    df.loc[50, 'temperature'] = np.nan
    df.loc[0, 'pv_power'] = np.nan
    
    return df


class TestDataPipeline:
    """Test suite for DataPipeline class."""
    
    def test_initialization(self, sample_config):
        """Test DataPipeline initialization."""
        pipeline = DataPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.scaler is None
        assert pipeline.feature_names == []
    
    def test_preprocess_no_missing_values(self, sample_config, sample_dataframe):
        """Test preprocessing with no missing values."""
        pipeline = DataPipeline(sample_config)
        df_clean = pipeline.preprocess(sample_dataframe)
        
        # Should have no missing values
        assert df_clean.isnull().sum().sum() == 0
        
        # Should have same shape
        assert df_clean.shape == sample_dataframe.shape
    
    def test_preprocess_with_missing_values(self, sample_config, sample_dataframe_with_missing):
        """Test preprocessing handles missing values correctly."""
        pipeline = DataPipeline(sample_config)
        
        # Verify we have missing values before
        assert sample_dataframe_with_missing.isnull().sum().sum() > 0
        
        df_clean = pipeline.preprocess(sample_dataframe_with_missing)
        
        # Should have no missing values after preprocessing
        assert df_clean.isnull().sum().sum() == 0
        
        # Should have same shape
        assert df_clean.shape == sample_dataframe_with_missing.shape
    
    def test_normalize_minmax(self, sample_config, sample_dataframe):
        """Test MinMax normalization produces values in [0, 1] range."""
        pipeline = DataPipeline(sample_config)
        
        # Drop timestamp column for normalization
        df_numeric = sample_dataframe.drop(columns=['timestamp'])
        
        normalized_data, scaler = pipeline.normalize(df_numeric, method='minmax')
        
        # Check that values are in [0, 1] range
        assert normalized_data.min() >= 0.0
        assert normalized_data.max() <= 1.0
        
        # Check scaler properties
        assert scaler.scaler_type == 'minmax'
        assert len(scaler.feature_names) == len(df_numeric.columns)
        
        # Check that scaler is stored
        assert pipeline.scaler is not None
    
    def test_normalize_standard(self, sample_config, sample_dataframe):
        """Test Standard normalization."""
        pipeline = DataPipeline(sample_config)
        
        # Drop timestamp column for normalization
        df_numeric = sample_dataframe.drop(columns=['timestamp'])
        
        normalized_data, scaler = pipeline.normalize(df_numeric, method='standard')
        
        # Check scaler properties
        assert scaler.scaler_type == 'standard'
        assert len(scaler.feature_names) == len(df_numeric.columns)
        
        # Standard scaler should produce data with mean ≈ 0 and std ≈ 1
        assert np.abs(normalized_data.mean()) < 0.1  # Close to 0
        assert np.abs(normalized_data.std() - 1.0) < 0.2  # Close to 1
    
    def test_normalize_invalid_method(self, sample_config, sample_dataframe):
        """Test that invalid normalization method raises error."""
        pipeline = DataPipeline(sample_config)
        
        df_numeric = sample_dataframe.drop(columns=['timestamp'])
        
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            pipeline.normalize(df_numeric, method='invalid')
    
    def test_engineer_features_creates_lags(self, sample_config, sample_dataframe):
        """Test that feature engineering creates lagged features."""
        pipeline = DataPipeline(sample_config)
        
        df_features = pipeline.engineer_features(sample_dataframe)
        
        # Should have more columns than original (lagged features added)
        assert len(df_features.columns) > len(sample_dataframe.columns)
        
        # Check that lag columns exist
        lag_cols = [col for col in df_features.columns if '_lag_' in col]
        assert len(lag_cols) > 0
    
    def test_engineer_features_creates_temporal_encodings(self, sample_config, sample_dataframe):
        """Test that feature engineering creates temporal encodings."""
        pipeline = DataPipeline(sample_config)
        
        df_features = pipeline.engineer_features(sample_dataframe)
        
        # Check for temporal encoding columns
        assert 'hour_sin' in df_features.columns
        assert 'hour_cos' in df_features.columns
        assert 'dow_sin' in df_features.columns
        assert 'dow_cos' in df_features.columns
        assert 'month_sin' in df_features.columns
        assert 'month_cos' in df_features.columns
    
    def test_engineer_features_removes_nan_rows(self, sample_config, sample_dataframe):
        """Test that feature engineering removes NaN rows created by lagging."""
        pipeline = DataPipeline(sample_config)
        
        df_features = pipeline.engineer_features(sample_dataframe)
        
        # Should have no NaN values
        assert df_features.isnull().sum().sum() == 0
        
        # Should have fewer rows than original (due to lagging)
        assert len(df_features) < len(sample_dataframe)
    
    def test_create_sequences_shape(self, sample_config):
        """Test that create_sequences produces correct shapes."""
        pipeline = DataPipeline(sample_config)
        
        # Create sample data
        n_samples = 100
        n_features = 5
        data = np.random.randn(n_samples, n_features)
        
        sequence_length = 12
        X, y = pipeline.create_sequences(data, sequence_length)
        
        # Check shapes
        expected_samples = n_samples - sequence_length
        assert X.shape == (expected_samples, sequence_length, n_features)
        assert y.shape == (expected_samples,)
    
    def test_create_sequences_target_column(self, sample_config):
        """Test that create_sequences uses correct target column."""
        pipeline = DataPipeline(sample_config)
        
        # Create sample data with known pattern
        n_samples = 50
        n_features = 3
        data = np.arange(n_samples * n_features).reshape(n_samples, n_features)
        
        sequence_length = 5
        target_column_index = 1
        
        X, y = pipeline.create_sequences(data, sequence_length, target_column_index)
        
        # Check that y contains values from the target column
        # y[0] should be data[5, 1]
        assert y[0] == data[sequence_length, target_column_index]
    
    def test_split_data_maintains_temporal_order(self, sample_config):
        """Test that split_data maintains temporal order."""
        pipeline = DataPipeline(sample_config)
        
        # Create sequential data
        n_samples = 100
        X = np.arange(n_samples * 10).reshape(n_samples, 10)
        y = np.arange(n_samples)
        
        train_ratio = 0.8
        split = pipeline.split_data(X, y, train_ratio)
        
        # Check split sizes
        expected_train_size = int(n_samples * train_ratio)
        assert len(split.X_train) == expected_train_size
        assert len(split.X_test) == n_samples - expected_train_size
        assert len(split.y_train) == expected_train_size
        assert len(split.y_test) == n_samples - expected_train_size
        
        # Check temporal order is maintained
        # Last training sample should come before first test sample
        assert split.y_train[-1] < split.y_test[0]
    
    def test_split_data_default_ratio(self, sample_config):
        """Test split_data with default train ratio."""
        pipeline = DataPipeline(sample_config)
        
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        
        split = pipeline.split_data(X, y)
        
        # Default ratio is 0.8
        assert len(split.X_train) == 80
        assert len(split.X_test) == 20
    
    def test_get_sequence_length_for_horizon(self, sample_config):
        """Test that sequence length adjusts based on forecast horizon."""
        pipeline = DataPipeline(sample_config)
        
        # Test different horizons
        seq_5min = pipeline.get_sequence_length_for_horizon(ForecastHorizon.FIVE_MIN)
        seq_15min = pipeline.get_sequence_length_for_horizon(ForecastHorizon.FIFTEEN_MIN)
        seq_1hour = pipeline.get_sequence_length_for_horizon(ForecastHorizon.ONE_HOUR)
        
        # Longer horizons should have longer or equal sequence lengths
        assert seq_5min >= 12
        assert seq_15min >= 24
        assert seq_1hour >= 48
        assert seq_1hour >= seq_15min >= seq_5min
    
    def test_scaler_transform_inverse_transform(self, sample_config, sample_dataframe):
        """Test that scaler transform and inverse_transform work correctly."""
        pipeline = DataPipeline(sample_config)
        
        df_numeric = sample_dataframe.drop(columns=['timestamp'])
        original_data = df_numeric.values.copy()
        
        # Normalize
        normalized_data, scaler = pipeline.normalize(df_numeric, method='minmax')
        
        # Inverse transform
        reconstructed_data = scaler.inverse_transform(normalized_data)
        
        # Should be close to original (within numerical precision)
        np.testing.assert_allclose(reconstructed_data, original_data, rtol=1e-5)
    
    def test_end_to_end_pipeline(self, sample_config, sample_dataframe_with_missing):
        """Test complete pipeline workflow."""
        pipeline = DataPipeline(sample_config)
        
        # Step 1: Preprocess
        df_clean = pipeline.preprocess(sample_dataframe_with_missing)
        assert df_clean.isnull().sum().sum() == 0
        
        # Step 2: Engineer features
        df_features = pipeline.engineer_features(df_clean)
        assert len(df_features.columns) > len(df_clean.columns)
        
        # Step 3: Normalize
        normalized_data, scaler = pipeline.normalize(df_features, method='minmax')
        assert normalized_data.min() >= 0.0
        assert normalized_data.max() <= 1.0
        
        # Step 4: Create sequences
        sequence_length = 12
        X, y = pipeline.create_sequences(normalized_data, sequence_length)
        assert X.ndim == 3
        assert y.ndim == 1
        
        # Step 5: Split data
        split = pipeline.split_data(X, y, train_ratio=0.8)
        assert len(split.X_train) + len(split.X_test) == len(X)
        assert len(split.y_train) + len(split.y_test) == len(y)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
