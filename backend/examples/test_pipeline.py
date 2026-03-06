"""Example script demonstrating DataPipeline usage."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np
from src.data.pipeline import DataPipeline
from src.config.schemas import (
    Configuration,
    ModelConfig,
    MicrogridConfig,
    TrainingConfig,
    ForecastHorizon,
    ModelType,
    MicrogridMode,
)


def create_sample_data(n_samples=1000):
    """Create sample time-series data for demonstration."""
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
    
    # Simulate solar irradiance with daily pattern
    hours = np.array([t.hour + t.minute/60 for t in timestamps])
    base_irradiance = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
    irradiance = base_irradiance * np.random.uniform(0.8, 1.2, n_samples)
    
    # Simulate temperature with daily pattern
    temperature = 20 + 10 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 2, n_samples)
    
    # Other weather features
    cloud_cover = np.random.uniform(0, 0.5, n_samples)
    humidity = np.random.uniform(0.4, 0.8, n_samples)
    wind_speed = np.random.uniform(0, 8, n_samples)
    
    # PV power output (correlated with irradiance)
    pv_power = irradiance * 8.0 * (1 - 0.004 * (temperature - 25))
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    irradiance[missing_indices[:len(missing_indices)//2]] = np.nan
    temperature[missing_indices[len(missing_indices)//2:]] = np.nan
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'irradiance': irradiance,
        'temperature': temperature,
        'cloud_cover': cloud_cover,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pv_power': pv_power
    })
    
    return df


def main():
    """Demonstrate DataPipeline functionality."""
    print("=" * 70)
    print("DataPipeline Example")
    print("=" * 70)
    
    # Create configuration
    config = Configuration(
        experiment_name="pipeline_demo",
        forecast_horizon=ForecastHorizon.FIFTEEN_MIN,
        model_configuration=ModelConfig(
            model_type=ModelType.LSTM,
            sequence_length=12,
            hyperparameters={'hidden_size': 64, 'num_layers': 2}
        ),
        microgrid_configuration=MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        ),
        training_configuration=TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.001
        ),
        data_path="sample_data.csv",
        output_dir="results"
    )
    
    print(f"\nConfiguration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Forecast Horizon: {config.forecast_horizon.value}")
    print(f"  Model Type: {config.model_configuration.model_type.value}")
    print(f"  Sequence Length: {config.model_configuration.sequence_length}")
    
    # Create sample data
    print(f"\n{'Step 1: Generate Sample Data':-^70}")
    df = create_sample_data(n_samples=1000)
    print(f"Generated {len(df)} samples with {len(df.columns)} features")
    print(f"Features: {', '.join(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Step 1: Preprocess
    print(f"\n{'Step 2: Preprocess Data':-^70}")
    df_clean = pipeline.preprocess(df)
    print(f"After preprocessing:")
    print(f"  Rows: {len(df_clean)}")
    print(f"  Missing values: {df_clean.isnull().sum().sum()}")
    
    # Step 2: Engineer features
    print(f"\n{'Step 3: Engineer Features':-^70}")
    df_features = pipeline.engineer_features(df_clean)
    print(f"After feature engineering:")
    print(f"  Rows: {len(df_features)} (reduced due to lagging)")
    print(f"  Features: {len(df_features.columns)} (increased from {len(df_clean.columns)})")
    
    # Show some engineered features
    lag_features = [col for col in df_features.columns if '_lag_' in col]
    temporal_features = [col for col in df_features.columns if '_sin' in col or '_cos' in col]
    print(f"  Lagged features: {len(lag_features)}")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Example lag features: {lag_features[:5]}")
    print(f"  Temporal features: {temporal_features}")
    
    # Step 3: Normalize
    print(f"\n{'Step 4: Normalize Features':-^70}")
    normalized_data, scaler = pipeline.normalize(df_features, method='minmax')
    print(f"After normalization (MinMax):")
    print(f"  Shape: {normalized_data.shape}")
    print(f"  Min value: {normalized_data.min():.6f}")
    print(f"  Max value: {normalized_data.max():.6f}")
    print(f"  Mean: {normalized_data.mean():.6f}")
    print(f"  Scaler type: {scaler.scaler_type}")
    
    # Step 4: Create sequences
    print(f"\n{'Step 5: Create Sequences':-^70}")
    sequence_length = config.model_configuration.sequence_length
    X, y = pipeline.create_sequences(normalized_data, sequence_length, target_column_index=0)
    print(f"After sequence creation:")
    print(f"  X shape: {X.shape} (samples, sequence_length, features)")
    print(f"  y shape: {y.shape} (samples,)")
    print(f"  Sequence length: {sequence_length}")
    
    # Step 5: Split data
    print(f"\n{'Step 6: Split Data':-^70}")
    split = pipeline.split_data(X, y, train_ratio=0.8)
    print(f"After train/test split (80/20):")
    print(f"  Training set:")
    print(f"    X_train: {split.X_train.shape}")
    print(f"    y_train: {split.y_train.shape}")
    print(f"  Test set:")
    print(f"    X_test: {split.X_test.shape}")
    print(f"    y_test: {split.y_test.shape}")
    
    # Demonstrate inverse transform
    print(f"\n{'Step 7: Inverse Transform':-^70}")
    sample_normalized = normalized_data[:5]
    sample_original = scaler.inverse_transform(sample_normalized)
    print(f"Inverse transform demonstration:")
    print(f"  Normalized sample (first 3 features):")
    print(f"    {sample_normalized[0, :3]}")
    print(f"  After inverse transform:")
    print(f"    {sample_original[0, :3]}")
    
    # Show sequence length recommendations
    print(f"\n{'Sequence Length Recommendations':-^70}")
    for horizon in ForecastHorizon:
        recommended_length = pipeline.get_sequence_length_for_horizon(horizon)
        print(f"  {horizon.value:12s}: {recommended_length} time steps")
    
    print(f"\n{'Pipeline Complete!':-^70}")
    print("\nThe data is now ready for model training!")
    print(f"  - Training samples: {len(split.X_train)}")
    print(f"  - Test samples: {len(split.X_test)}")
    print(f"  - Input shape per sample: ({sequence_length}, {X.shape[2]})")
    print(f"  - Output shape per sample: (1,)")


if __name__ == '__main__':
    main()
