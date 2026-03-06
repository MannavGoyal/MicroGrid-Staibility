"""
End-to-end example demonstrating the complete workflow:
Upload data → Train model → Run simulation → View results → Export

This example shows how all components wire together.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config.schemas import (
    Configuration, ModelConfig, MicrogridConfig, TrainingConfig,
    ForecastHorizon, ModelType, MicrogridMode
)
from src.data.parser import Parser
from src.data.pipeline import DataPipeline
from src.data.validator import DataValidator
from src.models.classical import ClassicalPredictor
from src.models.lstm import LSTMPredictor
from src.simulation.simulator import MicrogridSimulator
from src.simulation.ems_controller import EMSController
from src.analysis.stability_analyzer import StabilityAnalyzer
from src.analysis.comparative_engine import ComparativeEngine
from src.analysis.results_exporter import ResultsExporter


def generate_sample_data(num_days=30, resolution_minutes=5):
    """Generate sample time-series data for testing."""
    print(f"Generating {num_days} days of sample data...")
    
    # Create time index
    start_date = datetime(2024, 1, 1)
    periods = num_days * 24 * 60 // resolution_minutes
    timestamps = [start_date + timedelta(minutes=i*resolution_minutes) for i in range(periods)]
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Solar irradiance (0-1 kW/m², follows daily pattern)
    hours = np.array([(ts.hour + ts.minute/60) for ts in timestamps])
    irradiance = np.maximum(0, np.sin((hours - 6) * np.pi / 12) * 0.9 + np.random.normal(0, 0.1, len(hours)))
    irradiance = np.clip(irradiance, 0, 1.0)
    
    # Temperature (15-35°C, follows daily pattern)
    temperature = 25 + 8 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 2, len(hours))
    
    # Cloud cover (0-1, random with some persistence)
    cloud_cover = np.random.beta(2, 5, len(hours))
    
    # Humidity (0.3-0.9)
    humidity = 0.6 + 0.2 * np.sin(hours * np.pi / 12) + np.random.normal(0, 0.1, len(hours))
    humidity = np.clip(humidity, 0.3, 0.9)
    
    # Wind speed (0-10 m/s)
    wind_speed = np.abs(np.random.normal(3, 2, len(hours)))
    wind_speed = np.clip(wind_speed, 0, 10)
    
    # PV power (affected by irradiance and cloud cover)
    pv_power = irradiance * (1 - 0.5 * cloud_cover) * 10.0  # 10 kW capacity
    
    # Load profile (0.3-0.8 kW, higher during day)
    load = 0.4 + 0.3 * np.sin((hours - 8) * np.pi / 14) + np.random.normal(0, 0.05, len(hours))
    load = np.clip(load, 0.3, 0.8)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'irradiance': irradiance,
        'temperature': temperature,
        'cloud_cover': cloud_cover,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pv_power': pv_power,
        'load': load
    })
    df.set_index('timestamp', inplace=True)
    
    return df


def main():
    """Run end-to-end workflow."""
    print("=" * 80)
    print("MICROGRID STABILITY ENHANCEMENT - END-TO-END WORKFLOW")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Step 1: Generate and validate data
    # ========================================================================
    print("STEP 1: Generate and validate data")
    print("-" * 80)
    
    df = generate_sample_data(num_days=30, resolution_minutes=5)
    print(f"Generated data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Save data
    data_path = "data/sample_data.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path)
    print(f"Saved data to {data_path}")
    print()
    
    # Validate data
    validator = DataValidator()
    validation_result = validator.validate_timeseries(df)
    print(f"Data validation: {'PASSED' if validation_result.valid else 'FAILED'}")
    if validation_result.warnings:
        print(f"Warnings: {validation_result.warnings}")
    if validation_result.critical_errors:
        print(f"Errors: {validation_result.critical_errors}")
    print()
    
    # ========================================================================
    # Step 2: Configure experiment
    # ========================================================================
    print("STEP 2: Configure experiment")
    print("-" * 80)
    
    config = Configuration(
        experiment_name="end_to_end_test",
        forecast_horizon=ForecastHorizon.FIFTEEN_MIN,
        model_configuration=ModelConfig(
            model_type=ModelType.LSTM,
            hyperparameters={
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.2
            },
            sequence_length=12
        ),
        microgrid_configuration=MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        ),
        training_configuration=TrainingConfig(
            epochs=20,  # Reduced for demo
            batch_size=64,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=5
        ),
        data_path=data_path,
        output_dir="results/end_to_end"
    )
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model_configuration.model_type}")
    print(f"Forecast horizon: {config.forecast_horizon}")
    print(f"Microgrid mode: {config.microgrid_configuration.mode}")
    print()
    
    # ========================================================================
    # Step 3: Preprocess data
    # ========================================================================
    print("STEP 3: Preprocess data")
    print("-" * 80)
    
    parser = Parser()
    df_loaded = parser.parse_timeseries_data(data_path)
    
    pipeline = DataPipeline(config)
    df_clean = pipeline.preprocess(df_loaded)
    X_normalized, scaler = pipeline.normalize(df_clean)
    X, y = pipeline.create_sequences(
        X_normalized,
        config.model_configuration.sequence_length
    )
    
    print(f"Preprocessed data shape: {df_clean.shape}")
    print(f"Sequence shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = pipeline.split_data(
        X, y,
        train_ratio=1.0 - config.training_configuration.validation_split
    )
    
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print()
    
    # ========================================================================
    # Step 4: Train model
    # ========================================================================
    print("STEP 4: Train model")
    print("-" * 80)
    
    # Prepare model config with input_size
    model_config = config.model_configuration.hyperparameters.copy()
    model_config['input_size'] = X_train.shape[2]  # Number of features
    model_config['epochs'] = config.training_configuration.epochs
    model_config['batch_size'] = config.training_configuration.batch_size
    model_config['learning_rate'] = config.training_configuration.learning_rate
    model_config['early_stopping_patience'] = config.training_configuration.early_stopping_patience
    
    model = LSTMPredictor(model_config)
    print(f"Training {config.model_configuration.model_type} model...")
    
    training_result = model.train(X_train, y_train, X_val, y_val)
    
    print(f"Training completed!")
    print(f"Final train loss: {training_result['train_loss']:.4f}")
    print(f"Final val loss: {training_result['val_loss']:.4f}")
    
    # Calculate metrics on validation set
    y_val_pred = model.predict(X_val)
    metrics = model.evaluate(y_val, y_val_pred)
    print(f"Metrics: MAE={metrics['mae']:.4f}, "
          f"RMSE={metrics['rmse']:.4f}, "
          f"R²={metrics['r2']:.4f}")
    print()
    
    # Save model
    model_dir = "models/saved_models/end_to_end_test"
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    print(f"Model saved to {model_dir}")
    print()
    
    # ========================================================================
    # Step 5: Generate predictions
    # ========================================================================
    print("STEP 5: Generate predictions")
    print("-" * 80)
    
    # Use validation set for predictions
    predictions = model.predict(X_val)
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Actual range: [{y_val.min():.4f}, {y_val.max():.4f}]")
    print()
    
    # ========================================================================
    # Step 6: Run microgrid simulation
    # ========================================================================
    print("STEP 6: Run microgrid simulation")
    print("-" * 80)
    
    # Denormalize predictions and actuals
    predictions_denorm = scaler.inverse_transform(
        np.column_stack([predictions] + [np.zeros((len(predictions), X_normalized.shape[1]-1))])
    )[:, 0]
    
    actuals_denorm = scaler.inverse_transform(
        np.column_stack([y_val] + [np.zeros((len(y_val), X_normalized.shape[1]-1))])
    )[:, 0]
    
    # Get load profile (use actual load from data)
    load_profile = df_clean['load'].values[-len(predictions):]
    
    # Create simulator
    simulator = MicrogridSimulator(config.microgrid_configuration)
    
    print("Running simulation with forecasts...")
    result_with_forecast = simulator.simulate(
        pv_forecast=predictions_denorm,
        actual_pv=actuals_denorm,
        load_profile=load_profile,
        timestep_seconds=300  # 5 minutes
    )
    
    print("Running baseline simulation (no forecast)...")
    result_no_forecast = simulator.simulate(
        pv_forecast=None,  # No forecast
        actual_pv=actuals_denorm,
        load_profile=load_profile,
        timestep_seconds=300
    )
    
    print(f"Simulation completed for {len(result_with_forecast.timestamps)} timesteps")
    print()
    
    # ========================================================================
    # Step 7: Analyze stability
    # ========================================================================
    print("STEP 7: Analyze stability")
    print("-" * 80)
    
    analyzer = StabilityAnalyzer(
        battery_capacity_kwh=config.microgrid_configuration.battery_capacity_kwh
    )
    
    metrics_with_forecast = analyzer.analyze(result_with_forecast)
    metrics_no_forecast = analyzer.analyze(result_no_forecast)
    
    print("STABILITY METRICS COMPARISON")
    print()
    print(f"{'Metric':<30} {'No Forecast':<15} {'With Forecast':<15} {'Improvement':<15}")
    print("-" * 75)
    
    # Frequency metrics
    freq_std_no = metrics_no_forecast.frequency.std_deviation
    freq_std_with = metrics_with_forecast.frequency.std_deviation
    freq_improvement = (freq_std_no - freq_std_with) / freq_std_no * 100
    print(f"{'Frequency Std Dev (Hz)':<30} {freq_std_no:<15.4f} {freq_std_with:<15.4f} {freq_improvement:<15.1f}%")
    
    # Battery cycles
    cycles_no = metrics_no_forecast.battery.num_cycles
    cycles_with = metrics_with_forecast.battery.num_cycles
    cycles_improvement = (cycles_no - cycles_with) / cycles_no * 100 if cycles_no > 0 else 0
    print(f"{'Battery Cycles':<30} {cycles_no:<15.1f} {cycles_with:<15.1f} {cycles_improvement:<15.1f}%")
    
    # SOC range
    soc_range_no = metrics_no_forecast.battery.soc_range
    soc_range_with = metrics_with_forecast.battery.soc_range
    soc_improvement = (soc_range_no - soc_range_with) / soc_range_no * 100 if soc_range_no > 0 else 0
    print(f"{'SOC Range (kWh)':<30} {soc_range_no:<15.4f} {soc_range_with:<15.4f} {soc_improvement:<15.1f}%")
    
    print()
    
    # ========================================================================
    # Step 8: Export results
    # ========================================================================
    print("STEP 8: Export results")
    print("-" * 80)
    
    export_dir = config.output_dir
    os.makedirs(export_dir, exist_ok=True)
    exporter = ResultsExporter(export_dir)
    
    # Export timeseries
    timeseries_data = {
        "time": result_with_forecast.timestamps,
        "actual_pv": actuals_denorm,
        "predicted_pv": predictions_denorm,
        "load": load_profile,
        "soc_no_forecast": result_no_forecast.battery_soc,
        "soc_with_forecast": result_with_forecast.battery_soc,
        "freq_dev_no_forecast": result_no_forecast.frequency_deviation,
        "freq_dev_with_forecast": result_with_forecast.frequency_deviation
    }
    
    csv_path = exporter.export_timeseries(timeseries_data, "end_to_end_results")
    print(f"Exported timeseries to {csv_path}")
    
    # Export metrics
    metrics_data = {
        "prediction_metrics": metrics,
        "stability_no_forecast": metrics_no_forecast,
        "stability_with_forecast": metrics_with_forecast
    }
    
    json_path = exporter.export_metrics(metrics_data, "end_to_end_metrics")
    print(f"Exported metrics to {json_path}")
    
    # Export configuration
    config_path = exporter.export_configuration(config, "end_to_end_config")
    print(f"Exported configuration to {config_path}")
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("END-TO-END WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Trained {config.model_configuration.model_type} model with R²={metrics['r2']:.4f}")
    print(f"  - Generated {len(predictions)} predictions")
    print(f"  - Ran microgrid simulation for {len(result_with_forecast.timestamps)} timesteps")
    print(f"  - Frequency stability improved by {freq_improvement:.1f}%")
    print(f"  - Battery cycles reduced by {cycles_improvement:.1f}%")
    print(f"  - Results exported to {export_dir}")
    print()
    print("All components successfully wired together!")
    print()


if __name__ == "__main__":
    main()
