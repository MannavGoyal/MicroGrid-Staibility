"""
Example usage of ComparativeEngine for comparing multiple prediction models.

This script demonstrates how to:
1. Load multiple trained models
2. Run comparative analysis on test data
3. Calculate improvements over baseline
4. Rank models by different metrics
5. Perform statistical significance tests
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classical import ClassicalPredictor
from src.models.lstm import LSTMPredictor
from src.analysis.comparative_engine import ComparativeEngine
from src.config.schemas import MicrogridConfig, MicrogridMode


def generate_synthetic_data(n_samples: int = 288) -> tuple:
    """
    Generate synthetic test data for demonstration.
    
    Args:
        n_samples: Number of samples (default: 288 = 24 hours at 5-min resolution)
    
    Returns:
        Tuple of (test_data, test_targets, actual_pv, load_profile)
    """
    # Time array (5-minute intervals)
    time = np.arange(n_samples) * 5 / 60  # Hours
    
    # Synthetic PV power (sinusoidal pattern for daylight hours)
    actual_pv = np.maximum(0, 5.0 * np.sin(np.pi * time / 12) ** 2)
    
    # Add some noise
    actual_pv += np.random.normal(0, 0.2, n_samples)
    actual_pv = np.maximum(0, actual_pv)
    
    # Load profile (higher during day, lower at night)
    load_profile = 2.0 + 1.5 * np.sin(np.pi * time / 12) + np.random.normal(0, 0.1, n_samples)
    load_profile = np.maximum(0.5, load_profile)
    
    # Create test data (features: irradiance, temperature, cloud_cover, humidity, wind_speed)
    # For simplicity, use lagged PV values and synthetic weather
    sequence_length = 12
    n_features = 5
    
    test_data = np.zeros((n_samples, sequence_length, n_features))
    
    for i in range(n_samples):
        # Create lagged features
        for j in range(sequence_length):
            idx = max(0, i - sequence_length + j)
            
            # Feature 0: Normalized irradiance (proxy from PV)
            test_data[i, j, 0] = actual_pv[idx] / 5.0
            
            # Feature 1: Temperature (25-35°C)
            test_data[i, j, 1] = 30 + 5 * np.sin(np.pi * time[idx] / 12)
            
            # Feature 2: Cloud cover (0-1)
            test_data[i, j, 2] = 0.3 + 0.2 * np.random.random()
            
            # Feature 3: Humidity (0-1)
            test_data[i, j, 3] = 0.6 + 0.1 * np.random.random()
            
            # Feature 4: Wind speed (m/s)
            test_data[i, j, 4] = 3.0 + 2.0 * np.random.random()
    
    test_targets = actual_pv
    
    return test_data, test_targets, actual_pv, load_profile


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ComparativeEngine Example")
    print("=" * 80)
    
    # Generate synthetic test data
    print("\n1. Generating synthetic test data...")
    test_data, test_targets, actual_pv, load_profile = generate_synthetic_data(n_samples=288)
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Actual PV range: {actual_pv.min():.2f} - {actual_pv.max():.2f} kW")
    print(f"   Load range: {load_profile.min():.2f} - {load_profile.max():.2f} kW")
    
    # Configure microgrid
    print("\n2. Configuring microgrid...")
    microgrid_config = MicrogridConfig(
        mode=MicrogridMode.ISLANDED,
        pv_capacity_kw=10.0,
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        inverter_capacity_kw=12.0,
        initial_soc_kwh=2.5
    )
    print(f"   Mode: {microgrid_config.mode}")
    print(f"   PV Capacity: {microgrid_config.pv_capacity_kw} kW")
    print(f"   Battery: {microgrid_config.battery_capacity_kwh} kWh / {microgrid_config.battery_power_kw} kW")
    
    # Create models
    print("\n3. Creating prediction models...")
    models = {}
    
    # Persistence model
    persistence_config = {'method': 'persistence'}
    persistence_model = ClassicalPredictor(persistence_config, method='persistence')
    persistence_model.build_model()
    # Train on dummy data (persistence doesn't need training)
    dummy_X = test_data[:10]
    dummy_y = test_targets[:10]
    persistence_model.train(dummy_X, dummy_y, dummy_X, dummy_y)
    models['persistence'] = persistence_model
    print("   ✓ Persistence model created")
    
    # ARIMA model
    arima_config = {'method': 'arima', 'order': (5, 1, 0)}
    arima_model = ClassicalPredictor(arima_config, method='arima')
    arima_model.build_model()
    # For demo, train on small subset
    train_X = test_data[:100]
    train_y = test_targets[:100]
    val_X = test_data[100:120]
    val_y = test_targets[100:120]
    arima_model.train(train_X, train_y, val_X, val_y)
    models['arima'] = arima_model
    print("   ✓ ARIMA model created")
    
    # Initialize ComparativeEngine
    print("\n4. Initializing ComparativeEngine...")
    engine = ComparativeEngine(
        microgrid_config=microgrid_config,
        ems_strategy='reactive'  # Use reactive for simplicity
    )
    print("   ✓ Engine initialized with reactive EMS strategy")
    
    # Run comparison
    print("\n5. Running comparative analysis...")
    print("   This may take a moment...")
    
    comparison_result = engine.run_comparison(
        models=models,
        test_data=test_data,
        test_targets=test_targets,
        actual_pv=actual_pv,
        load_profile=load_profile,
        baseline='no_forecast'
    )
    
    print("   ✓ Comparison complete")
    
    # Display results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Prediction metrics
    print("\n6. Prediction Metrics:")
    print("-" * 80)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12} {'R²':<12}")
    print("-" * 80)
    
    for model_name, result in comparison_result.models.items():
        metrics = result.prediction_metrics
        if np.isinf(metrics['mae']):
            print(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"{model_name:<20} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} "
                  f"{metrics['mape']:<12.2f} {metrics['r2']:<12.4f}")
    
    # Stability metrics
    print("\n7. Stability Metrics:")
    print("-" * 80)
    print(f"{'Model':<20} {'Freq Std (Hz)':<15} {'Volt Std (%)':<15} {'Battery Cycles':<15}")
    print("-" * 80)
    
    for model_name, result in comparison_result.models.items():
        freq_std = result.stability_metrics.frequency.std_deviation
        volt_std = result.stability_metrics.voltage.std_deviation
        cycles = result.stability_metrics.battery.num_cycles
        print(f"{model_name:<20} {freq_std:<15.4f} {volt_std:<15.4f} {cycles:<15.2f}")
    
    # Rankings
    print("\n8. Model Rankings:")
    print("-" * 80)
    
    for metric_name, ranked_models in comparison_result.rankings.items():
        print(f"\n{metric_name.upper()}:")
        for i, model_name in enumerate(ranked_models, 1):
            print(f"  {i}. {model_name}")
    
    # Improvements
    print("\n9. Improvements over Baseline (no_forecast):")
    print("-" * 80)
    
    for model_name, improvements in comparison_result.improvements.items():
        if model_name == 'no_forecast':
            continue
        
        print(f"\n{model_name.upper()}:")
        for metric_name, improvement in improvements.items():
            print(f"  {metric_name}: {improvement:+.2f}%")
    
    # Statistical significance
    print("\n10. Statistical Significance Tests:")
    print("-" * 80)
    
    # Compare persistence vs no_forecast
    if 'persistence' in comparison_result.models and 'no_forecast' in comparison_result.models:
        sig_test = engine.statistical_significance(
            results=comparison_result.models,
            model_a='persistence',
            model_b='no_forecast',
            metric='freq_std'
        )
        
        print(f"\nComparing: {sig_test.model_a} vs {sig_test.model_b}")
        print(f"Metric: {sig_test.metric}")
        print(f"Test statistic: {sig_test.statistic:.4f}")
        print(f"P-value: {sig_test.p_value:.4f}")
        print(f"Significant: {sig_test.significant}")
        print(f"Interpretation: {sig_test.interpretation}")
    
    # Compare arima vs persistence
    if 'arima' in comparison_result.models and 'persistence' in comparison_result.models:
        sig_test = engine.statistical_significance(
            results=comparison_result.models,
            model_a='arima',
            model_b='persistence',
            metric='freq_std'
        )
        
        print(f"\nComparing: {sig_test.model_a} vs {sig_test.model_b}")
        print(f"Metric: {sig_test.metric}")
        print(f"Test statistic: {sig_test.statistic:.4f}")
        print(f"P-value: {sig_test.p_value:.4f}")
        print(f"Significant: {sig_test.significant}")
        print(f"Interpretation: {sig_test.interpretation}")
    
    # Generate comparison table
    print("\n11. Generating comparison table...")
    comparison_table = engine.generate_comparison_table(comparison_result)
    print("   ✓ Comparison table generated")
    print(f"   Models in table: {list(comparison_table['models'].keys())}")
    print(f"   Ranking categories: {list(comparison_table['rankings'].keys())}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
