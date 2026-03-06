# ComparativeEngine Documentation

## Overview

The `ComparativeEngine` class provides a systematic framework for comparing multiple PV forecasting models and quantifying their impact on microgrid stability. It orchestrates the execution of different prediction models on identical test data, runs microgrid simulations with each forecast, and calculates comparative metrics.

## Key Features

1. **Multi-Model Comparison**: Execute and compare multiple forecasting approaches (Classical, LSTM, Advanced models)
2. **Identical Test Conditions**: Ensures all models are evaluated on the same test data and microgrid configuration
3. **Comprehensive Metrics**: Calculates both prediction metrics (MAE, RMSE, MAPE, R²) and stability metrics (frequency, voltage, battery stress)
4. **Improvement Calculations**: Computes percentage improvements relative to a baseline (typically no-forecast)
5. **Model Rankings**: Ranks models by various metrics to identify best performers
6. **Statistical Significance**: Performs paired t-tests to determine if differences are statistically significant
7. **Comparison Tables**: Generates structured comparison data for visualization and reporting

## Usage Example

```python
from src.analysis.comparative_engine import ComparativeEngine
from src.config.schemas import MicrogridConfig, MicrogridMode

# Configure microgrid
microgrid_config = MicrogridConfig(
    mode=MicrogridMode.ISLANDED,
    pv_capacity_kw=10.0,
    battery_capacity_kwh=5.0,
    battery_power_kw=3.0,
    inverter_capacity_kw=12.0,
    initial_soc_kwh=2.5
)

# Initialize engine
engine = ComparativeEngine(
    microgrid_config=microgrid_config,
    ems_strategy='mpc'  # or 'rule_based', 'reactive'
)

# Prepare models dictionary
models = {
    'persistence': persistence_model,
    'arima': arima_model,
    'lstm': lstm_model
}

# Run comparison
comparison_result = engine.run_comparison(
    models=models,
    test_data=test_data,
    test_targets=test_targets,
    actual_pv=actual_pv,
    load_profile=load_profile,
    baseline='no_forecast'
)

# Access results
for model_name, result in comparison_result.models.items():
    print(f"{model_name}: MAE={result.prediction_metrics['mae']:.4f}")

# View rankings
print("Best models by frequency stability:")
print(comparison_result.rankings['freq_stability'])

# View improvements
for model_name, improvements in comparison_result.improvements.items():
    print(f"{model_name} improvements:")
    for metric, value in improvements.items():
        print(f"  {metric}: {value:+.2f}%")
```

## Main Methods

### `run_comparison()`
Executes all models and generates complete comparison results.

**Parameters:**
- `models`: Dictionary mapping model names to predictor instances
- `test_data`: Test input features
- `test_targets`: Test target values (actual PV power)
- `actual_pv`: Actual PV power for simulation
- `load_profile`: Load demand profile
- `baseline`: Name of baseline model for improvement calculations

**Returns:** `ComparisonResult` object

### `calculate_improvements()`
Calculates percentage improvements over baseline for all metrics.

**Parameters:**
- `results`: Dictionary of model results
- `baseline`: Name of baseline model

**Returns:** Dictionary mapping model names to improvement percentages

### `rank_models()`
Ranks models by a specified metric.

**Parameters:**
- `results`: Dictionary of model results
- `metric`: Metric name to rank by

**Returns:** List of (model_name, metric_value) tuples sorted by rank

### `statistical_significance()`
Performs statistical significance test comparing two models.

**Parameters:**
- `results`: Dictionary of model results
- `model_a`: Name of first model
- `model_b`: Name of second model
- `metric`: Metric to compare

**Returns:** `SignificanceTest` object with test results

### `generate_comparison_table()`
Generates structured comparison table with all metrics.

**Parameters:**
- `comparison_result`: Complete comparison results

**Returns:** Dictionary containing formatted comparison data

## Data Structures

### `ModelResult`
Contains results for a single model:
- `model_name`: Name of the model
- `predictions`: Predicted PV power values
- `prediction_metrics`: Dictionary of prediction metrics (MAE, RMSE, MAPE, R²)
- `simulation_result`: SimulationResult object
- `stability_metrics`: StabilityMetrics object

### `ComparisonResult`
Contains complete comparison results:
- `models`: Dictionary of ModelResult objects
- `rankings`: Dictionary mapping metric names to ranked model lists
- `improvements`: Dictionary of improvement percentages
- `baseline_model`: Name of baseline model

### `SignificanceTest`
Contains statistical test results:
- `model_a`, `model_b`: Names of compared models
- `metric`: Metric being compared
- `statistic`: Test statistic value
- `p_value`: P-value from test
- `significant`: Boolean indicating if difference is significant (p < 0.05)
- `interpretation`: Human-readable interpretation

## Requirements Validated

This implementation validates the following requirements:

- **11.1**: Execute simulations for no-forecast baseline, Classical_Predictor methods, and Advanced_Predictor methods
- **11.2**: Use identical input data, microgrid configuration, and simulation parameters across all comparison runs
- **11.3**: Generate comparison table with prediction metrics and stability metrics for each method
- **11.4**: Calculate percentage improvement in each stability metric relative to baseline
- **11.5**: Identify best-performing method for each metric category
- **11.6**: Generate statistical significance tests comparing methods

## Testing

Unit tests are provided in `backend/tests/unit/test_comparative_engine.py`.

Example usage is demonstrated in `backend/examples/test_comparative_engine.py`.

Run tests with:
```bash
python -m pytest backend/tests/unit/test_comparative_engine.py -v
```

Run example with:
```bash
python backend/examples/test_comparative_engine.py
```
