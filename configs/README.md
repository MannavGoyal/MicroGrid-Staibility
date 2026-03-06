# Configuration Files

This directory contains example configuration files for the Microgrid Stability Enhancement application.

## Available Configurations

### 1. example_config.json (LSTM - 15 minute forecast)
Default configuration using LSTM model for 15-minute ahead forecasting in islanded mode.

**Use case**: Standard microgrid with moderate battery capacity, suitable for most applications.

**Key parameters**:
- Model: LSTM (64 hidden units, 2 layers)
- Forecast horizon: 15 minutes
- Microgrid mode: Islanded
- PV capacity: 10 kW
- Battery: 5 kWh capacity, 3 kW power

### 2. example_arima_config.json (ARIMA - 5 minute forecast)
Classical ARIMA model for short-term forecasting in grid-connected mode.

**Use case**: Grid-connected system with larger battery, faster response needed.

**Key parameters**:
- Model: ARIMA (5,1,0)
- Forecast horizon: 5 minutes
- Microgrid mode: Grid-connected
- PV capacity: 15 kW
- Battery: 10 kWh capacity, 5 kW power

### 3. example_svr_config.json (SVR - 1 hour forecast)
Support Vector Regression for longer-term forecasting with diesel backup.

**Use case**: Larger islanded system with diesel generator backup, longer planning horizon.

**Key parameters**:
- Model: SVR (RBF kernel)
- Forecast horizon: 1 hour
- Microgrid mode: Islanded with diesel generator
- PV capacity: 20 kW
- Battery: 15 kWh capacity, 8 kW power
- Diesel generator: 10 kW

## Configuration Schema

All configuration files follow this structure:

```json
{
  "experiment_name": "string",
  "forecast_horizon": "5min" | "15min" | "1hour",
  "model_configuration": {
    "model_type": "lstm" | "arima" | "svr" | "persistence",
    "hyperparameters": {
      // Model-specific hyperparameters
    },
    "sequence_length": integer
  },
  "microgrid_configuration": {
    "mode": "islanded" | "grid_connected" | "hybrid_ac_dc",
    "pv_capacity_kw": float,
    "battery_capacity_kwh": float,
    "battery_power_kw": float,
    "inverter_capacity_kw": float,
    "initial_soc_kwh": float,
    "has_diesel_generator": boolean,
    "diesel_capacity_kw": float | null
  },
  "training_configuration": {
    "epochs": integer,
    "batch_size": integer,
    "learning_rate": float,
    "validation_split": float,
    "early_stopping_patience": integer
  },
  "data_path": "string",
  "output_dir": "string"
}
```

## Model-Specific Hyperparameters

### LSTM
```json
"hyperparameters": {
  "hidden_size": 32-128,
  "num_layers": 1-3,
  "dropout": 0.0-0.5
}
```

### ARIMA
```json
"hyperparameters": {
  "arima_order": [p, d, q]
}
```
- p: autoregressive order (typically 1-10)
- d: differencing order (typically 0-2)
- q: moving average order (typically 0-5)

### SVR
```json
"hyperparameters": {
  "C": 0.1-10.0,
  "epsilon": 0.01-0.5,
  "kernel": "rbf" | "linear" | "poly"
}
```

### Persistence (no hyperparameters)
```json
"hyperparameters": {}
```

## Validation Rules

The configuration is validated using Pydantic schemas. Key constraints:

1. **Inverter capacity** must be ≥ PV capacity
2. **Initial SOC** must be between 0 and battery capacity
3. **Battery power** should be reasonable relative to capacity (typically 0.2-2.0 C-rate)
4. **Sequence length** should match forecast horizon:
   - 5min: 12-24 steps (1-2 hours history)
   - 15min: 8-16 steps (2-4 hours history)
   - 1hour: 6-12 steps (6-12 hours history)
5. **Validation split** must be between 0.0 and 1.0

## Usage Examples

### Using with Python API

```python
from backend.src.data.parser import Parser

parser = Parser()
config = parser.parse_config('configs/example_config.json')

print(f"Experiment: {config.experiment_name}")
print(f"Model: {config.model_configuration.model_type}")
print(f"Forecast horizon: {config.forecast_horizon}")
```

### Using with REST API

```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d @configs/example_config.json
```

### Using with CLI (if implemented)

```bash
python backend/src/cli.py train --config configs/example_config.json
```

## Creating Custom Configurations

1. Copy one of the example files
2. Modify parameters for your use case
3. Validate using the parser:

```python
from backend.src.data.parser import Parser

parser = Parser()
try:
    config = parser.parse_config('configs/my_config.json')
    print("Configuration valid!")
except Exception as e:
    print(f"Configuration error: {e}")
```

## Tips for Configuration

### Choosing Forecast Horizon
- **5min**: Fast-changing conditions, need quick response
- **15min**: Balanced approach, most common
- **1hour**: Slower dynamics, longer planning horizon

### Choosing Model Type
- **Persistence**: Baseline, assumes no change
- **ARIMA**: Good for stationary time series, fast training
- **SVR**: Good for non-linear patterns, moderate training time
- **LSTM**: Best for complex patterns, longer training time

### Sizing Battery
- **Capacity**: Typically 0.5-2.0x peak PV capacity (in hours)
- **Power**: Typically 0.5-1.0x PV capacity
- **Initial SOC**: Start at 50% (0.5 * capacity) for balanced operation

### Training Configuration
- **Epochs**: 20-50 for LSTM, 1 for classical models
- **Batch size**: 32-128, larger for more data
- **Learning rate**: 0.001 is a good starting point
- **Validation split**: 0.2 (20%) is standard

## Troubleshooting

### "Inverter capacity must be >= PV capacity"
Increase `inverter_capacity_kw` to at least match `pv_capacity_kw`.

### "Initial SOC out of range"
Ensure `initial_soc_kwh` is between 0 and `battery_capacity_kwh`.

### "Training fails to converge"
- Reduce learning rate
- Increase epochs
- Increase early stopping patience
- Check data quality

### "Simulation unstable"
- Increase battery capacity or power
- Adjust forecast horizon
- Check PV/load balance

## References

- Configuration schema: `backend/src/config/schemas.py`
- Parser implementation: `backend/src/data/parser.py`
- API documentation: `backend/API_DOCUMENTATION.md`
