# Microgrid Stability Analysis Report

**Generated:** 2026-03-03 18:16:13

## Configuration

- **Experiment Name:** lstm_baseline_experiment
- **Forecast Horizon:** 15min
- **Model Type:** lstm
- **Battery Capacity:** 5.0 kWh
- **PV Capacity:** 10.0 kW
- **Microgrid Mode:** islanded

## Prediction Performance

| Model | MAE | RMSE | MAPE (%) | R² |
|-------|-----|------|----------|----|
| lstm | 0.3420 | 0.4560 | 8.90 | 0.9120 |

## Stability Improvements

Comparison against baseline: **no_forecast**

### Frequency Stability

| Model | Mean Abs Dev (Hz) | Std Dev (Hz) | Max Dev (Hz) | Improvement (%) |
|-------|-------------------|--------------|--------------|------------------|
| lstm | 0.0890 | 0.0670 | 0.2340 | 45.5 |
| no_forecast | 0.1560 | 0.1230 | 0.4560 | 0.0 |

### Voltage Stability

| Model | Mean Abs Dev (%) | Std Dev (%) | Max Dev (%) | Improvement (%) |
|-------|------------------|-------------|-------------|------------------|
| lstm | 1.4500 | 1.1200 | 3.8900 | 40.7 |
| no_forecast | 2.3400 | 1.8900 | 6.1200 | 0.0 |

### Battery Stress

| Model | SOC Range (kWh) | Cycles | Max DOD | Throughput (kWh) | Improvement (%) |
|-------|-----------------|--------|---------|------------------|------------------|
| lstm | 3.20 | 4.5 | 0.58 | 15.60 | 42.3 |
| no_forecast | 4.10 | 7.8 | 0.82 | 23.40 | 0.0 |

## Model Rankings

### Mae

1. lstm

### Freq Stability

1. lstm
2. no_forecast

### Battery Stress

1. lstm
2. no_forecast

## Key Findings

- **Best Prediction Model:** lstm
- **Best Frequency Stability:** lstm
- **Lowest Battery Stress:** lstm

## Conclusion

This analysis demonstrates the impact of PV forecasting accuracy on microgrid stability. The results show quantifiable improvements in frequency stability, voltage stability, and battery stress when using advanced forecasting methods compared to reactive control.
