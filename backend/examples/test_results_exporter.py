"""
Example script demonstrating ResultsExporter functionality.

This script shows how to use the ResultsExporter to export simulation results,
metrics, visualizations, and generate comprehensive reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.results_exporter import ResultsExporter
from src.analysis.comparative_engine import ComparisonResult, ModelResult
from src.analysis.stability_analyzer import (
    StabilityMetrics, FrequencyMetrics, VoltageMetrics,
    BatteryMetrics, PowerQualityMetrics, EnergyMetrics, ControlEffortMetrics
)
from src.simulation.simulator import SimulationResult
from src.config.schemas import (
    Configuration, ModelConfig, MicrogridConfig, TrainingConfig,
    ForecastHorizon, ModelType, MicrogridMode
)


def create_sample_data():
    """Create sample simulation data for demonstration."""
    n_samples = 288  # 24 hours at 5-minute intervals
    
    # Create time-series data
    timestamps = np.arange(n_samples) * 300  # 5-minute intervals in seconds
    
    # Simulate PV power (solar curve)
    hours = timestamps / 3600
    pv_power = 10 * np.maximum(0, np.sin((hours - 6) * np.pi / 12))
    pv_power += np.random.randn(n_samples) * 0.5  # Add noise
    pv_power = np.maximum(0, pv_power)
    
    # Simulate load (higher during day)
    load_power = 5 + 2 * np.sin((hours - 12) * np.pi / 12)
    load_power += np.random.randn(n_samples) * 0.3
    
    # Simulate battery operation
    battery_power = pv_power - load_power
    battery_power = np.clip(battery_power, -3, 3)  # Power limits
    
    # Simulate battery SOC
    battery_soc = np.zeros(n_samples)
    battery_soc[0] = 2.5  # Start at 50%
    for i in range(1, n_samples):
        dt_hours = 300 / 3600
        if battery_power[i] > 0:
            battery_soc[i] = battery_soc[i-1] + battery_power[i] * dt_hours * 0.95
        else:
            battery_soc[i] = battery_soc[i-1] + battery_power[i] * dt_hours / 0.95
        battery_soc[i] = np.clip(battery_soc[i], 0, 5)
    
    # Simulate frequency and voltage deviations
    power_imbalance = pv_power + battery_power - load_power
    frequency_deviation = power_imbalance * 0.02  # Simplified model
    voltage_deviation = power_imbalance * 0.5
    
    return {
        'timestamps': timestamps,
        'pv_power': pv_power,
        'load_power': load_power,
        'battery_power': battery_power,
        'battery_soc': battery_soc,
        'frequency_deviation': frequency_deviation,
        'voltage_deviation': voltage_deviation
    }


def create_sample_comparison_result(data):
    """Create sample comparison result."""
    # Create simulation result
    sim_result = SimulationResult(
        timestamps=data['timestamps'],
        pv_power=data['pv_power'],
        load_power=data['load_power'],
        battery_power=data['battery_power'],
        battery_soc=data['battery_soc'],
        frequency_deviation=data['frequency_deviation'],
        voltage_deviation=data['voltage_deviation'],
        grid_power=np.zeros_like(data['pv_power']),
        states=[]
    )
    
    # Create stability metrics
    stability_metrics = StabilityMetrics(
        frequency=FrequencyMetrics(
            mean_absolute_deviation=0.089,
            std_deviation=0.067,
            max_deviation=0.234,
            time_outside_limits=0.03,
            rate_of_change=0.012
        ),
        voltage=VoltageMetrics(
            mean_absolute_deviation=1.45,
            std_deviation=1.12,
            max_deviation=3.89,
            time_outside_limits=0.01
        ),
        battery=BatteryMetrics(
            soc_range=3.2,
            num_cycles=4.5,
            max_depth_of_discharge=0.58,
            avg_c_rate=0.28,
            total_throughput=15.6
        ),
        power_quality=PowerQualityMetrics(
            thd_proxy=0.12,
            power_factor=0.96
        ),
        energy=EnergyMetrics(
            total_unmet_load=0.3,
            total_curtailed_pv=0.8,
            energy_efficiency=0.94
        ),
        control_effort=ControlEffortMetrics(
            sum_absolute_battery_changes=52.3,
            num_control_actions=34
        )
    )
    
    # Create model result
    model_result = ModelResult(
        model_name='lstm',
        predictions=data['pv_power'] + np.random.randn(len(data['pv_power'])) * 0.3,
        prediction_metrics={
            'mae': 0.342,
            'rmse': 0.456,
            'mape': 8.9,
            'r2': 0.912
        },
        simulation_result=sim_result,
        stability_metrics=stability_metrics
    )
    
    # Create baseline result (worse metrics)
    baseline_metrics = StabilityMetrics(
        frequency=FrequencyMetrics(
            mean_absolute_deviation=0.156,
            std_deviation=0.123,
            max_deviation=0.456,
            time_outside_limits=0.08,
            rate_of_change=0.023
        ),
        voltage=VoltageMetrics(
            mean_absolute_deviation=2.34,
            std_deviation=1.89,
            max_deviation=6.12,
            time_outside_limits=0.05
        ),
        battery=BatteryMetrics(
            soc_range=4.1,
            num_cycles=7.8,
            max_depth_of_discharge=0.82,
            avg_c_rate=0.45,
            total_throughput=23.4
        ),
        power_quality=PowerQualityMetrics(
            thd_proxy=0.21,
            power_factor=0.89
        ),
        energy=EnergyMetrics(
            total_unmet_load=1.2,
            total_curtailed_pv=2.1,
            energy_efficiency=0.87
        ),
        control_effort=ControlEffortMetrics(
            sum_absolute_battery_changes=89.7,
            num_control_actions=67
        )
    )
    
    baseline_result = ModelResult(
        model_name='no_forecast',
        predictions=np.zeros_like(data['pv_power']),
        prediction_metrics={
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
            'r2': -float('inf')
        },
        simulation_result=sim_result,
        stability_metrics=baseline_metrics
    )
    
    return ComparisonResult(
        models={'lstm': model_result, 'no_forecast': baseline_result},
        rankings={
            'mae': ['lstm'],
            'freq_stability': ['lstm', 'no_forecast'],
            'battery_stress': ['lstm', 'no_forecast']
        },
        improvements={
            'lstm': {
                'freq_std_improvement': 45.5,
                'volt_std_improvement': 40.7,
                'battery_cycles_improvement': 42.3
            }
        },
        baseline_model='no_forecast'
    )


def create_sample_config():
    """Create sample configuration."""
    return Configuration(
        experiment_name="lstm_baseline_experiment",
        forecast_horizon=ForecastHorizon.FIFTEEN_MIN,
        model_configuration=ModelConfig(
            model_type=ModelType.LSTM,
            hyperparameters={
                "hidden_size": 64,
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
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=10
        ),
        data_path="data/solar_data.csv",
        output_dir="results"
    )


def create_sample_visualizations(data):
    """Create sample visualization figures."""
    figures = []
    
    # Figure 1: PV Power and Load
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    hours = data['timestamps'] / 3600
    ax1.plot(hours, data['pv_power'], label='PV Power', linewidth=2)
    ax1.plot(hours, data['load_power'], label='Load Power', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('PV Power and Load Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    figures.append(fig1)
    
    # Figure 2: Battery SOC
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(hours, data['battery_soc'], label='Battery SOC', linewidth=2, color='green')
    ax2.axhline(y=5.0, color='r', linestyle='--', label='Max Capacity', alpha=0.5)
    ax2.axhline(y=0.0, color='r', linestyle='--', label='Min Capacity', alpha=0.5)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('SOC (kWh)')
    ax2.set_title('Battery State of Charge')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    figures.append(fig2)
    
    # Figure 3: Frequency Deviation
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(hours, data['frequency_deviation'], linewidth=1.5, color='orange')
    ax3.axhline(y=0.5, color='r', linestyle='--', label='Upper Limit', alpha=0.5)
    ax3.axhline(y=-0.5, color='r', linestyle='--', label='Lower Limit', alpha=0.5)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Frequency Deviation (Hz)')
    ax3.set_title('Frequency Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    figures.append(fig3)
    
    return figures


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("ResultsExporter Demonstration")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/exporter_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = ResultsExporter(output_dir=str(output_dir))
    print(f"\n✓ Initialized ResultsExporter with output directory: {output_dir}")
    
    # Create timestamped export directory
    export_dir = exporter.create_export_directory("demo_experiment")
    print(f"✓ Created timestamped export directory: {export_dir.name}")
    
    # Generate sample data
    print("\n" + "-" * 60)
    print("Generating sample data...")
    data = create_sample_data()
    print(f"✓ Generated {len(data['timestamps'])} time steps (24 hours)")
    
    # Export time-series data
    print("\n" + "-" * 60)
    print("Exporting time-series data...")
    timeseries_path = exporter.export_timeseries(
        data={
            'actual_pv': data['pv_power'],
            'load': data['load_power'],
            'battery_soc': data['battery_soc'],
            'battery_power': data['battery_power'],
            'frequency_deviation': data['frequency_deviation'],
            'voltage_deviation': data['voltage_deviation']
        },
        filename='simulation_timeseries',
        timestamps=data['timestamps']
    )
    print(f"✓ Exported time-series to: {Path(timeseries_path).name}")
    
    # Export metrics
    print("\n" + "-" * 60)
    print("Exporting metrics...")
    metrics = {
        'prediction_metrics': {
            'mae': 0.342,
            'rmse': 0.456,
            'mape': 8.9,
            'r2': 0.912
        },
        'stability_metrics': {
            'frequency': {
                'mean_abs_dev': 0.089,
                'std_dev': 0.067,
                'max_dev': 0.234
            },
            'voltage': {
                'mean_abs_dev': 1.45,
                'std_dev': 1.12,
                'max_dev': 3.89
            },
            'battery': {
                'soc_range': 3.2,
                'num_cycles': 4.5,
                'throughput': 15.6
            }
        }
    }
    
    metrics_json_path = exporter.export_metrics(metrics, 'metrics', format='json')
    print(f"✓ Exported metrics (JSON) to: {Path(metrics_json_path).name}")
    
    metrics_csv_path = exporter.export_metrics(metrics, 'metrics_flat', format='csv')
    print(f"✓ Exported metrics (CSV) to: {Path(metrics_csv_path).name}")
    
    # Export visualizations
    print("\n" + "-" * 60)
    print("Exporting visualizations...")
    figures = create_sample_visualizations(data)
    viz_paths = exporter.export_visualizations(figures, 'plot', format='png')
    print(f"✓ Exported {len(viz_paths)} visualizations:")
    for path in viz_paths:
        print(f"  - {Path(path).name}")
    
    # Export configuration
    print("\n" + "-" * 60)
    print("Exporting configuration...")
    config = create_sample_config()
    config_path = exporter.export_configuration(config, 'experiment_config')
    print(f"✓ Exported configuration to: {Path(config_path).name}")
    
    # Generate comprehensive report
    print("\n" + "-" * 60)
    print("Generating comprehensive report...")
    comparison_result = create_sample_comparison_result(data)
    report_path = exporter.generate_report(comparison_result, config, format='markdown')
    print(f"✓ Generated report: {Path(report_path).name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"All artifacts exported to: {export_dir}")
    print("\nExported files:")
    for file in sorted(export_dir.iterdir()):
        print(f"  - {file.name}")
    
    print("\n✓ Demonstration complete!")
    print("\nYou can now:")
    print("  1. View the time-series CSV in Excel or similar")
    print("  2. Review the metrics JSON for detailed analysis")
    print("  3. Check the visualizations (PNG files)")
    print("  4. Read the comprehensive Markdown report")
    print("  5. Use the configuration JSON to reproduce the experiment")


if __name__ == '__main__':
    main()
