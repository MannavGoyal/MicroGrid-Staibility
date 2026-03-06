"""
Example usage of StabilityAnalyzer.

This script demonstrates how to use the StabilityAnalyzer to calculate
comprehensive stability metrics from simulation results.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to enable backend imports
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from backend.src.analysis.stability_analyzer import StabilityAnalyzer
from backend.src.simulation.simulator import MicrogridSimulator, SimulationResult
from backend.src.config.schemas import MicrogridConfig, MicrogridMode


def create_sample_simulation_result():
    """Create a sample simulation result for testing."""
    n_steps = 288  # 24 hours at 5-minute intervals
    timestamps = np.arange(n_steps) * 300.0  # 5-minute timesteps in seconds
    
    # Create realistic PV profile (solar curve)
    hours = np.arange(n_steps) / 12.0  # Convert to hours
    pv_power = 5.0 * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))  # Peak at noon
    
    # Create load profile (higher in morning and evening)
    load_power = 2.0 + 1.0 * np.sin(2 * np.pi * hours / 24) + 0.5 * np.random.randn(n_steps) * 0.1
    load_power = np.maximum(0.5, load_power)
    
    # Create battery power (trying to balance)
    battery_power = -(pv_power - load_power) * 0.8  # 80% compensation
    battery_power = np.clip(battery_power, -3.0, 3.0)  # Power limits
    
    # Create battery SOC (integrate power)
    battery_soc = np.zeros(n_steps)
    battery_soc[0] = 2.5  # Start at 50%
    for i in range(1, n_steps):
        dt_hours = 5.0 / 60.0  # 5 minutes
        battery_soc[i] = battery_soc[i-1] + battery_power[i-1] * dt_hours * 0.95
        battery_soc[i] = np.clip(battery_soc[i], 0.5, 4.5)  # SOC limits
    
    # Create frequency deviation (proportional to power imbalance)
    power_imbalance = pv_power + battery_power - load_power
    frequency_deviation = power_imbalance * 0.1 + np.random.randn(n_steps) * 0.05
    
    # Create voltage deviation (smaller variations)
    voltage_deviation = power_imbalance * 0.05 + np.random.randn(n_steps) * 0.02
    
    return SimulationResult(
        timestamps=timestamps,
        pv_power=pv_power,
        load_power=load_power,
        battery_power=battery_power,
        battery_soc=battery_soc,
        frequency_deviation=frequency_deviation,
        voltage_deviation=voltage_deviation,
        grid_power=np.zeros(n_steps),
        states=[]
    )


def print_metrics(metrics, scenario_name):
    """Print metrics in a readable format."""
    print(f"\n{'='*60}")
    print(f"Stability Metrics - {scenario_name}")
    print(f"{'='*60}")
    
    print("\n--- Frequency Metrics ---")
    print(f"  Mean Absolute Deviation: {metrics.frequency.mean_absolute_deviation:.4f} Hz")
    print(f"  Standard Deviation:      {metrics.frequency.std_deviation:.4f} Hz")
    print(f"  Maximum Deviation:       {metrics.frequency.max_deviation:.4f} Hz")
    print(f"  Time Outside Limits:     {metrics.frequency.time_outside_limits*100:.2f}%")
    print(f"  Rate of Change:          {metrics.frequency.rate_of_change:.4f} Hz/step")
    
    print("\n--- Voltage Metrics ---")
    print(f"  Mean Absolute Deviation: {metrics.voltage.mean_absolute_deviation:.4f}%")
    print(f"  Standard Deviation:      {metrics.voltage.std_deviation:.4f}%")
    print(f"  Maximum Deviation:       {metrics.voltage.max_deviation:.4f}%")
    print(f"  Time Outside Limits:     {metrics.voltage.time_outside_limits*100:.2f}%")
    
    print("\n--- Battery Stress Metrics ---")
    print(f"  SOC Range:               {metrics.battery.soc_range:.4f} kWh")
    print(f"  Number of Cycles:        {metrics.battery.num_cycles:.2f}")
    print(f"  Max Depth of Discharge:  {metrics.battery.max_depth_of_discharge*100:.2f}%")
    print(f"  Average C-Rate:          {metrics.battery.avg_c_rate:.4f}")
    print(f"  Total Throughput:        {metrics.battery.total_throughput:.4f} kWh")
    
    print("\n--- Power Quality Metrics ---")
    print(f"  THD Proxy:               {metrics.power_quality.thd_proxy:.4f}")
    print(f"  Power Factor:            {metrics.power_quality.power_factor:.4f}")
    
    print("\n--- Energy Balance Metrics ---")
    print(f"  Total Unmet Load:        {metrics.energy.total_unmet_load:.4f} kWh")
    print(f"  Total Curtailed PV:      {metrics.energy.total_curtailed_pv:.4f} kWh")
    print(f"  Energy Efficiency:       {metrics.energy.energy_efficiency*100:.2f}%")
    
    print("\n--- Control Effort Metrics ---")
    print(f"  Sum Abs Battery Changes: {metrics.control_effort.sum_absolute_battery_changes:.4f} kW")
    print(f"  Number Control Actions:  {metrics.control_effort.num_control_actions}")


def compare_scenarios():
    """Compare forecast-enabled vs no-forecast scenarios."""
    print("\n" + "="*60)
    print("Comparing Forecast-Enabled vs No-Forecast Scenarios")
    print("="*60)
    
    # Create analyzer
    analyzer = StabilityAnalyzer(battery_capacity_kwh=5.0)
    
    # Scenario 1: With forecast (good battery control)
    print("\nGenerating forecast-enabled scenario...")
    sim_result_forecast = create_sample_simulation_result()
    metrics_forecast = analyzer.analyze(sim_result_forecast)
    print_metrics(metrics_forecast, "Forecast-Enabled")
    
    # Scenario 2: No forecast (reactive control - worse performance)
    print("\nGenerating no-forecast scenario...")
    sim_result_no_forecast = create_sample_simulation_result()
    # Make it worse by adding more frequency deviation
    sim_result_no_forecast.frequency_deviation *= 1.5
    sim_result_no_forecast.voltage_deviation *= 1.3
    # More battery cycling
    sim_result_no_forecast.battery_power *= 1.2
    
    metrics_no_forecast = analyzer.analyze(sim_result_no_forecast)
    print_metrics(metrics_no_forecast, "No Forecast (Baseline)")
    
    # Calculate improvements
    print("\n" + "="*60)
    print("Improvements with Forecast")
    print("="*60)
    
    freq_improvement = (
        (metrics_no_forecast.frequency.std_deviation - metrics_forecast.frequency.std_deviation)
        / metrics_no_forecast.frequency.std_deviation * 100
    )
    print(f"\nFrequency Std Deviation Improvement: {freq_improvement:.2f}%")
    
    volt_improvement = (
        (metrics_no_forecast.voltage.std_deviation - metrics_forecast.voltage.std_deviation)
        / metrics_no_forecast.voltage.std_deviation * 100
    )
    print(f"Voltage Std Deviation Improvement:   {volt_improvement:.2f}%")
    
    if metrics_no_forecast.battery.num_cycles > 0:
        cycle_improvement = (
            (metrics_no_forecast.battery.num_cycles - metrics_forecast.battery.num_cycles)
            / metrics_no_forecast.battery.num_cycles * 100
        )
        print(f"Battery Cycles Reduction:            {cycle_improvement:.2f}%")
    else:
        print(f"Battery Cycles Reduction:            N/A (no cycles detected)")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    print("StabilityAnalyzer Example")
    print("="*60)
    
    # Run comparison
    compare_scenarios()
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
