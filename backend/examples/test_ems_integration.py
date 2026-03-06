"""
Integration example: EMSController with MicrogridSimulator.

This script demonstrates how to use the EMS controller with the microgrid
simulator to optimize battery dispatch and improve stability.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from backend.src.simulation.ems_controller import EMSController, EMSConfig, SystemState
from backend.src.simulation.simulator import MicrogridSimulator
from backend.src.config.schemas import MicrogridConfig, MicrogridMode


def create_sample_data(n_steps: int = 96):
    """
    Create sample data for 24 hours with 15-minute resolution.
    
    Args:
        n_steps: Number of timesteps (default: 96 for 24 hours at 15-min intervals)
    
    Returns:
        Tuple of (pv_forecast, actual_pv, load_profile, irradiance, temperature)
    """
    # Time array (hours)
    time = np.arange(n_steps) * 0.25  # 15-minute intervals
    
    # PV generation: sinusoidal pattern with some noise
    base_pv = 8.0 * np.maximum(0, np.sin(np.pi * (time - 6) / 12))
    actual_pv = base_pv + np.random.normal(0, 0.3, n_steps)
    actual_pv = np.maximum(0, actual_pv)
    
    # PV forecast: similar but with forecast error
    pv_forecast = base_pv + np.random.normal(0, 0.5, n_steps)
    pv_forecast = np.maximum(0, pv_forecast)
    
    # Load profile: relatively constant with daily variation
    load_profile = 3.0 + 1.5 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 0.2, n_steps)
    load_profile = np.maximum(0.5, load_profile)
    
    # Weather data (for PV array model)
    irradiance = base_pv / 8.0  # Normalized to 0-1 kW/m²
    temperature = 25.0 + 10.0 * np.sin(2 * np.pi * time / 24)
    
    return pv_forecast, actual_pv, load_profile, irradiance, temperature


def run_simulation_with_ems(strategy: str = 'mpc'):
    """
    Run microgrid simulation with EMS controller.
    
    Args:
        strategy: EMS control strategy ('mpc', 'rule_based', 'reactive')
    
    Returns:
        Simulation result
    """
    print(f"\nRunning simulation with {strategy.upper()} control...")
    
    # Create microgrid configuration
    microgrid_config = MicrogridConfig(
        mode=MicrogridMode.ISLANDED,
        pv_capacity_kw=10.0,
        battery_capacity_kwh=10.0,
        battery_power_kw=5.0,
        inverter_capacity_kw=12.0,
        initial_soc_kwh=5.0  # Start at 50% SOC
    )
    
    # Create EMS configuration
    ems_config = EMSConfig(
        battery_capacity_kwh=microgrid_config.battery_capacity_kwh,
        battery_power_kw=microgrid_config.battery_power_kw,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9
    )
    
    # Create simulator and controller
    simulator = MicrogridSimulator(microgrid_config)
    controller = EMSController(ems_config, strategy=strategy)
    
    # Generate sample data
    pv_forecast, actual_pv, load_profile, irradiance, temperature = create_sample_data(96)
    
    # Compute battery dispatch using EMS controller
    if strategy == 'reactive':
        # Reactive control doesn't use forecast
        battery_dispatch = None
    else:
        # Use EMS to compute optimal dispatch
        initial_state = SystemState(
            soc_kwh=microgrid_config.initial_soc_kwh,
            pv_power=actual_pv[0],
            load_power=load_profile[0],
            frequency_hz=60.0,
            voltage_pu=1.0
        )
        
        battery_dispatch = controller.compute_dispatch(
            initial_state,
            pv_forecast,
            load_profile,
            horizon=len(pv_forecast),
            timestep_hours=0.25  # 15 minutes
        )
    
    # Run simulation
    result = simulator.simulate(
        pv_forecast=pv_forecast,
        actual_pv=actual_pv,
        load_profile=load_profile,
        timestep_seconds=900,  # 15 minutes
        battery_dispatch=battery_dispatch
    )
    
    return result, pv_forecast, actual_pv, load_profile


def compare_strategies():
    """Compare different EMS strategies in simulation."""
    print("=" * 70)
    print("EMS CONTROLLER + MICROGRID SIMULATOR INTEGRATION")
    print("=" * 70)
    
    strategies = ['reactive', 'rule_based', 'mpc']
    results = {}
    
    for strategy in strategies:
        result, pv_forecast, actual_pv, load_profile = run_simulation_with_ems(strategy)
        results[strategy] = result
        
        # Calculate metrics
        freq_mae = np.mean(np.abs(result.frequency_deviation))
        freq_std = np.std(result.frequency_deviation)
        freq_max = np.max(np.abs(result.frequency_deviation))
        
        volt_mae = np.mean(np.abs(result.voltage_deviation))
        volt_std = np.std(result.voltage_deviation)
        
        soc_range = np.max(result.battery_soc) - np.min(result.battery_soc)
        battery_throughput = np.sum(np.abs(result.battery_power)) * 0.25  # kWh
        
        print(f"\n{strategy.upper()} Strategy Results:")
        print("-" * 50)
        print(f"  Frequency Deviation:")
        print(f"    MAE:     {freq_mae:.4f} Hz")
        print(f"    Std Dev: {freq_std:.4f} Hz")
        print(f"    Max:     {freq_max:.4f} Hz")
        print(f"  Voltage Deviation:")
        print(f"    MAE:     {volt_mae:.4f} %")
        print(f"    Std Dev: {volt_std:.4f} %")
        print(f"  Battery Metrics:")
        print(f"    SOC Range:   {soc_range:.2f} kWh")
        print(f"    Throughput:  {battery_throughput:.2f} kWh")
        print(f"    Final SOC:   {result.battery_soc[-1]:.2f} kWh")
    
    # Visualize comparison
    visualize_results(results, actual_pv, load_profile)
    
    return results


def visualize_results(results, actual_pv, load_profile):
    """
    Visualize simulation results for all strategies.
    
    Args:
        results: Dictionary of simulation results for each strategy
        actual_pv: Actual PV power array
        load_profile: Load power array
    """
    time_hours = np.arange(len(actual_pv)) * 0.25
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: PV and Load
    ax1 = axes[0]
    ax1.plot(time_hours, actual_pv, label='PV Generation', color='orange', linewidth=2)
    ax1.plot(time_hours, load_profile, label='Load Demand', color='blue', linewidth=2)
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('PV Generation and Load Demand')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery SOC
    ax2 = axes[1]
    for strategy, result in results.items():
        ax2.plot(time_hours, result.battery_soc, 
                label=strategy.replace('_', ' ').title(), linewidth=2)
    ax2.set_ylabel('SOC (kWh)')
    ax2.set_title('Battery State of Charge')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency Deviation
    ax3 = axes[2]
    for strategy, result in results.items():
        ax3.plot(time_hours, result.frequency_deviation,
                label=strategy.replace('_', ' ').title(), linewidth=1.5, alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=0.5, color='r', linestyle=':', alpha=0.3, label='±0.5 Hz limit')
    ax3.axhline(y=-0.5, color='r', linestyle=':', alpha=0.3)
    ax3.set_ylabel('Frequency Deviation (Hz)')
    ax3.set_title('Frequency Stability')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Battery Power
    ax4 = axes[3]
    for strategy, result in results.items():
        ax4.plot(time_hours, result.battery_power,
                label=strategy.replace('_', ' ').title(), linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Battery Power (kW)')
    ax4.set_title('Battery Dispatch (Positive = Charge, Negative = Discharge)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ems_simulator_integration.png', dpi=150, bbox_inches='tight')
    print(f"\n{'=' * 70}")
    print("Visualization saved to: ems_simulator_integration.png")
    print(f"{'=' * 70}")
    plt.close()


def calculate_improvements(results):
    """Calculate improvements relative to reactive baseline."""
    print("\n" + "=" * 70)
    print("IMPROVEMENTS RELATIVE TO REACTIVE BASELINE")
    print("=" * 70)
    
    baseline = results['reactive']
    baseline_freq_mae = np.mean(np.abs(baseline.frequency_deviation))
    baseline_freq_std = np.std(baseline.frequency_deviation)
    
    for strategy in ['rule_based', 'mpc']:
        result = results[strategy]
        freq_mae = np.mean(np.abs(result.frequency_deviation))
        freq_std = np.std(result.frequency_deviation)
        
        freq_mae_improvement = (baseline_freq_mae - freq_mae) / baseline_freq_mae * 100
        freq_std_improvement = (baseline_freq_std - freq_std) / baseline_freq_std * 100
        
        print(f"\n{strategy.upper()} vs REACTIVE:")
        print(f"  Frequency MAE improvement:    {freq_mae_improvement:+.1f}%")
        print(f"  Frequency Std Dev improvement: {freq_std_improvement:+.1f}%")


if __name__ == '__main__':
    # Run comparison
    results = compare_strategies()
    
    # Calculate improvements
    calculate_improvements(results)
    
    print("\n" + "=" * 70)
    print("Integration test completed successfully!")
    print("=" * 70)
