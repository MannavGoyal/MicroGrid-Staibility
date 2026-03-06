"""
Example script demonstrating MicrogridSimulator usage.

This script shows how to:
1. Configure a microgrid
2. Create test data (PV, load profiles)
3. Run simulations in different modes
4. Analyze results
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import numpy as np
import matplotlib.pyplot as plt
from src.simulation.simulator import MicrogridSimulator
from src.config.schemas import MicrogridConfig, MicrogridMode


def create_test_data(n_steps: int = 144) -> tuple:
    """
    Create synthetic test data for 24 hours at 10-minute intervals.
    
    Args:
        n_steps: Number of timesteps (default: 144 = 24 hours at 10-min intervals)
    
    Returns:
        Tuple of (pv_power, load_profile, timestamps_hours)
    """
    # Time array (hours)
    time_hours = np.linspace(0, 24, n_steps)
    
    # PV generation: sinusoidal pattern peaking at noon
    # Zero at night (before 6am and after 6pm)
    pv_power = np.zeros(n_steps)
    for i, t in enumerate(time_hours):
        if 6 <= t <= 18:  # Daylight hours
            # Peak at noon (t=12)
            pv_power[i] = 8.0 * np.sin(np.pi * (t - 6) / 12)
    
    # Add some variability (clouds)
    pv_power += np.random.normal(0, 0.3, n_steps)
    pv_power = np.maximum(0, pv_power)  # Ensure non-negative
    
    # Load profile: typical residential pattern
    # Higher in morning (7-9am) and evening (6-10pm)
    load_profile = np.zeros(n_steps)
    for i, t in enumerate(time_hours):
        if 0 <= t < 6:
            # Night: low load
            load_profile[i] = 2.0 + np.random.normal(0, 0.2)
        elif 6 <= t < 9:
            # Morning: increasing load
            load_profile[i] = 3.0 + 2.0 * (t - 6) / 3 + np.random.normal(0, 0.3)
        elif 9 <= t < 17:
            # Day: moderate load
            load_profile[i] = 4.0 + np.random.normal(0, 0.3)
        elif 17 <= t < 22:
            # Evening: high load
            load_profile[i] = 5.0 + 1.0 * np.sin(np.pi * (t - 17) / 5) + np.random.normal(0, 0.3)
        else:
            # Late night: decreasing load
            load_profile[i] = 3.0 - 1.0 * (t - 22) / 2 + np.random.normal(0, 0.2)
    
    load_profile = np.maximum(1.0, load_profile)  # Minimum 1 kW load
    
    return pv_power, load_profile, time_hours


def run_islanded_simulation():
    """Run simulation in islanded mode."""
    print("=" * 60)
    print("ISLANDED MODE SIMULATION")
    print("=" * 60)
    
    # Configure microgrid
    config = MicrogridConfig(
        mode=MicrogridMode.ISLANDED,
        pv_capacity_kw=10.0,
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        inverter_capacity_kw=12.0,
        initial_soc_kwh=2.5
    )
    
    # Create simulator
    sim = MicrogridSimulator(config)
    
    # Generate test data
    pv_power, load_profile, time_hours = create_test_data(n_steps=144)
    pv_forecast = pv_power.copy()  # Perfect forecast for this example
    
    # Run simulation
    result = sim.simulate(
        pv_forecast=pv_forecast,
        actual_pv=pv_power,
        load_profile=load_profile,
        timestep_seconds=600  # 10 minutes
    )
    
    # Print summary statistics
    print(f"\nSimulation Summary:")
    print(f"  Duration: {len(result.timestamps)} timesteps ({len(result.timestamps) * 10 / 60:.1f} hours)")
    print(f"  Average PV: {np.mean(result.pv_power):.2f} kW")
    print(f"  Average Load: {np.mean(result.load_power):.2f} kW")
    print(f"  Average Battery Power: {np.mean(result.battery_power):.2f} kW")
    print(f"\nBattery Statistics:")
    print(f"  Initial SOC: {result.battery_soc[0]:.2f} kWh")
    print(f"  Final SOC: {result.battery_soc[-1]:.2f} kWh")
    print(f"  Min SOC: {np.min(result.battery_soc):.2f} kWh")
    print(f"  Max SOC: {np.max(result.battery_soc):.2f} kWh")
    print(f"\nFrequency Stability:")
    print(f"  Mean Deviation: {np.mean(np.abs(result.frequency_deviation)):.3f} Hz")
    print(f"  Std Deviation: {np.std(result.frequency_deviation):.3f} Hz")
    print(f"  Max Deviation: {np.max(np.abs(result.frequency_deviation)):.3f} Hz")
    print(f"\nVoltage Stability:")
    print(f"  Mean Deviation: {np.mean(np.abs(result.voltage_deviation)):.2f} %")
    print(f"  Std Deviation: {np.std(result.voltage_deviation):.2f} %")
    print(f"  Max Deviation: {np.max(np.abs(result.voltage_deviation)):.2f} %")
    
    return result, time_hours


def run_grid_connected_simulation():
    """Run simulation in grid-connected mode."""
    print("\n" + "=" * 60)
    print("GRID-CONNECTED MODE SIMULATION")
    print("=" * 60)
    
    # Configure microgrid
    config = MicrogridConfig(
        mode=MicrogridMode.GRID_CONNECTED,
        pv_capacity_kw=10.0,
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        inverter_capacity_kw=12.0,
        initial_soc_kwh=2.5
    )
    
    # Create simulator
    sim = MicrogridSimulator(config)
    
    # Generate test data
    pv_power, load_profile, time_hours = create_test_data(n_steps=144)
    pv_forecast = pv_power.copy()
    
    # Run simulation
    result = sim.simulate(
        pv_forecast=pv_forecast,
        actual_pv=pv_power,
        load_profile=load_profile,
        timestep_seconds=600
    )
    
    # Print summary statistics
    print(f"\nSimulation Summary:")
    print(f"  Duration: {len(result.timestamps)} timesteps ({len(result.timestamps) * 10 / 60:.1f} hours)")
    print(f"  Average Grid Power: {np.mean(result.grid_power):.2f} kW")
    print(f"  Total Grid Import: {np.sum(np.maximum(0, result.grid_power)) * 10/60:.2f} kWh")
    print(f"  Total Grid Export: {np.sum(np.maximum(0, -result.grid_power)) * 10/60:.2f} kWh")
    print(f"\nFrequency Stability:")
    print(f"  Mean Deviation: {np.mean(np.abs(result.frequency_deviation)):.3f} Hz")
    print(f"  Max Deviation: {np.max(np.abs(result.frequency_deviation)):.3f} Hz")
    
    return result, time_hours


def plot_results(result_islanded, result_grid, time_hours):
    """Plot simulation results."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Microgrid Simulation Results: Islanded vs Grid-Connected', fontsize=14, fontweight='bold')
    
    # Plot 1: Power flows (Islanded)
    ax = axes[0, 0]
    ax.plot(time_hours, result_islanded.pv_power, label='PV Power', linewidth=2)
    ax.plot(time_hours, result_islanded.load_power, label='Load', linewidth=2)
    ax.plot(time_hours, result_islanded.battery_power, label='Battery Power', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Flows - Islanded Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Power flows (Grid-Connected)
    ax = axes[0, 1]
    ax.plot(time_hours, result_grid.pv_power, label='PV Power', linewidth=2)
    ax.plot(time_hours, result_grid.load_power, label='Load', linewidth=2)
    ax.plot(time_hours, result_grid.battery_power, label='Battery Power', linewidth=2)
    ax.plot(time_hours, result_grid.grid_power, label='Grid Power', linewidth=2, linestyle='--')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Flows - Grid-Connected Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Battery SOC (Islanded)
    ax = axes[1, 0]
    ax.plot(time_hours, result_islanded.battery_soc, linewidth=2, color='green')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Min SOC (10%)')
    ax.axhline(y=4.5, color='r', linestyle='--', alpha=0.5, label='Max SOC (90%)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('SOC (kWh)')
    ax.set_title('Battery State of Charge - Islanded')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Battery SOC (Grid-Connected)
    ax = axes[1, 1]
    ax.plot(time_hours, result_grid.battery_soc, linewidth=2, color='green')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Min SOC (10%)')
    ax.axhline(y=4.5, color='r', linestyle='--', alpha=0.5, label='Max SOC (90%)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('SOC (kWh)')
    ax.set_title('Battery State of Charge - Grid-Connected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Frequency deviation (Islanded)
    ax = axes[2, 0]
    ax.plot(time_hours, result_islanded.frequency_deviation, linewidth=2, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit (±2 Hz)')
    ax.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Frequency Deviation (Hz)')
    ax.set_title('Frequency Stability - Islanded')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Frequency deviation (Grid-Connected)
    ax = axes[2, 1]
    ax.plot(time_hours, result_grid.frequency_deviation, linewidth=2, color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit (±2 Hz)')
    ax.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Frequency Deviation (Hz)')
    ax.set_title('Frequency Stability - Grid-Connected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('microgrid_simulation_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'microgrid_simulation_results.png'")
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("MICROGRID SIMULATOR DEMONSTRATION")
    print("=" * 60)
    print("\nThis example demonstrates the MicrogridSimulator with:")
    print("  - 10 kW PV array")
    print("  - 5 kWh / 3 kW battery")
    print("  - 12 kW inverter")
    print("  - 24-hour simulation with realistic PV and load profiles")
    
    # Run simulations
    result_islanded, time_hours = run_islanded_simulation()
    result_grid, _ = run_grid_connected_simulation()
    
    # Plot results
    print("\n" + "=" * 60)
    print("Generating plots...")
    plot_results(result_islanded, result_grid, time_hours)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
