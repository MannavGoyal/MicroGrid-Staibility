"""
Example script demonstrating EMS controller usage.

This script shows how to use different control strategies and visualize results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from backend.src.simulation.ems_controller import EMSController, EMSConfig, SystemState


def create_sample_forecasts(n_steps: int = 24):
    """
    Create sample PV and load forecasts for testing.
    
    Args:
        n_steps: Number of timesteps (default: 24 for 24 hours with 1-hour steps)
    
    Returns:
        Tuple of (pv_forecast, load_forecast)
    """
    # Create time array (hours)
    time = np.arange(n_steps)
    
    # PV forecast: sinusoidal pattern (solar generation during day)
    # Peak at noon (hour 12), zero at night
    pv_forecast = 5.0 * np.maximum(0, np.sin(np.pi * (time - 6) / 12))
    
    # Load forecast: relatively constant with small variations
    load_forecast = 2.0 + 0.5 * np.sin(2 * np.pi * time / 24)
    
    return pv_forecast, load_forecast


def compare_control_strategies():
    """Compare different EMS control strategies."""
    print("=" * 60)
    print("EMS Controller Comparison")
    print("=" * 60)
    
    # Create EMS configuration
    config = EMSConfig(
        battery_capacity_kwh=10.0,
        battery_power_kw=5.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9
    )
    
    # Create forecasts
    horizon = 24
    pv_forecast, load_forecast = create_sample_forecasts(horizon)
    
    # Initial system state
    initial_state = SystemState(
        soc_kwh=5.0,  # 50% SOC
        pv_power=pv_forecast[0],
        load_power=load_forecast[0],
        frequency_hz=60.0,
        voltage_pu=1.0
    )
    
    # Test each control strategy
    strategies = ['reactive', 'rule_based', 'mpc']
    results = {}
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Control Strategy:")
        print("-" * 40)
        
        controller = EMSController(config, strategy=strategy)
        
        # Compute dispatch
        dispatch = controller.compute_dispatch(
            initial_state,
            pv_forecast,
            load_forecast,
            horizon,
            timestep_hours=1.0
        )
        
        # Calculate power imbalance with and without battery
        imbalance_without_battery = pv_forecast - load_forecast
        imbalance_with_battery = pv_forecast + dispatch - load_forecast
        
        # Calculate metrics
        mae_without = np.mean(np.abs(imbalance_without_battery))
        mae_with = np.mean(np.abs(imbalance_with_battery))
        improvement = (mae_without - mae_with) / mae_without * 100
        
        print(f"  Power Imbalance MAE (without battery): {mae_without:.3f} kW")
        print(f"  Power Imbalance MAE (with battery):    {mae_with:.3f} kW")
        print(f"  Improvement:                            {improvement:.1f}%")
        print(f"  Max battery power used:                 {np.max(np.abs(dispatch)):.2f} kW")
        print(f"  Total battery energy throughput:        {np.sum(np.abs(dispatch)):.2f} kWh")
        
        results[strategy] = {
            'dispatch': dispatch,
            'imbalance_with': imbalance_with_battery,
            'imbalance_without': imbalance_without_battery,
            'mae_improvement': improvement
        }
    
    # Visualize results
    visualize_comparison(pv_forecast, load_forecast, results, horizon)
    
    return results


def visualize_comparison(pv_forecast, load_forecast, results, horizon):
    """
    Visualize comparison of control strategies.
    
    Args:
        pv_forecast: PV power forecast
        load_forecast: Load power forecast
        results: Dictionary of results for each strategy
        horizon: Number of timesteps
    """
    time = np.arange(horizon)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: PV and Load forecasts
    ax1 = axes[0]
    ax1.plot(time, pv_forecast, 'o-', label='PV Forecast', color='orange')
    ax1.plot(time, load_forecast, 's-', label='Load Forecast', color='blue')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('PV and Load Forecasts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery dispatch for each strategy
    ax2 = axes[1]
    for strategy, data in results.items():
        ax2.plot(time, data['dispatch'], 'o-', label=strategy.replace('_', ' ').title())
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Battery Power (kW)')
    ax2.set_title('Battery Dispatch Strategies (Positive = Charge, Negative = Discharge)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Power imbalance comparison
    ax3 = axes[2]
    ax3.plot(time, results['reactive']['imbalance_without'], 
             'x-', label='No Battery', color='red', alpha=0.5)
    for strategy, data in results.items():
        ax3.plot(time, data['imbalance_with'], 'o-', 
                label=f"{strategy.replace('_', ' ').title()}")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Power Imbalance (kW)')
    ax3.set_title('Power Imbalance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ems_controller_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'=' * 60}")
    print("Visualization saved to: ems_controller_comparison.png")
    print(f"{'=' * 60}")
    plt.close()


def test_single_step_control():
    """Test single-step control computation."""
    print("\n" + "=" * 60)
    print("Single-Step Control Test")
    print("=" * 60)
    
    config = EMSConfig(
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0
    )
    
    # Test scenarios
    scenarios = [
        ("Surplus Power", SystemState(2.5, 3.0, 1.5, 60.0, 1.0)),
        ("Deficit Power", SystemState(2.5, 1.0, 2.5, 60.0, 1.0)),
        ("Balanced Power", SystemState(2.5, 2.0, 2.0, 60.0, 1.0)),
    ]
    
    for name, state in scenarios:
        print(f"\n{name}:")
        print(f"  PV: {state.pv_power:.1f} kW, Load: {state.load_power:.1f} kW")
        print(f"  Imbalance: {state.pv_power - state.load_power:+.1f} kW")
        
        for strategy in ['reactive', 'rule_based', 'mpc']:
            controller = EMSController(config, strategy=strategy)
            dispatch = controller.compute_single_step(state)
            print(f"  {strategy:12s}: {dispatch:+.2f} kW")


def test_soc_constraints():
    """Test SOC constraint enforcement."""
    print("\n" + "=" * 60)
    print("SOC Constraint Test")
    print("=" * 60)
    
    config = EMSConfig(
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        soc_min=0.1,
        soc_max=0.9
    )
    
    # Test near max SOC with surplus power
    print("\nScenario 1: Near Max SOC with Surplus Power")
    state_high_soc = SystemState(4.4, 5.0, 1.0, 60.0, 1.0)  # 88% SOC
    print(f"  Initial SOC: {state_high_soc.soc_kwh:.2f} kWh ({state_high_soc.soc_kwh/5.0*100:.0f}%)")
    print(f"  Max SOC limit: {config.soc_max * 5.0:.2f} kWh ({config.soc_max*100:.0f}%)")
    
    controller = EMSController(config, strategy='rule_based')
    forecast = np.ones(5) * 5.0
    load_forecast = np.ones(5) * 1.0
    dispatch = controller.compute_dispatch(state_high_soc, forecast, load_forecast, 5, 1/60)
    
    print(f"  Battery dispatch: {dispatch[0]:.2f} kW (should be limited)")
    
    # Test near min SOC with deficit power
    print("\nScenario 2: Near Min SOC with Deficit Power")
    state_low_soc = SystemState(0.6, 0.5, 3.0, 60.0, 1.0)  # 12% SOC
    print(f"  Initial SOC: {state_low_soc.soc_kwh:.2f} kWh ({state_low_soc.soc_kwh/5.0*100:.0f}%)")
    print(f"  Min SOC limit: {config.soc_min * 5.0:.2f} kWh ({config.soc_min*100:.0f}%)")
    
    forecast = np.ones(5) * 0.5
    load_forecast = np.ones(5) * 3.0
    dispatch = controller.compute_dispatch(state_low_soc, forecast, load_forecast, 5, 1/60)
    
    print(f"  Battery dispatch: {dispatch[0]:.2f} kW (should be limited)")


if __name__ == '__main__':
    # Run all tests
    print("\n" + "=" * 60)
    print("EMS CONTROLLER DEMONSTRATION")
    print("=" * 60)
    
    # Compare control strategies
    results = compare_control_strategies()
    
    # Test single-step control
    test_single_step_control()
    
    # Test SOC constraints
    test_soc_constraints()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
