"""
Example usage of microgrid component models.

Demonstrates how to use PVArray, Battery, and Inverter components
for microgrid simulations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.components import PVArray, Battery, Inverter
import numpy as np


def example_pv_array():
    """Demonstrate PV array usage."""
    print("=" * 60)
    print("PV Array Example")
    print("=" * 60)
    
    # Create a 10 kW PV array
    pv = PVArray(capacity_kw=10.0, area_m2=50.0, base_efficiency=0.18)
    
    # Simulate different conditions
    conditions = [
        (1.0, 25.0, "Standard Test Conditions"),
        (0.8, 30.0, "Sunny afternoon (hot)"),
        (0.5, 20.0, "Partly cloudy"),
        (0.2, 15.0, "Overcast"),
        (0.0, 10.0, "Night time"),
    ]
    
    print(f"\nPV Array: {pv.capacity_kw} kW rated capacity")
    print(f"Panel area: {pv.area_m2} m²")
    print(f"Base efficiency: {pv.base_efficiency * 100:.1f}%")
    print(f"Temperature coefficient: {pv.temp_coefficient * 100:.2f}%/°C\n")
    
    for irradiance, temp, description in conditions:
        power = pv.calculate_output(irradiance, temp)
        print(f"{description:30s} | Irr: {irradiance:.2f} kW/m² | "
              f"Temp: {temp:5.1f}°C | Power: {power:6.2f} kW")


def example_battery():
    """Demonstrate battery usage."""
    print("\n" + "=" * 60)
    print("Battery Example")
    print("=" * 60)
    
    # Create a 5 kWh battery with 3 kW max power
    battery = Battery(
        capacity_kwh=5.0,
        max_power_kw=3.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc_kwh=2.5
    )
    
    print(f"\nBattery: {battery.capacity_kwh} kWh capacity")
    print(f"Max power: {battery.max_power_kw} kW")
    print(f"Charge efficiency: {battery.charge_efficiency * 100:.1f}%")
    print(f"Discharge efficiency: {battery.discharge_efficiency * 100:.1f}%")
    print(f"Initial SOC: {battery.get_soc():.2f} kWh ({battery.get_soc_fraction() * 100:.1f}%)\n")
    
    # Simulate charging
    print("Charging at 2 kW for 30 minutes:")
    actual_power = battery.charge(power_kw=2.0, dt_hours=0.5)
    print(f"  Actual charge power: {actual_power:.2f} kW")
    print(f"  New SOC: {battery.get_soc():.2f} kWh ({battery.get_soc_fraction() * 100:.1f}%)")
    
    # Simulate discharging
    print("\nDischarging at 1.5 kW for 1 hour:")
    actual_power = battery.discharge(power_kw=1.5, dt_hours=1.0)
    print(f"  Actual discharge power: {actual_power:.2f} kW")
    print(f"  New SOC: {battery.get_soc():.2f} kWh ({battery.get_soc_fraction() * 100:.1f}%)")
    
    # Try to exceed power limit
    print("\nAttempting to charge at 5 kW (exceeds max power):")
    actual_power = battery.charge(power_kw=5.0, dt_hours=0.5)
    print(f"  Actual charge power: {actual_power:.2f} kW (limited to max)")
    print(f"  New SOC: {battery.get_soc():.2f} kWh ({battery.get_soc_fraction() * 100:.1f}%)")


def example_inverter():
    """Demonstrate inverter usage."""
    print("\n" + "=" * 60)
    print("Inverter Example")
    print("=" * 60)
    
    # Create a 12 kW inverter
    inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96, power_factor=0.95)
    
    print(f"\nInverter: {inverter.capacity_kw} kW rated capacity")
    print(f"Peak efficiency: {inverter.base_efficiency * 100:.1f}%")
    print(f"Power factor: {inverter.power_factor:.2f}\n")
    
    # Test at different loading levels
    dc_powers = [0.6, 2.4, 6.0, 9.6, 12.0, 14.4]  # 5%, 20%, 50%, 80%, 100%, 120%
    
    print(f"{'DC Power (kW)':>15} | {'Loading':>8} | {'AC Power (kW)':>15} | "
          f"{'Efficiency':>10} | {'Reactive (kVAR)':>16}")
    print("-" * 85)
    
    for dc_power in dc_powers:
        ac_power, reactive_power = inverter.convert(dc_power)
        loading = (dc_power / inverter.capacity_kw) * 100
        efficiency = (ac_power / dc_power * 100) if dc_power > 0 else 0
        
        print(f"{dc_power:15.2f} | {loading:7.1f}% | {ac_power:15.2f} | "
              f"{efficiency:9.2f}% | {reactive_power:16.2f}")


def example_integrated_simulation():
    """Demonstrate integrated component usage."""
    print("\n" + "=" * 60)
    print("Integrated Microgrid Simulation Example")
    print("=" * 60)
    
    # Create components
    pv = PVArray(capacity_kw=10.0)
    battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=2.5)
    inverter = Inverter(capacity_kw=12.0)
    
    # Simulate a few time steps
    print("\nSimulating 5 time steps (15-minute intervals):\n")
    
    # Weather and load data
    weather_data = [
        (0.8, 28.0),  # Morning
        (0.9, 30.0),  # Mid-morning
        (1.0, 32.0),  # Noon
        (0.85, 31.0), # Afternoon
        (0.6, 28.0),  # Late afternoon
    ]
    
    load_profile = [0.6, 0.7, 0.8, 0.75, 0.65]  # kW
    
    print(f"{'Time':>6} | {'Irr':>6} | {'Temp':>6} | {'PV':>7} | {'Load':>7} | "
          f"{'Batt':>7} | {'SOC':>7} | {'Grid':>7}")
    print("-" * 75)
    
    for i, ((irradiance, temp), load) in enumerate(zip(weather_data, load_profile)):
        # Calculate PV output
        pv_dc = pv.calculate_output(irradiance, temp)
        
        # Convert through inverter
        pv_ac, _ = inverter.convert(pv_dc)
        
        # Calculate power balance
        power_balance = pv_ac - load
        
        # Battery dispatch
        if power_balance > 0:
            # Excess PV: charge battery
            battery_power = battery.charge(power_kw=power_balance, dt_hours=0.25)
            grid_power = power_balance - battery_power
        else:
            # Deficit: discharge battery
            battery_power = -battery.discharge(power_kw=-power_balance, dt_hours=0.25)
            grid_power = power_balance - battery_power
        
        soc = battery.get_soc()
        
        print(f"{i*15:4d}min | {irradiance:5.2f} | {temp:5.1f}°C | "
              f"{pv_ac:6.2f}kW | {load:6.2f}kW | {battery_power:+6.2f}kW | "
              f"{soc:6.2f}kWh | {grid_power:+6.2f}kW")
    
    print("\nLegend:")
    print("  Batt: Positive = charging, Negative = discharging")
    print("  Grid: Positive = exporting, Negative = importing")


if __name__ == "__main__":
    example_pv_array()
    example_battery()
    example_inverter()
    example_integrated_simulation()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
