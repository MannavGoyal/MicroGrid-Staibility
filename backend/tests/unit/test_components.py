"""
Unit tests for microgrid component models.

Tests PVArray, Battery, and Inverter component behavior.
"""

import pytest
import numpy as np
from backend.src.simulation.components import PVArray, Battery, Inverter


class TestPVArray:
    """Test PVArray component."""
    
    def test_initialization(self):
        """Test PVArray initialization with default parameters."""
        pv = PVArray(capacity_kw=10.0)
        assert pv.capacity_kw == 10.0
        assert pv.area_m2 == 50.0
        assert pv.base_efficiency == 0.18
        assert pv.temp_coefficient == -0.004
    
    def test_calculate_output_standard_conditions(self):
        """Test PV output at standard test conditions (1 kW/m², 25°C)."""
        pv = PVArray(capacity_kw=10.0, area_m2=50.0, base_efficiency=0.18)
        
        # At STC: 1.0 kW/m² irradiance, 25°C temperature
        power = pv.calculate_output(irradiance_kw_m2=1.0, temperature_celsius=25.0)
        
        # Expected: 1.0 * 50 * 0.18 * 1.0 = 9.0 kW
        assert abs(power - 9.0) < 0.01
    
    def test_calculate_output_temperature_derating(self):
        """Test temperature coefficient effect on output."""
        pv = PVArray(capacity_kw=10.0, area_m2=50.0, base_efficiency=0.18, 
                     temp_coefficient=-0.004)
        
        # At 35°C (10°C above reference)
        power_hot = pv.calculate_output(irradiance_kw_m2=1.0, temperature_celsius=35.0)
        
        # Expected temp factor: 1 + (-0.004) * (35 - 25) = 0.96
        # Power: 1.0 * 50 * 0.18 * 0.96 = 8.64 kW
        assert abs(power_hot - 8.64) < 0.01
        
        # At 15°C (10°C below reference)
        power_cold = pv.calculate_output(irradiance_kw_m2=1.0, temperature_celsius=15.0)
        
        # Expected temp factor: 1 + (-0.004) * (15 - 25) = 1.04
        # Power: 1.0 * 50 * 0.18 * 1.04 = 9.36 kW
        assert abs(power_cold - 9.36) < 0.01
    
    def test_calculate_output_low_irradiance(self):
        """Test output at low irradiance."""
        pv = PVArray(capacity_kw=10.0, area_m2=50.0, base_efficiency=0.18)
        
        # At 0.3 kW/m² irradiance
        power = pv.calculate_output(irradiance_kw_m2=0.3, temperature_celsius=25.0)
        
        # Expected: 0.3 * 50 * 0.18 * 1.0 = 2.7 kW
        assert abs(power - 2.7) < 0.01
    
    def test_calculate_output_zero_irradiance(self):
        """Test output at night (zero irradiance)."""
        pv = PVArray(capacity_kw=10.0)
        power = pv.calculate_output(irradiance_kw_m2=0.0, temperature_celsius=20.0)
        assert power == 0.0
    
    def test_calculate_output_capacity_limit(self):
        """Test that output is capped at rated capacity."""
        pv = PVArray(capacity_kw=10.0, area_m2=100.0, base_efficiency=0.20)
        
        # High irradiance that would exceed capacity
        power = pv.calculate_output(irradiance_kw_m2=1.2, temperature_celsius=15.0)
        
        # Should be capped at 10.0 kW
        assert power <= 10.0
    
    def test_calculate_output_non_negative(self):
        """Test that output is always non-negative."""
        pv = PVArray(capacity_kw=10.0)
        
        # Extreme temperature that might cause negative temp factor
        power = pv.calculate_output(irradiance_kw_m2=0.5, temperature_celsius=300.0)
        assert power >= 0.0


class TestBattery:
    """Test Battery component."""
    
    def test_initialization_default_soc(self):
        """Test Battery initialization with default SOC."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0)
        
        assert battery.capacity_kwh == 5.0
        assert battery.max_power_kw == 3.0
        assert battery.charge_efficiency == 0.95
        assert battery.discharge_efficiency == 0.95
        assert battery.get_soc() == 2.5  # 50% of 5.0 kWh
        assert battery.get_soc_fraction() == 0.5
    
    def test_initialization_custom_soc(self):
        """Test Battery initialization with custom SOC."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=3.0)
        assert battery.get_soc() == 3.0
        assert battery.get_soc_fraction() == 0.6
    
    def test_charge_normal(self):
        """Test normal charging operation."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=2.0)
        
        # Charge at 2 kW for 0.5 hours (30 minutes)
        actual_power = battery.charge(power_kw=2.0, dt_hours=0.5)
        
        # Energy added: 2.0 * 0.5 * 0.95 = 0.95 kWh
        # New SOC: 2.0 + 0.95 = 2.95 kWh
        assert actual_power == 2.0
        assert abs(battery.get_soc() - 2.95) < 0.01
    
    def test_charge_power_limit(self):
        """Test charging respects max power limit."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=2.0)
        
        # Try to charge at 5 kW (exceeds max_power_kw)
        actual_power = battery.charge(power_kw=5.0, dt_hours=0.5)
        
        # Should be limited to 3.0 kW
        assert actual_power == 3.0
    
    def test_charge_soc_limit(self):
        """Test charging respects SOC max limit."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, 
                         initial_soc_kwh=4.4, soc_max=0.9)
        
        # SOC max is 0.9 * 5.0 = 4.5 kWh
        # Available capacity: 4.5 - 4.4 = 0.1 kWh
        # Try to charge at 2 kW for 1 hour
        actual_power = battery.charge(power_kw=2.0, dt_hours=1.0)
        
        # Should be limited by available capacity
        # 0.1 kWh / (1.0 * 0.95) ≈ 0.105 kW
        assert actual_power < 0.2
        assert abs(battery.get_soc() - 4.5) < 0.01
    
    def test_discharge_normal(self):
        """Test normal discharging operation."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=3.0)
        
        # Discharge at 2 kW for 0.5 hours (30 minutes)
        actual_power = battery.discharge(power_kw=2.0, dt_hours=0.5)
        
        # Energy removed: 2.0 * 0.5 / 0.95 ≈ 1.053 kWh
        # New SOC: 3.0 - 1.053 ≈ 1.947 kWh
        assert actual_power == 2.0
        assert abs(battery.get_soc() - 1.947) < 0.01
    
    def test_discharge_power_limit(self):
        """Test discharging respects max power limit."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, initial_soc_kwh=3.0)
        
        # Try to discharge at 5 kW (exceeds max_power_kw)
        actual_power = battery.discharge(power_kw=5.0, dt_hours=0.5)
        
        # Should be limited to 3.0 kW
        assert actual_power == 3.0
    
    def test_discharge_soc_limit(self):
        """Test discharging respects SOC min limit."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, 
                         initial_soc_kwh=0.6, soc_min=0.1)
        
        # SOC min is 0.1 * 5.0 = 0.5 kWh
        # Available energy: 0.6 - 0.5 = 0.1 kWh
        # Try to discharge at 2 kW for 1 hour
        actual_power = battery.discharge(power_kw=2.0, dt_hours=1.0)
        
        # Should be limited by available energy
        # 0.1 * 0.95 / 1.0 = 0.095 kW
        assert actual_power < 0.2
        assert abs(battery.get_soc() - 0.5) < 0.01
    
    def test_charge_discharge_efficiency(self):
        """Test that efficiency losses are applied correctly."""
        battery = Battery(capacity_kwh=5.0, max_power_kw=3.0, 
                         initial_soc_kwh=2.5,
                         charge_efficiency=0.9, discharge_efficiency=0.9)
        
        initial_soc = battery.get_soc()
        
        # Charge 1 kW for 1 hour
        battery.charge(power_kw=1.0, dt_hours=1.0)
        after_charge = battery.get_soc()
        
        # Energy added: 1.0 * 1.0 * 0.9 = 0.9 kWh
        assert abs(after_charge - initial_soc - 0.9) < 0.01
        
        # Discharge 1 kW for 1 hour
        battery.discharge(power_kw=1.0, dt_hours=1.0)
        after_discharge = battery.get_soc()
        
        # Energy removed: 1.0 * 1.0 / 0.9 ≈ 1.111 kWh
        assert abs(after_discharge - after_charge + 1.111) < 0.01


class TestInverter:
    """Test Inverter component."""
    
    def test_initialization(self):
        """Test Inverter initialization."""
        inverter = Inverter(capacity_kw=12.0)
        assert inverter.capacity_kw == 12.0
        assert inverter.base_efficiency == 0.96
        assert inverter.power_factor == 0.95
    
    def test_convert_zero_power(self):
        """Test inverter with zero input power."""
        inverter = Inverter(capacity_kw=12.0)
        ac_power, reactive_power = inverter.convert(dc_power_kw=0.0)
        
        assert ac_power == 0.0
        assert reactive_power == 0.0
    
    def test_convert_negative_power(self):
        """Test inverter with negative input power."""
        inverter = Inverter(capacity_kw=12.0)
        ac_power, reactive_power = inverter.convert(dc_power_kw=-5.0)
        
        assert ac_power == 0.0
        assert reactive_power == 0.0
    
    def test_convert_rated_power(self):
        """Test inverter at rated power (peak efficiency)."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # At rated power (12 kW DC input)
        ac_power, reactive_power = inverter.convert(dc_power_kw=12.0)
        
        # Should operate at peak efficiency
        expected_ac = 12.0 * 0.96
        assert abs(ac_power - expected_ac) < 0.1
        
        # Reactive power: P * tan(acos(0.95))
        expected_reactive = expected_ac * np.tan(np.arccos(0.95))
        assert abs(reactive_power - expected_reactive) < 0.1
    
    def test_convert_low_loading(self):
        """Test inverter at low loading (reduced efficiency)."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # At 5% loading (0.6 kW)
        ac_power, reactive_power = inverter.convert(dc_power_kw=0.6)
        
        # Efficiency should be reduced (around 85-90%)
        efficiency = ac_power / 0.6
        assert 0.80 < efficiency < 0.92
    
    def test_convert_medium_loading(self):
        """Test inverter at medium loading."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # At 30% loading (3.6 kW)
        ac_power, reactive_power = inverter.convert(dc_power_kw=3.6)
        
        # Efficiency should be between low and peak
        efficiency = ac_power / 3.6
        assert 0.88 < efficiency < 0.96
    
    def test_convert_high_loading(self):
        """Test inverter at high loading (peak efficiency)."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # At 80% loading (9.6 kW)
        ac_power, reactive_power = inverter.convert(dc_power_kw=9.6)
        
        # Should be at peak efficiency
        expected_ac = 9.6 * 0.96
        assert abs(ac_power - expected_ac) < 0.1
    
    def test_convert_overloading(self):
        """Test inverter overloading (reduced efficiency)."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # At 120% loading (14.4 kW)
        ac_power, reactive_power = inverter.convert(dc_power_kw=14.4)
        
        # Efficiency should be reduced
        efficiency = ac_power / 14.4
        assert efficiency < 0.96
        
        # Output should not exceed capacity
        assert ac_power <= 12.0
    
    def test_convert_capacity_limit(self):
        """Test that AC output is capped at inverter capacity."""
        inverter = Inverter(capacity_kw=12.0)
        
        # Very high DC input
        ac_power, reactive_power = inverter.convert(dc_power_kw=20.0)
        
        # Should be capped at 12.0 kW
        assert ac_power <= 12.0
    
    def test_convert_reactive_power_calculation(self):
        """Test reactive power calculation."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96, power_factor=0.95)
        
        ac_power, reactive_power = inverter.convert(dc_power_kw=10.0)
        
        # Reactive power should be positive
        assert reactive_power > 0
        
        # Q = P * tan(acos(pf))
        expected_reactive = ac_power * np.tan(np.arccos(0.95))
        assert abs(reactive_power - expected_reactive) < 0.01
    
    def test_efficiency_curve_consistency(self):
        """Test that efficiency curve is monotonic in normal operating range."""
        inverter = Inverter(capacity_kw=12.0, base_efficiency=0.96)
        
        # Test efficiency at various loading levels
        loadings = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        efficiencies = []
        
        for loading in loadings:
            dc_power = loading * 12.0
            ac_power, _ = inverter.convert(dc_power_kw=dc_power)
            efficiency = ac_power / dc_power if dc_power > 0 else 0
            efficiencies.append(efficiency)
        
        # Efficiency should generally increase up to rated power
        for i in range(len(efficiencies) - 1):
            # Allow small variations but check general trend
            if loadings[i] < 1.0:
                assert efficiencies[i+1] >= efficiencies[i] - 0.05
