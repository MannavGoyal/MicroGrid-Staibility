"""
Microgrid component models for simulation.

This module implements physical models for PV arrays, batteries, and inverters
used in microgrid stability simulations.
"""

import numpy as np
from typing import Tuple


class PVArray:
    """
    Photovoltaic array model with temperature coefficient.
    
    Models PV power output based on solar irradiance, temperature, and panel efficiency.
    """
    
    def __init__(self, capacity_kw: float, area_m2: float = 50.0, 
                 base_efficiency: float = 0.18, temp_coefficient: float = -0.004):
        """
        Initialize PV array.
        
        Args:
            capacity_kw: Rated capacity in kW (at STC: 1000 W/m², 25°C)
            area_m2: Panel area in square meters
            base_efficiency: Base conversion efficiency at 25°C (default 18%)
            temp_coefficient: Temperature coefficient per °C (default -0.4%/°C)
        """
        self.capacity_kw = capacity_kw
        self.area_m2 = area_m2
        self.base_efficiency = base_efficiency
        self.temp_coefficient = temp_coefficient
        
    def calculate_output(self, irradiance_kw_m2: float, temperature_celsius: float) -> float:
        """
        Calculate PV power output based on irradiance and temperature.
        
        Formula:
            P_pv = irradiance * area * efficiency * temp_factor
            temp_factor = 1 + temp_coefficient * (T - 25)
        
        Args:
            irradiance_kw_m2: Solar irradiance in kW/m² (0 to ~1.2)
            temperature_celsius: Panel temperature in °C
            
        Returns:
            Power output in kW (non-negative, capped at capacity)
        """
        # Temperature derating factor
        temp_factor = 1.0 + self.temp_coefficient * (temperature_celsius - 25.0)
        
        # Calculate raw power output
        power_kw = irradiance_kw_m2 * self.area_m2 * self.base_efficiency * temp_factor
        
        # Ensure non-negative and cap at rated capacity
        power_kw = max(0.0, min(power_kw, self.capacity_kw))
        
        return power_kw


class Battery:
    """
    Battery Energy Storage System (BESS) model.
    
    Models battery charge/discharge dynamics with efficiency, SOC constraints,
    and power limits.
    """
    
    def __init__(self, capacity_kwh: float, max_power_kw: float,
                 charge_efficiency: float = 0.95, discharge_efficiency: float = 0.95,
                 initial_soc_kwh: float = None, soc_min: float = 0.1, soc_max: float = 0.9):
        """
        Initialize battery.
        
        Args:
            capacity_kwh: Total energy capacity in kWh
            max_power_kw: Maximum charge/discharge power in kW
            charge_efficiency: Charging efficiency (0-1)
            discharge_efficiency: Discharging efficiency (0-1)
            initial_soc_kwh: Initial state of charge in kWh (default: 50% of capacity)
            soc_min: Minimum SOC as fraction of capacity (default: 10%)
            soc_max: Maximum SOC as fraction of capacity (default: 90%)
        """
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_min_kwh = soc_min * capacity_kwh
        self.soc_max_kwh = soc_max * capacity_kwh
        
        # Initialize SOC to 50% if not specified
        if initial_soc_kwh is None:
            self.soc_kwh = 0.5 * capacity_kwh
        else:
            self.soc_kwh = max(self.soc_min_kwh, min(initial_soc_kwh, self.soc_max_kwh))
    
    def charge(self, power_kw: float, dt_hours: float) -> float:
        """
        Charge the battery.
        
        Formula:
            SOC_new = SOC + (power * dt * charge_efficiency)
        
        Args:
            power_kw: Charging power in kW (positive value)
            dt_hours: Time step in hours
            
        Returns:
            Actual power charged in kW (may be less than requested due to constraints)
        """
        # Ensure positive power
        power_kw = abs(power_kw)
        
        # Limit to max power
        power_kw = min(power_kw, self.max_power_kw)
        
        # Calculate energy that would be added
        energy_kwh = power_kw * dt_hours * self.charge_efficiency
        
        # Check SOC constraint
        available_capacity = self.soc_max_kwh - self.soc_kwh
        if energy_kwh > available_capacity:
            # Limit charging to available capacity
            energy_kwh = available_capacity
            power_kw = energy_kwh / (dt_hours * self.charge_efficiency) if dt_hours > 0 else 0.0
        
        # Update SOC
        self.soc_kwh += energy_kwh
        
        return power_kw
    
    def discharge(self, power_kw: float, dt_hours: float) -> float:
        """
        Discharge the battery.
        
        Formula:
            SOC_new = SOC - (power * dt / discharge_efficiency)
        
        Args:
            power_kw: Discharging power in kW (positive value)
            dt_hours: Time step in hours
            
        Returns:
            Actual power discharged in kW (may be less than requested due to constraints)
        """
        # Ensure positive power
        power_kw = abs(power_kw)
        
        # Limit to max power
        power_kw = min(power_kw, self.max_power_kw)
        
        # Calculate energy that would be removed (accounting for efficiency loss)
        energy_kwh = power_kw * dt_hours / self.discharge_efficiency
        
        # Check SOC constraint
        available_energy = self.soc_kwh - self.soc_min_kwh
        if energy_kwh > available_energy:
            # Limit discharging to available energy
            energy_kwh = available_energy
            power_kw = energy_kwh * self.discharge_efficiency / dt_hours if dt_hours > 0 else 0.0
        
        # Update SOC
        self.soc_kwh -= energy_kwh
        
        return power_kw
    
    def get_soc(self) -> float:
        """Get current state of charge in kWh."""
        return self.soc_kwh
    
    def get_soc_fraction(self) -> float:
        """Get current state of charge as fraction of total capacity (0-1)."""
        return self.soc_kwh / self.capacity_kwh


class Inverter:
    """
    Power inverter model with efficiency curves and reactive power capability.
    
    Models DC to AC power conversion with power-dependent efficiency and
    reactive power generation.
    """
    
    def __init__(self, capacity_kw: float, base_efficiency: float = 0.96,
                 power_factor: float = 0.95):
        """
        Initialize inverter.
        
        Args:
            capacity_kw: Rated AC power capacity in kW
            base_efficiency: Peak efficiency at rated power (default: 96%)
            power_factor: Power factor for reactive power calculation (default: 0.95)
        """
        self.capacity_kw = capacity_kw
        self.base_efficiency = base_efficiency
        self.power_factor = power_factor
    
    def _efficiency_curve(self, dc_power_kw: float) -> float:
        """
        Calculate efficiency based on loading.
        
        Efficiency curve model:
        - Low loading (< 10%): 85% efficiency
        - Medium loading (10-50%): Linear increase to peak
        - High loading (50-100%): Peak efficiency
        - Overloading (> 100%): Decreased efficiency
        
        Args:
            dc_power_kw: DC input power in kW
            
        Returns:
            Efficiency factor (0-1)
        """
        if dc_power_kw <= 0:
            return 0.0
        
        # Calculate loading as fraction of capacity
        loading = dc_power_kw / self.capacity_kw
        
        if loading < 0.1:
            # Low loading: reduced efficiency
            efficiency = 0.85 + (self.base_efficiency - 0.85) * (loading / 0.1)
        elif loading < 0.5:
            # Medium loading: linear increase to peak
            efficiency = 0.85 + (self.base_efficiency - 0.85) * (loading / 0.5)
        elif loading <= 1.0:
            # High loading: peak efficiency
            efficiency = self.base_efficiency
        else:
            # Overloading: decreased efficiency
            efficiency = self.base_efficiency * (1.0 - 0.1 * (loading - 1.0))
            efficiency = max(0.7, efficiency)  # Minimum 70% efficiency
        
        return efficiency
    
    def convert(self, dc_power_kw: float) -> Tuple[float, float]:
        """
        Convert DC power to AC power with reactive power calculation.
        
        Formula:
            ac_power = dc_power * efficiency_curve(dc_power)
            reactive_power = ac_power * tan(acos(power_factor))
        
        Args:
            dc_power_kw: DC input power in kW
            
        Returns:
            Tuple of (ac_active_power_kw, reactive_power_kvar)
        """
        # Handle negative or zero power
        if dc_power_kw <= 0:
            return (0.0, 0.0)
        
        # Calculate efficiency at this power level
        efficiency = self._efficiency_curve(dc_power_kw)
        
        # Convert to AC power
        ac_power_kw = dc_power_kw * efficiency
        
        # Cap at inverter capacity
        ac_power_kw = min(ac_power_kw, self.capacity_kw)
        
        # Calculate reactive power
        # Q = P * tan(acos(pf))
        reactive_power_kvar = ac_power_kw * np.tan(np.arccos(self.power_factor))
        
        return (ac_power_kw, reactive_power_kvar)
