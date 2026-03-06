"""
Microgrid simulator for stability analysis.

This module implements the main simulation engine that orchestrates component
interactions to model microgrid behavior over time.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.simulation.components import PVArray, Battery, Inverter
from src.config.schemas import MicrogridConfig, MicrogridMode


@dataclass
class PowerBalance:
    """Power balance at a single timestep."""
    pv_power: float
    load_power: float
    battery_power: float  # Positive = charging, negative = discharging
    grid_power: float  # Positive = import, negative = export
    power_imbalance: float


@dataclass
class SystemState:
    """System state at a single timestep."""
    time_index: int
    soc_kwh: float
    frequency_hz: float
    voltage_pu: float  # Per-unit voltage (1.0 = nominal)
    power_balance: PowerBalance


@dataclass
class SimulationResult:
    """Complete simulation results."""
    timestamps: np.ndarray
    pv_power: np.ndarray
    load_power: np.ndarray
    battery_power: np.ndarray
    battery_soc: np.ndarray
    frequency_deviation: np.ndarray  # Hz deviation from nominal
    voltage_deviation: np.ndarray  # Percentage deviation from nominal
    grid_power: np.ndarray  # Only for grid-connected mode
    states: List[SystemState]


class MicrogridSimulator:
    """
    Physics-based microgrid simulator.
    
    Simulates microgrid component interactions including PV generation,
    battery storage, inverter operation, and grid stability metrics.
    """
    
    def __init__(self, config: MicrogridConfig):
        """
        Initialize microgrid simulator.
        
        Args:
            config: Microgrid configuration specifying components and operating mode
        """
        self.config = config
        self.components = self._initialize_components()
        
        # System parameters
        self.nominal_frequency = 60.0  # Hz (can be 50 Hz for other regions)
        self.nominal_voltage = 1.0  # Per-unit
        
        # Inertia constant for frequency dynamics (seconds)
        # Lower inertia = more frequency deviation for same power imbalance
        self.inertia_constant = 3.0 if config.mode == MicrogridMode.ISLANDED else 5.0
        
        # System reactance for voltage dynamics (per-unit)
        self.system_reactance = 0.1
        
        # Frequency and voltage limits
        self.freq_limit_hz = 2.0  # ±2 Hz from nominal
        self.voltage_limit_pct = 10.0  # ±10% from nominal
    
    def _initialize_components(self) -> Dict[str, any]:
        """
        Initialize microgrid components based on configuration.
        
        Returns:
            Dictionary of component instances
        """
        components = {}
        
        # Initialize PV array
        components['pv'] = PVArray(
            capacity_kw=self.config.pv_capacity_kw,
            area_m2=self.config.pv_capacity_kw * 5.0,  # Assume ~5 m²/kW
            base_efficiency=0.18
        )
        
        # Initialize battery
        components['battery'] = Battery(
            capacity_kwh=self.config.battery_capacity_kwh,
            max_power_kw=self.config.battery_power_kw,
            initial_soc_kwh=self.config.initial_soc_kwh,
            charge_efficiency=0.95,
            discharge_efficiency=0.95
        )
        
        # Initialize inverter
        components['inverter'] = Inverter(
            capacity_kw=self.config.inverter_capacity_kw,
            base_efficiency=0.96,
            power_factor=0.95
        )
        
        return components

    
    def simulate(self, 
                 pv_forecast: np.ndarray,
                 actual_pv: np.ndarray,
                 load_profile: np.ndarray,
                 timestep_seconds: int = 60,
                 battery_dispatch: Optional[np.ndarray] = None) -> SimulationResult:
        """
        Run microgrid simulation.
        
        Args:
            pv_forecast: Forecasted PV power in kW (used by EMS for dispatch)
            actual_pv: Actual PV power in kW (what actually happens)
            load_profile: Load demand in kW
            timestep_seconds: Simulation timestep in seconds (default: 60s)
            battery_dispatch: Optional pre-computed battery dispatch in kW
                            (positive = charge, negative = discharge)
                            If None, reactive control is used
        
        Returns:
            SimulationResult containing time-series data and system states
        """
        n_steps = len(actual_pv)
        dt_hours = timestep_seconds / 3600.0
        
        # Initialize result arrays
        battery_power = np.zeros(n_steps)
        battery_soc = np.zeros(n_steps)
        frequency_deviation = np.zeros(n_steps)
        voltage_deviation = np.zeros(n_steps)
        grid_power = np.zeros(n_steps)
        states = []
        
        # Initialize system state
        current_frequency = self.nominal_frequency
        current_voltage = self.nominal_voltage
        
        for t in range(n_steps):
            # Get current values
            pv_power = actual_pv[t]
            load = load_profile[t]
            
            # Determine battery dispatch
            if battery_dispatch is not None:
                desired_battery_power = battery_dispatch[t]
            else:
                # Reactive control: try to balance power
                desired_battery_power = -(pv_power - load)
            
            # Calculate power balance and update battery
            power_bal = self._calculate_power_balance(
                pv_power=pv_power,
                load_power=load,
                desired_battery_power=desired_battery_power,
                dt_hours=dt_hours,
                t=t
            )
            
            # Update battery state
            actual_battery_power = self._update_battery_state(
                desired_battery_power, dt_hours
            )
            
            # Recalculate power balance with actual battery power
            power_bal.battery_power = actual_battery_power
            power_bal.power_imbalance = (
                pv_power + actual_battery_power - load
            )
            
            # Handle grid power for grid-connected mode
            if self.config.mode == MicrogridMode.GRID_CONNECTED:
                power_bal.grid_power = -power_bal.power_imbalance
                power_bal.power_imbalance = 0.0  # Grid absorbs imbalance
            else:
                power_bal.grid_power = 0.0
            
            # Calculate frequency deviation
            freq_dev = self._calculate_frequency_deviation(power_bal.power_imbalance)
            current_frequency = self.nominal_frequency + freq_dev
            
            # Enforce frequency limits
            freq_dev = np.clip(freq_dev, -self.freq_limit_hz, self.freq_limit_hz)
            current_frequency = self.nominal_frequency + freq_dev
            
            # Calculate voltage deviation
            # Estimate reactive power from inverter
            _, reactive_power = self.components['inverter'].convert(pv_power)
            volt_dev = self._calculate_voltage_deviation(reactive_power)
            current_voltage = self.nominal_voltage + volt_dev / 100.0
            
            # Enforce voltage limits
            volt_dev = np.clip(volt_dev, -self.voltage_limit_pct, self.voltage_limit_pct)
            current_voltage = self.nominal_voltage + volt_dev / 100.0
            
            # Store results
            battery_power[t] = actual_battery_power
            battery_soc[t] = self.components['battery'].get_soc()
            frequency_deviation[t] = freq_dev
            voltage_deviation[t] = volt_dev
            grid_power[t] = power_bal.grid_power
            
            # Store system state
            state = SystemState(
                time_index=t,
                soc_kwh=battery_soc[t],
                frequency_hz=current_frequency,
                voltage_pu=current_voltage,
                power_balance=power_bal
            )
            states.append(state)
        
        # Create timestamps
        timestamps = np.arange(n_steps) * timestep_seconds
        
        return SimulationResult(
            timestamps=timestamps,
            pv_power=actual_pv,
            load_power=load_profile,
            battery_power=battery_power,
            battery_soc=battery_soc,
            frequency_deviation=frequency_deviation,
            voltage_deviation=voltage_deviation,
            grid_power=grid_power,
            states=states
        )
    
    def _calculate_power_balance(self,
                                 pv_power: float,
                                 load_power: float,
                                 desired_battery_power: float,
                                 dt_hours: float,
                                 t: int) -> PowerBalance:
        """
        Calculate power balance at a single timestep.
        
        Args:
            pv_power: PV generation in kW
            load_power: Load demand in kW
            desired_battery_power: Desired battery power in kW
                                  (positive = charge, negative = discharge)
            dt_hours: Timestep in hours
            t: Current time index
        
        Returns:
            PowerBalance object with power flows
        """
        # Initial power imbalance before battery action
        initial_imbalance = pv_power - load_power
        
        # Battery will try to compensate
        # Note: actual battery power may differ due to constraints
        battery_power = desired_battery_power
        
        # Calculate net power imbalance
        power_imbalance = pv_power + battery_power - load_power
        
        return PowerBalance(
            pv_power=pv_power,
            load_power=load_power,
            battery_power=battery_power,
            grid_power=0.0,  # Will be updated later if grid-connected
            power_imbalance=power_imbalance
        )
    
    def _update_battery_state(self, desired_power: float, dt_hours: float) -> float:
        """
        Update battery state of charge based on desired power.
        
        Args:
            desired_power: Desired battery power in kW
                          (positive = charge, negative = discharge)
            dt_hours: Timestep in hours
        
        Returns:
            Actual battery power in kW (may differ from desired due to constraints)
        """
        battery = self.components['battery']
        
        if desired_power > 0:
            # Charging
            actual_power = battery.charge(desired_power, dt_hours)
            return actual_power
        elif desired_power < 0:
            # Discharging
            actual_power = battery.discharge(-desired_power, dt_hours)
            return -actual_power
        else:
            # No battery action
            return 0.0
    
    def _calculate_frequency_deviation(self, power_imbalance: float) -> float:
        """
        Calculate frequency deviation based on power imbalance and system inertia.
        
        Formula:
            Δf = (P_gen - P_load) / (2 * H * S_base) * f_nominal
        
        Simplified for small systems:
            Δf ≈ power_imbalance / (inertia_constant * system_capacity) * f_nominal
        
        Args:
            power_imbalance: Net power imbalance in kW (positive = excess generation)
        
        Returns:
            Frequency deviation in Hz (positive = above nominal)
        """
        # Use total system capacity as base power
        system_capacity = self.config.pv_capacity_kw + self.config.battery_power_kw
        
        if system_capacity == 0:
            return 0.0
        
        # Frequency deviation proportional to power imbalance
        # Normalized by inertia and system capacity
        freq_deviation = (power_imbalance / (self.inertia_constant * system_capacity)) * self.nominal_frequency
        
        return freq_deviation
    
    def _calculate_voltage_deviation(self, reactive_power: float) -> float:
        """
        Calculate voltage deviation based on reactive power balance.
        
        Formula:
            ΔV = (Q_gen - Q_load) * X / V_nominal
        
        Simplified:
            ΔV_pct ≈ reactive_power * reactance * scaling_factor
        
        Args:
            reactive_power: Reactive power in kVAR (positive = generation)
        
        Returns:
            Voltage deviation in percentage (positive = above nominal)
        """
        # Simplified voltage deviation model
        # In reality, this depends on network impedance and load characteristics
        
        # Use system reactance and capacity for scaling
        system_capacity = self.config.inverter_capacity_kw
        
        if system_capacity == 0:
            return 0.0
        
        # Voltage deviation as percentage
        # Positive reactive power tends to increase voltage
        voltage_deviation_pct = (reactive_power / system_capacity) * self.system_reactance * 100.0
        
        return voltage_deviation_pct
