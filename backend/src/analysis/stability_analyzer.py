"""
Stability analyzer for microgrid simulation results.

This module calculates comprehensive stability and performance metrics from
simulation results to quantify the impact of forecasting on microgrid stability.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from src.simulation.simulator import SimulationResult


@dataclass
class FrequencyMetrics:
    """Frequency stability metrics."""
    mean_absolute_deviation: float  # Hz
    std_deviation: float  # Hz
    max_deviation: float  # Hz
    time_outside_limits: float  # Fraction of time (0-1)
    rate_of_change: float  # Hz/timestep


@dataclass
class VoltageMetrics:
    """Voltage stability metrics."""
    mean_absolute_deviation: float  # Percentage
    std_deviation: float  # Percentage
    max_deviation: float  # Percentage
    time_outside_limits: float  # Fraction of time (0-1)


@dataclass
class BatteryMetrics:
    """Battery stress metrics."""
    soc_range: float  # kWh
    num_cycles: float  # Number of charge/discharge cycles
    max_depth_of_discharge: float  # Fraction (0-1)
    avg_c_rate: float  # Average C-rate
    total_throughput: float  # kWh


@dataclass
class PowerQualityMetrics:
    """Power quality metrics."""
    thd_proxy: float  # Total harmonic distortion proxy
    power_factor: float  # Average power factor


@dataclass
class EnergyMetrics:
    """Energy balance metrics."""
    total_unmet_load: float  # kWh
    total_curtailed_pv: float  # kWh
    energy_efficiency: float  # Fraction (0-1)


@dataclass
class ControlEffortMetrics:
    """Control effort metrics."""
    sum_absolute_battery_changes: float  # kW
    num_control_actions: int  # Number of non-zero battery power changes


@dataclass
class StabilityMetrics:
    """Complete stability analysis metrics."""
    frequency: FrequencyMetrics
    voltage: VoltageMetrics
    battery: BatteryMetrics
    power_quality: PowerQualityMetrics
    energy: EnergyMetrics
    control_effort: ControlEffortMetrics


class StabilityAnalyzer:
    """
    Analyzer for calculating comprehensive stability metrics from simulation results.
    
    Calculates metrics for frequency, voltage, battery stress, power quality,
    energy balance, and control effort to quantify microgrid stability.
    """
    
    def __init__(self, battery_capacity_kwh: float):
        """
        Initialize stability analyzer.
        
        Args:
            battery_capacity_kwh: Battery capacity for C-rate calculations
        """
        self.battery_capacity_kwh = battery_capacity_kwh
    
    def analyze(self, sim_result: SimulationResult) -> StabilityMetrics:
        """
        Calculate all stability metrics from simulation results.
        
        Args:
            sim_result: Complete simulation results
        
        Returns:
            StabilityMetrics containing all calculated metrics
        """
        frequency = self.frequency_metrics(sim_result.frequency_deviation)
        voltage = self.voltage_metrics(sim_result.voltage_deviation)
        battery = self.battery_stress_metrics(
            sim_result.battery_soc,
            sim_result.battery_power,
            sim_result.timestamps
        )
        power_quality = self.power_quality_metrics(
            sim_result.pv_power,
            sim_result.battery_power,
            sim_result.load_power
        )
        energy = self.energy_balance_metrics(
            sim_result.pv_power,
            sim_result.battery_power,
            sim_result.load_power,
            sim_result.timestamps
        )
        control_effort = self.control_effort_metrics(sim_result.battery_power)
        
        return StabilityMetrics(
            frequency=frequency,
            voltage=voltage,
            battery=battery,
            power_quality=power_quality,
            energy=energy,
            control_effort=control_effort
        )
    
    def frequency_metrics(self, freq_deviation: np.ndarray) -> FrequencyMetrics:
        """
        Calculate frequency-related stability metrics.
        
        Args:
            freq_deviation: Frequency deviation time series in Hz
        
        Returns:
            FrequencyMetrics object
        """
        # Handle empty arrays
        if len(freq_deviation) == 0:
            return FrequencyMetrics(
                mean_absolute_deviation=0.0,
                std_deviation=0.0,
                max_deviation=0.0,
                time_outside_limits=0.0,
                rate_of_change=0.0
            )
        
        # Mean absolute deviation
        mean_abs_dev = np.mean(np.abs(freq_deviation))
        
        # Standard deviation
        std_dev = np.std(freq_deviation)
        
        # Maximum deviation
        max_dev = np.max(np.abs(freq_deviation))
        
        # Time outside limits (±0.5 Hz is typical limit for good stability)
        freq_limit = 0.5  # Hz
        time_outside = np.sum(np.abs(freq_deviation) > freq_limit) / len(freq_deviation)
        
        # Rate of change (average absolute change between timesteps)
        if len(freq_deviation) > 1:
            rate_of_change = np.mean(np.abs(np.diff(freq_deviation)))
        else:
            rate_of_change = 0.0
        
        return FrequencyMetrics(
            mean_absolute_deviation=float(mean_abs_dev),
            std_deviation=float(std_dev),
            max_deviation=float(max_dev),
            time_outside_limits=float(time_outside),
            rate_of_change=float(rate_of_change)
        )
    
    def voltage_metrics(self, voltage_deviation: np.ndarray) -> VoltageMetrics:
        """
        Calculate voltage-related stability metrics.
        
        Args:
            voltage_deviation: Voltage deviation time series in percentage
        
        Returns:
            VoltageMetrics object
        """
        # Handle empty arrays
        if len(voltage_deviation) == 0:
            return VoltageMetrics(
                mean_absolute_deviation=0.0,
                std_deviation=0.0,
                max_deviation=0.0,
                time_outside_limits=0.0
            )
        
        # Mean absolute deviation
        mean_abs_dev = np.mean(np.abs(voltage_deviation))
        
        # Standard deviation
        std_dev = np.std(voltage_deviation)
        
        # Maximum deviation
        max_dev = np.max(np.abs(voltage_deviation))
        
        # Time outside limits (±5% is typical limit for good stability)
        voltage_limit = 5.0  # Percentage
        time_outside = np.sum(np.abs(voltage_deviation) > voltage_limit) / len(voltage_deviation)
        
        return VoltageMetrics(
            mean_absolute_deviation=float(mean_abs_dev),
            std_deviation=float(std_dev),
            max_deviation=float(max_dev),
            time_outside_limits=float(time_outside)
        )
    
    def battery_stress_metrics(self,
                               soc: np.ndarray,
                               power: np.ndarray,
                               timestamps: np.ndarray) -> BatteryMetrics:
        """
        Calculate battery stress metrics.
        
        Args:
            soc: State of charge time series in kWh
            power: Battery power time series in kW (positive = charge, negative = discharge)
            timestamps: Timestamp array in seconds
        
        Returns:
            BatteryMetrics object
        """
        # SOC range
        soc_range = float(np.max(soc) - np.min(soc))
        
        # Count charge/discharge cycles using rainflow counting approximation
        num_cycles = self._count_cycles(soc)
        
        # Maximum depth of discharge
        # DOD = 1 - (min_SOC / capacity)
        max_dod = 1.0 - (np.min(soc) / self.battery_capacity_kwh)
        
        # Average C-rate (power / capacity)
        avg_c_rate = np.mean(np.abs(power)) / self.battery_capacity_kwh
        
        # Total throughput (total energy cycled through battery)
        # Calculate timestep in hours
        if len(timestamps) > 1:
            dt_hours = (timestamps[1] - timestamps[0]) / 3600.0
        else:
            dt_hours = 1.0 / 60.0  # Default 1 minute
        
        total_throughput = np.sum(np.abs(power)) * dt_hours
        
        return BatteryMetrics(
            soc_range=float(soc_range),
            num_cycles=float(num_cycles),
            max_depth_of_discharge=float(max_dod),
            avg_c_rate=float(avg_c_rate),
            total_throughput=float(total_throughput)
        )
    
    def _count_cycles(self, soc: np.ndarray) -> float:
        """
        Count battery charge/discharge cycles using peak counting method.
        
        A cycle is defined as a full charge-discharge or discharge-charge sequence.
        This uses a simplified peak-counting approach.
        
        Args:
            soc: State of charge time series
        
        Returns:
            Number of cycles (can be fractional)
        """
        if len(soc) < 3:
            return 0.0
        
        # Find local maxima and minima
        peaks = []
        valleys = []
        
        for i in range(1, len(soc) - 1):
            if soc[i] > soc[i-1] and soc[i] > soc[i+1]:
                peaks.append(soc[i])
            elif soc[i] < soc[i-1] and soc[i] < soc[i+1]:
                valleys.append(soc[i])
        
        # Count cycles as pairs of peaks and valleys
        # Each peak-valley or valley-peak pair represents a half cycle
        num_extrema = len(peaks) + len(valleys)
        num_cycles = num_extrema / 2.0
        
        return num_cycles
    
    def power_quality_metrics(self,
                             pv_power: np.ndarray,
                             battery_power: np.ndarray,
                             load_power: np.ndarray) -> PowerQualityMetrics:
        """
        Calculate power quality metrics.
        
        Args:
            pv_power: PV power time series in kW
            battery_power: Battery power time series in kW
            load_power: Load power time series in kW
        
        Returns:
            PowerQualityMetrics object
        """
        # THD proxy: Use standard deviation of power imbalance as a proxy
        # for harmonic distortion (higher variability = more harmonics)
        power_imbalance = pv_power + battery_power - load_power
        thd_proxy = np.std(power_imbalance) / (np.mean(np.abs(load_power)) + 1e-6)
        
        # Power factor: Calculate from active and apparent power
        # For simplicity, estimate based on power balance quality
        # Perfect balance = PF close to 1.0, poor balance = lower PF
        mean_imbalance = np.mean(np.abs(power_imbalance))
        mean_load = np.mean(load_power) + 1e-6
        
        # Power factor estimate: 1.0 - normalized imbalance
        # Capped between 0.7 and 1.0
        power_factor = 1.0 - min(0.3, mean_imbalance / mean_load)
        
        return PowerQualityMetrics(
            thd_proxy=float(thd_proxy),
            power_factor=float(power_factor)
        )
    
    def energy_balance_metrics(self,
                               pv_power: np.ndarray,
                               battery_power: np.ndarray,
                               load_power: np.ndarray,
                               timestamps: np.ndarray) -> EnergyMetrics:
        """
        Calculate energy balance metrics.
        
        Args:
            pv_power: PV power time series in kW
            battery_power: Battery power time series in kW
            load_power: Load power time series in kW
            timestamps: Timestamp array in seconds
        
        Returns:
            EnergyMetrics object
        """
        # Calculate timestep in hours
        if len(timestamps) > 1:
            dt_hours = (timestamps[1] - timestamps[0]) / 3600.0
        else:
            dt_hours = 1.0 / 60.0  # Default 1 minute
        
        # Available power at each timestep
        available_power = pv_power + np.minimum(0, battery_power)  # PV + discharge
        
        # Unmet load: when load exceeds available power
        unmet_power = np.maximum(0, load_power - available_power)
        total_unmet_load = np.sum(unmet_power) * dt_hours
        
        # Curtailed PV: when PV + battery charging exceeds load and battery is full
        # Simplified: excess power that cannot be used or stored
        excess_power = pv_power + battery_power - load_power
        curtailed_power = np.maximum(0, excess_power)
        total_curtailed_pv = np.sum(curtailed_power) * dt_hours
        
        # Energy efficiency: delivered energy / generated energy
        total_generated = np.sum(pv_power) * dt_hours
        total_delivered = np.sum(np.minimum(load_power, available_power)) * dt_hours
        
        if total_generated > 0:
            energy_efficiency = total_delivered / total_generated
        else:
            energy_efficiency = 0.0
        
        return EnergyMetrics(
            total_unmet_load=float(total_unmet_load),
            total_curtailed_pv=float(total_curtailed_pv),
            energy_efficiency=float(energy_efficiency)
        )
    
    def control_effort_metrics(self, battery_power: np.ndarray) -> ControlEffortMetrics:
        """
        Calculate control effort metrics.
        
        Args:
            battery_power: Battery power time series in kW
        
        Returns:
            ControlEffortMetrics object
        """
        # Sum of absolute battery power changes
        if len(battery_power) > 1:
            power_changes = np.diff(battery_power)
            sum_abs_changes = np.sum(np.abs(power_changes))
        else:
            sum_abs_changes = 0.0
        
        # Number of control actions (non-zero power changes)
        # Use a threshold to avoid counting noise
        threshold = 0.01  # kW
        if len(battery_power) > 1:
            significant_changes = np.abs(power_changes) > threshold
            num_actions = np.sum(significant_changes)
        else:
            num_actions = 0
        
        return ControlEffortMetrics(
            sum_absolute_battery_changes=float(sum_abs_changes),
            num_control_actions=int(num_actions)
        )
