"""
Energy Management System (EMS) controller for microgrid battery dispatch.

This module implements various control strategies for optimizing battery dispatch
based on PV forecasts to minimize frequency deviation and improve stability.
"""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class SystemState:
    """Current system state for EMS decision-making."""
    soc_kwh: float
    pv_power: float
    load_power: float
    frequency_hz: float
    voltage_pu: float


@dataclass
class EMSConfig:
    """EMS controller configuration."""
    battery_capacity_kwh: float
    battery_power_kw: float
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    soc_min: float = 0.1  # Minimum SOC as fraction
    soc_max: float = 0.9  # Maximum SOC as fraction
    control_weight_frequency: float = 1.0  # Weight for frequency deviation
    control_weight_battery: float = 0.1  # Weight for battery usage


class EMSController:
    """
    Energy Management System controller for battery dispatch optimization.
    
    Implements multiple control strategies:
    - Model Predictive Control (MPC): Optimization-based control using forecasts
    - Rule-based control: Heuristic rules based on forecast trends
    - Reactive control: No forecast, react to current power imbalance
    """
    
    def __init__(self, config: EMSConfig, strategy: Literal['mpc', 'rule_based', 'reactive'] = 'mpc'):
        """
        Initialize EMS controller.
        
        Args:
            config: EMS configuration with battery parameters
            strategy: Control strategy to use ('mpc', 'rule_based', 'reactive')
        """
        self.config = config
        self.strategy = strategy
        
        # Calculate SOC limits in kWh
        self.soc_min_kwh = config.soc_min * config.battery_capacity_kwh
        self.soc_max_kwh = config.soc_max * config.battery_capacity_kwh
    
    def compute_dispatch(self,
                        current_state: SystemState,
                        forecast: np.ndarray,
                        load_forecast: np.ndarray,
                        horizon: int,
                        timestep_hours: float = 1/60) -> np.ndarray:
        """
        Compute optimal battery dispatch over the forecast horizon.
        
        Args:
            current_state: Current system state (SOC, power, frequency, voltage)
            forecast: PV power forecast in kW for next 'horizon' steps
            load_forecast: Load power forecast in kW for next 'horizon' steps
            horizon: Number of timesteps to optimize over
            timestep_hours: Timestep duration in hours (default: 1 minute)
        
        Returns:
            Array of battery dispatch commands in kW (positive = charge, negative = discharge)
            Length equals horizon
        """
        if self.strategy == 'mpc':
            return self._mpc_optimization(current_state, forecast, load_forecast, horizon, timestep_hours)
        elif self.strategy == 'rule_based':
            return self._rule_based_control(current_state, forecast, load_forecast, horizon, timestep_hours)
        elif self.strategy == 'reactive':
            return self._reactive_control(current_state, forecast, load_forecast, horizon)
        else:
            raise ValueError(f"Unknown control strategy: {self.strategy}")
    
    def _mpc_optimization(self,
                         current_state: SystemState,
                         forecast: np.ndarray,
                         load_forecast: np.ndarray,
                         horizon: int,
                         timestep_hours: float) -> np.ndarray:
        """
        Model Predictive Control optimization using scipy.optimize.
        
        Objective: Minimize frequency deviation and battery usage
        Constraints: SOC limits, power limits, battery dynamics
        
        Args:
            current_state: Current system state
            forecast: PV power forecast
            load_forecast: Load power forecast
            horizon: Optimization horizon
            timestep_hours: Timestep in hours
        
        Returns:
            Optimal battery dispatch sequence
        """
        # Ensure forecasts have correct length
        forecast = forecast[:horizon]
        load_forecast = load_forecast[:horizon]
        
        # Initial guess: zero battery power
        x0 = np.zeros(horizon)
        
        # Define objective function
        def objective(battery_dispatch):
            """
            Objective: minimize frequency deviation + battery usage penalty.
            
            Frequency deviation is proportional to power imbalance.
            """
            total_cost = 0.0
            soc = current_state.soc_kwh
            
            for t in range(horizon):
                # Power imbalance
                power_imbalance = forecast[t] + battery_dispatch[t] - load_forecast[t]
                
                # Frequency deviation cost (squared for smoothness)
                freq_cost = self.config.control_weight_frequency * (power_imbalance ** 2)
                
                # Battery usage cost (penalize large power changes)
                battery_cost = self.config.control_weight_battery * (battery_dispatch[t] ** 2)
                
                total_cost += freq_cost + battery_cost
                
                # Update SOC for next iteration (simplified)
                if battery_dispatch[t] > 0:
                    soc += battery_dispatch[t] * timestep_hours * self.config.battery_charge_efficiency
                else:
                    soc -= abs(battery_dispatch[t]) * timestep_hours / self.config.battery_discharge_efficiency
            
            return total_cost
        
        # Define constraints
        constraints = []
        
        # SOC constraints (must stay within limits throughout horizon)
        def soc_constraint_factory(t_idx):
            def soc_min_constraint(battery_dispatch):
                """Ensure SOC stays above minimum."""
                soc = current_state.soc_kwh
                for t in range(t_idx + 1):
                    if battery_dispatch[t] > 0:
                        soc += battery_dispatch[t] * timestep_hours * self.config.battery_charge_efficiency
                    else:
                        soc -= abs(battery_dispatch[t]) * timestep_hours / self.config.battery_discharge_efficiency
                return soc - self.soc_min_kwh
            
            def soc_max_constraint(battery_dispatch):
                """Ensure SOC stays below maximum."""
                soc = current_state.soc_kwh
                for t in range(t_idx + 1):
                    if battery_dispatch[t] > 0:
                        soc += battery_dispatch[t] * timestep_hours * self.config.battery_charge_efficiency
                    else:
                        soc -= abs(battery_dispatch[t]) * timestep_hours / self.config.battery_discharge_efficiency
                return self.soc_max_kwh - soc
            
            return soc_min_constraint, soc_max_constraint
        
        # Add SOC constraints for each timestep
        for t in range(horizon):
            soc_min_con, soc_max_con = soc_constraint_factory(t)
            constraints.append({'type': 'ineq', 'fun': soc_min_con})
            constraints.append({'type': 'ineq', 'fun': soc_max_con})
        
        # Power limits (bounds)
        bounds = [(-self.config.battery_power_kw, self.config.battery_power_kw) for _ in range(horizon)]
        
        # Solve optimization problem
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            return result.x
        else:
            # If optimization fails, fall back to reactive control
            return self._reactive_control(current_state, forecast, load_forecast, horizon)
    
    def _rule_based_control(self,
                           current_state: SystemState,
                           forecast: np.ndarray,
                           load_forecast: np.ndarray,
                           horizon: int,
                           timestep_hours: float) -> np.ndarray:
        """
        Rule-based control strategy using forecast trends.
        
        Rules:
        1. If forecast shows surplus (PV > load), charge battery
        2. If forecast shows deficit (PV < load), discharge battery
        3. Respect SOC and power limits
        4. Adjust power based on magnitude of imbalance
        
        Args:
            current_state: Current system state
            forecast: PV power forecast
            load_forecast: Load power forecast
            horizon: Control horizon
            timestep_hours: Timestep in hours
        
        Returns:
            Battery dispatch sequence
        """
        dispatch = np.zeros(horizon)
        soc = current_state.soc_kwh
        
        for t in range(horizon):
            # Calculate expected power imbalance
            power_imbalance = forecast[t] - load_forecast[t]
            
            # Rule 1: Surplus power -> charge battery
            if power_imbalance > 0.1:  # Threshold to avoid small fluctuations
                # Charge with available surplus, limited by battery power rating
                desired_charge = min(power_imbalance, self.config.battery_power_kw)
                
                # Check SOC limit
                energy_to_add = desired_charge * timestep_hours * self.config.battery_charge_efficiency
                if soc + energy_to_add <= self.soc_max_kwh:
                    dispatch[t] = desired_charge
                    soc += energy_to_add
                else:
                    # Limit charging to available capacity
                    available_capacity = self.soc_max_kwh - soc
                    dispatch[t] = available_capacity / (timestep_hours * self.config.battery_charge_efficiency)
                    soc = self.soc_max_kwh
            
            # Rule 2: Deficit power -> discharge battery
            elif power_imbalance < -0.1:
                # Discharge to cover deficit, limited by battery power rating
                desired_discharge = min(abs(power_imbalance), self.config.battery_power_kw)
                
                # Check SOC limit
                energy_to_remove = desired_discharge * timestep_hours / self.config.battery_discharge_efficiency
                if soc - energy_to_remove >= self.soc_min_kwh:
                    dispatch[t] = -desired_discharge
                    soc -= energy_to_remove
                else:
                    # Limit discharging to available energy
                    available_energy = soc - self.soc_min_kwh
                    dispatch[t] = -(available_energy * self.config.battery_discharge_efficiency / timestep_hours)
                    soc = self.soc_min_kwh
            
            # Rule 3: Balanced power -> maintain SOC
            else:
                dispatch[t] = 0.0
        
        return dispatch
    
    def _reactive_control(self,
                         current_state: SystemState,
                         forecast: np.ndarray,
                         load_forecast: np.ndarray,
                         horizon: int) -> np.ndarray:
        """
        Reactive control (no-forecast baseline).
        
        Simply reacts to current power imbalance without using forecast.
        This serves as the baseline for comparison.
        
        Args:
            current_state: Current system state
            forecast: PV power forecast (not used in reactive mode)
            load_forecast: Load power forecast (not used in reactive mode)
            horizon: Control horizon
        
        Returns:
            Battery dispatch sequence (only first value is meaningful)
        """
        dispatch = np.zeros(horizon)
        
        # React to current power imbalance only
        current_imbalance = current_state.pv_power - current_state.load_power
        
        # Try to compensate for imbalance
        # Positive imbalance (excess PV) -> charge battery
        # Negative imbalance (deficit) -> discharge battery
        desired_power = -current_imbalance
        
        # Limit to battery power rating
        desired_power = np.clip(desired_power, -self.config.battery_power_kw, self.config.battery_power_kw)
        
        # Set only the first timestep (reactive control doesn't plan ahead)
        dispatch[0] = desired_power
        
        return dispatch
    
    def compute_single_step(self,
                           current_state: SystemState,
                           forecast: Optional[np.ndarray] = None,
                           load_forecast: Optional[np.ndarray] = None) -> float:
        """
        Compute battery dispatch for a single timestep.
        
        Convenience method for single-step control.
        
        Args:
            current_state: Current system state
            forecast: Optional PV power forecast
            load_forecast: Optional load power forecast
        
        Returns:
            Battery dispatch command in kW (positive = charge, negative = discharge)
        """
        if forecast is None or load_forecast is None:
            # No forecast available, use reactive control
            horizon = 1
            forecast = np.array([current_state.pv_power])
            load_forecast = np.array([current_state.load_power])
        else:
            horizon = len(forecast)
        
        dispatch_sequence = self.compute_dispatch(
            current_state, forecast, load_forecast, horizon
        )
        
        # Return only the first timestep command
        return dispatch_sequence[0]
