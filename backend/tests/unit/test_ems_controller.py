"""
Unit tests for EMS controller.

Tests battery dispatch optimization, rule-based control, and reactive control.
"""

import pytest
import numpy as np
from backend.src.simulation.ems_controller import (
    EMSController, EMSConfig, SystemState
)


@pytest.fixture
def ems_config():
    """Create standard EMS configuration for testing."""
    return EMSConfig(
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        soc_min=0.1,
        soc_max=0.9,
        control_weight_frequency=1.0,
        control_weight_battery=0.1
    )


@pytest.fixture
def current_state():
    """Create standard system state for testing."""
    return SystemState(
        soc_kwh=2.5,  # 50% SOC
        pv_power=2.0,
        load_power=1.5,
        frequency_hz=60.0,
        voltage_pu=1.0
    )


class TestEMSControllerInitialization:
    """Test EMS controller initialization."""
    
    def test_initialization_mpc(self, ems_config):
        """Test MPC controller initialization."""
        controller = EMSController(ems_config, strategy='mpc')
        assert controller.strategy == 'mpc'
        assert controller.config == ems_config
        assert controller.soc_min_kwh == 0.5  # 10% of 5.0 kWh
        assert controller.soc_max_kwh == 4.5  # 90% of 5.0 kWh
    
    def test_initialization_rule_based(self, ems_config):
        """Test rule-based controller initialization."""
        controller = EMSController(ems_config, strategy='rule_based')
        assert controller.strategy == 'rule_based'
    
    def test_initialization_reactive(self, ems_config):
        """Test reactive controller initialization."""
        controller = EMSController(ems_config, strategy='reactive')
        assert controller.strategy == 'reactive'
    
    def test_invalid_strategy(self, ems_config):
        """Test that invalid strategy raises error."""
        controller = EMSController(ems_config, strategy='invalid')
        with pytest.raises(ValueError, match="Unknown control strategy"):
            controller.compute_dispatch(
                SystemState(2.5, 2.0, 1.5, 60.0, 1.0),
                np.array([2.0]),
                np.array([1.5]),
                1
            )


class TestReactiveControl:
    """Test reactive control strategy."""
    
    def test_reactive_surplus_power(self, ems_config, current_state):
        """Test reactive control with surplus PV power."""
        controller = EMSController(ems_config, strategy='reactive')
        
        # Surplus: PV=2.0, Load=1.5, imbalance=+0.5
        # Should charge battery with -0.5 kW (negative of imbalance)
        dispatch = controller.compute_dispatch(
            current_state,
            np.array([2.0, 2.0, 2.0]),
            np.array([1.5, 1.5, 1.5]),
            horizon=3
        )
        
        # Only first timestep should be non-zero in reactive mode
        assert dispatch[0] == pytest.approx(-0.5, abs=0.01)
        assert dispatch[1] == 0.0
        assert dispatch[2] == 0.0
    
    def test_reactive_deficit_power(self, ems_config):
        """Test reactive control with power deficit."""
        controller = EMSController(ems_config, strategy='reactive')
        
        # Deficit: PV=1.0, Load=2.0, imbalance=-1.0
        state = SystemState(2.5, 1.0, 2.0, 60.0, 1.0)
        
        dispatch = controller.compute_dispatch(
            state,
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            horizon=2
        )
        
        # Should discharge battery with +1.0 kW
        assert dispatch[0] == pytest.approx(1.0, abs=0.01)
        assert dispatch[1] == 0.0
    
    def test_reactive_power_limit(self, ems_config):
        """Test reactive control respects power limits."""
        controller = EMSController(ems_config, strategy='reactive')
        
        # Large deficit: PV=0.0, Load=5.0, imbalance=-5.0
        # But battery power limited to 3.0 kW
        state = SystemState(2.5, 0.0, 5.0, 60.0, 1.0)
        
        dispatch = controller.compute_dispatch(
            state,
            np.array([0.0]),
            np.array([5.0]),
            horizon=1
        )
        
        # Should be limited to max battery power
        assert dispatch[0] == pytest.approx(3.0, abs=0.01)


class TestRuleBasedControl:
    """Test rule-based control strategy."""
    
    def test_rule_based_surplus(self, ems_config, current_state):
        """Test rule-based control charges battery during surplus."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        # Forecast shows consistent surplus
        forecast = np.array([3.0, 3.5, 4.0])
        load_forecast = np.array([1.5, 1.5, 1.5])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=3, timestep_hours=1/60
        )
        
        # Should charge battery (positive dispatch)
        assert np.all(dispatch >= 0)
        # Should not exceed power limit
        assert np.all(dispatch <= ems_config.battery_power_kw)
    
    def test_rule_based_deficit(self, ems_config, current_state):
        """Test rule-based control discharges battery during deficit."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        # Forecast shows consistent deficit
        forecast = np.array([0.5, 0.3, 0.2])
        load_forecast = np.array([2.0, 2.0, 2.0])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=3, timestep_hours=1/60
        )
        
        # Should discharge battery (negative dispatch)
        assert np.all(dispatch <= 0)
        # Should not exceed power limit
        assert np.all(dispatch >= -ems_config.battery_power_kw)
    
    def test_rule_based_respects_soc_limits(self, ems_config):
        """Test rule-based control respects SOC limits."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        # Start near max SOC
        state = SystemState(4.4, 3.0, 1.0, 60.0, 1.0)  # 88% SOC
        
        # Forecast shows surplus (would want to charge)
        forecast = np.array([5.0, 5.0, 5.0])
        load_forecast = np.array([1.0, 1.0, 1.0])
        
        dispatch = controller.compute_dispatch(
            state, forecast, load_forecast, horizon=3, timestep_hours=1/60
        )
        
        # Should limit charging to avoid exceeding max SOC
        # Verify SOC doesn't exceed limit
        soc = state.soc_kwh
        for p in dispatch:
            if p > 0:
                soc += p * (1/60) * ems_config.battery_charge_efficiency
            else:
                soc -= abs(p) * (1/60) / ems_config.battery_discharge_efficiency
            assert soc <= controller.soc_max_kwh + 0.01  # Small tolerance
    
    def test_rule_based_balanced_power(self, ems_config, current_state):
        """Test rule-based control with balanced power."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        # Forecast shows balanced power (PV ≈ load)
        forecast = np.array([1.5, 1.5, 1.5])
        load_forecast = np.array([1.5, 1.5, 1.5])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=3, timestep_hours=1/60
        )
        
        # Should maintain SOC (minimal dispatch)
        assert np.all(np.abs(dispatch) < 0.2)


class TestMPCOptimization:
    """Test Model Predictive Control optimization."""
    
    def test_mpc_basic_optimization(self, ems_config, current_state):
        """Test MPC produces valid dispatch sequence."""
        controller = EMSController(ems_config, strategy='mpc')
        
        forecast = np.array([2.0, 3.0, 1.5, 0.5])
        load_forecast = np.array([1.5, 1.5, 1.5, 1.5])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=4, timestep_hours=1/60
        )
        
        # Should return array of correct length
        assert len(dispatch) == 4
        # Should respect power limits
        assert np.all(dispatch >= -ems_config.battery_power_kw)
        assert np.all(dispatch <= ems_config.battery_power_kw)
    
    def test_mpc_respects_power_limits(self, ems_config, current_state):
        """Test MPC respects battery power limits."""
        controller = EMSController(ems_config, strategy='mpc')
        
        # Large power variations
        forecast = np.array([10.0, 0.0, 10.0, 0.0])
        load_forecast = np.array([1.0, 5.0, 1.0, 5.0])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=4, timestep_hours=1/60
        )
        
        # All dispatch values should be within limits
        assert np.all(dispatch >= -ems_config.battery_power_kw)
        assert np.all(dispatch <= ems_config.battery_power_kw)
    
    def test_mpc_minimizes_imbalance(self, ems_config, current_state):
        """Test MPC attempts to minimize power imbalance."""
        controller = EMSController(ems_config, strategy='mpc')
        
        # Predictable surplus then deficit
        forecast = np.array([4.0, 4.0, 0.5, 0.5])
        load_forecast = np.array([1.5, 1.5, 1.5, 1.5])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=4, timestep_hours=1/60
        )
        
        # Calculate power imbalances with battery dispatch
        imbalance_with_battery = forecast + dispatch - load_forecast
        imbalance_without_battery = forecast - load_forecast
        
        # MPC should reduce overall imbalance (measured by sum of squared imbalances)
        cost_with = np.sum(imbalance_with_battery ** 2)
        cost_without = np.sum(imbalance_without_battery ** 2)
        
        # With battery should be better than without (or at least not much worse)
        assert cost_with <= cost_without * 1.1  # Allow 10% tolerance
    
    def test_mpc_fallback_on_failure(self, ems_config):
        """Test MPC falls back to reactive control if optimization fails."""
        controller = EMSController(ems_config, strategy='mpc')
        
        # Create infeasible scenario (very low SOC, large deficit)
        state = SystemState(0.5, 0.0, 5.0, 60.0, 1.0)  # 10% SOC, large deficit
        
        forecast = np.array([0.0, 0.0, 0.0])
        load_forecast = np.array([5.0, 5.0, 5.0])
        
        # Should not crash, should return some dispatch
        dispatch = controller.compute_dispatch(
            state, forecast, load_forecast, horizon=3, timestep_hours=1/60
        )
        
        assert len(dispatch) == 3
        assert np.all(np.isfinite(dispatch))


class TestSingleStepCompute:
    """Test single-step dispatch computation."""
    
    def test_single_step_with_forecast(self, ems_config, current_state):
        """Test single-step computation with forecast."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        forecast = np.array([3.0, 3.5, 4.0])
        load_forecast = np.array([1.5, 1.5, 1.5])
        
        dispatch = controller.compute_single_step(
            current_state, forecast, load_forecast
        )
        
        # Should return single float value
        assert isinstance(dispatch, (float, np.floating))
        # Should be within power limits
        assert -ems_config.battery_power_kw <= dispatch <= ems_config.battery_power_kw
    
    def test_single_step_without_forecast(self, ems_config, current_state):
        """Test single-step computation without forecast (reactive mode)."""
        controller = EMSController(ems_config, strategy='reactive')
        
        dispatch = controller.compute_single_step(current_state)
        
        # Should return single float value
        assert isinstance(dispatch, (float, np.floating))
        # Should react to current imbalance
        # PV=2.0, Load=1.5, imbalance=+0.5, should charge with -0.5
        assert dispatch == pytest.approx(-0.5, abs=0.01)


class TestDispatchOutputShapes:
    """Test dispatch output shapes and types."""
    
    def test_dispatch_array_length(self, ems_config, current_state):
        """Test dispatch array has correct length."""
        controller = EMSController(ems_config, strategy='mpc')
        
        for horizon in [1, 5, 10, 20]:
            forecast = np.ones(horizon) * 2.0
            load_forecast = np.ones(horizon) * 1.5
            
            dispatch = controller.compute_dispatch(
                current_state, forecast, load_forecast, horizon
            )
            
            assert len(dispatch) == horizon
    
    def test_dispatch_array_type(self, ems_config, current_state):
        """Test dispatch array is numpy array."""
        controller = EMSController(ems_config, strategy='rule_based')
        
        forecast = np.array([2.0, 2.5, 3.0])
        load_forecast = np.array([1.5, 1.5, 1.5])
        
        dispatch = controller.compute_dispatch(
            current_state, forecast, load_forecast, horizon=3
        )
        
        assert isinstance(dispatch, np.ndarray)
        assert dispatch.dtype in [np.float64, np.float32]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
