"""
Unit tests for MicrogridSimulator.

Tests the main simulation engine and its methods.
"""

import pytest
import numpy as np
from backend.src.simulation.simulator import MicrogridSimulator, PowerBalance, SystemState, SimulationResult
from backend.src.config.schemas import MicrogridConfig, MicrogridMode


class TestMicrogridSimulator:
    """Test MicrogridSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        assert sim.config == config
        assert 'pv' in sim.components
        assert 'battery' in sim.components
        assert 'inverter' in sim.components
        assert sim.nominal_frequency == 60.0
        assert sim.nominal_voltage == 1.0
    
    def test_initialize_components(self):
        """Test component initialization."""
        config = MicrogridConfig(
            mode=MicrogridMode.GRID_CONNECTED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        )
        
        sim = MicrogridSimulator(config)
        
        # Check PV array
        assert sim.components['pv'].capacity_kw == 10.0
        
        # Check battery
        assert sim.components['battery'].capacity_kwh == 5.0
        assert sim.components['battery'].max_power_kw == 3.0
        assert sim.components['battery'].get_soc() == 2.5
        
        # Check inverter
        assert sim.components['inverter'].capacity_kw == 12.0
    
    def test_calculate_power_balance(self):
        """Test power balance calculation."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Test with excess PV generation
        power_bal = sim._calculate_power_balance(
            pv_power=8.0,
            load_power=5.0,
            desired_battery_power=2.0,  # Charge battery
            dt_hours=1.0/60.0,
            t=0
        )
        
        assert power_bal.pv_power == 8.0
        assert power_bal.load_power == 5.0
        assert power_bal.battery_power == 2.0
        # Power imbalance: 8.0 + 2.0 - 5.0 = 5.0 kW
        assert power_bal.power_imbalance == 5.0
    
    def test_update_battery_state_charging(self):
        """Test battery state update during charging."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.0
        )
        
        sim = MicrogridSimulator(config)
        initial_soc = sim.components['battery'].get_soc()
        
        # Charge at 2 kW for 1 minute
        actual_power = sim._update_battery_state(desired_power=2.0, dt_hours=1.0/60.0)
        
        assert actual_power == 2.0
        assert sim.components['battery'].get_soc() > initial_soc
    
    def test_update_battery_state_discharging(self):
        """Test battery state update during discharging."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=3.0
        )
        
        sim = MicrogridSimulator(config)
        initial_soc = sim.components['battery'].get_soc()
        
        # Discharge at 2 kW for 1 minute
        actual_power = sim._update_battery_state(desired_power=-2.0, dt_hours=1.0/60.0)
        
        assert actual_power == -2.0
        assert sim.components['battery'].get_soc() < initial_soc
    
    def test_update_battery_state_idle(self):
        """Test battery state when idle."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        )
        
        sim = MicrogridSimulator(config)
        initial_soc = sim.components['battery'].get_soc()
        
        # No battery action
        actual_power = sim._update_battery_state(desired_power=0.0, dt_hours=1.0/60.0)
        
        assert actual_power == 0.0
        assert sim.components['battery'].get_soc() == initial_soc
    
    def test_calculate_frequency_deviation(self):
        """Test frequency deviation calculation."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Test with power surplus (should increase frequency)
        freq_dev_surplus = sim._calculate_frequency_deviation(power_imbalance=2.0)
        assert freq_dev_surplus > 0
        
        # Test with power deficit (should decrease frequency)
        freq_dev_deficit = sim._calculate_frequency_deviation(power_imbalance=-2.0)
        assert freq_dev_deficit < 0
        
        # Test with balanced power
        freq_dev_balanced = sim._calculate_frequency_deviation(power_imbalance=0.0)
        assert freq_dev_balanced == 0.0
    
    def test_calculate_voltage_deviation(self):
        """Test voltage deviation calculation."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Test with positive reactive power (should increase voltage)
        volt_dev_positive = sim._calculate_voltage_deviation(reactive_power=2.0)
        assert volt_dev_positive > 0
        
        # Test with negative reactive power (should decrease voltage)
        volt_dev_negative = sim._calculate_voltage_deviation(reactive_power=-2.0)
        assert volt_dev_negative < 0
        
        # Test with zero reactive power
        volt_dev_zero = sim._calculate_voltage_deviation(reactive_power=0.0)
        assert volt_dev_zero == 0.0
    
    def test_simulate_basic(self):
        """Test basic simulation run."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        )
        
        sim = MicrogridSimulator(config)
        
        # Create simple test data (10 timesteps)
        n_steps = 10
        pv_forecast = np.ones(n_steps) * 5.0
        actual_pv = np.ones(n_steps) * 5.0
        load_profile = np.ones(n_steps) * 4.0
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # Check result structure
        assert isinstance(result, SimulationResult)
        assert len(result.timestamps) == n_steps
        assert len(result.pv_power) == n_steps
        assert len(result.load_power) == n_steps
        assert len(result.battery_power) == n_steps
        assert len(result.battery_soc) == n_steps
        assert len(result.frequency_deviation) == n_steps
        assert len(result.voltage_deviation) == n_steps
        assert len(result.states) == n_steps
    
    def test_simulate_grid_connected_mode(self):
        """Test simulation in grid-connected mode."""
        config = MicrogridConfig(
            mode=MicrogridMode.GRID_CONNECTED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Create test data with power imbalance
        n_steps = 5
        actual_pv = np.array([8.0, 7.0, 6.0, 5.0, 4.0])
        load_profile = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        pv_forecast = actual_pv.copy()
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # In grid-connected mode, grid should absorb imbalances
        # So frequency deviation should be minimal or zero
        assert np.all(np.abs(result.frequency_deviation) < 0.5)
    
    def test_simulate_islanded_mode(self):
        """Test simulation in islanded mode."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Create test data with power imbalance
        n_steps = 5
        actual_pv = np.array([8.0, 7.0, 6.0, 5.0, 4.0])
        load_profile = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        pv_forecast = actual_pv.copy()
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # In islanded mode, there should be frequency deviations
        # (unless battery perfectly compensates)
        assert len(result.frequency_deviation) == n_steps
    
    def test_simulate_with_battery_dispatch(self):
        """Test simulation with pre-computed battery dispatch."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        n_steps = 5
        actual_pv = np.ones(n_steps) * 6.0
        load_profile = np.ones(n_steps) * 5.0
        pv_forecast = actual_pv.copy()
        
        # Pre-compute battery dispatch (charge at 1 kW)
        battery_dispatch = np.ones(n_steps) * 1.0
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60,
            battery_dispatch=battery_dispatch
        )
        
        # Battery should be charging
        assert np.all(result.battery_power > 0)
        # SOC should be increasing
        assert result.battery_soc[-1] > result.battery_soc[0]
    
    def test_simulate_frequency_limits(self):
        """Test that frequency deviation is constrained within limits."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        # Create extreme power imbalance
        n_steps = 5
        actual_pv = np.ones(n_steps) * 15.0  # Very high PV
        load_profile = np.ones(n_steps) * 2.0  # Low load
        pv_forecast = actual_pv.copy()
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # Frequency deviation should be within ±2 Hz
        assert np.all(result.frequency_deviation >= -2.0)
        assert np.all(result.frequency_deviation <= 2.0)
    
    def test_simulate_voltage_limits(self):
        """Test that voltage deviation is constrained within limits."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0
        )
        
        sim = MicrogridSimulator(config)
        
        n_steps = 5
        actual_pv = np.ones(n_steps) * 8.0
        load_profile = np.ones(n_steps) * 5.0
        pv_forecast = actual_pv.copy()
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # Voltage deviation should be within ±10%
        assert np.all(result.voltage_deviation >= -10.0)
        assert np.all(result.voltage_deviation <= 10.0)
    
    def test_simulate_soc_constraints(self):
        """Test that battery SOC respects constraints."""
        config = MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        )
        
        sim = MicrogridSimulator(config)
        
        # Long simulation with continuous charging
        n_steps = 100
        actual_pv = np.ones(n_steps) * 8.0  # High PV
        load_profile = np.ones(n_steps) * 3.0  # Low load
        pv_forecast = actual_pv.copy()
        
        result = sim.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60
        )
        
        # SOC should stay within limits (0.1 to 0.9 of capacity)
        assert np.all(result.battery_soc >= 0.5)  # 10% of 5.0 kWh
        assert np.all(result.battery_soc <= 4.5)  # 90% of 5.0 kWh
