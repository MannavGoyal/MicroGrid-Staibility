"""
Unit tests for StabilityAnalyzer.
"""

import pytest
import numpy as np

from backend.src.analysis.stability_analyzer import (
    StabilityAnalyzer,
    FrequencyMetrics,
    VoltageMetrics,
    BatteryMetrics,
    PowerQualityMetrics,
    EnergyMetrics,
    ControlEffortMetrics
)
from backend.src.simulation.simulator import SimulationResult


class TestStabilityAnalyzer:
    """Test suite for StabilityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StabilityAnalyzer(battery_capacity_kwh=5.0)
    
    @pytest.fixture
    def simple_sim_result(self):
        """Create simple simulation result for testing."""
        n_steps = 100
        timestamps = np.arange(n_steps) * 60.0  # 1-minute timesteps
        
        return SimulationResult(
            timestamps=timestamps,
            pv_power=np.ones(n_steps) * 2.0,
            load_power=np.ones(n_steps) * 1.5,
            battery_power=np.zeros(n_steps),
            battery_soc=np.ones(n_steps) * 2.5,
            frequency_deviation=np.zeros(n_steps),
            voltage_deviation=np.zeros(n_steps),
            grid_power=np.zeros(n_steps),
            states=[]
        )
    
    def test_frequency_metrics_zero_deviation(self, analyzer):
        """Test frequency metrics with zero deviation."""
        freq_dev = np.zeros(100)
        metrics = analyzer.frequency_metrics(freq_dev)
        
        assert metrics.mean_absolute_deviation == 0.0
        assert metrics.std_deviation == 0.0
        assert metrics.max_deviation == 0.0
        assert metrics.time_outside_limits == 0.0
        assert metrics.rate_of_change == 0.0
    
    def test_frequency_metrics_constant_deviation(self, analyzer):
        """Test frequency metrics with constant deviation."""
        freq_dev = np.ones(100) * 0.3
        metrics = analyzer.frequency_metrics(freq_dev)
        
        assert metrics.mean_absolute_deviation == 0.3
        assert metrics.std_deviation == 0.0
        assert metrics.max_deviation == 0.3
        assert metrics.time_outside_limits == 0.0  # Below 0.5 Hz limit
        assert metrics.rate_of_change == 0.0
    
    def test_frequency_metrics_exceeds_limits(self, analyzer):
        """Test frequency metrics when exceeding limits."""
        freq_dev = np.ones(100) * 0.6  # Above 0.5 Hz limit
        metrics = analyzer.frequency_metrics(freq_dev)
        
        assert metrics.time_outside_limits == 1.0  # All timesteps outside
    
    def test_voltage_metrics_zero_deviation(self, analyzer):
        """Test voltage metrics with zero deviation."""
        volt_dev = np.zeros(100)
        metrics = analyzer.voltage_metrics(volt_dev)
        
        assert metrics.mean_absolute_deviation == 0.0
        assert metrics.std_deviation == 0.0
        assert metrics.max_deviation == 0.0
        assert metrics.time_outside_limits == 0.0
    
    def test_voltage_metrics_exceeds_limits(self, analyzer):
        """Test voltage metrics when exceeding limits."""
        volt_dev = np.ones(100) * 6.0  # Above 5% limit
        metrics = analyzer.voltage_metrics(volt_dev)
        
        assert metrics.time_outside_limits == 1.0  # All timesteps outside
    
    def test_battery_stress_metrics_no_usage(self, analyzer):
        """Test battery metrics with no usage."""
        soc = np.ones(100) * 2.5
        power = np.zeros(100)
        timestamps = np.arange(100) * 60.0
        
        metrics = analyzer.battery_stress_metrics(soc, power, timestamps)
        
        assert metrics.soc_range == 0.0
        assert metrics.num_cycles == 0.0
        assert metrics.avg_c_rate == 0.0
        assert metrics.total_throughput == 0.0
    
    def test_battery_stress_metrics_with_cycling(self, analyzer):
        """Test battery metrics with charge/discharge cycling."""
        # Create a cycling pattern
        soc = np.array([2.0, 2.5, 3.0, 2.5, 2.0, 2.5, 3.0])
        power = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0])
        timestamps = np.arange(7) * 60.0
        
        metrics = analyzer.battery_stress_metrics(soc, power, timestamps)
        
        assert metrics.soc_range == 1.0  # 3.0 - 2.0
        assert metrics.num_cycles > 0.0  # Should detect cycles
        assert metrics.avg_c_rate > 0.0
        assert metrics.total_throughput > 0.0
    
    def test_battery_stress_max_dod(self, analyzer):
        """Test maximum depth of discharge calculation."""
        soc = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Discharge from full to 1 kWh
        power = np.ones(5) * -1.0
        timestamps = np.arange(5) * 60.0
        
        metrics = analyzer.battery_stress_metrics(soc, power, timestamps)
        
        # DOD = 1 - (min_SOC / capacity) = 1 - (1.0 / 5.0) = 0.8
        assert abs(metrics.max_depth_of_discharge - 0.8) < 0.01
    
    def test_power_quality_metrics_perfect_balance(self, analyzer):
        """Test power quality with perfect power balance."""
        pv_power = np.ones(100) * 2.0
        battery_power = np.zeros(100)
        load_power = np.ones(100) * 2.0
        
        metrics = analyzer.power_quality_metrics(pv_power, battery_power, load_power)
        
        assert metrics.thd_proxy == 0.0  # Perfect balance
        assert metrics.power_factor > 0.9  # High power factor
    
    def test_power_quality_metrics_with_imbalance(self, analyzer):
        """Test power quality with power imbalance."""
        pv_power = np.random.uniform(1.0, 3.0, 100)
        battery_power = np.zeros(100)
        load_power = np.ones(100) * 2.0
        
        metrics = analyzer.power_quality_metrics(pv_power, battery_power, load_power)
        
        assert metrics.thd_proxy > 0.0  # Some distortion
        assert 0.7 <= metrics.power_factor <= 1.0
    
    def test_energy_balance_metrics_no_losses(self, analyzer):
        """Test energy balance with no unmet load or curtailment."""
        pv_power = np.ones(100) * 2.0
        battery_power = np.zeros(100)
        load_power = np.ones(100) * 2.0
        timestamps = np.arange(100) * 60.0
        
        metrics = analyzer.energy_balance_metrics(pv_power, battery_power, load_power, timestamps)
        
        assert metrics.total_unmet_load == 0.0
        assert metrics.energy_efficiency == 1.0
    
    def test_energy_balance_metrics_with_unmet_load(self, analyzer):
        """Test energy balance with unmet load."""
        pv_power = np.ones(100) * 1.0
        battery_power = np.zeros(100)
        load_power = np.ones(100) * 2.0  # Load exceeds generation
        timestamps = np.arange(100) * 60.0
        
        metrics = analyzer.energy_balance_metrics(pv_power, battery_power, load_power, timestamps)
        
        assert metrics.total_unmet_load > 0.0
    
    def test_energy_balance_metrics_with_curtailment(self, analyzer):
        """Test energy balance with PV curtailment."""
        pv_power = np.ones(100) * 3.0
        battery_power = np.zeros(100)
        load_power = np.ones(100) * 1.0  # Generation exceeds load
        timestamps = np.arange(100) * 60.0
        
        metrics = analyzer.energy_balance_metrics(pv_power, battery_power, load_power, timestamps)
        
        assert metrics.total_curtailed_pv > 0.0
    
    def test_control_effort_metrics_no_changes(self, analyzer):
        """Test control effort with no battery power changes."""
        battery_power = np.ones(100) * 1.0  # Constant power
        
        metrics = analyzer.control_effort_metrics(battery_power)
        
        assert metrics.sum_absolute_battery_changes == 0.0
        assert metrics.num_control_actions == 0
    
    def test_control_effort_metrics_with_changes(self, analyzer):
        """Test control effort with battery power changes."""
        battery_power = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.0])
        
        metrics = analyzer.control_effort_metrics(battery_power)
        
        assert metrics.sum_absolute_battery_changes > 0.0
        assert metrics.num_control_actions > 0
    
    def test_analyze_complete(self, analyzer, simple_sim_result):
        """Test complete analysis with all metrics."""
        metrics = analyzer.analyze(simple_sim_result)
        
        # Check that all metric categories are present
        assert isinstance(metrics.frequency, FrequencyMetrics)
        assert isinstance(metrics.voltage, VoltageMetrics)
        assert isinstance(metrics.battery, BatteryMetrics)
        assert isinstance(metrics.power_quality, PowerQualityMetrics)
        assert isinstance(metrics.energy, EnergyMetrics)
        assert isinstance(metrics.control_effort, ControlEffortMetrics)
    
    def test_frequency_metrics_with_varying_deviation(self, analyzer):
        """Test frequency metrics with varying deviation."""
        freq_dev = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.4
        metrics = analyzer.frequency_metrics(freq_dev)
        
        assert metrics.mean_absolute_deviation > 0.0
        assert metrics.std_deviation > 0.0
        assert metrics.rate_of_change > 0.0
    
    def test_cycle_counting_single_cycle(self, analyzer):
        """Test cycle counting with a single charge-discharge cycle."""
        # One complete cycle: charge up, discharge down
        soc = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 3.5, 3.0, 2.5, 2.0])
        
        num_cycles = analyzer._count_cycles(soc)
        
        # Should detect approximately 1 cycle
        assert 0.5 <= num_cycles <= 2.0
    
    def test_edge_case_single_timestep(self, analyzer):
        """Test metrics with single timestep (edge case)."""
        freq_dev = np.array([0.1])
        metrics = analyzer.frequency_metrics(freq_dev)
        
        assert metrics.mean_absolute_deviation == 0.1
        assert metrics.rate_of_change == 0.0  # No change with single point
    
    def test_edge_case_empty_arrays(self, analyzer):
        """Test that empty arrays are handled gracefully."""
        # This should not crash
        freq_dev = np.array([])
        
        # NumPy operations on empty arrays should return valid values
        # (though they may be NaN or 0)
        try:
            metrics = analyzer.frequency_metrics(freq_dev)
            # If it doesn't crash, that's good enough
            assert True
        except Exception:
            pytest.fail("Should handle empty arrays gracefully")
