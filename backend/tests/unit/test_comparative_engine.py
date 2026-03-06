"""
Unit tests for ComparativeEngine.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.comparative_engine import ComparativeEngine, ModelResult, ComparisonResult
from src.models.classical import ClassicalPredictor
from src.config.schemas import MicrogridConfig, MicrogridMode
from src.simulation.simulator import SimulationResult, PowerBalance, SystemState
from src.analysis.stability_analyzer import (
    StabilityMetrics, FrequencyMetrics, VoltageMetrics, 
    BatteryMetrics, PowerQualityMetrics, EnergyMetrics, ControlEffortMetrics
)


@pytest.fixture
def microgrid_config():
    """Create test microgrid configuration."""
    return MicrogridConfig(
        mode=MicrogridMode.ISLANDED,
        pv_capacity_kw=10.0,
        battery_capacity_kwh=5.0,
        battery_power_kw=3.0,
        inverter_capacity_kw=12.0,
        initial_soc_kwh=2.5
    )


@pytest.fixture
def test_data():
    """Generate synthetic test data."""
    n_samples = 100
    sequence_length = 12
    n_features = 5
    
    # Create test data
    test_data = np.random.randn(n_samples, sequence_length, n_features) * 0.5 + 0.5
    test_data = np.clip(test_data, 0, 1)
    
    # Create targets (actual PV power)
    test_targets = np.random.rand(n_samples) * 5.0
    
    # Create actual PV and load profiles
    actual_pv = test_targets.copy()
    load_profile = np.random.rand(n_samples) * 3.0 + 1.0
    
    return {
        'test_data': test_data,
        'test_targets': test_targets,
        'actual_pv': actual_pv,
        'load_profile': load_profile
    }


@pytest.fixture
def comparative_engine(microgrid_config):
    """Create ComparativeEngine instance."""
    return ComparativeEngine(
        microgrid_config=microgrid_config,
        ems_strategy='reactive'
    )


def test_comparative_engine_initialization(comparative_engine, microgrid_config):
    """Test ComparativeEngine initialization."""
    assert comparative_engine.microgrid_config == microgrid_config
    assert comparative_engine.ems_strategy == 'reactive'
    assert comparative_engine.simulator is not None
    assert comparative_engine.analyzer is not None
    assert comparative_engine.ems_controller is None  # Reactive mode


def test_run_no_forecast_baseline(comparative_engine, test_data):
    """Test running no-forecast baseline."""
    result = comparative_engine._run_no_forecast_baseline(
        actual_pv=test_data['actual_pv'],
        load_profile=test_data['load_profile']
    )
    
    assert isinstance(result, ModelResult)
    assert result.model_name == 'no_forecast'
    assert len(result.predictions) == len(test_data['actual_pv'])
    assert np.all(result.predictions == 0)
    assert result.prediction_metrics['mae'] == float('inf')
    assert result.simulation_result is not None
    assert result.stability_metrics is not None


def test_run_single_model(comparative_engine, test_data):
    """Test running a single model."""
    # Create a simple persistence model
    model_config = {'method': 'persistence'}
    model = ClassicalPredictor(model_config, method='persistence')
    model.build_model()
    
    # Train on small subset
    train_X = test_data['test_data'][:10]
    train_y = test_data['test_targets'][:10]
    model.train(train_X, train_y, train_X, train_y)
    
    # Run single model
    result = comparative_engine._run_single_model(
        model_name='persistence',
        model=model,
        test_data=test_data['test_data'],
        test_targets=test_data['test_targets'],
        actual_pv=test_data['actual_pv'],
        load_profile=test_data['load_profile']
    )
    
    assert isinstance(result, ModelResult)
    assert result.model_name == 'persistence'
    assert len(result.predictions) == len(test_data['test_targets'])
    assert 'mae' in result.prediction_metrics
    assert 'rmse' in result.prediction_metrics
    assert result.simulation_result is not None
    assert result.stability_metrics is not None


def test_calculate_improvements(comparative_engine, test_data):
    """Test improvement calculations."""
    # Create mock results
    baseline_metrics = StabilityMetrics(
        frequency=FrequencyMetrics(0.5, 0.3, 1.0, 0.1, 0.05),
        voltage=VoltageMetrics(2.0, 1.5, 5.0, 0.05),
        battery=BatteryMetrics(3.0, 10.0, 0.8, 0.5, 15.0),
        power_quality=PowerQualityMetrics(0.1, 0.95),
        energy=EnergyMetrics(0.5, 0.3, 0.9),
        control_effort=ControlEffortMetrics(50.0, 20)
    )
    
    improved_metrics = StabilityMetrics(
        frequency=FrequencyMetrics(0.3, 0.2, 0.7, 0.05, 0.03),
        voltage=VoltageMetrics(1.5, 1.0, 3.5, 0.02),
        battery=BatteryMetrics(2.5, 8.0, 0.7, 0.4, 12.0),
        power_quality=PowerQualityMetrics(0.08, 0.97),
        energy=EnergyMetrics(0.3, 0.2, 0.95),
        control_effort=ControlEffortMetrics(40.0, 15)
    )
    
    # Create mock simulation results
    n = 100
    mock_sim = SimulationResult(
        timestamps=np.arange(n),
        pv_power=np.random.rand(n),
        load_power=np.random.rand(n),
        battery_power=np.random.rand(n),
        battery_soc=np.random.rand(n),
        frequency_deviation=np.random.rand(n),
        voltage_deviation=np.random.rand(n),
        grid_power=np.zeros(n),
        states=[]
    )
    
    results = {
        'no_forecast': ModelResult(
            model_name='no_forecast',
            predictions=np.zeros(n),
            prediction_metrics={'mae': float('inf'), 'rmse': float('inf'), 
                              'mape': float('inf'), 'r2': -float('inf')},
            simulation_result=mock_sim,
            stability_metrics=baseline_metrics
        ),
        'model_a': ModelResult(
            model_name='model_a',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.5, 'rmse': 0.7, 'mape': 10.0, 'r2': 0.85},
            simulation_result=mock_sim,
            stability_metrics=improved_metrics
        )
    }
    
    improvements = comparative_engine.calculate_improvements(results, baseline='no_forecast')
    
    assert 'no_forecast' in improvements
    assert 'model_a' in improvements
    assert len(improvements['no_forecast']) == 0  # Baseline has no improvements
    assert len(improvements['model_a']) > 0
    
    # Check that improvement metrics exist
    model_a_improvements = improvements['model_a']
    assert 'freq_mean_abs_dev_improvement' in model_a_improvements
    assert 'freq_std_improvement' in model_a_improvements
    assert 'battery_cycles_improvement' in model_a_improvements


def test_calculate_rankings(comparative_engine):
    """Test model ranking calculations."""
    # Create mock results with different metric values
    n = 100
    mock_sim = SimulationResult(
        timestamps=np.arange(n),
        pv_power=np.random.rand(n),
        load_power=np.random.rand(n),
        battery_power=np.random.rand(n),
        battery_soc=np.random.rand(n),
        frequency_deviation=np.random.rand(n),
        voltage_deviation=np.random.rand(n),
        grid_power=np.zeros(n),
        states=[]
    )
    
    results = {
        'model_a': ModelResult(
            model_name='model_a',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.5, 'rmse': 0.7, 'mape': 10.0, 'r2': 0.85},
            simulation_result=mock_sim,
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.3, 0.2, 0.7, 0.05, 0.03),
                voltage=VoltageMetrics(1.5, 1.0, 3.5, 0.02),
                battery=BatteryMetrics(2.5, 8.0, 0.7, 0.4, 12.0),
                power_quality=PowerQualityMetrics(0.08, 0.97),
                energy=EnergyMetrics(0.3, 0.2, 0.95),
                control_effort=ControlEffortMetrics(40.0, 15)
            )
        ),
        'model_b': ModelResult(
            model_name='model_b',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.6, 'rmse': 0.8, 'mape': 12.0, 'r2': 0.80},
            simulation_result=mock_sim,
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.4, 0.25, 0.8, 0.08, 0.04),
                voltage=VoltageMetrics(1.8, 1.2, 4.0, 0.03),
                battery=BatteryMetrics(3.0, 9.0, 0.75, 0.45, 13.5),
                power_quality=PowerQualityMetrics(0.09, 0.96),
                energy=EnergyMetrics(0.35, 0.25, 0.93),
                control_effort=ControlEffortMetrics(45.0, 18)
            )
        )
    }
    
    rankings = comparative_engine._calculate_rankings(results)
    
    assert 'mae' in rankings
    assert 'rmse' in rankings
    assert 'r2' in rankings
    assert 'freq_stability' in rankings
    assert 'volt_stability' in rankings
    assert 'battery_stress' in rankings
    
    # Check that model_a ranks better than model_b for MAE
    assert rankings['mae'][0] == 'model_a'
    assert rankings['mae'][1] == 'model_b'
    
    # Check that model_a ranks better for R²
    assert rankings['r2'][0] == 'model_a'


def test_rank_models_by_metric(comparative_engine):
    """Test ranking models by specific metric."""
    n = 100
    mock_sim = SimulationResult(
        timestamps=np.arange(n),
        pv_power=np.random.rand(n),
        load_power=np.random.rand(n),
        battery_power=np.random.rand(n),
        battery_soc=np.random.rand(n),
        frequency_deviation=np.random.rand(n),
        voltage_deviation=np.random.rand(n),
        grid_power=np.zeros(n),
        states=[]
    )
    
    results = {
        'model_a': ModelResult(
            model_name='model_a',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.5, 'rmse': 0.7, 'mape': 10.0, 'r2': 0.85},
            simulation_result=mock_sim,
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.3, 0.2, 0.7, 0.05, 0.03),
                voltage=VoltageMetrics(1.5, 1.0, 3.5, 0.02),
                battery=BatteryMetrics(2.5, 8.0, 0.7, 0.4, 12.0),
                power_quality=PowerQualityMetrics(0.08, 0.97),
                energy=EnergyMetrics(0.3, 0.2, 0.95),
                control_effort=ControlEffortMetrics(40.0, 15)
            )
        ),
        'model_b': ModelResult(
            model_name='model_b',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.6, 'rmse': 0.8, 'mape': 12.0, 'r2': 0.80},
            simulation_result=mock_sim,
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.4, 0.25, 0.8, 0.08, 0.04),
                voltage=VoltageMetrics(1.8, 1.2, 4.0, 0.03),
                battery=BatteryMetrics(3.0, 9.0, 0.75, 0.45, 13.5),
                power_quality=PowerQualityMetrics(0.09, 0.96),
                energy=EnergyMetrics(0.35, 0.25, 0.93),
                control_effort=ControlEffortMetrics(45.0, 18)
            )
        )
    }
    
    # Rank by MAE
    ranked = comparative_engine.rank_models(results, 'mae')
    assert len(ranked) == 2
    assert ranked[0][0] == 'model_a'
    assert ranked[0][1] == 0.5
    
    # Rank by R²
    ranked = comparative_engine.rank_models(results, 'r2')
    assert len(ranked) == 2
    assert ranked[0][0] == 'model_a'
    assert ranked[0][1] == 0.85


def test_statistical_significance(comparative_engine):
    """Test statistical significance testing."""
    n = 100
    
    # Create results with different frequency deviations
    results = {
        'model_a': ModelResult(
            model_name='model_a',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.5, 'rmse': 0.7, 'mape': 10.0, 'r2': 0.85},
            simulation_result=SimulationResult(
                timestamps=np.arange(n),
                pv_power=np.random.rand(n),
                load_power=np.random.rand(n),
                battery_power=np.random.rand(n),
                battery_soc=np.random.rand(n),
                frequency_deviation=np.random.randn(n) * 0.2,  # Lower deviation
                voltage_deviation=np.random.randn(n) * 1.0,
                grid_power=np.zeros(n),
                states=[]
            ),
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.3, 0.2, 0.7, 0.05, 0.03),
                voltage=VoltageMetrics(1.5, 1.0, 3.5, 0.02),
                battery=BatteryMetrics(2.5, 8.0, 0.7, 0.4, 12.0),
                power_quality=PowerQualityMetrics(0.08, 0.97),
                energy=EnergyMetrics(0.3, 0.2, 0.95),
                control_effort=ControlEffortMetrics(40.0, 15)
            )
        ),
        'model_b': ModelResult(
            model_name='model_b',
            predictions=np.random.rand(n),
            prediction_metrics={'mae': 0.6, 'rmse': 0.8, 'mape': 12.0, 'r2': 0.80},
            simulation_result=SimulationResult(
                timestamps=np.arange(n),
                pv_power=np.random.rand(n),
                load_power=np.random.rand(n),
                battery_power=np.random.rand(n),
                battery_soc=np.random.rand(n),
                frequency_deviation=np.random.randn(n) * 0.5,  # Higher deviation
                voltage_deviation=np.random.randn(n) * 1.5,
                grid_power=np.zeros(n),
                states=[]
            ),
            stability_metrics=StabilityMetrics(
                frequency=FrequencyMetrics(0.4, 0.25, 0.8, 0.08, 0.04),
                voltage=VoltageMetrics(1.8, 1.2, 4.0, 0.03),
                battery=BatteryMetrics(3.0, 9.0, 0.75, 0.45, 13.5),
                power_quality=PowerQualityMetrics(0.09, 0.96),
                energy=EnergyMetrics(0.35, 0.25, 0.93),
                control_effort=ControlEffortMetrics(45.0, 18)
            )
        )
    }
    
    sig_test = comparative_engine.statistical_significance(
        results=results,
        model_a='model_a',
        model_b='model_b',
        metric='freq_std'
    )
    
    assert sig_test.model_a == 'model_a'
    assert sig_test.model_b == 'model_b'
    assert sig_test.metric == 'Frequency Deviation'
    assert isinstance(sig_test.statistic, float)
    assert isinstance(sig_test.p_value, float)
    assert isinstance(sig_test.significant, bool)
    assert len(sig_test.interpretation) > 0


def test_generate_comparison_table(comparative_engine):
    """Test comparison table generation."""
    n = 100
    mock_sim = SimulationResult(
        timestamps=np.arange(n),
        pv_power=np.random.rand(n),
        load_power=np.random.rand(n),
        battery_power=np.random.rand(n),
        battery_soc=np.random.rand(n),
        frequency_deviation=np.random.rand(n),
        voltage_deviation=np.random.rand(n),
        grid_power=np.zeros(n),
        states=[]
    )
    
    comparison_result = ComparisonResult(
        models={
            'model_a': ModelResult(
                model_name='model_a',
                predictions=np.random.rand(n),
                prediction_metrics={'mae': 0.5, 'rmse': 0.7, 'mape': 10.0, 'r2': 0.85},
                simulation_result=mock_sim,
                stability_metrics=StabilityMetrics(
                    frequency=FrequencyMetrics(0.3, 0.2, 0.7, 0.05, 0.03),
                    voltage=VoltageMetrics(1.5, 1.0, 3.5, 0.02),
                    battery=BatteryMetrics(2.5, 8.0, 0.7, 0.4, 12.0),
                    power_quality=PowerQualityMetrics(0.08, 0.97),
                    energy=EnergyMetrics(0.3, 0.2, 0.95),
                    control_effort=ControlEffortMetrics(40.0, 15)
                )
            )
        },
        rankings={'mae': ['model_a']},
        improvements={'model_a': {}},
        baseline_model='no_forecast'
    )
    
    table = comparative_engine.generate_comparison_table(comparison_result)
    
    assert 'models' in table
    assert 'rankings' in table
    assert 'improvements' in table
    assert 'model_a' in table['models']
    assert 'prediction_metrics' in table['models']['model_a']
    assert 'stability_metrics' in table['models']['model_a']
