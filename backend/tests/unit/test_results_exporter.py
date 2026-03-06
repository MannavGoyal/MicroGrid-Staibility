"""
Unit tests for ResultsExporter class.
"""

import pytest
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from backend.src.analysis.results_exporter import ResultsExporter
from backend.src.analysis.comparative_engine import ComparisonResult, ModelResult
from backend.src.analysis.stability_analyzer import (
    StabilityMetrics, FrequencyMetrics, VoltageMetrics,
    BatteryMetrics, PowerQualityMetrics, EnergyMetrics, ControlEffortMetrics
)
from backend.src.simulation.simulator import SimulationResult
from backend.src.config.schemas import (
    Configuration, ModelConfig, MicrogridConfig, TrainingConfig,
    ForecastHorizon, ModelType, MicrogridMode
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def exporter(temp_output_dir):
    """Create ResultsExporter instance with temporary directory."""
    return ResultsExporter(output_dir=temp_output_dir)


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return Configuration(
        experiment_name="test_experiment",
        forecast_horizon=ForecastHorizon.FIFTEEN_MIN,
        model_configuration=ModelConfig(
            model_type=ModelType.LSTM,
            hyperparameters={"hidden_size": 64, "num_layers": 2},
            sequence_length=12
        ),
        microgrid_configuration=MicrogridConfig(
            mode=MicrogridMode.ISLANDED,
            pv_capacity_kw=10.0,
            battery_capacity_kwh=5.0,
            battery_power_kw=3.0,
            inverter_capacity_kw=12.0,
            initial_soc_kwh=2.5
        ),
        training_configuration=TrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.001
        ),
        data_path="data/test.csv",
        output_dir="results"
    )


@pytest.fixture
def sample_timeseries_data():
    """Create sample time-series data."""
    n_samples = 100
    return {
        'actual_pv': np.random.rand(n_samples) * 10,
        'predicted_pv': np.random.rand(n_samples) * 10,
        'load': np.random.rand(n_samples) * 5 + 3,
        'battery_soc': np.random.rand(n_samples) * 5,
        'frequency_deviation': np.random.randn(n_samples) * 0.1
    }


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        'prediction_metrics': {
            'mae': 0.0456,
            'rmse': 0.0623,
            'mape': 12.34,
            'r2': 0.8912
        },
        'stability_metrics': {
            'frequency': {
                'mean_abs_dev': 0.123,
                'std_dev': 0.089,
                'max_dev': 0.456
            },
            'voltage': {
                'mean_abs_dev': 1.23,
                'std_dev': 0.89,
                'max_dev': 4.56
            }
        }
    }


@pytest.fixture
def sample_comparison_result():
    """Create sample comparison result."""
    # Create minimal simulation result
    n_samples = 50
    sim_result = SimulationResult(
        timestamps=np.arange(n_samples) * 60,
        pv_power=np.random.rand(n_samples) * 10,
        load_power=np.random.rand(n_samples) * 5 + 3,
        battery_power=np.random.randn(n_samples) * 2,
        battery_soc=np.random.rand(n_samples) * 5,
        frequency_deviation=np.random.randn(n_samples) * 0.1,
        voltage_deviation=np.random.randn(n_samples) * 2,
        grid_power=np.zeros(n_samples),  # Add grid_power
        states=[]  # Add empty states list
    )
    
    # Create stability metrics
    stability_metrics = StabilityMetrics(
        frequency=FrequencyMetrics(
            mean_absolute_deviation=0.123,
            std_deviation=0.089,
            max_deviation=0.456,
            time_outside_limits=0.05,
            rate_of_change=0.012
        ),
        voltage=VoltageMetrics(
            mean_absolute_deviation=1.23,
            std_deviation=0.89,
            max_deviation=4.56,
            time_outside_limits=0.02
        ),
        battery=BatteryMetrics(
            soc_range=3.5,
            num_cycles=5.2,
            max_depth_of_discharge=0.65,
            avg_c_rate=0.3,
            total_throughput=12.5
        ),
        power_quality=PowerQualityMetrics(
            thd_proxy=0.15,
            power_factor=0.95
        ),
        energy=EnergyMetrics(
            total_unmet_load=0.5,
            total_curtailed_pv=1.2,
            energy_efficiency=0.92
        ),
        control_effort=ControlEffortMetrics(
            sum_absolute_battery_changes=45.6,
            num_control_actions=23
        )
    )
    
    # Create model results
    model_result = ModelResult(
        model_name='lstm',
        predictions=np.random.rand(n_samples) * 10,
        prediction_metrics={
            'mae': 0.0456,
            'rmse': 0.0623,
            'mape': 12.34,
            'r2': 0.8912
        },
        simulation_result=sim_result,
        stability_metrics=stability_metrics
    )
    
    baseline_result = ModelResult(
        model_name='no_forecast',
        predictions=np.zeros(n_samples),
        prediction_metrics={
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
            'r2': -float('inf')
        },
        simulation_result=sim_result,
        stability_metrics=StabilityMetrics(
            frequency=FrequencyMetrics(
                mean_absolute_deviation=0.234,
                std_deviation=0.156,
                max_deviation=0.678,
                time_outside_limits=0.12,
                rate_of_change=0.023
            ),
            voltage=VoltageMetrics(
                mean_absolute_deviation=2.34,
                std_deviation=1.56,
                max_deviation=6.78,
                time_outside_limits=0.08
            ),
            battery=BatteryMetrics(
                soc_range=4.2,
                num_cycles=8.5,
                max_depth_of_discharge=0.85,
                avg_c_rate=0.5,
                total_throughput=18.3
            ),
            power_quality=PowerQualityMetrics(
                thd_proxy=0.25,
                power_factor=0.88
            ),
            energy=EnergyMetrics(
                total_unmet_load=1.2,
                total_curtailed_pv=2.5,
                energy_efficiency=0.85
            ),
            control_effort=ControlEffortMetrics(
                sum_absolute_battery_changes=78.9,
                num_control_actions=45
            )
        )
    )
    
    return ComparisonResult(
        models={'lstm': model_result, 'no_forecast': baseline_result},
        rankings={
            'mae': ['lstm'],
            'freq_stability': ['lstm', 'no_forecast'],
            'battery_stress': ['lstm', 'no_forecast']
        },
        improvements={
            'lstm': {
                'freq_std_improvement': 42.9,
                'battery_cycles_improvement': 38.8
            }
        },
        baseline_model='no_forecast'
    )


class TestResultsExporter:
    """Test suite for ResultsExporter class."""
    
    def test_initialization(self, temp_output_dir):
        """Test exporter initialization creates output directory."""
        exporter = ResultsExporter(output_dir=temp_output_dir)
        assert exporter.output_dir.exists()
        assert exporter.output_dir.is_dir()
    
    def test_export_timeseries_csv(self, exporter, sample_timeseries_data):
        """Test exporting time-series data to CSV."""
        output_path = exporter.export_timeseries(
            data=sample_timeseries_data,
            filename='test_timeseries',
            format='csv'
        )
        
        # Check file exists
        assert Path(output_path).exists()
        
        # Read and verify CSV content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check number of rows
            assert len(rows) == len(sample_timeseries_data['actual_pv'])
            
            # Check headers
            assert 'timestamp' in rows[0]
            assert 'actual_pv' in rows[0]
            assert 'predicted_pv' in rows[0]
            
            # Check data types (should be strings in CSV)
            assert isinstance(rows[0]['actual_pv'], str)
    
    def test_export_timeseries_with_timestamps(self, exporter, sample_timeseries_data):
        """Test exporting time-series data with custom timestamps."""
        n_samples = len(sample_timeseries_data['actual_pv'])
        timestamps = np.arange(n_samples) * 300  # 5-minute intervals
        
        output_path = exporter.export_timeseries(
            data=sample_timeseries_data,
            filename='test_timeseries_custom',
            format='csv',
            timestamps=timestamps
        )
        
        assert Path(output_path).exists()
    
    def test_export_timeseries_invalid_format(self, exporter, sample_timeseries_data):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_timeseries(
                data=sample_timeseries_data,
                filename='test',
                format='xml'
            )
    
    def test_export_metrics_json(self, exporter, sample_metrics):
        """Test exporting metrics to JSON."""
        output_path = exporter.export_metrics(
            metrics=sample_metrics,
            filename='test_metrics',
            format='json'
        )
        
        # Check file exists
        assert Path(output_path).exists()
        
        # Read and verify JSON content
        with open(output_path, 'r') as f:
            loaded_metrics = json.load(f)
            
            assert 'prediction_metrics' in loaded_metrics
            assert loaded_metrics['prediction_metrics']['mae'] == 0.0456
            assert loaded_metrics['stability_metrics']['frequency']['mean_abs_dev'] == 0.123
    
    def test_export_metrics_csv(self, exporter, sample_metrics):
        """Test exporting metrics to CSV."""
        output_path = exporter.export_metrics(
            metrics=sample_metrics,
            filename='test_metrics',
            format='csv'
        )
        
        # Check file exists
        assert Path(output_path).exists()
        
        # Read and verify CSV content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check that nested keys are flattened
            metric_names = [row['metric'] for row in rows]
            assert 'prediction_metrics.mae' in metric_names
            assert 'stability_metrics.frequency.mean_abs_dev' in metric_names
    
    def test_export_metrics_with_numpy_types(self, exporter):
        """Test exporting metrics with numpy types."""
        metrics = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3])
        }
        
        output_path = exporter.export_metrics(
            metrics=metrics,
            filename='test_numpy',
            format='json'
        )
        
        # Should not raise error and should be JSON serializable
        with open(output_path, 'r') as f:
            loaded = json.load(f)
            assert loaded['numpy_int'] == 42
            assert loaded['numpy_float'] == 3.14
            assert loaded['numpy_array'] == [1, 2, 3]
    
    def test_export_visualizations_png(self, exporter):
        """Test exporting visualizations as PNG."""
        # Create sample figures
        figures = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            ax.set_title(f'Test Plot {i+1}')
            figures.append(fig)
        
        output_paths = exporter.export_visualizations(
            figures=figures,
            prefix='test_plot',
            format='png'
        )
        
        # Check all files exist
        assert len(output_paths) == 3
        for path in output_paths:
            assert Path(path).exists()
            assert path.endswith('.png')
    
    def test_export_visualizations_svg(self, exporter):
        """Test exporting visualizations as SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_paths = exporter.export_visualizations(
            figures=[fig],
            prefix='test_svg',
            format='svg'
        )
        
        assert len(output_paths) == 1
        assert Path(output_paths[0]).exists()
        assert output_paths[0].endswith('.svg')
    
    def test_export_visualizations_invalid_format(self, exporter):
        """Test that invalid format raises error."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_visualizations(
                figures=[fig],
                prefix='test',
                format='pdf'
            )
    
    def test_export_configuration(self, exporter, sample_config):
        """Test exporting configuration."""
        output_path = exporter.export_configuration(
            config=sample_config,
            filename='test_config'
        )
        
        # Check file exists
        assert Path(output_path).exists()
        
        # Read and verify JSON content
        with open(output_path, 'r') as f:
            loaded_config = json.load(f)
            
            assert loaded_config['experiment_name'] == 'test_experiment'
            assert loaded_config['forecast_horizon'] == '15min'
            assert loaded_config['model_configuration']['model_type'] == 'lstm'
            assert loaded_config['microgrid_configuration']['battery_capacity_kwh'] == 5.0
    
    def test_generate_markdown_report(self, exporter, sample_comparison_result, sample_config):
        """Test generating Markdown report."""
        output_path = exporter.generate_report(
            comparison_result=sample_comparison_result,
            config=sample_config,
            format='markdown'
        )
        
        # Check file exists
        assert Path(output_path).exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
            
            # Check for key sections
            assert '# Microgrid Stability Analysis Report' in content
            assert '## Configuration' in content
            assert '## Prediction Performance' in content
            assert '## Stability Improvements' in content
            assert '## Model Rankings' in content
            assert '## Key Findings' in content
            
            # Check for configuration details
            assert 'test_experiment' in content
            assert '15min' in content
            assert 'lstm' in content
            
            # Check for metrics tables
            assert '| Model |' in content
            assert '| lstm |' in content
    
    def test_generate_report_pdf_not_implemented(self, exporter, sample_comparison_result, sample_config):
        """Test that PDF report generation raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            exporter.generate_report(
                comparison_result=sample_comparison_result,
                config=sample_config,
                format='pdf'
            )
    
    def test_create_export_directory(self, exporter):
        """Test creating timestamped export directory."""
        export_dir = exporter.create_export_directory('my_experiment')
        
        # Check directory exists
        assert export_dir.exists()
        assert export_dir.is_dir()
        
        # Check directory name format
        dir_name = export_dir.name
        assert dir_name.startswith('my_experiment_')
        assert len(dir_name) > len('my_experiment_')
        
        # Check that output_dir is updated
        assert exporter.output_dir == export_dir
    
    def test_make_json_serializable(self, exporter):
        """Test JSON serialization helper."""
        obj = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.float32(2.71)
            },
            'list': [np.int32(1), np.int32(2)]
        }
        
        result = exporter._make_json_serializable(obj)
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
        
        # Check types
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['array'], list)
        assert isinstance(result['nested']['value'], float)
    
    def test_flatten_dict(self, exporter):
        """Test dictionary flattening."""
        nested_dict = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            },
            'f': 4
        }
        
        flattened = exporter._flatten_dict(nested_dict)
        
        assert flattened['a'] == 1
        assert flattened['b.c'] == 2
        assert flattened['b.d.e'] == 3
        assert flattened['f'] == 4
        assert len(flattened) == 4
    
    def test_export_empty_timeseries(self, exporter):
        """Test exporting empty time-series data."""
        empty_data = {
            'col1': np.array([]),
            'col2': np.array([])
        }
        
        # Should handle empty arrays gracefully
        # This might raise an error or create an empty file depending on implementation
        # For now, we expect it to work with 0 rows
        try:
            output_path = exporter.export_timeseries(
                data=empty_data,
                filename='empty_test',
                format='csv'
            )
            # If it succeeds, check the file
            assert Path(output_path).exists()
        except (ValueError, IndexError):
            # If it raises an error, that's also acceptable behavior
            pass
    
    def test_export_metrics_invalid_format(self, exporter, sample_metrics):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_metrics(
                metrics=sample_metrics,
                filename='test',
                format='yaml'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
