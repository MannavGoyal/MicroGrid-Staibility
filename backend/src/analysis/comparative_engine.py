"""
Comparative analysis engine for evaluating multiple forecasting approaches.

This module orchestrates the comparison of different prediction models by running
them on identical test data, simulating microgrid behavior with each forecast,
and calculating relative improvements over baseline approaches.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats

from src.models.base import BasePredictor
from src.simulation.simulator import MicrogridSimulator, SimulationResult
from src.simulation.ems_controller import EMSController
from src.analysis.stability_analyzer import StabilityAnalyzer, StabilityMetrics
from src.config.schemas import MicrogridConfig


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    predictions: np.ndarray
    prediction_metrics: Dict[str, float]
    simulation_result: SimulationResult
    stability_metrics: StabilityMetrics


@dataclass
class ComparisonResult:
    """Complete comparison results across all models."""
    models: Dict[str, ModelResult]
    rankings: Dict[str, List[str]]  # metric_name -> ranked list of model names
    improvements: Dict[str, Dict[str, float]]  # model_name -> {metric: improvement %}
    baseline_model: str


@dataclass
class SignificanceTest:
    """Statistical significance test results."""
    model_a: str
    model_b: str
    metric: str
    statistic: float
    p_value: float
    significant: bool  # True if p < 0.05
    interpretation: str


class ComparativeEngine:
    """
    Engine for systematic comparison of multiple forecasting approaches.
    
    Executes simulations for multiple models on identical test data and
    calculates comparative metrics to quantify improvements.
    """
    
    def __init__(self, microgrid_config: MicrogridConfig, ems_strategy: str = 'mpc'):
        """
        Initialize comparative engine.
        
        Args:
            microgrid_config: Microgrid configuration for simulations
            ems_strategy: EMS control strategy ('mpc', 'rule_based', 'reactive')
        """
        self.microgrid_config = microgrid_config
        self.ems_strategy = ems_strategy
        self.simulator = MicrogridSimulator(microgrid_config)
        self.analyzer = StabilityAnalyzer(microgrid_config.battery_capacity_kwh)
        
        if ems_strategy != 'reactive':
            self.ems_controller = EMSController(
                battery_capacity_kwh=microgrid_config.battery_capacity_kwh,
                battery_power_kw=microgrid_config.battery_power_kw,
                strategy=ems_strategy
            )
        else:
            self.ems_controller = None
    
    def run_comparison(
        self,
        models: Dict[str, BasePredictor],
        test_data: np.ndarray,
        test_targets: np.ndarray,
        actual_pv: np.ndarray,
        load_profile: np.ndarray,
        baseline: str = 'no_forecast'
    ) -> ComparisonResult:
        """
        Run all models and compare results.
        
        Args:
            models: Dictionary mapping model names to predictor instances
            test_data: Test input features, shape (n_samples, sequence_length, n_features)
            test_targets: Test target values (actual PV power), shape (n_samples,)
            actual_pv: Actual PV power for simulation, shape (n_samples,)
            load_profile: Load demand profile, shape (n_samples,)
            baseline: Name of baseline model for improvement calculations
        
        Returns:
            ComparisonResult containing all comparison data
        """
        results = {}
        
        # Run each model
        for model_name, model in models.items():
            print(f"Running model: {model_name}")
            result = self._run_single_model(
                model_name=model_name,
                model=model,
                test_data=test_data,
                test_targets=test_targets,
                actual_pv=actual_pv,
                load_profile=load_profile
            )
            results[model_name] = result
        
        # Add no-forecast baseline if not already included
        if 'no_forecast' not in results:
            print("Running no-forecast baseline")
            no_forecast_result = self._run_no_forecast_baseline(
                actual_pv=actual_pv,
                load_profile=load_profile
            )
            results['no_forecast'] = no_forecast_result
        
        # Calculate rankings
        rankings = self._calculate_rankings(results)
        
        # Calculate improvements relative to baseline
        improvements = self.calculate_improvements(results, baseline)
        
        return ComparisonResult(
            models=results,
            rankings=rankings,
            improvements=improvements,
            baseline_model=baseline
        )
    
    def _run_single_model(
        self,
        model_name: str,
        model: BasePredictor,
        test_data: np.ndarray,
        test_targets: np.ndarray,
        actual_pv: np.ndarray,
        load_profile: np.ndarray
    ) -> ModelResult:
        """
        Run single model and simulation.
        
        Args:
            model_name: Name of the model
            model: Predictor instance
            test_data: Test input features
            test_targets: Test target values
            actual_pv: Actual PV power for simulation
            load_profile: Load demand profile
        
        Returns:
            ModelResult containing predictions, metrics, and simulation results
        """
        # Generate predictions
        predictions = model.predict(test_data)
        
        # Calculate prediction metrics
        prediction_metrics = model.evaluate(test_targets, predictions)
        
        # Compute battery dispatch using EMS controller if available
        if self.ems_controller is not None:
            battery_dispatch = self._compute_battery_dispatch(
                predictions, actual_pv, load_profile
            )
        else:
            battery_dispatch = None
        
        # Run simulation
        simulation_result = self.simulator.simulate(
            pv_forecast=predictions,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60,
            battery_dispatch=battery_dispatch
        )
        
        # Analyze stability
        stability_metrics = self.analyzer.analyze(simulation_result)
        
        return ModelResult(
            model_name=model_name,
            predictions=predictions,
            prediction_metrics=prediction_metrics,
            simulation_result=simulation_result,
            stability_metrics=stability_metrics
        )
    
    def _run_no_forecast_baseline(
        self,
        actual_pv: np.ndarray,
        load_profile: np.ndarray
    ) -> ModelResult:
        """
        Run no-forecast baseline (reactive control only).
        
        Args:
            actual_pv: Actual PV power
            load_profile: Load demand profile
        
        Returns:
            ModelResult for no-forecast baseline
        """
        # No predictions - use zeros as placeholder
        predictions = np.zeros_like(actual_pv)
        
        # No prediction metrics for baseline
        prediction_metrics = {
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf'),
            'r2': -float('inf')
        }
        
        # Run simulation with reactive control (no forecast)
        simulation_result = self.simulator.simulate(
            pv_forecast=predictions,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=60,
            battery_dispatch=None  # Reactive control
        )
        
        # Analyze stability
        stability_metrics = self.analyzer.analyze(simulation_result)
        
        return ModelResult(
            model_name='no_forecast',
            predictions=predictions,
            prediction_metrics=prediction_metrics,
            simulation_result=simulation_result,
            stability_metrics=stability_metrics
        )
    
    def _compute_battery_dispatch(
        self,
        pv_forecast: np.ndarray,
        actual_pv: np.ndarray,
        load_profile: np.ndarray
    ) -> np.ndarray:
        """
        Compute battery dispatch using EMS controller.
        
        Args:
            pv_forecast: Forecasted PV power
            actual_pv: Actual PV power
            load_profile: Load demand profile
        
        Returns:
            Battery dispatch array (kW, positive = charge, negative = discharge)
        """
        n_steps = len(pv_forecast)
        battery_dispatch = np.zeros(n_steps)
        
        # Initialize system state
        current_soc = self.microgrid_config.initial_soc_kwh or (
            self.microgrid_config.battery_capacity_kwh * 0.5
        )
        
        for t in range(n_steps):
            # Get forecast horizon (use remaining steps or fixed horizon)
            horizon = min(12, n_steps - t)  # 12-step lookahead
            forecast_window = pv_forecast[t:t+horizon]
            load_window = load_profile[t:t+horizon]
            
            # Compute dispatch
            dispatch = self.ems_controller.compute_dispatch(
                current_soc=current_soc,
                pv_forecast=forecast_window,
                load_forecast=load_window,
                horizon=horizon
            )
            
            battery_dispatch[t] = dispatch[0] if len(dispatch) > 0 else 0.0
            
            # Update SOC estimate (simplified)
            dt_hours = 1.0 / 60.0  # 1 minute timestep
            if battery_dispatch[t] > 0:
                # Charging
                current_soc += battery_dispatch[t] * dt_hours * 0.95
            else:
                # Discharging
                current_soc += battery_dispatch[t] * dt_hours / 0.95
            
            # Enforce SOC limits
            current_soc = np.clip(
                current_soc,
                0.0,
                self.microgrid_config.battery_capacity_kwh
            )
        
        return battery_dispatch
    
    def calculate_improvements(
        self,
        results: Dict[str, ModelResult],
        baseline: str = 'no_forecast'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate percentage improvements over baseline.
        
        Args:
            results: Dictionary of model results
            baseline: Name of baseline model
        
        Returns:
            Dictionary mapping model names to improvement percentages
            Positive values indicate improvement (lower is better for most metrics)
        """
        if baseline not in results:
            raise ValueError(f"Baseline model '{baseline}' not found in results")
        
        baseline_result = results[baseline]
        improvements = {}
        
        for model_name, result in results.items():
            if model_name == baseline:
                # Baseline has 0% improvement
                improvements[model_name] = {}
                continue
            
            model_improvements = {}
            
            # Prediction metric improvements (lower is better)
            for metric in ['mae', 'rmse', 'mape']:
                baseline_val = baseline_result.prediction_metrics[metric]
                model_val = result.prediction_metrics[metric]
                
                if baseline_val != 0 and not np.isinf(baseline_val):
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    model_improvements[f'{metric}_improvement'] = improvement
            
            # R² improvement (higher is better)
            baseline_r2 = baseline_result.prediction_metrics['r2']
            model_r2 = result.prediction_metrics['r2']
            if not np.isinf(baseline_r2):
                r2_improvement = ((model_r2 - baseline_r2) / abs(baseline_r2 + 1e-6)) * 100
                model_improvements['r2_improvement'] = r2_improvement
            
            # Frequency stability improvements (lower deviation is better)
            baseline_freq = baseline_result.stability_metrics.frequency
            model_freq = result.stability_metrics.frequency
            
            freq_metrics = {
                'freq_mean_abs_dev': baseline_freq.mean_absolute_deviation,
                'freq_std': baseline_freq.std_deviation,
                'freq_max_dev': baseline_freq.max_deviation
            }
            
            for metric_name, baseline_val in freq_metrics.items():
                # Map metric names to attribute names
                attr_map = {
                    'freq_mean_abs_dev': 'mean_absolute_deviation',
                    'freq_std': 'std_deviation',
                    'freq_max_dev': 'max_deviation'
                }
                model_val = getattr(model_freq, attr_map[metric_name])
                if baseline_val != 0:
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    model_improvements[f'{metric_name}_improvement'] = improvement
            
            # Voltage stability improvements
            baseline_volt = baseline_result.stability_metrics.voltage
            model_volt = result.stability_metrics.voltage
            
            volt_metrics = {
                'volt_mean_abs_dev': baseline_volt.mean_absolute_deviation,
                'volt_std': baseline_volt.std_deviation,
                'volt_max_dev': baseline_volt.max_deviation
            }
            
            for metric_name, baseline_val in volt_metrics.items():
                # Map metric names to attribute names
                attr_map = {
                    'volt_mean_abs_dev': 'mean_absolute_deviation',
                    'volt_std': 'std_deviation',
                    'volt_max_dev': 'max_deviation'
                }
                model_val = getattr(model_volt, attr_map[metric_name])
                if baseline_val != 0:
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    model_improvements[f'{metric_name}_improvement'] = improvement
            
            # Battery stress improvements
            baseline_batt = baseline_result.stability_metrics.battery
            model_batt = result.stability_metrics.battery
            
            if baseline_batt.num_cycles != 0:
                cycles_improvement = ((baseline_batt.num_cycles - model_batt.num_cycles) / 
                                     baseline_batt.num_cycles) * 100
                model_improvements['battery_cycles_improvement'] = cycles_improvement
            
            if baseline_batt.total_throughput != 0:
                throughput_improvement = ((baseline_batt.total_throughput - model_batt.total_throughput) / 
                                         baseline_batt.total_throughput) * 100
                model_improvements['battery_throughput_improvement'] = throughput_improvement
            
            improvements[model_name] = model_improvements
        
        return improvements
    
    def _calculate_rankings(self, results: Dict[str, ModelResult]) -> Dict[str, List[str]]:
        """
        Rank models by various metrics.
        
        Args:
            results: Dictionary of model results
        
        Returns:
            Dictionary mapping metric names to ranked lists of model names
        """
        rankings = {}
        
        # Prediction metrics (lower is better)
        for metric in ['mae', 'rmse', 'mape']:
            metric_values = []
            for model_name, result in results.items():
                value = result.prediction_metrics[metric]
                if not np.isinf(value):
                    metric_values.append((model_name, value))
            
            # Sort by value (ascending - lower is better)
            metric_values.sort(key=lambda x: x[1])
            rankings[metric] = [name for name, _ in metric_values]
        
        # R² (higher is better)
        r2_values = []
        for model_name, result in results.items():
            value = result.prediction_metrics['r2']
            if not np.isinf(value):
                r2_values.append((model_name, value))
        
        r2_values.sort(key=lambda x: x[1], reverse=True)
        rankings['r2'] = [name for name, _ in r2_values]
        
        # Frequency stability (lower is better)
        freq_std_values = [
            (name, result.stability_metrics.frequency.std_deviation)
            for name, result in results.items()
        ]
        freq_std_values.sort(key=lambda x: x[1])
        rankings['freq_stability'] = [name for name, _ in freq_std_values]
        
        # Voltage stability (lower is better)
        volt_std_values = [
            (name, result.stability_metrics.voltage.std_deviation)
            for name, result in results.items()
        ]
        volt_std_values.sort(key=lambda x: x[1])
        rankings['volt_stability'] = [name for name, _ in volt_std_values]
        
        # Battery stress (lower cycles is better)
        battery_cycles_values = [
            (name, result.stability_metrics.battery.num_cycles)
            for name, result in results.items()
        ]
        battery_cycles_values.sort(key=lambda x: x[1])
        rankings['battery_stress'] = [name for name, _ in battery_cycles_values]
        
        return rankings
    
    def rank_models(self, results: Dict[str, ModelResult], metric: str) -> List[Tuple[str, float]]:
        """
        Rank models by a specified metric.
        
        Args:
            results: Dictionary of model results
            metric: Metric name to rank by
        
        Returns:
            List of (model_name, metric_value) tuples sorted by rank
        """
        metric_values = []
        
        # Check if it's a prediction metric
        if metric in ['mae', 'rmse', 'mape', 'r2']:
            for model_name, result in results.items():
                value = result.prediction_metrics[metric]
                if not np.isinf(value):
                    metric_values.append((model_name, value))
            
            # Sort (ascending for mae/rmse/mape, descending for r2)
            reverse = (metric == 'r2')
            metric_values.sort(key=lambda x: x[1], reverse=reverse)
        
        # Check if it's a stability metric
        elif 'freq' in metric or 'frequency' in metric:
            for model_name, result in results.items():
                if 'std' in metric:
                    value = result.stability_metrics.frequency.std_deviation
                elif 'mean' in metric:
                    value = result.stability_metrics.frequency.mean_absolute_deviation
                elif 'max' in metric:
                    value = result.stability_metrics.frequency.max_deviation
                else:
                    continue
                metric_values.append((model_name, value))
            
            metric_values.sort(key=lambda x: x[1])
        
        elif 'volt' in metric or 'voltage' in metric:
            for model_name, result in results.items():
                if 'std' in metric:
                    value = result.stability_metrics.voltage.std_deviation
                elif 'mean' in metric:
                    value = result.stability_metrics.voltage.mean_absolute_deviation
                elif 'max' in metric:
                    value = result.stability_metrics.voltage.max_deviation
                else:
                    continue
                metric_values.append((model_name, value))
            
            metric_values.sort(key=lambda x: x[1])
        
        elif 'battery' in metric or 'cycles' in metric:
            for model_name, result in results.items():
                value = result.stability_metrics.battery.num_cycles
                metric_values.append((model_name, value))
            
            metric_values.sort(key=lambda x: x[1])
        
        return metric_values
    
    def statistical_significance(
        self,
        results: Dict[str, ModelResult],
        model_a: str,
        model_b: str,
        metric: str = 'freq_std'
    ) -> SignificanceTest:
        """
        Perform statistical significance test comparing two models.
        
        Uses paired t-test to compare time-series metrics between models.
        
        Args:
            results: Dictionary of model results
            model_a: Name of first model
            model_b: Name of second model
            metric: Metric to compare ('freq_std', 'volt_std', 'battery_power', etc.)
        
        Returns:
            SignificanceTest object with test results
        """
        if model_a not in results or model_b not in results:
            raise ValueError(f"Models '{model_a}' or '{model_b}' not found in results")
        
        result_a = results[model_a]
        result_b = results[model_b]
        
        # Extract time-series data for comparison
        if metric == 'freq_std' or metric == 'frequency':
            data_a = result_a.simulation_result.frequency_deviation
            data_b = result_b.simulation_result.frequency_deviation
            metric_name = 'Frequency Deviation'
        elif metric == 'volt_std' or metric == 'voltage':
            data_a = result_a.simulation_result.voltage_deviation
            data_b = result_b.simulation_result.voltage_deviation
            metric_name = 'Voltage Deviation'
        elif metric == 'battery_power':
            data_a = result_a.simulation_result.battery_power
            data_b = result_b.simulation_result.battery_power
            metric_name = 'Battery Power'
        elif metric == 'battery_soc':
            data_a = result_a.simulation_result.battery_soc
            data_b = result_b.simulation_result.battery_soc
            metric_name = 'Battery SOC'
        else:
            raise ValueError(f"Unknown metric for significance testing: {metric}")
        
        # Perform paired t-test
        # Tests if the mean difference between paired samples is significantly different from zero
        statistic, p_value = stats.ttest_rel(np.abs(data_a), np.abs(data_b))
        
        # Determine significance (p < 0.05)
        significant = bool(p_value < 0.05)
        
        # Generate interpretation
        mean_a = np.mean(np.abs(data_a))
        mean_b = np.mean(np.abs(data_b))
        
        if significant:
            if mean_a < mean_b:
                interpretation = f"{model_a} performs significantly better than {model_b} (p={p_value:.4f})"
            else:
                interpretation = f"{model_b} performs significantly better than {model_a} (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference between {model_a} and {model_b} (p={p_value:.4f})"
        
        return SignificanceTest(
            model_a=model_a,
            model_b=model_b,
            metric=metric_name,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            interpretation=interpretation
        )
    
    def generate_comparison_table(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """
        Generate comparison table with all metrics.
        
        Args:
            comparison_result: Complete comparison results
        
        Returns:
            Dictionary containing formatted comparison data
        """
        table = {
            'models': {},
            'rankings': comparison_result.rankings,
            'improvements': comparison_result.improvements
        }
        
        for model_name, result in comparison_result.models.items():
            model_data = {
                'prediction_metrics': result.prediction_metrics,
                'stability_metrics': {
                    'frequency': {
                        'mean_abs_dev': result.stability_metrics.frequency.mean_absolute_deviation,
                        'std_dev': result.stability_metrics.frequency.std_deviation,
                        'max_dev': result.stability_metrics.frequency.max_deviation,
                        'time_outside_limits': result.stability_metrics.frequency.time_outside_limits
                    },
                    'voltage': {
                        'mean_abs_dev': result.stability_metrics.voltage.mean_absolute_deviation,
                        'std_dev': result.stability_metrics.voltage.std_deviation,
                        'max_dev': result.stability_metrics.voltage.max_deviation,
                        'time_outside_limits': result.stability_metrics.voltage.time_outside_limits
                    },
                    'battery': {
                        'soc_range': result.stability_metrics.battery.soc_range,
                        'num_cycles': result.stability_metrics.battery.num_cycles,
                        'max_dod': result.stability_metrics.battery.max_depth_of_discharge,
                        'avg_c_rate': result.stability_metrics.battery.avg_c_rate,
                        'total_throughput': result.stability_metrics.battery.total_throughput
                    },
                    'power_quality': {
                        'thd_proxy': result.stability_metrics.power_quality.thd_proxy,
                        'power_factor': result.stability_metrics.power_quality.power_factor
                    },
                    'energy': {
                        'unmet_load': result.stability_metrics.energy.total_unmet_load,
                        'curtailed_pv': result.stability_metrics.energy.total_curtailed_pv,
                        'efficiency': result.stability_metrics.energy.energy_efficiency
                    }
                }
            }
            
            table['models'][model_name] = model_data
        
        return table
