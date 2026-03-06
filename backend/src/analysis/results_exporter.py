"""
Results exporter for formatting and exporting analysis results.

This module provides functionality to export simulation results, metrics,
and visualizations in various formats (CSV, JSON, PNG, SVG, Markdown).
"""

import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.analysis.comparative_engine import ComparisonResult, ModelResult
from src.config.schemas import Configuration


class ResultsExporter:
    """
    Exporter for analysis results in multiple formats.
    
    Exports time-series data, metrics, visualizations, and configurations
    to support reproducibility and publication-ready outputs.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize results exporter.
        
        Args:
            output_dir: Base directory for exported results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_timeseries(
        self,
        data: Dict[str, np.ndarray],
        filename: str,
        format: str = 'csv',
        timestamps: Optional[np.ndarray] = None
    ) -> str:
        """
        Export time-series data to CSV format.
        
        Args:
            data: Dictionary mapping column names to data arrays
            filename: Output filename (without extension)
            format: Export format ('csv' only for now)
            timestamps: Optional timestamp array (seconds since start)
        
        Returns:
            Path to exported file
        """
        if format != 'csv':
            raise ValueError(f"Unsupported format: {format}. Only 'csv' is supported.")
        
        output_path = self.output_dir / f"{filename}.csv"
        
        # Determine number of rows
        n_rows = len(next(iter(data.values())))
        
        # Create timestamp column if not provided
        if timestamps is None:
            timestamps = np.arange(n_rows) * 60  # Assume 1-minute intervals
        
        # Convert timestamps to datetime strings
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        datetime_strings = [
            (start_time + timedelta(seconds=int(t))).strftime('%Y-%m-%d %H:%M:%S')
            for t in timestamps
        ]
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['timestamp'] + list(data.keys())
            writer.writerow(header)
            
            # Write data rows
            for i in range(n_rows):
                row = [datetime_strings[i]] + [data[col][i] for col in data.keys()]
                writer.writerow(row)
        
        return str(output_path)
    
    def export_metrics(
        self,
        metrics: Dict[str, Any],
        filename: str,
        format: str = 'json'
    ) -> str:
        """
        Export metrics to JSON format.
        
        Args:
            metrics: Dictionary containing metrics data
            filename: Output filename (without extension)
            format: Export format ('json' or 'csv')
        
        Returns:
            Path to exported file
        """
        if format == 'json':
            output_path = self.output_dir / f"{filename}.json"
            
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = self._make_json_serializable(metrics)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        elif format == 'csv':
            output_path = self.output_dir / f"{filename}.csv"
            
            # Flatten nested dictionary for CSV
            flattened = self._flatten_dict(metrics)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                for key, value in flattened.items():
                    writer.writerow([key, value])
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)
    
    def export_visualizations(
        self,
        figures: List[plt.Figure],
        prefix: str,
        format: str = 'png'
    ) -> List[str]:
        """
        Export matplotlib figures as PNG or SVG files.
        
        Args:
            figures: List of matplotlib Figure objects
            prefix: Filename prefix for exported files
            format: Export format ('png' or 'svg')
        
        Returns:
            List of paths to exported files
        """
        if format not in ['png', 'svg']:
            raise ValueError(f"Unsupported format: {format}. Use 'png' or 'svg'.")
        
        exported_paths = []
        
        for i, fig in enumerate(figures):
            filename = f"{prefix}_{i+1}.{format}"
            output_path = self.output_dir / filename
            
            # Set DPI for high-resolution output
            dpi = 300 if format == 'png' else None
            
            fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
            exported_paths.append(str(output_path))
            
            # Close figure to free memory
            plt.close(fig)
        
        return exported_paths
    
    def export_configuration(
        self,
        config: Configuration,
        filename: str = "configuration"
    ) -> str:
        """
        Export configuration for reproducibility.
        
        Args:
            config: Configuration object
            filename: Output filename (without extension)
        
        Returns:
            Path to exported configuration file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert Pydantic model to dict with mode='json' to handle enums properly
        config_dict = config.model_dump(mode='json')
        
        # Make JSON serializable (handles any remaining numpy types)
        serializable_config = self._make_json_serializable(config_dict)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        return str(output_path)
    
    def generate_report(
        self,
        comparison_result: ComparisonResult,
        config: Configuration,
        format: str = 'markdown'
    ) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            comparison_result: Complete comparison results
            config: Configuration used for the analysis
            format: Report format ('markdown' or 'pdf')
        
        Returns:
            Path to generated report
        """
        if format == 'markdown':
            return self._generate_markdown_report(comparison_result, config)
        elif format == 'pdf':
            # PDF generation would require additional dependencies (e.g., reportlab)
            raise NotImplementedError("PDF report generation not yet implemented")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(
        self,
        comparison_result: ComparisonResult,
        config: Configuration
    ) -> str:
        """
        Generate Markdown summary report.
        
        Args:
            comparison_result: Complete comparison results
            config: Configuration used for the analysis
        
        Returns:
            Path to generated Markdown report
        """
        output_path = self.output_dir / "analysis_report.md"
        
        with open(output_path, 'w') as f:
            # Title
            f.write("# Microgrid Stability Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration section
            f.write("## Configuration\n\n")
            f.write(f"- **Experiment Name:** {config.experiment_name}\n")
            f.write(f"- **Forecast Horizon:** {config.forecast_horizon.value}\n")
            f.write(f"- **Model Type:** {config.model_configuration.model_type.value}\n")
            f.write(f"- **Battery Capacity:** {config.microgrid_configuration.battery_capacity_kwh} kWh\n")
            f.write(f"- **PV Capacity:** {config.microgrid_configuration.pv_capacity_kw} kW\n")
            f.write(f"- **Microgrid Mode:** {config.microgrid_configuration.mode.value}\n\n")
            
            # Prediction performance section
            f.write("## Prediction Performance\n\n")
            f.write("| Model | MAE | RMSE | MAPE (%) | R² |\n")
            f.write("|-------|-----|------|----------|----|\n")
            
            for model_name, result in comparison_result.models.items():
                if model_name == 'no_forecast':
                    continue  # Skip baseline for prediction metrics
                
                metrics = result.prediction_metrics
                f.write(f"| {model_name} | ")
                f.write(f"{metrics['mae']:.4f} | ")
                f.write(f"{metrics['rmse']:.4f} | ")
                f.write(f"{metrics['mape']:.2f} | ")
                f.write(f"{metrics['r2']:.4f} |\n")
            
            f.write("\n")
            
            # Stability improvements section
            f.write("## Stability Improvements\n\n")
            f.write("Comparison against baseline: **{}**\n\n".format(
                comparison_result.baseline_model
            ))
            
            # Frequency stability table
            f.write("### Frequency Stability\n\n")
            f.write("| Model | Mean Abs Dev (Hz) | Std Dev (Hz) | Max Dev (Hz) | Improvement (%) |\n")
            f.write("|-------|-------------------|--------------|--------------|------------------|\n")
            
            baseline_result = comparison_result.models[comparison_result.baseline_model]
            baseline_freq = baseline_result.stability_metrics.frequency
            
            for model_name, result in comparison_result.models.items():
                freq = result.stability_metrics.frequency
                
                # Calculate improvement
                if model_name in comparison_result.improvements:
                    improvement = comparison_result.improvements[model_name].get(
                        'freq_std_improvement', 0.0
                    )
                else:
                    improvement = 0.0
                
                f.write(f"| {model_name} | ")
                f.write(f"{freq.mean_absolute_deviation:.4f} | ")
                f.write(f"{freq.std_deviation:.4f} | ")
                f.write(f"{freq.max_deviation:.4f} | ")
                f.write(f"{improvement:.1f} |\n")
            
            f.write("\n")
            
            # Voltage stability table
            f.write("### Voltage Stability\n\n")
            f.write("| Model | Mean Abs Dev (%) | Std Dev (%) | Max Dev (%) | Improvement (%) |\n")
            f.write("|-------|------------------|-------------|-------------|------------------|\n")
            
            for model_name, result in comparison_result.models.items():
                volt = result.stability_metrics.voltage
                
                # Calculate improvement
                if model_name in comparison_result.improvements:
                    improvement = comparison_result.improvements[model_name].get(
                        'volt_std_improvement', 0.0
                    )
                else:
                    improvement = 0.0
                
                f.write(f"| {model_name} | ")
                f.write(f"{volt.mean_absolute_deviation:.4f} | ")
                f.write(f"{volt.std_deviation:.4f} | ")
                f.write(f"{volt.max_deviation:.4f} | ")
                f.write(f"{improvement:.1f} |\n")
            
            f.write("\n")
            
            # Battery stress table
            f.write("### Battery Stress\n\n")
            f.write("| Model | SOC Range (kWh) | Cycles | Max DOD | Throughput (kWh) | Improvement (%) |\n")
            f.write("|-------|-----------------|--------|---------|------------------|------------------|\n")
            
            for model_name, result in comparison_result.models.items():
                batt = result.stability_metrics.battery
                
                # Calculate improvement
                if model_name in comparison_result.improvements:
                    improvement = comparison_result.improvements[model_name].get(
                        'battery_cycles_improvement', 0.0
                    )
                else:
                    improvement = 0.0
                
                f.write(f"| {model_name} | ")
                f.write(f"{batt.soc_range:.2f} | ")
                f.write(f"{batt.num_cycles:.1f} | ")
                f.write(f"{batt.max_depth_of_discharge:.2f} | ")
                f.write(f"{batt.total_throughput:.2f} | ")
                f.write(f"{improvement:.1f} |\n")
            
            f.write("\n")
            
            # Rankings section
            f.write("## Model Rankings\n\n")
            
            for metric_name, ranked_models in comparison_result.rankings.items():
                f.write(f"### {metric_name.replace('_', ' ').title()}\n\n")
                for i, model_name in enumerate(ranked_models, 1):
                    f.write(f"{i}. {model_name}\n")
                f.write("\n")
            
            # Key findings section
            f.write("## Key Findings\n\n")
            
            # Best model for prediction
            if 'mae' in comparison_result.rankings:
                best_prediction_model = comparison_result.rankings['mae'][0]
                f.write(f"- **Best Prediction Model:** {best_prediction_model}\n")
            
            # Best model for frequency stability
            if 'freq_stability' in comparison_result.rankings:
                best_freq_model = comparison_result.rankings['freq_stability'][0]
                f.write(f"- **Best Frequency Stability:** {best_freq_model}\n")
            
            # Best model for battery stress
            if 'battery_stress' in comparison_result.rankings:
                best_battery_model = comparison_result.rankings['battery_stress'][0]
                f.write(f"- **Lowest Battery Stress:** {best_battery_model}\n")
            
            f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("This analysis demonstrates the impact of PV forecasting accuracy on ")
            f.write("microgrid stability. The results show quantifiable improvements in ")
            f.write("frequency stability, voltage stability, and battery stress when using ")
            f.write("advanced forecasting methods compared to reactive control.\n")
        
        return str(output_path)
    
    def create_export_directory(self, experiment_name: str) -> Path:
        """
        Create timestamped directory for exported artifacts.
        
        Args:
            experiment_name: Name of the experiment
        
        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{experiment_name}_{timestamp}"
        export_dir = self.output_dir / dir_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Update output_dir to use the new directory
        self.output_dir = export_dir
        
        return export_dir
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types and other non-serializable types to JSON-compatible types.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclass or custom objects
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary for CSV export.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
