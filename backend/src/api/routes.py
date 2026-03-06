"""API routes for the microgrid stability backend."""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from flask import Blueprint, request, jsonify, send_file
from pydantic import ValidationError
import numpy as np
import pandas as pd

from src.api.schemas import (
    TrainRequest, TrainResponse, TrainingStatus,
    PredictRequest, PredictResponse,
    SimulateRequest, SimulationResponse,
    CompareRequest, ComparisonResponse,
    ModelListResponse, ModelMetadata,
    DataUploadResponse, DataValidateRequest, ValidationReport,
    ErrorResponse
)
from src.data.parser import Parser
from src.data.pipeline import DataPipeline
from src.data.validator import DataValidator
from src.models.classical import ClassicalPredictor
from src.models.lstm import LSTMPredictor
from src.simulation.simulator import MicrogridSimulator
from src.simulation.ems_controller import EMSController
from src.analysis.stability_analyzer import StabilityAnalyzer
from src.analysis.comparative_engine import ComparativeEngine
from src.analysis.results_exporter import ResultsExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# In-memory storage for jobs and results (in production, use Redis or database)
training_jobs: Dict[str, Dict[str, Any]] = {}
comparison_jobs: Dict[str, Dict[str, Any]] = {}
simulation_results: Dict[str, Dict[str, Any]] = {}

# Directories
MODELS_DIR = Path("models/saved_models")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def error_response(code: str, message: str, details: Dict = None, status_code: int = 400):
    """Create standardized error response."""
    error_data = {
        "error": {
            "code": code,
            "message": message
        }
    }
    if details:
        error_data["error"]["details"] = details
    return jsonify(error_data), status_code


# ============================================================================
# Training Endpoints
# ============================================================================

@api_bp.route('/train', methods=['POST'])
def start_training():
    """Start model training job."""
    try:
        # Parse request
        data = request.get_json()
        train_req = TrainRequest(**data)
        
        # Generate job ID
        job_id = f"train_{uuid.uuid4().hex[:8]}"
        
        # Initialize job status
        training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "config": train_req.config.model_dump(),
            "created_at": datetime.now().isoformat()
        }
        
        # In a real implementation, this would be async (Celery task)
        # For now, we'll run synchronously
        try:
            _run_training_job(job_id, train_req)
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {str(e)}")
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
        
        response = TrainResponse(
            job_id=job_id,
            status=training_jobs[job_id]["status"],
            message=f"Training job {job_id} started"
        )
        return jsonify(response.model_dump()), 202
        
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Invalid request data", {"errors": e.errors()})
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


def _run_training_job(job_id: str, train_req: TrainRequest):
    """Execute training job (would be async in production)."""
    try:
        training_jobs[job_id]["status"] = "running"
        config = train_req.config
        
        # Load and preprocess data
        logger.info(f"Loading data from {config.data_path}")
        parser = Parser()
        df = parser.parse_timeseries_data(config.data_path)
        
        # Validate data
        validator = DataValidator()
        validation_result = validator.validate_timeseries(df)
        if not validation_result.valid:
            raise ValueError(f"Data validation failed: {validation_result.critical_errors}")
        
        # Preprocess data
        pipeline = DataPipeline(config)
        df_clean = pipeline.preprocess(df)
        X_normalized, scaler = pipeline.normalize(df_clean)
        X, y = pipeline.create_sequences(
            X_normalized,
            config.model_configuration.sequence_length
        )
        X_train, X_val, y_train, y_val = pipeline.split_data(
            X, y,
            train_ratio=1.0 - config.training_configuration.validation_split
        )
        
        # Create model
        model_type = config.model_configuration.model_type
        if model_type in ["persistence", "arima", "svr"]:
            model = ClassicalPredictor(config.model_configuration, method=model_type)
        elif model_type == "lstm":
            model = LSTMPredictor(config.model_configuration)
        else:
            raise ValueError(f"Model type {model_type} not yet implemented")
        
        # Train model
        logger.info(f"Training {model_type} model")
        training_result = model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = MODELS_DIR / model_id
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        
        # Update job status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 1.0
        training_jobs[job_id]["model_id"] = model_id
        training_jobs[job_id]["metrics"] = training_result["metrics"]
        
        logger.info(f"Training completed for job {job_id}, model saved as {model_id}")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        raise


@api_bp.route('/train/<job_id>/status', methods=['GET'])
def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        return error_response("NOT_FOUND", f"Training job {job_id} not found", status_code=404)
    
    job = training_jobs[job_id]
    status = TrainingStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_epoch=job.get("current_epoch"),
        metrics=job.get("metrics"),
        error=job.get("error")
    )
    return jsonify(status.model_dump()), 200


@api_bp.route('/train/<job_id>', methods=['DELETE'])
def cancel_training(job_id: str):
    """Cancel training job."""
    if job_id not in training_jobs:
        return error_response("NOT_FOUND", f"Training job {job_id} not found", status_code=404)
    
    # In production, would signal Celery task to stop
    training_jobs[job_id]["status"] = "failed"
    training_jobs[job_id]["error"] = "Cancelled by user"
    
    return '', 204


# ============================================================================
# Model Management Endpoints
# ============================================================================

@api_bp.route('/models', methods=['GET'])
def list_models():
    """List all available trained models."""
    try:
        model_type_filter = request.args.get('model_type')
        sort_by = request.args.get('sort_by', 'created_at')
        
        models = []
        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Apply filter
            if model_type_filter and metadata.get("model_type") != model_type_filter:
                continue
            
            models.append(ModelMetadata(**metadata))
        
        # Sort models
        if sort_by == 'created_at':
            models.sort(key=lambda m: m.created_at, reverse=True)
        elif sort_by in ['mae', 'rmse'] and models:
            models.sort(key=lambda m: m.metrics.get(sort_by, float('inf')) if m.metrics else float('inf'))
        
        response = ModelListResponse(models=models)
        return jsonify(response.model_dump()), 200
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@api_bp.route('/models/<model_id>', methods=['GET'])
def get_model_details(model_id: str):
    """Get detailed model information."""
    try:
        model_path = MODELS_DIR / model_id / "metadata.json"
        if not model_path.exists():
            return error_response("NOT_FOUND", f"Model {model_id} not found", status_code=404)
        
        with open(model_path, 'r') as f:
            metadata = json.load(f)
        
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@api_bp.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id: str):
    """Delete saved model."""
    try:
        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            return error_response("NOT_FOUND", f"Model {model_id} not found", status_code=404)
        
        # Delete model directory
        import shutil
        shutil.rmtree(model_path)
        
        logger.info(f"Deleted model {model_id}")
        return '', 204
        
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@api_bp.route('/models/<model_id>/load', methods=['POST'])
def load_model(model_id: str):
    """Load model into memory for inference."""
    try:
        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            return error_response("NOT_FOUND", f"Model {model_id} not found", status_code=404)
        
        # In production, would load model into cache/memory
        logger.info(f"Model {model_id} loaded (placeholder)")
        
        return jsonify({"message": f"Model {model_id} loaded successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# ============================================================================
# Prediction Endpoints
# ============================================================================

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Generate predictions using trained model."""
    try:
        data = request.get_json()
        pred_req = PredictRequest(**data)
        
        # Load model
        model_path = MODELS_DIR / pred_req.model_id
        if not model_path.exists():
            return error_response("NOT_FOUND", f"Model {pred_req.model_id} not found", status_code=404)
        
        # Load metadata to determine model type
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata["model_type"]
        
        # Create and load model
        if model_type in ["persistence", "arima", "svr"]:
            from src.config.schemas import ModelConfig, ModelType
            model_config = ModelConfig(model_type=ModelType(model_type))
            model = ClassicalPredictor(model_config, method=model_type)
        elif model_type == "lstm":
            from src.config.schemas import ModelConfig, ModelType
            model_config = ModelConfig(
                model_type=ModelType.LSTM,
                hyperparameters=metadata.get("architecture", {})
            )
            model = LSTMPredictor(model_config)
        else:
            return error_response("NOT_IMPLEMENTED", f"Model type {model_type} not yet supported")
        
        model.load(str(model_path))
        
        # Make predictions
        X = np.array(pred_req.input_data)
        predictions = model.predict(X)
        
        response = PredictResponse(
            predictions=predictions.flatten().tolist(),
            confidence_intervals=None  # TODO: Implement confidence intervals
        )
        return jsonify(response.model_dump()), 200
        
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Invalid request data", {"errors": e.errors()})
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# ============================================================================
# Simulation Endpoints
# ============================================================================

@api_bp.route('/simulate', methods=['POST'])
def simulate():
    """Run microgrid simulation."""
    try:
        data = request.get_json()
        sim_req = SimulateRequest(**data)
        
        # Create simulator
        simulator = MicrogridSimulator(sim_req.microgrid_config)
        
        # Run simulation
        pv_forecast = np.array(sim_req.predictions)
        actual_pv = np.array(sim_req.actual_pv)
        load_profile = np.array(sim_req.load_profile)
        
        result = simulator.simulate(
            pv_forecast=pv_forecast,
            actual_pv=actual_pv,
            load_profile=load_profile,
            timestep_seconds=300  # 5 minutes
        )
        
        # Analyze stability
        analyzer = StabilityAnalyzer(
            battery_capacity_kwh=sim_req.microgrid_config.battery_capacity_kwh
        )
        metrics = analyzer.analyze(result)
        
        # Convert metrics to dict for JSON serialization
        from dataclasses import asdict
        metrics_dict = asdict(metrics)
        
        # Generate result ID and store
        result_id = f"sim_{uuid.uuid4().hex[:8]}"
        simulation_results[result_id] = {
            "result_id": result_id,
            "result": result,
            "metrics": metrics,
            "created_at": datetime.now().isoformat()
        }
        
        # Prepare response
        response = SimulationResponse(
            result_id=result_id,
            timeseries={
                "time": result.timestamps.tolist(),
                "soc": result.battery_soc.tolist(),
                "frequency_deviation": result.frequency_deviation.tolist(),
                "voltage_deviation": result.voltage_deviation.tolist(),
                "battery_power": result.battery_power.tolist()
            },
            metrics=metrics_dict
        )
        
        return jsonify(response.model_dump()), 200
        
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Invalid request data", {"errors": e.errors()})
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# ============================================================================
# Comparison Endpoints
# ============================================================================

@api_bp.route('/compare', methods=['POST'])
def start_comparison():
    """Start comparative analysis."""
    try:
        data = request.get_json()
        comp_req = CompareRequest(**data)
        
        # Generate comparison ID
        comparison_id = f"comp_{uuid.uuid4().hex[:8]}"
        
        # Initialize comparison job
        comparison_jobs[comparison_id] = {
            "comparison_id": comparison_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat()
        }
        
        # Run comparison (would be async in production)
        try:
            _run_comparison_job(comparison_id, comp_req)
        except Exception as e:
            logger.error(f"Comparison failed for {comparison_id}: {str(e)}")
            comparison_jobs[comparison_id]["status"] = "failed"
            comparison_jobs[comparison_id]["error"] = str(e)
        
        response = ComparisonResponse(
            comparison_id=comparison_id,
            results=comparison_jobs[comparison_id].get("results", {}),
            rankings=comparison_jobs[comparison_id].get("rankings", {}),
            improvements=comparison_jobs[comparison_id].get("improvements", {})
        )
        return jsonify(response.model_dump()), 202
        
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Invalid request data", {"errors": e.errors()})
    except Exception as e:
        logger.error(f"Error starting comparison: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


def _run_comparison_job(comparison_id: str, comp_req: CompareRequest):
    """Execute comparison job."""
    try:
        comparison_jobs[comparison_id]["status"] = "running"
        
        # Load test data
        parser = Parser()
        test_df = parser.parse_timeseries_data(comp_req.test_data_path)
        
        # Load models
        models = []
        for model_id in comp_req.model_ids:
            model_path = MODELS_DIR / model_id
            if not model_path.exists():
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata["model_type"]
            if model_type in ["persistence", "arima", "svr"]:
                from src.config.schemas import ModelConfig, ModelType
                model_config = ModelConfig(model_type=ModelType(model_type))
                model = ClassicalPredictor(model_config, method=model_type)
            elif model_type == "lstm":
                from src.config.schemas import ModelConfig, ModelType
                model_config = ModelConfig(
                    model_type=ModelType.LSTM,
                    hyperparameters=metadata.get("architecture", {})
                )
                model = LSTMPredictor(model_config)
            else:
                logger.warning(f"Model type {model_type} not supported, skipping")
                continue
            
            model.load(str(model_path))
            models.append((model_id, model))
        
        # Run comparative analysis
        engine = ComparativeEngine(comp_req.microgrid_config)
        comparison_result = engine.run_comparison(
            models=models,
            test_data=test_df
        )
        
        # Update job status
        comparison_jobs[comparison_id]["status"] = "completed"
        comparison_jobs[comparison_id]["progress"] = 1.0
        comparison_jobs[comparison_id]["results"] = comparison_result["results"]
        comparison_jobs[comparison_id]["rankings"] = comparison_result["rankings"]
        comparison_jobs[comparison_id]["improvements"] = comparison_result["improvements"]
        
        logger.info(f"Comparison completed for {comparison_id}")
        
    except Exception as e:
        logger.error(f"Comparison job {comparison_id} failed: {str(e)}")
        comparison_jobs[comparison_id]["status"] = "failed"
        comparison_jobs[comparison_id]["error"] = str(e)
        raise


@api_bp.route('/compare/<comparison_id>/status', methods=['GET'])
def get_comparison_status(comparison_id: str):
    """Get comparison job status."""
    if comparison_id not in comparison_jobs:
        return error_response("NOT_FOUND", f"Comparison {comparison_id} not found", status_code=404)
    
    job = comparison_jobs[comparison_id]
    return jsonify({
        "comparison_id": job["comparison_id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job.get("error")
    }), 200


@api_bp.route('/compare/<comparison_id>/results', methods=['GET'])
def get_comparison_results(comparison_id: str):
    """Get comparison results."""
    if comparison_id not in comparison_jobs:
        return error_response("NOT_FOUND", f"Comparison {comparison_id} not found", status_code=404)
    
    job = comparison_jobs[comparison_id]
    if job["status"] != "completed":
        return error_response("NOT_READY", f"Comparison {comparison_id} not yet completed")
    
    return jsonify({
        "comparison_id": comparison_id,
        "results": job["results"],
        "rankings": job["rankings"],
        "improvements": job["improvements"]
    }), 200


# ============================================================================
# Export Endpoints
# ============================================================================

@api_bp.route('/export/<result_id>', methods=['GET'])
def export_results(result_id: str):
    """Export simulation results."""
    try:
        if result_id not in simulation_results:
            return error_response("NOT_FOUND", f"Result {result_id} not found", status_code=404)
        
        format_type = request.args.get('format', 'json')
        include = request.args.get('include', 'all')
        
        result_data = simulation_results[result_id]
        
        # Create exporter
        export_dir = RESULTS_DIR / result_id
        export_dir.mkdir(parents=True, exist_ok=True)
        exporter = ResultsExporter(str(export_dir))
        
        # Export based on format
        if format_type == 'json':
            export_path = export_dir / f"{result_id}.json"
            # Convert SimulationResult to dict
            from dataclasses import asdict
            result_obj = result_data["result"]
            timeseries_dict = {
                "timestamps": result_obj.timestamps.tolist(),
                "pv_power": result_obj.pv_power.tolist(),
                "load_power": result_obj.load_power.tolist(),
                "battery_power": result_obj.battery_power.tolist(),
                "battery_soc": result_obj.battery_soc.tolist(),
                "frequency_deviation": result_obj.frequency_deviation.tolist(),
                "voltage_deviation": result_obj.voltage_deviation.tolist(),
                "grid_power": result_obj.grid_power.tolist()
            }
            with open(export_path, 'w') as f:
                json.dump({
                    "timeseries": timeseries_dict,
                    "metrics": result_data["metrics"]
                }, f, indent=2, default=str)
            return send_file(export_path, as_attachment=True)
        
        elif format_type == 'csv':
            # Export timeseries as CSV
            result_obj = result_data["result"]
            timeseries_dict = {
                "timestamps": result_obj.timestamps,
                "pv_power": result_obj.pv_power,
                "load_power": result_obj.load_power,
                "battery_power": result_obj.battery_power,
                "battery_soc": result_obj.battery_soc,
                "frequency_deviation": result_obj.frequency_deviation,
                "voltage_deviation": result_obj.voltage_deviation,
                "grid_power": result_obj.grid_power
            }
            export_path = exporter.export_timeseries(
                timeseries_dict,
                f"{result_id}_timeseries"
            )
            return send_file(export_path, as_attachment=True)
        
        else:
            return error_response("INVALID_FORMAT", f"Format {format_type} not supported")
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# ============================================================================
# Data Management Endpoints
# ============================================================================

@api_bp.route('/data/upload', methods=['POST'])
def upload_data():
    """Upload time-series data."""
    try:
        if 'file' not in request.files:
            return error_response("MISSING_FILE", "No file provided")
        
        file = request.files['file']
        if file.filename == '':
            return error_response("INVALID_FILE", "Empty filename")
        
        # Save file
        data_id = f"data_{uuid.uuid4().hex[:8]}"
        file_path = DATA_DIR / f"{data_id}.csv"
        file.save(file_path)
        
        # Validate data
        parser = Parser()
        df = parser.parse_timeseries_data(str(file_path))
        
        validator = DataValidator()
        validation_result = validator.validate_timeseries(df)
        
        # Create validation report
        report = ValidationReport(
            valid=validation_result.valid,
            warnings=validation_result.warnings,
            errors=validation_result.critical_errors,
            statistics=None  # TODO: Add statistics
        )
        
        response = DataUploadResponse(
            data_id=data_id,
            validation_report=report
        )
        
        return jsonify(response.model_dump()), 200
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@api_bp.route('/data/validate', methods=['POST'])
def validate_data():
    """Validate data quality."""
    try:
        data = request.get_json()
        val_req = DataValidateRequest(**data)
        
        # Load and validate data
        parser = Parser()
        df = parser.parse_timeseries_data(val_req.data_path)
        
        validator = DataValidator()
        validation_result = validator.validate_timeseries(df)
        
        report = ValidationReport(
            valid=validation_result.valid,
            warnings=validation_result.warnings,
            errors=validation_result.critical_errors,
            statistics=None  # TODO: Add statistics
        )
        
        return jsonify(report.model_dump()), 200
        
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Invalid request data", {"errors": e.errors()})
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@api_bp.route('/data/<data_id>', methods=['GET'])
def get_data_metadata(data_id: str):
    """Get data metadata."""
    try:
        file_path = DATA_DIR / f"{data_id}.csv"
        if not file_path.exists():
            return error_response("NOT_FOUND", f"Data {data_id} not found", status_code=404)
        
        # Load data and get basic info
        parser = Parser()
        df = parser.parse_timeseries_data(str(file_path))
        
        metadata = {
            "data_id": data_id,
            "num_samples": len(df),
            "columns": list(df.columns),
            "date_range": [str(df.index.min()), str(df.index.max())] if hasattr(df.index, 'min') else None
        }
        
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error getting data metadata: {str(e)}")
        return error_response("INTERNAL_ERROR", str(e), status_code=500)
