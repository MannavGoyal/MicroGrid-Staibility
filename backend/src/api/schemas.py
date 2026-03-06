"""API request/response schemas using Pydantic."""

from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Literal, Dict, Any
from src.config.schemas import Configuration, MicrogridConfig


# Training Schemas
class TrainRequest(BaseModel):
    """Request to start model training."""
    config: Configuration
    data: Optional[dict] = None  # Can upload data or reference path


class TrainResponse(BaseModel):
    """Response from training request."""
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str


class TrainingStatus(BaseModel):
    """Training job status."""
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = Field(ge=0.0, le=1.0)
    current_epoch: Optional[int] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None


# Prediction Schemas
class PredictRequest(BaseModel):
    """Request for predictions."""
    model_id: str
    input_data: List[List[float]]  # Shape: (samples, features)


class PredictResponse(BaseModel):
    """Prediction response."""
    predictions: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


# Simulation Schemas
class SimulateRequest(BaseModel):
    """Request to run microgrid simulation."""
    predictions: List[float]
    actual_pv: List[float]
    load_profile: List[float]
    microgrid_config: MicrogridConfig


class SimulationResponse(BaseModel):
    """Simulation results."""
    result_id: str
    timeseries: dict  # SOC, frequency, voltage over time
    metrics: dict     # Stability metrics


# Comparison Schemas
class CompareRequest(BaseModel):
    """Request for comparative analysis."""
    model_ids: List[str]
    test_data_path: str
    microgrid_config: MicrogridConfig


class ComparisonResponse(BaseModel):
    """Comparison results."""
    comparison_id: str
    results: dict
    rankings: dict
    improvements: dict


# Model Management Schemas
class ModelMetadata(BaseModel):
    """Model metadata."""
    model_id: str
    model_type: str
    created_at: str
    metrics: Optional[dict] = None
    architecture: Optional[dict] = None
    training: Optional[dict] = None
    data_info: Optional[dict] = None


class ModelListResponse(BaseModel):
    """List of models."""
    models: List[ModelMetadata]


# Data Management Schemas
class ValidationReport(BaseModel):
    """Data validation report."""
    valid: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    statistics: Optional[dict] = None


class DataUploadResponse(BaseModel):
    """Response from data upload."""
    data_id: str
    validation_report: ValidationReport


class DataValidateRequest(BaseModel):
    """Request to validate data."""
    data_path: str


# Error Response Schema
class ErrorDetail(BaseModel):
    """Error details."""
    field: Optional[str] = None
    provided: Optional[Any] = None
    required: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: Dict[str, Any]
