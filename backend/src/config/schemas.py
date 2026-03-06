"""Configuration schemas using Pydantic for validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum


class ForecastHorizon(str, Enum):
    """Supported forecast horizons."""
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1hour"


class MicrogridMode(str, Enum):
    """Supported microgrid operating modes."""
    GRID_CONNECTED = "grid_connected"
    ISLANDED = "islanded"
    HYBRID = "hybrid_ac_dc"


class ModelType(str, Enum):
    """Supported prediction model types."""
    PERSISTENCE = "persistence"
    ARIMA = "arima"
    SVR = "svr"
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"


class ModelConfig(BaseModel):
    """Model configuration."""
    model_type: ModelType
    hyperparameters: dict = Field(default_factory=dict)
    sequence_length: int = Field(default=12, ge=4, le=48)


class MicrogridConfig(BaseModel):
    """Microgrid configuration."""
    mode: MicrogridMode
    pv_capacity_kw: float = Field(gt=0)
    battery_capacity_kwh: float = Field(gt=0)
    battery_power_kw: float = Field(gt=0)
    inverter_capacity_kw: float = Field(gt=0)
    initial_soc_kwh: Optional[float] = None
    has_diesel_generator: bool = False
    diesel_capacity_kw: Optional[float] = None

    @field_validator('inverter_capacity_kw')
    @classmethod
    def inverter_must_exceed_pv(cls, v, info):
        """Validate that inverter capacity is >= PV capacity."""
        if 'pv_capacity_kw' in info.data and v < info.data['pv_capacity_kw']:
            raise ValueError('Inverter capacity must be >= PV capacity')
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = Field(default=50, ge=10, le=500)
    batch_size: int = Field(default=64, ge=16, le=512)
    learning_rate: float = Field(default=0.001, gt=0, lt=1)
    validation_split: float = Field(default=0.2, gt=0, lt=0.5)
    early_stopping_patience: int = Field(default=10, ge=3)


class Configuration(BaseModel):
    """Complete system configuration."""
    experiment_name: str
    forecast_horizon: ForecastHorizon
    model_configuration: ModelConfig
    microgrid_configuration: MicrogridConfig
    training_configuration: TrainingConfig
    data_path: str
    output_dir: str = "results"
