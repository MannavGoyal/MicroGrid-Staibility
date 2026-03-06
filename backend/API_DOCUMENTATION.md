# Microgrid Stability Enhancement - API Documentation

## Overview

This document describes the REST API endpoints for the Microgrid Stability Enhancement backend. The API provides comprehensive functionality for training prediction models, running simulations, and analyzing microgrid stability.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, no authentication is required (development mode).

## Endpoints

### Health Check

#### GET /health

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "microgrid-backend"
}
```

---

### Training Endpoints

#### POST /api/train

Start a model training job.

**Request Body:**
```json
{
  "config": {
    "experiment_name": "lstm_baseline",
    "forecast_horizon": "15min",
    "model_configuration": {
      "model_type": "lstm",
      "hyperparameters": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2
      },
      "sequence_length": 12
    },
    "microgrid_configuration": {
      "mode": "islanded",
      "pv_capacity_kw": 10.0,
      "battery_capacity_kwh": 5.0,
      "battery_power_kw": 3.0,
      "inverter_capacity_kw": 12.0,
      "initial_soc_kwh": 2.5
    },
    "training_configuration": {
      "epochs": 50,
      "batch_size": 64,
      "learning_rate": 0.001,
      "validation_split": 0.2,
      "early_stopping_patience": 10
    },
    "data_path": "data/sample_data.csv",
    "output_dir": "results"
  }
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "train_abc123",
  "status": "queued",
  "message": "Training job train_abc123 started"
}
```

#### GET /api/train/{job_id}/status

Get training job status and progress.

**Response (200 OK):**
```json
{
  "job_id": "train_abc123",
  "status": "running",
  "progress": 0.65,
  "current_epoch": 33,
  "metrics": {
    "train_loss": 0.0234,
    "val_loss": 0.0289
  }
}
```

**Status values:** `queued`, `running`, `completed`, `failed`

#### DELETE /api/train/{job_id}

Cancel a running training job.

**Response:** 204 No Content

---

### Model Management Endpoints

#### GET /api/models

List all available trained models.

**Query Parameters:**
- `model_type` (optional): Filter by model type (lstm, arima, svr, etc.)
- `sort_by` (optional): Sort by 'created_at', 'mae', 'rmse'

**Response (200 OK):**
```json
{
  "models": [
    {
      "model_id": "lstm_20240115_143022",
      "model_type": "lstm",
      "created_at": "2024-01-15T14:30:22Z",
      "metrics": {
        "mae": 0.0456,
        "rmse": 0.0623,
        "r2": 0.8912
      },
      "architecture": {
        "hidden_size": 64,
        "num_layers": 2
      }
    }
  ]
}
```

#### GET /api/models/{model_id}

Get detailed model information.

**Response (200 OK):**
```json
{
  "model_id": "lstm_20240115_143022",
  "model_type": "lstm",
  "created_at": "2024-01-15T14:30:22Z",
  "metrics": {
    "mae": 0.0456,
    "rmse": 0.0623,
    "r2": 0.8912
  },
  "architecture": {
    "input_size": 7,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2
  },
  "training": {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

#### DELETE /api/models/{model_id}

Delete a saved model.

**Response:** 204 No Content

#### POST /api/models/{model_id}/load

Load model into memory for inference.

**Response (200 OK):**
```json
{
  "message": "Model lstm_20240115_143022 loaded successfully"
}
```

---

### Prediction Endpoints

#### POST /api/predict

Generate predictions using a trained model.

**Request Body:**
```json
{
  "model_id": "lstm_20240115_143022",
  "input_data": [
    [0.8, 28.5, 0.3, 0.65, 3.2],
    [0.82, 29.1, 0.25, 0.62, 3.5]
  ]
}
```

**Response (200 OK):**
```json
{
  "predictions": [0.756, 0.768, 0.781],
  "confidence_intervals": [[0.72, 0.79], [0.73, 0.81]]
}
```

---

### Simulation Endpoints

#### POST /api/simulate

Run microgrid simulation with predictions.

**Request Body:**
```json
{
  "predictions": [0.5, 0.6, 0.7],
  "actual_pv": [0.48, 0.62, 0.68],
  "load_profile": [0.45, 0.46, 0.47],
  "microgrid_config": {
    "mode": "islanded",
    "pv_capacity_kw": 10.0,
    "battery_capacity_kwh": 5.0,
    "battery_power_kw": 3.0,
    "inverter_capacity_kw": 12.0
  }
}
```

**Response (200 OK):**
```json
{
  "result_id": "sim_xyz789",
  "timeseries": {
    "time": [0, 300, 600],
    "soc": [2.5, 2.6, 2.7],
    "frequency_deviation": [0.1, 0.05, 0.02],
    "voltage_deviation": [0.5, 0.3, 0.2],
    "battery_power": [0.5, 0.3, 0.1]
  },
  "metrics": {
    "frequency": {
      "mean_absolute_deviation": 0.05,
      "std_deviation": 0.03,
      "max_deviation": 0.15
    },
    "battery": {
      "num_cycles": 2.5,
      "soc_range": 1.2
    }
  }
}
```

---

### Comparison Endpoints

#### POST /api/compare

Start comparative analysis across multiple models.

**Request Body:**
```json
{
  "model_ids": ["persistence", "arima_001", "lstm_20240115"],
  "test_data_path": "data/test_set.csv",
  "microgrid_config": {
    "mode": "islanded",
    "pv_capacity_kw": 10.0,
    "battery_capacity_kwh": 5.0,
    "battery_power_kw": 3.0,
    "inverter_capacity_kw": 12.0
  }
}
```

**Response (202 Accepted):**
```json
{
  "comparison_id": "comp_def456",
  "results": {},
  "rankings": {},
  "improvements": {}
}
```

#### GET /api/compare/{comparison_id}/status

Get comparison job status.

**Response (200 OK):**
```json
{
  "comparison_id": "comp_def456",
  "status": "running",
  "progress": 0.5
}
```

#### GET /api/compare/{comparison_id}/results

Get comparison results.

**Response (200 OK):**
```json
{
  "comparison_id": "comp_def456",
  "results": {
    "lstm": {
      "mae": 0.045,
      "frequency_std": 0.12
    },
    "arima": {
      "mae": 0.067,
      "frequency_std": 0.18
    }
  },
  "rankings": {
    "mae": ["lstm", "arima", "persistence"],
    "frequency_std": ["lstm", "arima", "persistence"]
  },
  "improvements": {
    "lstm": {
      "mae_improvement": 45.2,
      "frequency_improvement": 33.3
    }
  }
}
```

---

### Export Endpoints

#### GET /api/export/{result_id}

Download exported results.

**Query Parameters:**
- `format`: 'csv', 'json', 'markdown', 'pdf'
- `include`: 'timeseries', 'metrics', 'plots', 'config', 'all'

**Response (200 OK):** File download

---

### Data Management Endpoints

#### POST /api/data/upload

Upload time-series data file.

**Request:** Multipart form data with CSV file

**Response (200 OK):**
```json
{
  "data_id": "data_abc123",
  "validation_report": {
    "valid": true,
    "warnings": ["3 outliers detected"],
    "errors": [],
    "statistics": {
      "num_samples": 8640,
      "date_range": ["2024-01-01", "2024-01-30"]
    }
  }
}
```

#### POST /api/data/validate

Validate data quality.

**Request Body:**
```json
{
  "data_path": "data/sample_data.csv"
}
```

**Response (200 OK):**
```json
{
  "valid": true,
  "warnings": ["3 outliers detected"],
  "errors": [],
  "statistics": {
    "num_samples": 8640,
    "features": ["irradiance", "temperature", "cloud_cover"]
  }
}
```

#### GET /api/data/{data_id}

Get data metadata.

**Response (200 OK):**
```json
{
  "data_id": "data_abc123",
  "num_samples": 8640,
  "columns": ["irradiance", "temperature", "pv_power"],
  "date_range": ["2024-01-01", "2024-01-30"]
}
```

---

## Error Responses

All endpoints follow a consistent error response format:

```json
{
  "error": {
    "code": "INVALID_CONFIG",
    "message": "Inverter capacity must be >= PV capacity",
    "details": {
      "field": "microgrid_config.inverter_capacity_kw",
      "provided": 8.0,
      "required": ">= 10.0"
    }
  }
}
```

### HTTP Status Codes

- **200 OK**: Request successful
- **202 Accepted**: Async job started
- **204 No Content**: Successful deletion
- **400 Bad Request**: Invalid input
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error

### Error Codes

- `VALIDATION_ERROR`: Request validation failed
- `NOT_FOUND`: Resource not found
- `INVALID_CONFIG`: Configuration validation failed
- `INVALID_FORMAT`: Unsupported format
- `MISSING_FILE`: Required file not provided
- `INTERNAL_ERROR`: Internal server error

---

## Complete Workflow Example

### 1. Upload and Validate Data

```bash
curl -X POST http://localhost:5000/api/data/validate \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/sample_data.csv"}'
```

### 2. Start Training

```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d @config.json
```

### 3. Check Training Status

```bash
curl http://localhost:5000/api/train/train_abc123/status
```

### 4. List Available Models

```bash
curl http://localhost:5000/api/models
```

### 5. Generate Predictions

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "lstm_20240115_143022",
    "input_data": [[0.8, 28.5, 0.3, 0.65, 3.2]]
  }'
```

### 6. Run Simulation

```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d @simulation_request.json
```

### 7. Export Results

```bash
curl "http://localhost:5000/api/export/sim_xyz789?format=csv&include=all" \
  -o results.csv
```

---

## Running the Server

### Development Mode

```bash
python backend/src/app.py
```

The server will start on `http://localhost:5000` with debug mode enabled.

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.src.app:create_app()
```

---

## Testing

### Run End-to-End Test

```bash
python backend/examples/test_end_to_end.py
```

### Test API Endpoints

```bash
# Start the server first
python backend/src/app.py

# In another terminal
python backend/examples/test_api.py
```

---

## Notes

- All timestamps are in ISO 8601 format
- All power values are in kW
- All energy values are in kWh
- All time durations are in seconds
- Training and comparison jobs run synchronously in the current implementation
- For production, implement async job processing with Celery/Redis
