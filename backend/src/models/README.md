# Prediction Models

This module contains all prediction models for PV power forecasting.

## BasePredictor

The `BasePredictor` abstract class provides a common interface for all prediction models.

### Features

- **Abstract Interface**: Defines `build_model()`, `train()`, and `predict()` methods
- **Performance Metrics**: Built-in `evaluate()` method calculating MAE, RMSE, MAPE, and R²
- **Model Persistence**: `save()` and `load()` methods for model serialization
- **Metadata Tracking**: Automatic tracking of hyperparameters, training time, and version

### Usage

```python
from src.models.base import BasePredictor
import numpy as np

class MyPredictor(BasePredictor):
    def build_model(self):
        # Implement model architecture
        self.model = ...
    
    def train(self, X_train, y_train, X_val, y_val):
        # Implement training logic
        self.build_model()
        # ... training code ...
        return {
            'train_loss': ...,
            'val_loss': ...,
            'epochs_completed': ...,
            'convergence_status': ...,
            'loss_history': [...]
        }
    
    def predict(self, X):
        # Implement prediction logic
        return predictions

# Create and use predictor
config = {'hidden_size': 64, 'num_layers': 2}
predictor = MyPredictor(config)

# Train
result = predictor.train(X_train, y_train, X_val, y_val)

# Predict
predictions = predictor.predict(X_test)

# Evaluate
metrics = predictor.evaluate(y_test, predictions)
print(f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")

# Save
predictor.save('models/my_model')

# Load
new_predictor = MyPredictor(config)
new_predictor.load('models/my_model')
```

### Metrics

The `evaluate()` method calculates:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **MAPE** (Mean Absolute Percentage Error): Average percentage error (handles zero values)
- **R²** (Coefficient of Determination): Proportion of variance explained by the model

### Model Persistence

Models are saved in the following structure:

```
{path}/
├── model.pkl       # Model weights/parameters
├── scaler.pkl      # Fitted data scaler (if used)
└── metadata.json   # Configuration and metrics
```

### Requirements Satisfied

- **Requirement 3.5**: Consistent prediction interface across all model types
- **Requirement 10.1-10.4**: MAE, RMSE, MAPE, R² metrics
- **Requirement 15.1-15.5**: Model persistence with metadata

### Examples

See `backend/examples/test_base_predictor.py` for a complete working example.

### Tests

Unit tests are located in `backend/tests/unit/test_base_predictor.py`.

Run tests with:
```bash
pytest backend/tests/unit/test_base_predictor.py -v
```
