"""Classical prediction models: Persistence, ARIMA, and SVR."""

from typing import Dict, Any, Optional
import numpy as np
import time
from pathlib import Path
import pickle

from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class PersistenceModel:
    """
    Naive persistence forecast model.
    
    Predicts that the next value will be equal to the current value.
    This serves as a simple baseline for comparison.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PersistenceModel':
        """
        Fit method for API consistency (no actual training needed).
        
        Args:
            X: Input features (not used)
            y: Target values (not used)
        
        Returns:
            self
        """
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate persistence predictions.
        
        Uses the last value in the sequence as the prediction.
        
        Args:
            X: Input features, shape (n_samples, sequence_length, n_features)
               or (n_samples, n_features) for flattened input
        
        Returns:
            Predictions, shape (n_samples,)
        """
        # Handle both 3D and 2D input shapes
        if X.ndim == 3:
            # Take the last timestep, first feature (assumed to be PV power)
            return X[:, -1, 0]
        elif X.ndim == 2:
            # Take the first feature
            return X[:, 0]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")


class ClassicalPredictor(BasePredictor):
    """
    Classical forecasting methods for PV power prediction.
    
    Supports three methods:
    - Persistence: Naive forecast (next value = current value)
    - ARIMA: Auto-Regressive Integrated Moving Average
    - SVR: Support Vector Regression with RBF kernel
    
    All methods provide a consistent prediction interface for comparison
    with deep learning approaches.
    """
    
    def __init__(self, config: Dict[str, Any], method: str = 'persistence'):
        """
        Initialize classical predictor.
        
        Args:
            config: Model configuration dictionary
            method: Forecasting method - 'persistence', 'arima', or 'svr'
        
        Raises:
            ValueError: If method is not supported
        """
        super().__init__(config)
        
        if method not in ['persistence', 'arima', 'svr']:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Must be 'persistence', 'arima', or 'svr'"
            )
        
        self.method = method
        self.metadata['method'] = method
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def build_model(self) -> None:
        """
        Build the prediction model based on selected method.
        
        - Persistence: Simple model, no parameters
        - ARIMA: Uses order from config, defaults to (5, 1, 0)
        - SVR: Uses hyperparameters from config, defaults to RBF kernel
        """
        if self.method == 'persistence':
            self.model = PersistenceModel()
        
        elif self.method == 'arima':
            # ARIMA parameters from config or defaults
            order = self.config.get('arima_order', (5, 1, 0))
            seasonal_order = self.config.get('seasonal_order', (0, 0, 0, 0))
            self.metadata['arima_order'] = order
            self.metadata['seasonal_order'] = seasonal_order
            # ARIMA model will be fitted during training
            self.model = None
        
        elif self.method == 'svr':
            # SVR hyperparameters from config or defaults
            C = self.config.get('C', 1.0)
            epsilon = self.config.get('epsilon', 0.1)
            kernel = self.config.get('kernel', 'rbf')
            gamma = self.config.get('gamma', 'scale')
            
            self.model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon,
                gamma=gamma
            )
            
            self.metadata['svr_params'] = {
                'C': C,
                'epsilon': epsilon,
                'kernel': kernel,
                'gamma': gamma
            }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the classical model.
        
        Args:
            X_train: Training input features, shape (n_samples, sequence_length, n_features)
            y_train: Training target values, shape (n_samples,)
            X_val: Validation input features, shape (n_samples, sequence_length, n_features)
            y_val: Validation target values, shape (n_samples,)
        
        Returns:
            Dictionary containing training metrics:
                - train_loss: Training MAE
                - val_loss: Validation MAE
                - epochs_completed: 1 (classical models train in one pass)
                - convergence_status: 'converged'
                - loss_history: [train_loss]
        """
        start_time = time.time()
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Flatten sequences for classical models (they don't use temporal structure)
        # Shape: (n_samples, sequence_length * n_features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        # Train based on method
        if self.method == 'persistence':
            # No training needed for persistence
            self.model.fit(X_train, y_train)
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
        
        elif self.method == 'arima':
            # ARIMA uses only the target time series
            # Fit ARIMA model on training data
            order = self.config.get('arima_order', (5, 1, 0))
            seasonal_order = self.config.get('seasonal_order', (0, 0, 0, 0))
            
            try:
                arima_model = ARIMA(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order
                )
                self.model = arima_model.fit()
                
                # Generate predictions
                train_pred = self.model.fittedvalues
                # For validation, use forecast
                val_pred = self.model.forecast(steps=len(y_val))
                
            except Exception as e:
                # If ARIMA fails, fall back to persistence
                print(f"ARIMA fitting failed: {e}. Falling back to persistence.")
                self.method = 'persistence'
                self.model = PersistenceModel()
                self.model.fit(X_train, y_train)
                train_pred = self.model.predict(X_train)
                val_pred = self.model.predict(X_val)
        
        elif self.method == 'svr':
            # Fit scaler on training data
            self.scaler.fit(X_train_flat)
            X_train_scaled = self.scaler.transform(X_train_flat)
            X_val_scaled = self.scaler.transform(X_val_flat)
            
            # Train SVR
            self.model.fit(X_train_scaled, y_train)
            
            # Generate predictions
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
        
        self._is_fitted = True
        
        # Calculate losses (using MAE)
        train_loss = np.mean(np.abs(y_train - train_pred))
        val_loss = np.mean(np.abs(y_val - val_pred))
        
        # Track training time
        self._track_training_time(start_time)
        
        # Update metadata
        self._update_metadata(
            train_samples=len(X_train),
            val_samples=len(X_val),
            train_loss=float(train_loss),
            val_loss=float(val_loss)
        )
        
        return {
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'epochs_completed': 1,
            'convergence_status': 'converged',
            'loss_history': [float(train_loss)]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Input features, shape (n_samples, sequence_length, n_features)
        
        Returns:
            Predictions, shape (n_samples,)
        
        Raises:
            RuntimeError: If model hasn't been trained
        """
        if not self._is_fitted and self.method != 'persistence':
            raise RuntimeError("Model must be trained before making predictions")
        
        if self.method == 'persistence':
            return self.model.predict(X)
        
        elif self.method == 'arima':
            # ARIMA forecasting
            # Note: This is a simplified implementation
            # In practice, ARIMA would need the full time series context
            n_samples = X.shape[0]
            predictions = self.model.forecast(steps=n_samples)
            return np.array(predictions)
        
        elif self.method == 'svr':
            # Flatten and scale input
            X_flat = X.reshape(X.shape[0], -1)
            X_scaled = self.scaler.transform(X_flat)
            return self.model.predict(X_scaled)
    
    def save(self, path: str) -> None:
        """
        Save model weights and metadata to disk.
        
        Args:
            path: Directory path to save the model
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = save_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save method and fitted status
        method_path = save_path / "method.pkl"
        with open(method_path, 'wb') as f:
            pickle.dump({'method': self.method, 'is_fitted': self._is_fitted}, f)
        
        # Save metadata using parent class method
        import json
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load model weights and metadata from disk.
        
        Args:
            path: Directory path containing saved model
        
        Raises:
            FileNotFoundError: If model files don't exist
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load metadata
        import json
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load method and fitted status
        method_path = load_path / "method.pkl"
        if method_path.exists():
            with open(method_path, 'rb') as f:
                method_data = pickle.load(f)
                self.method = method_data['method']
                self._is_fitted = method_data['is_fitted']
        
        # Load model
        model_path = load_path / "model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
