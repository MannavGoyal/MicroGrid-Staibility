"""Base predictor abstract class for all prediction models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import pickle
import time
from datetime import datetime, timezone


class BasePredictor(ABC):
    """
    Abstract base class for all prediction models.
    
    Provides common interface and functionality for:
    - Model training and prediction
    - Performance evaluation (MAE, RMSE, MAPE, R²)
    - Model persistence (save/load)
    - Metadata tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize predictor with configuration.
        
        Args:
            config: Model configuration dictionary containing hyperparameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {
            'hyperparameters': config.copy(),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0',
            'training_time_seconds': None,
            'framework': self.__class__.__name__
        }
    
    @abstractmethod
    def build_model(self) -> None:
        """
        Build the prediction model architecture.
        
        Must be implemented by subclasses to create the specific model.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the model and return training metrics.
        
        Args:
            X_train: Training input features, shape (n_samples, sequence_length, n_features)
            y_train: Training target values, shape (n_samples,)
            X_val: Validation input features, shape (n_samples, sequence_length, n_features)
            y_val: Validation target values, shape (n_samples,)
        
        Returns:
            Dictionary containing training metrics:
                - train_loss: Final training loss
                - val_loss: Final validation loss
                - epochs_completed: Number of epochs trained
                - convergence_status: 'converged', 'early_stopped', or 'max_epochs'
                - loss_history: List of loss values per epoch
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Input features, shape (n_samples, sequence_length, n_features)
        
        Returns:
            Predictions, shape (n_samples,)
        """
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction performance metrics.
        
        Computes:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error)
        - R² (Coefficient of Determination)
        
        Args:
            y_true: True target values, shape (n_samples,)
            y_pred: Predicted values, shape (n_samples,)
        
        Returns:
            Dictionary containing metrics:
                - mae: Mean Absolute Error
                - rmse: Root Mean Squared Error
                - mape: Mean Absolute Percentage Error (%)
                - r2: R-squared coefficient
        """
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Calculate MAPE (avoid division by zero)
        # Only calculate for non-zero true values
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def save(self, path: str) -> None:
        """
        Save model weights and metadata to disk.
        
        Creates a directory structure:
        {path}/
            ├── model.pkl or model.pt (model weights)
            ├── scaler.pkl (fitted scaler)
            ├── metadata.json (model configuration and metrics)
        
        Args:
            path: Directory path to save the model
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model (implementation depends on subclass)
        model_path = save_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler if it exists
        if self.scaler is not None:
            scaler_path = save_path / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save metadata
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
            ValueError: If loaded model architecture doesn't match
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load metadata
        metadata_path = load_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Validate framework matches
        if self.metadata.get('framework') != self.__class__.__name__:
            raise ValueError(
                f"Model framework mismatch: expected {self.__class__.__name__}, "
                f"got {self.metadata.get('framework')}"
            )
        
        # Load model
        model_path = load_path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if it exists
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def _track_training_time(self, start_time: float) -> None:
        """
        Track training time in metadata.
        
        Args:
            start_time: Training start time from time.time()
        """
        training_time = time.time() - start_time
        self.metadata['training_time_seconds'] = training_time
    
    def _update_metadata(self, **kwargs) -> None:
        """
        Update metadata with additional information.
        
        Args:
            **kwargs: Key-value pairs to add to metadata
        """
        self.metadata.update(kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        return self.metadata.copy()
