"""LSTM prediction model for time-series forecasting."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any
import time

from .base import BasePredictor


class LSTMModel(nn.Module):
    """
    LSTM neural network for time-series prediction.
    
    Architecture:
    - Multi-layer LSTM with configurable hidden size and layers
    - Dropout for regularization
    - Fully connected output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability for regularization
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        # Shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        # Shape: (batch_size, 1)
        output = self.fc(last_output)
        
        return output


class LSTMPredictor(BasePredictor):
    """
    LSTM-based predictor implementing the BasePredictor interface.
    
    Features:
    - Configurable LSTM architecture
    - Adam optimizer with learning rate scheduling
    - Early stopping based on validation loss
    - MSE loss function
    - Training metrics tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM predictor.
        
        Args:
            config: Configuration dictionary with keys:
                - input_size: Number of input features
                - hidden_size: LSTM hidden units (default: 64)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout probability (default: 0.2)
                - learning_rate: Learning rate (default: 0.001)
                - batch_size: Batch size (default: 64)
                - epochs: Maximum epochs (default: 50)
                - early_stopping_patience: Patience for early stopping (default: 10)
        """
        super().__init__(config)
        
        # Extract hyperparameters with defaults
        self.input_size = config.get('input_size')
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 50)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
    
    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        if self.input_size is None:
            raise ValueError("input_size must be specified in config")
        
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Update metadata
        self._update_metadata(
            architecture={
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input features, shape (n_samples, sequence_length, n_features)
            y_train: Training target values, shape (n_samples,)
            X_val: Validation input features, shape (n_samples, sequence_length, n_features)
            y_val: Validation target values, shape (n_samples,)
        
        Returns:
            Dictionary containing:
                - train_loss: Final training loss
                - val_loss: Final validation loss
                - epochs_completed: Number of epochs trained
                - convergence_status: 'converged', 'early_stopped', or 'max_epochs'
                - train_loss_history: List of training losses per epoch
                - val_loss_history: List of validation losses per epoch
        """
        start_time = time.time()
        
        # Build model if not already built
        if self.model is None:
            # Infer input_size from data if not specified
            if self.input_size is None:
                self.input_size = X_train.shape[2]
                self.config['input_size'] = self.input_size
            self.build_model()
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        convergence_status = 'max_epochs'
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            train_loss_history.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
                val_loss_value = val_loss.item()
                val_loss_history.append(val_loss_value)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss_value)
            
            # Early stopping check
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_loss_value:.6f}")
            
            # Check early stopping
            if patience_counter >= self.early_stopping_patience:
                convergence_status = 'early_stopped'
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Track training time
        self._track_training_time(start_time)
        
        # Update metadata with training info
        self._update_metadata(
            training={
                'epochs_completed': epoch + 1,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            },
            convergence_status=convergence_status
        )
        
        return {
            'train_loss': train_loss_history[-1],
            'val_loss': val_loss_history[-1],
            'epochs_completed': epoch + 1,
            'convergence_status': convergence_status,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Input features, shape (n_samples, sequence_length, n_features)
        
        Returns:
            Predictions, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Convert back to numpy and flatten
        return predictions.cpu().numpy().flatten()
    
    def save(self, path: str) -> None:
        """
        Save model weights and metadata to disk.
        
        Saves:
        - model.pt: PyTorch model state dict
        - metadata.json: Model configuration and metrics
        - scaler.pkl: Fitted scaler (if exists)
        
        Args:
            path: Directory path to save the model
        """
        from pathlib import Path
        import json
        import pickle
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model state dict
        if self.model is not None:
            model_path = save_path / "model.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'architecture': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout
                }
            }, model_path)
        
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
        from pathlib import Path
        import json
        import pickle
        
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
        
        # Load PyTorch model
        model_path = load_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract architecture from checkpoint
        arch = checkpoint['architecture']
        self.input_size = arch['input_size']
        self.hidden_size = arch['hidden_size']
        self.num_layers = arch['num_layers']
        self.dropout = arch['dropout']
        
        # Build model with loaded architecture
        self.build_model()
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler if it exists
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Set model to evaluation mode
        self.model.eval()
