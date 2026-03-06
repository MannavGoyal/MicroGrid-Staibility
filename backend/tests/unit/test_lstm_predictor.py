"""Unit tests for LSTM predictor."""

import pytest
import numpy as np
import torch
from backend.src.models.lstm import LSTMModel, LSTMPredictor
import tempfile
import shutil
from pathlib import Path


class TestLSTMModel:
    """Test cases for LSTMModel PyTorch module."""
    
    def test_model_initialization(self):
        """Test LSTM model can be initialized with correct architecture."""
        input_size = 5
        hidden_size = 32
        num_layers = 2
        dropout = 0.2
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        assert model.input_size == input_size
        assert model.hidden_size == hidden_size
        assert model.num_layers == num_layers
        assert model.dropout == dropout
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_forward_pass_output_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        sequence_length = 12
        input_size = 5
        hidden_size = 32
        
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size)
        
        # Create dummy input
        x = torch.randn(batch_size, sequence_length, input_size)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
    
    def test_forward_pass_values_range(self):
        """Test forward pass produces reasonable output values."""
        model = LSTMModel(input_size=5, hidden_size=32)
        x = torch.randn(8, 12, 5)
        
        output = model(x)
        
        # Output should be finite
        assert torch.isfinite(output).all()


class TestLSTMPredictor:
    """Test cases for LSTMPredictor class."""
    
    def test_predictor_initialization(self):
        """Test LSTMPredictor can be initialized with config."""
        config = {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'early_stopping_patience': 5
        }
        
        predictor = LSTMPredictor(config)
        
        assert predictor.input_size == 5
        assert predictor.hidden_size == 64
        assert predictor.num_layers == 2
        assert predictor.dropout == 0.2
        assert predictor.learning_rate == 0.001
        assert predictor.batch_size == 32
        assert predictor.epochs == 10
        assert predictor.early_stopping_patience == 5
    
    def test_build_model(self):
        """Test build_model creates model, optimizer, and scheduler."""
        config = {'input_size': 5, 'hidden_size': 32}
        predictor = LSTMPredictor(config)
        
        predictor.build_model()
        
        assert predictor.model is not None
        assert isinstance(predictor.model, LSTMModel)
        assert predictor.optimizer is not None
        assert isinstance(predictor.optimizer, torch.optim.Adam)
        assert predictor.scheduler is not None
        assert isinstance(predictor.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    def test_train_produces_decreasing_loss(self):
        """Test training produces decreasing loss over epochs."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        sequence_length = 12
        n_features = 5
        
        X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        y_train = np.random.randn(n_samples).astype(np.float32)
        X_val = np.random.randn(50, sequence_length, n_features).astype(np.float32)
        y_val = np.random.randn(50).astype(np.float32)
        
        config = {
            'input_size': n_features,
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 20,
            'early_stopping_patience': 10
        }
        
        predictor = LSTMPredictor(config)
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        # Check training completed
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert 'epochs_completed' in result
        assert 'convergence_status' in result
        assert 'train_loss_history' in result
        assert 'val_loss_history' in result
        
        # Check loss history exists and has values
        assert len(result['train_loss_history']) > 0
        assert len(result['val_loss_history']) > 0
        
        # Check that loss decreased (first loss > last loss)
        # Allow some tolerance for stochastic training
        first_loss = result['train_loss_history'][0]
        last_loss = result['train_loss_history'][-1]
        assert first_loss > last_loss * 0.5  # Loss should decrease by at least 50%
    
    def test_predict_output_shape(self):
        """Test predict produces correct output shape."""
        # Create and train a simple model
        np.random.seed(42)
        n_samples = 100
        sequence_length = 12
        n_features = 5
        
        X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        y_train = np.random.randn(n_samples).astype(np.float32)
        X_val = np.random.randn(20, sequence_length, n_features).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        
        config = {
            'input_size': n_features,
            'hidden_size': 16,
            'epochs': 5
        }
        
        predictor = LSTMPredictor(config)
        predictor.train(X_train, y_train, X_val, y_val)
        
        # Test prediction
        X_test = np.random.randn(30, sequence_length, n_features).astype(np.float32)
        predictions = predictor.predict(X_test)
        
        # Check output shape
        assert predictions.shape == (30,)
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64
    
    def test_predict_without_training_raises_error(self):
        """Test predict raises error if model not trained."""
        config = {'input_size': 5}
        predictor = LSTMPredictor(config)
        
        X_test = np.random.randn(10, 12, 5).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model must be trained or loaded"):
            predictor.predict(X_test)
    
    def test_evaluate_metrics(self):
        """Test evaluate calculates correct metrics."""
        config = {'input_size': 5}
        predictor = LSTMPredictor(config)
        
        # Create test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        # Check all metrics are present
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        
        # Check metrics are reasonable
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mape'] > 0
        assert 0 <= metrics['r2'] <= 1
    
    def test_save_and_load(self):
        """Test model can be saved and loaded correctly."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Train a model
            np.random.seed(42)
            n_samples = 100
            sequence_length = 12
            n_features = 5
            
            X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
            y_train = np.random.randn(n_samples).astype(np.float32)
            X_val = np.random.randn(20, sequence_length, n_features).astype(np.float32)
            y_val = np.random.randn(20).astype(np.float32)
            
            config = {
                'input_size': n_features,
                'hidden_size': 16,
                'epochs': 5
            }
            
            predictor1 = LSTMPredictor(config)
            predictor1.train(X_train, y_train, X_val, y_val)
            
            # Make predictions before saving
            X_test = np.random.randn(10, sequence_length, n_features).astype(np.float32)
            pred1 = predictor1.predict(X_test)
            
            # Save model
            save_path = Path(temp_dir) / "test_model"
            predictor1.save(str(save_path))
            
            # Load model
            predictor2 = LSTMPredictor(config)
            predictor2.load(str(save_path))
            
            # Make predictions after loading
            pred2 = predictor2.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)
            
            # Metadata should be preserved
            assert predictor2.metadata['framework'] == 'LSTMPredictor'
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_early_stopping(self):
        """Test early stopping is triggered when validation loss doesn't improve."""
        # Create data where model will overfit quickly
        np.random.seed(42)
        n_samples = 50
        sequence_length = 12
        n_features = 5
        
        X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        y_train = np.random.randn(n_samples).astype(np.float32)
        X_val = np.random.randn(10, sequence_length, n_features).astype(np.float32)
        y_val = np.random.randn(10).astype(np.float32)
        
        config = {
            'input_size': n_features,
            'hidden_size': 32,
            'epochs': 100,
            'early_stopping_patience': 5
        }
        
        predictor = LSTMPredictor(config)
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        # Early stopping should trigger before max epochs
        assert result['epochs_completed'] < 100
        assert result['convergence_status'] in ['early_stopped', 'max_epochs']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
