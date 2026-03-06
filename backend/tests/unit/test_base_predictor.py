"""Unit tests for BasePredictor abstract class."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from backend.src.models.base import BasePredictor


class DummyPredictor(BasePredictor):
    """Concrete implementation of BasePredictor for testing."""
    
    def build_model(self):
        """Build a dummy model."""
        self.model = {'type': 'dummy', 'weights': np.array([1.0, 2.0, 3.0])}
    
    def train(self, X_train, y_train, X_val, y_val):
        """Dummy training that returns mock metrics."""
        self.build_model()
        return {
            'train_loss': 0.05,
            'val_loss': 0.06,
            'epochs_completed': 10,
            'convergence_status': 'converged',
            'loss_history': [0.1, 0.08, 0.06, 0.05]
        }
    
    def predict(self, X):
        """Dummy prediction that returns mean of inputs."""
        return np.mean(X, axis=(1, 2))


class TestBasePredictor:
    """Test suite for BasePredictor class."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        config = {'hidden_size': 64, 'num_layers': 2}
        predictor = DummyPredictor(config)
        
        assert predictor.config == config
        assert predictor.model is None
        assert predictor.scaler is None
        assert 'hyperparameters' in predictor.metadata
        assert predictor.metadata['hyperparameters'] == config
        assert 'created_at' in predictor.metadata
        assert predictor.metadata['framework'] == 'DummyPredictor'
    
    def test_evaluate_mae(self):
        """Test MAE calculation."""
        predictor = DummyPredictor({})
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert abs(metrics['mae'] - expected_mae) < 1e-6
    
    def test_evaluate_rmse(self):
        """Test RMSE calculation."""
        predictor = DummyPredictor({})
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(metrics['rmse'] - expected_rmse) < 1e-6
    
    def test_evaluate_mape(self):
        """Test MAPE calculation."""
        predictor = DummyPredictor({})
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert abs(metrics['mape'] - expected_mape) < 1e-6
    
    def test_evaluate_mape_with_zeros(self):
        """Test MAPE calculation with zero values."""
        predictor = DummyPredictor({})
        y_true = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([0.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        # MAPE should only consider non-zero true values
        non_zero_mask = y_true != 0
        expected_mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        assert abs(metrics['mape'] - expected_mape) < 1e-6
    
    def test_evaluate_r2(self):
        """Test R² calculation."""
        predictor = DummyPredictor({})
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)
        assert abs(metrics['r2'] - expected_r2) < 1e-6
    
    def test_evaluate_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        predictor = DummyPredictor({})
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = predictor.evaluate(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mape'] == 0.0
        assert abs(metrics['r2'] - 1.0) < 1e-6
    
    def test_save_and_load(self):
        """Test model save and load functionality."""
        config = {'hidden_size': 64, 'num_layers': 2}
        predictor = DummyPredictor(config)
        predictor.build_model()
        predictor.metadata['test_metric'] = 0.95
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save model
            save_path = Path(temp_dir) / "test_model"
            predictor.save(str(save_path))
            
            # Verify files exist
            assert (save_path / "model.pkl").exists()
            assert (save_path / "metadata.json").exists()
            
            # Load model
            new_predictor = DummyPredictor(config)
            new_predictor.load(str(save_path))
            
            # Verify loaded data
            assert new_predictor.model is not None
            assert new_predictor.model['type'] == 'dummy'
            assert np.array_equal(new_predictor.model['weights'], np.array([1.0, 2.0, 3.0]))
            assert new_predictor.metadata['test_metric'] == 0.95
            assert new_predictor.metadata['framework'] == 'DummyPredictor'
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        predictor = DummyPredictor({})
        
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            predictor.load("/nonexistent/path")
    
    def test_load_framework_mismatch(self):
        """Test loading model with mismatched framework."""
        config = {'hidden_size': 64}
        predictor = DummyPredictor(config)
        predictor.build_model()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            save_path = Path(temp_dir) / "test_model"
            predictor.save(str(save_path))
            
            # Modify metadata to simulate framework mismatch
            metadata_path = save_path / "metadata.json"
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['framework'] = 'DifferentPredictor'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Try to load with wrong framework
            new_predictor = DummyPredictor(config)
            with pytest.raises(ValueError, match="Model framework mismatch"):
                new_predictor.load(str(save_path))
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_metadata(self):
        """Test metadata retrieval."""
        config = {'hidden_size': 64}
        predictor = DummyPredictor(config)
        
        metadata = predictor.get_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata['hyperparameters'] == config
        assert 'created_at' in metadata
        assert metadata['framework'] == 'DummyPredictor'
        
        # Verify it's a copy (modifying returned dict shouldn't affect original)
        metadata['new_key'] = 'new_value'
        assert 'new_key' not in predictor.metadata
    
    def test_train_integration(self):
        """Test training integration."""
        predictor = DummyPredictor({'hidden_size': 32})
        
        # Create dummy data
        X_train = np.random.rand(100, 12, 5)
        y_train = np.random.rand(100)
        X_val = np.random.rand(20, 12, 5)
        y_val = np.random.rand(20)
        
        # Train
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert 'epochs_completed' in result
        assert 'convergence_status' in result
        assert predictor.model is not None
    
    def test_predict_integration(self):
        """Test prediction integration."""
        predictor = DummyPredictor({'hidden_size': 32})
        predictor.build_model()
        
        # Create dummy data
        X = np.random.rand(10, 12, 5)
        
        # Predict
        predictions = predictor.predict(X)
        
        assert predictions.shape == (10,)
        assert np.all(np.isfinite(predictions))
