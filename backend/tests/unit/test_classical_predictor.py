"""Unit tests for ClassicalPredictor models."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from backend.src.models.classical import ClassicalPredictor, PersistenceModel


class TestPersistenceModel:
    """Test suite for PersistenceModel."""
    
    def test_predict_3d_input(self):
        """Test persistence prediction with 3D input."""
        model = PersistenceModel()
        X = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Sample 1
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  # Sample 2
        ])
        
        predictions = model.predict(X)
        
        # Should return last timestep, first feature
        assert predictions.shape == (2,)
        assert predictions[0] == 5.0
        assert predictions[1] == 11.0
    
    def test_predict_2d_input(self):
        """Test persistence prediction with 2D input."""
        model = PersistenceModel()
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        predictions = model.predict(X)
        
        # Should return first feature
        assert predictions.shape == (2,)
        assert predictions[0] == 1.0
        assert predictions[1] == 4.0
    
    def test_predict_invalid_shape(self):
        """Test persistence prediction with invalid input shape."""
        model = PersistenceModel()
        X = np.array([1.0, 2.0, 3.0])  # 1D array
        
        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            model.predict(X)
    
    def test_fit_returns_self(self):
        """Test that fit returns self for API consistency."""
        model = PersistenceModel()
        X = np.random.rand(10, 5, 3)
        y = np.random.rand(10)
        
        result = model.fit(X, y)
        assert result is model


class TestClassicalPredictor:
    """Test suite for ClassicalPredictor class."""
    
    def test_initialization_persistence(self):
        """Test initialization with persistence method."""
        config = {}
        predictor = ClassicalPredictor(config, method='persistence')
        
        assert predictor.method == 'persistence'
        assert predictor.metadata['method'] == 'persistence'
        assert not predictor._is_fitted
    
    def test_initialization_svr(self):
        """Test initialization with SVR method."""
        config = {'C': 2.0, 'epsilon': 0.2}
        predictor = ClassicalPredictor(config, method='svr')
        
        assert predictor.method == 'svr'
        assert predictor.metadata['method'] == 'svr'
    
    def test_initialization_arima(self):
        """Test initialization with ARIMA method."""
        config = {'arima_order': (3, 1, 2)}
        predictor = ClassicalPredictor(config, method='arima')
        
        assert predictor.method == 'arima'
        assert predictor.metadata['method'] == 'arima'
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        config = {}
        
        with pytest.raises(ValueError, match="Unsupported method"):
            ClassicalPredictor(config, method='invalid_method')
    
    def test_build_model_persistence(self):
        """Test building persistence model."""
        predictor = ClassicalPredictor({}, method='persistence')
        predictor.build_model()
        
        assert isinstance(predictor.model, PersistenceModel)
    
    def test_build_model_svr(self):
        """Test building SVR model."""
        config = {'C': 1.5, 'epsilon': 0.15, 'kernel': 'rbf'}
        predictor = ClassicalPredictor(config, method='svr')
        predictor.build_model()
        
        assert predictor.model is not None
        assert predictor.metadata['svr_params']['C'] == 1.5
        assert predictor.metadata['svr_params']['epsilon'] == 0.15
        assert predictor.metadata['svr_params']['kernel'] == 'rbf'
    
    def test_build_model_arima(self):
        """Test building ARIMA model."""
        config = {'arima_order': (2, 1, 1)}
        predictor = ClassicalPredictor(config, method='arima')
        predictor.build_model()
        
        assert predictor.metadata['arima_order'] == (2, 1, 1)
    
    def test_train_persistence(self):
        """Test training persistence model."""
        np.random.seed(42)
        predictor = ClassicalPredictor({}, method='persistence')
        
        X_train = np.random.rand(50, 12, 5)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert result['epochs_completed'] == 1
        assert result['convergence_status'] == 'converged'
        assert len(result['loss_history']) == 1
        assert predictor._is_fitted
    
    def test_train_svr(self):
        """Test training SVR model."""
        np.random.seed(42)
        config = {'C': 1.0, 'epsilon': 0.1}
        predictor = ClassicalPredictor(config, method='svr')
        
        X_train = np.random.rand(50, 12, 5)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert result['epochs_completed'] == 1
        assert result['convergence_status'] == 'converged'
        assert predictor._is_fitted
        assert predictor.metadata['train_samples'] == 50
        assert predictor.metadata['val_samples'] == 10
    
    def test_train_arima(self):
        """Test training ARIMA model."""
        np.random.seed(42)
        config = {'arima_order': (2, 1, 0)}
        predictor = ClassicalPredictor(config, method='arima')
        
        # Create more realistic time series
        n_train = 50
        t = np.linspace(0, 10, n_train)
        y_train = 0.5 + 0.3 * np.sin(t) + 0.05 * np.random.randn(n_train)
        X_train = np.random.rand(n_train, 12, 5)
        
        n_val = 10
        t_val = np.linspace(10, 12, n_val)
        y_val = 0.5 + 0.3 * np.sin(t_val) + 0.05 * np.random.randn(n_val)
        X_val = np.random.rand(n_val, 12, 5)
        
        result = predictor.train(X_train, y_train, X_val, y_val)
        
        assert 'train_loss' in result
        assert 'val_loss' in result
        assert result['epochs_completed'] == 1
        assert result['convergence_status'] == 'converged'
        assert predictor._is_fitted
    
    def test_predict_persistence(self):
        """Test prediction with persistence model."""
        np.random.seed(42)
        predictor = ClassicalPredictor({}, method='persistence')
        
        X_train = np.random.rand(50, 12, 5)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        X_test = np.random.rand(5, 12, 5)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (5,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_svr(self):
        """Test prediction with SVR model."""
        np.random.seed(42)
        predictor = ClassicalPredictor({'C': 1.0}, method='svr')
        
        X_train = np.random.rand(50, 12, 5)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        X_test = np.random.rand(5, 12, 5)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (5,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_arima(self):
        """Test prediction with ARIMA model."""
        np.random.seed(42)
        predictor = ClassicalPredictor({'arima_order': (2, 1, 0)}, method='arima')
        
        n_train = 50
        t = np.linspace(0, 10, n_train)
        y_train = 0.5 + 0.3 * np.sin(t) + 0.05 * np.random.randn(n_train)
        X_train = np.random.rand(n_train, 12, 5)
        
        n_val = 10
        y_val = np.random.rand(n_val)
        X_val = np.random.rand(n_val, 12, 5)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        X_test = np.random.rand(5, 12, 5)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (5,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_before_training(self):
        """Test that prediction fails before training for non-persistence models."""
        predictor = ClassicalPredictor({'C': 1.0}, method='svr')
        X_test = np.random.rand(5, 12, 5)
        
        with pytest.raises(RuntimeError, match="Model must be trained"):
            predictor.predict(X_test)
    
    def test_evaluate_metrics(self):
        """Test evaluation metrics calculation."""
        np.random.seed(42)
        predictor = ClassicalPredictor({}, method='persistence')
        
        X_train = np.random.rand(50, 12, 5)
        y_train = np.random.rand(50)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        predictions = predictor.predict(X_val)
        
        metrics = predictor.evaluate(y_val, predictions)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert all(np.isfinite(v) for k, v in metrics.items() if k != 'mape')
    
    def test_save_and_load_persistence(self):
        """Test save and load for persistence model."""
        np.random.seed(42)
        predictor = ClassicalPredictor({}, method='persistence')
        
        X_train = np.random.rand(30, 12, 5)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        # Make predictions before saving
        X_test = np.random.rand(5, 12, 5)
        pred_before = predictor.predict(X_test)
        
        # Save model
        temp_dir = tempfile.mkdtemp()
        try:
            save_path = Path(temp_dir) / "persistence_model"
            predictor.save(str(save_path))
            
            # Verify files exist
            assert (save_path / "model.pkl").exists()
            assert (save_path / "scaler.pkl").exists()
            assert (save_path / "method.pkl").exists()
            assert (save_path / "metadata.json").exists()
            
            # Load model
            new_predictor = ClassicalPredictor({}, method='persistence')
            new_predictor.load(str(save_path))
            
            # Make predictions after loading
            pred_after = new_predictor.predict(X_test)
            
            # Compare predictions
            np.testing.assert_array_almost_equal(pred_before, pred_after)
            assert new_predictor.method == 'persistence'
            assert new_predictor._is_fitted
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_and_load_svr(self):
        """Test save and load for SVR model."""
        np.random.seed(42)
        config = {'C': 1.0, 'epsilon': 0.1}
        predictor = ClassicalPredictor(config, method='svr')
        
        X_train = np.random.rand(30, 12, 5)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        # Make predictions before saving
        X_test = np.random.rand(5, 12, 5)
        pred_before = predictor.predict(X_test)
        
        # Save model
        temp_dir = tempfile.mkdtemp()
        try:
            save_path = Path(temp_dir) / "svr_model"
            predictor.save(str(save_path))
            
            # Load model
            new_predictor = ClassicalPredictor(config, method='svr')
            new_predictor.load(str(save_path))
            
            # Make predictions after loading
            pred_after = new_predictor.predict(X_test)
            
            # Compare predictions
            np.testing.assert_array_almost_equal(pred_before, pred_after)
            assert new_predictor.method == 'svr'
            assert new_predictor._is_fitted
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_consistent_interface_across_methods(self):
        """Test that all methods provide consistent interface."""
        np.random.seed(42)
        
        X_train = np.random.rand(30, 12, 5)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        X_test = np.random.rand(5, 12, 5)
        
        methods = ['persistence', 'svr']
        
        for method in methods:
            predictor = ClassicalPredictor({}, method=method)
            predictor.build_model()
            
            # Train
            result = predictor.train(X_train, y_train, X_val, y_val)
            assert 'train_loss' in result
            assert 'val_loss' in result
            assert 'epochs_completed' in result
            assert 'convergence_status' in result
            assert 'loss_history' in result
            
            # Predict
            predictions = predictor.predict(X_test)
            assert predictions.shape == (len(X_test),)
            assert np.all(np.isfinite(predictions))
            
            # Evaluate (use validation set for evaluation)
            val_predictions = predictor.predict(X_val)
            metrics = predictor.evaluate(y_val, val_predictions)
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'mape' in metrics
            assert 'r2' in metrics
    
    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        np.random.seed(42)
        config = {'C': 1.5, 'epsilon': 0.15}
        predictor = ClassicalPredictor(config, method='svr')
        
        X_train = np.random.rand(30, 12, 5)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        metadata = predictor.get_metadata()
        
        assert metadata['method'] == 'svr'
        assert 'train_samples' in metadata
        assert 'val_samples' in metadata
        assert 'train_loss' in metadata
        assert 'val_loss' in metadata
        assert 'training_time_seconds' in metadata
        assert metadata['train_samples'] == 30
        assert metadata['val_samples'] == 10
    
    def test_output_shape_consistency(self):
        """Test that output shapes are consistent across methods."""
        np.random.seed(42)
        
        n_samples = 20
        X = np.random.rand(n_samples, 12, 5)
        y = np.random.rand(n_samples)
        
        methods = ['persistence', 'svr']
        
        for method in methods:
            predictor = ClassicalPredictor({}, method=method)
            predictor.build_model()
            
            # Train with subset
            predictor.train(X[:15], y[:15], X[15:], y[15:])
            
            # Predict
            predictions = predictor.predict(X)
            
            assert predictions.shape == (n_samples,), f"Method {method} produced wrong shape"
            assert predictions.ndim == 1, f"Method {method} produced wrong dimensions"
    
    def test_value_range_validity(self):
        """Test that predictions are in valid ranges."""
        np.random.seed(42)
        
        # Create data in [0, 1] range (typical for normalized PV power)
        X_train = np.random.rand(30, 12, 5)
        y_train = np.random.rand(30)
        X_val = np.random.rand(10, 12, 5)
        y_val = np.random.rand(10)
        X_test = np.random.rand(5, 12, 5)
        
        methods = ['persistence', 'svr']
        
        for method in methods:
            predictor = ClassicalPredictor({}, method=method)
            predictor.build_model()
            predictor.train(X_train, y_train, X_val, y_val)
            
            predictions = predictor.predict(X_test)
            
            # Predictions should be finite
            assert np.all(np.isfinite(predictions)), f"Method {method} produced non-finite values"
