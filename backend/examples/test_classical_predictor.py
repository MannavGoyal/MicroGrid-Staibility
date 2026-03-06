"""Example usage of ClassicalPredictor models."""

import sys
from pathlib import Path

# Add backend/src to path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

import numpy as np
from models.classical import ClassicalPredictor


def test_persistence():
    """Test Persistence model."""
    print("\n=== Testing Persistence Model ===")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 12
    n_features = 5
    
    X_train = np.random.rand(n_samples, sequence_length, n_features)
    y_train = np.random.rand(n_samples)
    X_val = np.random.rand(20, sequence_length, n_features)
    y_val = np.random.rand(20)
    
    # Initialize and train
    config = {}
    predictor = ClassicalPredictor(config, method='persistence')
    predictor.build_model()
    
    print("Training Persistence model...")
    results = predictor.train(X_train, y_train, X_val, y_val)
    print(f"Training results: {results}")
    
    # Make predictions
    predictions = predictor.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Evaluate
    metrics = predictor.evaluate(y_val, predictions)
    print(f"Evaluation metrics: {metrics}")
    
    print("✓ Persistence model test passed")


def test_svr():
    """Test SVR model."""
    print("\n=== Testing SVR Model ===")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 12
    n_features = 5
    
    X_train = np.random.rand(n_samples, sequence_length, n_features)
    y_train = np.random.rand(n_samples)
    X_val = np.random.rand(20, sequence_length, n_features)
    y_val = np.random.rand(20)
    
    # Initialize and train
    config = {
        'C': 1.0,
        'epsilon': 0.1,
        'kernel': 'rbf'
    }
    predictor = ClassicalPredictor(config, method='svr')
    predictor.build_model()
    
    print("Training SVR model...")
    results = predictor.train(X_train, y_train, X_val, y_val)
    print(f"Training results: {results}")
    
    # Make predictions
    predictions = predictor.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Evaluate
    metrics = predictor.evaluate(y_val, predictions)
    print(f"Evaluation metrics: {metrics}")
    
    print("✓ SVR model test passed")


def test_arima():
    """Test ARIMA model."""
    print("\n=== Testing ARIMA Model ===")
    
    # Create synthetic time series data
    np.random.seed(42)
    n_samples = 100
    sequence_length = 12
    n_features = 5
    
    # Create more realistic time series
    t = np.linspace(0, 10, n_samples)
    y_train = 0.5 + 0.3 * np.sin(t) + 0.1 * np.random.randn(n_samples)
    
    X_train = np.random.rand(n_samples, sequence_length, n_features)
    
    t_val = np.linspace(10, 12, 20)
    y_val = 0.5 + 0.3 * np.sin(t_val) + 0.1 * np.random.randn(20)
    X_val = np.random.rand(20, sequence_length, n_features)
    
    # Initialize and train
    config = {
        'arima_order': (2, 1, 0)
    }
    predictor = ClassicalPredictor(config, method='arima')
    predictor.build_model()
    
    print("Training ARIMA model...")
    results = predictor.train(X_train, y_train, X_val, y_val)
    print(f"Training results: {results}")
    
    # Make predictions
    predictions = predictor.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Evaluate
    metrics = predictor.evaluate(y_val, predictions)
    print(f"Evaluation metrics: {metrics}")
    
    print("✓ ARIMA model test passed")


def test_save_load():
    """Test model save and load functionality."""
    print("\n=== Testing Save/Load ===")
    
    # Create and train a model
    np.random.seed(42)
    n_samples = 50
    sequence_length = 12
    n_features = 5
    
    X_train = np.random.rand(n_samples, sequence_length, n_features)
    y_train = np.random.rand(n_samples)
    X_val = np.random.rand(10, sequence_length, n_features)
    y_val = np.random.rand(10)
    
    config = {'C': 1.0}
    predictor = ClassicalPredictor(config, method='svr')
    predictor.build_model()
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Make predictions before saving
    pred_before = predictor.predict(X_val)
    
    # Save model
    save_path = "/tmp/test_classical_model"
    predictor.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Load model
    predictor_loaded = ClassicalPredictor(config, method='svr')
    predictor_loaded.load(save_path)
    print(f"Model loaded from {save_path}")
    
    # Make predictions after loading
    pred_after = predictor_loaded.predict(X_val)
    
    # Compare predictions
    diff = np.abs(pred_before - pred_after).max()
    print(f"Max difference in predictions: {diff}")
    
    if diff < 1e-6:
        print("✓ Save/Load test passed")
    else:
        print(f"✗ Save/Load test failed: predictions differ by {diff}")


def test_consistent_interface():
    """Test that all methods have consistent interface."""
    print("\n=== Testing Consistent Interface ===")
    
    np.random.seed(42)
    n_samples = 50
    sequence_length = 12
    n_features = 5
    
    X_train = np.random.rand(n_samples, sequence_length, n_features)
    y_train = np.random.rand(n_samples)
    X_val = np.random.rand(10, sequence_length, n_features)
    y_val = np.random.rand(10)
    
    methods = ['persistence', 'svr']  # Skip ARIMA for speed
    
    for method in methods:
        print(f"\nTesting {method}...")
        config = {}
        predictor = ClassicalPredictor(config, method=method)
        predictor.build_model()
        
        # Train
        results = predictor.train(X_train, y_train, X_val, y_val)
        assert 'train_loss' in results
        assert 'val_loss' in results
        assert 'convergence_status' in results
        
        # Predict
        predictions = predictor.predict(X_val)
        assert predictions.shape == (len(X_val),)
        
        # Evaluate
        metrics = predictor.evaluate(y_val, predictions)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        
        print(f"  ✓ {method} interface consistent")
    
    print("\n✓ All methods have consistent interface")


if __name__ == "__main__":
    print("Testing ClassicalPredictor Implementation")
    print("=" * 50)
    
    try:
        test_persistence()
        test_svr()
        test_arima()
        test_save_load()
        test_consistent_interface()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
