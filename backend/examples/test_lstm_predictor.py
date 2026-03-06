"""Example script demonstrating LSTM predictor usage."""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.models.lstm import LSTMPredictor


def main():
    """Demonstrate LSTM predictor training and prediction."""
    print("=" * 60)
    print("LSTM Predictor Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic time-series data
    print("\n1. Generating synthetic data...")
    n_train = 500
    n_val = 100
    n_test = 50
    sequence_length = 12
    n_features = 5
    
    # Create synthetic sequences with some pattern
    def generate_data(n_samples):
        X = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
        # Create target as weighted sum of features with some noise
        y = (X[:, -1, :].sum(axis=1) * 0.5 + np.random.randn(n_samples) * 0.1).astype(np.float32)
        return X, y
    
    X_train, y_train = generate_data(n_train)
    X_val, y_val = generate_data(n_val)
    X_test, y_test = generate_data(n_test)
    
    print(f"   Training samples: {n_train}")
    print(f"   Validation samples: {n_val}")
    print(f"   Test samples: {n_test}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of features: {n_features}")
    
    # Configure LSTM model
    print("\n2. Configuring LSTM model...")
    config = {
        'input_size': n_features,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 30,
        'early_stopping_patience': 10
    }
    
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Number of layers: {config['num_layers']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Max epochs: {config['epochs']}")
    
    # Create and train predictor
    print("\n3. Training LSTM model...")
    predictor = LSTMPredictor(config)
    
    training_result = predictor.train(X_train, y_train, X_val, y_val)
    
    print(f"\n   Training completed!")
    print(f"   Epochs completed: {training_result['epochs_completed']}")
    print(f"   Convergence status: {training_result['convergence_status']}")
    print(f"   Final training loss: {training_result['train_loss']:.6f}")
    print(f"   Final validation loss: {training_result['val_loss']:.6f}")
    
    # Show loss history
    print("\n   Loss history (first 5 and last 5 epochs):")
    train_history = training_result['train_loss_history']
    val_history = training_result['val_loss_history']
    
    for i in range(min(5, len(train_history))):
        print(f"      Epoch {i+1:2d}: Train={train_history[i]:.6f}, Val={val_history[i]:.6f}")
    
    if len(train_history) > 10:
        print("      ...")
        for i in range(max(5, len(train_history) - 5), len(train_history)):
            print(f"      Epoch {i+1:2d}: Train={train_history[i]:.6f}, Val={val_history[i]:.6f}")
    
    # Make predictions on test set
    print("\n4. Making predictions on test set...")
    predictions = predictor.predict(X_test)
    
    print(f"   Generated {len(predictions)} predictions")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   True value range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    
    # Evaluate performance
    print("\n5. Evaluating model performance...")
    metrics = predictor.evaluate(y_test, predictions)
    
    print(f"   MAE (Mean Absolute Error): {metrics['mae']:.6f}")
    print(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}")
    print(f"   MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}")
    
    # Show sample predictions
    print("\n6. Sample predictions (first 10):")
    print("   " + "-" * 40)
    print("   {:>10s} {:>10s} {:>10s}".format("True", "Predicted", "Error"))
    print("   " + "-" * 40)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - predictions[i])
        print(f"   {y_test[i]:10.4f} {predictions[i]:10.4f} {error:10.4f}")
    print("   " + "-" * 40)
    
    # Test save and load
    print("\n7. Testing model persistence...")
    save_path = backend_path / "models" / "test_lstm_model"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    predictor.save(str(save_path))
    print(f"   Model saved to: {save_path}")
    
    # Load model
    predictor_loaded = LSTMPredictor(config)
    predictor_loaded.load(str(save_path))
    print(f"   Model loaded successfully")
    
    # Verify loaded model produces same predictions
    predictions_loaded = predictor_loaded.predict(X_test[:5])
    predictions_original = predictor.predict(X_test[:5])
    
    if np.allclose(predictions_loaded, predictions_original, rtol=1e-5):
        print(f"   ✓ Loaded model predictions match original")
    else:
        print(f"   ✗ Warning: Loaded model predictions differ")
    
    # Show metadata
    print("\n8. Model metadata:")
    metadata = predictor.get_metadata()
    print(f"   Framework: {metadata['framework']}")
    print(f"   Created at: {metadata['created_at']}")
    print(f"   Training time: {metadata['training_time_seconds']:.2f} seconds")
    print(f"   Convergence status: {metadata['convergence_status']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
