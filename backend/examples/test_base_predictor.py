"""Example demonstrating BasePredictor usage."""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import numpy as np
from src.models.base import BasePredictor


class SimpleLinearPredictor(BasePredictor):
    """
    Simple linear predictor for demonstration.
    
    This predictor uses a simple linear model that predicts
    the mean of the last time step across all features.
    """
    
    def build_model(self):
        """Build a simple linear model."""
        # Simple model: just store coefficients
        self.model = {
            'type': 'linear',
            'coefficients': np.ones(5)  # Assuming 5 features
        }
        print("Model built successfully")
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the linear model.
        
        For this simple example, we just compute mean coefficients.
        """
        print(f"Training on {len(X_train)} samples...")
        
        # Build model
        self.build_model()
        
        # Simple training: compute correlation between features and target
        # This is just for demonstration
        n_features = X_train.shape[2]
        self.model['coefficients'] = np.random.rand(n_features)
        
        # Generate predictions for metrics
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)
        
        # Calculate losses
        train_loss = np.mean((y_train - y_train_pred) ** 2)
        val_loss = np.mean((y_val - y_val_pred) ** 2)
        
        print(f"Training complete - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'epochs_completed': 1,
            'convergence_status': 'converged',
            'loss_history': [train_loss]
        }
    
    def predict(self, X):
        """
        Generate predictions.
        
        Simple prediction: weighted sum of last time step features.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get last time step
        last_step = X[:, -1, :]  # Shape: (n_samples, n_features)
        
        # Weighted sum
        predictions = np.dot(last_step, self.model['coefficients'])
        
        return predictions


def main():
    """Demonstrate BasePredictor usage."""
    print("=" * 60)
    print("BasePredictor Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_type': 'linear',
        'n_features': 5,
        'sequence_length': 12
    }
    
    # Create predictor
    print("\n1. Creating predictor...")
    predictor = SimpleLinearPredictor(config)
    print(f"   Predictor created: {predictor.__class__.__name__}")
    print(f"   Metadata: {predictor.get_metadata()}")
    
    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)
    n_train = 100
    n_val = 20
    n_test = 30
    seq_length = 12
    n_features = 5
    
    X_train = np.random.rand(n_train, seq_length, n_features)
    y_train = np.random.rand(n_train)
    X_val = np.random.rand(n_val, seq_length, n_features)
    y_val = np.random.rand(n_val)
    X_test = np.random.rand(n_test, seq_length, n_features)
    y_test = np.random.rand(n_test)
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train model
    print("\n3. Training model...")
    training_result = predictor.train(X_train, y_train, X_val, y_val)
    print(f"   Training result: {training_result}")
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = predictor.predict(X_test)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:5]}")
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics = predictor.evaluate(y_test, predictions)
    print(f"   Metrics:")
    print(f"     MAE:  {metrics['mae']:.4f}")
    print(f"     RMSE: {metrics['rmse']:.4f}")
    print(f"     MAPE: {metrics['mape']:.2f}%")
    print(f"     R²:   {metrics['r2']:.4f}")
    
    # Save model
    print("\n6. Saving model...")
    save_path = "models/example_linear_model"
    predictor.save(save_path)
    print(f"   Model saved to: {save_path}")
    
    # Load model
    print("\n7. Loading model...")
    new_predictor = SimpleLinearPredictor(config)
    new_predictor.load(save_path)
    print(f"   Model loaded successfully")
    print(f"   Loaded metadata: {new_predictor.get_metadata()}")
    
    # Verify loaded model works
    print("\n8. Verifying loaded model...")
    new_predictions = new_predictor.predict(X_test)
    print(f"   Predictions match: {np.allclose(predictions, new_predictions)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
