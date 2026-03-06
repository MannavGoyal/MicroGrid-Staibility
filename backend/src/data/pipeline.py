"""Data Pipeline module for preprocessing and feature engineering."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, NamedTuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataclasses import dataclass

from ..config.schemas import Configuration, ForecastHorizon


class DataSplit(NamedTuple):
    """Container for train/test data split."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class Scaler:
    """Wrapper for sklearn scalers with metadata."""
    scaler_type: str  # 'minmax' or 'standard'
    scaler: object  # sklearn scaler instance
    feature_names: list
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        return self.scaler.inverse_transform(X)


class DataPipeline:
    """Data preprocessing and feature engineering pipeline."""
    
    def __init__(self, config: Configuration):
        """
        Initialize DataPipeline.
        
        Args:
            config: Configuration object with pipeline parameters
        """
        self.config = config
        self.scaler: Optional[Scaler] = None
        self.feature_names: list = []
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess raw data.
        
        Handles missing values using forward-fill and interpolation methods.
        
        Args:
            df: Raw DataFrame with time-series data
            
        Returns:
            Preprocessed DataFrame
        """
        df_clean = df.copy()
        
        # Identify numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Handle missing values
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                # First try forward-fill (use last valid observation)
                df_clean[col] = df_clean[col].ffill()
                
                # Then backward-fill for any remaining NaNs at the start
                df_clean[col] = df_clean[col].bfill()
                
                # Finally, use linear interpolation for any remaining gaps
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
                
                # If still NaN (e.g., entire column is NaN), fill with 0
                df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
    
    def normalize(
        self, 
        df: pd.DataFrame, 
        method: str = 'minmax'
    ) -> Tuple[np.ndarray, Scaler]:
        """
        Normalize features to comparable scales.
        
        Args:
            df: DataFrame with features to normalize
            method: Normalization method ('minmax' or 'standard')
            
        Returns:
            Tuple of (normalized_array, fitted_scaler)
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols
        
        # Extract numeric data
        data = df[numeric_cols].values
        
        # Create and fit scaler
        if method == 'minmax':
            scaler_obj = MinMaxScaler(feature_range=(0, 1))
        elif method == 'standard':
            scaler_obj = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Fit and transform
        normalized_data = scaler_obj.fit_transform(data)
        
        # Create scaler wrapper
        scaler = Scaler(
            scaler_type=method,
            scaler=scaler_obj,
            feature_names=numeric_cols
        )
        
        self.scaler = scaler
        
        return normalized_data, scaler
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features and temporal encodings.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Get sequence length from config
        sequence_length = self.config.model_configuration.sequence_length
        
        # Create lagged features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Create lags spanning the sequence length
            for lag in range(1, min(sequence_length, 6)):  # Limit to 5 lags to avoid too many features
                df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Add temporal encodings if timestamp column exists
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if timestamp_cols:
            # Use first timestamp column
            ts_col = timestamp_cols[0]
            
            # Try to parse as datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_features[ts_col]):
                try:
                    df_features[ts_col] = pd.to_datetime(df_features[ts_col])
                except:
                    pass  # Skip temporal encoding if parsing fails
            
            # Add temporal features if we have datetime
            if pd.api.types.is_datetime64_any_dtype(df_features[ts_col]):
                df_features['hour'] = df_features[ts_col].dt.hour
                df_features['day_of_week'] = df_features[ts_col].dt.dayofweek
                df_features['month'] = df_features[ts_col].dt.month
                
                # Cyclical encoding for hour (24-hour cycle)
                df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
                df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
                
                # Cyclical encoding for day of week (7-day cycle)
                df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
                df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
                
                # Cyclical encoding for month (12-month cycle)
                df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
                df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Drop rows with NaN created by lagging (at the beginning)
        df_features = df_features.dropna()
        
        return df_features
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        sequence_length: int,
        target_column_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for time-series models.
        
        Args:
            data: Normalized feature matrix (samples, features)
            sequence_length: Length of input sequences
            target_column_index: Index of target column in data
            
        Returns:
            Tuple of (X, y) where:
                X: Input sequences (samples, sequence_length, num_features)
                y: Target values (samples,)
        """
        X, y = [], []
        
        num_samples = len(data)
        
        # Create sequences
        for i in range(sequence_length, num_samples):
            # Input sequence: previous sequence_length time steps
            X.append(data[i - sequence_length:i, :])
            
            # Target: next value of target column
            y.append(data[i, target_column_index])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        train_ratio: float = 0.8
    ) -> DataSplit:
        """
        Split data into train/test sets maintaining temporal order.
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of data to use for training (default: 0.8)
            
        Returns:
            DataSplit with train and test sets
        """
        # Calculate split point
        split_idx = int(len(X) * train_ratio)
        
        # Split maintaining temporal order (no shuffling)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
    
    def get_sequence_length_for_horizon(self, horizon: ForecastHorizon) -> int:
        """
        Get appropriate sequence length based on forecast horizon.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Recommended sequence length
        """
        # Adjust sequence length based on forecast horizon
        # Longer horizons need more historical context
        if horizon == ForecastHorizon.FIVE_MIN:
            return max(12, self.config.model_configuration.sequence_length)
        elif horizon == ForecastHorizon.FIFTEEN_MIN:
            return max(24, self.config.model_configuration.sequence_length)
        elif horizon == ForecastHorizon.ONE_HOUR:
            return max(48, self.config.model_configuration.sequence_length)
        else:
            return self.config.model_configuration.sequence_length
