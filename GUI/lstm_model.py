import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import platform

class LSTMStockPredictor:
    """
    LSTM model for stock price prediction that addresses specific challenges of financial time series.
    """
    
    def __init__(self, time_steps: int = 60, layers: List[int] = [50, 50], 
                 dropout_rate: float = 0.2, learning_rate: float = 0.001, 
                 regularization: float = 0.01, batch_size: int = 32, epochs: int = 50):
        """
        Initialize the LSTM Stock Predictor.
        
        Args:
            time_steps: Number of previous days to use for prediction
            layers: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            regularization: L2 regularization factor
            batch_size: Training batch size
            epochs: Number of training epochs
        """
        self.time_steps = time_steps
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.epochs = epochs
        
        # These will be initialized later
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def _create_dataset(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values from a time series dataset.
        
        Args:
            dataset: Time series data with shape (n_samples, n_features)
            
        Returns:
            Tuple of X (input sequences) and y (target values)
        """
        X, y = [], []
        for i in range(len(dataset) - self.time_steps):
            X.append(dataset[i:(i + self.time_steps)])
            y.append(dataset[i + self.time_steps, 0])  # Predict only the closing price
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture with regularization to prevent overfitting.
        
        Args:
            input_shape: Shape of input data (time_steps, n_features)
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=self.layers[0],
            return_sequences=len(self.layers) > 1,
            input_shape=input_shape,
            recurrent_regularizer=l2(self.regularization),
            kernel_regularizer=l2(self.regularization),
            recurrent_dropout=self.dropout_rate
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Middle LSTM layers (if any)
        for i in range(1, len(self.layers) - 1):
            model.add(LSTM(
                units=self.layers[i],
                return_sequences=True,
                recurrent_regularizer=l2(self.regularization),
                kernel_regularizer=l2(self.regularization),
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Last LSTM layer (if more than one layer)
        if len(self.layers) > 1:
            model.add(LSTM(
                units=self.layers[-1],
                recurrent_regularizer=l2(self.regularization),
                kernel_regularizer=l2(self.regularization),
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Check if we're on macOS (for M1/M2 chips)
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            # Use legacy optimizer for M1/M2 Macs
            try:
                # First try the recommended legacy optimizer
                opt = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
                print("Using legacy Adam optimizer for M1/M2 Mac")
            except AttributeError:
                # Fall back to regular optimizer if legacy is not available
                opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                print("Legacy optimizer not available, using standard Adam")
        else:
            # Use standard optimizer for non-Mac systems
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(optimizer=opt, loss='mean_squared_error')
        
        return model
    
    def preprocess_data(self, data: pd.DataFrame, feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        Preprocess data for LSTM training and testing.
        
        Args:
            data: DataFrame containing stock data
            feature_columns: List of column names to use as features (default: only 'Close')
            
        Returns:
            Dictionary containing preprocessed data components
        """
        # Default to using only 'Close' price if no features specified
        if feature_columns is None:
            feature_columns = ['Close']
        
        # Extract features and ensure no NaN values
        df = data[feature_columns].copy()
        df = df.dropna()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df)
        
        # Create sequences
        X, y = self._create_dataset(scaled_data)
        
        # Split into train and test sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape y arrays for scalar output
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Return as dictionary
        return {
            'scaled_data': scaled_data,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_size': train_size,
            'feature_columns': feature_columns
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
              y_test: np.ndarray, callbacks: List = None) -> None:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_test: Testing input sequences for validation
            y_test: Testing target values for validation
            callbacks: List of Keras callbacks (optional)
        """
        # Build the model if it doesn't exist
        if self.model is None:
            self.model = self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Default callbacks for training
        if callbacks is None:
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
            ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input sequences
            
        Returns:
            Array of predicted values (scaled)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform scaled predictions back to original scale.
        
        Args:
            data: Scaled predictions or values
            
        Returns:
            Data in original scale
        """
        # Ensure data is 2D
        if len(data.shape) > 2:
            # If data has more than 2 dimensions, flatten it to 2D
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data.copy()
            
        # If data is a single value, reshape for proper inverse transform
        if len(data_2d.shape) == 1:
            data_2d = data_2d.reshape(-1, 1)
        
        # Check if scaler has been fitted
        if not hasattr(self.scaler, 'scale_'):
            # If not fitted, fit it to the data (this is a fallback and shouldn't be common)
            dummy_data = np.zeros((data_2d.shape[0], 1))
            self.scaler.fit(dummy_data)
            # Warning: this is just to avoid errors, but proper scaling requires the scaler to be fit to training data
            print("Warning: Scaler was not fitted. Results may be inaccurate.")
        
        # Create a matrix with the same shape as the original input to the scaler
        dummy = np.zeros((data_2d.shape[0], len(self.scaler.scale_)))
        dummy[:, 0] = data_2d[:, 0]  # Put predictions in first column
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy)[:, 0:1]
        
        # Return with original dimensions, if needed
        if len(data.shape) > 2:
            return inverse_transformed.reshape(data.shape[0], 1)
        return inverse_transformed
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate directional accuracy (prediction of up/down movement)
        y_true_diff = np.diff(y_true.flatten())
        y_pred_diff = np.diff(y_pred.flatten())
        dir_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': dir_accuracy
        }
    
    def plot_training_history(self) -> plt.Figure:
        """
        Plot training and validation loss.
        
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_predictions(self, dates, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            dates: Date indices for the x-axis
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Ensure arrays are flattened for plotting
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        ax.plot(dates, y_true_flat, label='Actual Prices', color='blue', linewidth=2)
        ax.plot(dates, y_pred_flat, label='Predicted Prices', color='red', linewidth=2)
        
        # Calculate errors and prediction bands
        error = np.std(y_true_flat - y_pred_flat)
        ax.fill_between(dates, y_pred_flat - 2*error, y_pred_flat + 2*error, 
                        color='red', alpha=0.1, label='95% Confidence Interval')
        
        ax.set_title('LSTM Predictions vs Actual Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def predict_next_day(self, latest_data: np.ndarray) -> Dict[str, Any]:
        """
        Predict the next day's price.
        
        Args:
            latest_data: The most recent data (scaled), with shape matching time_steps
            
        Returns:
            Dictionary with the prediction and confidence interval
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert pandas Series to numpy if needed
        if hasattr(latest_data, 'values'):
            latest_data = latest_data.values

        # Reshape data for LSTM input - ensure correct dimensions
        if len(latest_data.shape) == 2:
            # If already 2D, just add batch dimension
            input_data = latest_data.reshape(1, latest_data.shape[0], latest_data.shape[1])
        else:
            # If not 2D, reshape accordingly
            input_data = latest_data.reshape(1, self.time_steps, -1)
        
        # Make prediction
        next_day_scaled = self.model.predict(input_data, verbose=0)
        
        # Check if scaler has been fitted
        if not hasattr(self.scaler, 'scale_'):
            # If not fitted, fit it to the data (this is a fallback and shouldn't be common)
            dummy_data = np.zeros((1, 1))
            self.scaler.fit(dummy_data)
            # Warning: this is just to avoid errors, but proper scaling requires the scaler to be fit to training data
            print("Warning: Scaler was not fitted. Results may be inaccurate.")
        
        # Convert back to original scale - ensure we handle the dimensions properly
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, 0] = next_day_scaled[0, 0] if len(next_day_scaled.shape) > 1 else next_day_scaled[0]
        next_day_price = float(self.scaler.inverse_transform(dummy)[0, 0])
        
        # Calculate confidence interval (based on training error)
        # For a proper implementation, you would use the prediction uncertainty
        # Here we use a simplified approach
        confidence = 0.05 * next_day_price  # 5% uncertainty 
        
        return {
            'prediction': next_day_price,
            'lower_bound': next_day_price - confidence,
            'upper_bound': next_day_price + confidence
        }
    
    def rolling_window_validation(self, data: np.ndarray, window_size: int = 30) -> Dict[str, List[float]]:
        """
        Perform rolling window validation to check model stability.
        
        Args:
            data: Full scaled dataset
            window_size: Size of rolling windows
            
        Returns:
            Dictionary of rolling evaluation metrics
        """
        total_samples = len(data) - self.time_steps - window_size
        
        # Initialize arrays to store metrics
        maes, rmses, dir_accuracies = [], [], []
        
        # Use a smaller number of windows to avoid excessive computation
        step_size = max(1, total_samples // 10)
        
        # Check if scaler has been fitted
        if not hasattr(self.scaler, 'scale_'):
            # If not fitted, fit it to the data
            self.scaler.fit(data)
            print("Fitted scaler to data in rolling_window_validation")
        
        for i in range(0, total_samples, step_size):
            try:
                # Split data for this window
                window_data = data[i:i+self.time_steps+window_size]
                X_window, y_window = self._create_dataset(window_data)
                
                # Reshape X if needed
                if len(X_window.shape) == 2:
                    X_window = X_window.reshape(X_window.shape[0], X_window.shape[1], 1)
                
                # Reshape y to match expected shape
                y_window = y_window.reshape(-1, 1)
                
                # Generate predictions with reduced verbosity
                y_pred_scaled = self.model.predict(X_window, verbose=0)
                
                # Create dummy array for inverse transform
                dummy_true = np.zeros((y_window.shape[0], len(self.scaler.scale_)))
                dummy_true[:, 0] = y_window.flatten()
                y_true = self.scaler.inverse_transform(dummy_true)[:, 0:1]
                
                dummy_pred = np.zeros((y_pred_scaled.shape[0], len(self.scaler.scale_)))
                dummy_pred[:, 0] = y_pred_scaled.flatten()
                y_pred = self.scaler.inverse_transform(dummy_pred)[:, 0:1]
                
                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                # Calculate directional accuracy
                y_true_diff = np.diff(y_true.flatten())
                y_pred_diff = np.diff(y_pred.flatten())
                
                # Avoid division by zero or empty arrays
                if len(y_true_diff) > 0:
                    dir_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
                else:
                    dir_accuracy = 0
                
                # Store metrics
                maes.append(mae)
                rmses.append(rmse)
                dir_accuracies.append(dir_accuracy)
                
            except Exception as e:
                # Skip this window if there's an error
                print(f"Error in window {i}: {str(e)}")
                continue
        
        return {
            'maes': maes,
            'rmses': rmses,
            'dir_accuracies': dir_accuracies
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str, scaler: MinMaxScaler, 
                  time_steps: int, **kwargs) -> 'LSTMStockPredictor':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            scaler: Fitted MinMaxScaler for data transformation
            time_steps: Number of time steps used in the model
            kwargs: Additional parameters for the predictor
            
        Returns:
            LSTMStockPredictor with loaded model
        """
        # Create a new predictor instance
        predictor = cls(time_steps=time_steps, **kwargs)
        
        # Load the model
        predictor.model = tf.keras.models.load_model(filepath)
        
        # Set the scaler
        predictor.scaler = scaler
        
        return predictor 