# Stock Price Prediction Models

A comprehensive collection of machine learning models for stock price prediction, including LSTM, Hybrid LSTM+XGBoost, and CNN approaches.

## Available Models

### 1. LSTM Model (`lstm_app.py`)
- Pure Long Short-Term Memory neural network model
- Specialized for time-series prediction 
- Captures temporal dependencies in stock price data
- Features comprehensive parameter tuning and optimization

### 2. Hybrid LSTM+XGBoost Model (`hybrid_stock_app.py`) Which is our final model for this project 
- Combines the power of LSTM and XGBoost
- LSTM component captures temporal patterns
- XGBoost component evaluates technical indicators
- Features weighting mechanism between LSTM and XGBoost predictions

### 3. CNN Model (`CNN_model.py`)
- Convolutional Neural Network approach
- Treats time-series data as patterns for convolutional analysis
- Useful for detecting local and global patterns in price movements

## Features

- **Interactive User Interfaces**: All models use Streamlit for intuitive interaction
- **Technical Analysis**: Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Parameter Customization**: Adjust all model hyperparameters through the UI
- **Data Transformations**: Options for raw prices, logarithmic transformation, or percent changes
- **Performance Metrics**: MAE, RMSE, R² Score, and Directional Accuracy
- **Visualization**: Interactive charts for predictions and model performance
- **Next-Day Predictions**: All models provide forecasts for future prices with confidence metrics

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-prediction-models.git
   cd stock-prediction-models
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Applications

### Go to GUI folder

### LSTM Model
```
streamlit run lstm_app.py
```

The LSTM application features:
- Parameter tuning with multiple configurations
- Market cycle visualization for long-term analysis 
- Moving average options (20, 50, 100, 200-day)
- Rolling window validation to test model stability

### Hybrid LSTM+XGBoost Model (Final model) 
```
streamlit run hybrid_stock_app.py
```

The Hybrid application features:
- Separate controls for LSTM and XGBoost parameters
- Technical indicator visualization
- Feature importance analysis from XGBoost
- Adjustable weighting between LSTM and XGBoost predictions

### CNN Model
```
streamlit run CNN_model.py
```

The CNN application features:
- Convolutional layer parameter adjustments
- Specialized time-series visualization
- Filter visualization options

## Model Architectures

### LSTM Architecture
- Multiple stacked LSTM layers with configurable sizes
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- L2 regularization for better generalization

### Hybrid LSTM+XGBoost Architecture
- LSTM neural network component for sequence learning
- XGBoost gradient boosting for technical indicator analysis
- Weighted combination of predictions from both models
- Feature importance analysis for technical indicators

### CNN Architecture
- 1D convolutional layers for time-series pattern detection
- Max pooling and flattening layers
- Dense layers for final prediction

## Recommended Usage

1. **For Beginners**: Start with the LSTM model to understand basic time-series prediction
2. **For Technical Analysis**: The Hybrid model provides the best integration of technical indicators
3. **For Pattern Detection**: The CNN model can detect unique patterns in price movements

## Common Parameters Explained

- **Time Steps**: Number of previous days used to make a prediction (typically 30-120)
- **LSTM Units**: Size of LSTM memory cell layers (higher = more complex patterns)
- **Dropout Rate**: Controls overfitting by randomly deactivating neurons (0.0-0.5)
- **Epochs**: Training iterations over the entire dataset
- **Batch Size**: Number of samples processed before model update

## Performance Optimization

Each application provides ways to optimize model performance:
- Parameter tuning to find optimal configurations
- Multiple data transformation options
- Feature selection
- Regularization techniques
- Early stopping and learning rate reduction

## Evaluation Metrics

All models' performance is evaluated using:
- **Mean Absolute Error (MAE)**: Average dollar error in predictions
- **Root Mean Squared Error (RMSE)**: Emphasizes larger errors
- **R² Score**: How well the model explains price variations
- **Directional Accuracy**: How often the model correctly predicts price movement direction


