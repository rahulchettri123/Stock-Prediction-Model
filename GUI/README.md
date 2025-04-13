# LSTM Stock Price Prediction

A sophisticated stock price prediction application that uses Long Short-Term Memory (LSTM) neural networks to forecast future stock prices based on historical data.

## Features

- **Advanced LSTM Model**: Multi-layer LSTM architecture with regularization, dropout, and batch normalization
- **Interactive User Interface**: Built with Streamlit for easy parameter selection and visualization
- **Comprehensive Model Optimization**: Automatic parameter tuning for time windows, layer sizes, dropout rates, and more
- **Multiple Data Transformations**: Options for raw prices, logarithmic transformation, or percent changes
- **Technical Indicators**: Moving averages (20, 50, 100, 200-day) for enhanced predictions
- **Market Cycle Visualization**: Identification of bull and bear markets for long-term analysis
- **Detailed Performance Metrics**: MAE, RMSE, MAPE, R² Score, and Directional Accuracy
- **Rolling Window Validation**: Tests model stability across different time periods
- **Next-Day Predictions**: Forecasts future prices with confidence intervals

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lstm-stock-prediction.git
   cd lstm-stock-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```
streamlit run lstm_app.py
```

This will start the application and open it in your default web browser. From there, you can:

1. Enter a stock ticker symbol
2. Select date range for historical data
3. Choose optimization metrics and tuning intensity
4. Select features and parameters to tune
5. Start model optimization
6. View results, predictions, and performance metrics

## Project Structure

### Core Files

#### `lstm_app.py`
The main application file that contains the Streamlit UI and orchestrates the workflow. It handles:
- User interface and parameter selection
- Data loading and preprocessing
- Parameter tuning loop
- Results visualization
- Next-day price predictions

#### `lstm_model.py`
The model implementation file that contains the `LSTMStockPredictor` class. It handles:
- LSTM architecture definition
- Data preprocessing
- Model training and prediction
- Performance evaluation
- Visualization functions

## Model Architecture

The LSTM model consists of:
- Multiple LSTM layers with configurable sizes
- Batch normalization layers for stable training
- Dropout layers to prevent overfitting
- L2 regularization for better generalization
- Dense output layer for price prediction

The model addresses common challenges in financial time series:
- High volatility and non-stationarity
- Noise and random market fluctuations
- Influence of global events and sentiment
- Balancing short and long-term dependencies

## Performance Optimization

The application provides several ways to optimize model performance:
- Parameter tuning to find optimal configurations
- Multiple data transformation options
- Feature selection
- Regularization techniques
- Early stopping and learning rate reduction

## Evaluation Metrics

The model's performance is evaluated using:
- **Mean Absolute Error (MAE)**: Average dollar error in predictions
- **Root Mean Squared Error (RMSE)**: Emphasizes larger errors
- **Mean Absolute Percentage Error (MAPE)**: Error as a percentage of actual price
- **R² Score**: How well the model explains price variations
- **Directional Accuracy**: How often the model correctly predicts price movement direction

## License

[Specify your license here]

## Acknowledgments

- [List any libraries, data sources, or inspirations] 