import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Hybrid LSTM+XGBoost Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0ea5e9;
    }
    .metric-label {
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Hybrid LSTM+XGBoost Stock Price Prediction")
st.markdown("""
This app combines the power of LSTM neural networks (for sequence learning) with XGBoost 
(for feature importance) to predict future stock prices with higher accuracy.
""")

# Feature engineering function
def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid warnings
    df = df.copy()
    
    # 1. Price Changes
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    
    # 2. Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 3. Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 4. MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.abs()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # 7. Volume Features - Fix for yfinance multi-column issue
    # Ensure Volume is handled as a Series 
    if isinstance(df['Volume'], pd.DataFrame):
        volume_series = df['Volume'].iloc[:, 0]
    else:
        volume_series = df['Volume']
    
    df['Volume_Change'] = volume_series.pct_change() * 100
    df['Volume_MA_5'] = volume_series.rolling(window=5).mean()
    # Fix the Volume_Spike calculation
    df['Volume_Spike'] = volume_series.div(df['Volume_MA_5'])
    
    # Drop missing values
    df = df.dropna()
    
    return df

# Hybrid model class
class HybridLSTM_XGBoost:
    def __init__(self, lstm_units=50, lstm_dropout=0.2, time_steps=60, xgb_max_depth=3, xgb_eta=0.1, lstm_weight=0.7):
        self.time_steps = time_steps
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.xgb_max_depth = xgb_max_depth
        self.xgb_eta = xgb_eta
        self.lstm_weight = lstm_weight
        self.xgb_weight = 1 - lstm_weight
        
        # Scalers
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Models
        self.lstm_model = None
        self.xgb_model = None
    
    def prepare_data(self, data):
        """Prepare data for both LSTM and XGBoost models"""
        # Scale price data for LSTM
        price_data = data[['Close']].values
        scaled_price = self.price_scaler.fit_transform(price_data)
        
        # Select features for XGBoost
        xgb_features = ['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'RSI', 
                         'BB_Width', 'Price_Change_Pct', 'Volume_Spike']
        
        feature_data = data[xgb_features].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Create sequences for LSTM
        X_lstm, y = [], []
        for i in range(len(scaled_price) - self.time_steps):
            X_lstm.append(scaled_price[i:i+self.time_steps, 0])
            y.append(scaled_price[i+self.time_steps, 0])
        
        X_lstm = np.array(X_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        y = np.array(y)
        
        # Create feature data for XGBoost (aligned with LSTM targets)
        X_xgb = scaled_features[self.time_steps:]
        if len(X_xgb) > len(y):
            X_xgb = X_xgb[:len(y)]
        
        # Split into train and test sets (80% train, 20% test)
        train_size = int(len(X_lstm) * 0.8)
        
        X_lstm_train, X_lstm_test = X_lstm[:train_size], X_lstm[train_size:]
        X_xgb_train, X_xgb_test = X_xgb[:train_size], X_xgb[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return {
            'X_lstm_train': X_lstm_train,
            'X_lstm_test': X_lstm_test,
            'X_xgb_train': X_xgb_train,
            'X_xgb_test': X_xgb_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_size': train_size,
            'xgb_features': xgb_features
        }
    
    def build_lstm_model(self):
        """Build the LSTM model"""
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, 
                      input_shape=(self.time_steps, 1)))
        model.add(Dropout(self.lstm_dropout))
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.lstm_dropout))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_xgb_model(self):
        """Build the XGBoost model"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_eta,
            n_estimators=100
        )
        return model
    
    def train(self, data, epochs=50, batch_size=32, verbose=1):
        """Train both models and return training history"""
        # Build models
        self.lstm_model = self.build_lstm_model()
        self.xgb_model = self.build_xgb_model()
        
        # Train LSTM
        lstm_history = self.lstm_model.fit(
            data['X_lstm_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_lstm_test'], data['y_test']),
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        # Train XGBoost - Handle different XGBoost versions
        # Convert data to DMatrix format which works more consistently
        import xgboost as xgb
        dtrain = xgb.DMatrix(data['X_xgb_train'], label=data['y_train'])
        dval = xgb.DMatrix(data['X_xgb_test'], label=data['y_test'])
        
        # Set parameters - these work with all XGBoost versions
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.xgb_max_depth,
            'eta': self.xgb_eta,
        }
        
        # Train using the lower-level API which is more consistent
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        num_round = 100
        
        try:
            xgb_model = xgb.train(
                params, 
                dtrain, 
                num_round, 
                watchlist, 
                early_stopping_rounds=10,
                verbose_eval=verbose
            )
            
            # Store the trained model
            self.xgb_model = xgb_model
        except:
            # If that approach fails, fall back to the sklearn API without early_stopping
            try:
                self.xgb_model.fit(
                    data['X_xgb_train'], data['y_train'],
                    eval_set=[(data['X_xgb_test'], data['y_test'])],
                    verbose=(verbose > 0)
                )
            except:
                # Last resort, most basic fit
                self.xgb_model.fit(
                    data['X_xgb_train'], data['y_train']
                )
        
        return lstm_history
    
    def predict(self, X_lstm, X_xgb):
        """Make hybrid predictions by combining LSTM and XGBoost"""
        # Make individual predictions
        lstm_pred = self.lstm_model.predict(X_lstm)
        
        # Handle prediction based on which XGBoost API we used
        # First, check if this is a scikit-learn style XGBoost model vs core API model
        use_scikit_api = hasattr(self.xgb_model, 'get_booster')
        
        # XGBoost prediction with error handling
        xgb_pred = None
        try:
            if use_scikit_api:
                # It's a scikit-learn wrapper
                xgb_pred = self.xgb_model.predict(X_xgb).reshape(-1, 1)
            else:
                # It's the core XGBoost model from train()
                import xgboost as xgb
                # Ensure data is numpy array
                dtest = xgb.DMatrix(np.asarray(X_xgb))
                xgb_pred = self.xgb_model.predict(dtest).reshape(-1, 1)
        except Exception as e:
            # If prediction fails, use LSTM prediction only
            print(f"XGBoost prediction failed: {str(e)}. Using LSTM prediction only.")
            xgb_pred = lstm_pred
            self.lstm_weight = 1.0
            self.xgb_weight = 0.0
        
        # Combine predictions using weights
        hybrid_pred = self.lstm_weight * lstm_pred + self.xgb_weight * xgb_pred
        
        return {
            'lstm': lstm_pred,
            'xgb': xgb_pred,
            'hybrid': hybrid_pred
        }
    
    def evaluate(self, y_true, predictions):
        """Evaluate model performance"""
        results = {}
        
        for model_name, y_pred in predictions.items():
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy
            y_true_diff = np.diff(y_true.flatten())
            y_pred_diff = np.diff(y_pred.flatten())
            dir_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
            
            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': dir_accuracy
            }
        
        return results
    
    def inverse_transform_price(self, scaled_data):
        """Convert scaled predictions back to original price range"""
        # For single value
        if isinstance(scaled_data, (int, float)):
            return self.price_scaler.inverse_transform([[scaled_data]])[0,0]
        
        # For arrays
        if len(scaled_data.shape) == 1:
            return self.price_scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
        
        return self.price_scaler.inverse_transform(scaled_data)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from XGBoost model"""
        if self.xgb_model is None:
            return None
        
        # Handle different XGBoost APIs
        try:
            if hasattr(self.xgb_model, 'feature_importances_'):
                # scikit-learn API
                importance = self.xgb_model.feature_importances_
            elif hasattr(self.xgb_model, 'get_score'):
                # Core XGBoost API
                importance_dict = self.xgb_model.get_score(importance_type='weight')
                # Match feature index with name
                importance = np.zeros(len(feature_names))
                for k, v in importance_dict.items():
                    if k.startswith('f'):
                        try:
                            idx = int(k[1:])
                            if idx < len(importance):
                                importance[idx] = v
                        except:
                            pass
            else:
                # Fallback - equal importance
                importance = np.ones(len(feature_names)) / len(feature_names)
                st.warning("Could not get feature importance - using equal weights")
        except:
            # If all else fails
            importance = np.ones(len(feature_names)) / len(feature_names)
            st.warning("Error getting feature importance - using equal weights")
        
        return dict(zip(feature_names, importance))

# Sidebar for inputs
with st.sidebar:
    st.header("Model Parameters")
    
    # Stock selection
    ticker_symbol = st.text_input("Stock Ticker Symbol", "AAPL")
    
    # Date range
    st.subheader("Date Range")
    today = datetime.now()
    start_date = st.date_input(
        "Start Date",
        value=today - timedelta(days=3*365),
        min_value=datetime(2000, 1, 1),
        max_value=today
    )
    
    end_date = st.date_input(
        "End Date",
        value=today
    )
    
    # Model parameters
    st.subheader("LSTM Parameters")
    time_steps = st.slider("Time Steps (Days)", 30, 120, 60, 10)
    lstm_units = st.slider("LSTM Units", 32, 128, 50, 8)
    lstm_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    
    st.subheader("XGBoost Parameters")
    xgb_max_depth = st.slider("Max Depth", 2, 10, 3, 1)
    xgb_eta = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    
    st.subheader("Hybrid Model")
    lstm_weight = st.slider("LSTM Weight", 0.0, 1.0, 0.7, 0.1)
    
    # Training parameters
    st.subheader("Training Parameters")
    epochs = st.slider("Epochs", 10, 200, 50, 10)
    batch_size = st.slider("Batch Size", 16, 128, 32, 8)
    
    # Train button
    train_button = st.button("Train Model", type="primary")

# Main containers
status_container = st.empty()
data_container = st.container()
metrics_container = st.container()
plots_container = st.container()
predict_container = st.container()

# Main execution logic
if train_button:
    status_container.info(f"Loading stock data for {ticker_symbol}...")
    
    try:
        # Download stock data
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            status_container.error(f"No data found for ticker {ticker_symbol}")
        else:
            # Display raw data
            with data_container:
                st.subheader("Stock Data Overview")
                st.dataframe(stock_data.tail())
                
                # Plot raw price
                fig = plt.figure(figsize=(10, 5))
                plt.plot(stock_data.index, stock_data['Close'])
                plt.title(f"{ticker_symbol} Close Price History")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Add technical indicators
            status_container.info("Adding technical indicators and preparing data...")
            stock_data_with_features = add_technical_indicators(stock_data)
            
            # Display features
            with data_container:
                st.subheader("Technical Indicators")
                st.dataframe(stock_data_with_features.tail())
            
            # Initialize model
            status_container.info("Building and training the hybrid model...")
            model = HybridLSTM_XGBoost(
                lstm_units=lstm_units,
                lstm_dropout=lstm_dropout,
                time_steps=time_steps,
                xgb_max_depth=xgb_max_depth,
                xgb_eta=xgb_eta,
                lstm_weight=lstm_weight
            )
            
            # Prepare data
            prepared_data = model.prepare_data(stock_data_with_features)
            
            # Train model
            history = model.train(
                prepared_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Make predictions
            status_container.info("Making predictions and evaluating model performance...")
            predictions = model.predict(
                prepared_data['X_lstm_test'],
                prepared_data['X_xgb_test']
            )
            
            # Convert predictions to original scale
            y_test_inv = model.inverse_transform_price(prepared_data['y_test'])
            pred_inv = {
                'lstm': model.inverse_transform_price(predictions['lstm']),
                'xgb': model.inverse_transform_price(predictions['xgb']),
                'hybrid': model.inverse_transform_price(predictions['hybrid'])
            }
            
            # Evaluate model
            eval_scaled = model.evaluate(prepared_data['y_test'], predictions)
            eval_original = model.evaluate(y_test_inv, pred_inv)
            
            # Display metrics
            with metrics_container:
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">LSTM Model</div>
                            <div class="metric-value">${eval_original['lstm']['mae']:.2f}</div>
                            <div>Mean Absolute Error</div>
                            <div>RMSE: ${eval_original['lstm']['rmse']:.2f}</div>
                            <div>Directional Accuracy: {eval_original['lstm']['directional_accuracy']:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Hybrid Model</div>
                            <div class="metric-value">${eval_original['hybrid']['mae']:.2f}</div>
                            <div>Mean Absolute Error</div>
                            <div>RMSE: ${eval_original['hybrid']['rmse']:.2f}</div>
                            <div>Directional Accuracy: {eval_original['hybrid']['directional_accuracy']:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Plot results
            with plots_container:
                st.subheader("Prediction Results")
                
                # Get dates for test data
                test_start_idx = prepared_data['train_size'] + time_steps
                test_dates = stock_data_with_features.index[test_start_idx:test_start_idx+len(y_test_inv)]
                
                # Ensure matching lengths
                min_len = min(len(test_dates), len(y_test_inv))
                test_dates = test_dates[:min_len]
                y_test_inv = y_test_inv[:min_len]
                pred_inv['lstm'] = pred_inv['lstm'][:min_len]
                pred_inv['hybrid'] = pred_inv['hybrid'][:min_len]
                
                # Ensure consistent dimensions by flattening all arrays
                if len(y_test_inv.shape) > 1:
                    y_test_inv = y_test_inv.flatten()
                
                if len(pred_inv['lstm'].shape) > 1:
                    pred_inv['lstm'] = pred_inv['lstm'].flatten()
                    
                if len(pred_inv['hybrid'].shape) > 1:
                    pred_inv['hybrid'] = pred_inv['hybrid'].flatten()
                
                # Convert all to numpy arrays explicitly (if not already)
                y_test_inv = np.array(y_test_inv, dtype=np.float64)
                pred_inv['lstm'] = np.array(pred_inv['lstm'], dtype=np.float64)
                pred_inv['hybrid'] = np.array(pred_inv['hybrid'], dtype=np.float64)
                
                # Log prediction arrays to debug
                st.write(f"Debug - Actual values shape: {y_test_inv.shape}, LSTM shape: {pred_inv['lstm'].shape}, Hybrid shape: {pred_inv['hybrid'].shape}")
                
                # Show sample values to ensure they're numeric and valid
                if len(y_test_inv) > 0:
                    st.write(f"Sample values - Actual: {y_test_inv[0]}, LSTM: {pred_inv['lstm'][0]}, Hybrid: {pred_inv['hybrid'][0]}")
                
                # Create plotly figure with explicit dimensions
                fig = go.Figure(layout=dict(
                    width=800,
                    height=500,
                    title=f"{ticker_symbol} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                ))
                
                # Add traces - ensure we're plotting non-empty data
                if len(y_test_inv) > 0:
                    # Actual values
                    fig.add_trace(go.Scatter(
                        x=test_dates, 
                        y=y_test_inv,
                        mode='lines',
                        name='Actual',
                        line=dict(color='black', width=2)
                    ))
                    
                    # LSTM predictions
                    fig.add_trace(go.Scatter(
                        x=test_dates, 
                        y=pred_inv['lstm'],
                        mode='lines',
                        name='LSTM',
                        line=dict(color='blue', width=1.5)
                    ))
                    
                    # Hybrid predictions
                    fig.add_trace(go.Scatter(
                        x=test_dates, 
                        y=pred_inv['hybrid'],
                        mode='lines',
                        name='Hybrid',
                        line=dict(color='red', width=2)
                    ))
                else:
                    st.warning("No prediction data available to plot")
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("XGBoost Feature Importance")
                
                try:
                    feature_importance = model.get_feature_importance(prepared_data['xgb_features'])
                    
                    if feature_importance:
                        feature_df = pd.DataFrame({
                            'Feature': list(feature_importance.keys()),
                            'Importance': list(feature_importance.values())
                        }).sort_values('Importance', ascending=False)
                        
                        fig_feat = px.bar(
                            feature_df, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title="Feature Importance"
                        )
                        
                        st.plotly_chart(fig_feat, use_container_width=True)
                    else:
                        st.info("Feature importance information is not available for this model.")
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
                    st.info("This may occur when using the core XGBoost API instead of scikit-learn wrapper.")
            
            # Predict next day
            status_container.info("Predicting next trading day price...")
            
            with predict_container:
                st.subheader("Next Day Prediction")
                
                try:
                    # Get last sequence for LSTM
                    last_sequence = stock_data_with_features['Close'].values[-time_steps:]
                    scaled_last_sequence = model.price_scaler.transform(last_sequence.reshape(-1, 1))
                    lstm_input = scaled_last_sequence.reshape(1, time_steps, 1)
                    
                    # Get last features for XGBoost
                    last_features = stock_data_with_features[prepared_data['xgb_features']].iloc[-1].values.reshape(1, -1)
                    scaled_last_features = model.feature_scaler.transform(last_features)
                    
                    # Make predictions
                    lstm_pred = model.lstm_model.predict(lstm_input, verbose=0)
                    
                    # Handle prediction based on which XGBoost API we used
                    # First, let's check if this is a scikit-learn style XGBoost model vs core API model
                    use_scikit_api = hasattr(model.xgb_model, 'get_booster')
                    
                    # XGBoost prediction with error handling
                    xgb_pred = None
                    try:
                        if use_scikit_api:
                            # It's a scikit-learn wrapper
                            xgb_pred = model.xgb_model.predict(scaled_last_features).reshape(-1, 1)
                        else:
                            # It's the core XGBoost model from train()
                            import xgboost as xgb
                            # Ensure data is numpy array
                            dtest = xgb.DMatrix(np.asarray(scaled_last_features))
                            xgb_pred = model.xgb_model.predict(dtest).reshape(-1, 1)
                    except Exception as e:
                        st.warning(f"XGBoost prediction failed: {str(e)}. Using LSTM prediction only.")
                        xgb_pred = lstm_pred
                        model.lstm_weight = 1.0
                        model.xgb_weight = 0.0
                    
                    # Combine predictions using weights
                    hybrid_pred = model.lstm_weight * lstm_pred + model.xgb_weight * xgb_pred
                    
                    # Convert to original scale with additional error handling
                    try:
                        next_price = {
                            'lstm': float(model.inverse_transform_price(lstm_pred)[0,0]),
                            'xgb': float(model.inverse_transform_price(xgb_pred)[0,0]),
                            'hybrid': float(model.inverse_transform_price(hybrid_pred)[0,0])
                        }
                    except Exception as e:
                        st.warning(f"Error converting predictions to original scale: {str(e)}")
                        # Fallback to simple conversion
                        next_price = {
                            'lstm': float(lstm_pred[0][0]),
                            'xgb': float(xgb_pred[0][0]),
                            'hybrid': float(hybrid_pred[0][0])
                        }
                    
                    # Get the last known price
                    last_price = float(stock_data['Close'].iloc[-1])
                    
                    # Print debug info
                    st.write(f"Debug - last_price type: {type(last_price)}, hybrid_pred type: {type(hybrid_pred)}")
                    
                    # Calculate change - ensure everything is scalar
                    price_change = float(next_price['hybrid']) - last_price
                    price_change_pct = (price_change / last_price) * 100
                    
                    # Direction - ensure scalar comparison
                    is_positive = float(price_change) > 0
                    direction = "📈 UP" if is_positive else "📉 DOWN"
                    color = "green" if is_positive else "red"
                    
                    # Display prediction
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 1rem 0;">
                        <h3 style="margin-top: 0; color: #0f172a;">Next Trading Day Prediction for {ticker_symbol}</h3>
                        <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                            <div>
                                <h4 style="margin: 0; color: #64748b;">Last Price</h4>
                                <h2 style="margin: 0;">${last_price:.2f}</h2>
                                <p style="color: #64748b; margin: 0;">As of {stock_data.index[-1].strftime('%Y-%m-%d')}</p>
                            </div>
                            <div>
                                <h4 style="margin: 0; color: #64748b;">Predicted Price</h4>
                                <h2 style="margin: 0; color: {color};">${next_price['hybrid']:.2f}</h2>
                                <p style="color: #64748b; margin: 0;">Hybrid Model</p>
                            </div>
                            <div>
                                <h4 style="margin: 0; color: #64748b;">Expected Change</h4>
                                <h2 style="margin: 0; color: {color};">{direction} {abs(price_change):.2f} ({price_change_pct:.2f}%)</h2>
                            </div>
                        </div>
                        <div style="margin-top: 1rem;">
                            <p><strong>LSTM Prediction:</strong> ${next_price['lstm']:.2f} (Weight: {model.lstm_weight:.1f})</p>
                            <p><strong>XGBoost Prediction:</strong> ${next_price['xgb']:.2f} (Weight: {model.xgb_weight:.1f})</p>
                        </div>
                        <p style="color: #64748b; font-style: italic; margin-top: 1rem;">Note: This prediction is based on historical patterns and should not be used as the sole basis for investment decisions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error predicting next day price: {str(e)}")
            
            # Final status
            status_container.success("Analysis completed successfully!")
    
    except Exception as e:
        status_container.error(f"An error occurred: {str(e)}")
        st.exception(e)

# Run the app with: streamlit run hybrid_stock_app.py 