import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from lstm_model import LSTMStockPredictor

# Set page config
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #0ea5e9;
    }
    h1, h2, h3 {
        margin-top: 1rem;
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
    .challenge-card {
        background-color: #f8fafc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0ea5e9;
    }
    .info-box {
        background-color: #f0f9ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #bae6fd;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("LSTM Stock Price Prediction")
st.markdown("""
<div class="info-box">
This app uses Long Short-Term Memory (LSTM) neural networks, which are specially designed to handle sequential data like stock prices.
Unlike standard neural networks, LSTMs maintain an internal memory state that helps them better capture long-term dependencies in time series data.
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Predictor", "Market Challenges & Solutions"])

with tab1:
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
            value=datetime(2000, 1, 3),  # Changed to start from Jan 3, 2000
            min_value=datetime(1980, 1, 1),  # Allow even earlier data if needed
            max_value=today - timedelta(days=365)  # Ensure at least 1 year of data
        )
        
        end_date = st.date_input(
            "End Date",
            value=today
        )
        
        # Add note about using longer historical data
        st.markdown("""
        <div style="background-color: #f0f9ff; border-radius: 0.5rem; padding: 0.5rem; margin: 0.5rem 0; border: 1px solid #bae6fd; font-size: 0.8rem;">
            <strong>Note:</strong> Using longer historical data (e.g., from 2000) can help the model learn long-term market cycles, including bull and bear markets.
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter tuning options
        st.subheader("Model Optimization")
        
        # Select optimization metric with explanation
        st.markdown("""
        <div style="background-color: #f0f9ff; border-radius: 0.5rem; padding: 0.7rem; margin: 0.5rem 0; border: 1px solid #bae6fd;">
            <h4 style="margin-top: 0;">What does "Optimize For" mean?</h4>
            <p style="margin-bottom: 0.5rem;">Select which metric the model should prioritize:</p>
            <ul style="margin-top: 0;">
                <li><strong>MAE:</strong> Mean Absolute Error - the average dollar error in predictions (lower is better)</li>
                <li><strong>RMSE:</strong> Root Mean Squared Error - emphasizes larger errors more (lower is better)</li>
                <li><strong>MAPE:</strong> Mean Absolute Percentage Error - error as a percentage of actual price (lower is better)</li>
                <li><strong>R² Score:</strong> How well the model explains price variations (higher is better)</li>
                <li><strong>Directional Accuracy:</strong> How often the model correctly predicts price movement direction (higher is better)</li>
            </ul>
            <p style="margin-bottom: 0;"><strong>For predicting next day's absolute closing price:</strong> MAE is usually the best choice.</p>
        </div>
        """, unsafe_allow_html=True)
        
        optimization_metric = st.selectbox(
            "Optimize For",
            ["MAE", "RMSE", "MAPE", "R² Score", "Directional Accuracy"],
            index=0,  # Default to MAE for absolute price predictions
            help="The metric that will be optimized during parameter tuning"
        )
        
        # Quick or comprehensive tuning
        st.markdown("""
        <div style="background-color: #f0f9ff; border-radius: 0.5rem; padding: 0.7rem; margin: 0.5rem 0; border: 1px solid #bae6fd;">
            <h4 style="margin-top: 0;">What does "Tuning Intensity" mean?</h4>
            <p>This controls how many different model configurations will be tested:</p>
            <ul style="margin-bottom: 0;">
                <li><strong>Quick:</strong> Tests 3 configurations - faster but less thorough</li>
                <li><strong>Medium:</strong> Tests 6 configurations - balanced approach</li>
                <li><strong>Comprehensive:</strong> Tests 9 configurations - most thorough but slowest</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        tuning_mode = st.radio(
            "Tuning Intensity",
            ["Quick (3 configs)", "Medium (6 configs)", "Comprehensive (9 configs)"],
            help="More configurations will take longer but may find better parameters"
        )
        
        # Select what to tune
        st.write("Parameters to tune:")
        tune_time_window = st.checkbox("Time Window", value=True)
        tune_layer_size = st.checkbox("Layer Size", value=True)
        tune_dropout = st.checkbox("Dropout Rate", value=True)
        tune_epochs = st.checkbox("Epochs", value=True)
        tune_batch_size = st.checkbox("Batch Size", value=True)
        
        # Number of epochs for tuning - now just a info message since we're tuning it
        if not tune_epochs:
            tuning_epochs = st.slider("Epochs per Config", 10, 50, 20, 5,
                                     help="Fewer epochs speed up tuning but may be less accurate")
        else:
            st.info("Epochs will be tuned (50, 75, 100). This may take longer to train.")
            tuning_epochs = 20  # Default only used if tune_epochs is False
        
        # Feature selection
        st.subheader("Features")
        use_close = st.checkbox("Close Price", value=True)
        
        # Moving Averages
        st.subheader("Moving Averages")
        use_ma20 = st.checkbox("20-day MA", value=True)
        use_ma50 = st.checkbox("50-day MA", value=True)
        use_ma100 = st.checkbox("100-day MA", value=False)
        use_ma200 = st.checkbox("200-day MA", value=True)
        
        # Add data transformation for long-term analysis with explanations
        st.subheader("Data Transformation")
        
        st.markdown("""
        <div style="background-color: #f0f9ff; border-radius: 0.5rem; padding: 0.7rem; margin: 0.5rem 0; border: 1px solid #bae6fd;">
            <h4 style="margin-top: 0;">What does "Price Transformation" mean?</h4>
            <p>This changes how price data is represented in the model:</p>
            <ul>
                <li><strong>None:</strong> Uses raw price data - good for short-term predictions in stable markets</li>
                <li><strong>Log Transform:</strong> Takes the logarithm of prices - <span style="color: #0369a1; font-weight: bold;">best for predicting absolute prices</span> in markets with long-term growth</li>
                <li><strong>Percent Change:</strong> Uses day-to-day percentage changes - good for predicting price direction but not absolute values</li>
            </ul>
            <p style="margin-bottom: 0;"><strong>For predicting next day's absolute closing price:</strong> "Log Transform" is usually better, especially for stocks with long-term upward trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
        transform_option = st.selectbox(
            "Price Transformation",
            options=["None", "Log Transform", "Percent Change"],
            index=1,  # Default to Log Transform
            help="Log transformation helps with long-term exponential growth; Percent Change focuses on relative movements"
        )
        
        # Train button - only need one version now
        st.write("")
        train_button = st.button("Start Model Optimization", type="primary")
    
    # Main content area
    status_container = st.empty()
    data_container = st.container()
    metrics_container = st.container()
    plots_container = st.container()
    prediction_container = st.container()
    rolling_container = st.container()
    
    # Main execution logic
    if train_button:
        # Update status
        status_container.info(f"Loading stock data for {ticker_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        # If the date range is very long, add a note about performance
        date_diff = (end_date - start_date).days
        if date_diff > 3650:  # More than 10 years
            status_container.warning("Training with more than 10 years of data may take longer. Please be patient.")
            
            # Suggest optimal configuration for long time series
            st.sidebar.markdown("""
            <div style="background-color: #fffbeb; border-radius: 0.5rem; padding: 0.5rem; margin: 0.5rem 0; border: 1px solid #fef3c7; font-size: 0.8rem;">
                <strong>Recommended settings for long-term data:</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1rem;">
                    <li>Use Log Transform for price data</li>
                    <li>Increase Time Window to 100-120 days</li>
                    <li>Include 200-day MA</li>
                    <li>Consider increasing LSTM layer size</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            # Load data with yfinance
            stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            if stock_data.empty:
                status_container.error(f"No data found for ticker {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Display the data
                with data_container:
                    st.subheader("Stock Data Overview")
                    st.dataframe(stock_data.tail())
                    
                    # Plot stock price history
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(stock_data['Close'])
                    plt.title(f"{ticker_symbol} Close Price History")
                    plt.xlabel("Date")
                    plt.ylabel("Price (USD)")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Feature selection
                selected_features = []
                if use_close:
                    # Apply data transformation if selected
                    if transform_option == "Log Transform":
                        stock_data['Close_Log'] = np.log(stock_data['Close'])
                        selected_features.append('Close_Log')
                        # Display message about log transform
                        st.info("Using logarithmic transformation of prices to handle long-term exponential growth.")
                    elif transform_option == "Percent Change":
                        stock_data['Close_Pct'] = stock_data['Close'].pct_change()
                        # Drop the first row which will have NaN
                        stock_data = stock_data.iloc[1:].copy()
                        selected_features.append('Close_Pct')
                        # Display message about percent change
                        st.info("Using percent change transformation to focus on relative price movements rather than absolute values.")
                    else:
                        selected_features.append('Close')
                
                # Add moving averages
                if use_ma20:
                    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
                    selected_features.append('MA20')
                
                if use_ma50:
                    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                    selected_features.append('MA50')
                    
                if use_ma100:
                    stock_data['MA100'] = stock_data['Close'].rolling(window=100).mean()
                    selected_features.append('MA100')
                    
                if use_ma200:
                    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
                    selected_features.append('MA200')
                
                # Identify major market cycles for very long data periods
                if date_diff > 3650:  # More than 10 years of data
                    with data_container:
                        st.subheader("Long-term Market Cycles")
                        
                        # Create figure for cycles
                        fig_cycles = plt.figure(figsize=(14, 7))
                        
                        # Plot closing price
                        plt.plot(stock_data.index, stock_data['Close'], label='Close Price', linewidth=1.5)
                        
                        # Calculate 200-day MA regardless of checkbox for trend identification
                        if 'MA200' not in stock_data.columns:
                            stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
                        
                        # Plot 200-day MA
                        plt.plot(stock_data.index, stock_data['MA200'], label='200-day MA', linewidth=2)
                        
                        # Force clean data before comparison - no NaN values
                        plot_data = stock_data.copy()
                        
                        # Check what columns are actually available in the dataset
                        st.write(f"Available columns: {list(plot_data.columns)}")
                        
                        # Check if DataFrame has multi-index columns
                        has_multi_index = isinstance(plot_data.columns, pd.MultiIndex)
                        
                        # Check if 'Close' and 'MA200' columns exist before dropping
                        # For safety, we'll make sure we have the columns we need
                        column_to_check = []
                        
                        # Look for 'Close' column or alternative formats
                        if has_multi_index:
                            # For multi-index columns like ('Close', 'AAPL')
                            close_cols = [col for col in plot_data.columns if col[0] == 'Close']
                            ma200_cols = [col for col in plot_data.columns if col[0] == 'MA200']
                            
                            if close_cols:
                                column_to_check.append(close_cols[0])  # Use the full tuple
                            
                            # Look for MA200 column
                            if ma200_cols:
                                column_to_check.append(ma200_cols[0])  # Use the full tuple
                            elif close_cols:
                                # Calculate MA200 if missing
                                close_col = close_cols[0]
                                ma200_col = ('MA200', close_col[1] if len(close_col) > 1 else '')
                                plot_data[ma200_col] = plot_data[close_col].rolling(window=200).mean()
                                column_to_check.append(ma200_col)
                        else:
                            # For standard columns
                            if 'Close' in plot_data.columns:
                                column_to_check.append('Close')
                            elif 'close' in plot_data.columns:
                                column_to_check.append('close')
                                plot_data.rename(columns={'close': 'Close'}, inplace=True)
                                
                            if 'MA200' in plot_data.columns:
                                column_to_check.append('MA200')
                            elif 'Close' in plot_data.columns:
                                plot_data['MA200'] = plot_data['Close'].rolling(window=200).mean()
                                column_to_check.append('MA200')
                        
                        # Only drop NA if we have columns to check
                        if column_to_check:
                            plot_data = plot_data.dropna(subset=column_to_check)
                            st.write(f"Dropping NaN values for columns: {column_to_check}")
                        else:
                            st.warning("Required columns for market cycle analysis not found in dataset.")
                        
                        # Instead of using pandas comparison operators, use numpy for element-wise comparison
                        # to avoid alignment issues
                        try:
                            # Make sure we have both columns before proceeding with comparison
                            has_required_columns = False
                            
                            if has_multi_index:
                                # For multi-index columns
                                close_cols = [col for col in plot_data.columns if col[0] == 'Close']
                                ma200_cols = [col for col in plot_data.columns if col[0] == 'MA200']
                                
                                if close_cols and ma200_cols:
                                    close_col = close_cols[0]
                                    ma200_col = ma200_cols[0]
                                    close_prices = plot_data[close_col].values
                                    ma200_values = plot_data[ma200_col].values
                                    dates = plot_data.index
                                    has_required_columns = True
                                else:
                                    st.warning("Missing necessary columns ('Close' and/or 'MA200') for market cycle identification")
                            elif 'Close' in plot_data.columns and 'MA200' in plot_data.columns:
                                # For standard columns
                                close_prices = plot_data['Close'].values
                                ma200_values = plot_data['MA200'].values
                                dates = plot_data.index
                                has_required_columns = True
                            else:
                                st.warning("Missing necessary columns ('Close' and/or 'MA200') for market cycle identification")
                            
                            # Only proceed if we have the necessary columns
                            if has_required_columns:
                                # Create boolean masks for bull and bear markets
                                bull_mask = close_prices > ma200_values
                                bear_mask = close_prices < ma200_values
                                
                                # Find the min price for the fill_between
                                min_price = np.min(close_prices)
                                
                                # Highlight bull periods
                                for i in range(1, len(bull_mask)):
                                    if bull_mask[i-1] == False and bull_mask[i] == True:
                                        # Start of bull market
                                        start_idx = i
                                        for j in range(i+1, len(bull_mask)):
                                            if bull_mask[j] == False:
                                                # End of bull market
                                                end_idx = j
                                                # Color this segment
                                                plt.fill_between(
                                                    dates[start_idx:end_idx],
                                                    min_price,
                                                    close_prices[start_idx:end_idx],
                                                    alpha=0.1, color='green'
                                                )
                                                break
                                        else:
                                            # If we reach the end without finding an end, color till the end
                                            plt.fill_between(
                                                dates[start_idx:],
                                                min_price,
                                                close_prices[start_idx:],
                                                alpha=0.1, color='green'
                                            )
                                
                                # Highlight bear periods
                                for i in range(1, len(bear_mask)):
                                    if bear_mask[i-1] == False and bear_mask[i] == True:
                                        # Start of bear market
                                        start_idx = i
                                        for j in range(i+1, len(bear_mask)):
                                            if bear_mask[j] == False:
                                                # End of bear market
                                                end_idx = j
                                                # Color this segment
                                                plt.fill_between(
                                                    dates[start_idx:end_idx],
                                                    min_price,
                                                    close_prices[start_idx:end_idx],
                                                    alpha=0.1, color='red'
                                                )
                                                break
                                        else:
                                            # If we reach the end without finding an end, color till the end
                                            plt.fill_between(
                                                dates[start_idx:],
                                                min_price,
                                                close_prices[start_idx:],
                                                alpha=0.1, color='red'
                                            )
                        except Exception as e:
                            st.warning(f"Could not highlight market cycles: {str(e)}")
                            
                        # Add major market events as vertical lines
                        try:
                            # Get the max price for text positioning
                            if has_multi_index:
                                close_cols = [col for col in stock_data.columns if col[0] == 'Close']
                                if close_cols:
                                    max_price = stock_data[close_cols[0]].max()
                                else:
                                    max_price = 1000  # Default value if Close column not found
                            else:
                                max_price = stock_data['Close'].max()
                            
                            if start_date.year <= 2000:
                                # Dot-com bubble burst
                                plt.axvline(x=pd.Timestamp('2000-03-10'), color='red', linestyle='--', alpha=0.7)
                                plt.text(pd.Timestamp('2000-03-10'), max_price*0.95, 'Dot-com Peak', rotation=90)
                                
                            if start_date.year <= 2008 and end_date.year >= 2008:
                                # 2008 Financial Crisis
                                plt.axvline(x=pd.Timestamp('2008-09-15'), color='red', linestyle='--', alpha=0.7)
                                plt.text(pd.Timestamp('2008-09-15'), max_price*0.95, '2008 Crisis', rotation=90)
                                
                            if start_date.year <= 2020 and end_date.year >= 2020:
                                # COVID-19 Crash
                                plt.axvline(x=pd.Timestamp('2020-03-23'), color='red', linestyle='--', alpha=0.7)
                                plt.text(pd.Timestamp('2020-03-23'), max_price*0.95, 'COVID-19', rotation=90)
                        except Exception as e:
                            st.warning(f"Could not add market event markers: {str(e)}")
                        
                        # Finalize plot
                        plt.title(f"{ticker_symbol} Price and Long-term Market Cycles")
                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_cycles)
                        
                        # Add explanation
                        st.markdown("""
                        <div class="info-box">
                            <h4>Understanding Market Cycles</h4>
                            <p>The chart above shows major market cycles:</p>
                            <ul>
                                <li><span style="color:green">Green areas</span>: Bull markets (price above 200-day MA)</li>
                                <li><span style="color:red">Red areas</span>: Bear markets (price below 200-day MA)</li>
                                <li>Vertical lines: Major market events that could affect model training</li>
                            </ul>
                            <p>LSTMs can learn these patterns to better understand market regime changes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Plot with moving averages
                with data_container:
                    st.subheader("Stock Price with Moving Averages")
                    fig_ma = plt.figure(figsize=(12, 6))
                    plt.plot(stock_data['Close'], label='Close Price', linewidth=2)
                    
                    if use_ma20:
                        plt.plot(stock_data['MA20'], label='20-day MA', linestyle='--')
                    if use_ma50:
                        plt.plot(stock_data['MA50'], label='50-day MA', linestyle='--')
                    if use_ma100:
                        plt.plot(stock_data['MA100'], label='100-day MA', linestyle='--')
                    if use_ma200:
                        plt.plot(stock_data['MA200'], label='200-day MA', linestyle='--')
                    
                    plt.title(f"{ticker_symbol} Price and Moving Averages")
                    plt.xlabel("Date")
                    plt.ylabel("Price (USD)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_ma)
                
                # Drop rows with NaN values
                stock_data = stock_data.dropna()
                
                # Update status
                status_container.info("Preparing data and starting model optimization...")
                
                # Parameter tuning mode (always use this mode now)
                status_container.info("Testing multiple model configurations to find the optimal parameters...")
                
                # Preprocess data once for all models
                # Create a temporary predictor just for preprocessing
                temp_predictor = LSTMStockPredictor(time_steps=60)
                preprocessed_data = temp_predictor.preprocess_data(stock_data, selected_features)
                
                # Define parameter combinations to test
                if tuning_mode == "Quick (3 configs)":
                    num_configs = 3
                elif tuning_mode == "Medium (6 configs)":
                    num_configs = 6
                else:  # Comprehensive
                    num_configs = 9
                
                # Initialize parameters to test
                time_windows_to_test = []
                layer_sizes_to_test = []
                dropout_rates_to_test = []
                epochs_to_test = []
                batch_sizes_to_test = []
                
                # Time window parameters
                if tune_time_window:
                    if num_configs == 3:
                        time_windows_to_test = [30, 45, 60]
                    elif num_configs == 6:
                        time_windows_to_test = [20, 30, 40, 50, 60, 90]
                    else:  # Comprehensive
                        time_windows_to_test = [20, 30, 40, 45, 50, 60, 75, 90, 120]
                else:
                    time_windows_to_test = [30, 60]  # Default to both requested values
                
                # Layer size parameters
                if tune_layer_size:
                    if num_configs == 3:
                        layer_sizes_to_test = [[32, 32], [64, 64]]
                    elif num_configs == 6:
                        layer_sizes_to_test = [[32, 32], [64, 32], [64, 64], [128, 64], [128, 128], [200, 100]]
                    else:
                        layer_sizes_to_test = [[32, 16], [32, 32], [64, 32], [64, 64], [128, 64], [128, 128], [200, 100], [200, 200], [300, 150]]
                else:
                    layer_sizes_to_test = [[50, 50]]  # Default
                
                # Dropout rate parameters
                if tune_dropout:
                    if num_configs == 3:
                        dropout_rates_to_test = [0.1, 0.2, 0.3]
                    elif num_configs == 6:
                        dropout_rates_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                    else:
                        dropout_rates_to_test = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
                else:
                    dropout_rates_to_test = [0.2]  # Default
                
                # Epochs parameters
                if tune_epochs:
                    epochs_to_test = [50, 75, 100]  # As requested
                else:
                    epochs_to_test = [tuning_epochs]  # Use the value from slider or default
                
                # Batch size parameters
                if tune_batch_size:
                    batch_sizes_to_test = [32, 64]  # Only the requested values
                else:
                    batch_sizes_to_test = [32]  # Default to smaller batch size
                
                # Create tuning grid
                tuning_results = []
                
                # Calculate total number of configurations
                total_configs = len(time_windows_to_test) * len(layer_sizes_to_test) * len(dropout_rates_to_test) * len(epochs_to_test) * len(batch_sizes_to_test)
                st.info(f"Testing {total_configs} different model configurations. This may take some time...")
                
                # Create configuration progress
                config_progress = st.progress(0)
                config_status = st.empty()
                
                # Counter for configurations
                config_counter = 0
                
                # Start tuning loop - limit to max 12 combinations to prevent excessive computation
                max_combinations = min(12, total_configs)
                tested_combinations = 0
                
                best_metric_value = float('inf')  # Lower is better for most metrics
                if optimization_metric == "R² Score" or optimization_metric == "Directional Accuracy":
                    best_metric_value = -float('inf')  # Higher is better for R² and directional accuracy
                
                best_config = None
                best_predictor = None
                
                # Helper function to get configs to test
                def get_configs_to_test(time_windows, layer_sizes, dropout_rates, epochs_list, batch_sizes, max_combinations):
                    combinations = []
                    for tw in time_windows:
                        for ls in layer_sizes:
                            for dr in dropout_rates:
                                for ep in epochs_list:
                                    for bs in batch_sizes:
                                        combinations.append((tw, ls, dr, ep, bs))
                                        if len(combinations) >= max_combinations:
                                            return combinations
                    return combinations
                
                # Generate configs
                configs_to_test = get_configs_to_test(
                    time_windows_to_test, 
                    layer_sizes_to_test, 
                    dropout_rates_to_test,
                    epochs_to_test,
                    batch_sizes_to_test,
                    max_combinations
                )
                
                # Test each configuration
                for time_window, layer_size, dropout_rate, epochs, batch_size in configs_to_test:
                    config_counter += 1
                    config_status.info(f"Testing configuration {config_counter}/{max_combinations}: Time Window={time_window}, Layers={layer_size}, Dropout={dropout_rate}, Epochs={epochs}, Batch Size={batch_size}")
                    
                    # Need to preprocess again if time window changes
                    if time_window != temp_predictor.time_steps:
                        temp_predictor = LSTMStockPredictor(time_steps=time_window)
                        preprocessed_data = temp_predictor.preprocess_data(stock_data, selected_features)
                    
                    # Create and train model
                    try:
                        predictor = LSTMStockPredictor(
                            time_steps=time_window,
                            layers=layer_size,
                            dropout_rate=dropout_rate,
                            batch_size=batch_size,
                            epochs=epochs
                        )
                        
                        # Transfer the fitted scaler from temp_predictor to predictor
                        # This is crucial to avoid "scaler not fitted" errors
                        if hasattr(temp_predictor.scaler, 'scale_'):
                            predictor.scaler = temp_predictor.scaler
                        
                        # Train model with suppress_output=True to reduce console output
                        predictor.train(
                            preprocessed_data['X_train'],
                            preprocessed_data['y_train'],
                            preprocessed_data['X_test'],
                            preprocessed_data['y_test'],
                            callbacks=[
                                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
                            ]
                        )
                        
                        # Make predictions
                        y_pred_scaled = predictor.predict(preprocessed_data['X_test'])
                        
                        # Convert back to original scale
                        y_test = preprocessed_data['y_test']
                        y_test_actual = predictor.inverse_transform(y_test)
                        y_pred = predictor.inverse_transform(y_pred_scaled)
                        
                        # Calculate metrics
                        metrics = predictor.evaluate(y_test_actual, y_pred)
                        
                        # Store results
                        result = {
                            "Time Window": time_window,
                            "Layer Size": str(layer_size),
                            "Dropout Rate": dropout_rate,
                            "Epochs": epochs,
                            "Batch Size": batch_size,
                            "MAE": metrics['mae'],
                            "RMSE": metrics['rmse'],
                            "MAPE": metrics['mape'],
                            "R² Score": metrics['r2'],
                            "Directional Accuracy": metrics['directional_accuracy']
                        }
                        
                        tuning_results.append(result)
                        
                        # Check if this is the best model based on the selected metric
                        current_metric = None
                        if optimization_metric == "R² Score":
                            current_metric = metrics['r2']
                        elif optimization_metric == "Directional Accuracy":
                            current_metric = metrics['directional_accuracy']
                        else:
                            current_metric = metrics[optimization_metric.lower()]
                        
                        is_better = False
                        if optimization_metric == "R² Score" or optimization_metric == "Directional Accuracy":
                            # Higher is better
                            if current_metric > best_metric_value:
                                is_better = True
                        else:
                            # Lower is better
                            if current_metric < best_metric_value:
                                is_better = True
                        
                        if is_better:
                            best_metric_value = current_metric
                            best_config = (time_window, layer_size, dropout_rate, epochs, batch_size)
                            best_predictor = predictor
                        
                    except Exception as e:
                        st.error(f"Error with configuration (TW={time_window}, LS={layer_size}, DR={dropout_rate}, E={epochs}, BS={batch_size}): {str(e)}")
                    
                    # Update progress
                    config_progress.progress(config_counter / max_combinations)
                
                # Display tuning results
                config_status.empty()
                config_progress.empty()
                
                # Create table of results
                tuning_df = pd.DataFrame(tuning_results)
                
                # Sort by the selected optimization metric
                sort_ascending = True
                if optimization_metric == "R² Score" or optimization_metric == "Directional Accuracy":
                    sort_ascending = False
                
                tuning_df = tuning_df.sort_values(by=optimization_metric, ascending=sort_ascending)
                
                # Display results
                st.subheader("Parameter Tuning Results")
                st.dataframe(tuning_df)
                
                # Display best configuration
                st.subheader("Best Configuration")
                st.success(f"Best {optimization_metric}: {best_metric_value:.4f} with Time Window={best_config[0]}, Layer Size={best_config[1]}, Dropout={best_config[2]}, Epochs={best_config[3]}, Batch Size={best_config[4]}")
                
                # Visualization of tuning results
                st.subheader("Parameter Tuning Visualization")
                
                # Check what parameters were tuned
                if tune_time_window and len(time_windows_to_test) > 1:
                    # Plot metrics by time window
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Group by time window and calculate mean of the numeric metrics only
                    # Convert Layer Size to string first to avoid errors with non-numeric data
                    numeric_df = tuning_df.drop(columns=['Layer Size'])
                    tw_results = numeric_df.groupby('Time Window').mean().reset_index()
                    
                    # Select relevant metrics to plot
                    metrics_to_plot = ["MAE", "RMSE", "MAPE", "R² Score", "Directional Accuracy"]
                    
                    # Plot each metric
                    for metric in metrics_to_plot:
                        # For metrics where lower is better, normalize so higher is better for consistent visualization
                        if metric not in ["R² Score", "Directional Accuracy"]:
                            # Invert and scale to 0-1 range
                            values = tw_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = 1 - ((values - min(values)) / (max(values) - min(values)))
                                ax.plot(tw_results['Time Window'], normalized, 'o-', label=f"{metric} (normalized)")
                        else:
                            # For metrics where higher is better, just normalize to 0-1
                            values = tw_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = (values - min(values)) / (max(values) - min(values))
                                ax.plot(tw_results['Time Window'], normalized, 'o-', label=f"{metric} (normalized)")
                    
                    ax.set_title('Impact of Time Window on Model Performance')
                    ax.set_xlabel('Time Window (days)')
                    ax.set_ylabel('Normalized Performance (higher is better)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Also visualize impact of epochs if tuned
                if tune_epochs and len(epochs_to_test) > 1:
                    # Plot metrics by epochs
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Group by epochs and calculate mean of the numeric metrics only
                    # Convert Layer Size to string if needed
                    if 'Layer Size' in tuning_df.columns:
                        numeric_df = tuning_df.drop(columns=['Layer Size'])
                    else:
                        numeric_df = tuning_df.copy()
                    
                    epochs_results = numeric_df.groupby('Epochs').mean().reset_index()
                    
                    # Select relevant metrics to plot
                    metrics_to_plot = ["MAE", "RMSE", "MAPE", "R² Score", "Directional Accuracy"]
                    
                    # Plot each metric
                    for metric in metrics_to_plot:
                        # For metrics where lower is better, normalize so higher is better for consistent visualization
                        if metric not in ["R² Score", "Directional Accuracy"]:
                            # Invert and scale to 0-1 range
                            values = epochs_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = 1 - ((values - min(values)) / (max(values) - min(values)))
                                ax.plot(epochs_results['Epochs'], normalized, 'o-', label=f"{metric} (normalized)")
                        else:
                            # For metrics where higher is better, just normalize to 0-1
                            values = epochs_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = (values - min(values)) / (max(values) - min(values))
                                ax.plot(epochs_results['Epochs'], normalized, 'o-', label=f"{metric} (normalized)")
                    
                    ax.set_title('Impact of Epochs on Model Performance')
                    ax.set_xlabel('Number of Epochs')
                    ax.set_ylabel('Normalized Performance (higher is better)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Also visualize impact of batch size if tuned
                if tune_batch_size and len(batch_sizes_to_test) > 1:
                    # Plot metrics by batch size
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Group by batch size and calculate mean of the numeric metrics only
                    # Convert Layer Size to string if needed
                    if 'Layer Size' in tuning_df.columns:
                        numeric_df = tuning_df.drop(columns=['Layer Size'])
                    else:
                        numeric_df = tuning_df.copy()
                    
                    batch_results = numeric_df.groupby('Batch Size').mean().reset_index()
                    
                    # Select relevant metrics to plot
                    metrics_to_plot = ["MAE", "RMSE", "MAPE", "R² Score", "Directional Accuracy"]
                    
                    # Plot each metric
                    for metric in metrics_to_plot:
                        # For metrics where lower is better, normalize so higher is better for consistent visualization
                        if metric not in ["R² Score", "Directional Accuracy"]:
                            # Invert and scale to 0-1 range
                            values = batch_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = 1 - ((values - min(values)) / (max(values) - min(values)))
                                ax.plot(batch_results['Batch Size'], normalized, 'o-', label=f"{metric} (normalized)")
                        else:
                            # For metrics where higher is better, just normalize to 0-1
                            values = batch_results[metric]
                            if max(values) > min(values):  # Avoid division by zero
                                normalized = (values - min(values)) / (max(values) - min(values))
                                ax.plot(batch_results['Batch Size'], normalized, 'o-', label=f"{metric} (normalized)")
                    
                    ax.set_title('Impact of Batch Size on Model Performance')
                    ax.set_xlabel('Batch Size')
                    ax.set_ylabel('Normalized Performance (higher is better)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Create a parallel coordinates plot for all parameters if multiple were tuned
                if (tune_time_window and len(time_windows_to_test) > 1) or \
                   (tune_layer_size and len(layer_sizes_to_test) > 1) or \
                   (tune_dropout and len(dropout_rates_to_test) > 1) or \
                   (tune_epochs and len(epochs_to_test) > 1) or \
                   (tune_batch_size and len(batch_sizes_to_test) > 1):
                    
                    try:
                        # Create figure for parallel coordinates
                        import plotly.graph_objects as go
                        
                        # Prepare data - we need to normalize all metrics to 0-1 range
                        # For MAE, RMSE, MAPE lower is better, so we invert them
                        
                        plot_df = tuning_df.copy()
                        
                        # Normalize metrics
                        for metric in ["MAE", "RMSE", "MAPE"]:
                            if metric in plot_df.columns:
                                values = plot_df[metric]
                                if max(values) > min(values):  # Avoid division by zero
                                    plot_df[f"{metric} (norm)"] = 1 - ((values - min(values)) / (max(values) - min(values)))
                        
                        for metric in ["R² Score", "Directional Accuracy"]:
                            if metric in plot_df.columns:
                                values = plot_df[metric]
                                if max(values) > min(values):  # Avoid division by zero
                                    plot_df[f"{metric} (norm)"] = (values - min(values)) / (max(values) - min(values))
                        
                        # Convert layer size to string for proper display
                        plot_df["Layer Size"] = plot_df["Layer Size"].astype(str)
                        
                        # Create parallel coordinates plot
                        dimensions = []
                        
                        # Add tuned parameters
                        if tune_time_window:
                            dimensions.append(dict(range=[min(plot_df["Time Window"]), max(plot_df["Time Window"])],
                                                 label='Time Window', values=plot_df["Time Window"]))
                        
                        if tune_dropout:
                            dimensions.append(dict(range=[min(plot_df["Dropout Rate"]), max(plot_df["Dropout Rate"])],
                                                 label='Dropout Rate', values=plot_df["Dropout Rate"]))
                        
                        if tune_epochs:
                            dimensions.append(dict(range=[min(plot_df["Epochs"]), max(plot_df["Epochs"])],
                                                 label='Epochs', values=plot_df["Epochs"]))
                        
                        if tune_batch_size:
                            dimensions.append(dict(range=[min(plot_df["Batch Size"]), max(plot_df["Batch Size"])],
                                                 label='Batch Size', values=plot_df["Batch Size"]))
                        
                        # Add normalized metrics (higher is better for all of these)
                        normalized_metrics = [col for col in plot_df.columns if "(norm)" in col]
                        for metric in normalized_metrics:
                            dimensions.append(dict(range=[0, 1],
                                                 label=metric, values=plot_df[metric]))
                        
                        # Create the figure if we have dimensions
                        if dimensions:
                            fig = go.Figure(data=
                                go.Parcoords(
                                    line=dict(color=plot_df[f"{optimization_metric} (norm)" if f"{optimization_metric} (norm)" in plot_df.columns else optimization_metric],
                                            colorscale='Viridis',
                                            showscale=True),
                                    dimensions=dimensions
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create parallel coordinates plot: {str(e)}")
                
                # Use the best model for further analysis
                predictor = best_predictor
                
                # Extract predictions and metrics from the best model
                y_pred_scaled = predictor.predict(preprocessed_data['X_test'])
                y_test = preprocessed_data['y_test']
                y_test_actual = predictor.inverse_transform(y_test)
                y_pred = predictor.inverse_transform(y_pred_scaled)
                metrics = predictor.evaluate(y_test_actual, y_pred)
            
            # Display metrics
            with metrics_container:
                st.subheader("Model Performance Metrics")
                
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Mean Absolute Error</div>
                            <div class="metric-value">${metrics['mae']:.2f}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with cols[1]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Root Mean Squared Error</div>
                            <div class="metric-value">${metrics['rmse']:.2f}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with cols[2]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Directional Accuracy</div>
                            <div class="metric-value">{metrics['directional_accuracy']:.1f}%</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Display additional metrics
            cols = st.columns(2)
            with cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Mean Absolute Percentage Error</div>
                        <div class="metric-value">{metrics['mape']:.2f}%</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with cols[1]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{metrics['r2']:.3f}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Plot training history and predictions
            with plots_container:
                st.subheader("Training Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Training and Validation Loss**")
                    try:
                        history_fig = predictor.plot_training_history()
                        st.pyplot(history_fig)
                    except Exception as e:
                        st.error(f"Error plotting training history: {str(e)}")
                
                with col2:
                    st.write("**LSTM Predictions vs Actual Prices**")
                    try:
                        # Get the date index for test data
                        try:
                            # Get the base index position
                            base_idx = preprocessed_data['train_size'] + predictor.time_steps
                            # Make sure we don't exceed the length of the index
                            needed_len = base_idx + len(y_test)
                            if needed_len <= len(stock_data.index):
                                test_indices = stock_data.index[base_idx:needed_len]
                            else:
                                # Handle case where index is too short
                                available_len = len(stock_data.index) - base_idx
                                test_indices = stock_data.index[base_idx:base_idx + available_len]
                                # Trim y_test and y_pred to match
                                y_test_actual = y_test_actual[:available_len]
                                y_pred = y_pred[:available_len]
                            
                            # Ensure we have the right number of indices
                            if len(test_indices) > len(y_test_actual):
                                test_indices = test_indices[:len(y_test_actual)]
                            elif len(test_indices) < len(y_test_actual):
                                y_test_actual = y_test_actual[:len(test_indices)]
                                y_pred = y_pred[:len(test_indices)]
                        except Exception as e:
                            st.error(f"Error preparing test indices: {str(e)}")
                            # Fallback to a simpler approach
                            test_indices = stock_data.index[-len(y_test_actual):]
                            if len(test_indices) != len(y_test_actual):
                                # Adjust lengths to match
                                min_len = min(len(test_indices), len(y_test_actual))
                                test_indices = test_indices[:min_len]
                                y_test_actual = y_test_actual[:min_len]
                                y_pred = y_pred[:min_len]
                        
                        prediction_fig = predictor.plot_predictions(test_indices, y_test_actual, y_pred)
                        st.pyplot(prediction_fig)
                    except Exception as e:
                        st.error(f"Error plotting predictions: {str(e)}")
                        st.text(f"Debug info - test_indices: {len(test_indices)}, y_test_actual: {y_test_actual.shape}, y_pred: {y_pred.shape}")
            
            # Predict next day's price
            status_container.info("Predicting next trading day's price...")
            
            try:
                # Get the most recent time_step days of data
                # Note: we need to use the predictor's time_steps, not the generic time_step variable
                predictor_time_steps = getattr(predictor, 'time_steps', 60)  # Default to 60 if not found
                
                try:
                    last_sequence = preprocessed_data['scaled_data'][-predictor_time_steps:]
                except Exception as e1:
                    # Try alternate approach if the above fails
                    available_length = len(preprocessed_data['scaled_data'])
                    seq_length = min(60, available_length)  # Use at most 60 days or whatever is available
                    last_sequence = preprocessed_data['scaled_data'][-seq_length:]
                    st.warning(f"Using sequence length of {seq_length}: {str(e1)}")
                
                # Make sure we have a proper 2D array for LSTM input
                if len(last_sequence.shape) == 1:
                    last_sequence = last_sequence.reshape(-1, 1)
                
                # Predict next day
                next_day_prediction = predictor.predict_next_day(last_sequence)
                
                # Handle the conversion back to original price scale if log transform was used
                if transform_option == "Log Transform":
                    prediction_value = next_day_prediction['prediction']
                    # If we used log transform, we need to exponentiate to get the original price
                    prediction_value = np.exp(prediction_value)
                    lower_bound = np.exp(next_day_prediction['lower_bound'])
                    upper_bound = np.exp(next_day_prediction['upper_bound'])
                    next_day_prediction = {
                        'prediction': prediction_value,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    st.info("Converted logarithmic prediction back to original price scale.")
                
                # Get the last known price for comparison - always use the actual Close, not transformed
                last_known_price = float(stock_data['Close'].iloc[-1])
                price_change = next_day_prediction['prediction'] - last_known_price
                price_change_pct = (price_change / last_known_price) * 100
                
                # Determine direction (up or down)
                direction = "📈 UP" if price_change > 0 else "📉 DOWN"
                color = "green" if price_change > 0 else "red"
                
                # Display next day prediction
                with prediction_container:
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 2rem 0;">
                        <h2 style="margin-top: 0; color: #0f172a;">Next Trading Day Prediction</h2>
                        <div style="display: flex; gap: 1rem; margin: 1.5rem 0;">
                            <div style="flex: 1;">
                                <h3 style="margin-top: 0; color: #64748b;">Predicted Price</h3>
                                <h1 style="color: {color}; margin-bottom: 0;">${next_day_prediction['prediction']:.2f}</h1>
                                <p style="color: #64748b;">95% Confidence Interval: ${next_day_prediction['lower_bound']:.2f} - ${next_day_prediction['upper_bound']:.2f}</p>
                            </div>
                            <div style="flex: 1;">
                                <h3 style="margin-top: 0; color: #64748b;">Expected Change</h3>
                                <h1 style="color: {color}; margin-bottom: 0;">{direction} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</h1>
                            </div>
                        </div>
                        <p>Last known price: <strong>${last_known_price:.2f}</strong> on {stock_data.index[-1].strftime('%Y-%m-%d')}</p>
                        <p style="color: #64748b; font-style: italic;">Note: This prediction is based on historical patterns and should not be used as the sole basis for investment decisions.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                with prediction_container:
                    st.error(f"Error predicting next day price: {str(e)}")
                    st.code(f"Debug info:\nlatest_data shape: {preprocessed_data['scaled_data'][-time_step:].shape}\ntype: {type(preprocessed_data['scaled_data'][-time_step:])}")
            
            # Perform rolling window validation
            status_container.info("Performing rolling window validation...")
            
            try:
                rolling_metrics = predictor.rolling_window_validation(
                    preprocessed_data['scaled_data'], 
                    window_size=30
                )
                
                # Check if we got enough data points for plotting
                if len(rolling_metrics['maes']) > 1:
                    # Plot rolling metrics
                    with rolling_container:
                        st.subheader("Model Stability (Rolling Window Validation)")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot MAE and RMSE
                        ax.plot(rolling_metrics['maes'], label='MAE', color='blue')
                        ax.plot(rolling_metrics['rmses'], label='RMSE', color='red')
                        
                        ax.set_title('Model Stability Over Different Time Windows')
                        ax.set_xlabel('Window')
                        ax.set_ylabel('Error ($)')
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
                        
                        # Plot directional accuracy
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(rolling_metrics['dir_accuracies'], label='Directional Accuracy', color='green')
                        
                        ax.set_title('Directional Accuracy Over Different Time Windows')
                        ax.set_xlabel('Window')
                        ax.set_ylabel('Accuracy (%)')
                        ax.axhline(y=50, color='r', linestyle='--', label='Random Guess (50%)')
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
                else:
                    with rolling_container:
                        st.warning("Not enough data points for rolling window validation. Try using more historical data.")
            except Exception as e:
                with rolling_container:
                    st.error(f"Error performing rolling window validation: {str(e)}")
            
            # Final status update
            status_container.success("Analysis completed successfully!")
            
            # Option to save model
            if st.button("Save Model"):
                predictor.save_model("lstm_model_saved")
                st.success("Model saved successfully!")
        
        except Exception as e:
            status_container.error(f"An error occurred: {str(e)}")
            st.exception(e)

with tab2:
    st.header("Stock Market Forecasting Challenges & LSTM Solutions")
    
    # Challenge 1
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Highly Volatile & Non-Stationary Data</h3>
        <p>Stock prices exhibit high volatility and changing statistical properties over time, making prediction difficult.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Memory retention:</strong> LSTMs can remember important patterns over long sequences</li>
            <li><strong>Adaptive learning:</strong> The model can adapt to changing market conditions</li>
            <li><strong>Feature normalization:</strong> Input data scaling helps handle volatility</li>
            <li><strong>Rolling window validation:</strong> Our implementation tests model stability across different time periods</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 2
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Noise & Random Market Fluctuations</h3>
        <p>Markets are influenced by random events and noise, which can mislead prediction models.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Gating mechanisms:</strong> LSTM cells can selectively ignore irrelevant noise</li>
            <li><strong>Dropout layers:</strong> Help prevent the model from focusing on random patterns</li>
            <li><strong>Multi-feature input:</strong> Including volume and technical indicators provides context</li>
            <li><strong>BatchNormalization:</strong> Stabilizes learning in the presence of noisy data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 3
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Influence of Global Events & Sentiment</h3>
        <p>Stock markets are affected by news, global events, and investor sentiment that are hard to quantify.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Pattern recognition:</strong> LSTMs can identify how markets historically responded to disruptions</li>
            <li><strong>Emphasis on recent data:</strong> The model gives more weight to recent patterns reflecting current sentiment</li>
            <li><strong>Directional accuracy:</strong> We track if the model correctly predicts market direction, not just exact prices</li>
            <li><strong>Confidence intervals:</strong> Predictions include uncertainty ranges to account for unpredictable events</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 4
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Overfitting to Small Patterns</h3>
        <p>Models can become too specialized to historical patterns that don't repeat in the future.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Regularization:</strong> L2 regularization prevents weights from becoming too specialized</li>
            <li><strong>Dropout:</strong> Randomly turns off neurons during training to prevent co-adaptation</li>
            <li><strong>Validation monitoring:</strong> Early stopping if performance on validation data deteriorates</li>
            <li><strong>Batch normalization:</strong> Helps the model generalize better across different market conditions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 5
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Balancing Long-term Dependencies vs. Short-term Trends</h3>
        <p>Stock prices are influenced by both long-term fundamentals and short-term momentum.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Memory cells:</strong> LSTM's explicit memory can retain important information from distant past</li>
            <li><strong>Forget gates:</strong> Allow the model to discard irrelevant historical information</li>
            <li><strong>Multiple LSTM layers:</strong> Different layers can focus on different time horizons</li>
            <li><strong>Technical indicators:</strong> Including moving averages helps bridge different time scales</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 6
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Pattern Detection in Noisy Environments</h3>
        <p>Identifying meaningful patterns when they're obscured by random market movements.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Sequential processing:</strong> LSTMs process data in sequence, maintaining temporal relationships</li>
            <li><strong>Input gating:</strong> Can learn to focus on relevant signals and ignore noise</li>
            <li><strong>Multiple features:</strong> Using price, volume and technical indicators provides context</li>
            <li><strong>Evaluation metrics:</strong> Directional accuracy helps assess if the model captures meaningful patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Challenge 7
    st.markdown("""
    <div class="challenge-card">
        <h3>Challenge: Handling Missing Data & Irregular Time Intervals</h3>
        <p>Market holidays, trading halts, and other irregularities create gaps in time series data.</p>
        
        <h4>How LSTM Addresses This:</h4>
        <ul>
            <li><strong>Data preprocessing:</strong> Our implementation handles missing values automatically</li>
            <li><strong>Business day focus:</strong> The model is trained on trading days only, ignoring weekends/holidays</li>
            <li><strong>Sequence length:</strong> The time window parameter can be adjusted to handle irregular intervals</li>
            <li><strong>Feature engineering:</strong> Technical indicators can help bridge gaps in raw price data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Evaluation Metrics Explanation
    st.header("Understanding Evaluation Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Mean Absolute Error (MAE)</h3>
            <p>Average of absolute differences between predicted and actual values. Shows the typical error in dollars.</p>
            <p><strong>Why it matters:</strong> Easy to interpret and less sensitive to outliers.</p>
            
            <h3>Root Mean Squared Error (RMSE)</h3>
            <p>Square root of the average of squared differences. Penalizes large errors more heavily.</p>
            <p><strong>Why it matters:</strong> Good for detecting harmful large prediction errors.</p>
            
            <h3>Mean Absolute Percentage Error (MAPE)</h3>
            <p>Average of percentage differences between predicted and actual values.</p>
            <p><strong>Why it matters:</strong> Shows error relative to price, useful for comparing across different stocks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Directional Accuracy</h3>
            <p>Percentage of times the model correctly predicts the direction of price movement (up or down).</p>
            <p><strong>Why it matters:</strong> For many trading strategies, direction is more important than exact price.</p>
            
            <h3>R² Score</h3>
            <p>Indicates how much of the price variation is explained by the model (1.0 is perfect).</p>
            <p><strong>Why it matters:</strong> Shows if the model captures the underlying patterns or just predicts randomly.</p>
            
            <h3>Rolling Window Validation</h3>
            <p>Tests model on different time periods to check consistency of performance.</p>
            <p><strong>Why it matters:</strong> Reveals if model only works in specific market conditions or time periods.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # LSTM Architecture Explanation
    st.header("LSTM Architecture for Stock Prediction")
    
    st.markdown("""
    <div class="info-box">
        <h3>Why LSTM Works Better Than Simple Neural Networks</h3>
        <p>Long Short-Term Memory networks have a specialized architecture that gives them key advantages:</p>
        
        <h4>Memory Cell</h4>
        <p>The core component that can maintain information over long sequences, allowing the network to remember important patterns from weeks or months ago.</p>
        
        <h4>Input, Output and Forget Gates</h4>
        <p>These control what information enters the memory cell, what gets output, and what gets forgotten. This selective memory is crucial for financial data where some past events matter more than others.</p>
        
        <h4>Sequence Processing</h4>
        <p>LSTMs process data in sequence, preserving the time relationship between observations - critical for time series data like stock prices.</p>
        
        <h4>Multi-layer Design</h4>
        <p>Stacking LSTM layers allows the model to learn increasingly abstract features and patterns at different time scales.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Best Practices
    st.header("Best Practices for Stock Price Prediction")
    
    st.markdown("""
    <div class="info-box">
        <h3>Data Preparation</h3>
        <ul>
            <li><strong>Sufficient history:</strong> Use at least 2-5 years of data to capture various market conditions</li>
            <li><strong>Feature normalization:</strong> Always scale inputs to [0,1] or [-1,1] range</li>
            <li><strong>Feature engineering:</strong> Add technical indicators that traders actually use</li>
            <li><strong>Train/test split:</strong> Use recent data for testing to reflect current market dynamics</li>
        </ul>
        
        <h3>Model Configuration</h3>
        <ul>
            <li><strong>Time window size:</strong> Match to your prediction horizon (longer horizons need longer windows)</li>
            <li><strong>Layer size:</strong> More complex markets may need more units (50-100 is typical)</li>
            <li><strong>Regularization:</strong> Always use dropout (0.2-0.5) to prevent overfitting</li>
            <li><strong>Learning rate:</strong> Start small (0.001) and use learning rate reduction</li>
        </ul>
        
        <h3>Realistic Expectations</h3>
        <ul>
            <li><strong>Directional accuracy:</strong> A good model achieves 55-65% directional accuracy (50% is random)</li>
            <li><strong>Error magnitude:</strong> Expect errors of 1-3% in stable markets, higher in volatile ones</li>
            <li><strong>Horizon limitations:</strong> Accuracy decreases rapidly beyond 1-5 days</li>
            <li><strong>Market-specific:</strong> Models trained on one stock may not transfer well to others</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Run the app with: streamlit run lstm_app.py 