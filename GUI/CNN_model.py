import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import streamlit_shadcn_ui as ui
from streamlit_shadcn_ui import button, card, tabs

# Set page config
st.set_page_config(page_title="Stock Price Prediction with CNN", layout="wide")

# Custom CSS for shadcn-inspired styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main {
        background-color: #f8fafc;
    }
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #0f172a;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1e293b;
    }
    .stProgress > div > div > div {
        background-color: #0ea5e9;
    }
    .sidebar .sidebar-content {
        background-color: #f1f5f9;
    }
    .alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }
    .alert-info {
        background-color: #f0f9ff;
        border-color: #bae6fd;
        color: #0c4a6e;
    }
    .alert-success {
        background-color: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    .alert-warning {
        background-color: #fffbeb;
        border-color: #fef3c7;
        color: #92400e;
    }
    .alert-error {
        background-color: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Custom alert component replacement
def custom_alert(title, description, variant="default"):
    variant_class = {
        "info": "alert-info",
        "success": "alert-success",
        "warning": "alert-warning",
        "destructive": "alert-error",
        "default": "alert-info"
    }.get(variant, "alert-info")
    
    return f"""
    <div class="alert {variant_class}">
        <h4 style="margin-top: 0;">{title}</h4>
        <p style="margin-bottom: 0;">{description}</p>
    </div>
    """

# App title
st.title("Stock Price Prediction with CNN and Pattern Detection")

# Create a simpler tab system that's more reliable
tab1, tab2 = st.tabs(["Prediction Model", "About"])

with tab1:
    # Sidebar for inputs
    with st.sidebar:
        st.header("Parameters")
        
        # Ticker input
        card(
            title="Ticker Symbol",
            content="""
            <div style="margin-bottom: 1rem;">
                Enter the ticker symbol for the stock you want to analyze
            </div>
            """
        )
        ticker = st.text_input("Ticker", "^GSPC", label_visibility="collapsed")
        
        # Date range selection
        card(
            title="Date Range",
            content="""
            <div style="margin-bottom: 1rem;">
                Select the date range for historical data
            </div>
            """
        )
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", datetime(2010, 1, 1))
        end_date = col2.date_input("End Date", datetime.today())
        
        # Time window input
        card(
            title="Model Parameters",
            content="""
            <div style="margin-bottom: 1rem;">
                Adjust the CNN model parameters
            </div>
            """
        )
        time_step = st.slider("Time Window", min_value=10, max_value=120, value=60, step=5, 
                             help="Number of previous days to use for prediction")
        
        # CNN parameters
        st.subheader("CNN Parameters")
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=50, step=5)
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        
        # Pattern detection threshold
        pattern_threshold = st.slider("Pattern Detection Sensitivity", 
                                     min_value=0.005, max_value=0.05, value=0.02, step=0.005,
                                     help="Lower values detect more patterns")
        
        # Add ensemble option
        st.subheader("Ensemble Options")
        use_ensemble = st.checkbox("Use Ensemble Approach", value=False, 
                                  help="Train multiple models and average predictions for better accuracy")
        
        if use_ensemble:
            num_models = st.slider("Number of Models", min_value=3, max_value=15, value=10, step=1,
                                 help="Number of models to train for ensemble prediction")
        else:
            num_models = 1
        
        # Train model button with explicit key
        st.write("")
        train_button = button("Train Model", key="train_button_unique", variant="default")

    # Main content
    st.markdown(custom_alert(
        title="Welcome to Stock Price Prediction",
        description="This app uses a CNN model to predict stock prices and detect technical patterns.",
        variant="default"
    ), unsafe_allow_html=True)

    # Function to create dataset
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    # Pattern detection function
    def detect_pattern(prices, peaks, dips, threshold=0.02):
        patterns = []
        
        # Ensure we have enough data points
        if len(peaks) < 3 or len(dips) < 3:
            return patterns
        
        for i in range(1, min(len(peaks), len(dips)) - 1):
            # Check if we're still within bounds
            if i-1 >= len(peaks) or i+1 >= len(peaks) or i-1 >= len(dips) or i+1 >= len(dips):
                continue
                
            left_peak, middle_peak, right_peak = peaks[i - 1], peaks[i], peaks[i + 1]
            left_dip, middle_dip, right_dip = dips[i - 1], dips[i], dips[i + 1]
            
            # Check bounds
            if (max(right_peak, right_dip) >= len(prices.index) or 
                max(left_peak, left_dip) >= len(prices.index) or
                max(middle_peak, middle_dip) >= len(prices.index)):
                continue
            
            left_peak_price = prices.iloc[left_peak]
            middle_peak_price = prices.iloc[middle_peak]
            right_peak_price = prices.iloc[right_peak]
            
            left_dip_price = prices.iloc[left_dip]
            middle_dip_price = prices.iloc[middle_dip]
            right_dip_price = prices.iloc[right_dip]
            
            # Head & Shoulders pattern
            if (left_peak_price < middle_peak_price and 
                right_peak_price < middle_peak_price and
                abs(left_peak_price - right_peak_price) < threshold * left_peak_price):
                patterns.append(("Head & Shoulders", middle_peak))
                
            # Inverted Head & Shoulders pattern
            if (left_dip_price > middle_dip_price and 
                right_dip_price > middle_dip_price and
                abs(left_dip_price - right_dip_price) < threshold * left_dip_price):
                patterns.append(("Inverted H&S", middle_dip))
                
            # Double Top pattern
            if abs(left_peak_price - right_peak_price) < threshold * left_peak_price:
                patterns.append(("Double Top", right_peak))
                
            # Double Bottom pattern
            if abs(left_dip_price - right_dip_price) < threshold * left_dip_price:
                patterns.append(("Double Bottom", right_dip))
        
        return patterns

    # Load data placeholder
    data_placeholder = st.empty()
    
    # Status placeholder - using custom alert
    status_container = st.empty()
    
    # Results containers
    col1, col2 = st.columns(2)
    training_plot_container = col1.empty()
    prediction_plot_container = col2.empty()

    metrics_container = st.empty()
    patterns_container = st.empty()

    # Next day prediction container
    next_day_container = st.empty()

    if train_button:
        try:
            # Update status
            status_container.markdown(custom_alert(
                title="Loading data...",
                description=f"Fetching historical data for {ticker}",
                variant="info"
            ), unsafe_allow_html=True)
            
            # Fetch stock data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                status_container.markdown(custom_alert(
                    title="Error",
                    description=f"No data found for ticker {ticker}. Please check the symbol and try again.",
                    variant="destructive"
                ), unsafe_allow_html=True)
            else:
                # Show data with latest dates first
                data_placeholder.write(data.sort_index(ascending=False).head(10))
                st.write(f"**Showing most recent data for {ticker}. Total records: {len(data)}**")
                
                # Create cards to display key stats
                latest_price = float(data['Close'].iloc[-1])
                high_price = float(data['High'].max())
                low_price = float(data['Low'].min())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    card(
                        title="Current Price",
                        content=f"<h2 style='color:#0ea5e9'>${latest_price:.2f}</h2>",
                    )
                with col2:
                    card(
                        title="Historical High",
                        content=f"<h2 style='color:#22c55e'>${high_price:.2f}</h2>",
                    )
                with col3:
                    card(
                        title="Historical Low",
                        content=f"<h2 style='color:#ef4444'>${low_price:.2f}</h2>",
                    )
                
                # Update progress status
                status_container.markdown(custom_alert(
                    title="Preprocessing data...",
                    description="Preparing data for model training",
                    variant="info"
                ), unsafe_allow_html=True)
                
                # Prepare data
                close_data = data[['Close']].dropna()
                
                # Normalize data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_data)
                
                # Create dataset
                X, y = create_dataset(scaled_data, time_step)
                
                # Reshape for CNN
                X = X.reshape(X.shape[0], X.shape[1], 1)
                
                # Split data
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Update status
                status_container.markdown(custom_alert(
                    title="Building model...",
                    description="Creating neural network architecture" + (f" for {num_models} ensemble models" if num_models > 1 else ""),
                    variant="info"
                ), unsafe_allow_html=True)
                
                # Function to create and train a CNN model
                def create_and_train_model(X_train, y_train, X_test, y_test):
                    # Define CNN model
                    model = Sequential([
                        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), 
                               input_shape=(time_step, 1)),
                        BatchNormalization(),
                        Dropout(0.3),
                        
                        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
                        BatchNormalization(),
                        Dropout(0.3),
                        
                        Flatten(),
                        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
                        Dropout(0.5),
                        
                        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
                        Dropout(0.5),
                        
                        Dense(1)
                    ])
                    
                    # Compile model
                    model.compile(optimizer='adam', loss='mean_absolute_error')
                    
                    # Callbacks
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=0,
                        callbacks=[reduce_lr]
                    )
                    
                    return model, history
                
                # Initialize progress tracking
                if num_models > 1:
                    ensemble_progress = st.progress(0)
                    ensemble_status = st.empty()
                
                # Train model(s)
                if num_models == 1:
                    # Single model approach
                    status_container.markdown(custom_alert(
                        title="Training model...",
                        description="This might take a while.",
                        variant="info"
                    ), unsafe_allow_html=True)
                    
                    # Use standard Streamlit progress bar
                    progress_bar = st.progress(0)
                    
                    # Custom callback to update progress bar
                    class ProgressCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress_bar.progress((epoch + 1) / epochs)
                    
                    # Define CNN model
                    cnn_model = Sequential([
                        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), 
                               input_shape=(time_step, 1)),
                        BatchNormalization(),
                        Dropout(0.3),
                        
                        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
                        BatchNormalization(),
                        Dropout(0.3),
                        
                        Flatten(),
                        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
                        Dropout(0.5),
                        
                        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
                        Dropout(0.5),
                        
                        Dense(1)
                    ])
                    
                    # Compile model
                    cnn_model.compile(optimizer='adam', loss='mean_absolute_error')
                    
                    # Callbacks
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
                    
                    # Train model
                    history = cnn_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=0,
                        callbacks=[reduce_lr, ProgressCallback()]
                    )
                    
                    # Plot training loss
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training and Validation Loss')
                    ax1.legend()
                    ax1.grid(True)
                    training_plot_container.pyplot(fig1)
                    
                    # Update status
                    status_container.markdown(custom_alert(
                        title="Making predictions...",
                        description="Generating predictions from trained model",
                        variant="info"
                    ), unsafe_allow_html=True)
                    
                    # Make predictions
                    y_pred_scaled = cnn_model.predict(X_test)
                    
                    # Convert back to original scale
                    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
                    
                else:
                    # Ensemble approach
                    models = []
                    all_histories = []
                    all_predictions = []
                    
                    for i in range(num_models):
                        ensemble_status.markdown(f"Training model {i+1} of {num_models}...")
                        
                        # Train a model
                        model, history = create_and_train_model(X_train, y_train, X_test, y_test)
                        models.append(model)
                        all_histories.append(history)
                        
                        # Make prediction with this model
                        y_pred_scaled = model.predict(X_test)
                        all_predictions.append(y_pred_scaled)
                        
                        # Update progress
                        ensemble_progress.progress((i + 1) / num_models)
                    
                    # Plot average training loss
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Plot individual model losses in light colors
                    for i, history in enumerate(all_histories):
                        ax1.plot(history.history['loss'], alpha=0.2, color='blue')
                        ax1.plot(history.history['val_loss'], alpha=0.2, color='orange')
                    
                    # Calculate and plot average losses
                    min_length = min([len(h.history['loss']) for h in all_histories])
                    avg_train_loss = np.mean([[h.history['loss'][i] for h in all_histories] for i in range(min_length)], axis=1)
                    avg_val_loss = np.mean([[h.history['val_loss'][i] for h in all_histories] for i in range(min_length)], axis=1)
                    
                    ax1.plot(avg_train_loss, label='Avg Training Loss', linewidth=2, color='blue')
                    ax1.plot(avg_val_loss, label='Avg Validation Loss', linewidth=2, color='orange')
                    
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Loss')
                    ax1.set_title(f'Average Training Loss Across {num_models} Models')
                    ax1.legend()
                    ax1.grid(True)
                    training_plot_container.pyplot(fig1)
                    
                    # Average the predictions
                    y_pred_scaled = np.mean(all_predictions, axis=0)
                    
                    # Store ensemble for later use
                    cnn_model = models[0]  # Use first model for saving purposes
                    
                    # Clear progress indicators
                    ensemble_progress.empty()
                    ensemble_status.empty()
                    
                    # Update status
                    status_container.markdown(custom_alert(
                        title="Ensemble prediction complete",
                        description=f"Successfully averaged predictions from {num_models} models",
                        variant="success"
                    ), unsafe_allow_html=True)
                
                # Convert back to original scale
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                cnn_mae = mean_absolute_error(y_test_actual, y_pred)
                
                # Display metrics with shadcn-style cards
                metrics_container.markdown("""
                <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
                    <div style="flex: 1; background-color: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin-top: 0; color: #64748b;">Mean Absolute Error</h4>
                        <h2 style="color: #0ea5e9; margin-bottom: 0;">$""" + f"{cnn_mae:.2f}" + """</h2>
                    </div>
                    <div style="flex: 1; background-color: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin-top: 0; color: #64748b;">Test Data Points</h4>
                        <h2 style="color: #6366f1; margin-bottom: 0;">""" + f"{len(y_test)}" + """</h2>
                    </div>""" + 
                    ("""
                    <div style="flex: 1; background-color: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin-top: 0; color: #64748b;">Ensemble Size</h4>
                        <h2 style="color: #22c55e; margin-bottom: 0;">""" + f"{num_models} models" + """</h2>
                    </div>""" if num_models > 1 else "") + """
                </div>
                """, unsafe_allow_html=True)
                
                # Update status for pattern detection
                status_container.markdown(custom_alert(
                    title="Detecting patterns...",
                    description="Identifying technical chart patterns",
                    variant="info"
                ), unsafe_allow_html=True)
                
                # Get the date index for test data
                test_dates = close_data.index[-len(y_test):]
                prices = pd.Series(y_test_actual.flatten(), index=test_dates)
                
                # Find peaks and dips
                peaks = argrelextrema(prices.values, np.greater, order=5)[0]
                dips = argrelextrema(prices.values, np.less, order=5)[0]
                
                # Detect patterns
                patterns = detect_pattern(prices, peaks, dips, threshold=pattern_threshold)
                
                # Plot predictions
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(prices.index, prices, label="Actual Prices", color='blue', linewidth=2)
                ax2.plot(prices.index, y_pred.flatten(), label="CNN Predictions", color='red', linewidth=2)
                
                # Pattern styles
                pattern_styles = {
                    "Double Bottom": ('^', 'purple'),
                    "Head & Shoulders": ('s', 'red'),
                    "Inverted H&S": ('D', 'blue'),
                    "Double Top": ('o', 'green'),
                }
                
                # Plot patterns
                pattern_count = {}
                for pattern_type, marker_idx in patterns:
                    if pattern_type not in pattern_count:
                        pattern_count[pattern_type] = 0
                    pattern_count[pattern_type] += 1
                    
                    symbol, color = pattern_styles.get(pattern_type, ('*', 'black'))
                    ax2.scatter(prices.index[marker_idx], prices.iloc[marker_idx],
                               marker=symbol, color=color, s=100, edgecolors='black',
                               label=f"{pattern_type}" if pattern_type not in ax2.get_legend_handles_labels()[1] else "")
                
                ax2.set_title("CNN Predictions vs Actual Prices with Pattern Detection")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Price")
                ax2.legend(loc="upper left")
                ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
                prediction_plot_container.pyplot(fig2)
                
                # Display pattern summary with shadcn styling
                if patterns:
                    patterns_df = pd.DataFrame([(p[0], prices.index[p[1]].strftime('%Y-%m-%d'), prices.iloc[p[1]]) 
                                            for p in patterns], 
                                            columns=['Pattern', 'Date', 'Price'])
                    
                    patterns_container.markdown("### Detected Patterns")
                    patterns_container.dataframe(patterns_df.sort_values('Date', ascending=False))
                    
                    # Pattern badge display using badges (plural) instead of badge
                    st.markdown("<h4>Pattern Types</h4>", unsafe_allow_html=True)
                    
                    # Create badges for pattern types using the correct badges function
                    badge_list = []
                    for pattern, count in pattern_count.items():
                        variant = "blue" if pattern == "Inverted H&S" else "red" if pattern == "Head & Shoulders" else "green" if pattern == "Double Top" else "purple"
                        badge_list.append((f"{pattern}: {count}", variant))
                    
                    # Make sure to use a unique key for badges
                    ui.badges(badge_list=badge_list, class_name="flex gap-2", key=f"pattern_badges_{time.time()}")
                else:
                    patterns_container.markdown(custom_alert(
                        title="No patterns detected",
                        description="Try adjusting the sensitivity threshold.",
                        variant="warning"
                    ), unsafe_allow_html=True)
                
                # Predict next day's closing price
                status_container.markdown(custom_alert(
                    title="Predicting next day's closing price...",
                    description="Using " + ("ensemble of models" if num_models > 1 else "trained model") + " to predict next trading day",
                    variant="info"
                ), unsafe_allow_html=True)
                
                # Get the most recent time_step days of data
                last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
                
                if num_models == 1:
                    # Single model prediction
                    next_day_scaled = cnn_model.predict(last_sequence)
                    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
                else:
                    # Ensemble prediction
                    ensemble_predictions = []
                    for model in models:
                        pred = model.predict(last_sequence)
                        ensemble_predictions.append(pred)
                    
                    # Average the predictions
                    next_day_scaled = np.mean(ensemble_predictions, axis=0)
                    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
                    
                    # Calculate prediction variance for confidence interval
                    individual_prices = [scaler.inverse_transform(pred)[0][0] for pred in ensemble_predictions]
                    price_std = np.std(individual_prices)
                    price_min = next_day_price - 1.96 * price_std
                    price_max = next_day_price + 1.96 * price_std
                
                # Get the last known price for comparison
                last_known_price = close_data.iloc[-1][0]
                price_change = next_day_price - last_known_price
                price_change_pct = (price_change / last_known_price) * 100
                
                # Determine direction (up or down)
                direction = "📈 UP" if price_change > 0 else "📉 DOWN"
                color = "green" if price_change > 0 else "red"
                
                # Display next day prediction with shadcn styling
                if num_models == 1:
                    # Single model prediction display
                    next_day_container.markdown(f"""
                    <div style="margin-top: 2rem; margin-bottom: 2rem;">
                        <div style="background-color: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <h2 style="margin-top: 0; color: #0f172a;">Next Trading Day Prediction</h2>
                            <div style="display: flex; gap: 1rem; margin: 1.5rem 0;">
                                <div style="flex: 1;">
                                    <h3 style="margin-top: 0; color: #64748b;">Predicted Price</h3>
                                    <h1 style="color: {color}; margin-bottom: 0;">${next_day_price:.2f}</h1>
                                </div>
                                <div style="flex: 1;">
                                    <h3 style="margin-top: 0; color: #64748b;">Change</h3>
                                    <h1 style="color: {color}; margin-bottom: 0;">{direction} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</h1>
                                </div>
                            </div>
                            <p>Last known price: <strong>${last_known_price:.2f}</strong> on {close_data.index[-1].strftime('%Y-%m-%d')}</p>
                            <p style="color: #64748b; font-style: italic;">Note: This prediction is based on historical patterns and should not be used for investment decisions.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Ensemble prediction display with confidence interval
                    next_day_container.markdown(f"""
                    <div style="margin-top: 2rem; margin-bottom: 2rem;">
                        <div style="background-color: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <h2 style="margin-top: 0; color: #0f172a;">Next Trading Day Prediction (Ensemble of {num_models} Models)</h2>
                            <div style="display: flex; gap: 1rem; margin: 1.5rem 0;">
                                <div style="flex: 1;">
                                    <h3 style="margin-top: 0; color: #64748b;">Predicted Price</h3>
                                    <h1 style="color: {color}; margin-bottom: 0;">${next_day_price:.2f}</h1>
                                    <p style="color: #64748b;">95% Confidence: $[{price_min:.2f} - {price_max:.2f}]</p>
                                </div>
                                <div style="flex: 1;">
                                    <h3 style="margin-top: 0; color: #64748b;">Change</h3>
                                    <h1 style="color: {color}; margin-bottom: 0;">{direction} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</h1>
                                </div>
                            </div>
                            <p>Last known price: <strong>${last_known_price:.2f}</strong> on {close_data.index[-1].strftime('%Y-%m-%d')}</p>
                            <p style="color: #64748b; font-style: italic;">Note: Ensemble prediction averages results from {num_models} independently trained models for improved reliability.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Final status
                status_container.markdown(custom_alert(
                    title="Analysis completed!",
                    description="Model training and prediction successful",
                    variant="success"
                ), unsafe_allow_html=True)
                
                # Save model button with explicit key
                st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
                save_button = button("Save Model", key=f"save_model_{time.time()}", variant="default")
                if save_button:
                    # Save the model
                    model_dir = "saved_model"
                    cnn_model.save(model_dir)
                    # Save the scaler
                    import joblib
                    joblib.dump(scaler, "scaler.save")
                    st.markdown(custom_alert(
                        title="Model Saved",
                        description="The model has been saved and can be used with pretrained_model_predictor.py",
                        variant="success"
                    ), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            status_container.markdown(custom_alert(
                title="An error occurred",
                description=str(e),
                variant="destructive"
            ), unsafe_allow_html=True)
            st.exception(e)

with tab2:
    # About section with shadcn styling
    card(
        title="About This App",
        content="""
        <div style="margin-bottom: 1rem;">
            <p>This app uses a Convolutional Neural Network (CNN) to predict stock prices based on historical data.
            It also identifies common technical chart patterns like Head & Shoulders, Double Top, and Double Bottom.</p>
            
            <h3>Key Features</h3>
            <ul>
                <li>Load any stock ticker with customizable date range</li>
                <li>Train a CNN model with adjustable parameters</li>
                <li>Use ensemble learning to average predictions from multiple models</li>
                <li>Visualize predictions vs actual prices</li>
                <li>Detect and mark technical patterns</li>
                <li>Predict next day's closing price with confidence intervals</li>
            </ul>
            
            <h3>How to Use</h3>
            <ol>
                <li>Enter a ticker symbol (default is "^GSPC" for S&P 500)</li>
                <li>Select date range and time window</li>
                <li>Adjust CNN parameters if needed</li>
                <li>Click "Train Model" to start the analysis</li>
                <li>View results including model metrics and detected patterns</li>
            </ol>
            
            <h3>Requirements</h3>
            <ul>
                <li><strong>Python Packages:</strong>
                    <ul>
                        <li>streamlit (UI framework)</li>
                        <li>streamlit-shadcn-ui (for modern UI components)</li>
                        <li>numpy, pandas (data manipulation)</li>
                        <li>matplotlib, plotly (visualization)</li>
                        <li>yfinance (stock data retrieval)</li>
                        <li>scikit-learn (preprocessing and metrics)</li>
                        <li>tensorflow (neural network framework)</li>
                        <li>scipy (for technical pattern detection)</li>
                    </ul>
                </li>
                <li><strong>Installation:</strong> <code>pip install streamlit streamlit-shadcn-ui yfinance tensorflow scikit-learn pandas numpy matplotlib plotly scipy</code></li>
                <li><strong>System Requirements:</strong> 
                    <ul>
                        <li>RAM: At least 4GB (8GB+ recommended)</li>
                        <li>CPU: Modern multi-core processor</li>
                        <li>Storage: 1GB free space for model storage</li>
                        <li>Internet connection (to fetch stock data)</li>
                    </ul>
                </li>
            </ul>

            <h3>Understanding the Model</h3>
            <ul>
                <li><strong>Time Window:</strong> Number of previous days used to predict the next day's price. Larger windows may capture longer-term patterns but require more data.</li>
                <li><strong>Epochs:</strong> Number of training iterations. More epochs can improve accuracy but may lead to overfitting.</li>
                <li><strong>Batch Size:</strong> Number of samples processed before updating model weights. Smaller batches often provide better accuracy but slower training.</li>
                <li><strong>Validation Split:</strong> Portion of data reserved for validation during training (not used for training).</li>
                <li><strong>Pattern Detection Sensitivity:</strong> Lower values detect more patterns but may include false positives.</li>
                <li><strong>Ensemble Approach:</strong> Training multiple models and averaging their predictions to reduce variance and improve prediction stability.</li>
            </ul>
            
            <h3>Model Architecture</h3>
            <ul>
                <li><strong>Input Layer:</strong> 1D CNN with 128 filters and kernel size 3</li>
                <li><strong>Hidden Layers:</strong> 
                    <ul>
                        <li>BatchNormalization for stability</li>
                        <li>Dropout layers (0.3, 0.5) to prevent overfitting</li>
                        <li>Additional Conv1D layer with 64 filters</li>
                        <li>Dense layers (128, 64 nodes) with L2 regularization</li>
                    </ul>
                </li>
                <li><strong>Output Layer:</strong> Single node for price prediction</li>
                <li><strong>Training Features:</strong> Early stopping and learning rate reduction to optimize training</li>
            </ul>

            <h3>Interpreting Results</h3>
            <ul>
                <li><strong>Mean Absolute Error (MAE):</strong> Average absolute difference between predicted and actual prices. Lower values indicate better performance.</li>
                <li><strong>Technical Patterns:</strong> Visual markers on the chart where the algorithm detected potential trading patterns.</li>
                <li><strong>Next Day Prediction:</strong> Forecast for the next trading day based on the most recent data window.</li>
            </ul>
            
            <h3>Limitations</h3>
            <ul>
                <li>Stock prices are influenced by many factors beyond historical prices (news, market sentiment, economic conditions)</li>
                <li>The model cannot predict unexpected events or market shocks</li>
                <li>Performance varies by ticker and market conditions</li>
                <li>This is an educational tool and should not be the sole basis for investment decisions</li>
            </ul>
        </div>
        """
    )
    
    # Additional information cards
    col1, col2 = st.columns(2)
    with col1:
        card(
            title="Technical Patterns Explained",
            content="""
            <div>
                <h4>Head & Shoulders</h4>
                <p>A bearish reversal pattern with three peaks, the middle being the highest. Often signals a trend reversal from bullish to bearish.</p>
                
                <h4>Inverted Head & Shoulders</h4>
                <p>A bullish reversal pattern with three troughs, the middle being the lowest. Typically signals a trend reversal from bearish to bullish.</p>
            </div>
            """
        )
    with col2:
        card(
            title="More Patterns",
            content="""
            <div>
                <h4>Double Top</h4>
                <p>A bearish reversal pattern with two peaks at approximately the same level. Indicates resistance and potential downward movement.</p>
                
                <h4>Double Bottom</h4>
                <p>A bullish reversal pattern with two troughs at approximately the same level. Suggests support and potential upward movement.</p>
            </div>
            """
        )
        
    # Pre-trained model information
    card(
        title="Using Pre-Trained Models",
        content="""
        <div>
            <h4>Saving and Loading Models</h4>
            <ol>
                <li>Train a model with your desired parameters</li>
                <li>Click "Save Model" button after training</li>
                <li>The model will be saved to the "saved_model" directory</li>
                <li>The scaler will be saved as "scaler.save"</li>
            </ol>
            
            <h4>Using Pre-trained Models</h4>
            <p>To use a pre-trained model without retraining:</p>
            <ol>
                <li>Ensure you have the saved model files</li>
                <li>Use the pretrained_model_predictor.py script</li>
                <li>Run: <code>streamlit run pretrained_model_predictor.py</code></li>
            </ol>
            
            <h4>Advantages of Pre-trained Models</h4>
            <ul>
                <li>Faster predictions without training overhead</li>
                <li>Consistent predictions across multiple tickers</li>
                <li>Ability to test model on different time periods</li>
            </ul>
            
            <h4>When to Retrain</h4>
            <p>Consider retraining your model when:</p>
            <ul>
                <li>Market conditions change significantly</li>
                <li>You want to optimize for a specific ticker</li>
                <li>You want to test different model architectures or parameters</li>
                <li>Current model shows consistently high prediction errors</li>
            </ul>
        </div>
        """
    )
    
    # Troubleshooting card
    card(
        title="Troubleshooting",
        content="""
        <div>
            <h4>Common Issues</h4>
            <ul>
                <li><strong>No data found for ticker:</strong> Verify the ticker symbol is correct and data is available for the selected date range</li>
                <li><strong>High prediction error:</strong> Try adjusting the time window or increasing epochs</li>
                <li><strong>No patterns detected:</strong> Decrease the pattern detection sensitivity threshold</li>
                <li><strong>Model training too slow:</strong> Reduce the number of epochs or increase batch size</li>
                <li><strong>Out of memory errors:</strong> Reduce batch size or use a smaller date range</li>
            </ul>
            
            <h4>Performance Tips</h4>
            <ul>
                <li>For faster training, use a smaller date range or larger batch size</li>
                <li>For better accuracy, use a longer date range and more epochs</li>
                <li>Different tickers may require different parameters for optimal results</li>
                <li>Major market indices (like S&P 500) often give more reliable predictions than individual stocks</li>
            </ul>
        </div>
        """
    )
    
    # Add a new card about ensemble learning
    card(
        title="Ensemble Learning Benefits",
        content="""
        <div>
            <p>The app now features ensemble learning, which provides several benefits:</p>
            
            <h4>Reduced Variance</h4>
            <p>By training multiple models and averaging their predictions, we reduce the impact of individual model quirks and random initialization effects.</p>
            
            <h4>Improved Stability</h4>
            <p>Ensemble models produce more stable predictions that are less likely to be affected by data noise or outliers.</p>
            
            <h4>Confidence Intervals</h4>
            <p>When using the ensemble approach, the app calculates a 95% confidence interval for predictions, giving you better insight into the uncertainty of the forecast.</p>
            
            <h4>Better Performance</h4>
            <p>Research shows that ensemble methods typically outperform individual models in most prediction tasks, including financial forecasting.</p>
            
            <h4>When to Use Ensembles</h4>
            <ul>
                <li>For more important predictions where stability matters</li>
                <li>When you need confidence intervals around your prediction</li>
                <li>For longer-term forecasting where small errors can compound</li>
                <li>To reduce the effect of model sensitivity to random initialization</li>
            </ul>
        </div>
        """
    ) 