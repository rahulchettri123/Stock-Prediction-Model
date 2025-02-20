import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ---- STEP 1: LOAD STOCK MARKET DATA ----
tickerSymbol = '^GSPC'  # S&P 500 Index
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-01-01', end='2025-01-01')

# Keep only relevant columns
tickerDf = tickerDf[['Close', 'Open', 'High', 'Low', 'Volume']]
tickerDf.dropna(inplace=True)

# ---- STEP 2: SCALE THE DATA ----
scaler = MinMaxScaler()
tickerDf_scaled = scaler.fit_transform(tickerDf)

# ---- STEP 3: CREATE SEQUENCES FOR LSTM ----
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

seq_length = 30  # Use past 30 days to predict next day
X, y = create_sequences(tickerDf_scaled, seq_length)

# ---- TRAIN-TEST SPLIT ----
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Ensure y_train and y_test have no NaN or Inf values
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

# Reshape for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# ---- STEP 4: BUILD LSTM MODEL USING FUNCTIONAL API ----
input_layer = Input(shape=(seq_length, X_train_lstm.shape[2]))
x = LSTM(64, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.1)(x)
output = Dense(1)(x)

model_lstm = Model(inputs=input_layer, outputs=output)
model_lstm.compile(optimizer='adam', loss='mse')

# ---- TRAIN LSTM ----
model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_data=(X_test_lstm, y_test), verbose=1)

# ---- FEATURE EXTRACTOR FOR XGBOOST ----
# Get the output from the Dense(16) layer (index -3)
feature_extractor = Model(inputs=model_lstm.input, outputs=model_lstm.layers[-3].output)

# Extract LSTM features for training XGBoost
X_train_xgb = feature_extractor.predict(X_train_lstm)
X_test_xgb = feature_extractor.predict(X_test_lstm)

# ---- STEP 5: TRAIN XGBOOST MODEL ON LSTM FEATURES ----
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_xgb, y_train)

# ---- STEP 6: PREDICT STOCK PRICES ----
y_pred_lstm = model_lstm.predict(X_test_lstm)
y_pred_xgb = xgb_model.predict(X_test_xgb)

# ---- RESCALE PREDICTIONS BACK TO ORIGINAL PRICE ----
y_pred_lstm = scaler.inverse_transform(np.column_stack([y_pred_lstm] * tickerDf.shape[1]))[:, 0]
y_pred_xgb = scaler.inverse_transform(np.column_stack([y_pred_xgb] * tickerDf.shape[1]))[:, 0]

# ---- STEP 7: SHAP EXPLAINABILITY FOR XGBOOST ----
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_xgb)

shap.summary_plot(shap_values, X_test_xgb)

# ---- STEP 8: EVALUATION METRICS ----
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(f"LSTM RMSE: {rmse(y_test, y_pred_lstm):.4f}")
print(f"XGBoost RMSE: {rmse(y_test, y_pred_xgb):.4f}")

# ---- STEP 9: VISUALIZATION ----
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(tickerDf_scaled[-len(y_test):, 0]), label='Actual Prices', color='black')
plt.plot(y_pred_lstm, label='LSTM Predictions', color='blue')
plt.plot(y_pred_xgb, label='XGBoost Predictions', color='red')
plt.legend()
plt.title("Stock Price Prediction: LSTM vs XGBoost")
plt.show()