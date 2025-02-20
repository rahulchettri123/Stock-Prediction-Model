# 📈 Stock Prediction Model: CNN-LSTM, XGBoost & ARIMA

## **🔍 Project Overview**
This project is a **stock market prediction system** that leverages **deep learning (CNN-LSTM, LSTM-XGBoost Hybrid)** and **machine learning (XGBoost, ARIMA)** to forecast stock prices based on **historical market data**.

📌 **Key Features:**
- Loads **real-time stock data** from **Yahoo Finance (`yfinance`)**
- Uses a **CNN+LSTM hybrid model** for time series forecasting
- Implements a **LSTM + XGBoost hybrid model** for improved interpretability
- Applies **SHAP explainability** to interpret model predictions
- Optimized for **speed and accuracy** with hyperparameter tuning (Optuna)
- **NEW:** 📊 **Pattern Detection for Trading Signals**
- **NEW:** 🎯 **Comparison of CNN Model vs Actual Price Trends**

🔗 **GitHub Repository:**  
[Stock Prediction Model](https://github.com/rahulchettri123/Stock-Prediction-Model)

---

## **📊 Dataset Details**
The model fetches real-time stock market data from **Yahoo Finance** using `yfinance`.

**✅ Dataset Source:**  
[Yahoo Finance](https://finance.yahoo.com/)

**✅ Features Used:**
- **Close Price** (Target Variable) 📉
- Open, High, Low Prices  
- Trading Volume 📊

**✅ Time Range:**  
- **Start:** January 1, 2010  
- **End:** January 1, 2025  

📌 **The dataset is dynamically fetched** in real-time, so no static dataset is required.

---

## **🧠 Stock Price Prediction Models**
This project implements the following **forecasting models**:

### **1️⃣ Hybrid CNN + LSTM Model**
- Uses **1D Convolutional Neural Networks (CNN)** to detect short-term price movement patterns.
- **LSTM (Long Short-Term Memory)** layers capture long-term dependencies in stock price movements.
- **Sliding Window (30-day) input sequence** for better learning.
- **Dropout Regularization** to prevent overfitting.

### **2️⃣ LSTM + XGBoost Hybrid Model**
- Combines **LSTM for sequence learning** and **XGBoost for boosting predictions**.
- LSTM extracts features from stock prices.
- XGBoost **fine-tunes predictions** based on LSTM output.
- **Explainability with SHAP** to understand key influences.

### **3️⃣ ARIMA Model for Baseline Forecasting**
- **Autoregressive Integrated Moving Average (ARIMA)** used as a benchmark.
- Provides a **statistical approach** to predict future prices.

---

## **📈 Trading Pattern Detection**
### **📌 New Feature: Technical Pattern Recognition**
This model **detects trading patterns** to provide **better trading insights**.

📌 **Patterns Detected:**
- 📉 **Head & Shoulders** (Bearish Reversal)
- 📈 **Inverted Head & Shoulders** (Bullish Reversal)
- 🔴 **Double Top** (Bearish Signal)
- 🟢 **Double Bottom** (Bullish Signal)

### **🖼️ Visualization**
- **Plots actual stock prices** and overlays detected **trading patterns** with unique markers.
- **Compares detected patterns on actual price vs CNN predictions**.
- **Legend for pattern symbols** in visualizations.

📌 **Example Visualizations:**
- **Stock Price with Detected Patterns**
- **CNN Model Predictions vs Actual**
- **CNN Model with Trading Patterns Overlayed**

---

## **⚙️ Installation & Setup**
Follow these steps to **clone the repository** and **run the stock prediction model**.

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/rahulchettri123/Stock-Prediction-Model.git
cd Stock-Prediction-Model
