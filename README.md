# 📈 Stock Prediction Model: CNN-LSTM, XGBoost & ARIMA

## **🔍 Project Overview**
This project is a **stock market prediction system** that leverages **deep learning (CNN-LSTM, LSTM-XGBoost Hybrid)** and **machine learning (XGBoost, ARIMA)** to forecast stock prices based on **historical market data**.

📌 **Key Features:**
- Loads **real-time stock data** from **Yahoo Finance (`yfinance`)**
- Uses a **CNN+LSTM hybrid model** for time series forecasting
- Implements a **LSTM + XGBoost hybrid model** for improved interpretability
- Applies **SHAP explainability** to interpret model predictions
- Optimized for **speed and accuracy** with hyperparameter tuning (Optuna)

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

## **⚙️ Installation & Setup**
Follow these steps to **clone the repository** and **run the stock prediction model**.

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/rahulchettri123/Stock-Prediction-Model.git
cd Stock-Prediction-Model
