
Model Training
•	The LSTM model is trained on scaled stock price data.
•	The training process involves multiple epochs and batch sizes to optimize performance.
•	The model performance is evaluated using various metrics:
o	Root Mean Square Error (RMSE)
o	Mean Squared Error (MSE)
o	R-squared (R²)
o	Mean Absolute Error (MAE)	



•	•  Epochs 30–80: Noticeable improvement in RMSE and R^2, but some fluctuations.
•	•  Batch 32 vs. 64: Smaller batches tend to slightly reduce error, as seen in the 80-epoch results.
•	•  100 Epochs: Shows best performance with RMSE = 95.1, MSE = 9044.6, and R2=0.9783.
