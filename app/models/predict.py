# app/models/predict.py

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# --- Configuration ---
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
PRICES_DIR = Path(__file__).resolve().parent.parent.parent / "data/structured/prices"

def predict_next_day_price(ticker: str) -> dict:
    """
    Predicts the next day's closing price for a given stock ticker.

    Args:
        ticker: The stock ticker (e.g., 'AAPL').

    Returns:
        A dictionary with the predicted price or an error message.
    """
    try:
        # Load the trained model and its required features
        model = joblib.load(MODEL_DIR / f"{ticker}_price_regressor.joblib")
        features_list = joblib.load(MODEL_DIR / f"{ticker}_features.joblib")

        # Load the latest historical data for the ticker
        df = pd.read_csv(PRICES_DIR / f"{ticker}_prices.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Take a slice of the last ~30 days to ensure rolling windows can be calculated
        latest_data = df.tail(30).copy()

        # 1. Lag Features
        for i in range(1, 11): # WINDOW_SIZE is 10
            latest_data[f'Close_lag_{i}'] = latest_data['Close'].shift(i)

        # 2. Rolling Window Features
        latest_data['MA_5'] = latest_data['Close'].rolling(window=5).mean()
        latest_data['MA_20'] = latest_data['Close'].rolling(window=20).mean()

        # 3. Volume-based Features
        latest_data['Volume_lag_1'] = latest_data['Volume'].shift(1)
        latest_data['Volume_MA_5'] = latest_data['Volume'].rolling(window=5).mean()

        # Get the very last row, which now contains all the features needed for prediction
        prediction_features = latest_data.tail(1)

        # Ensure the feature DataFrame has the correct columns in the correct order
        prediction_features = prediction_features[features_list]

        # Make the prediction
        predicted_price = model.predict(prediction_features)[0]

        return {
            "ticker": ticker,
            "predicted_next_day_close": round(float(predicted_price), 2)
        }

    except FileNotFoundError:
        return {"error": f"Model or data for ticker '{ticker}' not found. Please ensure it has been trained."}
    except Exception as e:
        return {"error": f"An error occurred during prediction for {ticker}: {e}"}

if __name__ == '__main__':
    sample_ticker = 'AAPL'
    prediction = predict_next_day_price(sample_ticker)

    if "error" in prediction:
        print(f"Error: {prediction['error']}")
    else:
        print(f"Prediction for {prediction['ticker']}:")
        print(f"  Predicted Close Price for Tomorrow: ${prediction['predicted_next_day_close']}")