# train_predictor.py

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import argparse
import lightgbm as lgb
from tqdm import tqdm

# --- Paths (project-relative based on this file's location) ---
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # Go up 2 levels to reach Financial_ADK_Agent_Graph_Database
PRICES_DIR = PROJECT_ROOT / "data/structured/prices"
MODEL_DIR = THIS_FILE.parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Engineering Parameters ---
WINDOW_SIZE = 10         # How many past days of data to use as features
PREDICTION_HORIZON = 1   # How many days into the future to predict (1 = next day)

def create_features(df):
    """Creates time-series features from a stock price DataFrame."""
    # Create a new DataFrame for features to avoid modifying the original
    featured_df = df[['Close', 'Volume']].copy()

    # 1. Lag Features (autoregressive part)
    # Use the previous 'WINDOW_SIZE' days' closing prices as features
    for i in range(1, WINDOW_SIZE + 1):
        featured_df[f'Close_lag_{i}'] = featured_df['Close'].shift(i)

    # 2. Rolling Window Features
    # Create rolling averages to capture recent trends
    featured_df['MA_5'] = featured_df['Close'].rolling(window=5).mean()
    featured_df['MA_20'] = featured_df['Close'].rolling(window=20).mean()

    # 3. Volume-based Features
    featured_df['Volume_lag_1'] = featured_df['Volume'].shift(1)
    featured_df['Volume_MA_5'] = featured_df['Volume'].rolling(window=5).mean()

    # 4. Create the target variable
    # The 'target' is the closing price 'PREDICTION_HORIZON' days in the future
    featured_df['target'] = featured_df['Close'].shift(-PREDICTION_HORIZON)

    # Drop rows with NaN values created by shifts and rolling windows
    featured_df.dropna(inplace=True)

    return featured_df

def _train_ticker_model(ticker: str) -> None:
    price_file = PRICES_DIR / f"{ticker}_prices.csv"
    if not price_file.exists():
        print(f"Skipping {ticker}: prices file not found at {price_file}")
        return

    # Load data
    df = pd.read_csv(price_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Create features
    data = create_features(df)

    if data.empty:
        print(f"Skipping {ticker}: Not enough data to create features.")
        return

    # Define features (X) and target (y)
    X = data.drop(columns=['target', 'Close', 'Volume'])
    y = data['target']

    print(f"\nTraining model for {ticker} with {len(X.columns)} features...")
    model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31
    )
    model.fit(X, y)

    # Save the trained model and the list of features it expects
    joblib.dump(model, MODEL_DIR / f"{ticker}_price_regressor.joblib")
    joblib.dump(X.columns.tolist(), MODEL_DIR / f"{ticker}_features.joblib")
    print(f"✓ Model for {ticker} saved to {MODEL_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train next-day price regressors from price CSVs.")
    parser.add_argument("--tickers", nargs="*", help="Optional list of tickers to train. If omitted, trains for all CSVs in prices directory.")
    args = parser.parse_args()

    if not PRICES_DIR.exists():
        print(f"Prices directory not found: {PRICES_DIR}")
        return

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted({p.name.split('_')[0] for p in PRICES_DIR.glob("*_prices.csv")})

    if not tickers:
        print("No tickers found to train. Ensure price CSVs exist.")
        return

    for ticker in tqdm(tickers, desc="Training models"):
        _train_ticker_model(ticker)

    print("\nTraining complete! All models saved.")


if __name__ == "__main__":
    main()