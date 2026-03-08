"""
Task 4: Prediction/Forecast Script
───────────────────────────────────
End-to-end pipeline that:
1. Fetches a time series record from the API
2. Preprocesses the data (same pipeline as Task 1)
3. Loads the trained model
4. Makes a prediction/forecast
"""

import requests
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
# scripts/ is one level below project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
API_BASE_URL = "http://localhost:5000"
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_list.pkl")


def fetch_recent_records(hours=48):
    """
    Step 1: Fetch time series records from the API.
    We need 48 hours of history to compute lag features and moving averages.
    """
    print("=" * 60)
    print("STEP 1: Fetching data from API")
    print("=" * 60)
    
    # Get the latest record to know the current time
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sql/readings/latest", timeout=5)
        resp.raise_for_status()
        latest = resp.json()
        latest_dt = pd.Timestamp(latest["datetime"])
        print(f"  Latest record: {latest_dt}")
    except requests.exceptions.ConnectionError:
        print("  ⚠️  API not running. Using fallback: reading directly from DB.")
        return fetch_from_db_directly(hours)
    
    # Fetch records from date range
    start_dt = (latest_dt - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    end_dt = latest_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    resp = requests.get(
        f"{API_BASE_URL}/api/sql/readings/daterange",
        params={"start": start_dt, "end": end_dt}
    )
    records = resp.json()
    print(f"  Fetched {len(records)} records from {start_dt} to {end_dt}")
    
    return records, latest


def fetch_from_db_directly(hours=48):
    """Fallback: Read directly from SQLite if API is not running."""
    import sqlite3
    DB_PATH = os.path.join(ROOT_DIR, "data", "airquality.db")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get latest record
    latest = dict(conn.execute("""
        SELECT r.*, m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        ORDER BY r.datetime DESC LIMIT 1
    """).fetchone())
    
    latest_dt = pd.Timestamp(latest["datetime"])
    start_dt = (latest_dt - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    
    rows = conn.execute("""
        SELECT r.reading_id, r.datetime, r.co_gt, r.co_sensor, r.benzene_gt,
               r.nmhc_sensor, r.nox_gt, r.nox_sensor, r.no2_gt, r.no2_sensor, r.o3_sensor,
               m.temperature, m.rel_humidity, m.abs_humidity
        FROM readings r LEFT JOIN meteorology m ON r.reading_id = m.reading_id
        WHERE r.datetime BETWEEN ? AND ?
        ORDER BY r.datetime
    """, (start_dt, latest_dt.strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
    
    records = [dict(row) for row in rows]
    conn.close()
    print(f"  [DB Fallback] Fetched {len(records)} records")
    return records, latest


def preprocess_data(records):
    """
    Step 2: Preprocess the data using the same pipeline as Task 1.
    - Convert to DataFrame with datetime index
    - Rename columns to match training format
    - Create lagged features
    - Create moving averages
    - Create temporal features
    """
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing data")
    print("=" * 60)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    df["DateTime"] = pd.to_datetime(df["datetime"])
    df.set_index("DateTime", inplace=True)
    df.sort_index(inplace=True)
    
    # Rename columns to match the training feature names
    column_map = {
        "co_gt": "CO(GT)",
        "co_sensor": "PT08.S1(CO)",
        "benzene_gt": "C6H6(GT)",
        "nmhc_sensor": "PT08.S2(NMHC)",
        "nox_gt": "NOx(GT)",
        "nox_sensor": "PT08.S3(NOx)",
        "no2_gt": "NO2(GT)",
        "no2_sensor": "PT08.S4(NO2)",
        "o3_sensor": "PT08.S5(O3)",
        "temperature": "T",
        "rel_humidity": "RH",
        "abs_humidity": "AH"
    }
    df.rename(columns=column_map, inplace=True)
    
    # Keep only the columns we need
    cols_to_keep = list(column_map.values())
    df = df[[c for c in cols_to_keep if c in df.columns]]
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Time range: {df.index.min()} → {df.index.max()}")
    
    # Create lagged features (same as Task 1)
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"CO_lag_{lag}"] = df["CO(GT)"].shift(lag)
    
    # Moving averages (shifted by 1 to avoid data leakage)
    df["CO_MA_6h"] = df["CO(GT)"].rolling(window=6, min_periods=1).mean().shift(1)
    df["CO_MA_24h"] = df["CO(GT)"].rolling(window=24, min_periods=1).mean().shift(1)
    
    # Temporal features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Drop rows with NaN (from lagging)
    df.dropna(inplace=True)
    
    print(f"  After feature engineering: {df.shape}")
    print(f"  Features: {list(df.columns)}")
    
    return df


def load_model():
    """
    Step 3: Load the trained model and feature list.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Loading trained model")
    print("=" * 60)
    
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    print(f"  Model: {type(model).__name__}")
    print(f"  Model params: {model.get_params()}")
    print(f"  Expected features: {len(features)}")
    
    return model, features


def make_prediction(model, features, df):
    """
    Step 4: Make a prediction using the last available record.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Making prediction")
    print("=" * 60)
    
    # Use the last row (most recent data point)
    last_row = df.iloc[[-1]]
    
    # Ensure feature order matches training
    X = last_row[features]
    
    print(f"  Predicting for: {last_row.index[0]}")
    print(f"  Input features:")
    for feat in features:
        print(f"    {feat}: {X[feat].values[0]:.4f}")
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    print(f"\n  {'='*40}")
    print(f"  🎯 PREDICTED CO(GT): {prediction:.3f} mg/m³")
    print(f"  📊 ACTUAL CO(GT):    {last_row['CO(GT)'].values[0]:.3f} mg/m³")
    print(f"  📏 DIFFERENCE:       {abs(prediction - last_row['CO(GT)'].values[0]):.3f} mg/m³")
    print(f"  {'='*40}")
    
    # Also predict the next few hours
    print(f"\n  Multi-step forecast (using last known features):")
    forecasts = []
    current_features = X.copy()
    
    for step in range(1, 4):
        pred = model.predict(current_features)[0]
        forecasts.append(pred)
        
        # Shift lag features for next step
        current_features = current_features.copy()
        for lag in [24, 12, 6, 3, 2]:
            if f"CO_lag_{lag}" in features:
                prev_lag = lag - 1
                if prev_lag > 0 and f"CO_lag_{prev_lag}" in features:
                    current_features[f"CO_lag_{lag}"] = current_features[f"CO_lag_{prev_lag}"].values
        if "CO_lag_1" in features:
            current_features["CO_lag_1"] = pred
        
        next_hour = last_row.index[0] + timedelta(hours=step)
        print(f"    +{step}h ({next_hour.strftime('%H:%M')}): {pred:.3f} mg/m³")
    
    return prediction, forecasts


def main():
    """Run the full prediction pipeline."""
    print("╔" + "═" * 58 + "╗")
    print("║    AIR QUALITY CO PREDICTION PIPELINE                    ║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Step 1: Fetch data
    records, latest = fetch_recent_records(hours=48)
    
    # Step 2: Preprocess
    df = preprocess_data(records)
    
    # Step 3: Load model
    model, features = load_model()
    
    # Step 4: Predict
    prediction, forecasts = make_prediction(model, features, df)
    
    print("\n✅ Prediction pipeline completed successfully!")
    
    return {
        "timestamp": str(df.index[-1]),
        "prediction": float(prediction),
        "actual": float(df.iloc[-1]["CO(GT)"]),
        "forecasts": [float(f) for f in forecasts]
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult JSON:\n{json.dumps(result, indent=2)}")
