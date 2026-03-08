# Air Quality Time Series Pipeline

A comprehensive time-series analysis and prediction pipeline for the **UCI Air Quality Dataset**, covering data exploration, modeling, database design, API development, and automated prediction.

## Dataset

**UCI Air Quality Dataset** — Hourly averaged sensor responses and air quality measurements from a multisensor device deployed in an Italian city (March 2004 – April 2005).

- **Source:** [Kaggle - Air Quality UCI](https://www.kaggle.com/datasets/harinarayanan22/airquality)
- **Records:** 9,357 hourly readings
- **Target Variable:** CO(GT) — Carbon Monoxide concentration (mg/m³)

## Project Structure

```
airquality/
├── airquality.ipynb        # Main Jupyter notebook (Tasks 1–4)
├── scripts/
│   ├── api.py               # Flask CRUD API (Task 3)
│   └── predict.py           # Prediction pipeline script (Task 4)
├── data/
│   └── airquality.db        # SQLite database (Task 2)
├── models/
│   ├── best_model.pkl       # Trained Random Forest model (Task 1C)
│   └── feature_list.pkl     # Feature list used for prediction
├── outputs/
│   └── erd-diagram.png      # Entity-Relationship Diagram              
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Tasks Overview

### Task 1: Time-Series Preprocessing & Exploratory Analysis

**A. Understanding the Dataset**
Hourly granularity: March 10, 2004–April 4, 2005
Handling missing values: forward-fill for sensor columns, linear interpolation for ground-truth columns, and discarded NMHC(GT) (90% missing)
Statistical distributions and the identification of outliers

**B. Analytical Questions (5 questions with visualizations)**

1. **Trend Analysis** — Is CO increasing/decreasing? (Linear regression + monthly averages)
2. **Correlation with Meteorology** — Do temperature/humidity correlate with CO? (Heatmap + hexbin plots)
3. **Lag Effects** ★ — Is CO related to previous 24 hours? (ACF/PACF + lag correlation analysis)
4. **Moving Averages** ★ — What do 6h, 24h, 7-day moving averages reveal? (SMA vs EMA comparison)
5. **Seasonality** — Diurnal and weekly patterns? (Hour×Day heatmap, weekday vs weekend profiles)

**C. Model Training**

| Experiment | Model                     | Test MAE   | Test RMSE  | Test R²    |
| ---------- | ------------------------- | ---------- | ---------- | ---------- |
| 1          | Random Forest (Default)   | 0.2632     | 0.4671     | 0.8808     |
| 2          | **Random Forest (Tuned)** | **0.2633** | **0.4630** | **0.8829** |
| 3          | Gradient Boosting (Tuned) | 0.2939     | 0.4981     | 0.8647     |

Best model: **Random Forest** with `max_depth=15, n_estimators=300` (Test R² = 0.8829)

### Task 2: Database Design (SQL + MongoDB)

- **SQL (SQLite):** 3 normalized tables — `stations`, `readings`, `meteorology` with ERD in 3NF
- **MongoDB:** Denormalized document model with embedded station/pollutant/meteorological sub-documents
- Each DB includes 3+ demonstration queries

### Task 3: CRUD API Endpoints

Flask API with full CRUD for both SQL and MongoDB:

| Method | SQL Endpoint                              | MongoDB Endpoint                            | Description      |
| ------ | ----------------------------------------- | ------------------------------------------- | ---------------- |
| POST   | `/api/sql/readings`                       | `/api/mongo/readings`                       | Create a reading |
| GET    | `/api/sql/readings` or `/<id>`            | `/api/mongo/readings` or `/<id>`            | Read readings    |
| PUT    | `/api/sql/readings/<id>`                  | `/api/mongo/readings/<id>`                  | Update a reading |
| DELETE | `/api/sql/readings/<id>`                  | `/api/mongo/readings/<id>`                  | Delete a reading |
| GET    | `/api/sql/readings/latest`                | `/api/mongo/readings/latest`                | Latest record    |
| GET    | `/api/sql/readings/daterange?start=&end=` | `/api/mongo/readings/daterange?start=&end=` | Date range query |

### Task 4: Prediction Script

End-to-end pipeline: **Fetch → Preprocess → Load Model → Predict**

Fetches recent data from the API (with direct-DB fallback), applies the same feature engineering from Task 1, loads the saved model, and generates a CO concentration prediction plus a 3-hour forecast.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `airquality.ipynb` in VS Code or Jupyter and run all cells. This executes Tasks 1–4 (preprocessing, analysis, DB creation, API testing, and prediction).

### 3. Start the API server

```bash
cd scripts
python api.py
```

The server starts on `http://localhost:5000`. Health check: `GET /api/health`.

### 4. Run prediction (standalone)

```bash
cd scripts
python predict.py
```

The script fetches data from the running API (or falls back to the SQLite database), preprocesses it, and outputs a predicted CO concentration.

## Dependencies

- **Python 3.10+**
- pandas, numpy, matplotlib, seaborn
- scikit-learn, statsmodels, scipy
- flask, pymongo, mongomock
- joblib, kagglehub

## Reproducing Results

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv` and activate it
3. Install packages: `pip install -r requirements.txt`
4. Run `airquality.ipynb` — this downloads the dataset, trains the model, creates the database, and tests the API
5. (Optional) Start the API separately with `python scripts/api.py`, then run `python scripts/predict.py`
