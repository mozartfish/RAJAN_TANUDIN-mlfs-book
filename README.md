# Air Quality Prediction with Lagged Features - Machine Learning Systems Lab

**Authors**: Pranav Rajan and David Tanudin  
**Course**: Machine Learning Feature Store Systems  
**Date**: November 18th 2025

## Lab Overview

This project extends the O'Reilly book's air quality prediction system by implementing **lagged time-series features** to improve PM2.5 (particulate matter) predictions for San Francisco air quality sensors. The system uses Hopsworks Feature Store, XGBoost regression, and processes data from 6 different air quality sensors across San Francisco.

## Project Objectives

1. **Implement Lagged Features**: Add 1-day, 2-day, and 3-day lagged PM2.5 values as features
2. **Measure Performance Impact**: Compare baseline model (weather only) vs. full model (weather + lagged features)
3. **Multi-Sensor Predictions**: Generate predictions for 6 San Francisco air quality sensors
4. **Feature Analysis**: Analyze correlation and feature importance of lagged features

## System Architecture

```
┌─────────────────────┐
│   Data Sources      │
│  - Weather API      │
│  - AQI Sensors      │
│   (s12-s19)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │
│  - PM2.5 Lagged     │
│  - Weather Features │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hopsworks Feature   │
│      Store          │
│  - air_quality FG   │
│  - weather FG       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  XGBoost Model      │
│  (7 features)       │
│  - pm25_lag1        │
│  - pm25_lag2        │
│  - pm25_lag3        │
│  - temperature      │
│  - precipitation    │
│  - wind_speed       │
│  - wind_direction   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Predictions       │
│  - Forecast (7 days)│
│  - Hindcast (daily) │
│  - 6 Sensors        │
└─────────────────────┘
```

## Key Implementation: Lagged Features

### Implementation Details

```python
# Lagged feature creation (Notebook 1 - Backfill)
df_aq['pm25_lag1'] = df_aq['pm25'].shift(1)
df_aq['pm25_lag2'] = df_aq['pm25'].shift(2)
df_aq['pm25_lag3'] = df_aq['pm25'].shift(3)
```

### Correlation Analysis Results

```
Feature Correlations with PM2.5:
- pm25_lag1: 0.78 (strong positive correlation)
- pm25_lag2: 0.65 (moderate correlation)
- pm25_lag3: 0.52 (moderate correlation)
- temperature: 0.35
- wind_speed: -0.28
```

## Performance Results

### Model Comparison: Baseline vs. Full Model

| Metric | Baseline (Weather Only) | Full Model (Weather + Lags) | Improvement |
|--------|-------------------------|-----------------------------|-------------|
| MSE    | 9.4910                     | 6.2133                     | 34.53%     |
| R²     | -1.0951                    | -0.3716                    | 66.07%     |

### Feature Importance

The XGBoost model ranked features by importance:
1. **pm25_lag1** (highest) - Yesterday's PM2.5 is the strongest predictor
2. temperature_2m_mean
3. **pm25_lag2**
4. wind_speed_10m_max
5. **pm25_lag3**
6. precipitation_sum
7. wind_direction_10m_dominant

## Multi-Sensor Deployment

### San Francisco Sensors

| Sensor ID | Location              | Latitude  | Longitude   | Data File |
|-----------|-----------------------|-----------|-------------|-----------|
| s12       | Berkeley-ATB1         | 37.8715   | -122.2730   | s12.csv   |
| s14       | UCSF Mount Zion       | 37.7866   | -122.4417   | s14.csv   |
| s15       | UCSF Mission Bay      | 37.7685   | -122.3936   | s15.csv   |
| s16       | San Francisco S16     | 37.7749   | -122.4194   | s16.csv   |
| s17       | San Francisco S17     | 37.7749   | -122.4194   | s17.csv   |
| s19       | San Francisco S19     | 37.7749   | -122.4194   | s19.csv   |

### Prediction Results

All sensors generate **7-day forecasts** with individual graphs. Example output:
```
Sensor s12 - Berkeley-ATB1:
  Lagged features: lag1=1.96, lag2=2.85, lag3=3.58
  Predictions: 16.2 - 17.7 µg/m³ (Good air quality)

Sensor s14 - UCSF Mount Zion:
  Lagged features: lag1=2.52, lag2=2.94, lag3=3.87
  Predictions: 16.2 - 17.7 µg/m³ (Good air quality)
```

**Note**: Predictions are similar across sensors because:
1. All sensors use the same weather forecast
2. San Francisco consistently shows good air quality (PM2.5 < 35)
3. Lagged features are all in the low range (2-5 µg/m³)

Run notebooks in order:

#### **Notebook 1: Feature Backfill** (`1_air_quality_feature_backfill.ipynb`)
- Load historical air quality data
- **Create lagged features (pm25_lag1, lag2, lag3)**
- Validate with Great Expectations
- Insert into Hopsworks Feature Store

```python
# Key addition: Lagged features
df_aq['pm25_lag1'] = df_aq['pm25'].shift(1)
df_aq['pm25_lag2'] = df_aq['pm25'].shift(2)
df_aq['pm25_lag3'] = df_aq['pm25'].shift(3)
df_aq = df_aq.dropna()  # Remove first 3 rows with NaN lags
```

#### **Notebook 2: Daily Feature Pipeline** (`2_air_quality_feature_pipeline.ipynb`)
- Fetch daily air quality updates
- Calculate lagged features from historical data
- Update feature store

#### **Notebook 3: Training Pipeline** (`3_air_quality_training_pipeline.ipynb`)
- Create feature view with lagged features
- Train XGBoost model (7 features)
- **Compare baseline vs. full model performance**
- Save model to registry

```python
# Feature selection includes lagged features
features = ['pm25_lag1', 'pm25_lag2', 'pm25_lag3',
            'temperature_2m_mean', 'precipitation_sum',
            'wind_speed_10m_max', 'wind_direction_10m_dominant']
```

#### **Notebook 4: Batch Inference** (`4_air_quality_batch_inference.ipynb`)
- Load trained model
- Fetch weather forecast
- Calculate lagged features for prediction
- Generate 7-day forecast
- Create hindcast comparison graph

#### **Multi-Sensor Predictions** (`MULTI_SENSOR_BATCH_PREDICTIONS.ipynb`)
- Generate predictions for all 6 sensors
- Individual lagged features per sensor
- Create separate forecast graphs
- Export combined predictions CSV
