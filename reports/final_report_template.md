# Weather Trend Forecasting Final Report

## 1. Objective

The objective of this project is to analyze global weather data and forecast future weather trends using the Global Weather Repository dataset.

## 2. PM Accelerator Mission

PM Accelerator's mission is to provide access to education and career growth opportunities. This project includes the mission as required by the assessment.

## 3. Dataset

The dataset was downloaded from Kaggle's Global Weather Repository. It contains daily weather information for cities around the world and includes temperature, precipitation, humidity, wind, pressure, air quality, and geographical features.

## 4. Data Cleaning

Steps performed:

- Converted `last_updated` to datetime format.
- Removed duplicate records.
- Filled missing numeric values with medians.
- Filled missing categorical values with `Unknown`.
- Detected outliers using IQR and Isolation Forest.

## 5. Exploratory Data Analysis

EDA included:

- Temperature trend over time.
- Precipitation trend over time.
- Correlation heatmap.
- Weather pattern comparison by city/country.
- Air quality and weather correlation.

## 6. Forecasting Models

Models compared:

- Baseline lag model
- Random Forest Regressor
- Gradient Boosting Regressor
- Ensemble average model

Evaluation metrics:

- MAE
- RMSE
- R2

## 7. Advanced Analysis

Advanced analysis included:

- Anomaly detection
- Feature importance
- Air quality correlation
- Spatial/geographical analysis

## 8. Key Insights

Add your final findings here after running the project.

Example:

- Temperature patterns vary strongly by city and region.
- Lag and rolling temperature features improve forecasting performance.
- Humidity, pressure, and wind can be useful forecasting features.
- Air quality indicators show measurable relationships with selected weather variables.

## 9. Conclusion

This project demonstrates a complete data science workflow, from data cleaning and EDA to forecasting, model comparison, and dashboard presentation.
