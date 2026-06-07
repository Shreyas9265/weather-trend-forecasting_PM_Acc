from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


DEFAULT_DATA_PATH = Path("data/raw/GlobalWeatherRepository.csv")


def load_weather_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle CSV and place it there."
        )
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = (
        clean.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return clean


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = standardize_columns(df)

    if "last_updated" in clean.columns:
        clean["last_updated"] = pd.to_datetime(clean["last_updated"], errors="coerce")

    clean = clean.drop_duplicates()

    numeric_cols = clean.select_dtypes(include=[np.number]).columns
    categorical_cols = clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        clean[col] = clean[col].fillna(clean[col].median())

    for col in categorical_cols:
        if col != "last_updated":
            clean[col] = clean[col].fillna("Unknown")

    if "last_updated" in clean.columns:
        clean = clean.dropna(subset=["last_updated"])

    return clean


def get_temperature_column(df: pd.DataFrame) -> str:
    candidates = ["temperature_celsius", "temp_c", "temperature"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError("No temperature column found. Expected temperature_celsius or similar.")


def get_precipitation_column(df: pd.DataFrame) -> str | None:
    candidates = ["precip_mm", "precipitation_mm", "precipitation"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["year"] = featured["last_updated"].dt.year
    featured["month"] = featured["last_updated"].dt.month
    featured["day"] = featured["last_updated"].dt.day
    featured["dayofyear"] = featured["last_updated"].dt.dayofyear
    featured["dayofweek"] = featured["last_updated"].dt.dayofweek
    return featured


def prepare_city_timeseries(
    df: pd.DataFrame,
    city: str,
    target_col: str | None = None,
) -> pd.DataFrame:
    target_col = target_col or get_temperature_column(df)

    if "location_name" not in df.columns:
        raise KeyError("Expected a location_name column in the dataset.")

    city_df = df[df["location_name"].astype(str).str.lower() == city.lower()].copy()
    if city_df.empty:
        raise ValueError(f"No records found for city: {city}")

    city_df = city_df.sort_values("last_updated")
    city_df = create_time_features(city_df)
    city_df[f"{target_col}_lag_1"] = city_df[target_col].shift(1)
    city_df[f"{target_col}_lag_7"] = city_df[target_col].shift(7)
    city_df[f"{target_col}_rolling_7"] = city_df[target_col].rolling(7).mean()
    city_df[f"{target_col}_rolling_14"] = city_df[target_col].rolling(14).mean()
    return city_df.dropna()


def detect_iqr_outliers(df: pd.DataFrame, column: str) -> pd.Series:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)


def add_anomaly_labels(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clean_features = df[feature_cols].dropna()
    result = df.copy()
    result["anomaly"] = 0

    if clean_features.empty:
        return result

    model = IsolationForest(contamination=0.03, random_state=42)
    labels = model.fit_predict(clean_features)
    result.loc[clean_features.index, "anomaly"] = labels
    return result
