from __future__ import annotations

import pandas as pd
import plotly.express as px


def daily_average(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    daily = (
        df.dropna(subset=["last_updated"])
        .assign(date=lambda x: x["last_updated"].dt.date)
        .groupby("date", as_index=False)[value_col]
        .mean()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def temperature_trend_figure(df: pd.DataFrame, temp_col: str):
    daily = daily_average(df, temp_col)
    return px.line(
        daily,
        x="date",
        y=temp_col,
        title="Average Temperature Trend Over Time",
        labels={"date": "Date", temp_col: "Average Temperature (C)"},
    )


def precipitation_trend_figure(df: pd.DataFrame, precip_col: str):
    daily = daily_average(df, precip_col)
    return px.line(
        daily,
        x="date",
        y=precip_col,
        title="Average Precipitation Trend Over Time",
        labels={"date": "Date", precip_col: "Average Precipitation"},
    )


def correlation_figure(df: pd.DataFrame):
    corr = df.select_dtypes("number").corr(numeric_only=True)
    return px.imshow(
        corr,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )


def spatial_temperature_figure(df: pd.DataFrame, temp_col: str):
    required = {"latitude", "longitude", temp_col}
    if not required.issubset(df.columns):
        return None

    sample = df.dropna(subset=["latitude", "longitude", temp_col]).copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    hover_cols = [col for col in ["location_name", "country", temp_col] if col in sample.columns]

    return px.scatter_geo(
        sample,
        lat="latitude",
        lon="longitude",
        color=temp_col,
        hover_name="location_name" if "location_name" in sample.columns else None,
        hover_data=hover_cols,
        title="Spatial Temperature Patterns",
        color_continuous_scale="Turbo",
    )
