from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_processing import (  # noqa: E402
    add_anomaly_labels,
    clean_weather_data,
    get_precipitation_column,
    get_temperature_column,
    load_weather_data,
    prepare_city_timeseries,
)
from modeling import get_model_features, train_forecasting_models  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "GlobalWeatherRepository.csv"

st.set_page_config(
    page_title="Weather Trend Forecasting",
    page_icon="W",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_clean_data(path: str) -> pd.DataFrame:
    raw = load_weather_data(path)
    return clean_weather_data(raw)


def iqr_outlier_summary(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((data[col] < lower) | (data[col] > upper)).sum())
        rows.append(
            {
                "Feature": col,
                "Lower Bound": round(lower, 3),
                "Upper Bound": round(upper, 3),
                "Outlier Count": count,
                "Outlier %": round((count / len(data)) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


st.title("Weather Trend Forecasting")
st.caption("Global Weather Repository data science assessment project")
st.success(
    "Dashboard loaded sucessfully! Use the tabs to explore the dataset, EDA, forecasting models, and advanced analyses."
)

with st.expander("PM Accelerator Mission", expanded=True):
    st.write(
        "PM Accelerator's mission is to provide access to education and career growth "
        "opportunities. This dashboard includes the mission as requested in the assessment."
    )

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Download the Kaggle CSV and place it at "
        "`data/raw/GlobalWeatherRepository.csv`."
    )
    st.stop()

df = load_clean_data(str(DATA_PATH))
temp_col = get_temperature_column(df)
precip_col = get_precipitation_column(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.header("Controls")

if "location_name" in df.columns:
    city_counts = df["location_name"].dropna().astype(str).value_counts()
    forecast_city_counts = city_counts[city_counts >= 30]

    if forecast_city_counts.empty:
        forecast_city_counts = city_counts

    city_options = forecast_city_counts.index.tolist()

    selected_city = st.sidebar.selectbox(
        "Forecast city",
        city_options,
        index=0,
    )

    st.sidebar.caption(f"{int(city_counts[selected_city]):,} records for selected city")
else:
    selected_city = None

st.sidebar.write("Rows:", f"{len(df):,}")

overview_cols = st.columns(4)
overview_cols[0].metric("Rows", f"{len(df):,}")
overview_cols[1].metric("Columns", f"{df.shape[1]:,}")
overview_cols[2].metric("Avg Temp", f"{df[temp_col].mean():.2f} C")
overview_cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

tab_requirements, tab_eda, tab_forecast, tab_advanced, tab_data = st.tabs(
    ["Requirements", "EDA", "Forecasting", "Advanced Analysis", "Data"]
)

with tab_requirements:
    st.subheader("Assessment Requirements Coverage")

    coverage = pd.DataFrame(
        [
            [
                "Handle missing values",
                "Cleaning fills numeric columns with medians and categorical columns with Unknown.",
            ],
            [
                "Handle outliers",
                "EDA shows IQR outlier counts; Advanced Analysis uses Isolation Forest anomaly detection.",
            ],
            [
                "Normalize data",
                "Data tab shows Min-Max normalized numeric features.",
            ],
            [
                "EDA trends/correlations/patterns",
                "EDA tab shows trend lines, heatmap, and country patterns.",
            ],
            [
                "Temperature and precipitation charts",
                "EDA tab includes both trend visualizations.",
            ],
            [
                "Use last_updated for time series",
                "Forecasting tab creates time features and uses a time-based split.",
            ],
            [
                "Build and compare models",
                "Baseline, Random Forest, Gradient Boosting, and Ensemble are compared.",
            ],
            [
                "Model evaluation metrics",
                "Forecasting tab reports MAE, RMSE, and R2.",
            ],
            [
                "Advanced analyses",
                "Anomaly detection, air quality, feature importance, climate, and spatial analysis are included.",
            ],
            [
                "PM Accelerator mission",
                "Displayed at the top of the dashboard.",
            ],
        ],
        columns=["Requirement", "How this app fulfills it"],
    )

    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Cleaning Summary")
    st.write(
        "The app standardizes column names, converts `last_updated` to datetime, removes duplicates, "
        "fills missing numeric values with medians, fills missing categorical values, and removes rows "
        "without a valid timestamp."
    )

with tab_eda:
    st.subheader("Exploratory Data Analysis")

    left, right = st.columns(2)

    with left:
        st.write("Average Temperature Trend Over Time")
        daily_temp = (
            df.assign(date=df["last_updated"].dt.date)
            .groupby("date", as_index=False)[temp_col]
            .mean()
        )
        daily_temp["date"] = pd.to_datetime(daily_temp["date"])
        st.line_chart(daily_temp, x="date", y=temp_col)

    with right:
        st.write("Average Precipitation Trend Over Time")

        if precip_col:
            daily_precip = (
                df.assign(date=df["last_updated"].dt.date)
                .groupby("date", as_index=False)[precip_col]
                .mean()
            )
            daily_precip["date"] = pd.to_datetime(daily_precip["date"])
            st.line_chart(daily_precip, x="date", y=precip_col)
        else:
            st.info("No precipitation column was found in this dataset.")

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes("number").corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig, clear_figure=True)

    if "country" in df.columns:
        country_summary = (
            df.groupby("country", as_index=False)[temp_col]
            .mean()
            .sort_values(temp_col, ascending=False)
            .head(20)
        )

        st.subheader("Top Countries by Average Temperature")
        st.bar_chart(country_summary, x="country", y=temp_col)

    st.subheader("IQR Outlier Summary")

    outlier_candidates = [
        col
        for col in [
            temp_col,
            precip_col,
            "humidity",
            "wind_kph",
            "pressure_mb",
            "uv_index",
        ]
        if col and col in df.columns
    ]

    if outlier_candidates:
        st.dataframe(
            iqr_outlier_summary(df, outlier_candidates),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No standard weather columns were available for IQR outlier analysis.")

with tab_forecast:
    st.subheader("Forecasting Models")

    if not selected_city:
        st.info("Enter a city name in the sidebar to run forecasting.")
    else:
        try:
            selected_count = int(
                df["location_name"].dropna().astype(str).value_counts().get(selected_city, 0)
            )

            if selected_count < 30:
                st.warning(
                    f"`{selected_city}` has only {selected_count} records. "
                    "Forecasting needs more time-series history, so choose a city with at least 30 records."
                )
                st.stop()

            city_df = prepare_city_timeseries(df, selected_city, temp_col)
            feature_cols = get_model_features(city_df, temp_col)

            metrics, models, prediction_frame = train_forecasting_models(
                city_df,
                temp_col,
                feature_cols,
            )

            st.write(f"Forecasting target: `{temp_col}` for `{selected_city}`")
            st.dataframe(metrics, use_container_width=True, hide_index=True)

            plot_cols = ["last_updated", temp_col] + [
                col
                for col in [
                    "Baseline",
                    "Random Forest",
                    "Gradient Boosting",
                    "Ensemble",
                ]
                if col in prediction_frame.columns
            ]

            long_preds = prediction_frame[plot_cols].melt(
                id_vars="last_updated",
                var_name="Series",
                value_name="Temperature",
            )

            forecast_chart = long_preds.pivot(
                index="last_updated",
                columns="Series",
                values="Temperature",
            )

            st.subheader(f"Actual vs Forecasted Temperature - {selected_city}")
            st.line_chart(forecast_chart)

            if "Random Forest" in models:
                importance = pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Importance": models["Random Forest"].feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)

                st.subheader("Feature Importance")
                st.bar_chart(importance.head(15), x="Feature", y="Importance")

        except Exception as exc:
            st.warning(f"Could not train forecast models for `{selected_city}`: {exc}")

            if "location_name" in df.columns:
                st.write("Example available cities:")
                st.write(", ".join(df["location_name"].dropna().astype(str).unique()[:20]))

with tab_advanced:
    st.subheader("Advanced Analysis")

    anomaly_features = [
        col
        for col in [
            temp_col,
            "humidity",
            "wind_kph",
            "precip_mm",
            "pressure_mb",
            "air_quality_pm2.5",
        ]
        if col in df.columns
    ]

    if len(anomaly_features) >= 2:
        anomaly_df = add_anomaly_labels(df, anomaly_features)

        st.metric("Detected Anomalies", f"{(anomaly_df['anomaly'] == -1).sum():,}")

        sample = anomaly_df.sample(min(len(anomaly_df), 5000), random_state=42)

        st.subheader("Temperature Anomaly Detection")

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(
            data=sample,
            x="last_updated",
            y=temp_col,
            hue="anomaly",
            palette={1: "#4c78a8", -1: "#f58518"},
            ax=ax,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature")
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Not enough numeric weather features for anomaly detection.")

    air_quality_cols = [
        col for col in df.columns if "air_quality" in col.lower() or "pm2" in col.lower()
    ]

    weather_cols = [
        col
        for col in [
            temp_col,
            "humidity",
            "wind_kph",
            "pressure_mb",
            precip_col,
        ]
        if col and col in df.columns
    ]

    if air_quality_cols and weather_cols:
        corr_cols = air_quality_cols + weather_cols

        st.subheader("Air Quality and Weather Correlation")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[corr_cols].corr(numeric_only=True),
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Air quality columns were not found or could not be matched.")

    st.subheader("Climate and Geographical Patterns")

    if "country" in df.columns:
        geo_summary = (
            df.groupby("country", as_index=False)
            .agg(
                average_temperature=(temp_col, "mean"),
                temperature_variability=(temp_col, "std"),
                observations=(temp_col, "count"),
            )
            .sort_values("average_temperature", ascending=False)
        )

        st.write("Countries with highest average temperature")
        st.dataframe(geo_summary.head(15), use_container_width=True, hide_index=True)

        st.write("Countries with highest temperature variability")
        variability = geo_summary.sort_values("temperature_variability", ascending=False)
        st.dataframe(variability.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("Country-level climate analysis requires a `country` column.")

    if {"latitude", "longitude"}.issubset(df.columns):
        st.subheader("Spatial Temperature Patterns")

        map_df = df.dropna(subset=["latitude", "longitude", temp_col]).copy()

        if len(map_df) > 5000:
            map_df = map_df.sample(5000, random_state=42)

        map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})

        st.map(map_df[["lat", "lon", temp_col]])
    else:
        st.info("Spatial map requires latitude and longitude columns.")

with tab_data:
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Normalized Numeric Feature Preview")

    if numeric_cols:
        scaler = MinMaxScaler()

        normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=[f"{col}_normalized" for col in numeric_cols],
            index=df.index,
        )

        st.dataframe(normalized.head(100), use_container_width=True)
    else:
        st.info("No numeric columns were available for normalization.")

    st.subheader("Column Summary")

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().to_numpy(),
            "unique": df.nunique(dropna=True).to_numpy(),
        }
    )

    st.dataframe(summary, use_container_width=True)