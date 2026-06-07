from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def get_model_features(df: pd.DataFrame, target_col: str) -> list[str]:
    ignore = {
        target_col,
        "last_updated",
        "location_name",
        "country",
        "region",
        "timezone",
        "condition_text",
        "wind_direction",
        "moon_phase",
        "sunrise",
        "sunset",
        "moonrise",
        "moonset",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in ignore]


def time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
):
    ordered = df.sort_values("last_updated")
    split_index = int(len(ordered) * (1 - test_size))

    train = ordered.iloc[:split_index]
    test = ordered.iloc[split_index:]

    x_train = train[feature_cols]
    y_train = train[target_col]
    x_test = test[feature_cols]
    y_test = test[target_col]

    return x_train, x_test, y_train, y_test, train, test


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def train_forecasting_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    feature_cols = feature_cols or get_model_features(df, target_col)
    x_train, x_test, y_train, y_test, _, test = time_based_split(df, feature_cols, target_col)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    predictions = {}
    metrics = []

    lag_col = f"{target_col}_lag_1"
    if lag_col in test.columns:
        baseline_pred = test[lag_col]
        predictions["Baseline"] = baseline_pred.to_numpy()
        metrics.append({"Model": "Baseline", **evaluate_predictions(y_test, baseline_pred)})

    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fitted_models[name] = model
        predictions[name] = pred
        metrics.append({"Model": name, **evaluate_predictions(y_test, pred)})

    if len(predictions) >= 2:
        model_preds = [
            pred for name, pred in predictions.items() if name in {"Random Forest", "Gradient Boosting"}
        ]
        if model_preds:
            ensemble_pred = np.mean(model_preds, axis=0)
            predictions["Ensemble"] = ensemble_pred
            metrics.append({"Model": "Ensemble", **evaluate_predictions(y_test, ensemble_pred)})

    prediction_frame = test[["last_updated", target_col]].copy()
    for name, pred in predictions.items():
        prediction_frame[name] = pred

    return pd.DataFrame(metrics).sort_values("RMSE"), fitted_models, prediction_frame


def random_split_model(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    x = df[feature_cols]
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, evaluate_predictions(y_test, pred)
