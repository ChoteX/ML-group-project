from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "creditsense-ai1215" / "credit_train.csv"
ARTIFACT_DIR = ROOT / "task_b" / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "task_b_metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "task_b_validation_predictions.csv"

RATE_BUCKET_ORDER = ["4.99", "5-8", "8-12", "12-20", "20+"]
MODEL_ORDER = [
    "LinearRegression",
    "MLPRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "HistGradientBoostingRegressor",
    "StackingRegressor",
]


def make_basic_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_nn_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_task_b_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    return {
        "XGBRegressor": Pipeline(
            [
                ("preprocess", make_basic_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=600,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocess", make_basic_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        max_depth=None,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            [
                ("preprocess", make_basic_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=600,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LinearRegression": Pipeline(
            [
                ("preprocess", make_basic_preprocessor(numeric_features, categorical_features)),
                ("model", LinearRegression()),
            ]
        ),
        "MLPRegressor": Pipeline(
            [
                ("preprocess", make_nn_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        batch_size="auto",
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize_target(y: pd.Series) -> dict[str, float | int]:
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    return {
        "count": int(y.shape[0]),
        "min": float(y.min()),
        "q1": q1,
        "median": float(y.median()),
        "mean": float(y.mean()),
        "q3": q3,
        "max": float(y.max()),
        "std": float(y.std()),
        "iqr": float(q3 - q1),
        "p90": float(y.quantile(0.90)),
        "p95": float(y.quantile(0.95)),
        "floor_value": 4.99,
        "floor_pct": float(np.isclose(y, 4.99).mean() * 100.0),
    }


def bucket_interest_rates(y: pd.Series) -> pd.Categorical:
    floor_mask = np.isclose(y, 4.99)
    labels = pd.Series(index=y.index, dtype="object")
    labels.loc[floor_mask] = "4.99"
    labels.loc[~floor_mask] = pd.cut(
        y.loc[~floor_mask],
        bins=[4.99, 8.0, 12.0, 20.0, np.inf],
        labels=["5-8", "8-12", "12-20", "20+"],
        right=False,
    ).astype(str)
    return pd.Categorical(labels, categories=RATE_BUCKET_ORDER, ordered=True)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["RiskTier", "InterestRate"])
    y = df["InterestRate"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    model_factories = make_task_b_models(numeric_features, categorical_features)
    model_aliases = {
        "LinearRegression": "lr",
        "MLPRegressor": "mlp",
        "RandomForestRegressor": "rf",
        "XGBRegressor": "xgb",
        "HistGradientBoostingRegressor": "hgb",
    }

    baseline_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    for model_name in MODEL_ORDER[:-1]:
        model = model_factories[model_name]
        print(f"Fitting {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        baseline_predictions[model_name] = y_pred
        metrics[model_name] = regression_metrics(y_val, y_pred)

    stack_estimators = [
        ("xgb", make_task_b_models(numeric_features, categorical_features)["XGBRegressor"]),
        ("rf", make_task_b_models(numeric_features, categorical_features)["RandomForestRegressor"]),
        ("hgb", make_task_b_models(numeric_features, categorical_features)["HistGradientBoostingRegressor"]),
        ("lr", make_task_b_models(numeric_features, categorical_features)["LinearRegression"]),
        ("mlp", make_task_b_models(numeric_features, categorical_features)["MLPRegressor"]),
    ]
    stack_model = StackingRegressor(
        estimators=stack_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    print("Fitting StackingRegressor...")
    stack_model.fit(X_train, y_train)
    y_pred_stack = stack_model.predict(X_val)
    metrics["StackingRegressor"] = regression_metrics(y_val, y_pred_stack)

    residual = y_val.to_numpy() - y_pred_stack
    abs_error = np.abs(residual)
    validation_predictions = pd.DataFrame(
        {
            "y_true": y_val.to_numpy(),
            "y_pred_stack": y_pred_stack,
            "residual": residual,
            "abs_error": abs_error,
            "rate_bucket": bucket_interest_rates(y_val),
        }
    )
    for model_name, y_pred in baseline_predictions.items():
        validation_predictions[f"y_pred_{model_aliases[model_name]}"] = y_pred

    target_summary = summarize_target(y)
    error_summary = {
        "mean_residual": float(residual.mean()),
        "median_absolute_error": float(np.median(abs_error)),
        "p90_absolute_error": float(np.quantile(abs_error, 0.90)),
        "within_1pt_pct": float((abs_error <= 1.0).mean() * 100.0),
    }

    best_single_model = max(
        ((name, values["r2"]) for name, values in metrics.items() if name != "StackingRegressor"),
        key=lambda item: item[1],
    )[0]

    payload = {
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(X_train.shape[0]),
            "validation_rows": int(X_val.shape[0]),
        },
        "model_order": MODEL_ORDER,
        "final_model_name": "StackingRegressor",
        "final_model_spec": "XGB + RF + HGB + LR + MLP -> Ridge(alpha=1.0)",
        "best_single_model": best_single_model,
        "models": metrics,
        "target_summary": target_summary,
        "final_error_summary": error_summary,
        "artifacts": {
            "validation_predictions_csv": str(PREDICTIONS_PATH),
        },
    }

    validation_predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload["models"]["StackingRegressor"], indent=2))
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
