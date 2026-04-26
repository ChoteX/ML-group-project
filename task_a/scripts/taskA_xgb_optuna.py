#!/usr/bin/env python3
"""
Task A XGBoost notebook backend with optional Optuna tuning.

This module intentionally copies the existing Task A preprocessing and submission I/O
contract from `taskA_nn.ipynb` so the model layer can be swapped to a single
`xgboost.XGBClassifier` without changing feature preparation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    joblib = None

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    optuna = None

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    XGBClassifier = None

warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "creditsense-ai1215"
TRAIN_PATH = DATA_DIR / "credit_train.csv"
TEST_PATH = DATA_DIR / "credit_test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

ARTIFACT_DIR = REPO_ROOT / "task_a" / "artifacts"
REPORT_DIR = REPO_ROOT / "task_a" / "reports"
SUBMISSION_PATH = REPO_ROOT / "submissions" / "submission.csv"

CLASS_LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["VeryLow(0)", "Low(1)", "Moderate(2)", "High(3)", "VeryHigh(4)"]

RANDOM_STATE = 42


TASK_A_ZERO_FILL_COLS = [
    "StudentLoanOutstandingBalance",
    "MortgageOutstandingBalance",
    "PropertyValue",
    "InvestmentPortfolioValue",
    "VehicleValue",
    "AutoLoanOutstandingBalance",
    "SecondaryMonthlyIncome",
    "CollateralValue",
]

TASK_A_MONEY_CLIP_COLS = [
    "AnnualIncome",
    "MonthlyGrossIncome",
    "SecondaryMonthlyIncome",
    "TotalMonthlyIncome",
    "SavingsBalance",
    "CheckingBalance",
    "InvestmentPortfolioValue",
    "PropertyValue",
    "VehicleValue",
    "TotalAssets",
    "MortgageOutstandingBalance",
    "AutoLoanOutstandingBalance",
    "StudentLoanOutstandingBalance",
    "TotalCreditLimit",
    "RequestedLoanAmount",
    "CollateralValue",
    "MonthlyPaymentEstimate",
]

TASK_A_RATIO_CLIP_COLS = ["LoanToIncomeRatio", "PaymentToIncomeRatio"]
TASK_A_LOG_COLS = TASK_A_MONEY_CLIP_COLS.copy()

DEFAULT_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@dataclass
class TrainEvalConfig:
    artifact_tag: str = "taskA_xgb_optuna"
    validation_size: float = 0.2
    random_state: int = RANDOM_STATE
    model_threads: int = 2
    verbose: bool = False
    metric_name: str = "accuracy"
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 40
    internal_early_stopping_size: float = 0.1
    run_full_train_prediction: bool = True
    preserve_interest_rate: bool = True
    save_model_bundle: bool = True
    xgb_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    use_scale_pos_weight: bool = False


@dataclass
class OptunaConfig:
    study_name: str = "taskA_xgb_optuna"
    artifact_tag: str = "taskA_xgb_optuna"
    n_trials: int = 25
    timeout: int | None = None
    cv_folds: int = 5
    metric_name: str = "accuracy"
    random_state: int = RANDOM_STATE
    model_threads: int = 2
    verbose: bool = False
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 40
    internal_early_stopping_size: float = 0.1
    include_scale_pos_weight: bool = False
    fixed_params: dict[str, Any] = field(default_factory=dict)


def _require_xgboost() -> None:
    if XGBClassifier is None:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "xgboost is required for Task A XGBoost training. "
            "Install it in the notebook environment, for example: `pip install xgboost`."
        )


def _require_optuna() -> None:
    if optuna is None:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "optuna is required for Bayesian optimization. "
            "Install it in the notebook environment, for example: `pip install optuna`."
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False, float_digits: int = 4) -> str:
    working = df.copy()
    if index:
        index_name = working.index.name or "index"
        working = working.reset_index().rename(columns={"index": index_name})

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if abs(value - round(value)) < 1e-12:
                return str(int(round(value)))
            return f"{value:.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, working.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(working.columns)) + " |"
    rows = [
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in working.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator] + rows)


def _fit_tabular_preprocessor(
    X_train_raw: pd.DataFrame,
    *,
    zero_fill_cols: list[str],
    clip_cols: list[str],
    log_cols: list[str],
) -> dict[str, Any]:
    X_train_raw = X_train_raw.copy()

    base_cols = X_train_raw.columns.tolist()
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_flag_cols = [col for col in base_cols if X_train_raw[col].isna().any()]

    zero_fill_cols = [col for col in zero_fill_cols if col in num_cols]
    median_fill_map = {
        col: float(X_train_raw[col].median())
        for col in num_cols
        if col not in zero_fill_cols
    }

    clip_map: dict[str, float] = {}
    for col in clip_cols:
        if col not in num_cols:
            continue
        non_null = X_train_raw[col].dropna()
        clip_map[col] = float(non_null.quantile(0.99)) if not non_null.empty else 0.0

    prep = {
        "base_cols": base_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "missing_flag_cols": missing_flag_cols,
        "zero_fill_cols": zero_fill_cols,
        "median_fill_map": median_fill_map,
        "clip_map": clip_map,
        "log_cols": [col for col in log_cols if col in num_cols],
    }

    transformed_train = _transform_tabular(X_train_raw, prep, align=False)
    prep["feature_columns"] = transformed_train.columns.tolist()
    return prep


def _transform_tabular(X_raw: pd.DataFrame, prep: dict[str, Any], align: bool = True) -> pd.DataFrame:
    working = X_raw.copy()

    for col in prep["missing_flag_cols"]:
        working[f"{col}_is_missing"] = working[col].isna().astype(np.int8)

    for col in prep["cat_cols"]:
        working[col] = working[col].fillna("Missing").astype(str)

    for col in prep["zero_fill_cols"]:
        working[col] = working[col].fillna(0.0)

    for col, med in prep["median_fill_map"].items():
        working[col] = working[col].fillna(med)

    for col, upper in prep["clip_map"].items():
        working[col] = working[col].clip(upper=upper)

    for col in prep["log_cols"]:
        working[col] = np.log1p(working[col])

    numeric_frame = working[prep["num_cols"]].apply(pd.to_numeric)
    missing_cols = [f"{col}_is_missing" for col in prep["missing_flag_cols"]]
    missing_frame = (
        working[missing_cols].astype(np.int8)
        if missing_cols
        else pd.DataFrame(index=working.index)
    )
    cat_frame = pd.get_dummies(
        working[prep["cat_cols"]],
        prefix=prep["cat_cols"],
        prefix_sep="__",
        drop_first=False,
        dtype=np.int8,
    )

    transformed = pd.concat([numeric_frame, missing_frame, cat_frame], axis=1).fillna(0)
    if align and "feature_columns" in prep:
        transformed = transformed.reindex(columns=prep["feature_columns"], fill_value=0)
    return transformed


def fit_task_a_preprocessor(X_train_raw: pd.DataFrame) -> dict[str, Any]:
    return _fit_tabular_preprocessor(
        X_train_raw,
        zero_fill_cols=TASK_A_ZERO_FILL_COLS,
        clip_cols=TASK_A_MONEY_CLIP_COLS + TASK_A_RATIO_CLIP_COLS,
        log_cols=TASK_A_LOG_COLS,
    )


def transform_task_a(X_raw: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    transformed = _transform_tabular(X_raw, prep, align=True)
    return transformed.reindex(columns=prep["feature_columns"], fill_value=0)


def load_task_a_data() -> dict[str, Any]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    X = train_df.drop(columns=["RiskTier", "InterestRate"])
    y = train_df["RiskTier"].astype(int)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "sample_submission": sample_submission,
        "X": X,
        "y": y,
    }


def compute_scale_pos_weight(y: pd.Series, enabled: bool) -> float | None:
    if not enabled:
        return None
    unique = sorted(pd.Series(y).unique())
    if len(unique) != 2:
        return None

    counts = pd.Series(y).value_counts()
    majority = float(counts.max())
    minority = float(counts.min())
    if minority == 0:
        return None
    return majority / minority


def build_xgb_classifier(
    *,
    params: dict[str, Any],
    n_classes: int,
    random_state: int,
    model_threads: int,
    scale_pos_weight: float | None = None,
) -> Any:
    _require_xgboost()

    model_params = dict(params)
    if n_classes == 2:
        model_params.setdefault("objective", "binary:logistic")
        model_params.setdefault("eval_metric", "logloss")
        if scale_pos_weight is not None:
            model_params["scale_pos_weight"] = scale_pos_weight
    else:
        model_params.setdefault("objective", "multi:softprob")
        model_params.setdefault("num_class", n_classes)
        model_params.setdefault("eval_metric", "mlogloss")

    model_params.setdefault("tree_method", "hist")
    model_params.setdefault("random_state", random_state)
    model_params.setdefault("n_jobs", model_threads)
    model_params.setdefault("verbosity", 0)
    return XGBClassifier(**model_params)


def extract_best_iteration(model: Any, fallback_n_estimators: int) -> tuple[int | None, int]:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return None, int(fallback_n_estimators)
    return int(best_iteration), int(best_iteration) + 1


@dataclass
class TaskAXGBFitMetadata:
    early_stopping_used: bool
    early_stopping_source: str
    best_iteration: int | None
    best_n_estimators: int
    feature_count: int
    scale_pos_weight: float | None
    evals_result: dict[str, Any] | None = None


class TaskAXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Existing Task A classifier interface assumptions:
    - input features are raw pandas DataFrames, not preprocessed matrices;
    - labels are integer-encoded `RiskTier` values;
    - preprocessing is fit on train folds only, then reused for validation/test;
    - `predict` returns integer class labels and `predict_proba` returns class probabilities.
    """

    def __init__(
        self,
        *,
        xgb_params: dict[str, Any] | None = None,
        random_state: int = RANDOM_STATE,
        model_threads: int = 2,
        enable_early_stopping: bool = True,
        early_stopping_rounds: int = 40,
        internal_early_stopping_size: float = 0.1,
        use_scale_pos_weight: bool = False,
        verbose: bool = False,
    ):
        self.xgb_params = dict(DEFAULT_XGB_PARAMS) if xgb_params is None else dict(xgb_params)
        self.random_state = random_state
        self.model_threads = model_threads
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.internal_early_stopping_size = internal_early_stopping_size
        self.use_scale_pos_weight = use_scale_pos_weight
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        *,
        refit_on_full_train: bool = True,
    ) -> "TaskAXGBClassifier":
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True).astype(int)
        n_classes = int(y.nunique())
        scale_pos_weight = compute_scale_pos_weight(y, self.use_scale_pos_weight)

        if self.enable_early_stopping and X_val is not None and y_val is not None:
            prep = fit_task_a_preprocessor(X)
            X_train_proc = transform_task_a(X, prep)
            X_val_proc = transform_task_a(X_val.reset_index(drop=True), prep)

            model = build_xgb_classifier(
                params=self.xgb_params,
                n_classes=n_classes,
                random_state=self.random_state,
                model_threads=self.model_threads,
                scale_pos_weight=scale_pos_weight,
            )
            model.fit(
                X_train_proc,
                y,
                eval_set=[(X_train_proc, y), (X_val_proc, pd.Series(y_val).reset_index(drop=True).astype(int))],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
            best_iteration, best_n_estimators = extract_best_iteration(
                model,
                fallback_n_estimators=int(self.xgb_params["n_estimators"]),
            )
            evals_result = model.evals_result() if hasattr(model, "evals_result") else None

            self.prep_ = prep
            self.model_ = model
            self.classes_ = np.sort(y.unique())
            self.fit_metadata_ = TaskAXGBFitMetadata(
                early_stopping_used=True,
                early_stopping_source="external_validation",
                best_iteration=best_iteration,
                best_n_estimators=best_n_estimators,
                feature_count=int(X_train_proc.shape[1]),
                scale_pos_weight=scale_pos_weight,
                evals_result=evals_result,
            )
            return self

        if self.enable_early_stopping and self.internal_early_stopping_size > 0:
            X_fit_raw, X_es_raw, y_fit, y_es = train_test_split(
                X,
                y,
                test_size=self.internal_early_stopping_size,
                random_state=self.random_state,
                stratify=y,
            )
            prep_es = fit_task_a_preprocessor(X_fit_raw)
            X_fit_proc = transform_task_a(X_fit_raw, prep_es)
            X_es_proc = transform_task_a(X_es_raw, prep_es)

            temp_model = build_xgb_classifier(
                params=self.xgb_params,
                n_classes=n_classes,
                random_state=self.random_state,
                model_threads=self.model_threads,
                scale_pos_weight=scale_pos_weight,
            )
            temp_model.fit(
                X_fit_proc,
                y_fit.reset_index(drop=True),
                eval_set=[(X_fit_proc, y_fit.reset_index(drop=True)), (X_es_proc, y_es.reset_index(drop=True))],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
            best_iteration, best_n_estimators = extract_best_iteration(
                temp_model,
                fallback_n_estimators=int(self.xgb_params["n_estimators"]),
            )
            evals_result = temp_model.evals_result() if hasattr(temp_model, "evals_result") else None

            if refit_on_full_train:
                final_prep = fit_task_a_preprocessor(X)
                X_full_proc = transform_task_a(X, final_prep)
                final_params = dict(self.xgb_params)
                final_params["n_estimators"] = best_n_estimators
                final_model = build_xgb_classifier(
                    params=final_params,
                    n_classes=n_classes,
                    random_state=self.random_state,
                    model_threads=self.model_threads,
                    scale_pos_weight=scale_pos_weight,
                )
                final_model.fit(X_full_proc, y, verbose=False)
                self.prep_ = final_prep
                self.model_ = final_model
                feature_count = int(X_full_proc.shape[1])
            else:
                self.prep_ = prep_es
                self.model_ = temp_model
                feature_count = int(X_fit_proc.shape[1])

            self.classes_ = np.sort(y.unique())
            self.fit_metadata_ = TaskAXGBFitMetadata(
                early_stopping_used=True,
                early_stopping_source="internal_holdout",
                best_iteration=best_iteration,
                best_n_estimators=best_n_estimators,
                feature_count=feature_count,
                scale_pos_weight=scale_pos_weight,
                evals_result=evals_result,
            )
            return self

        prep = fit_task_a_preprocessor(X)
        X_train_proc = transform_task_a(X, prep)
        model = build_xgb_classifier(
            params=self.xgb_params,
            n_classes=n_classes,
            random_state=self.random_state,
            model_threads=self.model_threads,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train_proc, y, verbose=False)
        self.prep_ = prep
        self.model_ = model
        self.classes_ = np.sort(y.unique())
        self.fit_metadata_ = TaskAXGBFitMetadata(
            early_stopping_used=False,
            early_stopping_source="disabled",
            best_iteration=None,
            best_n_estimators=int(self.xgb_params["n_estimators"]),
            feature_count=int(X_train_proc.shape[1]),
            scale_pos_weight=scale_pos_weight,
            evals_result=None,
        )
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return transform_task_a(X.reset_index(drop=True), self.prep_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.transform_features(X)
        return self.model_.predict_proba(X_proc)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.transform_features(X)
        return self.model_.predict(X_proc).astype(int)


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train.reset_index(drop=True).astype(int),
        y_val.reset_index(drop=True).astype(int),
    )


def build_artifact_paths(tag: str) -> dict[str, Path]:
    return {
        "validation_metrics_json": ARTIFACT_DIR / f"{tag}_validation_metrics.json",
        "validation_predictions_csv": ARTIFACT_DIR / f"{tag}_validation_predictions.csv",
        "validation_model_json": ARTIFACT_DIR / f"{tag}_validation_model.json",
        "validation_preprocessor_json": ARTIFACT_DIR / f"{tag}_validation_preprocessor.json",
        "validation_bundle_joblib": ARTIFACT_DIR / f"{tag}_validation_bundle.joblib",
        "report_md": REPORT_DIR / f"{tag}.md",
        "full_train_model_json": ARTIFACT_DIR / f"{tag}_full_train_model.json",
        "full_train_preprocessor_json": ARTIFACT_DIR / f"{tag}_full_train_preprocessor.json",
        "full_train_bundle_joblib": ARTIFACT_DIR / f"{tag}_full_train_bundle.joblib",
        "submission_copy_csv": ARTIFACT_DIR / f"{tag}_submission.csv",
        "optuna_best_params_json": ARTIFACT_DIR / f"{tag}_optuna_best_params.json",
        "optuna_trials_csv": ARTIFACT_DIR / f"{tag}_optuna_trials.csv",
        "optuna_summary_json": ARTIFACT_DIR / f"{tag}_optuna_summary.json",
    }


def compute_classification_outputs(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    confusion_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    report_df = (
        pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                labels=CLASS_LABELS,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
        )
        .T.reset_index()
        .rename(columns={"index": "label"})
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_df": confusion_df,
        "report_df": report_df,
    }


def score_from_metric(metric_name: str, *, accuracy: float, macro_f1: float) -> float:
    if metric_name == "accuracy":
        return accuracy
    if metric_name == "macro_f1":
        return macro_f1
    if metric_name == "blend":
        return 0.5 * accuracy + 0.5 * macro_f1
    raise ValueError(f"Unsupported metric_name={metric_name!r}")


def save_model_artifacts(
    classifier: TaskAXGBClassifier,
    *,
    model_json_path: Path,
    preprocessor_json_path: Path,
    bundle_joblib_path: Path,
) -> None:
    ensure_parent_dir(model_json_path)
    ensure_parent_dir(preprocessor_json_path)
    ensure_parent_dir(bundle_joblib_path)

    classifier.model_.save_model(str(model_json_path))
    write_json(preprocessor_json_path, classifier.prep_)
    bundle_payload = {
        "prep": classifier.prep_,
        "fit_metadata": asdict(classifier.fit_metadata_),
        "classes": classifier.classes_.tolist(),
        "model_path": str(model_json_path),
    }
    if joblib is not None:
        joblib.dump(bundle_payload, bundle_joblib_path)
    else:
        with bundle_joblib_path.open("wb") as fh:
            pickle.dump(bundle_payload, fh)


def build_submission(
    *,
    sample_submission: pd.DataFrame,
    risktier_pred: np.ndarray,
    preserve_interest_rate: bool,
) -> tuple[pd.DataFrame, str]:
    submission = sample_submission[["Id"]].copy()
    submission["RiskTier"] = risktier_pred.astype(int)

    interest_rate_source = "sample_submission.csv"
    if preserve_interest_rate and SUBMISSION_PATH.exists():
        existing_submission = pd.read_csv(SUBMISSION_PATH)
        if "InterestRate" in existing_submission.columns and len(existing_submission) == len(submission):
            submission["InterestRate"] = existing_submission["InterestRate"].to_numpy()
            interest_rate_source = str(SUBMISSION_PATH)
    elif "InterestRate" in sample_submission.columns:
        submission["InterestRate"] = sample_submission["InterestRate"].to_numpy()

    ordered_cols = ["Id", "RiskTier"] + ([col for col in ["InterestRate"] if col in submission.columns])
    return submission[ordered_cols], interest_rate_source


def write_train_report(
    *,
    config: TrainEvalConfig,
    metrics_payload: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    confusion_df = pd.DataFrame(metrics_payload["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    report_df = pd.DataFrame(metrics_payload["classification_report"])

    report_lines = [
        "# Task A XGBoost Classifier",
        "",
        "## Objective",
        "Train a single-model Task A classifier using `xgboost.XGBClassifier` while reusing the existing Task A preprocessing and submission I/O contract from the current ensemble notebook.",
        "",
        "## Assumptions About The Existing Interface",
        "- Raw feature input is a pandas `DataFrame` with the same schema as `credit_train.csv` minus `RiskTier` and `InterestRate`.",
        "- Preprocessing is done through the existing Task A `fit_task_a_preprocessor` / `transform_task_a` contract and is not modified here.",
        "- `RiskTier` labels are integer-encoded classes `[0, 1, 2, 3, 4]`.",
        "- Submission writing preserves an existing `InterestRate` column from `submissions/submission.csv` when available.",
        "",
        "## Training Config",
        "```json",
        json.dumps(asdict(config), indent=2, default=_json_default),
        "```",
        "",
        "## Validation Metrics",
        f"- Accuracy: `{metrics_payload['accuracy']:.4f}`",
        f"- Macro F1: `{metrics_payload['macro_f1']:.4f}`",
        f"- Selected score (`{config.metric_name}`): `{metrics_payload['selected_score']:.4f}`",
        f"- Early stopping used: `{metrics_payload['fit_metadata']['early_stopping_used']}`",
        f"- Early stopping source: `{metrics_payload['fit_metadata']['early_stopping_source']}`",
        f"- Best iteration: `{metrics_payload['fit_metadata']['best_iteration']}`",
        f"- Best n_estimators used for final model: `{metrics_payload['fit_metadata']['best_n_estimators']}`",
        f"- Feature count after preprocessing: `{metrics_payload['fit_metadata']['feature_count']}`",
        "",
        "## Confusion Matrix",
        dataframe_to_markdown(confusion_df, index=True, float_digits=0),
        "",
        "## Classification Report",
        dataframe_to_markdown(report_df, index=False),
        "",
        "## Saved Artifacts",
        f"- Validation metrics JSON: `{artifact_paths['validation_metrics_json']}`",
        f"- Validation predictions CSV: `{artifact_paths['validation_predictions_csv']}`",
        f"- Validation model JSON: `{artifact_paths['validation_model_json']}`",
        f"- Validation preprocessor JSON: `{artifact_paths['validation_preprocessor_json']}`",
        f"- Validation bundle: `{artifact_paths['validation_bundle_joblib']}`",
    ]

    if metrics_payload.get("submission_path"):
        report_lines.extend(
            [
                f"- Submission CSV: `{metrics_payload['submission_path']}`",
                f"- Artifact copy of submission: `{artifact_paths['submission_copy_csv']}`",
                f"- Full-train model JSON: `{artifact_paths['full_train_model_json']}`",
                f"- Full-train preprocessor JSON: `{artifact_paths['full_train_preprocessor_json']}`",
                f"- Full-train bundle: `{artifact_paths['full_train_bundle_joblib']}`",
            ]
        )

    ensure_parent_dir(artifact_paths["report_md"])
    artifact_paths["report_md"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_train_validation_experiment(config: TrainEvalConfig) -> dict[str, Any]:
    loaded = load_task_a_data()
    X_train, X_val, y_train, y_val = split_train_validation(
        loaded["X"],
        loaded["y"],
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    started = time.time()
    classifier = TaskAXGBClassifier(
        xgb_params=config.xgb_params,
        random_state=config.random_state,
        model_threads=config.model_threads,
        enable_early_stopping=config.enable_early_stopping,
        early_stopping_rounds=config.early_stopping_rounds,
        internal_early_stopping_size=config.internal_early_stopping_size,
        use_scale_pos_weight=config.use_scale_pos_weight,
        verbose=config.verbose,
    )
    classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    y_val_pred = classifier.predict(X_val)
    elapsed_seconds = round(time.time() - started, 3)

    metrics = compute_classification_outputs(y_val, y_val_pred)
    selected_score = score_from_metric(
        config.metric_name,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
    )

    artifact_paths = build_artifact_paths(config.artifact_tag)
    validation_predictions = X_val.copy()
    validation_predictions["y_true"] = y_val.to_numpy()
    validation_predictions["y_pred"] = y_val_pred
    ensure_parent_dir(artifact_paths["validation_predictions_csv"])
    validation_predictions.to_csv(artifact_paths["validation_predictions_csv"], index=False)

    save_model_artifacts(
        classifier,
        model_json_path=artifact_paths["validation_model_json"],
        preprocessor_json_path=artifact_paths["validation_preprocessor_json"],
        bundle_joblib_path=artifact_paths["validation_bundle_joblib"],
    )

    payload = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "selected_score": selected_score,
        "elapsed_seconds": elapsed_seconds,
        "fit_metadata": asdict(classifier.fit_metadata_),
        "validation_rows": int(len(X_val)),
        "train_rows": int(len(X_train)),
        "confusion_matrix": metrics["confusion_df"].to_numpy().tolist(),
        "classification_report": metrics["report_df"].to_dict(orient="records"),
        "config": asdict(config),
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
    }
    write_json(artifact_paths["validation_metrics_json"], payload)

    if config.run_full_train_prediction:
        final_classifier = TaskAXGBClassifier(
            xgb_params=config.xgb_params,
            random_state=config.random_state,
            model_threads=config.model_threads,
            enable_early_stopping=config.enable_early_stopping,
            early_stopping_rounds=config.early_stopping_rounds,
            internal_early_stopping_size=config.internal_early_stopping_size,
            use_scale_pos_weight=config.use_scale_pos_weight,
            verbose=config.verbose,
        )
        final_classifier.fit(loaded["X"], loaded["y"], refit_on_full_train=True)
        test_pred = final_classifier.predict(loaded["test_df"])
        submission, interest_rate_source = build_submission(
            sample_submission=loaded["sample_submission"],
            risktier_pred=test_pred,
            preserve_interest_rate=config.preserve_interest_rate,
        )
        ensure_parent_dir(SUBMISSION_PATH)
        submission.to_csv(SUBMISSION_PATH, index=False)
        submission.to_csv(artifact_paths["submission_copy_csv"], index=False)
        save_model_artifacts(
            final_classifier,
            model_json_path=artifact_paths["full_train_model_json"],
            preprocessor_json_path=artifact_paths["full_train_preprocessor_json"],
            bundle_joblib_path=artifact_paths["full_train_bundle_joblib"],
        )
        payload["submission_path"] = str(SUBMISSION_PATH)
        payload["artifact_submission_copy_path"] = str(artifact_paths["submission_copy_csv"])
        payload["interest_rate_source"] = interest_rate_source
        payload["submission_distribution"] = (
            submission["RiskTier"].value_counts().sort_index().rename_axis("RiskTier").reset_index(name="count")
        ).to_dict(orient="records")
        payload["full_train_fit_metadata"] = asdict(final_classifier.fit_metadata_)
        write_json(artifact_paths["validation_metrics_json"], payload)

    write_train_report(config=config, metrics_payload=payload, artifact_paths=artifact_paths)
    return payload


def suggest_xgb_params(
    trial: Any,
    *,
    include_scale_pos_weight: bool,
    fixed_params: dict[str, Any],
    n_classes: int,
) -> dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 25.0, log=True),
    }
    params.update(fixed_params)

    if include_scale_pos_weight and n_classes == 2:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, 10.0, log=True)
    return params


def build_optuna_objective(
    *,
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    config: OptunaConfig,
) -> Any:
    def objective(trial: Any) -> float:
        params = suggest_xgb_params(
            trial,
            include_scale_pos_weight=config.include_scale_pos_weight,
            fixed_params=config.fixed_params,
            n_classes=int(y_raw.nunique()),
        )
        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )

        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_raw, y_raw), start=1):
            X_train_fold = X_raw.iloc[train_idx].reset_index(drop=True)
            X_val_fold = X_raw.iloc[val_idx].reset_index(drop=True)
            y_train_fold = y_raw.iloc[train_idx].reset_index(drop=True).astype(int)
            y_val_fold = y_raw.iloc[val_idx].reset_index(drop=True).astype(int)

            model = TaskAXGBClassifier(
                xgb_params=params,
                random_state=config.random_state,
                model_threads=config.model_threads,
                enable_early_stopping=config.enable_early_stopping,
                early_stopping_rounds=config.early_stopping_rounds,
                internal_early_stopping_size=config.internal_early_stopping_size,
                use_scale_pos_weight=config.include_scale_pos_weight,
                verbose=config.verbose,
            )
            # Keep the outer fold validation set strictly for scoring.
            model.fit(X_train_fold, y_train_fold, refit_on_full_train=True)
            y_pred = model.predict(X_val_fold)

            accuracy = float(accuracy_score(y_val_fold, y_pred))
            macro_f1 = float(f1_score(y_val_fold, y_pred, average="macro"))
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "selected_score": score_from_metric(
                        config.metric_name,
                        accuracy=accuracy,
                        macro_f1=macro_f1,
                    ),
                    "best_n_estimators": model.fit_metadata_.best_n_estimators,
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        mean_accuracy = float(fold_df["accuracy"].mean())
        mean_macro_f1 = float(fold_df["macro_f1"].mean())
        mean_score = float(fold_df["selected_score"].mean())

        trial.set_user_attr("mean_accuracy", mean_accuracy)
        trial.set_user_attr("mean_macro_f1", mean_macro_f1)
        trial.set_user_attr("mean_selected_score", mean_score)
        trial.set_user_attr("fold_rows", fold_rows)
        return mean_score

    return objective


def run_optuna_study(config: OptunaConfig) -> dict[str, Any]:
    _require_optuna()
    _require_xgboost()

    loaded = load_task_a_data()
    artifact_paths = build_artifact_paths(config.artifact_tag)

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
        sampler=sampler,
    )
    objective = build_optuna_objective(
        X_raw=loaded["X"],
        y_raw=loaded["y"],
        config=config,
    )

    started = time.time()
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout, show_progress_bar=False)
    elapsed_seconds = round(time.time() - started, 3)

    trials_df = study.trials_dataframe()
    ensure_parent_dir(artifact_paths["optuna_trials_csv"])
    trials_df.to_csv(artifact_paths["optuna_trials_csv"], index=False)

    best_trial = study.best_trial
    best_payload = {
        "study_name": config.study_name,
        "metric_name": config.metric_name,
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "trials_csv": str(artifact_paths["optuna_trials_csv"]),
    }

    write_json(artifact_paths["optuna_best_params_json"], best_trial.params)
    write_json(artifact_paths["optuna_summary_json"], best_payload)
    return best_payload


def parse_key_value_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides


def build_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="run train/validation evaluation and optional test prediction")
    parser.add_argument("--artifact-tag", default="taskA_xgb_optuna")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--model-threads", type=int, default=2)
    parser.add_argument("--metric-name", default="accuracy", choices=["accuracy", "macro_f1", "blend"])
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--internal-early-stopping-size", type=float, default=0.1)
    parser.add_argument("--disable-full-train-prediction", action="store_true")
    parser.add_argument("--disable-interest-rate-preservation", action="store_true")
    parser.add_argument("--use-scale-pos-weight", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--xgb-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an XGBoost parameter, for example --xgb-param n_estimators=800",
    )


def build_tune_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("tune", help="run Optuna Bayesian optimization on Task A")
    parser.add_argument("--study-name", default="taskA_xgb_optuna")
    parser.add_argument("--artifact-tag", default="taskA_xgb_optuna")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--metric-name", default="accuracy", choices=["accuracy", "macro_f1", "blend"])
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--model-threads", type=int, default=2)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--internal-early-stopping-size", type=float, default=0.1)
    parser.add_argument("--include-scale-pos-weight", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="fix an XGBoost parameter during tuning, for example --fixed-param tree_method=\"hist\"",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task A XGBoost training and Optuna tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_train_parser(subparsers)
    build_tune_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainEvalConfig(
            artifact_tag=args.artifact_tag,
            validation_size=args.validation_size,
            random_state=args.random_state,
            model_threads=args.model_threads,
            verbose=args.verbose,
            metric_name=args.metric_name,
            enable_early_stopping=not args.disable_early_stopping,
            early_stopping_rounds=args.early_stopping_rounds,
            internal_early_stopping_size=args.internal_early_stopping_size,
            run_full_train_prediction=not args.disable_full_train_prediction,
            preserve_interest_rate=not args.disable_interest_rate_preservation,
            save_model_bundle=True,
            xgb_params={**DEFAULT_XGB_PARAMS, **parse_key_value_overrides(args.xgb_param)},
            use_scale_pos_weight=args.use_scale_pos_weight,
        )
        result = run_train_validation_experiment(config)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0

    if args.command == "tune":
        config = OptunaConfig(
            study_name=args.study_name,
            artifact_tag=args.artifact_tag,
            n_trials=args.n_trials,
            timeout=args.timeout,
            cv_folds=args.cv_folds,
            metric_name=args.metric_name,
            random_state=args.random_state,
            model_threads=args.model_threads,
            verbose=args.verbose,
            enable_early_stopping=not args.disable_early_stopping,
            early_stopping_rounds=args.early_stopping_rounds,
            internal_early_stopping_size=args.internal_early_stopping_size,
            include_scale_pos_weight=args.include_scale_pos_weight,
            fixed_params=parse_key_value_overrides(args.fixed_param),
        )
        result = run_optuna_study(config)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
