#!/usr/bin/env python3
"""
Hyperparameter optimization for Task A using SPSA-style gradient descent.

Why SPSA instead of plain gradient descent:
- The Task A model stack contains tree ensembles and rounded integer hyperparameters.
- Validation accuracy / macro F1 is not differentiable with respect to those parameters.
- SPSA is still a gradient-descent family method, but it estimates a usable descent
  direction from a small number of objective evaluations.

Better optimizer for this problem:
- Optuna with a TPE sampler
- Bayesian optimization / SMAC

Those methods are usually a better fit for mixed discrete + continuous search spaces.
This script keeps the requested gradient-descent-style approach while staying
mathematically honest about the underlying objective.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    lower: float
    upper: float
    log_scale: bool = False


PARAM_SPECS = [
    ParamSpec("rf_n_estimators", "int", 200, 600),
    ParamSpec("rf_min_samples_leaf", "int", 1, 8),
    ParamSpec("xgb_n_estimators", "int", 200, 500),
    ParamSpec("xgb_learning_rate", "float", 0.01, 0.15, log_scale=True),
    ParamSpec("xgb_max_depth", "int", 3, 10),
    ParamSpec("xgb_subsample", "float", 0.6, 1.0),
    ParamSpec("xgb_colsample_bytree", "float", 0.6, 1.0),
    ParamSpec("lgbm_n_estimators", "int", 200, 500),
    ParamSpec("lgbm_learning_rate", "float", 0.01, 0.15, log_scale=True),
    ParamSpec("lgbm_num_leaves", "int", 16, 64),
    ParamSpec("lgbm_subsample", "float", 0.6, 1.0),
    ParamSpec("lgbm_colsample_bytree", "float", 0.6, 1.0),
]


DEFAULT_PARAMS = {
    "rf_n_estimators": 300,
    "rf_min_samples_leaf": 2,
    "xgb_n_estimators": 350,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 6,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.8,
    "lgbm_n_estimators": 350,
    "lgbm_learning_rate": 0.05,
    "lgbm_num_leaves": 31,
    "lgbm_subsample": 0.9,
    "lgbm_colsample_bytree": 0.8,
}


class LinearStackedRiskTier:
    """Matches the current Task A notebook stacker."""

    def __init__(
        self,
        base_models: dict[str, Any],
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.base_models = base_models
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.meta_model = LinearRegression()
        self.fitted_models_: dict[str, Any] = {}
        self.classes_: np.ndarray | None = None
        self.oof_pred_: np.ndarray | None = None

    def _to_class_labels(self, raw_pred: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return np.clip(np.rint(raw_pred), self.classes_.min(), self.classes_.max()).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearStackedRiskTier":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.classes_ = np.sort(y.unique())
        n_classes = len(self.classes_)
        n_models = len(self.base_models)

        oof_meta = np.zeros((len(X), n_models * n_classes), dtype=float)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            start = model_idx * n_classes
            end = start + n_classes
            if self.verbose:
                print(f"  [stack] training {name}", flush=True)

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
                if self.verbose:
                    print(f"    fold {fold_idx}/{self.n_splits}", flush=True)
                fold_model = clone(model)
                fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                oof_meta[va_idx, start:end] = fold_model.predict_proba(X.iloc[va_idx])

            final_model = clone(model)
            final_model.fit(X, y)
            self.fitted_models_[name] = final_model

        self.meta_model.fit(oof_meta, y)
        self.oof_pred_ = self._to_class_labels(self.meta_model.predict(oof_meta))
        return self

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        proba_blocks = [model.predict_proba(X) for model in self.fitted_models_.values()]
        return np.hstack(proba_blocks)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_pred = self.meta_model.predict(self._meta_features(X))
        return self._to_class_labels(raw_pred)


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


def build_task_a_base_models(
    params: dict[str, float | int],
    random_state: int,
    model_threads: int,
) -> dict[str, Any]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=int(params["rf_n_estimators"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=model_threads,
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=int(params["xgb_n_estimators"]),
            learning_rate=float(params["xgb_learning_rate"]),
            max_depth=int(params["xgb_max_depth"]),
            subsample=float(params["xgb_subsample"]),
            colsample_bytree=float(params["xgb_colsample_bytree"]),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=model_threads,
        ),
        "lgbm": LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=int(params["lgbm_n_estimators"]),
            learning_rate=float(params["lgbm_learning_rate"]),
            num_leaves=int(params["lgbm_num_leaves"]),
            subsample=float(params["lgbm_subsample"]),
            colsample_bytree=float(params["lgbm_colsample_bytree"]),
            random_state=random_state,
            n_jobs=model_threads,
            verbosity=-1,
        ),
    }


def normalize_value(spec: ParamSpec, value: float | int) -> float:
    value = float(value)
    if spec.log_scale:
        low = math.log(spec.lower)
        high = math.log(spec.upper)
        return (math.log(value) - low) / (high - low)
    return (value - spec.lower) / (spec.upper - spec.lower)


def denormalize_value(spec: ParamSpec, theta: float) -> float | int:
    theta = float(np.clip(theta, 0.0, 1.0))
    if spec.log_scale:
        low = math.log(spec.lower)
        high = math.log(spec.upper)
        value = math.exp(low + theta * (high - low))
    else:
        value = spec.lower + theta * (spec.upper - spec.lower)

    value = min(max(value, spec.lower), spec.upper)
    if spec.kind == "int":
        return int(round(value))
    return float(value)


def params_to_theta(params: dict[str, float | int]) -> np.ndarray:
    return np.array([normalize_value(spec, params[spec.name]) for spec in PARAM_SPECS], dtype=float)


def theta_to_params(theta: np.ndarray) -> dict[str, float | int]:
    return {spec.name: denormalize_value(spec, value) for spec, value in zip(PARAM_SPECS, theta)}


def params_cache_key(params: dict[str, float | int]) -> tuple[tuple[str, float | int], ...]:
    return tuple(sorted(params.items()))


def metric_score(accuracy: float, macro_f1: float, metric_name: str) -> float:
    if metric_name == "accuracy":
        return accuracy
    if metric_name == "macro_f1":
        return macro_f1
    if metric_name == "blend":
        return 0.5 * accuracy + 0.5 * macro_f1
    raise ValueError(f"Unknown metric: {metric_name}")


def load_task_a_split(
    train_path: str,
    *,
    validation_size: float,
    train_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(train_path)
    X = df.drop(["RiskTier", "InterestRate"], axis=1)
    y_cls = df["RiskTier"]

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y_cls,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_cls,
    )

    if train_fraction < 1.0:
        X_train_raw, _, y_train, _ = train_test_split(
            X_train_raw,
            y_train,
            train_size=train_fraction,
            random_state=random_state,
            stratify=y_train,
        )

    prep = fit_task_a_preprocessor(X_train_raw)
    X_train = transform_task_a(X_train_raw, prep)
    X_val = transform_task_a(X_val_raw, prep)

    if X_train.isnull().sum().sum() != 0 or X_val.isnull().sum().sum() != 0:
        raise ValueError("Task A preprocessing left missing values in the optimization split.")
    if not X_train.columns.equals(X_val.columns):
        raise ValueError("Task A train/validation schemas do not match after preprocessing.")

    return X_train, X_val, y_train.reset_index(drop=True), y_val.reset_index(drop=True)


def evaluate_params(
    params: dict[str, float | int],
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    metric_name: str,
    stacking_folds: int,
    random_state: int,
    model_threads: int,
    verbose: bool,
    cache: dict[tuple[tuple[str, float | int], ...], dict[str, Any]],
) -> dict[str, Any]:
    key = params_cache_key(params)
    if key in cache:
        return cache[key]

    started = time.time()
    if verbose:
        compact = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"[eval] start {compact}", flush=True)
    clf = LinearStackedRiskTier(
        base_models=build_task_a_base_models(
            params,
            random_state=random_state,
            model_threads=model_threads,
        ),
        n_splits=stacking_folds,
        random_state=random_state,
        verbose=verbose,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    accuracy = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    score = float(metric_score(accuracy, macro_f1, metric_name))
    result = {
        "params": dict(params),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "score": score,
        "loss": 1.0 - score,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if verbose:
        print(
            f"[eval] done score={result['score']:.5f} "
            f"acc={result['accuracy']:.5f} macro_f1={result['macro_f1']:.5f} "
            f"elapsed={result['elapsed_sec']:.3f}s",
            flush=True,
        )
    cache[key] = result
    return result


def run_spsa(
    *,
    steps: int,
    base_lr: float,
    perturb_scale: float,
    metric_name: str,
    stacking_folds: int,
    random_state: int,
    model_threads: int,
    verbose: bool,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(random_state)
    cache: dict[tuple[tuple[str, float | int], ...], dict[str, Any]] = {}

    theta = np.clip(params_to_theta(DEFAULT_PARAMS), 0.0, 1.0)
    current = evaluate_params(
        theta_to_params(theta),
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        metric_name=metric_name,
        stacking_folds=stacking_folds,
        random_state=random_state,
        model_threads=model_threads,
        verbose=verbose,
        cache=cache,
    )
    best = current
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "initial",
            "score": current["score"],
            "accuracy": current["accuracy"],
            "macro_f1": current["macro_f1"],
            "params": current["params"],
        }
    ]

    for step_idx in range(1, steps + 1):
        lr = base_lr / math.sqrt(step_idx)
        ck = perturb_scale / (step_idx ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=len(theta))

        theta_plus = np.clip(theta + ck * delta, 0.0, 1.0)
        theta_minus = np.clip(theta - ck * delta, 0.0, 1.0)

        result_plus = evaluate_params(
            theta_to_params(theta_plus),
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            metric_name=metric_name,
            stacking_folds=stacking_folds,
            random_state=random_state,
            model_threads=model_threads,
            verbose=verbose,
            cache=cache,
        )
        result_minus = evaluate_params(
            theta_to_params(theta_minus),
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            metric_name=metric_name,
            stacking_folds=stacking_folds,
            random_state=random_state,
            model_threads=model_threads,
            verbose=verbose,
            cache=cache,
        )

        grad_hat = ((result_plus["loss"] - result_minus["loss"]) / (2.0 * ck)) * delta
        theta = np.clip(theta - lr * grad_hat, 0.0, 1.0)

        current = evaluate_params(
            theta_to_params(theta),
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            metric_name=metric_name,
            stacking_folds=stacking_folds,
            random_state=random_state,
            model_threads=model_threads,
            verbose=verbose,
            cache=cache,
        )
        if current["score"] > best["score"]:
            best = current

        history.append(
            {
                "step": step_idx,
                "kind": "spsa",
                "score": current["score"],
                "accuracy": current["accuracy"],
                "macro_f1": current["macro_f1"],
                "learning_rate": lr,
                "perturb_scale": ck,
                "params": current["params"],
                "best_score_so_far": best["score"],
            }
        )

        print(
            f"[step {step_idx:02d}] score={current['score']:.5f} "
            f"acc={current['accuracy']:.5f} macro_f1={current['macro_f1']:.5f} "
            f"best={best['score']:.5f}"
        )

    return best, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize Task A stack hyperparameters with SPSA-style gradient descent."
    )
    parser.add_argument(
        "--train-path",
        default="creditsense-ai1215/credit_train.csv",
        help="Path to the training CSV.",
    )
    parser.add_argument(
        "--save-path",
        default="taskA_hyperparams_spsa_results.json",
        help="Where to save the best result JSON.",
    )
    parser.add_argument(
        "--metric",
        choices=["blend", "accuracy", "macro_f1"],
        default="blend",
        help="Optimization target. 'blend' = 0.5 * accuracy + 0.5 * macro_f1.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of SPSA update steps. Each step evaluates the model three times.",
    )
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=0.04,
        help="Base SPSA learning rate in normalized parameter space.",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.12,
        help="Base SPSA perturbation scale in normalized parameter space.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split size.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Optional fraction of the training split to use during tuning for faster runs.",
    )
    parser.add_argument(
        "--stacking-folds",
        type=int,
        default=3,
        help="Stacking folds during optimization. Lower is faster; 5 matches the notebook.",
    )
    parser.add_argument(
        "--model-threads",
        type=int,
        default=1,
        help="Thread count for each base model during tuning. Keep this low to avoid runtime kills.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-evaluation logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_val, y_train, y_val = load_task_a_split(
        args.train_path,
        validation_size=args.validation_size,
        train_fraction=args.train_fraction,
        random_state=args.seed,
    )

    print("Task A tuning split loaded.")
    print(f"Train matrix: {X_train.shape} | Validation matrix: {X_val.shape}")
    print(
        "Optimizer: SPSA-style gradient descent "
        f"(metric={args.metric}, steps={args.steps}, lr={args.optimizer_lr}, "
        f"perturb_scale={args.perturb_scale}, stacking_folds={args.stacking_folds}, "
        f"model_threads={args.model_threads})"
    )
    print("Recommendation: Optuna/TPE or Bayesian optimization is usually better here.")
    print()

    best, history = run_spsa(
        steps=args.steps,
        base_lr=args.optimizer_lr,
        perturb_scale=args.perturb_scale,
        metric_name=args.metric,
        stacking_folds=args.stacking_folds,
        random_state=args.seed,
        model_threads=args.model_threads,
        verbose=not args.quiet,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
    )

    payload = {
        "method": "spsa_hyperparameter_descent",
        "why_not_plain_gradient_descent": (
            "Task A uses tree ensembles and integer hyperparameters, so the validation "
            "objective is not differentiable. SPSA provides a gradient-descent-like update "
            "using stochastic finite-difference estimates."
        ),
        "recommended_better_optimizer": "Optuna with TPE sampler or Bayesian optimization",
        "metric": args.metric,
        "steps": args.steps,
        "optimizer_lr": args.optimizer_lr,
        "perturb_scale": args.perturb_scale,
        "stacking_folds": args.stacking_folds,
        "model_threads": args.model_threads,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "best_result": best,
        "default_params": DEFAULT_PARAMS,
        "search_space": [spec.__dict__ for spec in PARAM_SPECS],
        "history": history,
    }

    Path(args.save_path).write_text(json.dumps(payload, indent=2))

    print()
    print("Best validation result:")
    print(
        f"score={best['score']:.5f} "
        f"acc={best['accuracy']:.5f} "
        f"macro_f1={best['macro_f1']:.5f}"
    )
    print("Best params:")
    print(json.dumps(best["params"], indent=2))
    print(f"Saved results to {args.save_path}")


if __name__ == "__main__":
    main()
