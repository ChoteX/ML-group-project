#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
MODEL_THREADS = -1
CLASS_LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES = ["VeryLow(0)", "Low(1)", "Moderate(2)", "High(3)", "VeryHigh(4)"]

DATA_DIR = Path("creditsense-ai1215")
TRAIN_PATH = DATA_DIR / "credit_train.csv"

NN_REPORT_PATH = Path("taskA_nn.md")
RESULTS_JSON_PATH = Path("taskA_tree_catboost_cv_results.json")
RESULTS_MD_PATH = Path("taskA_tree_catboost_vs_nn.md")


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


def fit_task_a_catboost_preprocessor(X_train_raw: pd.DataFrame) -> dict[str, Any]:
    X_train_raw = X_train_raw.copy()

    base_cols = X_train_raw.columns.tolist()
    num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_flag_cols = [col for col in base_cols if X_train_raw[col].isna().any()]

    zero_fill_cols = [col for col in TASK_A_ZERO_FILL_COLS if col in num_cols]
    clip_cols = [col for col in TASK_A_MONEY_CLIP_COLS + TASK_A_RATIO_CLIP_COLS if col in num_cols]
    log_cols = [col for col in TASK_A_LOG_COLS if col in num_cols]

    clip_map: dict[str, float] = {}
    for col in clip_cols:
        non_null = X_train_raw[col].dropna()
        clip_map[col] = float(non_null.quantile(0.99)) if not non_null.empty else 0.0

    prep = {
        "base_cols": base_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing_flag_cols": missing_flag_cols,
        "zero_fill_cols": zero_fill_cols,
        "clip_map": clip_map,
        "log_cols": log_cols,
    }

    transformed_train = transform_task_a_catboost(X_train_raw, prep)
    prep["feature_columns"] = transformed_train.columns.tolist()
    return prep


def transform_task_a_catboost(X_raw: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    working = X_raw.copy()

    for col in prep["missing_flag_cols"]:
        working[f"{col}_is_missing"] = working[col].isna().astype(np.int8)

    for col in prep["cat_cols"]:
        working[col] = working[col].fillna("Missing").astype(str)

    for col in prep["zero_fill_cols"]:
        working[col] = working[col].fillna(0.0)

    for col, upper in prep["clip_map"].items():
        working[col] = working[col].clip(upper=upper)

    for col in prep["log_cols"]:
        working[col] = np.log1p(working[col])

    for col in prep["num_cols"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    missing_cols = [f"{col}_is_missing" for col in prep["missing_flag_cols"]]
    ordered_cols = prep["base_cols"] + missing_cols
    transformed = working.reindex(columns=ordered_cols)
    if "feature_columns" in prep:
        transformed = transformed.reindex(columns=prep["feature_columns"], fill_value=0)
    return transformed


def build_onehot_base_model_builders(
    random_state: int,
    model_threads: int,
) -> dict[str, Callable[[], Any]]:
    return {
        "rf": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=model_threads,
        ),
        "xgb": lambda: XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=350,
            learning_rate=0.05011952008381389,
            max_depth=6,
            subsample=0.8996473404798764,
            colsample_bytree=0.8001695969176756,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=model_threads,
        ),
        "lgbm": lambda: LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=350,
            learning_rate=0.0500574425921408,
            num_leaves=31,
            subsample=0.8998304030823245,
            colsample_bytree=0.7998642491270678,
            random_state=random_state,
            n_jobs=model_threads,
            verbosity=-1,
        ),
    }


def build_catboost_model(random_state: int, model_threads: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=350,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        thread_count=model_threads,
        verbose=False,
        allow_writing_files=False,
    )


@dataclass(frozen=True)
class BaseModelDef:
    name: str
    fit_preprocessor: Callable[[pd.DataFrame], dict[str, Any]]
    transform: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]
    build_model: Callable[[], Any]
    uses_native_cat: bool = False


class HeterogeneousLinearStackedRiskTier:
    def __init__(
        self,
        *,
        base_model_defs: list[BaseModelDef],
        n_splits: int,
        random_state: int,
    ) -> None:
        self.base_model_defs = base_model_defs
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_model = LinearRegression()
        self.classes_: np.ndarray | None = None
        self.fitted_models_: dict[str, Any] = {}
        self.fitted_preprocessors_: dict[str, dict[str, Any]] = {}
        self.oof_pred_: np.ndarray | None = None

    def _to_class_labels(self, raw_pred: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return np.clip(np.rint(raw_pred), self.classes_.min(), self.classes_.max()).astype(int)

    def _fit_single_model(
        self,
        base_model_def: BaseModelDef,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        prep = base_model_def.fit_preprocessor(X_train_raw)
        X_train = base_model_def.transform(X_train_raw, prep)
        model = base_model_def.build_model()

        if base_model_def.uses_native_cat:
            model.fit(X_train, y_train, cat_features=prep["cat_cols"])
        else:
            model.fit(X_train, y_train)
        return model, prep, X_train

    def fit(self, X_raw: pd.DataFrame, y: pd.Series) -> "HeterogeneousLinearStackedRiskTier":
        X_raw = X_raw.reset_index(drop=True)
        y = y.reset_index(drop=True)

        self.classes_ = np.sort(y.unique())
        n_classes = len(self.classes_)
        oof_meta = np.zeros((len(X_raw), len(self.base_model_defs) * n_classes), dtype=float)

        inner_skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for model_idx, base_model_def in enumerate(self.base_model_defs):
            start = model_idx * n_classes
            end = start + n_classes

            for inner_train_idx, inner_val_idx in inner_skf.split(X_raw, y):
                X_inner_train_raw = X_raw.iloc[inner_train_idx].reset_index(drop=True)
                X_inner_val_raw = X_raw.iloc[inner_val_idx].reset_index(drop=True)
                y_inner_train = y.iloc[inner_train_idx].reset_index(drop=True)

                model, prep, _ = self._fit_single_model(base_model_def, X_inner_train_raw, y_inner_train)
                X_inner_val = base_model_def.transform(X_inner_val_raw, prep)
                oof_meta[inner_val_idx, start:end] = model.predict_proba(X_inner_val)

            final_model, final_prep, _ = self._fit_single_model(base_model_def, X_raw, y)
            self.fitted_models_[base_model_def.name] = final_model
            self.fitted_preprocessors_[base_model_def.name] = final_prep

        self.meta_model.fit(oof_meta, y)
        self.oof_pred_ = self._to_class_labels(self.meta_model.predict(oof_meta))
        return self

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        meta_blocks = []
        for base_model_def in self.base_model_defs:
            prep = self.fitted_preprocessors_[base_model_def.name]
            model = self.fitted_models_[base_model_def.name]
            X_model = base_model_def.transform(X_raw.reset_index(drop=True), prep)
            meta_blocks.append(model.predict_proba(X_model))
        meta_features = np.hstack(meta_blocks)
        raw_pred = self.meta_model.predict(meta_features)
        return self._to_class_labels(raw_pred)


def build_model_defs(random_state: int, model_threads: int) -> list[BaseModelDef]:
    onehot_model_builders = build_onehot_base_model_builders(
        random_state=random_state,
        model_threads=model_threads,
    )
    model_defs = [
        BaseModelDef(
            name=name,
            fit_preprocessor=fit_task_a_preprocessor,
            transform=transform_task_a,
            build_model=builder,
            uses_native_cat=False,
        )
        for name, builder in onehot_model_builders.items()
    ]
    model_defs.append(
        BaseModelDef(
            name="catboost",
            fit_preprocessor=fit_task_a_catboost_preprocessor,
            transform=transform_task_a_catboost,
            build_model=lambda: build_catboost_model(random_state=random_state, model_threads=model_threads),
            uses_native_cat=True,
        )
    )
    return model_defs


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


def parse_nn_summary(report_path: Path) -> pd.DataFrame:
    text = report_path.read_text(encoding="utf-8")

    accuracy_match = re.search(r"\| accuracy \| ([0-9.]+) \| ([0-9.]+) \|", text)
    macro_f1_match = re.search(r"\| macro_f1 \| ([0-9.]+) \| ([0-9.]+) \|", text)
    if accuracy_match is None or macro_f1_match is None:
        raise ValueError(f"Could not parse NN summary metrics from {report_path}")

    return pd.DataFrame(
        [
            {
                "model": "FC neural network",
                "accuracy_mean": float(accuracy_match.group(1)),
                "accuracy_std": float(accuracy_match.group(2)),
                "macro_f1_mean": float(macro_f1_match.group(1)),
                "macro_f1_std": float(macro_f1_match.group(2)),
            }
        ]
    )


def run_outer_cv(X_raw: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    outer_skf = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    started = time.time()
    oof_pred = np.full(len(X_raw), -1, dtype=int)
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_skf.split(X_raw, y), start=1):
        X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        stacker = HeterogeneousLinearStackedRiskTier(
            base_model_defs=build_model_defs(random_state=RANDOM_STATE, model_threads=MODEL_THREADS),
            n_splits=INNER_FOLDS,
            random_state=RANDOM_STATE,
        )
        stacker.fit(X_train_raw, y_train)
        val_pred = stacker.predict(X_val_raw)
        oof_pred[val_idx] = val_pred

        fold_accuracy = float(accuracy_score(y_val, val_pred))
        fold_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "accuracy": fold_accuracy,
                "macro_f1": fold_macro_f1,
            }
        )
        print(
            f"Fold {fold_idx}/{OUTER_FOLDS}: "
            f"accuracy={fold_accuracy:.4f}, macro_f1={fold_macro_f1:.4f}",
            flush=True,
        )

    if (oof_pred < 0).any():
        raise ValueError("Outer CV predictions did not cover all rows.")

    fold_metrics_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(
        [
            {
                "model": "Tree stack + CatBoost",
                "accuracy_mean": float(fold_metrics_df["accuracy"].mean()),
                "accuracy_std": float(fold_metrics_df["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(fold_metrics_df["macro_f1"].mean()),
                "macro_f1_std": float(fold_metrics_df["macro_f1"].std(ddof=1)),
                "elapsed_minutes": round((time.time() - started) / 60.0, 2),
            }
        ]
    )
    confusion_df = pd.DataFrame(
        confusion_matrix(y.to_numpy(), oof_pred, labels=CLASS_LABELS),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    confusion_df.index.name = "actual"
    classification_df = pd.DataFrame(
        classification_report(
            y.to_numpy(),
            oof_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T
    classification_df.index.name = "label"

    return {
        "fold_metrics_df": fold_metrics_df,
        "summary_df": summary_df,
        "confusion_df": confusion_df,
        "classification_df": classification_df,
    }


def write_report(
    *,
    results: dict[str, Any],
    nn_summary_df: pd.DataFrame,
) -> None:
    tree_summary_df = results["summary_df"].copy()
    comparison_df = pd.concat(
        [
            tree_summary_df.drop(columns=["elapsed_minutes"]),
            nn_summary_df,
        ],
        ignore_index=True,
    )
    comparison_df["accuracy_delta_vs_nn"] = comparison_df["accuracy_mean"] - float(
        nn_summary_df.loc[0, "accuracy_mean"]
    )
    comparison_df["macro_f1_delta_vs_nn"] = comparison_df["macro_f1_mean"] - float(
        nn_summary_df.loc[0, "macro_f1_mean"]
    )

    report = "\n".join(
        [
            "# Task A Tree Stack + CatBoost vs NN",
            "",
            "## Objective",
            "Evaluate a Task A stacked tree ensemble that adds a CatBoost base learner with native categorical handling, then compare the 5-fold cross-validation metrics against the previously executed FC neural network results in `taskA_nn.md`.",
            "",
            "## Tree Stack Configuration",
            "- Meta-learner: `LinearRegression` on concatenated base-model class probabilities, followed by `round` + `clip` to integer labels `[0, 4]`.",
            "- One-hot branch base learners: `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier` using the existing Task A preprocessing.",
            "- Native categorical branch base learner: `CatBoostClassifier(loss_function='MultiClass')` trained on raw categorical columns with missing categories filled as `Missing`.",
            "- CatBoost numeric preparation: structural zero-fill for the Task A zero-fill columns, train-fold clipping for Task A money/ratio columns, `log1p` on Task A money columns, and numeric missingness left as `NaN` where CatBoost can handle it natively.",
            "",
            "## Evaluation Protocol",
            f"- Outer CV: `StratifiedKFold(n_splits={OUTER_FOLDS}, shuffle=True, random_state={RANDOM_STATE})`.",
            f"- Inner stacking CV: `StratifiedKFold(n_splits={INNER_FOLDS}, shuffle=True, random_state={RANDOM_STATE})` inside each outer training fold.",
            "- NN comparison baseline: parsed from the executed `taskA_nn.md` report.",
            "",
            "## Tree Stack + CatBoost Fold Metrics",
            dataframe_to_markdown(results["fold_metrics_df"], index=False, float_digits=4),
            "",
            "## Summary Comparison",
            dataframe_to_markdown(comparison_df, index=False, float_digits=4),
            "",
            "## Tree Stack + CatBoost Confusion Matrix",
            dataframe_to_markdown(results["confusion_df"], index=True, float_digits=0),
            "",
            "## Tree Stack + CatBoost Classification Report",
            dataframe_to_markdown(results["classification_df"], index=True, float_digits=4),
            "",
            "## Interpretation",
            f"- Tree stack + CatBoost mean accuracy: {tree_summary_df.loc[0, 'accuracy_mean']:.4f}",
            f"- FC neural network mean accuracy: {nn_summary_df.loc[0, 'accuracy_mean']:.4f}",
            f"- Tree stack + CatBoost mean macro F1: {tree_summary_df.loc[0, 'macro_f1_mean']:.4f}",
            f"- FC neural network mean macro F1: {nn_summary_df.loc[0, 'macro_f1_mean']:.4f}",
            "",
            "A positive delta in the comparison table means the stacked tree ensemble outperformed the FC neural network under the same 5-fold outer CV setup.",
        ]
    )
    RESULTS_MD_PATH.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=["RiskTier", "InterestRate"])
    y = train_df["RiskTier"].astype(int)

    print(f"Loaded train data: {train_df.shape}", flush=True)
    results = run_outer_cv(X, y)
    nn_summary_df = parse_nn_summary(NN_REPORT_PATH)

    serializable_results = {
        "fold_metrics": results["fold_metrics_df"].to_dict(orient="records"),
        "summary": results["summary_df"].to_dict(orient="records"),
        "confusion_matrix": results["confusion_df"].reset_index().to_dict(orient="records"),
        "classification_report": results["classification_df"].reset_index().to_dict(orient="records"),
        "nn_summary": nn_summary_df.to_dict(orient="records"),
    }
    RESULTS_JSON_PATH.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    write_report(results=results, nn_summary_df=nn_summary_df)

    print("\nTree stack + CatBoost summary:")
    print(results["summary_df"].round(4).to_string(index=False))
    print(f"\nSaved JSON results to {RESULTS_JSON_PATH}")
    print(f"Saved markdown report to {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()
