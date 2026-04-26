from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditsense-ai1215" / "credit_train.csv"
FIG_DIR = ROOT / "docs" / "figures"
SUMMARY_PATH = ROOT / "docs" / "report_summary.json"

OPTUNA_SUMMARY_PATH = ROOT / "task_a" / "artifacts" / "taskA_xgb_optuna_optuna_summary.json"
OPTUNA_TRIALS_PATH = ROOT / "task_a" / "artifacts" / "taskA_xgb_optuna_optuna_trials.csv"
BEST_MODEL_REPORT_PATH = ROOT / "task_a" / "reports" / "taskA_taskb_style_stack_vs_baselines.md"
TASK_B_METRICS_PATH = ROOT / "task_b" / "artifacts" / "task_b_metrics.json"
TASK_B_VALIDATION_PATH = ROOT / "task_b" / "artifacts" / "task_b_validation_predictions.csv"

CORR_MATRIX_FEATURES = [
    "RiskTier",
    "InterestRate",
    "NumberOfChargeOffs",
    "NumberOfLatePayments30Days",
    "NumberOfLatePayments90Days",
    "NumberOfLatePayments60Days",
    "NumberOfCollections",
    "RevolvingUtilizationRate",
    "NumberOfBankruptcies",
    "AnnualIncome",
    "LoanToIncomeRatio",
    "DebtToIncomeRatio",
]
TASK_B_RATE_BUCKET_ORDER = ["4.99", "5-8", "8-12", "12-20", "20+"]
TASK_B_MODEL_LABELS = {
    "LinearRegression": "Linear Regression",
    "MLPRegressor": "MLP Regressor",
    "RandomForestRegressor": "Random Forest",
    "XGBRegressor": "XGBoost",
    "HistGradientBoostingRegressor": "HistGradientBoosting",
    "StackingRegressor": "Final Stack",
}


def parse_markdown_table(markdown: str, heading: str) -> pd.DataFrame:
    marker = markdown.index(heading)
    lines = markdown[marker:].splitlines()[1:]

    table_lines: list[str] = []
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            break

    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found after heading {heading!r}")

    headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(cells)

    return pd.DataFrame(rows, columns=headers)


def generate_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    risk_counts = df["RiskTier"].value_counts().sort_index()
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(8).sort_values()

    corr_risk = (
        df.corr(numeric_only=True)["RiskTier"]
        .drop(labels=["RiskTier"])
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    corr_interest = (
        df.corr(numeric_only=True)["InterestRate"]
        .drop(labels=["InterestRate"])
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )

    corr_matrix = df[CORR_MATRIX_FEATURES].corr(numeric_only=True)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "risk_counts": {str(int(k)): int(v) for k, v in risk_counts.items()},
        "interest_summary": {
            "min": round(float(df["InterestRate"].min()), 4),
            "q1": round(float(df["InterestRate"].quantile(0.25)), 4),
            "median": round(float(df["InterestRate"].median()), 4),
            "mean": round(float(df["InterestRate"].mean()), 4),
            "q3": round(float(df["InterestRate"].quantile(0.75)), 4),
            "max": round(float(df["InterestRate"].max()), 4),
            "std": round(float(df["InterestRate"].std()), 4),
            "iqr": round(float(df["InterestRate"].quantile(0.75) - df["InterestRate"].quantile(0.25)), 4),
            "p90": round(float(df["InterestRate"].quantile(0.90)), 4),
            "p95": round(float(df["InterestRate"].quantile(0.95)), 4),
            "floor_pct": round(float(np.isclose(df["InterestRate"], 4.99).mean() * 100.0), 2),
        },
        "top_missing_pct": {k: round(float(v), 2) for k, v in top_missing.sort_values(ascending=False).items()},
        "top_corr_risk": {k: round(float(v), 4) for k, v in corr_risk.items()},
        "top_corr_interest": {k: round(float(v), 4) for k, v in corr_interest.items()},
        "corr_matrix_features": CORR_MATRIX_FEATURES,
        "corr_matrix_values": {
            row: {col: round(float(corr_matrix.loc[row, col]), 4) for col in corr_matrix.columns}
            for row in corr_matrix.index
        },
    }


def generate_eda_figures(df: pd.DataFrame) -> None:
    risk_counts = df["RiskTier"].value_counts().sort_index()
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(8).sort_values()

    corr_risk = (
        df.corr(numeric_only=True)["RiskTier"]
        .drop(labels=["RiskTier"])
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    corr_interest = (
        df.corr(numeric_only=True)["InterestRate"]
        .drop(labels=["InterestRate"])
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    corr_matrix = df[CORR_MATRIX_FEATURES].corr(numeric_only=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["0", "1", "2", "3", "4"]
    values = [risk_counts.get(i, 0) for i in range(5)]
    bars = ax.bar(labels, values, color=["#315c8f", "#4e8fb5", "#6db2a8", "#f0a35e", "#c45a4a"])
    ax.set_title("RiskTier Distribution in Training Data")
    ax.set_xlabel("RiskTier")
    ax.set_ylabel("Applicants")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 60, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_tier_distribution.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.75))
    ax.barh(list(top_missing.index), list(top_missing.values), color="#c45a4a")
    ax.set_title("Top Missing Features")
    ax.set_xlabel("Missing Values (%)")
    ax.set_ylabel("")
    for idx, value in enumerate(top_missing.values):
        ax.text(value + 0.6, idx, f"{value:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_missing_features.svg", format="svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(list(reversed(corr_risk.index)), list(reversed(corr_risk.values)), color="#315c8f")
    axes[0].set_title("Top Absolute Correlations with RiskTier")
    axes[0].set_xlabel("|Pearson correlation|")
    axes[0].tick_params(axis="y", labelsize=9)

    axes[1].barh(list(reversed(corr_interest.index)), list(reversed(corr_interest.values)), color="#6db2a8")
    axes[1].set_title("Top Absolute Correlations with InterestRate")
    axes[1].set_xlabel("|Pearson correlation|")
    axes[1].tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_target_correlations.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="magma",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of the Most Informative Numeric Features", pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_task_b_artifacts() -> tuple[dict[str, Any], pd.DataFrame]:
    if not TASK_B_METRICS_PATH.exists() or not TASK_B_VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "Missing Task B report artifacts. Run task_b/scripts/generate_task_b_report_artifacts.py first."
        )

    metrics_payload = json.loads(TASK_B_METRICS_PATH.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(TASK_B_VALIDATION_PATH)
    if "rate_bucket" in predictions_df.columns:
        predictions_df["rate_bucket"] = pd.Categorical(
            predictions_df["rate_bucket"],
            categories=TASK_B_RATE_BUCKET_ORDER,
            ordered=True,
        )
    return metrics_payload, predictions_df


def generate_task_b_figures(df: pd.DataFrame) -> None:
    metrics_payload, predictions_df = load_task_b_artifacts()
    target_summary = metrics_payload["target_summary"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.histplot(
        df["InterestRate"],
        bins=50,
        kde=True,
        color="#c45a4a",
        edgecolor="white",
        alpha=0.9,
        ax=ax,
    )
    ax.axvline(target_summary["mean"], color="#315c8f", linestyle="--", linewidth=1.8, label="Mean")
    ax.axvline(target_summary["median"], color="#1f1f1f", linestyle=":", linewidth=2.0, label="Median")
    ax.axvline(target_summary["floor_value"], color="#6db2a8", linestyle="-.", linewidth=1.8, label="Rate floor")
    stats_text = (
        f"Mean: {target_summary['mean']:.2f}\n"
        f"Median: {target_summary['median']:.2f}\n"
        f"IQR: {target_summary['iqr']:.2f}\n"
        f"Std: {target_summary['std']:.2f}\n"
        f"At 4.99: {target_summary['floor_pct']:.1f}%\n"
        f"P90 / P95: {target_summary['p90']:.2f} / {target_summary['p95']:.2f}"
    )
    ax.text(
        0.985,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0"},
    )
    ax.set_title("Task B Target Distribution: InterestRate")
    ax.set_xlabel("InterestRate")
    ax.set_ylabel("Loans")
    ax.legend(loc="upper center", ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_interest_rate_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_df = df[["RiskTier", "InterestRate"]].copy()
    plot_df["RiskTier"] = pd.Categorical(
        plot_df["RiskTier"].astype(str),
        categories=["0", "1", "2", "3", "4"],
        ordered=True,
    )
    medians = plot_df.groupby("RiskTier", observed=False)["InterestRate"].median()

    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.boxplot(
        data=plot_df,
        x="RiskTier",
        y="InterestRate",
        palette=["#315c8f", "#4e8fb5", "#6db2a8", "#f0a35e", "#c45a4a"],
        width=0.65,
        showfliers=False,
        ax=ax,
    )
    ax.plot(range(len(medians)), medians.to_numpy(), color="#1f1f1f", marker="o", linewidth=1.5, markersize=5)
    ax.set_title("InterestRate Increases Systematically with RiskTier")
    ax.set_xlabel("RiskTier")
    ax.set_ylabel("InterestRate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_interest_rate_by_risktier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    corr_series = (
        df.corr(numeric_only=True)["InterestRate"]
        .drop(labels=["InterestRate"])
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(12)
        .sort_values()
    )
    corr_colors = ["#315c8f" if value < 0 else "#c45a4a" for value in corr_series.values]
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    bars = ax.barh(corr_series.index, corr_series.values, color=corr_colors)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_title("Top Numeric Correlations with InterestRate")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("")
    pad = 0.015
    for bar, value in zip(bars, corr_series.values):
        x = value + pad if value >= 0 else value - pad
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_top_correlations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    model_df = (
        pd.DataFrame.from_dict(metrics_payload["models"], orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
    )
    model_order = metrics_payload["model_order"]
    model_df["model"] = pd.Categorical(model_df["model"], categories=model_order, ordered=True)
    model_df = model_df.sort_values("model")
    model_labels = [TASK_B_MODEL_LABELS.get(name, name) for name in model_df["model"].astype(str)]
    bar_colors = [
        "#c45a4a" if name == "StackingRegressor" else "#6db2a8"
        for name in model_df["model"].astype(str)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    metric_specs = [
        ("rmse", "RMSE", "lower is better"),
        ("mae", "MAE", "lower is better"),
        ("r2", "R²", "higher is better"),
    ]
    for ax, (metric_key, title, hint) in zip(axes, metric_specs):
        bars = ax.barh(model_labels, model_df[metric_key], color=bar_colors)
        ax.set_title(f"{title} ({hint})")
        ax.set_xlabel(title)
        for bar, value in zip(bars, model_df[metric_key]):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8.5)
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.suptitle("Task B Model Comparison on the Shared 80/20 Validation Split", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    y_true = predictions_df["y_true"].to_numpy()
    y_pred = predictions_df["y_pred_stack"].to_numpy()
    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    hb = ax.hexbin(y_true, y_pred, gridsize=42, cmap="magma", mincnt=1)
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#1f1f1f", linewidth=1.5)
    metrics_box = metrics_payload["models"]["StackingRegressor"]
    ax.text(
        0.03,
        0.97,
        (
            f"RMSE: {metrics_box['rmse']:.4f}\n"
            f"MAE: {metrics_box['mae']:.4f}\n"
            f"R²: {metrics_box['r2']:.4f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0"},
    )
    ax.set_title("Task B Final Stack: Actual vs Predicted InterestRate")
    ax.set_xlabel("Actual InterestRate")
    ax.set_ylabel("Predicted InterestRate")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Validation rows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_actual_vs_predicted.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    sns.histplot(predictions_df["residual"], bins=45, kde=True, color="#4e8fb5", ax=axes[0])
    axes[0].axvline(0, color="#1f1f1f", linestyle="--", linewidth=1.3)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (actual - predicted)")
    axes[0].set_ylabel("Validation rows")

    axes[1].scatter(
        predictions_df["y_pred_stack"],
        predictions_df["residual"],
        s=14,
        alpha=0.18,
        color="#c45a4a",
        edgecolors="none",
    )
    axes[1].axhline(0, color="#1f1f1f", linestyle="--", linewidth=1.3)
    axes[1].set_title("Residuals vs Predicted InterestRate")
    axes[1].set_xlabel("Predicted InterestRate")
    axes[1].set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_residual_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    error_df = predictions_df.copy()
    error_df["rate_bucket"] = pd.Categorical(
        error_df["rate_bucket"],
        categories=TASK_B_RATE_BUCKET_ORDER,
        ordered=True,
    )
    bucket_counts = error_df["rate_bucket"].value_counts().reindex(TASK_B_RATE_BUCKET_ORDER)
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    sns.boxplot(
        data=error_df,
        x="rate_bucket",
        y="abs_error",
        order=TASK_B_RATE_BUCKET_ORDER,
        color="#6db2a8",
        showfliers=False,
        ax=ax,
    )
    y_top = float(error_df["abs_error"].quantile(0.98))
    for idx, count in enumerate(bucket_counts):
        ax.text(idx, y_top, f"n={int(count)}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Absolute Error by True InterestRate Bucket")
    ax.set_xlabel("True InterestRate bucket")
    ax.set_ylabel("Absolute error")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_error_by_bucket.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rebuild_optuna_study_from_trials(trials_csv: Path) -> tuple[optuna.study.Study, list[str]]:
    trials_df = pd.read_csv(trials_csv)
    completed_df = trials_df.loc[trials_df["state"] == "COMPLETE"].copy()
    if completed_df.empty:
        raise ValueError(f"No completed Optuna trials found in {trials_csv}")

    param_distributions: dict[str, Any] = {
        "n_estimators": IntDistribution(200, 1200),
        "max_depth": IntDistribution(3, 10),
        "learning_rate": FloatDistribution(1e-2, 3e-1, log=True),
        "min_child_weight": FloatDistribution(1.0, 12.0),
        "subsample": FloatDistribution(0.6, 1.0),
        "colsample_bytree": FloatDistribution(0.6, 1.0),
        "gamma": FloatDistribution(0.0, 5.0),
        "reg_alpha": FloatDistribution(1e-8, 10.0, log=True),
        "reg_lambda": FloatDistribution(1e-3, 25.0, log=True),
    }

    available_params = [
        name
        for name in param_distributions
        if f"params_{name}" in completed_df.columns and completed_df[f"params_{name}"].notna().any()
    ]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="taskA_xgb_optuna_slice_rebuilt")

    for _, row in completed_df.iterrows():
        params = {}
        distributions = {}
        for name in available_params:
            col = f"params_{name}"
            if pd.isna(row[col]):
                continue
            dist = param_distributions[name]
            value = int(row[col]) if isinstance(dist, IntDistribution) else float(row[col])
            params[name] = value
            distributions[name] = dist

        trial = create_trial(
            params=params,
            distributions=distributions,
            value=float(row["value"]),
            state=TrialState.COMPLETE,
            user_attrs={
                "mean_accuracy": float(row["user_attrs_mean_accuracy"]),
                "mean_macro_f1": float(row["user_attrs_mean_macro_f1"]),
                "mean_selected_score": float(row["user_attrs_mean_selected_score"]),
            },
        )
        study.add_trial(trial)

    return study, available_params


def generate_optuna_slice_figure() -> None:
    study, available_params = rebuild_optuna_study_from_trials(OPTUNA_TRIALS_PATH)
    n_cols = 3
    n_rows = math.ceil(len(available_params) / n_cols)
    target_fn = lambda trial: trial.user_attrs.get("mean_accuracy", trial.value)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=available_params,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for index, param_name in enumerate(available_params):
        row = index // n_cols + 1
        col = index % n_cols + 1
        slice_fig = optuna.visualization.plot_slice(
            study,
            params=[param_name],
            target=target_fn,
            target_name="Mean CV Accuracy",
        )
        for trace in slice_fig.data:
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(
            title_text=param_name,
            type=slice_fig.layout.xaxis.type,
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text="Mean CV Accuracy", row=row, col=col)

    fig.update_layout(
        height=max(420, 360 * n_rows),
        width=1500,
        showlegend=False,
        title_text="Optuna Slice Plot for Tuned XGBoost Hyperparameters",
        title_x=0.5,
    )
    fig.write_image(FIG_DIR / "optuna_sliced.png", scale=2)


def generate_best_model_confusion_matrix() -> None:
    report_md = BEST_MODEL_REPORT_PATH.read_text(encoding="utf-8")
    confusion_df = parse_markdown_table(report_md, "## Confusion Matrix")
    confusion_df = confusion_df.rename(columns={"index": "actual"}).set_index("actual")
    confusion_df = confusion_df.apply(pd.to_numeric)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.85, "label": "Count"},
        ax=ax,
    )
    ax.set_title("Best Final Task A Model: Confusion Matrix", pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_final_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    summary = generate_data_summary(df)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    generate_eda_figures(df)
    generate_task_b_figures(df)
    generate_optuna_slice_figure()
    generate_best_model_confusion_matrix()


if __name__ == "__main__":
    main()
