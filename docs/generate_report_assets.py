from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    generate_optuna_slice_figure()
    generate_best_model_confusion_matrix()


if __name__ == "__main__":
    main()
