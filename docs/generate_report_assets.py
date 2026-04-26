from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditsense-ai1215" / "credit_train.csv"
FIG_DIR = ROOT / "docs" / "figures"
SUMMARY_PATH = ROOT / "docs" / "report_summary.json"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

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

    summary = {
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
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")

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


if __name__ == "__main__":
    main()
