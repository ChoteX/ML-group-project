from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
OUTPUT_JSON = ROOT / "docs" / "task_a_classifier_results.json"

TASKA_LOG_PATH = ROOT / "TaskA.md"
FC_NN_REPORT_PATH = ROOT / "task_a" / "reports" / "taskA_fc_nn.md"
STACK_REPORT_PATH = ROOT / "task_a" / "reports" / "taskA_taskb_style_stack_vs_baselines.md"
TREE_JSON_PATH = ROOT / "task_a" / "artifacts" / "taskA_tree_catboost_cv_results.json"
OPTUNA_SUMMARY_PATH = ROOT / "task_a" / "artifacts" / "taskA_xgb_optuna_optuna_summary.json"
XGB_VALIDATION_PATH = ROOT / "task_a" / "artifacts" / "taskA_xgb_trainval_validation_metrics.json"


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
        if len(cells) == len(headers):
            rows.append(cells)

    return pd.DataFrame(rows, columns=headers)


def build_classifier_results() -> pd.DataFrame:
    results: list[dict[str, object]] = []

    taska_log = TASKA_LOG_PATH.read_text(encoding="utf-8")
    baseline_df = parse_markdown_table(taska_log, "### Baseline vs Upgraded Validation Metrics")
    protocol_map = {
        "Leakage-free baseline": "Single 80/20 validation",
        "Upgraded one-hot + clipping": "Single 80/20 validation",
    }
    for _, row in baseline_df.iterrows():
        results.append(
            {
                "model": row["Model"],
                "protocol": protocol_map[row["Model"]],
                "accuracy": float(row["Validation Accuracy"].strip("`")),
                "macro_f1": float(row["Validation Macro F1"].strip("`")),
            }
        )

    fc_nn_report = FC_NN_REPORT_PATH.read_text(encoding="utf-8")
    fc_metrics_df = parse_markdown_table(fc_nn_report, "### Mean And Standard Deviation")
    fc_accuracy = float(fc_metrics_df.loc[fc_metrics_df["metric"] == "accuracy", "mean"].iloc[0])
    fc_macro_f1 = float(fc_metrics_df.loc[fc_metrics_df["metric"] == "macro_f1", "mean"].iloc[0])
    results.append(
        {
            "model": "FC neural network",
            "protocol": "5-fold CV mean",
            "accuracy": fc_accuracy,
            "macro_f1": fc_macro_f1,
        }
    )

    tree_payload = json.loads(TREE_JSON_PATH.read_text(encoding="utf-8"))
    tree_summary = tree_payload["summary"][0]
    results.append(
        {
            "model": "Tree stack + CatBoost",
            "protocol": "5-fold CV mean",
            "accuracy": float(tree_summary["accuracy_mean"]),
            "macro_f1": float(tree_summary["macro_f1_mean"]),
        }
    )

    stack_report = STACK_REPORT_PATH.read_text(encoding="utf-8")
    stack_summary_df = parse_markdown_table(stack_report, "## Summary Comparison")
    stack_row = stack_summary_df.loc[
        stack_summary_df["model"] == "Task B-style StackingClassifier"
    ].iloc[0]
    results.append(
        {
            "model": "Task B-style StackingClassifier",
            "protocol": "5-fold CV mean",
            "accuracy": float(stack_row["accuracy_mean"]),
            "macro_f1": float(stack_row["macro_f1_mean"]),
        }
    )

    xgb_validation = json.loads(XGB_VALIDATION_PATH.read_text(encoding="utf-8"))
    results.append(
        {
            "model": "Single XGBoost",
            "protocol": "Single 80/20 validation",
            "accuracy": float(xgb_validation["accuracy"]),
            "macro_f1": float(xgb_validation["macro_f1"]),
        }
    )

    optuna_summary = json.loads(OPTUNA_SUMMARY_PATH.read_text(encoding="utf-8"))
    results.append(
        {
            "model": "Optuna-tuned single XGBoost",
            "protocol": "5-fold CV mean (best trial)",
            "accuracy": float(optuna_summary["best_user_attrs"]["mean_accuracy"]),
            "macro_f1": float(optuna_summary["best_user_attrs"]["mean_macro_f1"]),
        }
    )

    df = pd.DataFrame(results)
    return df.sort_values("accuracy", ascending=True).reset_index(drop=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = build_classifier_results()
    OUTPUT_JSON.write_text(df.to_json(orient="records", indent=2) + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    color_map = {
        "Single 80/20 validation": "#d08159",
        "5-fold CV mean": "#3b6ea8",
        "5-fold CV mean (best trial)": "#7d5fb2",
    }
    colors = [color_map.get(protocol, "#6a6a6a") for protocol in df["protocol"]]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    labels = [f"{model}\n{protocol}" for model, protocol in zip(df["model"], df["protocol"])]
    bars = ax.barh(labels, df["accuracy"], color=colors)
    ax.set_title("Task A Classifier Accuracy Comparison")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    ax.set_xlim(0.65, max(0.86, float(df["accuracy"].max()) + 0.01))

    for bar, value in zip(bars, df["accuracy"]):
        ax.text(value + 0.0025, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in color_map.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, title="Evaluation protocol")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_a_classifier_accuracy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
