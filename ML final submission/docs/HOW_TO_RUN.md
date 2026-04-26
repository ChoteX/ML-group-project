# How To Run The Final Submission

This folder is intended to be self-contained for reviewer inspection. The final submission flow has two steps:

1. run **Task A** to generate the classification submission file
2. run **Task B** to fill in the regression target and create the final combined CSV

These instructions assume you open the **`ML final submission`** folder itself as the project root in Jupyter / VS Code.

## Part A: Create `submissions/submission.csv`

Open:

- `task_a/notebooks/taskA_best.ipynb`

Run all cells from top to bottom.

What it reads:

- `data/creditsense-ai1215/credit_train.csv`
- `data/creditsense-ai1215/credit_test.csv`
- `data/creditsense-ai1215/sample_submission.csv`

What it writes:

- `submissions/submission.csv`
- `task_a/reports/taskA_taskb_style_stack_vs_baselines.md`

Notes:

- This is the strongest finalized Task A notebook in this folder.
- The notebook now resolves the repo root automatically, so the data / output paths do not depend on the notebook working directory.
- If `task_a/artifacts/taskA_tree_catboost_cv_results.json` is missing, the notebook will still continue and just skip the stored-baseline comparison rows.

## Part B: Create The Final Combined Submission

Open:

- `task_b/notebooks/Task_B.ipynb`

Run all cells from top to bottom **after Part A has finished**.

What it reads:

- `data/creditsense-ai1215/credit_train.csv`
- `data/creditsense-ai1215/credit_test.csv`
- `submissions/submission.csv` from Part A

What it writes:

- `task_b/artifacts/gukas_reg_submission.csv`
  - regression-only `InterestRate` predictions
- `submissions/submission_Dachi_class_Guka_reg.csv`
  - final combined submission with both `RiskTier` and `InterestRate`
- `task_b/notebooks/submission_DCGR2.csv`
  - notebook-local copy of the same final combined submission

## Path Check Summary

I checked the code paths in this finalized copy without running the notebooks:

- `task_a/notebooks/taskA_best.ipynb`
  - input and output paths now resolve from the `ML final submission` repo root
  - report output parent directory is created automatically
- `task_a/notebooks/improved_taskA.ipynb`
  - input and output paths now resolve from the `ML final submission` repo root
- `task_b/notebooks/Task_B.ipynb`
  - input and output paths now resolve from the `ML final submission` repo root
  - final merged CSV now writes into `submissions/` in addition to the notebook-local copy
- `task_a/notebooks/taskA_xgboost_optuna.ipynb`
  - already used repo-root resolution and did not need path changes

## Minimal Reviewer Flow

If a reviewer only wants the final prediction files:

1. Run `task_a/notebooks/taskA_best.ipynb`
2. Confirm `submissions/submission.csv` exists
3. Run `task_b/notebooks/Task_B.ipynb`
4. Use `submissions/submission_Dachi_class_Guka_reg.csv` as the final combined output
