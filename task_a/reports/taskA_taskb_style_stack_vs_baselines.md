# Task A Task-B Style Stacking Classifier

## Objective
Run a new Task A classification experiment that mirrors the Task B ensemble structure, optionally adds a native-categorical CatBoost base learner, and compares the 5-fold CV results against the stored FC NN and tree + CatBoost baselines.

## Repo Layout Used
- Raw data: `data/creditsense-ai1215/`
- Task A scripts, reports, artifacts: `task_a/`
- Task B notebook and regression artifact: `task_b/`
- Competition submissions: `submissions/`

## New Experiment
- Model: `Task B-style StackingClassifier`
- Base learners:
  - `XGBClassifier`
  - `RandomForestClassifier`
  - `HistGradientBoostingClassifier`
  - `LogisticRegression`
  - `MLPClassifier`
  - `CatBoostClassifier` included: `False`
- Meta-learner: `LogisticRegression(multi_class='multinomial')` inside `StackingClassifier`
- Outer CV folds: `5`
- Inner stacking folds: `2`

## Task A Preprocessing
- One-hot branch for XGB, RF, HGB, logistic, and MLP uses the upgraded Task A preprocessing family.
- Missing categorical values are filled with `Missing` and one-hot encoded.
- Numeric missing indicators are added before imputation.
- Structural Task A numeric columns are zero-filled where missingness semantically implies zero.
- Remaining numeric columns are median-imputed with train-fold statistics only.
- Money and ratio columns are clipped at the train-fold 99th percentile.
- Money columns receive `log1p` after clipping.
- Logistic and MLP branches additionally standardize only the processed numeric columns.
- CatBoost keeps native categorical handling on its own Task A preprocessing branch.

## Fold Metrics
| fold | train_rows | val_rows | feature_count | base_models | elapsed_minutes | accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 28000 | 7000 | 115 | 5 | 8.6000 | 0.8401 | 0.8407 |
| 2 | 28000 | 7000 | 115 | 5 | 8.5500 | 0.8381 | 0.8389 |
| 3 | 28000 | 7000 | 115 | 5 | 9.5900 | 0.8366 | 0.8374 |
| 4 | 28000 | 7000 | 115 | 5 | 10.3600 | 0.8333 | 0.8336 |
| 5 | 28000 | 7000 | 115 | 5 | 10.2100 | 0.8389 | 0.8397 |

## Summary Comparison
| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | elapsed_minutes | delta_vs_tree_accuracy | delta_vs_tree_macro_f1 | delta_vs_nn_accuracy | delta_vs_nn_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tree stack + CatBoost | 0.8231 | 0.0052 | 0.8247 | 0.0053 | 19.6000 | 0 | 0 | 0.0074 | 0.0081 |
| FC neural network | 0.8157 | 0.0065 | 0.8166 | 0.0068 |  | -0.0074 | -0.0081 | 0 | 0 |
| Task B-style StackingClassifier | 0.8374 | 0.0026 | 0.8380 | 0.0028 | 47.3200 | 0.0143 | 0.0133 | 0.0217 | 0.0214 |

## Confusion Matrix
| index | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 5763 | 916 | 44 | 1 | 0 |
| Low(1) | 969 | 5454 | 840 | 20 | 0 |
| Moderate(2) | 22 | 794 | 5566 | 609 | 7 |
| High(3) | 2 | 29 | 610 | 5898 | 273 |
| VeryHigh(4) | 0 | 0 | 22 | 533 | 6628 |

## Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.8530 | 0.8571 | 0.8550 | 6724 |
| Low(1) | 0.7582 | 0.7489 | 0.7535 | 7283 |
| Moderate(2) | 0.7859 | 0.7954 | 0.7906 | 6998 |
| High(3) | 0.8353 | 0.8658 | 0.8503 | 6812 |
| VeryHigh(4) | 0.9595 | 0.9227 | 0.9407 | 7183 |
| accuracy | 0.8374 | 0.8374 | 0.8374 | 0.8374 |
| macro avg | 0.8384 | 0.8380 | 0.8380 | 35000 |
| weighted avg | 0.8383 | 0.8374 | 0.8377 | 35000 |

## Artifacts
- Stored baseline summary source: `task_a/artifacts/taskA_tree_catboost_cv_results.json`
- Executed report: `task_a/reports/taskA_taskb_style_stack_vs_baselines.md`
- Submission artifact: `submissions/submission.csv`
- Submission rows: `15000`
