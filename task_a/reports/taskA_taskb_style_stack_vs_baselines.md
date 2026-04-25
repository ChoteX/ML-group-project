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
| 1 | 28000 | 7000 | 115 | 5 | 9.5000 | 0.8353 | 0.8356 |
| 2 | 28000 | 7000 | 115 | 5 | 8.9600 | 0.8321 | 0.8331 |
| 3 | 28000 | 7000 | 115 | 5 | 1.4400 | 0.8319 | 0.8327 |
| 4 | 28000 | 7000 | 115 | 5 | 9.8700 | 0.8289 | 0.8292 |
| 5 | 28000 | 7000 | 115 | 5 | 10.4000 | 0.8333 | 0.8341 |

## Summary Comparison
| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | elapsed_minutes | delta_vs_tree_accuracy | delta_vs_tree_macro_f1 | delta_vs_nn_accuracy | delta_vs_nn_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tree stack + CatBoost | 0.8231 | 0.0052 | 0.8247 | 0.0053 | 19.6000 | 0 | 0 | 0.0074 | 0.0081 |
| FC neural network | 0.8157 | 0.0065 | 0.8166 | 0.0068 |  | -0.0074 | -0.0081 | 0 | 0 |
| Task B-style StackingClassifier | 0.8323 | 0.0023 | 0.8330 | 0.0024 | 40.1700 | 0.0092 | 0.0082 | 0.0166 | 0.0164 |

## Confusion Matrix
| index | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 5736 | 945 | 42 | 1 | 0 |
| Low(1) | 1032 | 5377 | 855 | 19 | 0 |
| Moderate(2) | 33 | 873 | 5491 | 593 | 8 |
| High(3) | 2 | 30 | 614 | 5901 | 265 |
| VeryHigh(4) | 0 | 0 | 25 | 533 | 6625 |

## Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.8432 | 0.8531 | 0.8481 | 6724 |
| Low(1) | 0.7442 | 0.7383 | 0.7412 | 7283 |
| Moderate(2) | 0.7814 | 0.7847 | 0.7830 | 6998 |
| High(3) | 0.8374 | 0.8663 | 0.8516 | 6812 |
| VeryHigh(4) | 0.9604 | 0.9223 | 0.9410 | 7183 |
| accuracy | 0.8323 | 0.8323 | 0.8323 | 0.8323 |
| macro avg | 0.8333 | 0.8329 | 0.8330 | 35000 |
| weighted avg | 0.8332 | 0.8323 | 0.8326 | 35000 |

## Artifacts
- Stored baseline summary source: `task_a/artifacts/taskA_tree_catboost_cv_results.json`
- Executed report: `task_a/reports/taskA_taskb_style_stack_vs_baselines.md`
- Full-train submission generation was disabled for this run.
