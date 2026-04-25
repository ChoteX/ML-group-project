# Task A Tree Stack + CatBoost vs NN

## Objective
Evaluate a Task A stacked tree ensemble that adds a CatBoost base learner with native categorical handling, then compare the 5-fold cross-validation metrics against the previously executed FC neural network results in `taskA_nn.md`.

## Tree Stack Configuration
- Meta-learner: `LinearRegression` on concatenated base-model class probabilities, followed by `round` + `clip` to integer labels `[0, 4]`.
- One-hot branch base learners: `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier` using the existing Task A preprocessing.
- Native categorical branch base learner: `CatBoostClassifier(loss_function='MultiClass')` trained on raw categorical columns with missing categories filled as `Missing`.
- CatBoost numeric preparation: structural zero-fill for the Task A zero-fill columns, train-fold clipping for Task A money/ratio columns, `log1p` on Task A money columns, and numeric missingness left as `NaN` where CatBoost can handle it natively.

## Evaluation Protocol
- Outer CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Inner stacking CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` inside each outer training fold.
- NN comparison baseline: parsed from the executed `taskA_nn.md` report.

## Tree Stack + CatBoost Fold Metrics
| fold | train_rows | val_rows | accuracy | macro_f1 |
| --- | --- | --- | --- | --- |
| 1 | 28000 | 7000 | 0.8280 | 0.8298 |
| 2 | 28000 | 7000 | 0.8213 | 0.8226 |
| 3 | 28000 | 7000 | 0.8271 | 0.8288 |
| 4 | 28000 | 7000 | 0.8151 | 0.8167 |
| 5 | 28000 | 7000 | 0.8240 | 0.8256 |

## Summary Comparison
| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | accuracy_delta_vs_nn | macro_f1_delta_vs_nn |
| --- | --- | --- | --- | --- | --- | --- |
| Tree stack + CatBoost | 0.8231 | 0.0052 | 0.8247 | 0.0053 | 0.0074 | 0.0081 |
| FC neural network | 0.8157 | 0.0065 | 0.8166 | 0.0068 | 0 | 0 |

## Tree Stack + CatBoost Confusion Matrix
| actual | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 5385 | 1300 | 38 | 1 | 0 |
| Low(1) | 778 | 5787 | 699 | 19 | 0 |
| Moderate(2) | 12 | 1184 | 5148 | 645 | 9 |
| High(3) | 1 | 40 | 607 | 5920 | 244 |
| VeryHigh(4) | 0 | 0 | 23 | 591 | 6569 |

## Tree Stack + CatBoost Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.8719 | 0.8009 | 0.8349 | 6724 |
| Low(1) | 0.6963 | 0.7946 | 0.7422 | 7283 |
| Moderate(2) | 0.7902 | 0.7356 | 0.7619 | 6998 |
| High(3) | 0.8250 | 0.8691 | 0.8464 | 6812 |
| VeryHigh(4) | 0.9629 | 0.9145 | 0.9381 | 7183 |
| accuracy | 0.8231 | 0.8231 | 0.8231 | 0.8231 |
| macro avg | 0.8293 | 0.8229 | 0.8247 | 35000 |
| weighted avg | 0.8286 | 0.8231 | 0.8244 | 35000 |

## Interpretation
- Tree stack + CatBoost mean accuracy: 0.8231
- FC neural network mean accuracy: 0.8157
- Tree stack + CatBoost mean macro F1: 0.8247
- FC neural network mean macro F1: 0.8166

A positive delta in the comparison table means the stacked tree ensemble outperformed the FC neural network under the same 5-fold outer CV setup.
