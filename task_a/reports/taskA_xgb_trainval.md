# Task A XGBoost Classifier

## Objective
Train a single-model Task A classifier using `xgboost.XGBClassifier` while reusing the existing Task A preprocessing and submission I/O contract from the current ensemble notebook.

## Assumptions About The Existing Interface
- Raw feature input is a pandas `DataFrame` with the same schema as `credit_train.csv` minus `RiskTier` and `InterestRate`.
- Preprocessing is done through the existing Task A `fit_task_a_preprocessor` / `transform_task_a` contract and is not modified here.
- `RiskTier` labels are integer-encoded classes `[0, 1, 2, 3, 4]`.
- Submission writing preserves an existing `InterestRate` column from `submissions/submission.csv` when available.

## Training Config
```json
{
  "artifact_tag": "taskA_xgb_trainval",
  "validation_size": 0.2,
  "random_state": 42,
  "model_threads": 2,
  "verbose": false,
  "metric_name": "accuracy",
  "enable_early_stopping": true,
  "early_stopping_rounds": 40,
  "internal_early_stopping_size": 0.1,
  "run_full_train_prediction": true,
  "preserve_interest_rate": true,
  "save_model_bundle": true,
  "xgb_params": {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 4.442311657343007,
    "subsample": 0.8524337374155861,
    "colsample_bytree": 0.8837262828873732,
    "gamma": 1.1909377972289308,
    "reg_alpha": 0.015936847132187043,
    "reg_lambda": 0.0318485426459654
  },
  "use_scale_pos_weight": false
}
```

## Validation Metrics
- Accuracy: `0.6920`
- Macro F1: `0.6892`
- Selected score (`accuracy`): `0.6920`
- Early stopping used: `True`
- Early stopping source: `external_validation`
- Best iteration: `149`
- Best n_estimators used for final model: `150`
- Feature count after preprocessing: `115`

## Confusion Matrix
| index | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 987 | 318 | 17 | 23 | 0 |
| Low(1) | 404 | 850 | 159 | 43 | 0 |
| Moderate(2) | 106 | 318 | 675 | 295 | 6 |
| High(3) | 15 | 19 | 181 | 1049 | 98 |
| VeryHigh(4) | 1 | 0 | 11 | 142 | 1283 |

## Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.6523 | 0.7338 | 0.6907 | 1345 |
| Low(1) | 0.5648 | 0.5838 | 0.5741 | 1456 |
| Moderate(2) | 0.6472 | 0.4821 | 0.5526 | 1400 |
| High(3) | 0.6759 | 0.7702 | 0.7200 | 1362 |
| VeryHigh(4) | 0.9250 | 0.8928 | 0.9086 | 1437 |
| accuracy | 0.6920 | 0.6920 | 0.6920 | 0.6920 |
| macro avg | 0.6930 | 0.6926 | 0.6892 | 7000 |
| weighted avg | 0.6937 | 0.6920 | 0.6893 | 7000 |

## Saved Artifacts
- Validation metrics JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_metrics.json`
- Validation predictions CSV: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_predictions.csv`
- Validation model JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_model.json`
- Validation preprocessor JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_preprocessor.json`
- Validation bundle: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_bundle.joblib`
- Submission CSV: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/submissions/submission.csv`
- Artifact copy of submission: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_submission.csv`
- Full-train model JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_full_train_model.json`
- Full-train preprocessor JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_full_train_preprocessor.json`
- Full-train bundle: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_full_train_bundle.joblib`
