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
  "run_full_train_prediction": false,
  "preserve_interest_rate": true,
  "save_model_bundle": true,
  "xgb_params": {
    "n_estimators": 1035,
    "max_depth": 6,
    "learning_rate": 0.03992306657672681,
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
- Accuracy: `0.7896`
- Macro F1: `0.7903`
- Selected score (`accuracy`): `0.7896`
- Early stopping used: `True`
- Early stopping source: `external_validation`
- Best iteration: `1034`
- Best n_estimators used for final model: `1035`
- Feature count after preprocessing: `115`

## Confusion Matrix
| index | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 1110 | 219 | 15 | 1 | 0 |
| Low(1) | 231 | 1028 | 182 | 15 | 0 |
| Moderate(2) | 12 | 236 | 961 | 188 | 3 |
| High(3) | 2 | 16 | 165 | 1116 | 63 |
| VeryHigh(4) | 0 | 0 | 7 | 118 | 1312 |

## Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.8192 | 0.8253 | 0.8222 | 1345 |
| Low(1) | 0.6858 | 0.7060 | 0.6958 | 1456 |
| Moderate(2) | 0.7226 | 0.6864 | 0.7040 | 1400 |
| High(3) | 0.7761 | 0.8194 | 0.7971 | 1362 |
| VeryHigh(4) | 0.9521 | 0.9130 | 0.9321 | 1437 |
| accuracy | 0.7896 | 0.7896 | 0.7896 | 0.7896 |
| macro avg | 0.7911 | 0.7900 | 0.7903 | 7000 |
| weighted avg | 0.7910 | 0.7896 | 0.7900 | 7000 |

## Saved Artifacts
- Validation metrics JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_metrics.json`
- Validation predictions CSV: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_predictions.csv`
- Validation model JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_model.json`
- Validation preprocessor JSON: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_preprocessor.json`
- Validation bundle: `/Users/dachi.tchotashvili/local docs/VS_main/ML final/ML-group-project/task_a/artifacts/taskA_xgb_trainval_validation_bundle.joblib`
