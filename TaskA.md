# Task A Model Log

## Current Model (Classification: `RiskTier`)
Stacked ensemble classifier with a linear meta-learner:
- Base model 1: `RandomForestClassifier`
- Base model 2: `XGBClassifier`
- Base model 3: `LGBMClassifier`
- Final decision layer: `LinearRegression` on concatenated class probabilities from base models
- Final class mapping: `round` + `clip` to integer classes `[0, 4]`

## Preprocessing Used
- Adds `*_is_missing` indicator columns **before** imputing missing values
- Numeric features: median imputation
- Categorical features: mode imputation + label encoding
- Train/validation split: `train_test_split(test_size=0.2, random_state=42, stratify=y_cls)`
- Stacking folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

## Hyperparameters
### `RandomForestClassifier`
- `n_estimators=300`
- `min_samples_leaf=2`
- `class_weight='balanced_subsample'`
- `random_state=42`
- `n_jobs=-1`

### `XGBClassifier`
- `objective='multi:softprob'`
- `num_class=5`
- `n_estimators=350`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.9`
- `colsample_bytree=0.8`
- `eval_metric='mlogloss'`
- `tree_method='hist'`
- `random_state=42`
- `n_jobs=-1`

### `LGBMClassifier`
- `objective='multiclass'`
- `num_class=5`
- `n_estimators=350`
- `learning_rate=0.05`
- `num_leaves=31`
- `subsample=0.9`
- `colsample_bytree=0.8`
- `random_state=42`
- `n_jobs=-1`
- `verbosity=-1`

### Meta-Learner (`LinearRegression`)
- Default sklearn parameters
- Input features: stacked `predict_proba` outputs from all 3 base models

## Accuracy Log (Current Run)
Run date: **2026-04-14**

- Train accuracy (OOF on train split): **80.82%** (`0.808214`)
- Train accuracy (in-sample on train split): **92.14%** (`0.921393`)
- Validation accuracy: **81.17%** (`0.811714`)

`Train dataset accuracy` is logged above as in-sample train accuracy (`92.14%`).
