# Task A Model Log

## Current Model (Classification: `RiskTier`)
Stacked ensemble classifier with a linear meta-learner:
- Base model 1: `RandomForestClassifier`
- Base model 2: `XGBClassifier`
- Base model 3: `LGBMClassifier`
- Final decision layer: `LinearRegression` on concatenated class probabilities from base models
- Final class mapping: `round` + `clip` to integer classes `[0, 4]`

## Current Preprocessing Used
- Raw data is split first with `train_test_split(test_size=0.2, random_state=42, stratify=y_cls)`
- Task A preprocessing is fit only on the training split and then applied to validation and test
- Categorical features use full one-hot encoding with `drop_first=False`
- Missing categorical values are filled with the explicit category `"Missing"`
- Numeric `*_is_missing` indicator columns are added before imputation
- Structurally missing numeric fields are zero-imputed:
  - `StudentLoanOutstandingBalance`
  - `MortgageOutstandingBalance`
  - `PropertyValue`
  - `InvestmentPortfolioValue`
  - `VehicleValue`
  - `AutoLoanOutstandingBalance`
  - `SecondaryMonthlyIncome`
  - `CollateralValue`
- Remaining numeric features use train-only median imputation
- Upper-tail clipping at the training-set 99th percentile is applied to heavy-tailed money and ratio features
- `log1p` is applied after clipping to the money-like features
- Validation and test features are reindexed to the exact training schema after one-hot encoding
- Task A uses its own unscaled feature matrix; Task B remains on a separate preprocessing + scaling path
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

## Detailed Change Notes
The Task A pipeline was refactored so preprocessing is now fit only on the training split and then applied to validation and test data. This removes data leakage from the earlier workflow, where preprocessing was performed before the split and test-time encoding behavior was mixed into training preparation. Leakage inflates validation performance in a misleading way, so removing it gives a more reliable estimate of true generalization and makes later improvements meaningful.

Categorical preprocessing was changed from label encoding to full one-hot encoding for all nine categorical features: `EducationLevel`, `MaritalStatus`, `HomeOwnership`, `State`, `EmploymentStatus`, `EmployerType`, `JobCategory`, `LoanPurpose`, and `CollateralType`. This improves performance because these features are nominal categories, not ordered numeric values. Label encoding imposes a fake ordinal relationship between categories, while one-hot encoding lets the tree models learn separate effects for each category without assuming that one category sits numerically between two others.

Missing categorical values are now filled with the explicit category `"Missing"` instead of being collapsed into the most common class. This improves the model because missingness in credit data often carries signal of its own. For example, an absent `CollateralType` is not equivalent to the most frequent collateral category; it usually means no collateral was reported, which can affect risk.

Numeric missing-value indicators are added before imputation for every feature that has missing values in the training split. These `*_is_missing` columns help the model distinguish between a real observed value and an imputed placeholder. That improves performance because many missing patterns in this dataset are structured, especially around loans, collateral, and asset balances, so the fact that a value is missing can itself be predictive of `RiskTier`.

Several numeric features with structural missingness are now zero-imputed instead of median-imputed: `StudentLoanOutstandingBalance`, `MortgageOutstandingBalance`, `PropertyValue`, `InvestmentPortfolioValue`, `VehicleValue`, `AutoLoanOutstandingBalance`, `SecondaryMonthlyIncome`, and `CollateralValue`. This improves the representation of the data because these fields are often absent when the applicant simply does not have that asset, debt, or secondary income. Using zero preserves that semantic meaning better than replacing the value with a typical median borrowed from unrelated applicants.

All remaining numeric features are median-imputed using statistics learned only from the training split. Median imputation is robust to outliers and is safer than mean imputation for heavy-tailed financial data. Using train-only statistics also preserves a clean validation setup and avoids leaking information from validation or test examples into the fitted preprocessing state.

Upper-tail clipping at the 99th percentile was added for heavily skewed money and ratio features such as income, balances, portfolio values, loan amounts, `LoanToIncomeRatio`, and `PaymentToIncomeRatio`. This improves model stability because a small number of extreme financial values can dominate splits in tree-based models and create brittle decision boundaries. Clipping reduces that influence while still preserving relative ordering for the vast majority of observations.

After clipping, `log1p` is applied to the money-like features. Financial variables in this dataset are strongly right-skewed, so the log transform compresses extreme ranges and makes smaller but meaningful differences easier for the models to use. That usually improves separability between risk groups because the model no longer spends too much capacity on the far tail of a few very large values.

Task A and Task B were separated so the classifier trains on the unscaled Task A feature matrix, while the regression model keeps its own preprocessing and `StandardScaler`. This improves correctness because the old notebook fed the classifier the regression-scaled test matrix during inference, which mixed the two tasks and could distort classification behavior. Keeping the pipelines separate ensures that each model sees features prepared for its own objective.

Schema alignment checks were added so validation and test matrices are forced to match the exact training feature columns after one-hot encoding. This is important because one-hot encoded data can easily produce missing or extra columns when a category appears in one split but not another. Reindexing to the stored training schema prevents feature mismatches and makes the pipeline stable at inference time.

## Latest Recorded Results
Run date: **2026-04-24**

### Data / Schema Checks
- Raw split sizes: train `(28000, 55)`, validation `(7000, 55)`
- Leakage-free baseline Task A matrix: `(28000, 71)` -> `(7000, 71)`
- Upgraded Task A matrix: `(28000, 115)` -> `(7000, 115)`
- Task B matrix: `(28000, 115)` -> `(7000, 115)`
- Task A checks: `no missing=True`, `schema aligned=True`
- Task B checks: `schema aligned=True`

### Baseline vs Upgraded Validation Metrics
| Model | Train Accuracy (OOF) | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|---:|
| Leakage-free baseline | `0.8112` | `0.8103` | `0.8120` |
| Upgraded one-hot + clipping | `0.8092` | `0.8121` | `0.8139` |

### Upgraded Classification Report
| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| VeryLow(0) | `0.87` | `0.78` | `0.82` | `1345` |
| Low(1) | `0.69` | `0.79` | `0.73` | `1456` |
| Moderate(2) | `0.76` | `0.73` | `0.75` | `1400` |
| High(3) | `0.81` | `0.84` | `0.83` | `1362` |
| VeryHigh(4) | `0.96` | `0.92` | `0.94` | `1437` |
| Accuracy |  |  | `0.81` | `7000` |
| Macro avg | `0.82` | `0.81` | `0.81` | `7000` |
| Weighted avg | `0.82` | `0.81` | `0.81` | `7000` |

### Summary
The upgraded preprocessing improved validation accuracy from `0.8103` to `0.8121` and improved validation macro F1 from `0.8120` to `0.8139` against the leakage-free baseline. The gain is modest but consistent with a cleaner evaluation setup and a preprocessing design that matches the structure of the credit-risk data better than mode-imputation plus label encoding.
