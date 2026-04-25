# Task A FC Neural Network Report

## Objective
Implemented a new Task A classifier in `taskA_nn.ipynb` using a fully connected neural network with ReLU activations and the Adam optimizer. The notebook reuses the existing tree-ensemble preprocessing, adds fold-local engineered interaction features, evaluates the model with 5-fold stratified cross-validation, and generates Task A test predictions.

## Dataset And Task Scope
- Training data: 35000 rows, 57 columns
- Test data: 15000 rows, 55 columns
- Task A features before preprocessing: 55
- Classification target: `RiskTier` with labels [0, 1, 2, 3, 4]

## Preprocessing Reused From The Tree Ensemble Model
- Train-fold-only preprocessing fit and validation/test transform
- Missing categorical values filled with the explicit `Missing` category
- Full one-hot encoding for categorical variables
- Numeric missing-value indicators added before imputation
- Structural numeric missingness zero-filled for the Task A zero-fill columns
- Remaining numeric features median-imputed with train-fold statistics only
- 99th-percentile clipping applied to the Task A money and ratio columns
- `log1p` applied to the Task A money columns
- Validation and test matrices reindexed to the exact train-fold schema

## Engineered Feature Design
- Within each training fold, the top 10 processed features were ranked by absolute Pearson correlation with `RiskTier`.
- Candidate features were restricted to processed numeric columns plus `*_is_missing` indicator columns.
- One-hot dummy columns were excluded from the correlation ranking.
- All 45 pairwise products were added as engineered features.
- Final NN design matrix size: 115 base features + 45 interactions = 160 total features.
- Only the processed numeric columns and engineered interaction columns were standardized; one-hot columns and missing indicators remained as 0/1 inputs.

## FC Network Architecture And Adam / Backprop Configuration
- `MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam')`
- `learning_rate_init=3e-4`, `alpha=1e-4`, `batch_size=256`, `max_iter=400`
- `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=20`
- Adam moment parameters: `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-8`
- `shuffle=True`, `random_state=42`

Adam update summary used by the model:

```text
m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t
v_t = beta_2 * v_(t-1) + (1 - beta_2) * (g_t ** 2)
theta_(t+1) = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

## 5-Fold Evaluation Protocol
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- For each fold: fit preprocessing, select top correlated features, add pairwise products, scale continuous inputs, train the MLP, and score the held-out fold.
- Aggregate all held-out predictions into one out-of-fold prediction vector for overall confusion-matrix and classification-report evaluation.

## Executed Results
### Fold Metrics
| fold | train_rows | val_rows | base_feature_count | interaction_count | final_feature_count | epochs | accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 28000 | 7000 | 115 | 45 | 160 | 38 | 0.8127 | 0.8142 |
| 2 | 28000 | 7000 | 115 | 45 | 160 | 41 | 0.8240 | 0.8246 |
| 3 | 28000 | 7000 | 115 | 45 | 160 | 38 | 0.8193 | 0.8207 |
| 4 | 28000 | 7000 | 115 | 45 | 160 | 41 | 0.8159 | 0.8169 |
| 5 | 28000 | 7000 | 115 | 45 | 160 | 47 | 0.8069 | 0.8067 |

### Mean And Standard Deviation
| metric | mean | std |
| --- | --- | --- |
| accuracy | 0.8157 | 0.0065 |
| macro_f1 | 0.8166 | 0.0068 |

### Out-Of-Fold Confusion Matrix
| actual | VeryLow(0) | Low(1) | Moderate(2) | High(3) | VeryHigh(4) |
| --- | --- | --- | --- | --- | --- |
| VeryLow(0) | 5577 | 1094 | 52 | 1 | 0 |
| Low(1) | 1142 | 5294 | 818 | 29 | 0 |
| Moderate(2) | 31 | 1024 | 5345 | 582 | 16 |
| High(3) | 1 | 41 | 756 | 5706 | 308 |
| VeryHigh(4) | 0 | 0 | 26 | 528 | 6629 |

### Out-Of-Fold Classification Report
| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| VeryLow(0) | 0.8261 | 0.8294 | 0.8278 | 6724 |
| Low(1) | 0.7103 | 0.7269 | 0.7185 | 7283 |
| Moderate(2) | 0.7639 | 0.7638 | 0.7638 | 6998 |
| High(3) | 0.8335 | 0.8376 | 0.8356 | 6812 |
| VeryHigh(4) | 0.9534 | 0.9229 | 0.9379 | 7183 |
| accuracy | 0.8157 | 0.8157 | 0.8157 | 0.8157 |
| macro avg | 0.8174 | 0.8161 | 0.8167 | 35000 |
| weighted avg | 0.8171 | 0.8157 | 0.8164 | 35000 |

### Fold-Wise Top 10 Correlated Features
| fold | top_10_features |
| --- | --- |
| 1 | NumberOfLatePayments30Days, RevolvingUtilizationRate, NumberOfChargeOffs, NumberOfCollections, NumberOfLatePayments60Days, AnnualIncome, NumberOfBankruptcies, NumberOfLatePayments90Days, LoanToIncomeRatio, NumberOfHardInquiries12Mo |
| 2 | NumberOfLatePayments30Days, RevolvingUtilizationRate, NumberOfChargeOffs, NumberOfCollections, NumberOfLatePayments60Days, AnnualIncome, NumberOfBankruptcies, NumberOfLatePayments90Days, LoanToIncomeRatio, NumberOfHardInquiries12Mo |
| 3 | NumberOfLatePayments30Days, RevolvingUtilizationRate, NumberOfChargeOffs, NumberOfCollections, NumberOfLatePayments60Days, AnnualIncome, NumberOfBankruptcies, NumberOfLatePayments90Days, LoanToIncomeRatio, NumberOfHardInquiries12Mo |
| 4 | NumberOfLatePayments30Days, RevolvingUtilizationRate, NumberOfChargeOffs, NumberOfCollections, NumberOfLatePayments60Days, AnnualIncome, NumberOfBankruptcies, NumberOfLatePayments90Days, LoanToIncomeRatio, NumberOfHardInquiries12Mo |
| 5 | NumberOfLatePayments30Days, RevolvingUtilizationRate, NumberOfChargeOffs, NumberOfCollections, NumberOfLatePayments60Days, AnnualIncome, NumberOfBankruptcies, NumberOfLatePayments90Days, LoanToIncomeRatio, NumberOfHardInquiries12Mo |

### Detailed Fold Feature Ranking
| fold | rank | feature | correlation | abs_correlation |
| --- | --- | --- | --- | --- |
| 1 | 1 | NumberOfLatePayments30Days | 0.5698 | 0.5698 |
| 1 | 2 | RevolvingUtilizationRate | 0.5411 | 0.5411 |
| 1 | 3 | NumberOfChargeOffs | 0.4497 | 0.4497 |
| 1 | 4 | NumberOfCollections | 0.4417 | 0.4417 |
| 1 | 5 | NumberOfLatePayments60Days | 0.4296 | 0.4296 |
| 1 | 6 | AnnualIncome | -0.3804 | 0.3804 |
| 1 | 7 | NumberOfBankruptcies | 0.3671 | 0.3671 |
| 1 | 8 | NumberOfLatePayments90Days | 0.3216 | 0.3216 |
| 1 | 9 | LoanToIncomeRatio | 0.2730 | 0.2730 |
| 1 | 10 | NumberOfHardInquiries12Mo | 0.2555 | 0.2555 |
| 2 | 1 | NumberOfLatePayments30Days | 0.5691 | 0.5691 |
| 2 | 2 | RevolvingUtilizationRate | 0.5422 | 0.5422 |
| 2 | 3 | NumberOfChargeOffs | 0.4526 | 0.4526 |
| 2 | 4 | NumberOfCollections | 0.4404 | 0.4404 |
| 2 | 5 | NumberOfLatePayments60Days | 0.4302 | 0.4302 |
| 2 | 6 | AnnualIncome | -0.3789 | 0.3789 |
| 2 | 7 | NumberOfBankruptcies | 0.3702 | 0.3702 |
| 2 | 8 | NumberOfLatePayments90Days | 0.3185 | 0.3185 |
| 2 | 9 | LoanToIncomeRatio | 0.2744 | 0.2744 |
| 2 | 10 | NumberOfHardInquiries12Mo | 0.2545 | 0.2545 |
| 3 | 1 | NumberOfLatePayments30Days | 0.5717 | 0.5717 |
| 3 | 2 | RevolvingUtilizationRate | 0.5435 | 0.5435 |
| 3 | 3 | NumberOfChargeOffs | 0.4486 | 0.4486 |
| 3 | 4 | NumberOfCollections | 0.4392 | 0.4392 |
| 3 | 5 | NumberOfLatePayments60Days | 0.4366 | 0.4366 |
| 3 | 6 | AnnualIncome | -0.3807 | 0.3807 |
| 3 | 7 | NumberOfBankruptcies | 0.3702 | 0.3702 |
| 3 | 8 | NumberOfLatePayments90Days | 0.3220 | 0.3220 |
| 3 | 9 | LoanToIncomeRatio | 0.2773 | 0.2773 |
| 3 | 10 | NumberOfHardInquiries12Mo | 0.2560 | 0.2560 |
| 4 | 1 | NumberOfLatePayments30Days | 0.5738 | 0.5738 |
| 4 | 2 | RevolvingUtilizationRate | 0.5455 | 0.5455 |
| 4 | 3 | NumberOfChargeOffs | 0.4510 | 0.4510 |
| 4 | 4 | NumberOfCollections | 0.4425 | 0.4425 |
| 4 | 5 | NumberOfLatePayments60Days | 0.4322 | 0.4322 |
| 4 | 6 | AnnualIncome | -0.3825 | 0.3825 |
| 4 | 7 | NumberOfBankruptcies | 0.3678 | 0.3678 |
| 4 | 8 | NumberOfLatePayments90Days | 0.3230 | 0.3230 |
| 4 | 9 | LoanToIncomeRatio | 0.2789 | 0.2789 |
| 4 | 10 | NumberOfHardInquiries12Mo | 0.2619 | 0.2619 |
| 5 | 1 | NumberOfLatePayments30Days | 0.5691 | 0.5691 |
| 5 | 2 | RevolvingUtilizationRate | 0.5436 | 0.5436 |
| 5 | 3 | NumberOfChargeOffs | 0.4502 | 0.4502 |
| 5 | 4 | NumberOfCollections | 0.4427 | 0.4427 |
| 5 | 5 | NumberOfLatePayments60Days | 0.4304 | 0.4304 |
| 5 | 6 | AnnualIncome | -0.3798 | 0.3798 |
| 5 | 7 | NumberOfBankruptcies | 0.3696 | 0.3696 |
| 5 | 8 | NumberOfLatePayments90Days | 0.3224 | 0.3224 |
| 5 | 9 | LoanToIncomeRatio | 0.2697 | 0.2697 |
| 5 | 10 | NumberOfHardInquiries12Mo | 0.2528 | 0.2528 |

### Feature Frequency Across Folds
| feature | count |
| --- | --- |
| AnnualIncome | 5 |
| LoanToIncomeRatio | 5 |
| NumberOfBankruptcies | 5 |
| NumberOfChargeOffs | 5 |
| NumberOfCollections | 5 |
| NumberOfHardInquiries12Mo | 5 |
| NumberOfLatePayments30Days | 5 |
| NumberOfLatePayments60Days | 5 |
| NumberOfLatePayments90Days | 5 |
| RevolvingUtilizationRate | 5 |

### Global Top 10 Correlated Features Used For Final Inference
| feature | correlation | abs_correlation |
| --- | --- | --- |
| NumberOfLatePayments30Days | 0.5707 | 0.5707 |
| RevolvingUtilizationRate | 0.5432 | 0.5432 |
| NumberOfChargeOffs | 0.4504 | 0.4504 |
| NumberOfCollections | 0.4413 | 0.4413 |
| NumberOfLatePayments60Days | 0.4318 | 0.4318 |
| AnnualIncome | -0.3804 | 0.3804 |
| NumberOfBankruptcies | 0.3690 | 0.3690 |
| NumberOfLatePayments90Days | 0.3215 | 0.3215 |
| LoanToIncomeRatio | 0.2747 | 0.2747 |
| NumberOfHardInquiries12Mo | 0.2561 | 0.2561 |

### Test Prediction Distribution
| RiskTier | count |
| --- | --- |
| 0 | 2629 |
| 1 | 3656 |
| 2 | 2704 |
| 3 | 3170 |
| 4 | 2841 |

## Generated Artifacts
- Notebook: `taskA_nn.ipynb`
- Markdown report: `task_a/reports/taskA_fc_nn.md`
- Task A predictions: `taskA_nn_risktier_predictions.csv`
- Prediction rows written: 15000
