# CreditSense: Loan Risk Assessment Challenge

**Team name:** `[fill in before PDF export]`  
**Student names:** `[fill in before PDF export]`  
**Kaggle username(s):** `[fill in before PDF export]`

This report summarizes the current repository state for the AI1215 group project. Task A classification was primarily developed by **Dachi Tchotashvili**, while Task B regression was primarily developed by **Guram Matcharashvili**. The code and experiment artifacts used here are the assignment brief in [project_description.pdf](project_description.pdf), the Task A experiment log in [TaskA.md](../TaskA.md), the Task A reports under [`task_a/reports/`](../task_a/reports), and Guram's regression notebook in [Task_B.ipynb](../task_b/notebooks/Task_B.ipynb).

## 1. Data Exploration & Preprocessing

The training set contains **35,000 applicants**, **55 input features**, and two targets: `RiskTier` for classification and `InterestRate` for regression. The assignment splits the feature space into demographics, income/employment, assets/liabilities, credit history, and loan-request variables. The target distribution is nearly balanced across the five risk tiers, which is useful because accuracy is a meaningful classification metric here and performance is not driven by one dominant class.

![RiskTier distribution](figures/risk_tier_distribution.svg)

*Figure 1. Training-set class balance is close to uniform: tier 0 = 6,724, tier 1 = 7,283, tier 2 = 6,998, tier 3 = 6,812, tier 4 = 7,183.*

The second important observation is that missingness is highly structured rather than random. `StudentLoanOutstandingBalance` is missing for **59.98%** of rows, `CollateralType` for **55.06%**, `CollateralValue` for **54.64%**, and both `MortgageOutstandingBalance` and `PropertyValue` for roughly **45%** of the sample. This matches the project brief: many fields are absent because the applicant simply does not have that asset, liability, or collateral. Treating these values as ordinary random nulls would destroy signal.

![Top missing features](figures/top_missing_features.svg)

*Figure 2. The largest gaps come from structurally absent assets, collateral, and loan balances.*

The main exploratory findings that influenced the pipeline were:

- Credit-history features dominate both targets. The strongest absolute correlations with `RiskTier` are `NumberOfLatePayments30Days` (`0.5707`), `RevolvingUtilizationRate` (`0.5540`), `NumberOfChargeOffs` (`0.4504`), and `NumberOfCollections` (`0.4413`).
- Interest-rate pricing follows a similar pattern but is even more sensitive to severe derogatory marks: `NumberOfChargeOffs` (`0.6484`) is the strongest correlation with `InterestRate`, followed by `NumberOfLatePayments30Days` (`0.5877`) and `NumberOfLatePayments90Days` (`0.5500`).
- Income matters in the opposite direction: `AnnualIncome` is negatively correlated with `RiskTier` (`-0.3520`) and `InterestRate` (`-0.1475`), which is financially intuitive because higher-income borrowers are typically safer and receive better pricing.
- `InterestRate` itself is concentrated toward the lower bound of the allowed range. In the training set the summary is: min `4.99`, Q1 `4.99`, median `6.08`, mean `7.31`, Q3 `7.94`, max `35.99`. This makes regression harder in the upper tail because the distribution is asymmetric.

![Top target correlations](figures/top_target_correlations.svg)

*Figure 3. The same credit-history variables matter for both tasks, but pricing is even more sensitive to severe derogatory marks such as charge-offs and 90-day delinquencies.*

The preprocessing strategy therefore separated the two tasks but followed the same principles:

- **Task A** used split-first preprocessing to avoid leakage, full one-hot encoding for categorical features, an explicit `"Missing"` category for absent labels, missing-value indicators for numeric fields, structural zero-fill where absence meant “does not have this item”, train-fold median imputation for the remaining numerics, 99th-percentile clipping for heavy-tailed money/ratio variables, `log1p` transforms on money-like fields, and schema reindexing so validation/test columns always matched the training design matrix.
- **Task B** used Guram Matcharashvili's notebook pipeline: train/test split, median imputation for numerics, most-frequent imputation plus one-hot encoding for categoricals, and additional `StandardScaler` normalization only for the neural-network branch.

## 2. Feature Engineering

Most of the gains in this repository came from representation engineering rather than from adding many brand-new business variables. The important engineered features and transformations were:

| Addition | Motivation | Where used |
| --- | --- | --- |
| Numeric `*_is_missing` indicators | Preserve signal from structured missingness instead of hiding it behind imputation | Task A |
| Structural zero-fill for balances/assets | Encode “not present” as zero when that is the real financial meaning | Task A |
| 99th-percentile clipping | Reduce sensitivity to extreme outliers in heavily skewed financial variables | Task A |
| `log1p` on money-like features | Compress long right tails and make relative differences more learnable | Task A |
| Pairwise products among top 10 correlated numeric features | Let the FC neural network model interactions between delinquency, utilization, and affordability signals | Task A FC neural network |
| Task-specific scaling | Standardize continuous inputs for logistic/MLP branches without distorting tree inputs | Task A stack and Task B MLP |

For the FC neural-network experiment, the top 10 fold-stable features were `NumberOfLatePayments30Days`, `RevolvingUtilizationRate`, `NumberOfChargeOffs`, `NumberOfCollections`, `NumberOfLatePayments60Days`, `AnnualIncome`, `NumberOfBankruptcies`, `NumberOfLatePayments90Days`, `LoanToIncomeRatio`, and `NumberOfHardInquiries12Mo`. Dachi then created all **45 pairwise interaction terms** among those 10 features, expanding the neural-network design matrix from **115** processed base features to **160** total features.

The repository does not contain a one-by-one ablation for every preprocessing change, but it does contain a measured before/after comparison for the full upgraded Task A feature-engineering bundle:

| Task A preprocessing setup | Evaluation protocol | Accuracy | Macro F1 | Change vs previous |
| --- | --- | ---: | ---: | --- |
| Leakage-free repaired baseline | Single 80/20 validation split | `0.8103` | `0.8120` | Baseline |
| One-hot + missing indicators + structural zero-fill + clipping + `log1p` | Single 80/20 validation split | `0.8121` | `0.8139` | `+0.0018` accuracy, `+0.0019` macro F1 |

This improvement is small but meaningful because it came after removing leakage. In other words, the upgraded pipeline improved performance while also making the evaluation more trustworthy. After that stage, later gains came mostly from stronger model families and stacking rather than from further wholesale preprocessing changes.

## 3. Model Selection & Tuning

### Task A: Classification (`RiskTier`)

Task A is the part of the project with the richest experimentation history, so it is the clearest example of model comparison in the repository. Dachi moved from a repaired baseline to three stronger candidates evaluated under 5-fold cross-validation:

| Model | Evaluation protocol | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| FC neural network | 5-fold CV mean | `0.8157` | `0.8166` | `MLPClassifier` with 45 engineered pairwise interactions |
| Tree stack + CatBoost | 5-fold CV mean | `0.8231` | `0.8247` | RF + XGB + LGBM + native-categorical CatBoost, linear meta-layer |
| Tree stack + neural network | 5-fold CV mean | `0.8323` | `0.8330` | XGB + RF + HGB + logistic + MLP, multinomial logistic meta-learner |

The final Task A model was the **Task B-style StackingClassifier**, which improved on the tree stack by **0.0092** accuracy and **0.0082** macro F1, and improved on the FC neural network by **0.0166** accuracy and **0.0164** macro F1. The best explanation is diversity: tree models captured nonlinear tabular structure, while the logistic and MLP branches added smoother global boundaries and complementary probability estimates for the meta-learner.

The per-class report also shows where the problem remained difficult. In the final classifier, **VeryHigh risk** was easiest (`F1 = 0.9410`), while **Low risk** was the hardest class (`F1 = 0.7412`), followed by **Moderate risk** (`F1 = 0.7830`). This is consistent with the confusion matrix: middle tiers are naturally more ambiguous than the extremes.

### Task B: Regression (`InterestRate`)

Guram Matcharashvili's Task B notebook used a five-model regression stack:

- `XGBRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`
- `LinearRegression`
- `MLPRegressor`
- `Ridge(alpha=1.0)` as the final estimator in `StackingRegressor`

The executed notebook reports the following final result on an 80/20 split:

| Final Task B model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| StackingRegressor (XGB + RF + HGB + LR + MLP -> Ridge) | `1.7052` | `1.3433` | `0.8248` |

To make the comparison section more concrete, the same split and preprocessing were reproduced locally for the single sklearn models that were available in the environment:

| Model | RMSE | MAE | R² | Source |
| --- | ---: | ---: | ---: | --- |
| HistGradientBoostingRegressor | `1.7196` | `1.3472` | `0.8219` | Local rerun on Guram's notebook split |
| RandomForestRegressor | `1.7929` | `1.4025` | `0.8064` | Local rerun on Guram's notebook split |
| MLPRegressor | `1.8584` | `1.4310` | `0.7919` | Local rerun on Guram's notebook split |
| LinearRegression | `2.5464` | `1.7083` | `0.6094` | Local rerun on Guram's notebook split |
| StackingRegressor (final notebook artifact) | `1.7052` | `1.3433` | `0.8248` | Recorded in `Task_B.ipynb` |

The best single reproduced model was `HistGradientBoostingRegressor`, but the full stack still performed best. The gain was modest (`0.8248` vs `0.8219` in `R²`), which suggests that the stack improved robustness more than it changed the overall error profile. Even so, the final regression result was far above the assignment's linear baseline of roughly `R² ≈ 0.50`.

### Overfitting Control and Cross-Task Comparison

Overfitting was controlled in three main ways:

- preprocessing was always fit on training folds only, which removed leakage from the earlier Task A workflow;
- Task A model selection used outer 5-fold cross-validation, and the stacked models used inner folds to build meta-features;
- both neural-network branches used early stopping, while Task B's final estimator was a regularized ridge regressor.

The same features did **not** matter equally for both tasks, although the broad theme was shared. Credit-history variables such as late payments, charge-offs, collections, utilization, and bankruptcies dominated both targets. However, the classification task relied more on separating middle risk buckets from the extremes, so affordability features such as `AnnualIncome` and `LoanToIncomeRatio` were more visible in Dachi's selected Task A interaction set. Regression pricing was more sensitive to severity variables like charge-offs and 90-day late payments, which makes sense because interest-rate setting reacts strongly to the depth of derogatory credit history, not only to broad risk class membership.

## Conclusion

The repository already exceeds the course baselines for both tasks. Task A's strongest result is the stacking classifier at **0.8323 accuracy / 0.8330 macro F1**, and Task B's strongest recorded result is the regression stack at **`R² = 0.8248`**. The main lesson from the project is that careful handling of structured missingness and skewed financial variables was necessary before model complexity could pay off. After the pipeline was made leakage-free, the biggest improvements came from ensemble diversity: Dachi's classification work benefited from combining tree, linear, and neural branches, while Guram's regression work benefited from stacking strong tabular regressors rather than relying on a single model family.
