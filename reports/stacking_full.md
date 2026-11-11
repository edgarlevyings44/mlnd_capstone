# Stacking vs single models (full)

Outer 5-fold CV on the training split, inner 5-fold for the meta-learner, 2000 paired bootstrap resamples of the pooled out-of-fold predictions.
Bases: ['random_forest', 'xgboost', 'lightgbm', 'catboost']. Meta-learner: logistic regression (balanced).

## Verdict
- Stacking TIES with the best single model (catboost) on PR-AUC: difference +0.0016, 95% CI [-0.0003, +0.0034].
- Stacking TIES with the best single model (catboost) on best-threshold F1: difference +0.0014, 95% CI [-0.0028, +0.0059].
- Per-fold PR-AUC: stack beat catboost in 4 of 5 folds.

## Pooled out-of-fold performance (95% bootstrap CI)
| model | pr_auc | pr_auc_lo | pr_auc_hi | best_f1 | best_f1_lo | best_f1_hi | f1_at_0.5 | recall_at_0.5 | precision_at_0.5 |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.7337 | 0.7136 | 0.7528 | 0.6795 | 0.6669 | 0.6962 | 0.6786 | 0.7422 | 0.625 |
| random_forest | 0.727 | 0.7063 | 0.7464 | 0.6768 | 0.6638 | 0.6926 | 0.6718 | 0.6838 | 0.6603 |
| xgboost | 0.7514 | 0.7332 | 0.7699 | 0.6897 | 0.6766 | 0.7063 | 0.6871 | 0.7176 | 0.6591 |
| lightgbm | 0.7486 | 0.7301 | 0.7674 | 0.6895 | 0.6754 | 0.7053 | 0.6874 | 0.7141 | 0.6626 |
| catboost | 0.7523 | 0.7338 | 0.7705 | 0.6897 | 0.676 | 0.705 | 0.685 | 0.7172 | 0.6556 |
| stack | 0.7538 | 0.7351 | 0.772 | 0.6915 | 0.6774 | 0.707 | 0.6878 | 0.7374 | 0.6445 |

`best_f1` = highest F1 over thresholds. `*_at_0.5` use the default threshold and depend on calibration; they are information only.

## Per-fold PR-AUC
| fold | logistic_regression | random_forest | xgboost | lightgbm | catboost | stack |
|---|---|---|---|---|---|---|
| 1.0 | 0.73 | 0.709 | 0.745 | 0.743 | 0.747 | 0.749 |
| 2.0 | 0.736 | 0.736 | 0.756 | 0.752 | 0.754 | 0.755 |
| 3.0 | 0.715 | 0.707 | 0.731 | 0.727 | 0.734 | 0.735 |
| 4.0 | 0.742 | 0.742 | 0.774 | 0.77 | 0.771 | 0.774 |
| 5.0 | 0.753 | 0.746 | 0.759 | 0.76 | 0.765 | 0.763 |

## Stack minus each single model (paired)
| stack_vs | metric | mean_diff | ci_lo | ci_hi | verdict |
|---|---|---|---|---|---|
| logistic_regression | pr_auc | 0.0201 | 0.0132 | 0.0274 | WINS |
| logistic_regression | best_f1 | 0.0109 | 0.0023 | 0.0191 | WINS |
| random_forest | pr_auc | 0.0269 | 0.0195 | 0.0347 | WINS |
| random_forest | best_f1 | 0.0141 | 0.0064 | 0.0214 | WINS |
| xgboost | pr_auc | 0.0023 | -0.0003 | 0.0049 | TIES |
| xgboost | best_f1 | 0.001 | -0.0033 | 0.0054 | TIES |
| lightgbm | pr_auc | 0.0051 | 0.0023 | 0.008 | WINS |
| lightgbm | best_f1 | 0.0017 | -0.0034 | 0.0069 | TIES |
| catboost | pr_auc | 0.0016 | -0.0003 | 0.0034 | TIES |
| catboost | best_f1 | 0.0014 | -0.0028 | 0.0059 | TIES |

Hyperparameters were tuned on these same rows, so scores are slightly optimistic for every model; the held-out test is the unbiased comparison.
