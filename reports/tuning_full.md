# Tuned models (full, 5-fold CV, training split only)

Strategy: class_weight. Search metric: pr_auc. Mean +/- SD across folds at the best candidate.
The best CV score of a search is optimistically biased; the unbiased estimate is the held-out test.
`*_baseline` = baseline with the same strategy; `*_gain` = tuned minus baseline.

| model | n_iter | pr_auc | f1 | recall | precision | pr_auc_baseline | pr_auc_gain | f1_baseline | f1_gain |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 8 | 0.737 +/- 0.016 | 0.679 +/- 0.011 | 0.741 +/- 0.018 | 0.626 +/- 0.015 | 0.736 | 0.0 | 0.679 | 0.0 |
| random_forest | 30 | 0.732 +/- 0.014 | 0.667 +/- 0.014 | 0.679 +/- 0.026 | 0.656 +/- 0.014 | 0.719 | 0.013 | 0.659 | 0.008 |
| xgboost | 30 | 0.756 +/- 0.018 | 0.683 +/- 0.014 | 0.714 +/- 0.012 | 0.656 +/- 0.022 | 0.717 | 0.039 | 0.635 | 0.048 |
| lightgbm | 30 | 0.754 +/- 0.017 | 0.683 +/- 0.016 | 0.709 +/- 0.020 | 0.660 +/- 0.020 | 0.737 | 0.017 | 0.67 | 0.013 |
| catboost | 15 | 0.756 +/- 0.018 | 0.687 +/- 0.011 | 0.715 +/- 0.022 | 0.661 +/- 0.017 | 0.751 | 0.005 | 0.678 | 0.009 |
