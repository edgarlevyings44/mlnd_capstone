# Final evaluation on the held-out test split (catboost, threshold 0.40)

1579 test leaves, 570 infected. Scored once; nothing was changed after seeing these numbers.

## Overall (95% bootstrap CI)
| metric | estimate | ci_lo | ci_hi |
|---|---|---|---|
| precision | 0.586 | 0.55 | 0.622 |
| recall | 0.791 | 0.759 | 0.825 |
| specificity | 0.685 | 0.656 | 0.714 |
| f1 | 0.674 | 0.643 | 0.702 |
| mcc | 0.457 | 0.412 | 0.503 |
| gmean | 0.736 | 0.712 | 0.758 |
| roc_auc | 0.819 | 0.798 | 0.843 |
| pr_auc | 0.74 | 0.705 | 0.777 |

At the default 0.5 threshold, for reference: recall 0.702, precision 0.664, F1 0.683.

## Per county (test split; small samples, wide intervals)
| county | n | infected | pr_auc | roc_auc | recall | precision | f1 | mcc | pr_auc_lo | pr_auc_hi | recall_lo | recall_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bungoma | 271 | 101 | 0.764 | 0.839 | 0.822 | 0.597 | 0.692 | 0.476 | 0.673 | 0.844 | 0.747 | 0.897 |
| Kakamega | 249 | 92 | 0.729 | 0.802 | 0.783 | 0.6 | 0.679 | 0.461 | 0.639 | 0.817 | 0.69 | 0.858 |
| Nakuru | 267 | 84 | 0.738 | 0.813 | 0.726 | 0.53 | 0.613 | 0.404 | 0.646 | 0.821 | 0.635 | 0.819 |
| Narok | 283 | 112 | 0.767 | 0.835 | 0.839 | 0.61 | 0.707 | 0.48 | 0.689 | 0.84 | 0.764 | 0.906 |
| Trans Nzoia | 249 | 85 | 0.747 | 0.827 | 0.788 | 0.604 | 0.684 | 0.496 | 0.655 | 0.826 | 0.707 | 0.861 |
| Uasin Gishu | 260 | 96 | 0.712 | 0.792 | 0.771 | 0.569 | 0.655 | 0.414 | 0.628 | 0.801 | 0.685 | 0.857 |

Best-to-worst county gap: PR-AUC 0.055, recall 0.113.

## Calibration
Brier 0.1677 (constant-prevalence baseline 0.2307), ECE 0.084, mean predicted risk 0.445 vs true infection rate 0.361.

| n | mean_predicted | observed_rate |
|---|---|---|
| 158.0 | 0.09 | 0.076 |
| 158.0 | 0.156 | 0.108 |
| 158.0 | 0.213 | 0.139 |
| 158.0 | 0.282 | 0.146 |
| 158.0 | 0.352 | 0.247 |
| 157.0 | 0.433 | 0.268 |
| 158.0 | 0.537 | 0.456 |
| 158.0 | 0.663 | 0.563 |
| 158.0 | 0.798 | 0.747 |
| 158.0 | 0.928 | 0.861 |
