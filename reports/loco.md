# Leave-one-county-out (training split only)

Train on five counties, score the sixth, at threshold 0.40. Best-to-worst gap: PR-AUC 0.084, recall 0.024.
The settings and the threshold were chosen using all six counties, so these scores may be a little better than for a truly new county.

| held_out_county | n | infected | pr_auc | roc_auc | recall | precision | f1 | mcc | pr_auc_lo | pr_auc_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| Bungoma | 1034 | 345 | 0.696 | 0.811 | 0.8 | 0.538 | 0.643 | 0.43 | 0.642 | 0.749 |
| Kakamega | 1042 | 345 | 0.757 | 0.833 | 0.82 | 0.559 | 0.665 | 0.471 | 0.718 | 0.798 |
| Nakuru | 1054 | 414 | 0.78 | 0.834 | 0.814 | 0.624 | 0.706 | 0.485 | 0.744 | 0.817 |
| Narok | 1085 | 422 | 0.779 | 0.823 | 0.801 | 0.616 | 0.696 | 0.471 | 0.743 | 0.818 |
| Trans Nzoia | 1031 | 378 | 0.77 | 0.834 | 0.81 | 0.59 | 0.682 | 0.466 | 0.727 | 0.811 |
| Uasin Gishu | 1067 | 373 | 0.736 | 0.817 | 0.796 | 0.57 | 0.664 | 0.452 | 0.688 | 0.778 |
