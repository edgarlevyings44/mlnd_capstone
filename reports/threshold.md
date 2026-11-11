# Threshold tuning (catboost, policy: recall_target)

Chosen threshold: **0.40** (bootstrap median 0.40, 95% interval [0.38, 0.42]).

On out-of-fold predictions this flags 49.5% of leaves, finds 80.6% of infected leaves (95% CI [0.789, 0.821]), and 58.8% of flagged leaves are truly infected (95% CI [0.570, 0.605]).

## Operating points
| operating_point | threshold | recall | precision | specificity | f1 | f2 | gmean | flagged_share |
|---|---|---|---|---|---|---|---|---|
| default 0.5 | 0.5 | 0.717 | 0.656 | 0.787 | 0.685 | 0.704 | 0.751 | 0.395 |
| max F1 | 0.46 | 0.758 | 0.632 | 0.751 | 0.689 | 0.729 | 0.755 | 0.432 |
| max F2 | 0.25 | 0.918 | 0.479 | 0.436 | 0.63 | 0.776 | 0.633 | 0.692 |
| recall >= 0.8 | 0.4 | 0.806 | 0.588 | 0.681 | 0.68 | 0.75 | 0.741 | 0.495 |

## Trade-off every 0.05
| threshold | recall | precision | specificity | f1 | f2 | gmean | flagged_share |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.0 | 0.364 | 0.014 | 0.534 | 0.741 | 0.119 | 0.991 |
| 0.1 | 0.985 | 0.38 | 0.095 | 0.549 | 0.747 | 0.306 | 0.933 |
| 0.15 | 0.967 | 0.408 | 0.209 | 0.574 | 0.759 | 0.449 | 0.855 |
| 0.2 | 0.944 | 0.44 | 0.322 | 0.6 | 0.768 | 0.552 | 0.774 |
| 0.25 | 0.918 | 0.479 | 0.436 | 0.63 | 0.776 | 0.633 | 0.692 |
| 0.3 | 0.884 | 0.512 | 0.526 | 0.649 | 0.772 | 0.682 | 0.622 |
| 0.35 | 0.849 | 0.555 | 0.616 | 0.672 | 0.768 | 0.724 | 0.552 |
| 0.4 | 0.806 | 0.588 | 0.681 | 0.68 | 0.75 | 0.741 | 0.495 |
| 0.45 | 0.765 | 0.624 | 0.74 | 0.688 | 0.732 | 0.753 | 0.442 |
| 0.5 | 0.717 | 0.656 | 0.787 | 0.685 | 0.704 | 0.751 | 0.395 |
| 0.55 | 0.663 | 0.69 | 0.832 | 0.676 | 0.668 | 0.743 | 0.347 |
| 0.6 | 0.61 | 0.718 | 0.865 | 0.66 | 0.629 | 0.726 | 0.306 |
| 0.65 | 0.543 | 0.747 | 0.896 | 0.629 | 0.574 | 0.698 | 0.262 |
| 0.7 | 0.487 | 0.786 | 0.925 | 0.602 | 0.528 | 0.672 | 0.224 |
| 0.75 | 0.42 | 0.835 | 0.953 | 0.559 | 0.466 | 0.633 | 0.181 |
| 0.8 | 0.349 | 0.871 | 0.971 | 0.498 | 0.396 | 0.582 | 0.144 |
| 0.85 | 0.261 | 0.897 | 0.983 | 0.405 | 0.304 | 0.507 | 0.105 |
| 0.9 | 0.17 | 0.933 | 0.993 | 0.288 | 0.204 | 0.411 | 0.066 |
| 0.95 | 0.065 | 0.949 | 0.998 | 0.122 | 0.08 | 0.255 | 0.025 |

Notes: The threshold was chosen on out-of-fold predictions only, so the test split was not used. Each of the five cross-validation catboost models was trained on 80% of the training rows, while the deployed catboost model is trained on all of them, so real recall may differ slightly. Class weights make the scores run higher than the true infection rate, so they are not probabilities. We retune the threshold if the model changes.
