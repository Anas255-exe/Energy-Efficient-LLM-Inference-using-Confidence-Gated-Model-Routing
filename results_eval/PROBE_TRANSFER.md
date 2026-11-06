## Semantic-entropy probe (T4 fp16, test split, gate for →7B)

| Draft | SEP layer | AUROC on errors: entropy / SE (10 samples) / correctness probe / SEP | ΔAUROC SEP − entropy [95% CI] | Saving @95%: entropy / SE / SEP |
|---|---|---|---|---|
| 0.5b | answer L8 | 0.744 / 0.818 / 0.762 / 0.741 | -0.001 [-0.052, +0.051] | -26% / -333% / -26% |
| 1.5b | answer L9 | 0.808 / 0.836 / 0.836 / 0.815 | +0.008 [-0.027, +0.045] | -22% / -432% / -17% |
| 3b | answer L13 | 0.802 / 0.814 / 0.812 / 0.778 | -0.023 [-0.058, +0.013] | +4% / -466% / -6% |

## Cross-dataset transfer (AUROC on the held-out source's test rows)

| Draft | Held-out source | n | Entropy | Probe in-domain → transfer | SEP in-domain → transfer |
|---|---|---|---|---|---|
| 0.5b | triviaqa | 306 | 0.855 | 0.774 → 0.749 | 0.812 → 0.819 |
| 0.5b | sciq | 293 | 0.731 | 0.682 → 0.629 | 0.654 → 0.652 |
| 0.5b | nq | 301 | 0.628 | 0.732 → 0.625 | 0.621 → 0.579 |
| 1.5b | triviaqa | 306 | 0.832 | 0.831 → 0.771 | 0.811 → 0.774 |
| 1.5b | sciq | 293 | 0.750 | 0.775 → 0.777 | 0.732 → 0.729 |
| 1.5b | nq | 301 | 0.819 | 0.791 → 0.818 | 0.774 → 0.738 |
| 3b | triviaqa | 306 | 0.814 | 0.836 → 0.791 | 0.802 → 0.798 |
| 3b | sciq | 293 | 0.782 | 0.774 → 0.773 | 0.741 → 0.750 |
| 3b | nq | 301 | 0.782 | 0.732 → 0.712 | 0.703 → 0.748 |

## Threshold transfer: entropy threshold set for 95% retention, applied unchanged

| Pair | Held-out source | Retention achieved (target 95%) | Saving held-out vs in-sample |
|---|---|---|---|
| 7b->14b | triviaqa | 93.7% | +18% vs +13% |
| 7b->14b | sciq | 97.7% | +35% vs +45% |
| 7b->14b | nq | 91.6% | +16% vs +9% |
| 7b->32b | triviaqa | 93.8% | +36% vs +31% |
| 7b->32b | sciq | 97.2% | +47% vs +61% |
| 7b->32b | nq | 93.2% | +30% vs +21% |
| 14b->32b | triviaqa | 94.9% | +55% vs +55% |
| 14b->32b | sciq | 96.6% | +57% vs +59% |
| 14b->32b | nq | 91.7% | +59% vs +42% |
| 3b->7b fp16 | triviaqa | 94.2% | -4% vs -4% |
| 3b->7b fp16 | sciq | 96.7% | +9% vs +12% |
| 3b->7b fp16 | nq | 94.3% | -12% vs -15% |

| Threshold from → applied to | Retention achieved | Escalated | Saving |
|---|---|---|---|
| 7b->14b => 7b->32b | 91.3% | 22% | +56% |
| 7b->14b => 14b->32b | 95.7% | 9% | +50% |
| 7b->32b => 7b->14b | 98.1% | 39% | +7% |
| 7b->32b => 14b->32b | 97.4% | 19% | +40% |
| 14b->32b => 7b->14b | 91.6% | 13% | +34% |
| 14b->32b => 7b->32b | 87.4% | 12% | +66% |
