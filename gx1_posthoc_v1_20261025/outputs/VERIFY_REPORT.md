# Recomputation from per-case predictions

Overall: PASS (93 passed, 0 failed).
Tolerances: 1e-3 for AUC, QWK and C (probabilities exported to 6 decimals), 0.01 for hazard ratios, exact for counts.

| Check | Item | Recomputed | Locked RESULTS |
|---|---|---|---|
| PASS | every Korean case predicted once per seed, by the model whose test fold holds it | | |
| PASS | every Scottish case predicted by 25 models per encoder | | |
| PASS | ensembles equal the mean of their runs (max abs diff 8.0e-07) | | |
| PASS | conch korea AUC 3-4 vs 1-2 | 0.8785 | 0.8785 |
| PASS | conch korea AUC 2 vs 1 | 0.6916 | 0.6916 |
| PASS | conch korea QWK | 0.5825 | 0.5825 |
| PASS | conch scotland AUC 3-4 vs 1-2 | 0.8540 | 0.8540 |
| PASS | conch scotland AUC 2 vs 1 | 0.7328 | 0.7328 |
| PASS | conch scotland QWK | 0.5495 | 0.5495 |
| PASS | convnext_tiny korea AUC 3-4 vs 1-2 | 0.7974 | 0.7974 |
| PASS | convnext_tiny korea AUC 2 vs 1 | 0.6535 | 0.6535 |
| PASS | convnext_tiny korea QWK | 0.3940 | 0.3940 |
| PASS | convnext_tiny scotland AUC 3-4 vs 1-2 | 0.7692 | 0.7692 |
| PASS | convnext_tiny scotland AUC 2 vs 1 | 0.6917 | 0.6917 |
| PASS | convnext_tiny scotland QWK | 0.1882 | 0.1882 |
| PASS | gigapath korea AUC 3-4 vs 1-2 | 0.8940 | 0.8940 |
| PASS | gigapath korea AUC 2 vs 1 | 0.7584 | 0.7584 |
| PASS | gigapath korea QWK | 0.6441 | 0.6441 |
| PASS | gigapath scotland AUC 3-4 vs 1-2 | 0.8509 | 0.8509 |
| PASS | gigapath scotland AUC 2 vs 1 | 0.7283 | 0.7283 |
| PASS | gigapath scotland QWK | 0.4881 | 0.4881 |
| PASS | phikon-v2 korea AUC 3-4 vs 1-2 | 0.9154 | 0.9154 |
| PASS | phikon-v2 korea AUC 2 vs 1 | 0.7649 | 0.7649 |
| PASS | phikon-v2 korea QWK | 0.6711 | 0.6711 |
| PASS | phikon-v2 scotland AUC 3-4 vs 1-2 | 0.8406 | 0.8406 |
| PASS | phikon-v2 scotland AUC 2 vs 1 | 0.7389 | 0.7389 |
| PASS | phikon-v2 scotland QWK | 0.4828 | 0.4828 |
| PASS | resnet50 korea AUC 3-4 vs 1-2 | 0.7762 | 0.7762 |
| PASS | resnet50 korea AUC 2 vs 1 | 0.6092 | 0.6092 |
| PASS | resnet50 korea QWK | 0.3443 | 0.3443 |
| PASS | resnet50 scotland AUC 3-4 vs 1-2 | 0.7582 | 0.7582 |
| PASS | resnet50 scotland AUC 2 vs 1 | 0.6550 | 0.6550 |
| PASS | resnet50 scotland QWK | 0.4107 | 0.4107 |
| PASS | uni korea AUC 3-4 vs 1-2 | 0.9113 | 0.9113 |
| PASS | uni korea AUC 2 vs 1 | 0.7533 | 0.7533 |
| PASS | uni korea QWK | 0.6784 | 0.6784 |
| PASS | uni scotland AUC 3-4 vs 1-2 | 0.8632 | 0.8632 |
| PASS | uni scotland AUC 2 vs 1 | 0.7439 | 0.7439 |
| PASS | uni scotland QWK | 0.4805 | 0.4805 |
| PASS | uni2-h korea AUC 3-4 vs 1-2 | 0.9004 | 0.9004 |
| PASS | uni2-h korea AUC 2 vs 1 | 0.7504 | 0.7504 |
| PASS | uni2-h korea QWK | 0.6720 | 0.6720 |
| PASS | uni2-h scotland AUC 3-4 vs 1-2 | 0.8475 | 0.8475 |
| PASS | uni2-h scotland AUC 2 vs 1 | 0.7639 | 0.7639 |
| PASS | uni2-h scotland QWK | 0.5166 | 0.5166 |
| PASS | virchow2 korea AUC 3-4 vs 1-2 | 0.9056 | 0.9056 |
| PASS | virchow2 korea AUC 2 vs 1 | 0.7143 | 0.7143 |
| PASS | virchow2 korea QWK | 0.6293 | 0.6293 |
| PASS | virchow2 scotland AUC 3-4 vs 1-2 | 0.8813 | 0.8813 |
| PASS | virchow2 scotland AUC 2 vs 1 | 0.8078 | 0.8078 |
| PASS | virchow2 scotland QWK | 0.5588 | 0.5588 |
| PASS | vit_b_16 korea AUC 3-4 vs 1-2 | 0.8017 | 0.8017 |
| PASS | vit_b_16 korea AUC 2 vs 1 | 0.6750 | 0.6750 |
| PASS | vit_b_16 korea QWK | 0.4420 | 0.4420 |
| PASS | vit_b_16 scotland AUC 3-4 vs 1-2 | 0.7558 | 0.7558 |
| PASS | vit_b_16 scotland AUC 2 vs 1 | 0.6544 | 0.6544 |
| PASS | vit_b_16 scotland QWK | 0.3125 | 0.3125 |
| PASS | conch korea C, AI grade | 0.7937 | 0.7937 |
| PASS | conch scotland C, AI grade | 0.8030 | 0.8030 |
| PASS | convnext_tiny korea C, AI grade | 0.7366 | 0.7366 |
| PASS | convnext_tiny scotland C, AI grade | 0.7065 | 0.7065 |
| PASS | gigapath korea C, AI grade | 0.7292 | 0.7292 |
| PASS | gigapath scotland C, AI grade | 0.7377 | 0.7377 |
| PASS | phikon-v2 korea C, AI grade | 0.7371 | 0.7371 |
| PASS | korea recurrence set n, events | 295.0000, 48.0000 | 295.0000, 48.0000 |
| PASS | korea C, ISUP / Fuhrman / Leibovich | 0.5904, 0.7036, 0.8344 | 0.5904, 0.7036, 0.8344 |
| PASS | korea C within ISUP grade, AI grade | 0.7283 | 0.7283 |
| PASS | korea Cox(ISUP + AI): HR, CI | 4.2028, 2.5700, 6.8729 | 4.2028, 2.5700, 6.8729 |
| PASS | korea Cox(Leibovich + AI): HR | 1.2407 | 1.2407 |
| PASS | korea ISUP 1-2 split at own median: events above / below | 26.0000, 6.0000 | 26.0000, 6.0000 |
| PASS | korea ISUP 1-2 split: log-rank P | 0.0001 | 0.0001 |
| PASS | phikon-v2 scotland C, AI grade | 0.7100 | 0.7100 |
| PASS | scotland recurrence set n, events | 120.0000, 27.0000 | 120.0000, 27.0000 |
| PASS | scotland C, ISUP / Fuhrman / Leibovich | 0.6859, 0.7735, 0.8216 | 0.6859, 0.7735, 0.8216 |
| PASS | scotland C within ISUP grade, AI grade | 0.5973 | 0.5973 |
| PASS | scotland Cox(ISUP + AI): HR, CI | 3.2644, 1.0590, 10.0628 | 3.2644, 1.0590, 10.0628 |
| PASS | scotland Cox(Leibovich + AI): HR | 1.3043 | 1.3043 |
| PASS | scotland ISUP 1-2 split at own median: events above / below | 12.0000, 2.0000 | 12.0000, 2.0000 |
| PASS | scotland ISUP 1-2 split: log-rank P | 0.0075 | 0.0075 |
| PASS | resnet50 korea C, AI grade | 0.7443 | 0.7443 |
| PASS | resnet50 scotland C, AI grade | 0.7180 | 0.7180 |
| PASS | uni korea C, AI grade | 0.7322 | 0.7322 |
| PASS | uni scotland C, AI grade | 0.6978 | 0.6978 |
| PASS | uni2-h korea C, AI grade | 0.7386 | 0.7386 |
| PASS | uni2-h scotland C, AI grade | 0.7152 | 0.7152 |
| PASS | virchow2 korea C, AI grade | 0.7608 | 0.7608 |
| PASS | virchow2 scotland C, AI grade | 0.7239 | 0.7239 |
| PASS | vit_b_16 korea C, AI grade | 0.7464 | 0.7464 |
| PASS | vit_b_16 scotland C, AI grade | 0.6835 | 0.6835 |
| PASS | surv1 phikon-v2 Scotland C: Leibovich / H&E / Leibovich + H&E (frozen) | 0.8216, 0.7512, 0.8350 | 0.8216, 0.7512, 0.8350 |
| PASS | surv1 phikon-v2 Scotland split at Korean median: high n, log-rank P | 78.0000, 0.0670 | 78.0000, 0.0670 |
| PASS | surv1 uni Scotland C: Leibovich / H&E / Leibovich + H&E (frozen) | 0.8216, 0.7627, 0.8315 | 0.8216, 0.7627, 0.8315 |
| PASS | surv1 uni Scotland split at Korean median: high n, log-rank P | 69.0000, 0.0003 | 69.0000, 0.0003 |
