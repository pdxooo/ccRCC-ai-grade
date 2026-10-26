# Supplementary Information

**External validation of a continuous AI grade for clear cell renal cell carcinoma**

Contents: Supplementary Notes 1–5; Supplementary Tables 1–6; Supplementary Figures 1–10.

Unless explicitly labelled as adjusted, P values and confidence intervals are nominal. The post hoc diagnostic and multiplicity analyses added during revision are reported in Supplementary Note 5 and Tables 5–6; all earlier result values are retained.

## Supplementary Note 1. Development history, pre-specification and reproducibility records

**Development before the formal studies.** Between 10 and 21 October 2026, models were developed on the Korean cohort only, with the same locked five-fold split as the formal studies. Nine encoders and their concatenation were each trained with early stopping on the validation folds (50 runs), encoder subsets were concatenated or their probabilities averaged, and fixed three-encoder fusions were tried. The Korean out-of-fold results of these runs were inspected, and Phikon-v2, the best single encoder, became the primary encoder. The Scottish clinical tables, including their grade distributions, were inspected. The grading and direct-recurrence protocols were locked before their formal downstream Scottish predictions; this statement does not cover technical quality control or frozen feature extraction. Exploratory analyses were planned after the grading results, as shown below. The formal studies then fixed 40 training epochs without early stopping, the analyses and a decision rule per study. Their Korean out-of-fold estimates therefore come from the cohort used for development; only the Scottish estimates are external.

**Formal studies.** Every formal study followed the same sequence. A written protocol fixed the question, the primary encoder, the metrics and the decision rule. The code was written and tested. A lock manifest then recorded the SHA-256 hash of every code, configuration and input file, and every script checked these hashes before running and refused to run on a changed file. The scripts required a passing technical check comparing two-epoch training both uninterrupted and with a pause and resume on a second GPU, with bit-identical model, optimiser and random-number states and predictions. The current archive contains this check code and summary manifests, but not the original per-run check receipts or checkpoints; the present audit therefore verifies the requirement rather than independently reproducing the historical check. Code signatures below are SHA-256 hashes of the locked code; manifest hashes are SHA-256 hashes of the lock file itself (first 16 hexadecimal characters shown; full values are in the deposited code). Times are UTC+8.

| Study | Locked (UTC+8) | Code signature | Manifest | Verdict |
|---|---|---|---|---|
| Grading external validation and AI grade (main study) | 2026-10-22 17:42:29 | 83e88873fb8d1e02 | 9bbcb3d88a7fd960 | Grading success; AI grade positive |
| H&E recurrence model (main study) | 2026-10-22 16:45:38 | 6f5db834f21fd181 | 3ba39ee08be3746a | Inconclusive |
| Secondary analyses of the recurrence model | 2026-10-22 17:36:03 | 59064b2872c2b3af | ba13437c870b833e | Descriptive |
| Auxiliary Fuhrman supervision | 2026-10-22 16:01:02 | 6cfa394be44aff64 | 9145496a480a0ca6 | Inconclusive |
| Patch magnification pilot (stage 1) | 2026-10-21 21:19:54 | 26fd7cc1201ba7d3 | 02907d76af58ebf1 | Inconclusive; stage 2 not run |
| Encoder fusion | Comparisons fixed before training | 3a4132333fb1a781 | Read-only signed files | No gain |
| Exploratory analyses | 2026-10-23 11:17:03 | — | 600ffbe58fe53a5a | Exploratory |
| Post hoc analyses (Supplementary Note 4) | 2026-10-25 22:20:01 | — | 4b1f1be405dcf171 | Post hoc |

Decision rules, as fixed in each protocol (primary encoder Phikon-v2 unless stated):

- **Grading.** Success if the Scottish high-versus-low AUC is ≥ 0.80 with a 95% CI lower bound > 0.70; failure if the AUC is < 0.70; otherwise inconclusive.
- **AI grade.** Positive if the Scottish gain in Harrell's C over ISUP grade is ≥ 0.02 and the AI-grade HR adjusted for ISUP grade has a 95% CI lower bound > 1; negative if the gain is ≤ 0 or the HR ≤ 1; otherwise inconclusive.
- **H&E recurrence model.** Positive if, in Scotland with the Korean coefficients unchanged, adding the H&E risk to the Leibovich score raises Harrell's C by ≥ 0.02 and the H&E HR has a 95% CI lower bound > 1; negative if the gain is ≤ 0 or the HR ≤ 1; otherwise inconclusive.
- **Secondary analyses of the recurrence model.** Descriptive; no decision rule (Supplementary Note 2).
- **Auxiliary Fuhrman supervision.** Positive if grade 1 versus 2 AUC rises by ≥ 0.02 over ISUP-only training (at least 4 of 5 seeds positive) and by ≥ 0.01 over shuffled Fuhrman labels (at least 4 of 5 seeds positive); negative if either difference is ≤ 0; otherwise inconclusive.
- **Patch magnification pilot.** Proceed to stage 2 if, for UNI or Virchow2, higher-magnification sub-patches raise the grade 1 versus 2 AUC over standard patches by ≥ 0.03, with all three seeds positive, and beat their geometry control; negative if both encoders gain < 0.03 with a gain ≤ 0 or inconsistent seeds; otherwise the investigators decide.
- **Encoder fusion.** Three pre-specified comparisons (P1–P3) of the high-versus-low AUC against Phikon-v2 with paired bootstrap intervals, reported as found; declared exploratory in advance (Supplementary Note 3).
- **Exploratory analyses.** Plan written after the grading results were known and locked before computing; no decision rule.
- **Post hoc analyses.** Planned after an internal critical review of the draft, when all results were known, and locked before computing; no decision rule (Supplementary Note 4).

The original AI-grade decision rule compared AI grade alone with ISUP grade alone and tested an adjusted association. It did not test whether adding AI grade improved a combined predictor. The latter question was addressed separately by the post hoc analyses in Supplementary Note 4, which did not demonstrate external incremental discrimination.

## Supplementary Note 2. Pre-specified secondary analyses of the recurrence model

These analyses were specified and locked after the protocol of the recurrence study and before its results were seen. All compare the Leibovich score alone with the Leibovich score plus the H&E risk score; Korean values are cross-fitted within the development cohort and are therefore internal; Scottish values use the Korean coefficients unchanged. The H&E risk entering these models was standardised within each cohort (within each Korean fold, and within Scotland) from the H&E scores alone, without outcomes, so the Scottish analyses use the Scottish score distribution. Uno's C was truncated at 60 months. Calibration compares predicted and Kaplan–Meier observed recurrence by 36 months in five risk groups in Korea and three in Scotland. Net reclassification used predicted 36-month risk, either category-free or with cut-offs at 10% and 30%. Intervals are case-bootstrap percentile intervals (2,000 resamples). None of these analyses carried a decision rule.

**Supplementary Note Table 2a.** Discrimination. Cohort column gives recurrences/patients.

| Encoder | Cohort | Measure | Leibovich | + H&E | Δ (95% CI) |
|---|---|---|---|---|---|
| Phikon-v2 | Korea (48/295) | Uno's C (60 months) | 0.834 | 0.838 | +0.003 (−0.010 to +0.016) |
| Phikon-v2 | Korea (48/295) | AUC, 36 months | 0.879 | 0.882 | +0.003 (−0.012 to +0.017) |
| Phikon-v2 | Korea (48/295) | AUC, 60 months | 0.850 | 0.854 | +0.004 (−0.010 to +0.017) |
| Phikon-v2 | Scotland (27/120) | Uno's C (60 months) | 0.809 | 0.823 | +0.014 (−0.004 to +0.033) |
| Phikon-v2 | Scotland (27/120) | AUC, 36 months | 0.853 | 0.868 | +0.015 (−0.005 to +0.037) |
| Phikon-v2 | Scotland (27/120) | AUC, 60 months | 0.843 | 0.859 | +0.016 (−0.009 to +0.042) |
| UNI | Korea (48/295) | Uno's C (60 months) | 0.834 | 0.853 | +0.019 (−0.005 to +0.046) |
| UNI | Korea (48/295) | AUC, 36 months | 0.879 | 0.900 | +0.021 (−0.010 to +0.053) |
| UNI | Korea (48/295) | AUC, 60 months | 0.850 | 0.873 | +0.022 (−0.003 to +0.049) |
| UNI | Scotland (27/120) | Uno's C (60 months) | 0.809 | 0.815 | +0.007 (−0.017 to +0.032) |
| UNI | Scotland (27/120) | AUC, 36 months | 0.853 | 0.857 | +0.004 (−0.020 to +0.030) |
| UNI | Scotland (27/120) | AUC, 60 months | 0.843 | 0.845 | +0.002 (−0.028 to +0.030) |

**Supplementary Note Table 2b.** Calibration of predicted recurrence by 36 months (slope and observed-to-expected ratio, O/E) and net reclassification (NRI).

| Encoder | Cohort | Slope, Leibovich | O/E, Leibovich | Slope, + H&E | O/E, + H&E | NRI, category-free (95% CI) | NRI, 10%/30% |
|---|---|---|---|---|---|---|---|
| Phikon-v2 | Korea | 0.98 | 0.99 | 0.96 | 1.00 | −0.108 (−0.459 to +0.235) | +0.012 |
| Phikon-v2 | Scotland | 0.86 | 0.93 | 0.88 | 0.96 | +0.047 (−0.424 to +0.550) | 0.000 |
| UNI | Korea | 0.98 | 0.99 | 0.93 | 0.99 | −0.011 (−0.346 to +0.332) | +0.108 |
| UNI | Scotland | 0.86 | 0.93 | 0.87 | 1.06 | −0.078 (−0.524 to +0.400) | +0.180 |

**Supplementary Note Table 2c.** Net benefit for recurrence by 36 months at selected threshold probabilities (full curves in Fig. 7e and Supplementary Fig. 8).

| Encoder | Cohort | Threshold | Net benefit, Leibovich | Net benefit, + H&E | Net benefit, treat all |
|---|---|---|---|---|---|
| Phikon-v2 | Korea | 10% | 0.084 | 0.084 | 0.032 |
| Phikon-v2 | Korea | 20% | 0.064 | 0.059 | −0.089 |
| Phikon-v2 | Korea | 30% | 0.045 | 0.048 | −0.244 |
| Phikon-v2 | Scotland | 10% | 0.118 | 0.118 | 0.090 |
| Phikon-v2 | Scotland | 20% | 0.097 | 0.106 | −0.024 |
| Phikon-v2 | Scotland | 30% | 0.074 | 0.074 | −0.170 |
| UNI | Korea | 10% | 0.084 | 0.087 | 0.032 |
| UNI | Korea | 20% | 0.064 | 0.069 | −0.089 |
| UNI | Korea | 30% | 0.045 | 0.058 | −0.244 |
| UNI | Scotland | 10% | 0.118 | 0.130 | 0.090 |
| UNI | Scotland | 20% | 0.097 | 0.095 | −0.024 |
| UNI | Scotland | 30% | 0.074 | 0.089 | −0.170 |

**Supplementary Note Table 2d.** Recurrences/patients by H&E risk group (split at the Korean median) within Leibovich risk groups (0–2, 3–5, ≥ 6). Descriptive.

| Encoder | Cohort | Leibovich group | Patients (recurrences) | H&E high | H&E low | Log-rank P |
|---|---|---|---|---|---|---|
| Phikon-v2 | Korea | low | 132 (2) | 1/30 | 1/102 | 0.35 |
| Phikon-v2 | Korea | intermediate | 124 (23) | 18/84 | 5/40 | 0.23 |
| Phikon-v2 | Korea | high | 39 (23) | 21/33 | 2/6 | 0.33 |
| Phikon-v2 | Scotland | low | 47 (2) | 1/21 | 1/26 | 0.96 |
| Phikon-v2 | Scotland | intermediate | 50 (10) | 8/36 | 2/14 | 0.60 |
| Phikon-v2 | Scotland | high | 23 (15) | 13/21 | 2/2 | 0.29 |
| UNI | Korea | low | 132 (2) | 2/37 | 0/95 | 0.023 |
| UNI | Korea | intermediate | 124 (23) | 17/76 | 6/48 | 0.18 |
| UNI | Korea | high | 39 (23) | 23/34 | 0/5 | 0.019 |
| UNI | Scotland | low | 47 (2) | 1/17 | 1/30 | 0.76 |
| UNI | Scotland | intermediate | 50 (10) | 8/30 | 2/20 | 0.19 |
| UNI | Scotland | high | 23 (15) | 15/22 | 0/1 | 0.22 |

## Supplementary Note 3. Studies that did not improve the grading model

**Auxiliary Fuhrman supervision.** We asked whether Fuhrman grade, used as a second training target, helps the model separate ISUP grades 1 and 2. Three arms were trained on the locked Korean folds with five seeds each: A, ISUP only; B, ISUP plus an auxiliary head for Fuhrman 1–2 versus 3–4; C, ISUP plus the same head with Fuhrman labels shuffled within ISUP grade, which keeps any label correlation but removes the image correspondence. The primary metric was the out-of-fold grade 1 versus 2 AUC among ISUP grade 1–2 tumours (240 patients), scored by P(2)/(P(1) + P(2)) after averaging seeds.

| Encoder | Arm | Grade 1 vs 2 AUC (95% CI) | Per seed | High vs low AUC | QWK |
|---|---|---|---|---|---|
| Phikon-v2 | A | 0.764 (0.704–0.822) | 0.766, 0.755, 0.762, 0.766, 0.767 | 0.914 | 0.668 |
| Phikon-v2 | B | 0.776 (0.716–0.831) | 0.775, 0.773, 0.774, 0.773, 0.776 | 0.921 | 0.694 |
| Phikon-v2 | C | 0.765 (0.705–0.821) | 0.766, 0.749, 0.762, 0.766, 0.770 | 0.915 | 0.673 |
| UNI | A | 0.757 (0.695–0.816) | 0.756, 0.750, 0.759, 0.753, 0.761 | 0.911 | 0.677 |
| UNI | B | 0.757 (0.697–0.814) | 0.754, 0.752, 0.759, 0.753, 0.756 | 0.913 | 0.679 |
| UNI | C | 0.758 (0.695–0.816) | 0.763, 0.741, 0.757, 0.754, 0.758 | 0.912 | 0.682 |

With Phikon-v2, B − A was +0.012 (95% CI −0.002 to +0.026; all five seeds positive) and B − C +0.011 (−0.003 to +0.024). The gain was below the pre-specified 0.02, so the result was inconclusive. With UNI, B − A was 0.000.

**Patch magnification pilot.** We asked whether features from higher-magnification sub-patches separate grades 1 and 2 better. In 120 Korean cases (60 grade 1, 60 grade 2; stage 1 of a planned two-stage study), each 128 µm location was represented by its standard patch (L), by four 64 µm sub-patches decoded at native resolution (H), or by the same four quadrants cut from the standard patch and upsampled (Hd, which controls for the change in geometry without adding detail). A reference arm used standard features sampled from all patches of the slide (Lfull). UNI and Virchow2 were tested with three seeds and five folds.

| Encoder | Arm | Grade 1 vs 2 AUC (95% CI) | Per seed |
|---|---|---|---|
| UNI | L | 0.687 (0.586–0.778) | 0.684, 0.698, 0.678 |
| UNI | H | 0.697 (0.600–0.784) | 0.693, 0.699, 0.686 |
| UNI | Hd | 0.692 (0.596–0.781) | 0.687, 0.701, 0.684 |
| UNI | Lfull | 0.692 (0.592–0.781) | 0.691, 0.700, 0.679 |
| Virchow2 | L | 0.700 (0.599–0.792) | 0.698, 0.686, 0.697 |
| Virchow2 | H | 0.684 (0.582–0.774) | 0.692, 0.684, 0.684 |
| Virchow2 | Hd | 0.684 (0.581–0.774) | 0.691, 0.687, 0.681 |
| Virchow2 | Lfull | 0.699 (0.598–0.789) | 0.694, 0.692, 0.696 |

H − L was +0.010 (−0.018 to +0.036) with UNI and −0.016 (−0.060 to +0.025) with Virchow2. Neither encoder met the criterion for stage 2; the result was inconclusive and stage 2 was not run.

**Encoder fusion.** We asked whether combining pathology encoders improves grading beyond the best single encoder, after ruling out the effect of simply averaging more models. Four members (Phikon-v2, UNI, Prov-GigaPath and UNI2-h) were trained with four seeds each on the locked Korean folds, with model selection by validation cross-entropy. Three comparisons were fixed before training, all on the out-of-fold high-versus-low AUC with paired bootstrap intervals: P1, one model of each of the four encoders against four Phikon-v2 seeds (equal numbers of models); P2, late fusion of all 16 models against four Phikon-v2 seeds; P3, a learned fusion that projects each encoder to 128 dimensions, averages them and drops whole encoders at random during training, against Phikon-v2.

| Comparison | Δ high-vs-low AUC (95% CI) | Δ QWK (95% CI) | Δ macro-F1 (95% CI) |
|---|---|---|---|
| P1, 4 encoders vs Phikon-v2 × 4 seeds | −0.013 (−0.028 to 0.000) | +0.004 (−0.040 to +0.050) | −0.015 (−0.088 to +0.060) |
| P2, 16-model late fusion vs Phikon-v2 × 4 seeds | −0.012 (−0.025 to 0.000) | −0.003 (−0.049 to +0.043) | −0.022 (−0.096 to +0.053) |
| P3, learned fusion vs Phikon-v2 | −0.006 (−0.027 to +0.018) | +0.017 (−0.023 to +0.057) | −0.031 (−0.105 to +0.044) |

The reference (Phikon-v2, four seeds) had a high-versus-low AUC of 0.910. Two disclosures apply. The four members were chosen by validation macro-F1 in an earlier round, and validation patients in those folds were also out-of-fold test patients, so test labels influenced member choice. This was also the fourth fusion attempt on the same Korean out-of-fold predictions. We therefore declared the study exploratory before it ran.

## Supplementary Note 4. Post hoc analyses and per-case predictions

These analyses were added on 25 October 2026 after an internal critical review of the draft, when all results of the formal studies were known. Their plan and code (`gx1_posthoc_v1_20261025`) were locked by SHA-256 manifest (2026-10-25 22:20:01 UTC+8; manifest 4b1f1be405dcf171) before any of the numbers below was computed, and every planned analysis is reported. Encoder Phikon-v2; recurrence sets and predictions as in the grading study; Cox models, Harrell's C, log-rank tests and 2,000-replicate case bootstraps with the grading study's own functions. Categorical ISUP coding: indicators for grade 2 and grades 3–4 against grade 1 (grades 3 and 4 merged because Scotland has two grade 4 tumours).

**Supplementary Note Table 4a.** Discrimination of Cox models with ISUP grade alone and with ISUP grade plus the AI grade. Korea: cross-fitted over the locked folds. Scotland: models fitted on the Korean recurrence set and applied unchanged.

| ISUP coding | Cohort | C, ISUP | C, ISUP + AI grade | ΔC (95% CI) |
|---|---|---|---|---|
| Categorical | Korea | 0.551 | 0.727 | +0.176 (+0.074 to +0.281) |
| Categorical | Scotland | 0.677 | 0.623 | −0.054 (−0.223 to +0.112) |
| Numeric | Korea | 0.594 | 0.735 | +0.141 (+0.046 to +0.236) |
| Numeric | Scotland | 0.686 | 0.631 | −0.055 (−0.223 to +0.106) |

In the Korean model with categorical ISUP grade and the AI grade, the log hazard ratios were −0.68 (ISUP 2), −1.16 (ISUP 3–4) and +1.45 (AI grade, per unit); in the Korean model with ISUP grade alone they were +0.12 and +0.90. Once the AI grade was included, ISUP grade thus carried a negative weight in Korea, which did not hold in Scotland.

**Supplementary Note Table 4b.** Hazard ratio of the AI grade (per grade unit) adjusted for ISUP grade, fitted within each cohort.

| Cohort | ISUP coding | HR (95% CI) | P | Likelihood-ratio P |
|---|---|---|---|---|
| Korea | Categorical | 4.27 (2.61–6.98) | 7.2 × 10⁻⁹ | 1.8 × 10⁻⁸ |
| Korea | Numeric (formal study) | 4.20 (2.57–6.87) | 1.1 × 10⁻⁸ | 2.6 × 10⁻⁸ |
| Scotland | Categorical | 3.83 (1.26–11.66) | 0.018 | 0.017 |
| Scotland | Numeric (formal study) | 3.26 (1.06–10.06) | 0.039 | 0.039 |

**Low-grade split at the Korean threshold.** The threshold was the median AI grade of the Korean ISUP 1–2 recurrence-set patients (1.58). Applied unchanged to the 86 Scottish ISUP 1–2 patients, it placed 72 above it (14 recurrences) and 14 at or below it (no recurrence); log-rank P = 0.12. At the Scottish median (2.02), the split was 12 of 43 versus 2 of 43 (P = 0.0075). Scottish AI grades spanned a narrower range: the medians for ISUP grades 1–4 were 1.41, 1.75, 2.81 and 3.42 in Korea and 1.75, 2.14, 2.53 and 3.02 in Scotland.

**Per-case predictions.** The folder `gx1_posthoc_v1_20261025/outputs` of the code release holds:

- for all nine encoders, the per-run and ensemble ISUP probabilities and AI grades, with labels and outcomes;
- the locked Korean fold assignment;
- manifests of the 225 grading and 50 recurrence runs, with the SHA-256 hashes of their receipts and prediction files;
- the per-case H&E risk of the recurrence study.

`verify_from_predictions.py` recomputes the following from these files alone, with its own implementations and no study code:

- the out-of-fold structure of the Korean predictions;
- the grading AUCs and QWK of every encoder;
- the C indices, Cox hazard ratios and low-grade splits of the AI grade;
- the main results of the recurrence study.

All 93 checks agreed with the locked results.

## Supplementary Note 5. Post hoc statistical reporting audit

These analyses were added on 26 October 2026 after all existing results were known. They did not retrain any image model, change labels, or replace external predictions. Methods are described in the main text. The plan, R script, input hashes, complete term-level tests, coefficient comparisons and scaled Schoenfeld diagnostic plots are included in `statistical_audit_v1_20261026/`.

The audit refitted 42 Cox models with Breslow ties. Coefficients for all 36 AI-grade models with numeric ISUP or Leibovich adjustment agreed with the archived estimates (HR differences < 0.001; Wald P differences < 0.00001). Independent NumPy calculations reproduced the Holm-adjusted P values. There were no fitting warnings. The diagnostics test specified covariate effects over time; failure to reject is not evidence that an assumption is true. These checks did not include every exploratory biomarker or subgroup model.

None of the term-level or global tests in the ten core models was below 0.05 with either the Kaplan–Meier or rank time transformation, so the planned conditional time-interaction sensitivity for core models was not triggered. Across all 42 fits, one nominal diagnostic flag occurred: the Scottish CONCH AI-grade term with numeric ISUP adjustment (Kaplan–Meier transform P = 0.0308); it was not flagged by the rank transformation. No diagnostic multiplicity adjustment was used to dismiss this flag. CONCH's constant-HR interpretation is therefore qualified.

In the nine-encoder association screen, all nine Korean associations remained below 0.05 after within-cohort Holm correction. In Scotland, only CONCH and ResNet50 remained below 0.05. Phikon-v2 changed from nominal P = 0.0394 to adjusted P = 0.1102. This is a post hoc multiplicity sensitivity analysis, not retrospective registration or a new model-selection exercise. No encoder was selected or promoted on the basis of these results. The original Scottish HR and its nominal confidence interval remain reported alongside this limitation. Other secondary and exploratory P values are unadjusted, without a claim of study-wide error control.

## Supplementary Table 1. All 225 grading runs, by encoder and seed

Each seed comprises five fold models. Korean values are out-of-fold over the five folds; Scottish values are the mean prediction of the seed's five fold models. AUCs are for ISUP grade 3–4 versus 1–2 and for grade 2 versus 1 within grades 1–2; C is Harrell's C of the AI grade for recurrence. The ensembles used in the main analysis average all seeds (Korea) or all 25 models (Scotland).

| Encoder | Seed | Korea, 3–4 vs 1–2 | Korea, 2 vs 1 | Korea, C | Scotland, 3–4 vs 1–2 | Scotland, 2 vs 1 | Scotland, C |
|---|---|---|---|---|---|---|---|
| Phikon-v2 | 0 | 0.919 | 0.759 | 0.735 | 0.843 | 0.739 | 0.711 |
| Phikon-v2 | 1 | 0.909 | 0.768 | 0.732 | 0.841 | 0.737 | 0.710 |
| Phikon-v2 | 2 | 0.917 | 0.764 | 0.735 | 0.839 | 0.734 | 0.710 |
| Phikon-v2 | 3 | 0.910 | 0.761 | 0.738 | 0.840 | 0.737 | 0.708 |
| Phikon-v2 | 4 | 0.908 | 0.762 | 0.737 | 0.840 | 0.743 | 0.710 |
| UNI | 0 | 0.913 | 0.753 | 0.732 | 0.862 | 0.739 | 0.697 |
| UNI | 1 | 0.908 | 0.754 | 0.721 | 0.866 | 0.742 | 0.699 |
| UNI | 2 | 0.910 | 0.757 | 0.730 | 0.862 | 0.739 | 0.696 |
| UNI | 3 | 0.908 | 0.748 | 0.733 | 0.861 | 0.741 | 0.697 |
| UNI | 4 | 0.910 | 0.754 | 0.736 | 0.862 | 0.742 | 0.697 |
| UNI2-h | 0 | 0.902 | 0.755 | 0.736 | 0.850 | 0.770 | 0.718 |
| UNI2-h | 1 | 0.892 | 0.747 | 0.724 | 0.847 | 0.762 | 0.718 |
| UNI2-h | 2 | 0.897 | 0.752 | 0.740 | 0.846 | 0.764 | 0.717 |
| UNI2-h | 3 | 0.896 | 0.747 | 0.743 | 0.844 | 0.761 | 0.714 |
| UNI2-h | 4 | 0.903 | 0.746 | 0.741 | 0.847 | 0.765 | 0.714 |
| Virchow2 | 0 | 0.903 | 0.713 | 0.758 | 0.881 | 0.809 | 0.722 |
| Virchow2 | 1 | 0.899 | 0.722 | 0.759 | 0.881 | 0.807 | 0.726 |
| Virchow2 | 2 | 0.900 | 0.719 | 0.762 | 0.883 | 0.807 | 0.725 |
| Virchow2 | 3 | 0.909 | 0.709 | 0.759 | 0.878 | 0.808 | 0.722 |
| Virchow2 | 4 | 0.901 | 0.707 | 0.763 | 0.882 | 0.810 | 0.723 |
| Prov-GigaPath | 0 | 0.898 | 0.759 | 0.726 | 0.852 | 0.728 | 0.739 |
| Prov-GigaPath | 1 | 0.885 | 0.761 | 0.723 | 0.848 | 0.732 | 0.730 |
| Prov-GigaPath | 2 | 0.896 | 0.761 | 0.733 | 0.852 | 0.727 | 0.736 |
| Prov-GigaPath | 3 | 0.888 | 0.751 | 0.734 | 0.852 | 0.727 | 0.736 |
| Prov-GigaPath | 4 | 0.896 | 0.760 | 0.729 | 0.853 | 0.726 | 0.737 |
| CONCH | 0 | 0.880 | 0.695 | 0.794 | 0.854 | 0.730 | 0.802 |
| CONCH | 1 | 0.881 | 0.685 | 0.791 | 0.850 | 0.730 | 0.803 |
| CONCH | 2 | 0.877 | 0.694 | 0.790 | 0.856 | 0.729 | 0.802 |
| CONCH | 3 | 0.880 | 0.689 | 0.796 | 0.854 | 0.738 | 0.803 |
| CONCH | 4 | 0.864 | 0.687 | 0.794 | 0.853 | 0.728 | 0.802 |
| ResNet50 | 0 | 0.782 | 0.609 | 0.740 | 0.759 | 0.661 | 0.718 |
| ResNet50 | 1 | 0.773 | 0.598 | 0.734 | 0.752 | 0.644 | 0.707 |
| ResNet50 | 2 | 0.787 | 0.618 | 0.753 | 0.764 | 0.657 | 0.724 |
| ResNet50 | 3 | 0.766 | 0.608 | 0.733 | 0.750 | 0.639 | 0.706 |
| ResNet50 | 4 | 0.763 | 0.604 | 0.742 | 0.767 | 0.657 | 0.714 |
| ConvNeXt-T | 0 | 0.803 | 0.650 | 0.744 | 0.767 | 0.699 | 0.708 |
| ConvNeXt-T | 1 | 0.789 | 0.648 | 0.724 | 0.769 | 0.685 | 0.705 |
| ConvNeXt-T | 2 | 0.798 | 0.652 | 0.737 | 0.765 | 0.693 | 0.712 |
| ConvNeXt-T | 3 | 0.796 | 0.658 | 0.731 | 0.766 | 0.695 | 0.701 |
| ConvNeXt-T | 4 | 0.787 | 0.650 | 0.737 | 0.768 | 0.688 | 0.708 |
| ViT-B/16 | 0 | 0.797 | 0.670 | 0.740 | 0.754 | 0.647 | 0.685 |
| ViT-B/16 | 1 | 0.797 | 0.677 | 0.733 | 0.760 | 0.652 | 0.682 |
| ViT-B/16 | 2 | 0.808 | 0.672 | 0.749 | 0.757 | 0.647 | 0.682 |
| ViT-B/16 | 3 | 0.798 | 0.675 | 0.744 | 0.749 | 0.678 | 0.682 |
| ViT-B/16 | 4 | 0.799 | 0.676 | 0.751 | 0.763 | 0.658 | 0.687 |

## Supplementary Table 2. Exploratory analyses

Counts in brackets are recurrences/patients. High-grade-like patches have P(grade 3) + P(grade 4) > 0.5; the fraction of such patches was computed for each model and averaged over models. Morphology and P21+/MCM2− were measured on multiplex immunofluorescence images of the same tumours (71 Korean and 88 Scottish patients). All analyses were planned after the grading results were known and are exploratory.

| Analysis | Korea | Scotland |
|---|---|---|
| High-grade-like fraction, median by ISUP grade 1/2/3/4 | 0.012/0.047/0.336/0.539 | 0.027/0.103/0.313/0.363 |
| High-grade-like fraction vs AI grade, Spearman's ρ | 0.76 | 0.90 |
| High-grade-like fraction, ISUP 1–2: C; HR per s.d. | 0.772; 1.61 (1.35–1.94); P = 2.3 × 10⁻⁷ (32/237) | 0.648; 1.27 (0.83–1.94); P = 0.27 (14/86) |
| High-grade-like fraction, ISUP 2 only: C; HR per s.d. | 0.781; 1.74 (1.33–2.28); P = 6.4 × 10⁻⁵ (18/127) | 0.518; 0.94 (0.53–1.68); P = 0.83 (11/50) |
| AI grade, ISUP 2 only: C; HR per grade unit | 0.663; 2.78 (1.35–5.74); P = 0.0057 (18/127) | 0.561; 2.04 (0.37–11.13); P = 0.41 (11/50) |
| AI grade, ISUP 2 only: recurrences above vs below median | 13/63 vs 5/64; P = 0.043 | 7/25 vs 4/25; P = 0.39 |
| Spearman's ρ with nuclear area: AI grade; within ISUP grade; ISUP grade | 0.60; 0.42; 0.48 | 0.42; 0.21; 0.45 |
| Spearman's ρ with nuclear perimeter: AI grade; within ISUP grade; ISUP grade | 0.59; 0.41; 0.44 | 0.39; 0.20; 0.43 |
| Spearman's ρ with cell area: AI grade; within ISUP grade; ISUP grade | 0.60; 0.45; 0.47 | 0.48; 0.28; 0.47 |
| Spearman's ρ with nuclear roundness: AI grade; within ISUP grade; ISUP grade | −0.05; −0.05; 0.04 | −0.04; −0.08; −0.02 |
| AI grade vs P21+/MCM2− percentage, Spearman's ρ | 0.02 | −0.01 |
| Cox model with both markers: AI grade (per unit) | 1.06 (0.51–2.20); P = 0.87 | 6.36 (1.92–21.09); P = 0.0025 |
| Cox model with both markers: log(P21+/MCM2− % + 0.01) | 0.54 (0.36–0.82); P = 0.0036 | 0.87 (0.66–1.16); P = 0.35 |
| AI grade high and P21+/MCM2− < 2% vs all others: log-rank P | 0.0036 | 0.032 |

## Supplementary Table 3. Released database labels and distributions

Grades and outcomes were taken unchanged from the public clinical tables (S-BIAD2452, S-BIAD2454). The released ISUP grade was used as the reference label for training and grading evaluation. No patient was relabelled or excluded from grading. ISUP and Fuhrman were treated as separate grading variables, without conversion or reconciliation. The tables do not document the grading personnel or specimens, and the labels were not centrally re-reviewed for this study. One H&E slide per patient is public; a patient-level grade does not establish that its defining region is present in that slide.

**Supplementary Table 3a.** ISUP grade by Fuhrman grade, all patients.

| Cohort | ISUP grade | Fuhrman 1 | Fuhrman 2 | Fuhrman 3 | Fuhrman 4 |
|---|---|---|---|---|---|
| Korea | 1 | 2 | 70 | 34 | 5 |
| Korea | 2 | 0 | 54 | 71 | 4 |
| Korea | 3 | 0 | 2 | 38 | 4 |
| Korea | 4 | 0 | 0 | 8 | 8 |
| Scotland | 1 | 7 | 26 | 1 | 2 |
| Scotland | 2 | 2 | 31 | 12 | 5 |
| Scotland | 3 | 0 | 15 | 12 | 5 |
| Scotland | 4 | 0 | 0 | 0 | 2 |

**Supplementary Table 3b.** Immunofluorescence subsets: distributions from the released clinical tables, before excluding zero-time records from recurrence analyses.

| Item | Korea, released database | Scotland, released database |
|---|---|---|
| Patients | 71 | 88 |
| ISUP grade 1 / 2 / 3 / 4 | 32 / 26 / 12 / 1 | 29 / 39 / 19 / 1 |
| T stage 1 / 2 / 3–4 | 25 / 22 / 24 | 42 / 12 / 34 |
| Recurrences | 16 | 19 |
| Leibovich low / intermediate / high (Fuhrman version) | 0 / 64 / 7 | 35 / 37 / 16 |

**Supplementary Table 3c.** Recorded combinations retained without relabelling. Seven patients have ISUP grade 1 and Fuhrman grade 4 in the released tables: KR2, KR20, KR22, KR31 and KR285 (Korea), and 977 and 1045 (Scotland). These are observations across two grading systems, not adjudicated labelling errors. KR8 has a recorded ISUP grade of 2 and an ISUP-version Leibovich score of 7; both fields were retained as released. The latter score was used only in the alternative-baseline sensitivity analysis of the recurrence study and was not recalculated from the ISUP field. Grading used ISUP labels; the primary clinical prognostic baseline used the released Fuhrman-version Leibovich score.

## Supplementary Table 4. Cross-platform feature parity

One Scottish slide was extracted on both an NVIDIA A800 and an RTX A6000 with identical weights and patch coordinates. Acceptance required a cosine similarity of at least 0.999 for every patch; the result was PASS.

| Encoder | Patches | Minimum cosine similarity | Mean cosine similarity |
|---|---|---|---|
| Phikon-v2 | 7,291 | 0.999985 | 0.9999951 |
| UNI | 7,291 | 0.999999 | 0.9999999 |
| UNI2-h | 7,291 | 0.999937 | 0.9999999 |
| Virchow2 | 7,291 | 0.999978 | 0.9999987 |
| Prov-GigaPath | 7,291 | 0.999648 | 0.9999945 |
| CONCH | 7,291 | 1.000000 | 1.0000000 |
| ResNet50 | 7,291 | 1.000000 | 1.0000000 |
| ConvNeXt-T | 7,291 | 1.000000 | 1.0000000 |
| ViT-B/16 | 7,291 | 0.999999 | 0.9999998 |

## Supplementary Table 5. Proportional-hazards diagnostics for core models

Nominal score-test P values from scaled Schoenfeld residual diagnostics. AI/H&E refers to the model-derived score term. KM denotes the Kaplan–Meier time transformation; rank is the sensitivity transformation. Korea: 295 patients, 48 events; Scotland: 120 patients, 27 events. Models were refitted for diagnostics only; external predictive performance still uses the original frozen Korean predictors. All covariate tests for all 42 fits are in the accompanying CSV. No core-model term or global test was below 0.05 with either transformation; this does not prove proportionality.

| Cohort | Encoder | Model | KM: AI/H&E P | KM: global P | Rank: AI/H&E P | Rank: global P |
|---|---|---|---|---|---|---|
| Korea | phikon-v2 | AI + numeric ISUP | 0.882 | 0.751 | 0.858 | 0.725 |
| Korea | phikon-v2 | AI + Leibovich | 0.763 | 0.151 | 0.743 | 0.146 |
| Korea | phikon-v2 | AI + categorical ISUP | 0.906 | 0.861 | 0.882 | 0.849 |
| Korea | phikon-v2 | H&E + Leibovich | 0.419 | 0.192 | 0.420 | 0.182 |
| Korea | uni | H&E + Leibovich | 0.521 | 0.165 | 0.507 | 0.158 |
| Scotland | phikon-v2 | AI + numeric ISUP | 0.147 | 0.344 | 0.410 | 0.711 |
| Scotland | phikon-v2 | AI + Leibovich | 0.197 | 0.231 | 0.437 | 0.410 |
| Scotland | phikon-v2 | AI + categorical ISUP | 0.116 | 0.128 | 0.356 | 0.437 |
| Scotland | phikon-v2 | H&E + Leibovich | 0.460 | 0.292 | 0.823 | 0.377 |
| Scotland | uni | H&E + Leibovich | 0.131 | 0.189 | 0.301 | 0.398 |

## Supplementary Table 6. Nine-encoder association screen with post hoc Holm adjustment

AI grade adjusted for numeric ISUP in each cohort, using the original Breslow Cox specification. HR is per AI-grade unit; confidence intervals are nominal 95% intervals, not multiplicity-adjusted intervals. Holm adjustment treats the nine encoder association tests in each cohort as one family. The encoder family was defined for this revision sensitivity, after the original results were known. The Scottish CONCH fit had a PH diagnostic flag (Supplementary Note 5). These results do not support choosing a replacement primary encoder.

| Cohort | Encoder | HR (nominal 95% CI) | Raw P | Holm-adjusted P |
|---|---|---|---|---|
| Korea | conch | 6.18 (3.90–9.79) | 7.93e-15 | 7.14e-14 |
| Korea | convnext_tiny | 4.47 (2.59–7.71) | 7.94e-08 | 9.9e-08 |
| Korea | gigapath | 4.20 (2.57–6.87) | 1.06e-08 | 5.28e-08 |
| Korea | phikon-v2 | 4.20 (2.57–6.87) | 1.06e-08 | 5.28e-08 |
| Korea | resnet50 | 5.42 (3.07–9.57) | 5.93e-09 | 3.56e-08 |
| Korea | uni | 4.45 (2.71–7.32) | 3.91e-09 | 2.74e-08 |
| Korea | uni2-h | 4.20 (2.55–6.94) | 1.98e-08 | 5.95e-08 |
| Korea | virchow2 | 4.51 (2.74–7.42) | 3.24e-09 | 2.59e-08 |
| Korea | vit_b_16 | 5.26 (2.89–9.54) | 4.95e-08 | 9.9e-08 |
| Scotland | conch | 5.16 (2.35–11.31) | 4.19e-05 | 0.000377 |
| Scotland | convnext_tiny | 8.30 (1.48–46.56) | 0.016 | 0.110 |
| Scotland | gigapath | 2.56 (1.16–5.66) | 0.020 | 0.110 |
| Scotland | phikon-v2 | 3.26 (1.06–10.06) | 0.039 | 0.110 |
| Scotland | resnet50 | 19.83 (2.90–135.52) | 0.002 | 0.019 |
| Scotland | uni | 2.67 (0.91–7.88) | 0.075 | 0.110 |
| Scotland | uni2-h | 3.04 (1.23–7.48) | 0.016 | 0.110 |
| Scotland | virchow2 | 3.45 (1.19–9.99) | 0.023 | 0.110 |
| Scotland | vit_b_16 | 5.93 (1.17–30.15) | 0.032 | 0.110 |

## Supplementary Figures

![Supplementary Fig. 1](SuppFig1.png)

**Supplementary Fig. 1 | Patient flow.** Each cohort provided one H&E slide per patient. Patches came from a technical quality-control pool. Korean features were extracted on NVIDIA A800 GPUs; Scottish features on A800 (95 slides) and RTX A6000 (25 slides) GPUs, with cross-platform parity checked (Supplementary Fig. 10 and Supplementary Table 4). Five Korean patients with a disease-free interval of zero were excluded from recurrence analyses. Immunofluorescence summaries were available for 71 Korean and 88 Scottish patients. Recurrence was censored at 84 months.

![Supplementary Fig. 2](SuppFig2.png)

**Supplementary Fig. 2 | Receiver operating characteristic curves for all nine encoders.** ISUP grade 3–4 versus 1–2. **a)** Korean out-of-fold predictions (mean of five seeds). **b)** Scottish external predictions (mean of 25 models). Solid lines, pathology foundation models; dashed lines, ImageNet encoders. AUCs are given in the legends.

![Supplementary Fig. 3](SuppFig3.png)

**Supplementary Fig. 3 | Stability across random seeds.** Scottish external performance of each encoder: **a)** high-versus-low AUC, **b)** grade 1 versus 2 AUC and **c)** Harrell's C of the AI grade for recurrence. Dots show the five seeds, each the mean of its five fold models; bars show the 25-model ensemble used in the main analysis.

![Supplementary Fig. 4](SuppFig4.png)

**Supplementary Fig. 4 | AI grade by ISUP grade.** Phikon-v2 AI grade by pathologist ISUP grade in **a)** Korea (out-of-fold) and **b)** Scotland (external). Boxes show the median and interquartile range and whiskers extend to 1.5 times the interquartile range; dots are individual patients.

![Supplementary Fig. 5](SuppFig5.png)

**Supplementary Fig. 5 | Recurrence-free survival by AI-grade tertile and by ISUP grade.** **a)**, **b)** Patients split into tertiles of the Phikon-v2 AI grade, with cut-points from each cohort's own distribution, in Korea (295 patients) and Scotland (120 patients). **c)**, **d)** The same patients by pathologist ISUP grade, with grades 3 and 4 combined. Legends give recurrences/patients; ticks mark censored patients; numbers at risk are shown below each plot. The y axes start at 40%. Descriptive curves, without tests.

![Supplementary Fig. 6](SuppFig6.png)

**Supplementary Fig. 6 | Prognostic value of the AI grade by encoder.** **a)** Harrell's C of each encoder's AI grade for recurrence in Korea (out-of-fold) and Scotland (external); dashed lines mark the C of ISUP grade (0.590 and 0.686). **b)** Hazard ratio per AI-grade unit in Cox models adjusted for ISUP grade, with 95% Wald confidence intervals, on a log scale. The Scottish interval included 1 only for UNI (HR 2.67, 0.91–7.88). Intervals were widest for the ImageNet encoders (ResNet50: HR 19.8, 2.9–135.5). Confidence intervals are nominal; post hoc nine-encoder Holm-adjusted P values are reported in Supplementary Table 6.

![Supplementary Fig. 7](SuppFig7.png)

**Supplementary Fig. 7 | Patch-level maps of two Korean tumours.** Layout as in Fig. 4. **a)** ISUP grade 2 tumour with recurrence at 27 months and an AI grade of 3.33. **b)** ISUP grade 1 tumour without recurrence over 84 months and an AI grade of 1.17. For each tumour: per-patch P(grade 3) + P(grade 4), relative attention on a log₁₀ scale, and the four patches with the largest attention × (P(grade 3) + P(grade 4)), at least about 1 mm apart; frames are red when the patch's P(grade 3) + P(grade 4) exceeds 0.5. Examples were chosen by the rule used for Fig. 4, applied to the Korean slides.

![Supplementary Fig. 8](SuppFig8.png)

**Supplementary Fig. 8 | Pre-specified secondary analyses of the recurrence model in the other encoder and cohort combinations.** Rows: **a)** Korea, Phikon-v2; **b)** Korea, UNI; **c)** Scotland, UNI (Scotland with Phikon-v2 is shown in Fig. 7d–f). Left, calibration of predicted recurrence by 36 months: Kaplan–Meier observed risk with 95% CI in five risk groups in Korea and three in Scotland, with calibration slope and observed-to-expected ratio; the two models are offset slightly sideways for legibility. Middle, decision curves at 36 months, for threshold probabilities up to 40%. Right, Uno's C at 60 months and time-dependent AUC at 36 and 60 months for the Leibovich score alone (grey) and with the H&E risk score (red), with the difference and its 95% CI. Korean values are cross-fitted; Scottish values use the Korean coefficients unchanged.

![Supplementary Fig. 9](SuppFig9.png)

**Supplementary Fig. 9 | Additional studies that did not improve the grading model.** **a)** Auxiliary Fuhrman supervision. Grade 1 versus 2 AUC (Korean out-of-fold, 240 ISUP grade 1–2 patients) for arm A (ISUP only), arm B (ISUP plus a Fuhrman 1–2 versus 3–4 auxiliary target) and arm C (ISUP plus Fuhrman labels shuffled within ISUP grade, a control), with Phikon-v2 and UNI. **b)** Patch magnification pilot in 120 Korean cases (60 grade 1, 60 grade 2). Grade 1 versus 2 AUC with features from the standard 128 µm patch (L); from four 64 µm sub-patches decoded at native resolution (H); from the same four quadrants cut from the standard patch and upsampled (Hd, a control for geometry); and from standard features sampled over all patches of the slide (Lfull, a reference for sampling loss). Dots in a and b are seeds (five and three) and bars their ensemble. **c)** Fusion of Phikon-v2, UNI, Prov-GigaPath and UNI2-h. Difference in high-versus-low AUC (Korean out-of-fold, 300 patients) against Phikon-v2, with paired bootstrap 95% CIs. P1, four encoders with one seed each against Phikon-v2 with four seeds; P2, late fusion of 16 models against four Phikon-v2 seeds; P3, learned fusion with encoder dropout against Phikon-v2. Panel c reports exploratory comparisons; earlier Korean predictions had informed encoder selection (Supplementary Note 3).

![Supplementary Fig. 10](SuppFig10.png)

**Supplementary Fig. 10 | Cross-platform feature parity.** One Scottish slide (7,291 patches) was extracted on both an NVIDIA A800 and an RTX A6000. The plot shows 1 − cosine similarity between the two feature vectors of each patch, by encoder, on a log scale. The dashed line marks the acceptance limit (cosine similarity ≥ 0.999). Values below 10⁻⁹, including identical vectors, are drawn at 10⁻⁹. The lowest cosine similarity across all encoders was 0.99965.
