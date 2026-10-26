# Post hoc analyses and per-case exports for the grading study (gx1)

Written 2026-10-25, after all results of the grading study (`../grade_ext_v1_20261022`, locked 2026-10-22 17:42:29)
and the recurrence study (`../surv_he_clin_v1_20261022`) were known, in response to a critical review of the draft.
These analyses are **post hoc**. This plan, and the code that carries it out, were locked (`LOCK.json`, SHA-256)
before any of the numbers below was computed. Every analysis listed here is reported in full, whatever its result.
Nothing is trained, no label is changed, no case is removed, and the Scottish cohort is not used to choose any model,
threshold or method.

Primary encoder: Phikon-v2. Recurrence set: the locked survival sets of gx1 (Korea 295 patients, 48 recurrences;
Scotland 120, 27), 84-month censoring. Predictions: the gx1 ensembles (Korea: out-of-fold mean of five seeds;
Scotland: mean of 25 models), read through gx1's own gates (`evaluate.collect`). Statistics use gx1's own functions
(`common.cox_fit`, `common.harrell_c`, `common.logrank`, `evaluate.boot`; 2,000 case-bootstrap replicates with
models held fixed, seed 20261026).

## A. Incremental discrimination of the AI grade over ISUP grade

The draft's ΔC compared the AI grade alone with ISUP grade alone. The question "does adding the AI grade to ISUP
grade improve discrimination?" is answered here by comparing a Cox model with ISUP grade against a Cox model with ISUP
grade plus the AI grade (per grade unit).

- **ISUP coding.** Two codings, both reported:
  - categorical (the primary coding here): indicators for ISUP 2 and ISUP 3–4 against ISUP 1. Grades 3 and 4 are merged because Scotland has two grade 4 tumours;
  - numeric 1–4, as in the locked study.
- **A1. External, frozen.** Both models are fitted on the Korean recurrence set. Their linear predictors are applied
  unchanged to Scotland. Report Scottish Harrell's C of each model and ΔC = C(ISUP + AI) − C(ISUP), with a bootstrap
  95% CI.
- **A2. Korean, cross-fitted.** For each locked fold, both models are fitted on the recurrence-set patients outside the
  fold's test cases and applied to its test cases. Report the pooled out-of-fold C of each model and ΔC with a bootstrap
  95% CI.

## B. Hazard ratio of the AI grade with categorical ISUP adjustment

Within each cohort, fit Cox(ISUP 2, ISUP 3–4, AI grade). Report the AI-grade HR per grade unit (Wald 95% CI, P) and
the likelihood-ratio test against Cox(ISUP 2, ISUP 3–4). The numeric-ISUP results of the locked study are listed
alongside for comparison.

## C. Low-grade split at the Korean threshold

The locked study split Scottish ISUP 1–2 tumours at the Scottish median AI grade. Here the threshold is the median AI
grade of the Korean ISUP 1–2 recurrence-set patients (the threshold of the locked Korean low-grade analysis), applied
unchanged to the Scottish ISUP 1–2 patients. Report recurrences/patients above and at or below the threshold, and the
log-rank P.

## D. Per-case exports (for independent recomputation)

`export_predictions.py` writes the following, through each study's own gates (nothing is refitted):
- per-run and ensemble ISUP probabilities, the AI grade, labels and outcomes for all nine encoders;
- the locked Korean five-fold split;
- a manifest of the 225 grading runs and their receipts;
- the per-case standardised H&E risk of the recurrence study.

`verify_from_predictions.py` recomputes the main reported numbers from these files alone. It uses its own
implementations of the statistics and imports no study code. It writes a PASS/FAIL report against the locked
RESULTS.json files.
