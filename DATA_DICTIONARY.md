# Main prediction table fields

All patient IDs refer to released study records. A row is one patient/encoder/cohort prediction, not one image patch.

| Field | Meaning |
|---|---|
| `encoder`, `cohort`, `patient_id` | Unique ensemble key; cohort is `korea` or `scotland` |
| `korean_test_fold` | Korean outer held-out fold; blank for Scotland |
| `isup` | Released ISUP grade 1–4, the main grading reference |
| `fuhrman` | Separately released Fuhrman grade; not a substitute for ISUP |
| `leibovich_fuhrman` | Released Fuhrman-based Leibovich score |
| `in_recurrence_set` | 1 if included in recurrence analysis; all patients remain in grading |
| `time_months_censored84` | Follow-up/recurrence time after administrative censoring at 84 months |
| `recurrence` | Event indicator after that censoring rule; not overall mortality |
| `p1`–`p4` | Mean predicted probabilities for grades 1–4, rounded to six decimals |
| `ai_grade` | Expected grade, sum of k × P(k), 1–4 |

Per-run grading records additionally identify seed and fold. Korean cases have five seed records at their held-out fold; Scottish cases have all 25 seed/fold records per encoder. Split roles `train` and `val` were pooled for fixed-epoch formal training; `test` remained excluded. In direct-recurrence exports, `he_risk_standardised` is the stored standardised model risk, not the continuous AI grade.
