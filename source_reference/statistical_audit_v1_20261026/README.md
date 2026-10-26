# Statistical reporting audit

Run from any path supported by the local R installation:

`Rscript --vanilla statistical_audit.R PATH_TO_THIS_FOLDER`

Dependencies: R 4.5.2, survival 3.8-3. Relative inputs: `../gx1_posthoc_v1_20261025/outputs/grading_predictions.csv` and `recurrence_he_risk.csv`. The Windows run used an ASCII staging path because the native R locale could not open the long Unicode workspace path; input byte hashes were preserved. This is an execution-path workaround, not a data transformation.

Outputs contain all 42 model coefficients; every term/global diagnostic under KM and rank time transforms; raw and Holm-adjusted association P values; and core diagnostic plots. No core PH flag was found, so the conditional core time-interaction sensitivity did not run. The non-core Scottish CONCH AI/ISUP model has a KM diagnostic flag and is disclosed.

`independent_validation.json` checks all 36 AI/numeric-ISUP or AI/Leibovich coefficient sets against archived estimates and independently validates Holm adjustment with NumPy. The plots/checks do not cover every exploratory model. New results are post hoc and original P values elsewhere remain nominal unless explicitly identified as adjusted. Plan and data hashes are provenance aids, not public preregistration.
