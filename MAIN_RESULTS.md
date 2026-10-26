# Principal results and their recomputation

| Main finding | Recomputed output | Interpretation |
|---|---|---|
| Phikon-v2 external high/low-grade AUC about 0.841 | `MAIN_RESULTS.json`: `external_phikon_auc` | Discrimination against Scottish database ISUP labels |
| Pathology-encoder mean external AUC about 0.856; ImageNet mean about 0.761 | Same file: family mean AUC fields; individual values in `main_grading_results.csv` | Descriptive means of six versus three encoders, not a randomised family-level significance claim |
| Scottish AI-grade association after numeric ISUP adjustment: nominal P about 0.039, Holm P about 0.110 | Same file: raw and adjusted P; HR/CI checks in `VERIFY_REPORT.md` | Does not survive the post hoc nine-encoder adjustment |
| Korean-fitted ISUP+AI model does not improve Scottish C | Same file: C about 0.623 versus 0.677; delta about −0.054 | No demonstrated external incremental prognostic utility; categorical ISUP uses 1, 2, 3–4 |

All encoder results remain available. The guide focuses on the main findings without selectively omitting the negative prognostic result. Bootstrap intervals, morphology analyses, high-resolution pilots and exploratory maps are not rerun by the main command. They must not be labelled newly reproduced by this release.
