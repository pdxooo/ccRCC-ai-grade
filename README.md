# Continuous AI grading of clear cell renal cell carcinoma

Code and derived results for **External validation of a continuous AI grade for clear cell renal cell carcinoma**. This repository accompanies a manuscript prepared for submission; it does not imply journal acceptance.

## Reproduce the main results

Use Python 3.10 or later on CPU. No GPU, raw slide download or model weights are needed for result recomputation.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py
```

Expected: **93 archived-verifier checks and three frozen-model C-index checks pass**. The entry point verifies input hashes and patient/probability structure, writes a nine-encoder grading table, and recalculates the main external-validation results, the nine-encoder Holm adjustment, and the Korean-fitted combined model's Scottish C-index. Errors produce a non-zero exit code. Outputs are in `reproduction_output/`.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for inputs, outputs, training settings, source locations and limits. [MAIN_RESULTS.md](MAIN_RESULTS.md) maps the principal claims to reproducible quantities. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines the exported fields.

## Study

Korea: 300 patients for grading; 295 for recurrence after excluding five zero-time records. Scotland: 120 patients. ISUP is the released database label; no central pathology re-review is claimed. Nine frozen encoders are compared under the same attention-based MIL protocol. The primary encoder is Phikon-v2. Six encoders are pathology models and three are ImageNet models.

The main findings are external high/low-grade discrimination and the limitations of prognostic transfer. No external incremental prognostic benefit was established. Patch probabilities are exploratory maps, not pathologist-verified local grades or additive contributions to expected slide grade.

## Data and source files

- `gx1_posthoc_v1_20261025/outputs/`: ensemble and per-run predictions, locked Korean folds, derived run summaries, recurrence scores and post hoc reference results.
- `grade_ext_v1_20261022/RESULTS.json`, `surv_he_clin_v1_20261022/RESULTS.json`: reference results for numerical verification.
- `qc_provenance/`: patient-level QC counts and source hashes, with local path columns removed.
- `source_reference/`: path-redacted extraction, training, analysis and figure source. Historical lock records do not validate these adapted files. Read its README before attempting new training.

Original images and clinical tables: [Korea S-BIAD2452](https://www.ebi.ac.uk/bioimage-archive/studies/S-BIAD2452/) and [Scotland S-BIAD2454](https://www.ebi.ac.uk/bioimage-archive/studies/S-BIAD2454/). Both archived study metadata records specify CC0; see `DATA_SOURCES.json`. Public study identifiers are retained to permit record matching.

## Access and scope

The repository is publicly accessible. Publication of the files does not itself grant an open-source licence; no licence for original code has been selected. Third-party software, data and model weights retain their own terms. Pretrained weights, raw WSIs, personal credentials and internal correspondence are not included.

Result recomputation is tested. Full retraining from the original images is not: complete retained coordinates, cached features, trained checkpoints and raw historical run receipts are not included. Training-source Python files were syntax-checked, not rerun end to end. The guide distinguishes exported evidence from missing training artefacts; no missing material is represented as present.

OpenAI Codex assisted manuscript editing, code review and statistical verification; the authors remain responsible for the research record.
