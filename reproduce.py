"""Recompute the main reported results on CPU; no model training or slide downloads."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRED = ROOT / 'gx1_posthoc_v1_20261025' / 'outputs'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduction_output')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(out != PRED.resolve(), 'Use a separate output directory; do not overwrite the source predictions.')
    inputs = json.loads((ROOT / 'reproduction_inputs.sha256.json').read_text('utf-8'))
    for name, expected in inputs.items():
        path = ROOT / name
        require(path.is_file(), f'Missing input: {name}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Input hash mismatch: {name}')

    cfg = json.loads((ROOT / 'source_reference/grade_ext_v1_20261022/config.json').read_text())
    encoders = set(cfg['encoders'])
    ensemble = rows(PRED / 'grading_predictions.csv')
    per_run = rows(PRED / 'grading_predictions_per_run.csv')
    require(len(ensemble) == 3780 and len(per_run) == 40500, 'Unexpected grading row counts.')
    grouped = defaultdict(list)
    for r in ensemble:
        grouped[r['encoder'], r['cohort']].append(r)
    require(set(grouped) == {(e, c) for e in encoders for c in ('korea', 'scotland')}, 'Missing encoder/cohort.')
    for (encoder, cohort), rs in grouped.items():
        n = 300 if cohort == 'korea' else 120
        require(len(rs) == len({r['patient_id'] for r in rs}) == n, f'Invalid cases: {encoder}/{cohort}')
        expected_grades = {1:111, 2:129, 3:44, 4:16} if cohort == 'korea' else {1:36, 2:50, 3:32, 4:2}
        require(dict(Counter(int(r['isup']) for r in rs)) == expected_grades, 'Grade distribution mismatch.')
        P = np.array([[float(r[f'p{k}']) for k in range(1,5)] for r in rs])
        require(np.isfinite(P).all() and ((P >= 0) & (P <= 1)).all(), 'Invalid probabilities.')
        require(np.max(np.abs(P.sum(1)-1)) < 3e-6, 'Probabilities do not sum to one.')
        require(np.max(np.abs(P @ np.arange(1,5)-[float(r['ai_grade']) for r in rs])) < 8e-6, 'Invalid expected grades.')

    # The archived verifier writes a report but does not set a failure exit code.
    # Run it on copies and explicitly enforce its report status here.
    out.mkdir(parents=True, exist_ok=True)
    for name in ('grading_predictions.csv', 'grading_predictions_per_run.csv', 'korea_split.csv', 'recurrence_he_risk.csv'):
        shutil.copy2(PRED / name, out / name)
    verifier = ROOT / 'gx1_posthoc_v1_20261025/verify_from_predictions.py'
    result = subprocess.run([sys.executable, str(verifier), '--outputs', str(out)], capture_output=True, text=True)
    (out / 'verification_stdout.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    require(result.returncode == 0, 'Verifier crashed; see verification_stdout.txt.')
    report = (out / 'VERIFY_REPORT.md').read_text()
    match = re.search(r'Overall: PASS \((\d+) passed, 0 failed\)', report)
    require(match is not None and int(match[1]) == 93 and '| FAIL' not in report, 'Metric verification failed; see VERIFY_REPORT.md.')

    spec = importlib.util.spec_from_file_location('metric_verifier', verifier)
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    table = []
    raw_p = {}
    for (encoder, cohort), rs in sorted(grouped.items()):
        I = np.array([int(r['isup']) for r in rs])
        P = np.array([[float(r[f'p{k}']) for k in range(1,5)] for r in rs])
        table.append(dict(encoder=encoder, cohort=cohort, n=len(rs),
                          highlow_auc=V.auc(I >= 3, P[:,2:].sum(1)),
                          four_class_accuracy=float(np.mean(P.argmax(1)+1 == I))))
        surv = [r for r in rs if r['in_recurrence_set'] == '1']
        X = np.array([[float(r['isup']), float(r['ai_grade'])] for r in surv])
        t = np.array([float(r['time_months_censored84']) for r in surv])
        e = np.array([int(r['recurrence']) for r in surv])
        b, se = V.cox(X, t, e)
        raw_p[encoder, cohort] = math.erfc(abs(b[-1]/se[-1])/math.sqrt(2))
    adjusted = {}
    for cohort in ('korea', 'scotland'):
        ordered = sorted(encoders, key=lambda enc: raw_p[enc, cohort])
        running = 0.0
        for i, enc in enumerate(ordered):
            running = max(running, (len(ordered)-i)*raw_p[enc, cohort])
            adjusted[enc, cohort] = min(1.0, running)
    with (out / 'main_grading_results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    # Frozen Korean Cox models: ISUP categories 1, 2, and 3-4 as in posthoc.py.
    arrays = {}
    for cohort in ('korea', 'scotland'):
        rs = [r for r in grouped['phikon-v2', cohort] if r['in_recurrence_set'] == '1']
        I = np.array([int(r['isup']) for r in rs])
        g = np.array([float(r['ai_grade']) for r in rs])
        t = np.array([float(r['time_months_censored84']) for r in rs])
        e = np.array([int(r['recurrence']) for r in rs])
        arrays[cohort] = (np.c_[I == 2, I >= 3].astype(float), g, t, e)
    Xk, gk, tk, ek = arrays['korea']; Xs, gs, ts, es = arrays['scotland']
    b0, _ = V.cox(Xk, tk, ek); b1, _ = V.cox(np.c_[Xk, gk], tk, ek)
    c0 = V.cindex(ts, es, Xs @ b0); c1 = V.cindex(ts, es, np.c_[Xs, gs] @ b1)
    ref = json.loads((PRED / 'POSTHOC_RESULTS.json').read_text())['A_incremental_discrimination']['categorical']['scotland_frozen_korean_models']
    for name, value in [('c_isup', c0), ('c_isup_plus_ai', c1), ('delta_c', c1-c0)]:
        require(abs(value-ref[name]) < 1e-3, f'Frozen model mismatch: {name}')
    external = [r for r in table if r['cohort'] == 'scotland']
    headline = dict(
        external_phikon_auc=next(r['highlow_auc'] for r in external if r['encoder']=='phikon-v2'),
        external_pathology_mean_auc=float(np.mean([r['highlow_auc'] for r in external if cfg['encoder_family'][r['encoder']]=='pathology'])),
        external_imagenet_mean_auc=float(np.mean([r['highlow_auc'] for r in external if cfg['encoder_family'][r['encoder']]=='imagenet'])),
        external_phikon_isup_adjusted_p=raw_p['phikon-v2','scotland'],
        external_phikon_isup_adjusted_p_holm=adjusted['phikon-v2','scotland'],
        external_frozen_c_isup=c0, external_frozen_c_isup_plus_ai=c1, external_frozen_delta_c=c1-c0,
        note='Bootstrap confidence intervals and model training were not rerun. Holm covers nine encoders within each cohort.')
    (out / 'MAIN_RESULTS.json').write_text(json.dumps(headline, indent=2), encoding='utf-8')
    receipt = dict(status='PASS', archived_verifier_checks=93, frozen_model_checks=3,
                   hashed_inputs=len(inputs), python=platform.python_version(), numpy=np.__version__,
                   platform=platform.platform(), operation='CPU recomputation from exported predictions; no training')
    (out / 'RUN_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f'FAILED: {error}', file=sys.stderr)
        raise SystemExit(1)
