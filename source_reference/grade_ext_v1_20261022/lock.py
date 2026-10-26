"""Lock the grading external-validation code BEFORE any formal result exists (PROTOCOL.md).
  python lock.py write    refuse if any formal training output exists (runs/, RESULTS.*); write LOCK.json with the sha256 of
                          every code, protocol, campaign and input file and the code signature
  python lock.py verify   exit 0 if every locked file still has its sha256 (train.py and evaluate.py check the same)
Changing anything after the lock means a new version directory, recorded before any formal result is seen."""
import sys
from common import *
LOCKED=['PROTOCOL.md','config.json','common.py','train.py','check.py','evaluate.py','make_campaign.py','lock.py','CAMPAIGN.json',
        'inputs/korea_surv.csv','inputs/scotland_surv.csv','inputs/labels_provenance.json','tests/test_local.py']
def files():return list(LOCKED)
def main(argv):
    cmd=argv[0] if argv else 'verify'
    if cmd=='verify':
        ok,bad=lock_status();print('LOCK','OK' if ok else 'BROKEN',bad);return 0 if ok else 1
    if cmd!='write':print(__doc__);return 2
    if LOCK_JSON.exists():print('LOCK.json exists already; a change needs a new version directory');return 1
    outputs=[str(p) for p in RUNS.rglob('complete.json')]+[str(p) for p in RUNS.rglob('predictions.json')]+\
        [n for n in ('RESULTS.json','RESULTS.md') if (HERE/n).exists()]
    if outputs:print('REFUSED: formal training outputs exist already:',outputs[:5]);return 1
    cfg=config();cases,assign=load_korea(cfg);label_structure(cases,assign,load_scotland(cfg),cfg)
    rec=dict(state='LOCKED',locked_at=now(),code_signature=signature(),files={n:sha(HERE/n) for n in files()},
             labels_sha256=cfg['labels_sha256'],
             evidence=dict(training_outputs_existing=[],results_existing=False,
                           check_attempts=sorted(p.name for p in (HERE/'check_runs').glob('attempt_*'))),
             statement='Written before any formal training run or analysis of the grading external-validation study.')
    save_json(LOCK_JSON,rec);print('LOCKED',rec['code_signature'],len(rec['files']),'files')
    return 0
if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
