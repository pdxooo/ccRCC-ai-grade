"""Lock the stage-1 training and analysis code BEFORE any stage-1 result exists (PROTOCOL.md; protocol section 11).
  python lock.py write    refuse if any stage-1 training output exists (runs/, RESULTS.*); write LOCK.json with the sha256 of
                          every code, protocol and input file, the code signature, the locked case list and the state of the
                          stage-1 extraction at that moment (evidence that no result could have been seen)
  python lock.py verify   exit 0 if every locked file still has its sha256 (train.py and evaluate.py check the same)
Changing anything after the lock means a new version directory, recorded before any stage-1 result is seen."""
import sys,glob
from common import *
LOCKED=['PROTOCOL.md','config.json','common.py','train.py','check76.py','evaluate.py','make_prior_errors.py','lock.py',
        'inputs/prior_all_wrong_isup2_v1.json','inputs/hpc_sha256.txt','tests/test_local.py']
def files():
    return LOCKED+sorted(str(p.relative_to(HERE)) for p in (HERE/'inputs/mil_first_round_v2_test_predictions').rglob('*.json'))
def main(argv):
    cmd=argv[0] if argv else 'verify'
    if cmd=='verify':
        ok,bad=lock_status();print('LOCK','OK' if ok else 'BROKEN',bad);return 0 if ok else 1
    if cmd!='write':print(__doc__);return 2
    if LOCK_JSON.exists():print('LOCK.json exists already; a change needs a new version directory');return 1
    outputs=[str(p) for p in RUNS.rglob('complete.json')]+[str(p) for p in RUNS.rglob('predictions.json')]+\
        [n for n in ('RESULTS.json','RESULTS.md') if (HERE/n).exists()]
    if outputs:print('REFUSED: stage-1 training outputs exist already:',outputs[:5]);return 1
    cfg=config();cases,_=load_cases(cfg)
    extracted=sorted(c['stem'] for c in cases if (HR/c['stem']/'complete.json').exists())
    rec=dict(state='LOCKED',locked_at=now(),code_signature=signature(),files={n:sha(HERE/n) for n in files()},
             stage1_list_sha256=cfg['stage1_list_sha256'],
             evidence=dict(training_outputs_existing=[],results_existing=False,stage1_cases_extracted=len(extracted),
                           stage1_cases_extracted_list=extracted,check76_attempts=sorted(p.name for p in (HERE/'check76_runs').glob('attempt_*')),
                           release_json_present=(HR/'RELEASE.json').exists()),
             statement='Written before any stage-1 training run or analysis; the extraction outputs are features, not results.')
    save_json(LOCK_JSON,rec);print('LOCKED',rec['code_signature'],len(rec['files']),'files; stage-1 cases extracted so far:',len(extracted))
    return 0
if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
