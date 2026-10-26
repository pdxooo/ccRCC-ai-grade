"""Lock PLAN.md and the code of this directory before any post hoc number is computed: writes LOCK.json (SHA-256 of
each file, time) and makes the files read-only. posthoc.py refuses to run if any locked file changes."""
import hashlib,json,datetime,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;FILES=['PLAN.md','export_predictions.py','posthoc.py','verify_from_predictions.py','lock.py','run.sh']
assert not (HERE/'LOCK.json').exists(),'already locked';assert not (HERE/'outputs'/'POSTHOC_RESULTS.json').exists(),'results already exist'
lk=dict(locked_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),files={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in FILES},
        depends_on=dict(gx1_LOCK_json=hashlib.sha256((HERE.parent/'grade_ext_v1_20261022'/'LOCK.json').read_bytes()).hexdigest(),
                        surv1_LOCK_json=hashlib.sha256((HERE.parent/'surv_he_clin_v1_20261022'/'LOCK.json').read_bytes()).hexdigest()))
(HERE/'LOCK.json').write_text(json.dumps(lk,indent=1))
for n in FILES+['LOCK.json']:os.chmod(HERE/n,0o444)
print(json.dumps(lk,indent=1))
