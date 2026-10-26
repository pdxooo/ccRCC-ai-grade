"""Shared paths, locked inputs and checks for the high-res stage-1 training and analysis (see PROTOCOL.md).
No torch import here (evaluate.py and the label checks run without it)."""
import os,sys,json,csv,hashlib,datetime
# Forced, as in fusion_gain_v2: an inherited value must not silently change the cuBLAS workspace between legs.
INHERITED_CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'  # CUDA indices == nvidia-smi indices
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_k,'4')
from pathlib import Path
# HE_PATH_PREFIX (lab-a6000: P/hpcroot) is prepended to every HPC absolute path; unset -> the HPC paths themselves.
HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
PREFIX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')
assert PREFIX=='' or PREFIX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
ROOT=Path(PREFIX+HPC_ROOT)
sys.path.insert(0,str(ROOT/'qc_deps'))
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('HE_PROJECT_DIR','/data/workstation/he_ccrcc_isup_20261021'))
TRANSFER=PROJECT/'transfer'
LOCK_FILE=TRANSFER/'lists/STAGE1_CASES_LOCK.json'   # the 120 locked cases (fold, ISUP grade, slide stem)
CASE_LIST=TRANSFER/'lists/stage1_cases.txt'          # transfer order; defines the 7.6 check cases
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
SPLITS=ROOT/'bib_phase1/first_round_v1_20261010'
FULL=ROOT/'features/nine_encoder_full_v1_20261010'            # whole-slide L features (reference arm Lfull)
HR=ROOT/'features/highres_pilot_v1_20261021_stage1_a6000'     # stage-1 L / H / Hd features (QUARANTINE.json, RELEASE.json)
STAGE0=ROOT/'features/highres_pilot_v1_20261021_a6000'
PRIOR=HERE/'inputs/prior_all_wrong_isup2_v1.json'
RUNS=HERE/'runs'
CHECK_PASS=HERE/'CHECK76_PASS.json'
LOCK_JSON=HERE/'LOCK.json'
SIGNATURE_FILES=['common.py','train.py','check76.py','evaluate.py','config.json','inputs/prior_all_wrong_isup2_v1.json']
HR_ARMS=('L','H','Hd')
# the release conditions of highres_a6000/stage1_extract.py QUARANTINE.json, pinned: an edited QUARANTINE.json cannot drop one
RELEASE_CONDITIONS=('stage0_automated_checks_passed_on_a6000','stage0_visual_review_signed_off','stage0_platform_parity_passed',
                    'stage0_training_check_passed')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp.%d'%os.getpid());tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default
def config():return json.loads((HERE/'config.json').read_text())
def signature():return hashlib.sha256(json.dumps({n:sha(HERE/n) for n in SIGNATURE_FILES},sort_keys=True).encode()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:  # Windows exports may carry a BOM
        return list(csv.DictReader(stream))
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def model_seed(cfg,seed_index,fold):return cfg['seed_base']+100*seed_index+fold
def sample_locations(n,k,cfg,seed_index,fold,epoch,patient):
    """Training locations of one case in one epoch: k of n without replacement, sorted (all if n <= k). Depends only on
    (seed, fold, epoch, case), so L, H, Hd and both encoders see the same locations."""
    if n<=k:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,patient)).choice(n,k,replace=False))
def train_order(n,cfg,seed_index,fold,epoch):
    return np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,'order')).permutation(n)

# ------------------------------------------------------------------------------------------------ locked inputs
def load_inputs():
    """Manifest and five-fold table, validated exactly as in fusion_gain_v2 / mil_first_round_v2."""
    rows=read_csv(POOL/'patient_manifest.csv')
    lock=json.loads((POOL/'LOCK.json').read_text());sl=json.loads((SPLITS/'SPLITS_LOCK.json').read_text())
    assert len(rows)==300 and len({r['patient_id'] for r in rows})==300
    assert sha(POOL/'patient_manifest.csv')==lock['patient_manifest_sha256']==sl['patient_manifest_sha256']
    assert sha(POOL/'LOCK.json')==sl['qc_lock_sha256'] and sha(SPLITS/'patient_splits.csv')==sl['split_sha256']
    lookup={r['patient_id']:r for r in rows};assign=read_csv(SPLITS/'patient_splits.csv')
    assert len(assign)==1500
    for fold in range(5):
        a=[r for r in assign if int(r['fold'])==fold]
        assert len(a)==300 and len({r['patient_id'] for r in a})==300 and sum(r['role']=='test' for r in a)==60
        for r in a:
            assert r['role'] in ('train','val','test')
            for key in ('slide_filename','ISUP_grade'):assert r[key]==lookup[r['patient_id']][key]
    tests=[r['patient_id'] for r in assign if r['role']=='test'];assert len(tests)==len(set(tests))==300
    return lookup,assign

def load_cases(cfg):
    """The 120 locked cases with labels y = ISUP - 1 (G1 -> 0, G2 -> 1); the lock, the manifest and the five-fold table must
    agree on every case's grade, fold and slide. Returns (cases sorted by patient_id, transfer-order slide stems)."""
    lookup,assign=load_inputs()
    lock=json.loads(LOCK_FILE.read_text());raw=lock['cases']
    assert hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()==lock['list_sha256']==cfg['stage1_list_sha256'],'stage-1 case list is not the locked one'
    assert lock['manifest_sha256']==sha(POOL/'patient_manifest.csv') and lock['splits_sha256']==sha(SPLITS/'patient_splits.csv')
    test_fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'}
    cases=[]
    for c in raw:
        row=lookup[c['patient_id']];g=int(row['ISUP_grade'])
        assert g==int(c['ISUP_grade']) and str(g) in cfg['label_mapping'],f"{c['patient_id']}: grade {g} vs lock {c['ISUP_grade']}"
        assert test_fold[c['patient_id']]==int(c['fold']),f"{c['patient_id']}: test fold differs from the five-fold table"
        assert Path(row['slide_filename']).stem==c['slide_stem']
        cases.append(dict(patient_id=c['patient_id'],fold=int(c['fold']),grade=g,y=int(cfg['label_mapping'][str(g)]),
                          stem=c['slide_stem'],primary_count=int(row['primary_count'])))
    cases.sort(key=lambda c:c['patient_id'])
    assert len(cases)==120 and len({c['patient_id'] for c in cases})==120
    for f in cfg['folds']:
        for y in (0,1):assert sum(1 for c in cases if c['fold']==f and c['y']==y)==12,f'fold {f} class {y}'
    order=CASE_LIST.read_text().split()
    assert len(order)==120 and sorted(order)==sorted(c['stem'] for c in cases),'stage1_cases.txt != the locked cases'
    return cases,order

def roles(cases,fold):
    return dict(train=[c for c in cases if c['fold']!=fold],test=[c for c in cases if c['fold']==fold])

def check_cases(cases,order,cfg):
    """Protocol 7.6 cases, fixed in advance: the first k ISUP-1 and the first k ISUP-2 cases in stage1_cases.txt order."""
    by={c['stem']:c for c in cases};k=cfg['check76']['cases_per_grade']
    pick=[by[s] for s in order if by[s]['y']==0][:k]+[by[s] for s in order if by[s]['y']==1][:k]
    assert len(pick)==2*k;return sorted(pick,key=lambda c:c['patient_id'])

def label_structure(cases,cfg):
    """The structural part of the 7.6 label-mapping check over all 120 cases (no training, no predictions)."""
    out=dict(cases=len(cases),y_values=sorted({c['y'] for c in cases}),mapping=cfg['label_mapping'])
    assert out['y_values']==[0,1] and all(c['y']==c['grade']-1 for c in cases)
    folds={}
    for f in cfg['folds']:
        r=roles(cases,f);tr={c['patient_id'] for c in r['train']};te={c['patient_id'] for c in r['test']}
        assert not tr&te and len(tr|te)==120
        folds[f]=dict(train=[sum(c['y']==y for c in r['train']) for y in (0,1)],test=[sum(c['y']==y for c in r['test']) for y in (0,1)])
        assert folds[f]==dict(train=[48,48],test=[12,12]),folds[f]
    tested=[c['patient_id'] for f in cfg['folds'] for c in roles(cases,f)['test']]
    assert sorted(tested)==sorted(c['patient_id'] for c in cases)
    out['folds']=folds;out['each_case_tested_once']=True
    return out

# ----------------------------------------------------------------------------------------------- feature files
def feature_path(case,enc,arm):
    return FULL/case['stem']/f'{enc}.npy' if arm=='Lfull' else HR/case['stem']/f'{enc}_{arm}.npy'
def case_record(case):
    """The stage-1 extraction record of a case (None if not extracted yet). Raises if it is not a usable PASS record."""
    rec=read_json(HR/case['stem']/'complete.json')
    if rec is None:return None
    assert rec.get('state')=='complete' and rec.get('patient_id')==case['patient_id'],f"{case['stem']}: bad complete.json"
    assert rec.get('status')=='PASS',f"{case['stem']}: stage-1 status {rec.get('status')} (L parity failed: needs review)"
    return rec
def expected_feature(case,enc,arm,cfg):
    """(path, sha256, shape) a case's feature file must have, from the extraction records."""
    rec=case_record(case);assert rec is not None,f"{case['stem']}: not extracted"
    d=cfg['encoders'][enc]
    if arm=='Lfull':
        ref=json.loads((FULL/case['stem']/'complete.json').read_text())
        assert ref['n']==case['primary_count'] and ref['features'][enc]['sha256']==rec['stored_feature_sha256'][enc]
        return feature_path(case,enc,arm),ref['features'][enc]['sha256'],(case['primary_count'],d)
    m=rec['n_locations'];assert rec['primary_count']==case['primary_count'] and m==min(case['primary_count'],cfg['max_locations']),f"{case['stem']}: location count"
    shape=(m,d) if arm=='L' else (m,cfg['subtiles'],d)
    return feature_path(case,enc,arm),rec['files'][f'{enc}_{arm}.npy'],shape
def missing_data(cases,enc,arm):
    """Cases whose inputs for (enc, arm) do not exist yet (no complete stage-1 record, or a missing file)."""
    out=[]
    for c in cases:
        f=HR/c['stem']/'complete.json'
        if not f.exists() or not feature_path(c,enc,arm).exists():out.append(c['stem']);continue
        assert read_json(f) is not None,f'{f}: unreadable extraction record'  # corrupt, not "not yet extracted"
    return out
def record_digest(case,arm):
    """sha256 of the extraction record(s) a run depends on (part of every run fingerprint)."""
    d=dict(stage1=sha(HR/case['stem']/'complete.json'))
    if arm=='Lfull':d['full']=sha(FULL/case['stem']/'complete.json')
    return d

# ---------------------------------------------------------------------------------------------- release & lock
def release_status():
    """(released, missing conditions): every condition listed in the stage-1 QUARANTINE.json must be recorded as met (with
    who and when) by a human in RELEASE.json next to it."""
    q=read_json(HR/'QUARANTINE.json')
    if not q or 'release_conditions' not in q:return False,['QUARANTINE.json missing or unreadable']
    names=[c.split(':',1)[0].strip() for c in q['release_conditions']]
    if sorted(names)!=sorted(RELEASE_CONDITIONS):return False,['QUARANTINE.json lists %s, expected %s'%(names,list(RELEASE_CONDITIONS))]
    if q.get('list_sha256')!=config()['stage1_list_sha256']:return False,['QUARANTINE.json belongs to another case list']
    r=read_json(HR/'RELEASE.json') or {};conds=r.get('conditions') if isinstance(r,dict) else None
    if not isinstance(conds,dict):return False,['RELEASE.json missing or malformed (needs {"conditions": {name: {met, by, date}}})']
    missing=[n for n in names if not (isinstance(conds.get(n),dict) and conds[n].get('met') is True and conds[n].get('by') and conds[n].get('date'))]
    return not missing,missing
def lock_status():
    """(ok, problems): LOCK.json exists and every file it lists still has the recorded sha256."""
    lk=read_json(LOCK_JSON)
    if not lk:return False,['LOCK.json missing']
    bad=[n for n,h in lk['files'].items() if not (HERE/n).exists() or sha(HERE/n)!=h]
    if lk.get('code_signature')!=signature():bad.append('code_signature')
    return not bad,bad

# ----------------------------------------------------------------------------------------------------- metrics
def auc(y,score):
    """Mann-Whitney AUC of score for y == 1 against y == 0 (ties count 1/2)."""
    y=np.asarray(y).astype(bool);s=np.asarray(score,dtype=float);p=s[y];n=s[~y]
    assert len(p)>0 and len(n)>0 and np.isfinite(s).all()
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
