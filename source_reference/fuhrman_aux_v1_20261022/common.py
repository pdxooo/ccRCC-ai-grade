"""Shared paths, locked inputs, label handling and metrics for direction 3 (Fuhrman as auxiliary supervision; PROTOCOL.md).
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
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
SPLITS=ROOT/'bib_phase1/first_round_v1_20261010'
FULL=ROOT/'features/nine_encoder_full_v1_20261010'
LABELS=HERE/'inputs/korea_labels.csv'
RUNS=HERE/'runs'
CHECK_PASS=HERE/'CHECK_PASS.json'
LOCK_JSON=HERE/'LOCK.json'
SIGNATURE_FILES=['common.py','train.py','check.py','evaluate.py','config.json','inputs/korea_labels.csv']

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
def sample_patches(n,k,cfg,seed_index,fold,epoch,patient):
    """Training patches of one case in one epoch: k of n without replacement, sorted (all if n <= k). Depends only on
    (seed, fold, epoch, case): every arm and both encoders see the same patches."""
    if n<=k:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,patient)).choice(n,k,replace=False))
def train_order(n,cfg,seed_index,fold,epoch):
    return np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,'order')).permutation(n)

# ------------------------------------------------------------------------------------------------ locked inputs
def load_cases(cfg):
    """All 300 Korean cases: manifest + five-fold table (validated as in fusion_gain_v2) joined with the database labels.
    The database ISUP grade must equal the manifest/five-fold grade of every case. Returns (cases by patient_id, assign)."""
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
    assert sha(LABELS)==cfg['labels_csv_sha256'],'korea_labels.csv is not the locked label file'
    lab={r['patient_id']:r for r in read_csv(LABELS)}
    assert set(lab)==set(lookup),'label table and manifest name different patients'
    cases=[]
    for p,row in sorted(lookup.items()):
        g=int(lab[p]['ISUP_grade']);fu=int(lab[p]['Fuhrman_grade'])
        assert g==int(row['ISUP_grade']),f'{p}: database ISUP {g} != manifest {row["ISUP_grade"]}'
        cases.append(dict(patient_id=p,stem=Path(row['slide_filename']).stem,primary_count=int(row['primary_count']),
                          isup=g,fuhrman=fu,y=int(cfg['isup_mapping'][str(g)]),z=int(cfg['fuhrman_aux_mapping'][str(fu)])))
    return cases,assign

def roles(cases,assign,fold,cfg):
    """Training set = train + val of the locked fold (240 cases), test = its 60 test cases."""
    role={r['patient_id']:r['role'] for r in assign if int(r['fold'])==fold}
    tr=('train','val') if cfg['training_set']=='train_plus_val' else ('train',)
    return dict(train=[c for c in cases if role[c['patient_id']] in tr],test=[c for c in cases if role[c['patient_id']]=='test'])

def aux_labels(train,arm,cfg,seed_index,fold):
    """Auxiliary target per training case: none (arm A), the case's own Fuhrman 1-2 vs 3-4 (B), or the training cases' own
    Fuhrman labels permuted within each ISUP grade (C: keeps the ISUP-Fuhrman association and the class counts, breaks
    the link to the case's image). Fixed per (seed, fold)."""
    kind=cfg['arms'][arm]['aux_labels']
    if kind=='none':return {}
    if kind=='true':return {c['patient_id']:c['z'] for c in train}
    assert kind=='shuffled_within_isup_grade'
    out={}
    for g in sorted({c['isup'] for c in train}):
        grp=sorted((c for c in train if c['isup']==g),key=lambda c:c['patient_id'])
        perm=np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,'shuffle',g)).permutation(len(grp))
        for c,j in zip(grp,perm):out[c['patient_id']]=grp[int(j)]['z']
    return out

def check_cases(cases,assign,cfg):
    """Pre-defined technical-check cases: per ISUP grade the first k (by patient_id) of fold 0's training set."""
    tr=roles(cases,assign,0,cfg)['train'];k=cfg['check']['cases_per_isup_grade']
    pick=[c for g in (1,2,3,4) for c in [x for x in tr if x['isup']==g][:k]]
    assert len(pick)==4*k;return pick

def label_structure(cases,assign,cfg):
    """Structural label checks over all 300 cases (no training)."""
    out=dict(cases=len(cases),isup=dict(sorted(__import__('collections').Counter(c['isup'] for c in cases).items())),
             aux=dict(sorted(__import__('collections').Counter(c['z'] for c in cases).items())))
    assert all(c['y']==c['isup']-1 and c['z']==(0 if c['fuhrman']<=2 else 1) for c in cases)
    tested=[]
    for f in cfg['folds']:
        r=roles(cases,assign,f,cfg);a={c['patient_id'] for c in r['train']};b={c['patient_id'] for c in r['test']}
        assert not a&b and len(a)==240 and len(b)==60;tested+=sorted(b)
        for s in cfg['seeds']:
            sh=aux_labels(r['train'],'C',cfg,s,f)
            for g in (1,2,3,4):  # the shuffle keeps each ISUP grade's aux-label counts
                grp=[c for c in r['train'] if c['isup']==g];assert sorted(sh[c['patient_id']] for c in grp)==sorted(c['z'] for c in grp)
    assert sorted(tested)==sorted(c['patient_id'] for c in cases)
    out['each_case_tested_once']=True;return out

# ----------------------------------------------------------------------------------------------- feature files
def feature_path(case,enc):return FULL/case['stem']/f'{enc}.npy'
def expected_feature(case,enc,cfg):
    """(path, sha256, shape) from the extraction record nine_encoder_full_v1/<stem>/complete.json."""
    ref=json.loads((FULL/case['stem']/'complete.json').read_text())
    assert ref['n']==case['primary_count'],f"{case['stem']}: patch count {ref['n']} != manifest {case['primary_count']}"
    return feature_path(case,enc),ref['features'][enc]['sha256'],(case['primary_count'],cfg['encoders'][enc])
def missing_data(cases,enc):
    """Cases whose features are not verified on the mirror yet (batch marker and per-file .verified marker)."""
    if PREFIX and not (TRANSFER/f'verified_feat_{enc}').exists():return [c['stem'] for c in cases]
    out=[]
    for c in cases:
        f=feature_path(c,enc)
        if not f.exists() or (PREFIX and not Path(str(f)+'.verified').exists()) or not (FULL/c['stem']/'complete.json').exists():out.append(c['stem'])
    return out

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
def g12_auc(isup,P):
    """G1 vs G2 among ISUP 1-2 cases, score P(G2) / (P(G1) + P(G2)) from the 4-grade probabilities."""
    isup=np.asarray(isup);P=np.asarray(P);m=isup<=2
    return auc(isup[m]==2,P[m,1]/np.maximum(P[m,0]+P[m,1],1e-12))
def highlow_auc(isup,P):return auc(np.asarray(isup)>=3,np.asarray(P)[:,2:].sum(1))
def grade_metrics(isup,P):
    """4-grade argmax metrics: accuracy, macro-F1, quadratic kappa."""
    y=np.asarray(isup)-1;p=np.asarray(P).argmax(1);cm=np.zeros((4,4));np.add.at(cm,(y,p),1)
    tp=np.diag(cm);d=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,d,out=np.zeros(4),where=d>0)
    w=(np.arange(4)[:,None]-np.arange(4)[None,:])**2/9;e=np.outer(cm.sum(1),cm.sum(0))/cm.sum()
    return dict(accuracy=float((y==p).mean()),macro_f1=float(f1.mean()),qwk=float(1-(w*cm).sum()/(w*e).sum()))
