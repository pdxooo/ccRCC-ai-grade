"""Shared paths, locked inputs, survival labels and statistics for HE + clinical recurrence prediction (PROTOCOL.md).
No torch import here (evaluate.py and the label checks run without it)."""
import os,sys,json,csv,hashlib,datetime,math
INHERITED_CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'  # CUDA indices == nvidia-smi indices
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_k,'4')
from pathlib import Path
PREFIX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')
assert PREFIX=='' or PREFIX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
ROOT=Path(PREFIX+'/data/cluster/projects/korean_ccrcc_SBIAD2452')
SROOT=Path(PREFIX+'/data/cluster/projects/scottish_ccrcc_SBIAD2454')
sys.path.insert(0,str(ROOT/'qc_deps'))
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('HE_PROJECT_DIR','/data/workstation/he_ccrcc_isup_20261021'))
TRANSFER=PROJECT/'transfer'
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
SPLITS=ROOT/'bib_phase1/first_round_v1_20261010'
FULL=ROOT/'features/nine_encoder_full_v1_20261010'
SPOOL=SROOT/'qc/external_primary_technical_pool_v1_20261013'
SFULL=SROOT/'features/nine_encoder_full_v1_20261021'
IN=HERE/'inputs'
RUNS=HERE/'runs'
CHECK_PASS=HERE/'CHECK_PASS.json'
LOCK_JSON=HERE/'LOCK.json'
SIGNATURE_FILES=['common.py','train.py','check.py','evaluate.py','config.json','inputs/korea_surv.csv','inputs/scotland_surv.csv',
                 'inputs/korea_if.csv','inputs/scotland_if.csv']
FEATURE_DIR={'korea':FULL,'scotland':SFULL}
MARKER={'korea':'verified_feat_{enc}','scotland':'verified_scotland_feat_{enc}'}

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
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def model_seed(cfg,seed_index,fold):return cfg['seed_base']+100*seed_index+fold
def sample_patches(n,k,cfg,seed_index,fold,epoch,patient):
    """k of n patches without replacement, sorted (all if n <= k); depends only on (seed, fold, epoch, case)."""
    if n<=k:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,patient)).choice(n,k,replace=False))
def train_order(n,cfg,seed_index,fold,epoch):
    return np.random.default_rng(stable_seed(cfg['seed_base'],seed_index,fold,epoch,'order')).permutation(n)

# ------------------------------------------------------------------------------------------------ locked inputs
def labels(cfg,name):
    p=IN/name;assert sha(p)==cfg['labels_sha256'][name],f'{name} is not the locked label file'
    return {r['patient_id']:r for r in read_csv(p)}
def outcome(row,cfg,zero_dfs_time=None):
    """(time, event) after 84-month administrative censoring. A DFS of 0 becomes zero_dfs_time (sensitivity only)."""
    t=float(row['dfs_months']);e=int(row['recurrence'])
    if t==0 and zero_dfs_time is not None:t=zero_dfs_time
    c=cfg['censor_months']
    return (c,0) if t>c else (t,e)
def case_from(row,lab,cohort,cfg):
    t,e=outcome(lab,cfg)
    return dict(patient_id=row['patient_id'],cohort=cohort,stem=Path(row['slide_filename']).stem,primary_count=int(row['primary_count']),
                isup=int(lab['isup']),time=t,event=e,dfs_raw=float(lab['dfs_months']),zero_dfs=float(lab['dfs_months'])==0,
                leibovich_fuhrman=float(lab['leibovich_fuhrman']),leibovich_isup=float(lab['leibovich_isup']))

def load_korea(cfg):
    """All 300 Korean cases (manifest + five-fold table validated as in fusion_gain_v2) joined with the database outcome
    table; the database ISUP must equal the manifest's. zero_dfs cases are flagged (excluded from training and the primary
    analysis, predicted for the sensitivity analysis)."""
    rows=read_csv(POOL/'patient_manifest.csv')
    lock=json.loads((POOL/'LOCK.json').read_text());sl=json.loads((SPLITS/'SPLITS_LOCK.json').read_text())
    assert len(rows)==300 and len({r['patient_id'] for r in rows})==300
    assert sha(POOL/'patient_manifest.csv')==lock['patient_manifest_sha256']==sl['patient_manifest_sha256']
    assert sha(POOL/'LOCK.json')==sl['qc_lock_sha256'] and sha(SPLITS/'patient_splits.csv')==sl['split_sha256']
    assign=read_csv(SPLITS/'patient_splits.csv');assert len(assign)==1500
    lookup={r['patient_id']:r for r in rows}
    for fold in range(5):
        a=[r for r in assign if int(r['fold'])==fold]
        assert len(a)==300 and len({r['patient_id'] for r in a})==300 and sum(r['role']=='test' for r in a)==60
        for r in a:
            assert r['role'] in ('train','val','test')
            for key in ('slide_filename','ISUP_grade'):assert r[key]==lookup[r['patient_id']][key]
    lab=labels(cfg,'korea_surv.csv');assert set(lab)=={r['patient_id'] for r in rows},'outcome table and manifest name different patients'
    cases=[]
    for r in sorted(rows,key=lambda r:r['patient_id']):
        c=case_from(r,lab[r['patient_id']],'korea',cfg)
        assert c['isup']==int(r['ISUP_grade']),f"{r['patient_id']}: database ISUP {c['isup']} != manifest {r['ISUP_grade']}"
        cases.append(c)
    return cases,assign

def load_scotland(cfg):
    """All 120 Scottish cases of the locked external technical pool, joined with the database outcome table."""
    rows=read_csv(SPOOL/'patient_manifest.csv');lock=json.loads((SPOOL/'LOCK.json').read_text())
    assert sha(SPOOL/'patient_manifest.csv')==lock['patient_manifest_sha256'] and len(rows)==120
    lab=labels(cfg,'scotland_surv.csv');assert set(lab)=={r['patient_id'] for r in rows},'outcome table and manifest name different patients'
    cases=[]
    for r in sorted(rows,key=lambda r:r['patient_id']):
        c=case_from(r,lab[r['patient_id']],'scotland',cfg)
        assert c['isup']==int(r['ISUP_grade']),f"{r['patient_id']}: database ISUP {c['isup']} != manifest {r['ISUP_grade']}"
        cases.append(c)
    return cases

def primary(cases,cfg):return [c for c in cases if not (cfg['exclude_zero_dfs'] and c['zero_dfs'])]
def roles(cases,assign,fold,cfg):
    """Training set = train + val of the locked fold without DFS-0 cases; test = the fold's test cases (DFS-0 cases
    included: they are predicted for the sensitivity analysis only)."""
    role={r['patient_id']:r['role'] for r in assign if int(r['fold'])==fold}
    tr=('train','val') if cfg['training_set']=='train_plus_val' else ('train',)
    return dict(train=[c for c in primary(cases,cfg) if role[c['patient_id']] in tr],test=[c for c in cases if role[c['patient_id']]=='test'])
def time_bins(train,cfg):
    """Cut points of the discrete-time model: quartiles of the training set's recurrence times."""
    t=np.array([c['time'] for c in train if c['event']]);assert len(t)>=cfg['n_bins'],'too few events'
    q=[float(x) for x in np.quantile(t,[k/cfg['n_bins'] for k in range(1,cfg['n_bins'])])]
    assert all(b>a for a,b in zip(q,q[1:])),f'tied bin edges {q}'
    return q
def bin_index(t,cuts):return int(np.searchsorted(np.asarray(cuts),t,side='right'))
def check_cases(cases,assign,cfg):
    """Pre-defined technical-check cases: the first k recurrences and first k non-recurrences (by patient_id) of fold 0's
    training set."""
    tr=roles(cases,assign,0,cfg)['train'];k=cfg['check']['cases_per_status']
    pick=[c for c in tr if c['event']][:k]+[c for c in tr if not c['event']][:k];assert len(pick)==2*k;return pick
def label_structure(cases,assign,scot,cfg):
    out=dict(korea=len(cases),korea_primary=len(primary(cases,cfg)),korea_events=sum(c['event'] for c in primary(cases,cfg)),
             korea_zero_dfs=[c['patient_id'] for c in cases if c['zero_dfs']],scotland=len(scot),scotland_events=sum(c['event'] for c in scot),
             max_event_time_korea=max(c['time'] for c in cases if c['event']),censor_months=cfg['censor_months'])
    tested=[]
    for f in cfg['folds']:
        r=roles(cases,assign,f,cfg);a={c['patient_id'] for c in r['train']};b={c['patient_id'] for c in r['test']}
        assert not a&b and not any(c['zero_dfs'] for c in r['train']);tested+=sorted(b);time_bins(r['train'],cfg)
    assert sorted(tested)==sorted(c['patient_id'] for c in cases);out['each_case_tested_once']=True
    return out

# ----------------------------------------------------------------------------------------------- feature files
def feature_path(case,enc):return FEATURE_DIR[case['cohort']]/case['stem']/f'{enc}.npy'
def expected_feature(case,enc,cfg):
    ref=json.loads((FEATURE_DIR[case['cohort']]/case['stem']/'complete.json').read_text())
    assert ref['n']==case['primary_count'],f"{case['stem']}: patch count {ref['n']} != manifest {case['primary_count']}"
    return feature_path(case,enc),ref['features'][enc]['sha256'],(case['primary_count'],cfg['encoders'][enc])
def missing_data(cases,enc):
    """Cases whose features are not verified on the mirror yet (cohort batch marker and per-file .verified marker)."""
    out=[]
    for c in cases:
        f=feature_path(c,enc)
        if PREFIX and not (TRANSFER/MARKER[c['cohort']].format(enc=enc)).exists():out.append(c['stem']);continue
        if not f.exists() or (PREFIX and not Path(str(f)+'.verified').exists()) or not (f.parent/'complete.json').exists():out.append(c['stem'])
    return out
def lock_status():
    lk=read_json(LOCK_JSON)
    if not lk:return False,['LOCK.json missing']
    bad=[n for n,h in lk['files'].items() if not (HERE/n).exists() or sha(HERE/n)!=h]
    if lk.get('code_signature')!=signature():bad.append('code_signature')
    return not bad,bad

# ----------------------------------------------------------------------------------------------------- statistics
def harrell_c(time,event,risk):
    """Harrell's C: among pairs where the earlier time is a recurrence (a recurrence and a censoring at the same time also
    count, censored later; as R survival::concordance), the share with the higher risk (risk ties 1/2)."""
    t=np.asarray(time,float);e=np.asarray(event,int);r=np.asarray(risk,float);num=den=0.
    for i in np.flatnonzero(e==1):
        m=(t>t[i])|((t==t[i])&(e==0));den+=m.sum();num+=(r[i]>r[m]).sum()+.5*(r[i]==r[m]).sum()
    assert den>0;return float(num/den)
COX_WARNINGS=[]  # fits that did not converge (gradient not ~0) or show separation (|beta| > 15); evaluate records them
def cox_fit(X,time,event,iters=100):
    """Cox proportional hazards (Breslow ties), Newton-Raphson with step halving. Returns beta, standard errors,
    log-likelihood."""
    X=np.asarray(X,float);t=np.asarray(time,float);e=np.asarray(event,int);n,p=X.shape;b=np.zeros(p)
    R=(t[None,:]>=t[:,None]).astype(float)  # R[i, j] = 1: j is at risk at time t_i
    ev=np.flatnonzero(e==1);assert len(ev)>0
    def ll(b):
        eta=X@b;c=eta.max();w=np.exp(eta-c);return float((eta[ev]-c-np.log(R[ev]@w)).sum())
    def grad_hess(b):
        eta=X@b;w=np.exp(eta-eta.max());s0=R[ev]@w;s1=R[ev]@(w[:,None]*X);m=s1/s0[:,None]
        g=(X[ev]-m).sum(0);WX=w[:,None]*X;H=np.zeros((p,p))
        for k,i in enumerate(ev):H-=(R[i][:,None]*WX).T@X/s0[k]-np.outer(m[k],m[k])
        return g,H
    cur=ll(b)
    for _ in range(iters):
        g,H=grad_hess(b);step=np.linalg.solve(H,-g);a=1.;moved=False
        while a>1e-8:
            nb=b+a*step;nl=ll(nb)
            if nl>=cur-1e-12:moved=True;break
            a/=2
        if not moved:break  # no step improves the likelihood: keep b
        prev=cur;b,cur=nb,nl
        if abs(cur-prev)<1e-10 and np.abs(a*step).max()<1e-8:break
    g,H=grad_hess(b);cov=np.linalg.inv(-H)
    if np.abs(g).max()>1e-6*len(ev) or np.abs(b).max()>15:COX_WARNINGS.append(dict(n=n,events=len(ev),beta=[float(x) for x in b],max_grad=float(np.abs(g).max())))
    return b,np.sqrt(np.diag(cov)),cur
def norm_sf(z):return 0.5*math.erfc(z/math.sqrt(2))
def logrank(time,event,group):
    """Two-group log-rank test: (chi2, p)."""
    t=np.asarray(time,float);e=np.asarray(event,int);g=np.asarray(group,bool);O=E=V=0.
    for u in np.unique(t[e==1]):
        at=t>=u;n=at.sum();n1=(at&g).sum();d=((t==u)&(e==1)).sum();d1=((t==u)&(e==1)&g).sum()
        O+=d1;E+=d*n1/n
        if n>1:V+=d*(n1/n)*(1-n1/n)*(n-d)/(n-1)
    chi=(O-E)**2/V if V>0 else 0.;return float(chi),float(math.erfc(math.sqrt(chi/2)))
def avg_rank(x):
    """Ranks 1..n with ties given their average rank."""
    x=np.asarray(x,float);o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(1,len(x)+1)
    for v in np.unique(x):
        m=x==v
        if m.sum()>1:r[m]=r[m].mean()
    return r
def spearman(a,b):return float(np.corrcoef(avg_rank(a),avg_rank(b))[0,1])
