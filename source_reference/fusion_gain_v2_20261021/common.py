import os,sys,json,csv,hashlib
# Forced (not setdefault): a value inherited from the shared account's shell must not silently change the cuBLAS workspace
# between the smoke, pause and resume legs. The inherited value, if any, is recorded in metadata.json.
INHERITED_CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
# CUDA device indices = nvidia-smi indices (PCI bus order), so CUDA_VISIBLE_DEVICES picks the card the scheduler means.
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# The account's nproc limit is shared: never let numpy's OpenBLAS (qc_deps) start one thread per core on import.
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_k,'4')
from pathlib import Path
# HE_PATH_PREFIX (lab-a6000: /data/workstation/he_ccrcc_isup_20261021/hpcroot) is prepended to the HPC absolute path.
# Unset or empty -> exactly the original HPC path. Every other data path derives from ROOT.
HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
PREFIX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')
assert PREFIX=='' or PREFIX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
ROOT=Path(PREFIX+HPC_ROOT)
sys.path.insert(0,str(ROOT/'qc_deps'))
import numpy as np
HERE=Path(__file__).resolve().parent
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
FEATURES=ROOT/'features/nine_encoder_full_v1_20261010'
SPLITS=ROOT/'bib_phase1/first_round_v1_20261010'
DATA_READY=ROOT/'bib_phase1/mil_first_round_v2_20261015/DATA_READY.json'
DIMS={'resnet50':2048,'convnext_tiny':768,'vit_b_16':768,'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def config():return json.loads((HERE/'config.json').read_text())
def signature():
    names=['common.py','train.py','smoke.py','evaluate.py','ready.py','config.json']
    return hashlib.sha256(json.dumps({n:sha(HERE/n) for n in names},sort_keys=True).encode()).hexdigest()
def read_csv(path):
    # Windows exports may carry a UTF-8 BOM; never change locked input bytes.
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))
def load_inputs():
    rows=read_csv(POOL/'patient_manifest.csv')
    lock=json.loads((POOL/'LOCK.json').read_text());sl=json.loads((SPLITS/'SPLITS_LOCK.json').read_text())
    assert len(rows)==300 and len({r['patient_id'] for r in rows})==300
    assert sha(POOL/'patient_manifest.csv')==lock['patient_manifest_sha256']==sl['patient_manifest_sha256']
    assert sha(POOL/'LOCK.json')==sl['qc_lock_sha256'] and sha(SPLITS/'patient_splits.csv')==sl['split_sha256']
    lookup={r['patient_id']:r for r in rows};assign=read_csv(SPLITS/'patient_splits.csv')
    assert len(assign)==1500
    tests=[]
    for fold in range(5):
        a=[r for r in assign if int(r['fold'])==fold]
        assert len(a)==300 and len({r['patient_id'] for r in a})==300
        for role in ('train','val','test'):
            part=[r for r in a if r['role']==role];assert {r['ISUP_grade'] for r in part}==set('1234')
        assert sum(r['role']=='test' for r in a)==60
        for r in a:
            assert r['role'] in ('train','val','test')
            for key in ('slide_filename','ISUP_grade'):assert r[key]==lookup[r['patient_id']][key]
            if r['role']=='test':tests.append(r['patient_id'])
    assert len(tests)==len(set(tests))==300
    return lookup,assign
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def patch_indices(n,limit,seed,fold,epoch,patient):
    if limit is None or n<=limit:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(stable_seed(seed,fold,epoch,patient)).choice(n,limit,replace=False))
def metrics(labels,probs):
    y=np.asarray(labels,dtype=int);p=np.asarray(probs,dtype=float)
    assert len(y)>0 and p.shape==(len(y),4) and np.isfinite(p).all()
    assert np.all(p>=0) and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1);cm=np.zeros((4,4),dtype=int);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);row=cm.sum(1);col=cm.sum(0)
    f1=np.divide(2*tp,row+col,out=np.zeros(4),where=(row+col)>0)
    rec=np.divide(tp,row,out=np.zeros(4),where=row>0)
    w=(np.arange(4)[:,None]-np.arange(4)[None,:])**2/9
    expected=np.outer(row,col)/len(y);den=float((w*expected).sum())
    return dict(n=len(y),macro_f1=float(f1.mean()),balanced_accuracy=float(rec[row>0].mean()),accuracy=float((pred==y).mean()),quadratic_kappa=float(1-(w*cm).sum()/den) if den>0 else None,grade_mae=float(np.abs(pred-y).mean()),cross_entropy=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),confusion_matrix=cm.tolist(),per_class_f1=f1.tolist(),per_class_recall=rec.tolist())

def arm_seed(cfg,seed_index,fold):return cfg['seed_base']+100*seed_index+fold
