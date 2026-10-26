"""High-resolution pilot, stage 0: technical acceptance only. No training, no labels beyond case selection.

Same 128 um locations as the locked pool; one native decode per location yields three inputs:
  L  : full 128 um crop resized to 224 (current pipeline, ~0.57 um/px)
  H  : 2x2 physical sub-tiles (64 um) each resized to 224 (~0.29 um/px)
  Hd : 2x2 quadrants of the L image, each upsampled to 224 (H geometry, L detail)

A6000 port (lab-a6000, no Slurm; see CHANGES.md). Scientific recipe unchanged. Differences:
  * every HPC absolute path is prefixed with $HE_PATH_PREFIX ('' or unset => byte-identical HPC paths);
  * import shim: this directory's patched czi_pilot_v2.py / validate.py are the modules used,
    full_patch_qc.py is imported unchanged from ROOT/bib_phase1 (path + sha256 checked in the main process at start
    and again in every decode worker by the pool initializer);
  * output directory features/highres_pilot_v1_20261021_a6000 (never mixes with the HPC run, job 5397630);
  * with a prefix (A6000 mirror) it waits unless $HE_PROJECT_DIR/transfer/verified_stage0 and the per-file .verified
    markers exist; patient_manifest.csv must match the pool LOCK.json; weight sha256s must match the HPC smoke summary;
  * shared GPU: a busy card or a CUDA out-of-memory => exit 3 (retry later), never reported as FAIL; a crash =>
    STAGE0_REPORT.json state 'error', exit 2; a PASS already recorded for this exact code is not recomputed (--force);
  * the embedding loop is factored into module-level functions so stage1_extract.py imports (not copies) it.
Exit codes: 0 PASS_PENDING_HUMAN_VISUAL_REVIEW (now, or already recorded for this code) | 1 FAIL: automated checks
failed, or weight / manifest sha256 mismatch (a human must look) | 2 error: crash, state 'error' (retrying is safe) |
3 waiting: transferred inputs not verified yet, or the shared GPU is busy / ran out of memory (retry later) |
5 busy: another stage-0 process holds the run lock (nothing done).
"""
import os
if __name__=='__main__':  # A6000: exit 1 means FAIL only (the scheduler stops all high-res work on it); any uncaught crash exits 2
    import sys as _sys,traceback as _tb
    def _crash(*exc):
        _tb.print_exception(*exc);_sys.stdout.flush();_sys.stderr.flush();os._exit(2)
    _sys.excepthook=_crash
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,csv,json,hashlib,time,importlib,importlib.util,collections,multiprocessing,fcntl,signal,traceback
from pathlib import Path
PFX=os.environ.get('HE_PATH_PREFIX','').rstrip('/')  # A6000 mirror root prepended to every HPC absolute path; '' => HPC original
def hpc(p):return PFX+p if PFX else p
ROOT=Path(hpc('/data/cluster/projects/korean_ccrcc_SBIAD2452'))
sys.path.insert(0,str(ROOT/'qc_deps'));sys.path.insert(0,str(ROOT/'bib_phase1'))
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))  # A6000 import shim: patched czi_pilot_v2.py here shadows ROOT/bib_phase1's; full_patch_qc.py has no copy here
import numpy as np
from concurrent.futures import ProcessPoolExecutor,TimeoutError as FuturesTimeout
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
STORED=ROOT/'features/nine_encoder_full_v1_20261010'
OUT=ROOT/'features/highres_pilot_v1_20261021_a6000'  # A6000: the HPC run (job 5397630) writes features/highres_pilot_v1_20261021; kept apart even without a prefix
RUN_LOCK=OUT.with_name(OUT.name+'.lock')  # single-instance lock, outside OUT so a directory-level parity comparison never sees it
OLD=Path(hpc('/data/cluster/projects/gastritis_uni2h/outputs'))
VALIDATE=HERE/'validate.py'  # A6000: prefix-patched copy of OLD/'encoder-validation-20261008/validate.py'
SMOKE=ROOT/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json'  # weights that produced the stored features
PROJECT=Path(os.environ.get('HE_PROJECT_DIR','/data/workstation/he_ccrcc_isup_20261021'))
TRANSFER=PROJECT/'transfer'
GATED=bool(PFX)  # transfer markers exist only on the A6000 mirror; without a prefix (HPC) the data are local, as in the original
# sha256 of the helper modules this script must import (patched copies here; full_patch_qc unchanged on the mirror).
PINNED={'czi_pilot_v2.py':'b776d23b53c5ddfbb7987fd515dbf561808d386d86a4f9b9857687ae8fa3a1d8',
        'validate.py':'637fde1a91a80f3f8b74a4d312ee94c12d719058e1c3dfe8fe7b3627f42c2305',
        'full_patch_qc.py':'29e7ffa8623a910ff15cd09dd00966c52aa8c0d153fc00e73c83a6a683a4df9a'}
ENCODERS={'uni':(1024,'bilinear'),'virchow2':(2560,'bicubic')}  # interpolation as in extract_nine_korea.py
LOCATIONS=2048;SEED=20261021;STAGE0_G1=2;STAGE0_G2=2;BATCH=32;WORKERS=6
WAITING=3
EXIT_OK,EXIT_FAIL,EXIT_ERROR,EXIT_BUSY=0,1,2,5
PASS_STATE='PASS_PENDING_HUMAN_VISUAL_REVIEW'
MIN_FREE_GIB=12  # shared GPU: with less free memory than this, do not load models (exit 3, retry later)
DECODE_TIMEOUT=1800  # seconds to wait for one decoded batch (normally seconds): a hung CZI read becomes an error, not a hang
_reader=None;_path=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(p)
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def location_indices(n,patient):
    if n<=LOCATIONS:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(stable_seed('highres_v1',SEED,patient)).choice(n,LOCATIONS,replace=False))

def subtile_boxes(w,h):
    """Exact physical halves in native pixels; union is the full region with no overlap."""
    xm=int(round(w/2));ym=int(round(h/2))
    return [(0,0,xm,ym),(xm,0,w,ym),(0,ym,xm,h),(xm,ym,w,h)]  # row-major: TL, TR, BL, BR

def make_inputs(rgb,interp):
    from PIL import Image
    mode={'bilinear':Image.Resampling.BILINEAR,'bicubic':Image.Resampling.BICUBIC}[interp]
    im=Image.fromarray(rgb);h,w=rgb.shape[:2]
    L=np.asarray(im.resize((224,224),mode))
    H=np.stack([np.asarray(im.crop(b).resize((224,224),mode)) for b in subtile_boxes(w,h)])
    Lim=Image.fromarray(L)
    Hd=np.stack([np.asarray(Lim.crop(b).resize((224,224),mode)) for b in subtile_boxes(224,224)])
    return L,H,Hd

def decode(task):
    global _reader,_path
    from full_patch_qc import IndexedReader
    path,start,coords=task
    if _path!=path:
        if _reader is not None:_reader.close()
        _reader=IndexedReader(path);_path=path
    out={k:([],[],[]) for k in ('bilinear','bicubic')};native=[]
    for scene,x,y,w,h in coords:
        rgb,cov,_=_reader.region(int(scene),int(x),int(y),int(w),int(h))
        assert float(cov.mean())>=.999,'QC coordinate coverage changed'
        assert rgb.shape[:2]==(int(h),int(w))
        native.append((int(w),int(h)))
        for interp,(Ls,Hs,Hds) in out.items():
            L,H,Hd=make_inputs(rgb,interp);Ls.append(L);Hs.append(H);Hds.append(Hd)
    return start,{k:tuple(np.stack(a) for a in v) for k,v in out.items()},native

class DecodeTimeout(RuntimeError):pass
def buffered(pool,tasks,depth=8):
    it=iter(tasks);pending=collections.deque()
    for _ in range(depth):
        try:pending.append(pool.submit(decode,next(it)))
        except StopIteration:break
    while pending:
        try:result=pending.popleft().result(timeout=DECODE_TIMEOUT)  # A6000: bounded wait (original: unbounded)
        except FuturesTimeout:raise DecodeTimeout(f'no decoded batch within {DECODE_TIMEOUT}s (hung CZI read or stalled I/O)') from None
        try:pending.append(pool.submit(decode,next(it)))
        except StopIteration:pass
        yield result

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def select_cases(rows):
    rng=np.random.default_rng(SEED)
    g1=sorted(r['patient_id'] for r in rows if r['ISUP_grade']=='1');g2=sorted(r['patient_id'] for r in rows if r['ISUP_grade']=='2')
    return sorted(rng.choice(g1,STAGE0_G1,replace=False).tolist()+rng.choice(g2,STAGE0_G2,replace=False).tolist())

def visual(dst,patient,coords_sel,reader_path,picks):
    """Side-by-side PNG per location: L shown at 448 (nearest) | H 2x2 mosaic at 448. For human review."""
    from PIL import Image
    from full_patch_qc import IndexedReader
    r=IndexedReader(reader_path);files=[]
    try:
        for j in picks:
            scene,x,y,w,h=[int(v) for v in coords_sel[j]]
            rgb,_,_=r.region(scene,x,y,w,h);L,H,_=make_inputs(rgb,'bicubic')
            mosaic=np.vstack([np.hstack([H[0],H[1]]),np.hstack([H[2],H[3]])])
            left=np.asarray(Image.fromarray(L).resize((448,448),Image.Resampling.NEAREST))
            canvas=np.full((448,448*2+16,3),255,np.uint8);canvas[:,:448]=left;canvas[:,464:]=mosaic
            f=dst/f'visual_{patient.replace(":","_")}_{j}.png';Image.fromarray(canvas).save(f);files.append(f.name)
    finally:r.close()
    return files

# ---- A6000 additions: gating, provenance, weight verification, shared-GPU handling (no scientific effect) ----
def marker(f):return Path(f).with_name(Path(f).name+'.verified')  # written by the transfer lanes after sha256 verification
def case_inputs(row):
    """Transferred files one case needs: the CZI and the stored features of every encoder (for the L parity check)."""
    stem=Path(row['slide_filename']).stem
    return [ROOT/'data/he'/row['slide_filename']]+[STORED/stem/f'{nm}.npy' for nm in ENCODERS]
def missing_markers(files):
    """Transfer gate: the files without a .verified marker. Always [] without a prefix (HPC: data are local, no transfer)."""
    return [str(f) for f in files if not marker(f).exists()] if GATED else []

def require_gpu(device):
    import torch
    assert torch.cuda.is_available(),'GPU is required'
    d=torch.device(device);assert d.type=='cuda','GPU is required'
    if d.index is not None:torch.cuda.set_device(d)

def gpu_free_gib(device):
    import torch
    d=torch.device(device)
    return torch.cuda.mem_get_info(d if d.index is not None else None)[0]/1024**3 if d.type=='cuda' else None

OOM_MARKERS=('out of memory','CUBLAS_STATUS_ALLOC_FAILED','CUDNN_STATUS_ALLOC_FAILED','CUBLAS_STATUS_NOT_INITIALIZED')
def is_gpu_oom(e):
    """A CUDA allocation failure (another lab user's job took the shared card's memory): retry later, not a failure."""
    return type(e).__name__=='OutOfMemoryError' or (isinstance(e,RuntimeError) and any(s in str(e) for s in OOM_MARKERS))

def env_fingerprint():
    """Versions of everything that can change decoded pixels or features; no CUDA initialisation. Recorded in the
    stage-0 report; stage 1 puts it into its recipe and requires stage 0 to have passed under the same set."""
    import platform,torch,PIL
    def version(name):
        try:return getattr(importlib.import_module(name),'__version__',None)
        except Exception:return None
    return dict(python=platform.python_version(),torch=torch.__version__,torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                torchvision=version('torchvision'),timm=version('timm'),pillow=PIL.__version__,numpy=np.__version__,
                numpy_path=np.__file__,imagecodecs=version('imagecodecs'),czifile=version('czifile'))

def platform_info(device,query_gpu=True):
    import socket,torch
    d=torch.device(device)
    info=dict(env_fingerprint(),hostname=socket.gethostname(),device=str(d),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
              CUDA_DEVICE_ORDER=os.environ.get('CUDA_DEVICE_ORDER'),HE_PATH_PREFIX=PFX,HE_PROJECT_DIR=str(PROJECT),pid=os.getpid(),gpu=None)
    if query_gpu and d.type=='cuda':
        p=torch.cuda.get_device_properties(d)
        info.update(gpu=p.name,gpu_total_memory_gib=round(p.total_memory/1024**3,2),gpu_capability=f'{p.major}.{p.minor}')
    return info

def check_import_layout():
    """The patched czi_pilot_v2 (this dir), the patched validate.py (this dir) and the unchanged mirror full_patch_qc are the modules used."""
    import czi_pilot_v2,full_patch_qc
    got={'czi_pilot_v2.py':Path(czi_pilot_v2.__file__).resolve(),'full_patch_qc.py':Path(full_patch_qc.__file__).resolve(),'validate.py':VALIDATE.resolve()}
    want={'czi_pilot_v2.py':(HERE/'czi_pilot_v2.py').resolve(),'full_patch_qc.py':(ROOT/'bib_phase1/full_patch_qc.py').resolve(),'validate.py':(HERE/'validate.py').resolve()}
    assert got==want,f'Import layout differs from the A6000 design: {got} != {want}'
    assert czi_pilot_v2.ROOT==ROOT and full_patch_qc.ROOT==ROOT,(czi_pilot_v2.ROOT,ROOT)
    files={k:dict(path=str(v),sha256=sha(v)) for k,v in got.items()}
    for k,v in files.items():assert v['sha256']==PINNED[k],f'{k} sha256 {v["sha256"]} != pinned {PINNED[k]}'
    return files

def worker_check(expected):
    """Pool initializer, runs once in every spawn decode worker: the worker must import the very files (path and
    sha256) that check_import_layout() verified in the main process, else the pool breaks before any decode."""
    import czi_pilot_v2,full_patch_qc
    for k,m in (('czi_pilot_v2.py',czi_pilot_v2),('full_patch_qc.py',full_patch_qc)):
        p=Path(m.__file__).resolve()
        assert str(p)==expected[k]['path'] and sha(p)==expected[k]['sha256'],f'decode worker {os.getpid()} imports {k} from {p}; main process verified {expected[k]}'
    assert czi_pilot_v2.ROOT==ROOT==full_patch_qc.ROOT,(czi_pilot_v2.ROOT,ROOT)

def decode_pool(workers,modules):
    """The spawn decode pool (as in the original), with the per-worker module check."""
    return ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_check,initargs=(modules,))

def close_pool(pool,kill=False):
    """kill=True (after an error, a hung decode or a signal): terminate the workers first, so shutdown never blocks."""
    if pool is None:return
    if kill:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            try:p.terminate()
            except Exception:pass
    pool.shutdown(wait=not kill,cancel_futures=True)

def load_pathology():return module('pathology_adapters',VALIDATE)

class IntegrityError(RuntimeError):pass  # an input does not match its recorded sha256: FAIL (a human must look), not a crash
class WeightMismatch(IntegrityError):pass
def load_models(pathology,device='cuda'):
    """Build each encoder; its weight file sha256 must equal the HPC smoke summary before any extraction."""
    expected=json.loads(SMOKE.read_text())['models'];models={};infos={}
    for name in ENCODERS:
        model,_,info=pathology.build(name);weight=info.get('weight_path',info.get('weight'));got=sha(weight)  # prefixed: validate.build() builds it from the prefixed ASSETS/BIN
        want=expected[name]['weight_sha256']
        if got!=want:raise WeightMismatch(f'{name}: weight {weight} sha256 {got} != smoke summary {want}')
        infos[name]=dict(weight_path=str(weight),weight_sha256=got,weight_sha256_matches_smoke_summary=True)
        models[name]=model.eval().requires_grad_(False).to(device)
    return models,infos

def already_passed(cases,stems):
    """The recorded report if STAGE0_REPORT.json is a PASS made by this exact code for these cases and every per-case
    output still re-hashes to its complete.json; None otherwise (=> recompute)."""
    try:r=json.loads((OUT/'STAGE0_REPORT.json').read_text())
    except (OSError,ValueError):return None
    if r.get('state')!=PASS_STATE or r.get('code_sha256')!=sha(__file__) or r.get('cases')!=cases:return None
    for stem in stems:
        try:c=json.loads((OUT/stem/'complete.json').read_text())
        except (OSError,ValueError):return None
        if c.get('code_sha256')!=r['code_sha256']:return None
        for name,h in c.get('files',{}).items():
            if not (OUT/stem/name).is_file() or sha(OUT/stem/name)!=h:return None
    return r

def _terminate(signum,frame):raise SystemExit(128+signum)  # SIGTERM -> unwind: kill the decode workers, mark the report

# ---- shared extraction steps (moved out of main() unchanged so stage 1 imports them) ----
def embedder(pathology,models,device='cuda'):
    import torch
    dev=torch.device(device)
    mean=torch.tensor([.485,.456,.406],device=dev)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=dev)[None,:,None,None]
    def embed(name,arr):
        x=(torch.from_numpy(arr).permute(0,3,1,2).to(dev,dtype=torch.float32)/255.-mean)/std
        with torch.inference_mode(),torch.autocast(dev.type,dtype=torch.float16):
            y=pathology.forward(name,models[name],x).float()
        assert y.shape==(len(arr),ENCODERS[name][0]) and torch.isfinite(y).all()
        return y.half().cpu().numpy()
    embed.synchronize=(lambda:torch.cuda.synchronize(dev)) if dev.type=='cuda' else (lambda:None)
    return embed

def extract_locations(pool,slide,sel,embed,totals,hd_checked=False):
    """One native decode per location -> L (m,d), H (m,4,d), Hd (m,4,d) float16 features per encoder."""
    m=len(sel)
    feats={nm:dict(L=np.zeros((m,d),np.float16),H=np.zeros((m,4,d),np.float16),Hd=np.zeros((m,4,d),np.float16)) for nm,(d,_) in ENCODERS.items()}
    tasks=((slide,i,sel[i:i+BATCH]) for i in range(0,m,BATCH))
    t_wait=time.monotonic();sizes=[]
    for i,batch,native in buffered(pool,tasks):
        totals['decode_wait']+=time.monotonic()-t_wait;sizes+=native;t_gpu=time.monotonic()
        for nm,(d,interp) in ENCODERS.items():
            L,H,Hd=batch[interp];b=len(L)
            if not hd_checked:
                assert np.array_equal(quadrants_of_L(L[0],interp),Hd[0]);hd_checked=True
            feats[nm]['L'][i:i+b]=embed(nm,L)
            feats[nm]['H'][i:i+b]=embed(nm,H.reshape(-1,224,224,3)).reshape(b,4,d)
            feats[nm]['Hd'][i:i+b]=embed(nm,Hd.reshape(-1,224,224,3)).reshape(b,4,d)
        embed.synchronize();totals['gpu']+=time.monotonic()-t_gpu;totals['locations']+=len(native);t_wait=time.monotonic()
    return feats,sizes,hd_checked

def cosine_to_stored(stored_file,idx,L):
    stored=np.load(stored_file,mmap_mode='r')[idx].astype(np.float32);new=L.astype(np.float32)
    return (stored*new).sum(1)/np.linalg.norm(stored,axis=1)/np.linalg.norm(new,axis=1)

def geometry_record(patient,m,sizes,row):
    w=np.array([s[0] for s in sizes]);h=np.array([s[1] for s in sizes])
    mpp=float(row['mpp_x'])
    return dict(patient_id=patient,locations=m,native_w_range=[int(w.min()),int(w.max())],native_h_range=[int(h.min()),int(h.max())],
                mpp_x=mpp,mpp_y=float(row['mpp_y']),L_um_per_px=float(w.mean()*mpp/224),H_um_per_px=float(w.mean()/2*mpp/224))

def finish(report,state,code,**extra):
    """Record a non-PASS outcome in STAGE0_REPORT.json (a minimal report if the run stopped before writing one)."""
    r=report if report is not None else dict(stage=0,code_sha256=sha(__file__))
    r.update(state=state,exit_code=code,ended_unix=time.time(),**extra);save(OUT/'STAGE0_REPORT.json',r)
    print(state,json.dumps({k:(v[:300] if isinstance(v,str) else v) for k,v in extra.items()}),flush=True);return code

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description='High-res pilot stage 0 on lab-a6000 (4 locked cases, technical acceptance only).')
    ap.add_argument('--device',default='cuda',help="torch device, e.g. cuda:0 (default 'cuda' = current device, as on the HPC)")
    ap.add_argument('--workers',type=int,default=WORKERS,help='CZI decode processes (default %(default)s)')
    ap.add_argument('--min-free-gib',type=float,default=MIN_FREE_GIB,help='shared GPU: exit 3 (retry later) instead of loading models when less is free (default %(default)s)')
    ap.add_argument('--force',action='store_true',help='recompute although STAGE0_REPORT.json already records a PASS for this code')
    args=ap.parse_args(argv)
    assert not PFX or PFX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
    assert ROOT.is_dir(),f'{ROOT} does not exist (HE_PATH_PREFIX={PFX!r})'
    gate=TRANSFER/'verified_stage0'
    if GATED and not gate.exists():
        print(f'WAITING {gate} does not exist: stage-0 inputs not transferred and verified; refusing to start',flush=True);return WAITING
    RUN_LOCK.parent.mkdir(parents=True,exist_ok=True);run_lock=RUN_LOCK.open('a')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close();print(f'BUSY another stage-0 process holds {RUN_LOCK}; nothing done',flush=True);return EXIT_BUSY
    signal.signal(signal.SIGTERM,_terminate)
    ctx=dict(report=None,pool=None)
    try:return _run(args,gate,ctx)
    except IntegrityError as e:
        close_pool(ctx['pool'],kill=True);ctx['pool']=None
        return finish(ctx['report'],'FAIL',EXIT_FAIL,error=str(e))
    except Exception as e:
        close_pool(ctx['pool'],kill=True);ctx['pool']=None
        if is_gpu_oom(e):  # shared card: another user's job took the memory; retry later
            if ctx['report'] is None:print(json.dumps(dict(event='gpu_oom',device=args.device,error=str(e)[:300])),flush=True);return WAITING
            return finish(ctx['report'],'waiting_gpu_out_of_memory',WAITING,error=str(e)[:2000])
        tb=traceback.format_exc();print(tb,flush=True)
        if ctx['report'] is None:return EXIT_ERROR
        return finish(ctx['report'],'error',EXIT_ERROR,error=repr(e)[:2000],traceback=tb)
    except BaseException:  # SIGTERM (SystemExit) or Ctrl-C: stop the decode workers too, leave an honest report
        close_pool(ctx['pool'],kill=True);ctx['pool']=None
        if ctx['report'] is not None:finish(ctx['report'],'interrupted',None)
        raise
    finally:
        close_pool(ctx['pool']);run_lock.close()

def _run(args,gate,ctx):
    manifest=POOL/'patient_manifest.csv';got=sha(manifest)
    want=json.loads((POOL/'LOCK.json').read_text()).get('patient_manifest_sha256')
    if got!=want:raise IntegrityError(f'{manifest} sha256 {got} != {POOL/"LOCK.json"} patient_manifest_sha256 {want}')
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    lookup={r['patient_id']:r for r in rows};cases=select_cases(rows)
    missing=missing_markers([f for p in cases for f in case_inputs(lookup[p])])
    if missing:
        print('WAITING unverified stage-0 inputs (no .verified marker): '+json.dumps(missing),flush=True);return WAITING
    if not args.force and already_passed(cases,[Path(lookup[p]['slide_filename']).stem for p in cases]) is not None:
        print(f'{PASS_STATE} already recorded for this code in {OUT/"STAGE0_REPORT.json"}; outputs re-hashed OK, nothing recomputed (--force to redo)',flush=True)
        return EXIT_OK
    import torch
    torch.set_num_threads(2);torch.manual_seed(SEED)
    require_gpu(args.device)
    free=gpu_free_gib(args.device)
    if free is not None and free<args.min_free_gib:
        print(json.dumps(dict(event='gpu_busy',device=args.device,free_gib=round(free,1),need_gib=args.min_free_gib)),flush=True);return WAITING
    t_start=time.monotonic()
    report=ctx['report']=dict(state='running',stage=0,job_id=os.getenv('SLURM_JOB_ID'),scheduler_job=os.getenv('HE_SCHED_JOB'),code_sha256=sha(__file__),
                cases=cases,locations_per_case=LOCATIONS,encoders={k:dict(dim=v[0],interpolation=v[1]) for k,v in ENCODERS.items()},checks={},timing={},
                pool_lock_sha256=sha(POOL/'LOCK.json'),manifest_sha256=got,platform=platform_info(args.device),modules=check_import_layout(),
                transfer_gate=str(gate) if GATED else None,workers=args.workers,gpu_free_gib_at_start=None if free is None else round(free,2),
                started_unix=time.time(),inputs_verified={str(f):True for p in cases for f in case_inputs(lookup[p])} if GATED else None)
    OUT.mkdir(parents=True,exist_ok=True);save(OUT/'STAGE0_REPORT.json',report)
    pathology=load_pathology()
    models,infos=load_models(pathology,args.device)  # WeightMismatch -> FAIL (exit 1) in main()
    for name in ENCODERS:report['encoders'][name].update(infos[name])
    report['weights_reference']=dict(path=str(SMOKE),sha256=sha(SMOKE));save(OUT/'STAGE0_REPORT.json',report)
    embed=embedder(pathology,models,args.device)
    totals=dict(locations=0,decode_wait=0.,gpu=0.)
    sims={n:[] for n in ENCODERS};geometry=[];hd_checked=False
    pool=ctx['pool']=decode_pool(args.workers,report['modules'])
    for patient in cases:
        t_case=time.monotonic()
        row=lookup[patient];stem=Path(row['slide_filename']).stem;dst=OUT/stem;dst.mkdir(parents=True,exist_ok=True)
        assert sha(hpc(row['primary_path']))==row['primary_sha256']
        with np.load(hpc(row['primary_path']),allow_pickle=False) as z:coords=z['coords'].copy()
        n=int(row['primary_count']);assert len(coords)==n
        idx=location_indices(n,patient);sel=coords[idx];m=len(idx)
        np.save(dst/'location_indices.npy',idx);np.save(dst/'coords.npy',sel)
        slide=str(ROOT/'data/he'/row['slide_filename'])
        feats,sizes,hd_checked=extract_locations(pool,slide,sel,embed,totals,hd_checked)
        hashes={}
        for nm,parts in feats.items():
            for arm,a in parts.items():
                assert np.isfinite(a).all();f=dst/f'{nm}_{arm}.npy';np.save(f,a);hashes[f.name]=sha(f)
            sims[nm].append(cosine_to_stored(STORED/stem/f'{nm}.npy',idx,parts['L']))
        geometry.append(geometry_record(patient,m,sizes,row))
        save(dst/'complete.json',dict(state='complete',patient_id=patient,n_locations=m,files=hashes,code_sha256=report['code_sha256']))
        report['visual_files']=report.get('visual_files',[])+visual(dst,patient,sel,slide,np.linspace(0,m-1,4,dtype=int).tolist())
        print(json.dumps(dict(event='case',case=stem,locations=m,seconds=round(time.monotonic()-t_case,1),
                              L_min_cosine={nm:float(v[-1].min()) for nm,v in sims.items()})),flush=True)
    close_pool(pool);ctx['pool']=None
    cos_all={nm:np.concatenate(v) for nm,v in sims.items()}
    report['checks']=dict(
        L_reproduces_stored={nm:dict(min_cosine=float(c.min()),mean_cosine=float(c.mean()),pass_=bool(c.min()>=.999)) for nm,c in cos_all.items()},
        geometry=geometry,subtiles_union_exact=all(check_boxes(g) for g in [(581,581),(580,581),(582,583)]),
        Hd_equals_L_quadrants=hd_checked)
    loc_s=totals['locations']/(time.monotonic()-t_start)
    stage1_locations=120*LOCATIONS
    report['timing']=dict(**totals,wall_seconds=time.monotonic()-t_start,locations_per_second_overall=loc_s,
                          gpu_seconds_per_location=totals['gpu']/totals['locations'],
                          stage1_estimate_hours=stage1_locations/loc_s/3600,stage1_with_50pct_margin_hours=1.5*stage1_locations/loc_s/3600)
    ok=all(v['pass_'] for v in report['checks']['L_reproduces_stored'].values()) and report['checks']['subtiles_union_exact'] and hd_checked
    report['state']=PASS_STATE if ok else 'FAIL'
    report.update(exit_code=EXIT_OK if ok else EXIT_FAIL,ended_unix=time.time())
    save(OUT/'STAGE0_REPORT.json',report);print(report['state'],json.dumps(report['timing']),flush=True)
    return EXIT_OK if ok else EXIT_FAIL

def quadrants_of_L(L,interp):
    """Independent recomputation of Hd from a finished L image."""
    from PIL import Image
    mode={'bilinear':Image.Resampling.BILINEAR,'bicubic':Image.Resampling.BICUBIC}[interp]
    Lim=Image.fromarray(L)
    return np.stack([np.asarray(Lim.crop(b).resize((224,224),mode)) for b in subtile_boxes(224,224)])

def check_boxes(wh):
    w,h=wh;mask=np.zeros((h,w),int)
    for x0,y0,x1,y1 in subtile_boxes(w,h):mask[y0:y1,x0:x1]+=1
    return bool((mask==1).all())

if __name__=='__main__':sys.exit(main())
