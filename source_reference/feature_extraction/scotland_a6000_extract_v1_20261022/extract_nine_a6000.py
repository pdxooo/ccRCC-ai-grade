"""lab-a6000 port of the HPC Scottish nine-encoder extraction (scotland_nine_extract_v1_20261021/extract_nine_scotland.py,
sha256 9e6368ef...), used for the cases the HPC shards have not reached. The per-case method is the HPC code line for line:
one CZI decode through the same IndexedReader, identical 128 um field and resizes, the same encoders built by the same
adapters from weight files whose signature must equal the Korean extraction's, the same batch size, AMP precision and
float16 storage, the same per-encoder self-checks. Changes (no effect on the computation): every HPC path carries the
mirror prefix; cases come from a priority list instead of a shard (claimed with mkdir, skipped when the HPC has finished
or is working on them); the CZI and primary coordinates are checked against the manifest sha256 first; outputs are
staged, then installed into the mirror case folder with .verified markers and a provenance record.
Modes:
  --parity CASE [CASE ...] --out DIR   extract cases that the HPC has finished into DIR (compare with compare_parity.py)
  --work                               take cases from the order list until none is left
Exit: 0 done; 1 integrity failure; 2 usage; 3 nothing to do yet (inputs still arriving)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys, csv, json, hashlib, time, argparse, importlib.util, shutil, collections, multiprocessing, subprocess, socket
from pathlib import Path
PFX=os.environ['HE_PATH_PREFIX'].rstrip('/');PROJECT=Path(os.environ.get('HE_PROJECT_DIR','/data/workstation/he_ccrcc_isup_20261021'))
assert PFX.startswith('/'),'HE_PATH_PREFIX must be the mirror root'
ROOT=Path(PFX+'/data/cluster/projects/scottish_ccrcc_SBIAD2454')
KOREA=Path(PFX+'/data/cluster/projects/korean_ccrcc_SBIAD2452')
sys.path.insert(0,str(KOREA/'qc_deps'));sys.path.insert(0,str(KOREA/'bib_phase1'))  # same CZI reader as Korea
sys.path.insert(0,PFX+'/data/cluster/software/he-six-20261008/lib/python3.10/site-packages')  # only conch/ is mirrored there
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
POOL=ROOT/'qc/external_primary_technical_pool_v1_20261013'
FEAT=ROOT/'features/nine_encoder_full_v1_20261021'  # mirror of the HPC output folder
STAGE=PROJECT/'scotland_a6000_extract'
VALIDATE=PROJECT/'code/highres_a6000/validate.py'  # prefix-patched validate.py used for stage 0 (parity PASS)
GENERIC=HERE/'generic_encoders_a6000.py'          # prefix-patched generic_encoders.py
HPC_CODE_SHA='9e6368efe69183ff99cc9957fe25a374a8a80454332202ff65b287e81d5ecbce'
HPC_FEAT='/data/cluster/projects/scottish_ccrcc_SBIAD2454/features/nine_encoder_full_v1_20261021'
DIMS={'resnet50':2048,'convnext_tiny':768,'vit_b_16':768,'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
_reader=None;_path=None

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,r):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def mirror(p):return Path(PFX+p) if str(p).startswith('/work/') else Path(p)

def decode(task):
    global _reader,_path
    from full_patch_qc import IndexedReader
    from PIL import Image
    path,start,coords=task
    if _path!=path:
        if _reader is not None:_reader.close()
        _reader=IndexedReader(path);_path=path
    bilinear=[];bicubic=[];large=[]
    for scene,x,y,w,h in coords:
        rgb,cov,_=_reader.region(int(scene),int(x),int(y),int(w),int(h))
        assert float(cov.mean())>=.999,'QC coordinate coverage changed'
        im=Image.fromarray(rgb)
        bilinear.append(np.asarray(im.resize((224,224),Image.Resampling.BILINEAR)))
        bicubic.append(np.asarray(im.resize((224,224),Image.Resampling.BICUBIC)))
        large.append(np.asarray(im.resize((448,448),Image.Resampling.BICUBIC)))
    return start,np.stack(bilinear),np.stack(bicubic),np.stack(large)

def buffered(pool,tasks,depth=8):
    it=iter(tasks);pending=collections.deque()
    for _ in range(depth):
        try:pending.append(pool.submit(decode,next(it)))
        except StopIteration:break
    while pending:
        result=pending.popleft().result()
        try:pending.append(pool.submit(decode,next(it)))
        except StopIteration:pass
        yield result

def hpc_state(stem):
    """'done' when the HPC has complete.json for the case, 'busy' when it has a checkpoint (working on it), else 'free'."""
    q="d='%s/%s'; if [ -e \"$d/complete.json\" ]; then echo done; elif [ -e \"$d/checkpoint.json\" ]; then echo busy; else echo free; fi"%(HPC_FEAT,stem)
    r=subprocess.run(['ssh','-F',str(PROJECT/'transfer/ssh_config'),'-o',f'ControlPath={PROJECT}/transfer/cm-hd4','hd4',q],capture_output=True,text=True,timeout=120)
    out=[l for l in r.stdout.split() if l in ('done','busy','free')]
    if r.returncode!=0 or not out:raise RuntimeError(f'HPC state check failed for {stem}: {r.stderr[-300:]}')
    return out[-1]

def install(stage,dst,done):
    """Copy a finished staged case into the mirror folder: features, coords, .verified markers, provenance, complete.json last."""
    dst.mkdir(parents=True,exist_ok=True)
    assert not (dst/'complete.json').exists(),f'{dst} already has complete.json'
    for name in DIMS:
        tmp=dst/(name+'.npy.installing');shutil.copyfile(stage/(name+'.npy'),tmp)
        assert sha(tmp)==done['features'][name]['sha256'];tmp.replace(dst/(name+'.npy'));(dst/(name+'.npy.verified')).touch()
    shutil.copyfile(stage/'coords.npy',dst/'coords.npy')
    save(dst/'a6000_provenance.json',dict(extracted_on='lab-a6000',staged=str(stage),installed=time.strftime('%F %T'),complete=done))
    save(dst/'complete.json',done)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parity',nargs='*');ap.add_argument('--out');ap.add_argument('--work',action='store_true')
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if bool(args.parity)==bool(args.work) or (args.parity and not args.out):print(__doc__);return 2
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261010)
    assert torch.cuda.is_available(),'GPU is required'
    pool_lock=json.loads((POOL/'LOCK.json').read_text());pool_summary=json.loads((POOL/'summary.json').read_text())
    assert pool_lock['version']=='external_primary_technical_pool_v1_20261013'
    assert sha(POOL/'patient_manifest.csv')==pool_lock['patient_manifest_sha256']
    assert pool_summary['patients']==120 and pool_summary['primary_patches']==1927352
    assert args.batch==32,'the HPC extraction used batch 32'
    code_sha=sha(__file__);input_sha=sha(POOL/'LOCK.json')
    rows={r['patient_id']:r for r in csv.DictReader((POOL/'patient_manifest.csv').open())}
    pathology=module('pathology_adapters',VALIDATE)
    generic=module('generic_adapters',GENERIC)
    models={};infos={};checked=set()
    for name in DIMS:
        print('LOAD',name,flush=True)
        model,_,info=generic.build(name) if name in generic.SPECS else pathology.build(name)
        p=info.get('weight_path',info.get('weight'));info['weight_sha256']=sha(p)
        infos[name]=info
        models[name]=model.eval().requires_grad_(False).cuda()
        assert not any(p.requires_grad for p in models[name].parameters())
    assert torch.cuda.memory_allocated()<torch.cuda.get_device_properties(0).total_memory*.7
    signature=hashlib.sha256(json.dumps({n:i['weight_sha256'] for n,i in infos.items()},sort_keys=True).encode()).hexdigest()
    korea=json.loads((KOREA/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json').read_text())
    assert signature==korea['weights_signature'],'Encoder weights differ from the Korean extraction'
    expected_input={'code_sha256':code_sha,'input_lock_sha256':input_sha,'weights_signature':signature,'preprocessing':'same_128um_fov_resize_only_v1'}
    import timm,transformers,PIL,torchvision
    platform=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__,timm=timm.__version__,transformers=transformers.__version__,torchvision=torchvision.__version__,pillow=PIL.__version__)
    imagenet_mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    imagenet_std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    conch_mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    conch_std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    ctx=multiprocessing.get_context('spawn')

    def extract(row,dst):
        dst.mkdir(parents=True,exist_ok=True);completion=dst/'complete.json';cp=mirror(row['primary_path'])
        assert sha(cp)==row['primary_sha256'],f"{row['patient_id']}: primary coordinates sha256 differs from the manifest"
        czi=ROOT/'data/he'/row['slide_filename'];assert sha(czi)==row['source_sha256'],f"{row['patient_id']}: CZI sha256 differs from the manifest"
        if completion.exists():
            previous=json.loads(completion.read_text())
            assert all(previous[k]==v for k,v in expected_input.items()) and previous['primary_coords_sha256']==row['primary_sha256']
            for name in DIMS:assert sha(dst/(name+'.npy'))==previous['features'][name]['sha256']
            return previous
        with np.load(cp,allow_pickle=False) as ar:coords=ar['coords'].copy()
        n=len(coords);coord_hash=hashlib.sha256(coords.tobytes()).hexdigest()
        np.save(dst/'coords.npy',coords)
        checkpoint=dst/'checkpoint.json';next_index=0
        if checkpoint.exists():
            ck=json.loads(checkpoint.read_text())
            assert all(ck[k]==v for k,v in expected_input.items()) and ck['coords_array_sha256']==coord_hash
            next_index=ck['next_index']
        else:
            required=sum(n*d*2 for d in DIMS.values())
            assert shutil.disk_usage(dst).free>required+20*1024**3,'Keep 20 GiB free; extraction paused before allocating'
        arrays={}
        for name,dim in DIMS.items():
            fp=dst/(name+'.npy.partial')
            if not fp.exists() and (dst/(name+'.npy')).exists():(dst/(name+'.npy')).replace(fp)
            arrays[name]=np.lib.format.open_memmap(fp,mode='r+' if next_index else 'w+',dtype=np.float16,shape=(n,dim))
            assert arrays[name].shape==(n,dim) and arrays[name].dtype==np.float16
        tasks=((str(czi),i,coords[i:i+args.batch]) for i in range(next_index,n,args.batch))
        began=time.monotonic();processed_here=0
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
            for i,a,b,c in buffered(pool,tasks):
                assert i==next_index
                tensors={}
                for key,arr,mean,std in [('bilinear',a,imagenet_mean,imagenet_std),('bicubic',b,imagenet_mean,imagenet_std),('conch',c,conch_mean,conch_std)]:
                    tensors[key]=(torch.from_numpy(arr).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255.-mean)/std
                for name,model in models.items():
                    x=tensors['conch' if name=='conch' else 'bicubic' if name in ('phikon-v2','virchow2','gigapath') else 'bilinear']
                    amp=torch.bfloat16 if name in ('uni2-h','phikon-v2') else torch.float16
                    def forward():return model(x) if name in generic.SPECS else pathology.forward(name,model,x)
                    with torch.inference_mode(),torch.autocast('cuda',enabled=name!='conch',dtype=amp):
                        y=forward().float()
                        if name not in checked:
                            repeat=forward().float();assert torch.allclose(y,repeat,rtol=1e-4,atol=1e-4)
                    assert y.shape==(len(a),DIMS[name]) and torch.isfinite(y).all() and (y.norm(dim=1)>0).all()
                    z=y.half();assert torch.isfinite(z).all()
                    if name not in checked:
                        assert torch.nn.functional.cosine_similarity(y,z.float(),dim=1).min()>.9999
                        checked.add(name)
                    arrays[name][i:i+len(a)]=z.cpu().numpy()
                next_index=i+len(a);processed_here+=len(a)
                if next_index%1024==0 or next_index==n:
                    for ar in arrays.values():ar.flush()
                    save(checkpoint,dict(**expected_input,coords_array_sha256=coord_hash,next_index=next_index,total=n))
                    print('PROGRESS',row['patient_id'],next_index,n,'all_nine_patches_per_sec',round(processed_here/(time.monotonic()-began),2),flush=True)
        features={}
        for name,ar in arrays.items():
            ar.flush();assert np.isfinite(ar).all()
            assert np.any(np.asarray(ar[:min(64,n)],dtype=np.float32).std(axis=0)>1e-6)
            target=dst/(name+'.npy');(dst/(name+'.npy.partial')).replace(target)
            features[name]=dict(shape=list(ar.shape),dtype='float16',sha256=sha(target))
        done=dict(**expected_input,patient_id=row['patient_id'],slide_filename=row['slide_filename'],source_sha256=row['source_sha256'],
                  primary_coords_sha256=row['primary_sha256'],coords_array_sha256=coord_hash,n=n,features=features,
                  seconds_this_attempt=time.monotonic()-began,state='complete',extracted_on='lab-a6000',platform=platform,
                  reference_hpc_code_sha256=HPC_CODE_SHA)
        save(completion,done);print('COMPLETE',row['patient_id'],n,flush=True);return done

    if args.parity:
        out=Path(args.out)
        for p in args.parity:extract(rows[p],out/Path(rows[p]['slide_filename']).stem)
        print('PARITY_EXTRACTED',len(args.parity),flush=True);return 0
    order=[l.strip() for l in (PROJECT/'transfer/lists/a6000_extract.order').read_text().split() if l.strip().startswith('scotland:')][2:]  # minus the parity cases
    claims=STAGE/'claims';claims.mkdir(parents=True,exist_ok=True);waiting=True
    while waiting:
        waiting=False;did=False
        for p in order:
            row=rows[p];stem=Path(row['slide_filename']).stem;dst=FEAT/stem
            if (dst/'complete.json').exists() or (claims/stem/'done').exists():continue
            if not (ROOT/'data/he'/row['slide_filename']).exists() or not mirror(row['primary_path']).exists():waiting=True;continue  # still arriving
            try:os.mkdir(claims/stem)
            except FileExistsError:  # another GPU has it, unless its owner died before finishing (then resume its checkpoint)
                own=claims/stem/'owner';pid=int(own.read_text()) if own.exists() else None
                if any((claims/stem).glob('skipped_*')) or (claims/stem/'hpc_finished_first').exists() or pid is None:continue
                try:os.kill(pid,0);continue
                except ProcessLookupError:print('RESUME',p,'owner',pid,'is gone',flush=True)
            (claims/stem/'owner').write_text(str(os.getpid()))
            state=hpc_state(stem)
            if state!='free':(claims/stem/f'skipped_hpc_{state}').touch();print('SKIP',p,'HPC',state,flush=True);continue
            done=extract(row,STAGE/stem)
            if hpc_state(stem)=='done':(claims/stem/'hpc_finished_first').touch();print('HPC_FINISHED_FIRST',p,'(staged copy kept, not installed)',flush=True)
            else:install(STAGE/stem,dst,done);print('INSTALLED',p,flush=True)
            (claims/stem/'done').touch();did=True
        if waiting and not did:time.sleep(60)
    print('WORK_DONE',flush=True);return 0

if __name__=='__main__':sys.exit(main() or 0)
