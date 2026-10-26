"""Scotland external cohort: frozen nine-encoder extraction, byte-for-byte the Korean method (one CZI decode, identical 128 um FOV,
same resize/normalization/precision), resumable outputs. Only paths differ. No labels are read."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys, csv, json, hashlib, time, argparse, importlib.util, fcntl, shutil, collections, multiprocessing
from pathlib import Path
ROOT=Path('/data/cluster/projects/scottish_ccrcc_SBIAD2454')
KOREA=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
sys.path.insert(0,str(KOREA/'qc_deps'));sys.path.insert(0,str(KOREA/'bib_phase1'))  # same CZI reader as Korea
import numpy as np
from concurrent.futures import ProcessPoolExecutor
POOL=ROOT/'qc/external_primary_technical_pool_v1_20261013'
OLD=Path('/data/cluster/projects/gastritis_uni2h/outputs')
DIMS={'resnet50':2048,'convnext_tiny':768,'vit_b_16':768,'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
_reader=None;_path=None

def pause_required():
    stop_at=float(os.environ.get('NINE_STOP_AT','inf'))
    return time.time()>=stop_at

def pause_report(path,report):
    report['state']='paused_for_walltime'
    report['pause_time_unix']=time.time()
    save(path,report)
    print('SAFE_PAUSE_FOR_WALLTIME',flush=True)
    return 75

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(p,r):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False));tmp.replace(p)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    import torch
    torch.set_num_threads(2);torch.manual_seed(20261010)
    assert torch.cuda.is_available(),'GPU is required'
    # Scotland's pool has no export_verification.json; check its own lock and summary instead.
    pool_lock=json.loads((POOL/'LOCK.json').read_text());pool_summary=json.loads((POOL/'summary.json').read_text())
    assert pool_lock['version']=='external_primary_technical_pool_v1_20261013'
    assert sha(POOL/'patient_manifest.csv')==pool_lock['patient_manifest_sha256']
    assert pool_summary['patients']==120 and pool_summary['primary_patches']==1927352
    out=ROOT/'features'/('nine_encoder_smoke_v1_20261021' if args.smoke else 'nine_encoder_full_v1_20261021')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'shard{args.shard}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code_sha=sha(__file__);input_sha=sha(POOL/'LOCK.json')
    rows=sorted(csv.DictReader((POOL/'patient_manifest.csv').open()),key=lambda r:r['patient_id'])
    if args.smoke:rows=[rows[i] for i in [0,len(rows)//2,len(rows)-1]]
    else:
        check=json.loads((ROOT/'features/nine_encoder_smoke_v1_20261021/summary_shard0.json').read_text())
        assert check['state']=='complete' and check['code_sha256']==code_sha and check['input_lock_sha256']==input_sha
        assert check['preprocessing']=='same_128um_fov_resize_only_v1'
        rows=rows[args.shard::args.shards]
    pathology=module('pathology_adapters',OLD/'encoder-validation-20261008/validate.py')
    generic=module('generic_adapters',OLD/'generic-encoders-20261008/code/generic_encoders.py')
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
    report=dict(state='running',job_id=os.getenv('SLURM_JOB_ID'),array_id=os.getenv('SLURM_ARRAY_TASK_ID'),
                gpu=torch.cuda.get_device_name(),args=vars(args),**expected_input,models=infos,completed=[],
                precision='AMP bf16 for UNI2-h/Phikon-v2, fp16 for others except float32 CONCH; float16 storage',
                geometry_note='Standardized full 128um crop resized to input size; official mean/std and interpolation retained. No center-crop FOV shrink. This differs from official resize-then-crop evaluation recipes for some models.')
    reportfile=out/f'summary_shard{args.shard}.json';save(reportfile,report)
    imagenet_mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    imagenet_std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    conch_mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    conch_std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
        for row in rows:
            if pause_required():return pause_report(reportfile,report)
            dst=out/Path(row['slide_filename']).stem;dst.mkdir(exist_ok=True)
            completion=dst/'complete.json';cp=Path(row['primary_path'])
            assert sha(cp)==row['primary_sha256']
            if completion.exists():
                previous=json.loads(completion.read_text())
                assert all(previous[k]==v for k,v in expected_input.items()),'Resume configuration changed'
                assert previous['primary_coords_sha256']==row['primary_sha256']
                for name in DIMS:assert sha(dst/(name+'.npy'))==previous['features'][name]['sha256']
                report['completed'].append(previous);save(reportfile,report);continue
            with np.load(cp,allow_pickle=False) as ar:coords=ar['coords'].copy()
            if args.smoke:coords=coords[np.linspace(0,len(coords)-1,64,dtype=int)]
            n=len(coords);coord_hash=hashlib.sha256(coords.tobytes()).hexdigest()
            np.save(dst/'coords.npy',coords)
            checkpoint=dst/'checkpoint.json';next_index=0
            if checkpoint.exists():
                ck=json.loads(checkpoint.read_text())
                assert all(ck[k]==v for k,v in expected_input.items()) and ck['coords_array_sha256']==coord_hash
                next_index=ck['next_index']
            else:
                required=sum(n*d*2 for d in DIMS.values())
                assert shutil.disk_usage(out).free>required+20*1024**3,'Keep 20 GiB free; extraction paused before allocating'
            arrays={}
            for name,dim in DIMS.items():
                fp=dst/(name+'.npy.partial')
                if not fp.exists() and (dst/(name+'.npy')).exists():
                    # Recover a crash after final renames but before complete.json.
                    (dst/(name+'.npy')).replace(fp)
                arrays[name]=np.lib.format.open_memmap(fp,mode='r+' if next_index else 'w+',dtype=np.float16,shape=(n,dim))
                assert arrays[name].shape==(n,dim) and arrays[name].dtype==np.float16
            tasks=((str(ROOT/'data/he'/row['slide_filename']),i,coords[i:i+args.batch]) for i in range(next_index,n,args.batch))
            began=time.monotonic();processed_here=0
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
                pausing=pause_required()
                if next_index%1024==0 or next_index==n or pausing:
                    for ar in arrays.values():ar.flush()
                    save(checkpoint,dict(**expected_input,coords_array_sha256=coord_hash,next_index=next_index,total=n))
                    print('PROGRESS',row['patient_id'],next_index,n,'all_nine_patches_per_sec',round(processed_here/(time.monotonic()-began),2),flush=True)
                if pausing and next_index<n:return pause_report(reportfile,report)
            features={}
            for name,ar in arrays.items():
                ar.flush();assert np.isfinite(ar).all()
                assert np.any(np.asarray(ar[:min(64,n)],dtype=np.float32).std(axis=0)>1e-6)
                target=dst/(name+'.npy');(dst/(name+'.npy.partial')).replace(target)
                features[name]=dict(shape=list(ar.shape),dtype='float16',sha256=sha(target))
            done=dict(**expected_input,patient_id=row['patient_id'],slide_filename=row['slide_filename'],source_sha256=row['source_sha256'],
                primary_coords_sha256=row['primary_sha256'],coords_array_sha256=coord_hash,n=n,features=features,
                seconds_this_attempt=time.monotonic()-began,state='complete')
            save(completion,done);report['completed'].append(done);save(reportfile,report)
            print('COMPLETE',row['patient_id'],n,flush=True)
    report['state']='complete';save(reportfile,report)
    print('SHARD_COMPLETE',args.shard,len(rows),flush=True)

if __name__=='__main__':sys.exit(main() or 0)
