"""High-resolution pilot, stage 0: technical acceptance only. No training, no labels beyond case selection.

Same 128 um locations as the locked pool; one native decode per location yields three inputs:
  L  : full 128 um crop resized to 224 (current pipeline, ~0.57 um/px)
  H  : 2x2 physical sub-tiles (64 um) each resized to 224 (~0.29 um/px)
  Hd : 2x2 quadrants of the L image, each upsampled to 224 (H geometry, L detail)
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,csv,json,hashlib,time,importlib.util,collections,multiprocessing
from pathlib import Path
ROOT=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
sys.path.insert(0,str(ROOT/'qc_deps'));sys.path.insert(0,str(ROOT/'bib_phase1'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
POOL=ROOT/'qc/primary_technical_pool_v1_20261010'
STORED=ROOT/'features/nine_encoder_full_v1_20261010'
OUT=ROOT/'features/highres_pilot_v1_20261021'
OLD=Path('/data/cluster/projects/gastritis_uni2h/outputs')
ENCODERS={'uni':(1024,'bilinear'),'virchow2':(2560,'bicubic')}  # interpolation as in extract_nine_korea.py
LOCATIONS=2048;SEED=20261021;STAGE0_G1=2;STAGE0_G2=2;BATCH=32;WORKERS=6
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

def main():
    import torch
    torch.set_num_threads(2);torch.manual_seed(SEED)
    assert torch.cuda.is_available(),'GPU is required'
    t_start=time.monotonic()
    rows=list(csv.DictReader((POOL/'patient_manifest.csv').open(encoding='utf-8-sig')))
    lookup={r['patient_id']:r for r in rows};cases=select_cases(rows)
    report=dict(state='running',stage=0,job_id=os.getenv('SLURM_JOB_ID'),code_sha256=sha(__file__),pool_lock_sha256=sha(POOL/'LOCK.json'),
                cases=cases,locations_per_case=LOCATIONS,encoders={k:dict(dim=v[0],interpolation=v[1]) for k,v in ENCODERS.items()},checks={},timing={})
    save(OUT/'STAGE0_REPORT.json',report)
    pathology=module('pathology_adapters',OLD/'encoder-validation-20261008/validate.py')
    models={}
    for name in ENCODERS:
        model,_,info=pathology.build(name);report['encoders'][name]['weight_sha256']=sha(info.get('weight_path',info.get('weight')))
        models[name]=model.eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    def embed(name,arr):
        x=(torch.from_numpy(arr).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255.-mean)/std
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            y=pathology.forward(name,models[name],x).float()
        assert y.shape==(len(arr),ENCODERS[name][0]) and torch.isfinite(y).all()
        return y.half().cpu().numpy()
    totals=dict(locations=0,decode_wait=0.,gpu=0.)
    sims={n:[] for n in ENCODERS};geometry=[];hd_checked=False
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=ctx) as pool:
        for patient in cases:
            row=lookup[patient];stem=Path(row['slide_filename']).stem;dst=OUT/stem;dst.mkdir(parents=True,exist_ok=True)
            assert sha(row['primary_path'])==row['primary_sha256']
            with np.load(row['primary_path'],allow_pickle=False) as z:coords=z['coords'].copy()
            n=int(row['primary_count']);assert len(coords)==n
            idx=location_indices(n,patient);sel=coords[idx];m=len(idx)
            np.save(dst/'location_indices.npy',idx);np.save(dst/'coords.npy',sel)
            feats={nm:dict(L=np.zeros((m,d),np.float16),H=np.zeros((m,4,d),np.float16),Hd=np.zeros((m,4,d),np.float16)) for nm,(d,_) in ENCODERS.items()}
            slide=str(ROOT/'data/he'/row['slide_filename']);tasks=((slide,i,sel[i:i+BATCH]) for i in range(0,m,BATCH))
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
                torch.cuda.synchronize();totals['gpu']+=time.monotonic()-t_gpu;totals['locations']+=len(native);t_wait=time.monotonic()
            hashes={}
            for nm,parts in feats.items():
                for arm,a in parts.items():
                    assert np.isfinite(a).all();f=dst/f'{nm}_{arm}.npy';np.save(f,a);hashes[f.name]=sha(f)
                stored=np.load(STORED/stem/f'{nm}.npy',mmap_mode='r')[idx].astype(np.float32);new=parts['L'].astype(np.float32)
                cos=(stored*new).sum(1)/np.linalg.norm(stored,axis=1)/np.linalg.norm(new,axis=1);sims[nm].append(cos)
            w=np.array([s[0] for s in sizes]);h=np.array([s[1] for s in sizes])
            mpp=float(row['mpp_x'])
            geometry.append(dict(patient_id=patient,locations=m,native_w_range=[int(w.min()),int(w.max())],native_h_range=[int(h.min()),int(h.max())],
                                 mpp_x=mpp,mpp_y=float(row['mpp_y']),L_um_per_px=float(w.mean()*mpp/224),H_um_per_px=float(w.mean()/2*mpp/224)))
            save(dst/'complete.json',dict(state='complete',patient_id=patient,n_locations=m,files=hashes,code_sha256=report['code_sha256']))
            report['visual_files']=report.get('visual_files',[])+visual(dst,patient,sel,slide,np.linspace(0,m-1,4,dtype=int).tolist())
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
    report['state']='PASS_PENDING_HUMAN_VISUAL_REVIEW' if ok else 'FAIL'
    save(OUT/'STAGE0_REPORT.json',report);print(report['state'],json.dumps(report['timing']),flush=True)
    return 0 if ok else 1

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
