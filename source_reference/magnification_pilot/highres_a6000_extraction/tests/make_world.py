"""Build a fake lab-a6000 world for local CPU tests:
  <world>/hpcroot/...          = HE_PATH_PREFIX mirror of the HPC paths (manifest, primary coords, dummy CZIs + receipts,
                                 stored uni/virchow2 features computed with the fake encoders, smoke summary, weights,
                                 qc_deps/czifile stub, bib_phase1/{czi_pilot_v2.py (original bytes), full_patch_qc.py})
  <world>/project/transfer/... = HE_PROJECT_DIR (lists/STAGE1_CASES_LOCK.json, lists/stage1_cases.txt); no .verified markers
Usage: make_world.py <world_dir> fake|real   ('real' installs the unchanged mirror full_patch_qc.py and skips features)"""
import os,sys,json,csv,hashlib,shutil
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent
K='/data/cluster/projects/korean_ccrcc_SBIAD2452'
# (patient_id, ISUP grade, number of primary coordinates, native crop size)
PATIENTS=[('korea:KR901','1',40,121),('korea:KR902','1',33,120),('korea:KR903','2',40,122),('korea:KR904','2',20,121),('korea:KR905','1',37,121)]
STAGE1=[('korea:KR905',0,1),('korea:KR903',0,2)]  # (patient, fold, grade) in lock order
COLUMNS='patient_id,clinical_id,slide_filename,ISUP_grade,source_sha256,candidate_count,primary_count,review_count,excluded_count,mpp_x,mpp_y,field_um,label_scope,eligibility,primary_path,primary_sha256,review_reserve_path,review_reserve_sha256,excluded_path,excluded_sha256'.split(',')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stem_of=lambda pid:'HE_'+pid.split(':')[1]

def main(world,variant):
    world=Path(world).resolve();pfx=world/'hpcroot';root=Path(str(pfx)+K);project=world/'project'
    if world.exists():shutil.rmtree(world)
    for d in ['qc_deps','bib_phase1','qc/primary_technical_pool_v1_20261010/primary','data/he','features/nine_encoder_full_v1_20261010','features/nine_encoder_smoke_v1_20261010']:
        (root/d).mkdir(parents=True,exist_ok=True)
    (project/'transfer/lists').mkdir(parents=True);(world/'logs').mkdir()
    (root/'qc_deps/czifile.py').write_text('"""TEST STUB of czifile (no CZI decoding in local tests)."""\nclass CziFile:\n    def __init__(self,*a,**k):raise RuntimeError("czifile stub")\n')
    shutil.copy(COMP/'originals/czi_pilot_v2.py',root/'bib_phase1/czi_pilot_v2.py')
    shutil.copy(COMP/('originals/full_patch_qc.py' if variant=='real' else 'tests/fake_full_patch_qc.py'),root/'bib_phase1/full_patch_qc.py')
    import numpy as np
    rows=[]
    for k,(pid,grade,n,wh) in enumerate(PATIENTS):
        stem=stem_of(pid);czi=root/'data/he'/f'{stem}.czi';czi.write_bytes(f'fake czi {stem}\n'.encode()*1000)
        (root/'data/he'/f'{stem}.czi.receipt.json').write_text(json.dumps(dict(sha256_local=sha(czi))))
        xs=[(3000+k*50000)+i%8*wh for i in range(n)];ys=[(2000+k*7000)+i//8*wh for i in range(n)]
        coords=np.array([[0,x,y,wh+(i%3==1),wh+(i%3==2)] for i,(x,y) in enumerate(zip(xs,ys))],dtype=np.int64)
        prim=root/'qc/primary_technical_pool_v1_20261010/primary'/f'{stem}.npz';np.savez(prim,coords=coords)
        rows.append(dict(patient_id=pid,clinical_id=pid.split(':')[1],slide_filename=f'{stem}.czi',ISUP_grade=grade,source_sha256=sha(czi),
                         candidate_count=n,primary_count=n,review_count=0,excluded_count=0,mpp_x=.2201,mpp_y=.2201,field_um=128,label_scope='test',
                         eligibility='primary',primary_path=f'{K}/qc/primary_technical_pool_v1_20261010/primary/{stem}.npz',primary_sha256=sha(prim),
                         review_reserve_path='',review_reserve_sha256='',excluded_path='',excluded_sha256=''))
    man=root/'qc/primary_technical_pool_v1_20261010/patient_manifest.csv'
    with man.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
    (root/'qc/primary_technical_pool_v1_20261010/LOCK.json').write_text(json.dumps(dict(test=True,patient_manifest_sha256=sha(man))))  # as the real pool LOCK.json
    weights={'uni':Path(str(pfx)+'/data/cluster/software/uni2h/models/uni/pytorch_model.bin'),'virchow2':Path(str(pfx)+'/data/cluster/bin/pytorch_model(3).bin')}
    for nm,p in weights.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f'fake weights {nm}'.encode()*4096)
    (root/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json').write_text(json.dumps(dict(state='complete',models={nm:dict(weight=str(p),weight_sha256=sha(p)) for nm,p in weights.items()})))
    cases=[dict(patient_id=p,fold=f,ISUP_grade=g,slide_stem=stem_of(p)) for p,f,g in STAGE1]
    list_sha=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    (project/'transfer/lists/STAGE1_CASES_LOCK.json').write_text(json.dumps(dict(rule='test',seed=1,per_fold_per_grade=1,manifest_sha256=sha(man),splits_sha256='x',cases=cases,list_sha256=list_sha)))
    (project/'transfer/lists/stage1_cases.txt').write_text(''.join(c['slide_stem']+'\n' for c in cases))
    (world/'expected_list_sha256.txt').write_text(list_sha)
    if variant=='real':return
    # stored "nine-encoder" features: L of every primary coordinate through the fake encoders
    os.environ['HE_PATH_PREFIX']=str(pfx);os.environ['HE_PROJECT_DIR']=str(project)
    sys.path.insert(0,str(COMP));sys.path.insert(0,str(TESTS))
    import stage0_extract as s0,fake_validate
    from full_patch_qc import IndexedReader
    assert s0.ROOT==root,(s0.ROOT,root)
    models={nm:fake_validate.build(nm)[0].eval() for nm in s0.ENCODERS};embed=s0.embedder(fake_validate,models,'cpu')
    for row in rows:
        stem=Path(row['slide_filename']).stem;dst=root/'features/nine_encoder_full_v1_20261010'/stem;dst.mkdir()
        with np.load(s0.hpc(row['primary_path'])) as z:coords=z['coords']
        r=IndexedReader(root/'data/he'/row['slide_filename']);feats={}
        for nm,(d,interp) in s0.ENCODERS.items():
            L=np.stack([s0.make_inputs(r.region(*[int(v) for v in c])[0],interp)[0] for c in coords])
            feats[nm]=np.concatenate([embed(nm,L[i:i+32]) for i in range(0,len(L),32)]);np.save(dst/f'{nm}.npy',feats[nm])
        (dst/'complete.json').write_text(json.dumps(dict(n=len(coords),primary_coords_sha256=row['primary_sha256'],source_sha256=row['source_sha256'],
                                                     features={nm:dict(shape=list(a.shape),dtype='float16',sha256=sha(dst/f'{nm}.npy')) for nm,a in feats.items()})))

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
