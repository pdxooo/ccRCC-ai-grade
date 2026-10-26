"""A6000 vs HPC parity of the nine-encoder Scottish features on cases the HPC already finished. For every case and
encoder: same patch count and coordinate hash, finite values, and the cosine similarity of every patch's A6000 and HPC
feature vectors. PASS when every encoder's worst patch has cosine >= 0.999 (the stage-0 parity threshold).
Usage: compare_parity.py DIR CASE_STEM [CASE_STEM ...]  (HPC features: the mirror folder of each case)"""
import os,sys,json
from pathlib import Path
import numpy as np
PFX=os.environ['HE_PATH_PREFIX'].rstrip('/')
FEAT=Path(PFX+'/data/cluster/projects/scottish_ccrcc_SBIAD2454/features/nine_encoder_full_v1_20261021')
DIMS=['resnet50','convnext_tiny','vit_b_16','uni','uni2-h','phikon-v2','conch','virchow2','gigapath']
out=Path(sys.argv[1]);res={};ok=True
for stem in sys.argv[2:]:
    a=json.loads((out/stem/'complete.json').read_text());h=json.loads((FEAT/stem/'complete.json').read_text())
    r=dict(n=[a['n'],h['n']],coords_same=a['coords_array_sha256']==h['coords_array_sha256'],encoders={})
    ok&=a['n']==h['n'] and r['coords_same']
    for e in DIMS:
        x=np.load(out/stem/f'{e}.npy').astype(np.float32);y=np.load(FEAT/stem/f'{e}.npy').astype(np.float32)
        cos=(x*y).sum(1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
        r['encoders'][e]=dict(min_cos=float(cos.min()),mean_cos=float(cos.mean()),max_abs=float(np.abs(x-y).max()),identical=bool(np.array_equal(x,y)))
        ok&=bool(np.isfinite(x).all()) and float(cos.min())>=0.999
    res[stem]=r
print(json.dumps(dict(state='PASS' if ok else 'FAIL',threshold_min_cos=0.999,cases=res),indent=1))
sys.exit(0 if ok else 1)
