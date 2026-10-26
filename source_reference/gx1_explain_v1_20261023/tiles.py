"""Illustrative figure for the exploratory gx1 analyses (PROTOCOL.md section 2.5): for each cohort, among ISUP 1-2 cases
whose slide is on the lab-a6000 mirror, case A = the recurrence with the highest AI grade score and case B = the
non-recurrence followed >= 36 months with the lowest AI grade score (selection rule fixed before looking at any map).
Panels: patch map coloured by P(3)+P(4), attention map, and the 4 tiles with the largest attention x P(3)+P(4) contribution,
decoded from the CZI exactly as the extraction did (128 um field). Writes figures/examples_<cohort>.png and examples.json.
Usage (lab-a6000): HE_PATH_PREFIX=... python tiles.py"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
GX1=(HERE/CFG['gx1_dir']).resolve();sys.path.insert(0,str(GX1))
import common as C
import evaluate as E
import numpy as np
import analyze as X
sys.path.insert(0,str(C.ROOT/'qc_deps'));sys.path.insert(0,str(C.ROOT/'bib_phase1'))
def coords_of(c):
    """Patch coordinates in feature order: the QC pool's primary/<stem>.npz (the extraction's input), sha256-checked
    against the case's complete.json record and its feature length."""
    pool=C.POOL if c['cohort']=='korea' else C.SPOOL;p=pool/'primary'/f"{c['stem']}.npz"
    rec=json.loads((C.FEATURE_DIR[c['cohort']]/c['stem']/'complete.json').read_text())
    if 'primary_coords_sha256' in rec:assert C.sha(p)==rec['primary_coords_sha256'],f'{p}: coordinates differ from the extraction record'
    with np.load(p,allow_pickle=False) as a:xy=a['coords'].copy()
    assert len(xy)==rec['n'],(p,len(xy),rec['n']);return xy

def main():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from full_patch_qc import IndexedReader
    from PIL import Image
    ok,bad=X.lock_status();assert ok,f'not the locked analysis code or patch-map output: {bad[:5]}'
    feats={r['patient_id']:r for r in C.read_csv(HERE/'case_features.csv')}
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:5]
    enc=CFG['encoder'];seeds=cfg['seeds'];folds=cfg['folds']
    ai={p:float(C.grade_score(np.mean([kr[(enc,s)][p] for s in seeds],0)[None])[0]) for p in kr[(enc,seeds[0])]}
    ai.update({c['patient_id']:float(C.grade_score(np.mean([sc[(enc,s,f)][c['patient_id']] for s in seeds for f in folds],0)[None])[0]) for c in scot})
    picked={};(HERE/'figures').mkdir(exist_ok=True)
    for coh,cs,root in (('korea',cases,C.ROOT),('scotland',scot,C.SROOT)):
        czi=lambda c:root/'data/he'/(c['stem']+'.czi')
        low=[c for c in C.primary(cs,cfg) if c['isup']<=2 and czi(c).exists() and (C.POOL if coh=='korea' else C.SPOOL).joinpath('primary',c['stem']+'.npz').exists()]
        A=max([c for c in low if c['event']],key=lambda c:ai[c['patient_id']],default=None)
        B=min([c for c in low if not c['event'] and c['time']>=36],key=lambda c:ai[c['patient_id']],default=None)
        if A is None or B is None:print('SKIP',coh,'no eligible pair');continue
        fig,ax=plt.subplots(2,6,figsize=(22,7.5));picked[coh]=[]
        for r,(c,tag) in enumerate(((A,'A: recurrence, highest AI grade'),(B,'B: no recurrence >= 36 mo, lowest AI grade'))):
            m=np.load(HERE/'maps'/f"{c['stem']}.npz");ph=m['p_high'];at=m['attention'];xy=coords_of(c);assert len(xy)==len(ph)
            # lay scenes side by side: x offset per scene
            sx=np.zeros(len(xy));off=0.
            for s in np.unique(xy[:,0]):
                k=xy[:,0]==s;sx[k]=xy[k,1]-xy[k,1].min()+off;off=sx[k].max()+3*np.median(xy[:,3])
            sy=-(xy[:,2]);size=max(2,40000/len(xy))
            p0=ax[r,0].scatter(sx,sy,c=ph,cmap='RdYlBu_r',vmin=0,vmax=1,s=size,marker='s',linewidths=0);ax[r,0].set_aspect('equal');ax[r,0].axis('off')
            ax[r,0].set_title(f"{coh} {c['patient_id'].split(':')[1]} ISUP {c['isup']}, AI {ai[c['patient_id']]:.2f}\n{tag}\nP(3)+P(4) per patch",fontsize=9);plt.colorbar(p0,ax=ax[r,0],fraction=.04)
            p1=ax[r,1].scatter(sx,sy,c=at/at.max(),cmap='magma',vmin=0,vmax=1,s=size,marker='s',linewidths=0);ax[r,1].set_aspect('equal');ax[r,1].axis('off')
            ax[r,1].set_title('attention (relative)',fontsize=9);plt.colorbar(p1,ax=ax[r,1],fraction=.04)
            top=np.argsort(-(at*ph))[:4];rd=IndexedReader(str(czi(c)))
            for j,i in enumerate(top):
                s_,x,y,w,h=[int(v) for v in xy[i]];rgb,cov,_=rd.region(s_,x,y,w,h)
                ax[r,2+j].imshow(np.asarray(Image.fromarray(rgb).resize((448,448),Image.Resampling.BICUBIC)));ax[r,2+j].axis('off')
                ax[r,2+j].set_title(f"P(3)+P(4) {ph[i]:.2f}, attn rank {int((at>at[i]).sum())+1}",fontsize=9)
            rd.close();picked[coh].append(dict(case=c['patient_id'],role=tag,isup=c['isup'],ai_grade=ai[c['patient_id']],time=c['time'],event=c['event'],
                                               hg_frac=float(feats[c['patient_id']]['hg_frac']),patches=int(len(ph))))
        fig.tight_layout();fig.savefig(HERE/'figures'/f'examples_{coh}.png',dpi=120);plt.close(fig)
    C.save_json(HERE/'examples.json',picked);print('EXAMPLES',json.dumps(picked),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
