"""Patch-level decomposition of the locked gx1 Phikon-v2 grading models (exploratory; PROTOCOL.md section 2).
For gated attention-MIL with a linear classifier the slide logits are exactly the attention-weighted mean of patch logits,
classifier(sum a_i h_i) = sum a_i classifier(h_i), because the attention weights sum to 1. For every Phikon-v2 run
(5 seeds x 5 folds) and every case it predicted (its Korean test fold and all Scottish cases) this computes, per patch,
the attention a_i, the patch logits l_i, P_high = P(3)+P(4) of softmax(l_i) and the patch expected grade; checks that
softmax(sum a_i l_i) reproduces the slide probabilities the run stored; averages over the runs that predicted the case
(Korea: its fold's 5 seeds; Scotland: 25 models); writes maps/<stem>.npz and case_features.csv. No outcome is read.
Usage (lab-a6000): HE_PATH_PREFIX=... python patch_maps.py [--device cuda]"""
import sys,json,argparse,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
GX1=(HERE/CFG['gx1_dir']).resolve();sys.path.insert(0,str(GX1))
import common as C  # gx1's locked modules
import train as T
import numpy as np,torch
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args(argv)
    cfg=C.config();enc=CFG['encoder'];dev=torch.device(a.device)
    ok,bad=C.lock_status();assert ok,f'gx1 is not its locked version: {bad[:5]}'
    cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg);everyone=cases+scot
    bags=T.Bags(cfg,enc,dev);bags.plan(everyone);bags.preload(everyone)  # sha256 of every file checked against complete.json
    acc={c['patient_id']:dict(n=0,attn=0.,phigh=0.,egrade=0.,feat={}) for c in everyone}
    worst=0.;thr=CFG['high_patch_p']
    for s in cfg['seeds']:
        for f in cfg['folds']:
            d=C.RUNS/enc/f'seed{s}'/f'fold{f}';rc=json.loads((d/'complete.json').read_text())
            assert rc['state']=='COMPLETE' and all(C.sha(d/n)==h for n,h in rc['artifacts'].items()),f'{d}: receipt'
            st=torch.load(d/'last.pt',map_location='cpu',weights_only=False)
            m=T.GatedMIL(cfg['encoders'][enc],cfg['hidden'],cfg['attention_hidden'],cfg['dropout'],4);m.load_state_dict(st['model']);m.to(dev).eval()
            stored={x['patient_id']:np.array(x['probs']) for x in json.loads((d/'predictions.json').read_text())}
            todo=[c for c in everyone if c['patient_id'] in stored]
            with torch.no_grad():
                for c in todo:
                    hs=[];sc=[]
                    for x in bags.chunks(c['patient_id'],cfg['eval_chunk']):
                        h,att=m.encode(x);hs.append(h);sc.append(att)
                    h=torch.cat(hs);att=torch.softmax(torch.cat(sc),0);lg=m.classifier(h).double()
                    slide=torch.softmax((att.double()[:,None]*lg).sum(0),0).cpu().numpy()
                    worst=max(worst,float(np.abs(slide-stored[c['patient_id']]).max()))
                    pp=torch.softmax(lg,1);ph=(pp[:,2]+pp[:,3]).cpu().numpy();eg=(pp@torch.arange(1,5,dtype=torch.float64,device=dev)).cpu().numpy()
                    at=att.double().cpu().numpy();A=acc[c['patient_id']]
                    A['n']+=1;A['attn']=A['attn']+at;A['phigh']=A['phigh']+ph;A['egrade']=A['egrade']+eg
                    k=max(1,int(round(len(at)*CFG['top_attention_fraction'])));top=np.argsort(-at)[:k]
                    for key,v in (('hg_frac',float((ph>thr).mean())),('hg_attn',float(at[ph>thr].sum())),('top_attn_egrade',float(eg[top].mean())),
                                  ('mean_egrade',float(eg.mean()))):A['feat'][key]=A['feat'].get(key,0.)+v
            print('RUN',enc,s,f,len(todo),'cases; worst slide-probability difference so far %.2e'%worst,flush=True)
    assert worst<=CFG['max_reconstruction_error'],f'decomposition does not reproduce stored probabilities ({worst})'
    out=HERE/'maps';out.mkdir(exist_ok=True);rows=[]
    for c in everyone:
        A=acc[c['patient_id']];want=5 if c['cohort']=='korea' else 25;assert A['n']==want,(c['patient_id'],A['n'])
        coords=np.load(C.FEATURE_DIR[c['cohort']]/c['stem']/'coords.npy') if (C.FEATURE_DIR[c['cohort']]/c['stem']/'coords.npy').exists() else np.zeros((0,5))
        np.savez_compressed(out/f"{c['stem']}.npz",attention=(A['attn']/A['n']).astype(np.float32),p_high=(A['phigh']/A['n']).astype(np.float32),
                            egrade=(A['egrade']/A['n']).astype(np.float32),coords=coords)
        rows.append(dict(patient_id=c['patient_id'],cohort=c['cohort'],stem=c['stem'],models=A['n'],patches=len(A['attn']),**{k:v/A['n'] for k,v in A['feat'].items()}))
    with (HERE/'case_features.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    C.save_json(HERE/'PATCH_MAPS_DONE.json',dict(encoder=enc,cases=len(rows),worst_reconstruction_error=worst,finished=C.now()))
    print('DONE',len(rows),'cases; worst slide-probability reconstruction error %.2e'%worst,flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
