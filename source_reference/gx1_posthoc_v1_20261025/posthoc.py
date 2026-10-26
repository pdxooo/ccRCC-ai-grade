"""Post hoc analyses A-C of PLAN.md for the grading study (gx1). Refuses to run unless this directory's LOCK.json
matches (the plan and code were locked before any of these numbers existed). Predictions are read through gx1's own
gates; statistics use gx1's own functions. Writes outputs/POSTHOC_RESULTS.json and outputs/POSTHOC_RESULTS.md."""
import sys,json,math,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;GX1=HERE.parent/'grade_ext_v1_20261022';sys.path.insert(0,str(GX1))
import common as C,evaluate as E
np=C.np;OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)

def check_lock():
    lk=json.loads((HERE/'LOCK.json').read_text())
    bad=[n for n,h in lk['files'].items() if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h]
    assert not bad,f'changed since lock: {bad}';return lk
def X(I,g=None,coding='categorical'):
    I=np.asarray(I,float);cols=[(I==2).astype(float),(I>=3).astype(float)] if coding=='categorical' else [I]
    if g is not None:cols.append(np.asarray(g,float))
    return np.column_stack(cols)
def fit(Xm,t,e):
    n0=len(C.COX_WARNINGS);b,se,ll=C.cox_fit(Xm,t,e);assert len(C.COX_WARNINGS)==n0,C.COX_WARNINGS[n0:];return b,se,ll
def lr_p(ll1,ll0,df=1):
    x=max(0.,2*(ll1-ll0));return float(math.erfc(math.sqrt(x/2))) if df==1 else None

def main():
    lk=check_lock();cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    e=cfg['primary_encoder'];S=cfg['seeds'];F=cfg['folds'];B=cfg['analysis']['bootstrap_replicates'];seed=cfg['analysis']['bootstrap_seed']
    gk_all={c['patient_id']:float(C.grade_score(np.mean([kr[(e,s)][c['patient_id']] for s in S],0)[None])[0]) for c in cases}
    sk=C.primary(cases,cfg);fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'}
    tk=np.array([c['time'] for c in sk]);ek=np.array([c['event'] for c in sk]);Ik=np.array([c['isup'] for c in sk]);gk=np.array([gk_all[c['patient_id']] for c in sk]);fk=np.array([fold[c['patient_id']] for c in sk])
    ts=np.array([c['time'] for c in scot]);es=np.array([c['event'] for c in scot]);Is=np.array([c['isup'] for c in scot])
    gs=C.grade_score(np.array([np.mean([sc[(e,s,f)][c['patient_id']] for s in S for f in F],0) for c in scot]))
    R=dict(plan='PLAN.md',lock_sha256=hashlib.sha256((HERE/'LOCK.json').read_bytes()).hexdigest(),locked_at=lk['locked_at'],encoder=e,
           n=dict(korea=int(len(sk)),korea_events=int(ek.sum()),scotland=int(len(scot)),scotland_events=int(es.sum())))
    # A. incremental discrimination: ISUP vs ISUP + AI grade
    A={}
    for coding in ('categorical','numeric'):
        b0,_,_=fit(X(Ik,None,coding),tk,ek);b1,_,_=fit(X(Ik,gk,coding),tk,ek)
        l0=X(Is,None,coding)@b0;l1=X(Is,gs,coding)@b1;c0=C.harrell_c(ts,es,l0);c1=C.harrell_c(ts,es,l1)
        ext=dict(c_isup=c0,c_isup_plus_ai=c1,delta_c=c1-c0,delta_c_ci95=E.boot(len(ts),lambda ix:C.harrell_c(ts[ix],es[ix],l1[ix])-C.harrell_c(ts[ix],es[ix],l0[ix]),B,seed,events=es),
                 korean_coefficients_isup=[float(x) for x in b0],korean_coefficients_isup_plus_ai=[float(x) for x in b1])
        o0=np.zeros(len(sk));o1=np.zeros(len(sk))
        for f in F:
            m=fk!=f;c_b0,_,_=fit(X(Ik[m],None,coding),tk[m],ek[m]);c_b1,_,_=fit(X(Ik[m],gk[m],coding),tk[m],ek[m])
            o0[~m]=X(Ik[~m],None,coding)@c_b0;o1[~m]=X(Ik[~m],gk[~m],coding)@c_b1
        k0=C.harrell_c(tk,ek,o0);k1=C.harrell_c(tk,ek,o1)
        cross=dict(c_isup=k0,c_isup_plus_ai=k1,delta_c=k1-k0,delta_c_ci95=E.boot(len(tk),lambda ix:C.harrell_c(tk[ix],ek[ix],o1[ix])-C.harrell_c(tk[ix],ek[ix],o0[ix]),B,seed,events=ek))
        A[coding]=dict(scotland_frozen_korean_models=ext,korea_cross_fitted=cross)
    R['A_incremental_discrimination']=A
    # B. AI-grade HR with categorical ISUP adjustment, within each cohort
    Bres={}
    for name,t,ev,I,g in (('korea',tk,ek,Ik,gk),('scotland',ts,es,Is,gs)):
        b,se,ll=fit(X(I,g),t,ev);_,_,ll0=fit(X(I),t,ev);j=2
        bn,sen,lln=fit(X(I,g,'numeric'),t,ev);_,_,lln0=fit(X(I,None,'numeric'),t,ev)
        Bres[name]=dict(categorical=dict(hr_ai=float(np.exp(b[j])),hr_ai_ci95=[float(np.exp(b[j]-1.96*se[j])),float(np.exp(b[j]+1.96*se[j]))],p=float(2*C.norm_sf(abs(b[j]/se[j]))),
                                         lr_test_p=lr_p(ll,ll0),hr_isup2=float(np.exp(b[0])),hr_isup34=float(np.exp(b[1]))),
                        numeric=dict(hr_ai=float(np.exp(bn[1])),hr_ai_ci95=[float(np.exp(bn[1]-1.96*sen[1])),float(np.exp(bn[1]+1.96*sen[1]))],p=float(2*C.norm_sf(abs(bn[1]/sen[1]))),lr_test_p=lr_p(lln,lln0)))
    R['B_categorical_isup_adjustment']=Bres
    # C. low-grade split at the Korean threshold
    lo_k=Ik<=2;thr=float(np.median(gk[lo_k]));lo=Is<=2;hi=gs[lo]>thr;chi,p=C.logrank(ts[lo],es[lo],hi)
    own=float(np.median(gs[lo]));chi2,p2=C.logrank(ts[lo],es[lo],gs[lo]>own)
    R['C_low_grade_korean_threshold']=dict(korean_threshold=thr,scotland_n=int(lo.sum()),scotland_events=int(es[lo].sum()),
        above=dict(n=int(hi.sum()),events=int(es[lo][hi].sum())),at_or_below=dict(n=int((~hi).sum()),events=int(es[lo][~hi].sum())),logrank_chi2=chi,logrank_p=p,
        for_comparison_scottish_median=dict(threshold=own,above_n=int((gs[lo]>own).sum()),above_events=int(es[lo][gs[lo]>own].sum()),logrank_p=p2))
    R['cox_warnings']=list(C.COX_WARNINGS);R['finished']=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    (OUT/'POSTHOC_RESULTS.json').write_text(json.dumps(R,indent=1))
    md=['# Post hoc analyses (PLAN.md), Phikon-v2','',f"Locked {lk['locked_at']}; finished {R['finished']}.",'','## A. ISUP vs ISUP + AI grade (Harrell C)','']
    for coding,v in A.items():
        s=v['scotland_frozen_korean_models'];k=v['korea_cross_fitted']
        md.append(f"- {coding} ISUP. Scotland (Korean models frozen): {s['c_isup']:.3f} -> {s['c_isup_plus_ai']:.3f}, dC {s['delta_c']:+.3f} ({s['delta_c_ci95'][0]:+.3f} to {s['delta_c_ci95'][1]:+.3f}). "
                  f"Korea (cross-fitted): {k['c_isup']:.3f} -> {k['c_isup_plus_ai']:.3f}, dC {k['delta_c']:+.3f} ({k['delta_c_ci95'][0]:+.3f} to {k['delta_c_ci95'][1]:+.3f}).")
    md+=['','## B. AI-grade HR adjusted for ISUP','']
    for name,v in Bres.items():
        c=v['categorical'];n=v['numeric']
        md.append(f"- {name}: categorical ISUP HR {c['hr_ai']:.2f} ({c['hr_ai_ci95'][0]:.2f}-{c['hr_ai_ci95'][1]:.2f}), P = {c['p']:.2g}, LR P = {c['lr_test_p']:.2g}; numeric ISUP HR {n['hr_ai']:.2f} ({n['hr_ai_ci95'][0]:.2f}-{n['hr_ai_ci95'][1]:.2f}), P = {n['p']:.2g}.")
    c=R['C_low_grade_korean_threshold']
    md+=['','## C. Scottish ISUP 1-2 split at the Korean threshold','',f"- threshold {c['korean_threshold']:.3f}: above {c['above']['events']}/{c['above']['n']}, at or below {c['at_or_below']['events']}/{c['at_or_below']['n']}, log-rank P = {c['logrank_p']:.2g}"]
    (OUT/'POSTHOC_RESULTS.md').write_text('\n'.join(md)+'\n');print('\n'.join(md),flush=True)

if __name__=='__main__':main()
