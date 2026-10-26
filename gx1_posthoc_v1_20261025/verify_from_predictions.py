"""Recompute the manuscript's main numbers from the exported per-case predictions alone (PLAN.md section D).
Uses only numpy and its own implementations of AUC, quadratic weighted kappa, Harrell's C, the Cox model (Breslow
ties, Newton-Raphson) and the log-rank test; imports no study code. Compares each value with the locked RESULTS.json
files and writes outputs/VERIFY_REPORT.md (PASS/FAIL per item). Usage:
  python verify_from_predictions.py [--outputs DIR] [--gx1-results FILE] [--surv1-results FILE]"""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def read(p):return list(csv.DictReader(open(p)))
def auc(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float);p=s[y];n=s[~y]
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def qwk(isup,P):
    y=np.asarray(isup)-1;p=np.asarray(P).argmax(1);cm=np.zeros((4,4));np.add.at(cm,(y,p),1)
    w=(np.arange(4)[:,None]-np.arange(4)[None,:])**2/9;e=np.outer(cm.sum(1),cm.sum(0))/cm.sum();return float(1-(w*cm).sum()/(w*e).sum())
def cindex(t,e,r,strata=None):
    t=np.asarray(t,float);e=np.asarray(e,int);r=np.asarray(r,float);num=den=0.
    for i in np.flatnonzero(e==1):
        m=(t>t[i])|((t==t[i])&(e==0))
        if strata is not None:m&=np.asarray(strata)==np.asarray(strata)[i]
        den+=m.sum();num+=(r[i]>r[m]).sum()+.5*(r[i]==r[m]).sum()
    return float(num/den)
def cox(Xm,t,e):
    Xm=np.asarray(Xm,float);t=np.asarray(t,float);e=np.asarray(e,int);b=np.zeros(Xm.shape[1]);ev=np.flatnonzero(e==1)
    R=(t[None,:]>=t[:,None]).astype(float)
    def ll(b):eta=Xm@b;m=eta.max();return float((eta[ev]-m-np.log(R[ev]@np.exp(eta-m))).sum())
    for _ in range(200):
        eta=Xm@b;w=np.exp(eta-eta.max());s0=R[ev]@w;s1=R[ev]@(w[:,None]*Xm);mu=s1/s0[:,None];g=(Xm[ev]-mu).sum(0)
        H=-sum((R[i][:,None]*w[:,None]*Xm).T@Xm/s0[k]-np.outer(mu[k],mu[k]) for k,i in enumerate(ev))
        step=np.linalg.solve(-H,g);a=1.
        while ll(b+a*step)<ll(b)-1e-12 and a>1e-8:a/=2
        b=b+a*step
        if np.abs(a*step).max()<1e-10:break
    eta=Xm@b;w=np.exp(eta-eta.max());s0=R[ev]@w;s1=R[ev]@(w[:,None]*Xm);mu=s1/s0[:,None]
    H=-sum((R[i][:,None]*w[:,None]*Xm).T@Xm/s0[k]-np.outer(mu[k],mu[k]) for k,i in enumerate(ev))
    return b,np.sqrt(np.diag(np.linalg.inv(-H)))
def logrank(t,e,g):
    t=np.asarray(t,float);e=np.asarray(e,int);g=np.asarray(g,bool);O=E=V=0.
    for u in np.unique(t[e==1]):
        at=t>=u;n=at.sum();n1=(at&g).sum();d=((t==u)&(e==1)).sum();O+=((t==u)&(e==1)&g).sum();E+=d*n1/n
        if n>1:V+=d*(n1/n)*(1-n1/n)*(n-d)/(n-1)
    chi=(O-E)**2/V;return float(math.erfc(math.sqrt(chi/2)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outputs',default=str(HERE/'outputs'))
    ap.add_argument('--gx1-results',default=str(HERE.parent/'grade_ext_v1_20261022'/'RESULTS.json'))
    ap.add_argument('--surv1-results',default=str(HERE.parent/'surv_he_clin_v1_20261022'/'RESULTS.json'));a=ap.parse_args()
    O=Path(a.outputs);G=json.loads(Path(a.gx1_results).read_text());S1=json.loads(Path(a.surv1_results).read_text());rep=[];ok_all=True
    def check(name,got,want,tol):
        nonlocal ok_all
        if isinstance(want,(list,tuple)):good=all(abs(x-y)<=tol for x,y in zip(got,want));gs=', '.join(f'{x:.4f}' for x in got);ws=', '.join(f'{y:.4f}' for y in want)
        else:good=abs(got-want)<=tol;gs=f'{got:.4f}';ws=f'{want:.4f}'
        ok_all&=good;rep.append(f"| {'PASS' if good else 'FAIL'} | {name} | {gs} | {ws} |")
    # 1. structure: out-of-fold Korean predictions, 25 models per Scottish case, ensembles = means of runs
    split={r['patient_id']:int(r['fold']) for r in read(O/'korea_split.csv') if r['role']=='test'}
    per=read(O/'grading_predictions_per_run.csv');ens=read(O/'grading_predictions.csv')
    seen=defaultdict(list);runs=defaultdict(list)
    for r in per:
        seen[(r['encoder'],r['cohort'],r['patient_id'])].append(r)
        runs[(r['encoder'],r['cohort'],r['patient_id'])].append([float(r[f'p{k}']) for k in (1,2,3,4)])
    oof=all(len(v)==5 and sorted(int(x['seed']) for x in v)==[0,1,2,3,4] and all(int(x['fold'])==split[x['patient_id']] for x in v) for (e,c,p),v in seen.items() if c=='korea')
    ext=all(len(v)==25 for (e,c,p),v in seen.items() if c=='scotland')
    diff=max(abs(np.mean(runs[(r['encoder'],r['cohort'],r['patient_id'])],0)-np.array([float(r[f'p{k}']) for k in (1,2,3,4)])).max() for r in ens)
    ok_all&=oof and ext and diff<2e-6
    rep+=['| Check | Item | Recomputed | Locked RESULTS |','|---|---|---|---|',
          f"| {'PASS' if oof else 'FAIL'} | every Korean case predicted once per seed, by the model whose test fold holds it | | |",
          f"| {'PASS' if ext else 'FAIL'} | every Scottish case predicted by 25 models per encoder | | |",
          f"| {'PASS' if diff<2e-6 else 'FAIL'} | ensembles equal the mean of their runs (max abs diff {diff:.1e}) | | |"]
    # 2. grading metrics, all encoders
    by=defaultdict(list)
    for r in ens:by[(r['encoder'],r['cohort'])].append(r)
    for (e,coh),rs in sorted(by.items()):
        isup=np.array([int(r['isup']) for r in rs]);P=np.array([[float(r[f'p{k}']) for k in (1,2,3,4)] for r in rs]);g=G['encoders'][e][coh]['grading']
        check(f'{e} {coh} AUC 3-4 vs 1-2',auc(isup>=3,P[:,2:].sum(1)),g['highlow_auc'],1e-3)
        m=isup<=2;check(f'{e} {coh} AUC 2 vs 1',auc(isup[m]==2,P[m,1]/(P[m,0]+P[m,1])),g['g12_auc'],1e-3)
        check(f'{e} {coh} QWK',qwk(isup,P),g['qwk'],1e-3)
    # 3. prognosis of the AI grade (all encoders' C; Phikon-v2 Cox and low-grade split)
    for (e,coh),rs in sorted(by.items()):
        rs=[r for r in rs if r['in_recurrence_set']=='1'];t=np.array([float(r['time_months_censored84']) for r in rs]);ev=np.array([int(r['recurrence']) for r in rs])
        g=np.array([float(r['ai_grade']) for r in rs]);I=np.array([int(r['isup']) for r in rs]);p=G['encoders'][e][coh]['prognosis']
        check(f'{e} {coh} C, AI grade',cindex(t,ev,g),p['c_ai_grade'],1e-3)
        if e!='phikon-v2':continue
        Fu=np.array([int(r['fuhrman']) for r in rs]);L=np.array([float(r['leibovich_fuhrman']) for r in rs])
        check(f'{coh} recurrence set n, events',[len(rs),ev.sum()],[p['n'],p['events']],0)
        check(f'{coh} C, ISUP / Fuhrman / Leibovich',[cindex(t,ev,I),cindex(t,ev,Fu),cindex(t,ev,L)],[p['c_isup'],p['c_fuhrman'],p['c_leibovich']],1e-6)
        check(f'{coh} C within ISUP grade, AI grade',cindex(t,ev,g,I),p['c_within_isup_grade'],1e-3)
        b,se=cox(np.c_[I,g],t,ev);h=p['cox_isup_plus_ai']
        check(f'{coh} Cox(ISUP + AI): HR, CI',[math.exp(b[1]),math.exp(b[1]-1.96*se[1]),math.exp(b[1]+1.96*se[1])],[h['hr_ai_per_grade']]+h['hr_ai_ci95'],0.01)
        b,se=cox(np.c_[L,g],t,ev);h=p['cox_leibovich_plus_ai']
        check(f'{coh} Cox(Leibovich + AI): HR',math.exp(b[1]),h['hr_ai_per_grade'],0.01)
        lo=I<=2;thr=np.median(g[lo]);hi=g[lo]>thr;l=p['low_grade']
        check(f'{coh} ISUP 1-2 split at own median: events above / below',[ev[lo][hi].sum(),ev[lo][~hi].sum()],[l['high_events'],l['low_events']],0)
        check(f'{coh} ISUP 1-2 split: log-rank P',logrank(t[lo],ev[lo],hi),l['logrank_p'],max(1e-3*l['logrank_p'],1e-9))
    # 4. recurrence study (surv1): Leibovich, H&E alone, frozen Korean Cox(Leibovich + H&E) in Scotland, split at Korean median
    rr=read(O/'recurrence_he_risk.csv')
    for e in sorted({r['encoder'] for r in rr}):
        k=[r for r in rr if r['encoder']==e and r['cohort']=='korea' and r['in_recurrence_set']=='1'];s=[r for r in rr if r['encoder']==e and r['cohort']=='scotland']
        f=lambda rs,c:np.array([float(r[c]) for r in rs]);m=S1['encoders'][e]['main']['scotland']
        bK,_=cox(np.c_[f(k,'leibovich_fuhrman'),f(k,'he_risk_standardised')],f(k,'time_months_censored84'),f(k,'recurrence'))
        ts,es,Ls,zs=f(s,'time_months_censored84'),f(s,'recurrence'),f(s,'leibovich_fuhrman'),f(s,'he_risk_standardised')
        check(f'surv1 {e} Scotland C: Leibovich / H&E / Leibovich + H&E (frozen)',[cindex(ts,es,Ls),cindex(ts,es,zs),cindex(ts,es,np.c_[Ls,zs]@bK)],
              [m['c_baseline'],m['c_he_alone'],m['c_fused_frozen']],1e-3)
        med=np.median(f(k,'he_risk_standardised'));check(f'surv1 {e} Scotland split at Korean median: high n, log-rank P',[(zs>med).sum(),logrank(ts,es,zs>med)],
              [m['logrank_split_at_korean_median']['high_n'],m['logrank_split_at_korean_median']['p']],1e-4)
    head=['# Recomputation from per-case predictions','',f"Overall: {'PASS' if ok_all else 'FAIL'} ({sum(l.startswith('| PASS') for l in rep)} passed, {sum(l.startswith('| FAIL') for l in rep)} failed).",
          'Tolerances: 1e-3 for AUC, QWK and C (probabilities exported to 6 decimals), 0.01 for hazard ratios, exact for counts.','']
    (O/'VERIFY_REPORT.md').write_text('\n'.join(head+rep)+'\n');print('\n'.join(head+[l for l in rep if 'FAIL' in l][:40]),flush=True)

if __name__=='__main__':main()
