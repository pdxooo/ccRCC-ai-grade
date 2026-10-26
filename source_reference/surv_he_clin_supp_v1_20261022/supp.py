"""Pre-registered secondary analyses of surv_he_clin_v1_20261022 (PROTOCOL.md in this directory). Trains nothing: reads
the locked surv1 code and its 50 formal runs through surv1's own gates (evaluate.collect: hash-verified receipts from the
locked code signature, one platform, the technical check's PASS) and the same HE_z (evaluate.standardise).
Analyses per encoder and cohort (Korea: cross-fitted; Scotland: Korean models frozen): time-dependent AUC (Uno, IPCW),
Uno's C, calibration, decision curves, Kaplan-Meier within Leibovich risk groups, net reclassification.
Writes SUPP_RESULTS.json, SUPP_RESULTS.md, figures/*.png. Exit: 0 done; 1 refused.
Usage: supp.py [run] | supp.py lock-write | supp.py lock-verify"""
import sys,json,hashlib,math,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCFG=json.loads((HERE/'config.json').read_text())
SURV=(HERE/SCFG['surv1_dir']).resolve()
sys.path.insert(0,str(SURV))
import common as C  # surv1's locked modules (paths resolve to the surv1 directory)
import evaluate as E
import numpy as np
LOCKED=['supp.py','config.json','PROTOCOL.md','run_when_ready.sh','tests/test_supp.py']
SIGNATURE_FILES=['supp.py','config.json']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature():return hashlib.sha256(json.dumps({n:sha(HERE/n) for n in SIGNATURE_FILES},sort_keys=True).encode()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def lock_status():
    lk=C.read_json(HERE/'LOCK.json')
    if not lk:return False,['LOCK.json missing']
    bad=[n for n,h in lk['files'].items() if not (HERE/n).exists() or sha(HERE/n)!=h]
    if lk.get('code_signature')!=signature():bad.append('code_signature')
    return not bad,bad

# ----------------------------------------------------------------------------------------------------- statistics
def product_limit(t,d):
    """Product-limit estimate for 'events' d (1) at times t: (event times u, S just after each u, Greenwood sums)."""
    t=np.asarray(t,float);d=np.asarray(d,int);u=np.unique(t[d==1]);S=[];V=[];s=1.;v=0.
    for x in u:
        n=(t>=x).sum();k=((t==x)&(d==1)).sum();s*=1-k/n;S.append(s);v+=k/(n*(n-k)) if n>k else 0.;V.append(v)
    return u,np.array(S),np.array(V)
def step_at(u,S,x,left=False,one=1.):
    """Step function S at x (right-continuous), or just before x (left=True)."""
    i=int(np.searchsorted(u,x,side='left' if left else 'right'));return one if i==0 else float(S[i-1])
def km_at(t,e,h):
    """Kaplan-Meier recurrence-free probability at h, with its Greenwood standard error."""
    u,S,V=product_limit(t,e);s=step_at(u,S,h);v=step_at(u,V,h,one=0.);return s,s*math.sqrt(v)
def km_curve(t,e):
    u,S,_=product_limit(t,e);return dict(time=[0.]+[float(x) for x in u],surv=[1.]+[float(x) for x in S])
def td_auc(time,event,risk,h):
    """Cumulative/dynamic AUC at h (Uno 2007; IPCW with the cohort's censoring Kaplan-Meier): cases recur by h, controls
    are recurrence-free beyond h; risk ties count 1/2."""
    t=np.asarray(time,float);e=np.asarray(event,int);r=np.asarray(risk,float);u,G,_=product_limit(t,1-e)
    cases=np.flatnonzero((t<=h)&(e==1));ctrl=t>h;assert len(cases)>0 and ctrl.sum()>0
    w=np.array([1/step_at(u,G,t[i],left=True) for i in cases])
    hits=np.array([(r[i]>r[ctrl]).sum()+.5*(r[i]==r[ctrl]).sum() for i in cases])
    return float((w*hits).sum()/(w.sum()*ctrl.sum()))
def uno_c(time,event,risk,tau):
    """Uno's C truncated at tau (IPCW weights G(T_i-)^-2; pairs T_i < T_j, T_i < tau, i a recurrence)."""
    t=np.asarray(time,float);e=np.asarray(event,int);r=np.asarray(risk,float);u,G,_=product_limit(t,1-e);num=den=0.
    for i in np.flatnonzero((e==1)&(t<tau)):
        w=step_at(u,G,t[i],left=True)**-2;m=t>t[i];den+=w*m.sum();num+=w*((r[i]>r[m]).sum()+.5*(r[i]==r[m]).sum())
    assert den>0;return float(num/den)
def breslow(X,time,event,b):
    """Breslow cumulative baseline hazard of a fitted Cox model: (event times u, H0 just after each u)."""
    X=np.asarray(X,float);t=np.asarray(time,float);e=np.asarray(event,int);w=np.exp(X@b);u=np.unique(t[e==1])
    return u,np.cumsum([((t==x)&(e==1)).sum()/w[t>=x].sum() for x in u])
def abs_risk(model,X,h):
    """Predicted probability of recurrence by h: 1 - exp(-H0(h) exp(x b))."""
    b,u,H=model;return 1-np.exp(-step_at(u,H,h,one=0.)*np.exp(np.asarray(X,float)@b))
def cox_model(X,t,e):
    b,_,_=C.cox_fit(X,t,e);u,H=breslow(X,t,e,b);return b,u,H
def calibration(t,e,risk_h,h,groups):
    """Calibration slope (Cox of the outcome on log(-log(1 - predicted risk by h)) = log H0(h) + xb, the linear predictor
    of the risks being calibrated, including each Korean fold's own baseline; ideal 1), observed/expected recurrence by h
    (Kaplan-Meier vs mean predicted) and grouped calibration (equal-size groups by predicted risk)."""
    t=np.asarray(t,float);e=np.asarray(e,int);risk_h=np.asarray(risk_h,float)
    b,se,_=C.cox_fit(np.log(-np.log1p(-risk_h))[:,None],t,e);s,_=km_at(t,e,h)
    out=dict(slope=float(b[0]),slope_ci95=[float(b[0]-1.96*se[0]),float(b[0]+1.96*se[0])],observed=1-s,expected=float(risk_h.mean()),
             o_e=float((1-s)/risk_h.mean()),groups=[])
    for ix in np.array_split(np.argsort(risk_h,kind='stable'),groups):
        sg,seg=km_at(t[ix],e[ix],h)
        out['groups'].append(dict(n=int(len(ix)),events=int(e[ix].sum()),predicted=float(risk_h[ix].mean()),observed=1-sg,
                                  observed_ci95=[max(0.,1-sg-1.96*seg),min(1.,1-sg+1.96*seg)]))
    return out
def net_benefit(t,e,risk_h,h,thresholds):
    """Decision-curve net benefit at h (Vickers 2008, Kaplan-Meier version) of 'treat if predicted risk >= threshold'."""
    t=np.asarray(t,float);e=np.asarray(e,int);risk_h=np.asarray(risk_h,float);n=len(t);out=[]
    for pt in thresholds:
        hi=risk_h>=pt;nh=int(hi.sum())
        if nh==0:out.append(0.);continue
        s,_=km_at(t[hi],e[hi],h);out.append(float(((1-s)*nh-s*nh*pt/(1-pt))/n))
    return out
def treat_all(t,e,h,thresholds):
    s,_=km_at(t,e,h);return [float((1-s)-s*pt/(1-pt)) for pt in thresholds]
def nri(t,e,r0,r1,h,cuts=None):
    """Net reclassification improvement at h (Pencina 2011, Kaplan-Meier version) of model 1 over model 0: category-free
    (any increase or decrease in predicted risk) or between the risk categories given by cuts."""
    t=np.asarray(t,float);e=np.asarray(e,int);r0=np.asarray(r0,float);r1=np.asarray(r1,float)
    if cuts is None:up=r1>r0;down=r1<r0
    else:c0=np.digitize(r0,cuts);c1=np.digitize(r1,cuts);up=c1>c0;down=c1<c0
    s,_=km_at(t,e,h)
    def part(m):return (float(m.mean()),km_at(t[m],e[m],h)[0]) if m.any() else (0.,1.)
    pu,su=part(up);pd,sd=part(down);ev=(pu*(1-su)-pd*(1-sd))/(1-s);ne=(pd*sd-pu*su)/s
    return dict(events=float(ev),nonevents=float(ne),total=float(ev+ne),up=float(pu),down=float(pd))
def boot(n,events,fn,B,seed):
    """Case bootstrap (fixed models) of fn(index array) -> float; percentile 95% interval; resamples without an event or
    where fn is undefined are skipped."""
    rng=np.random.default_rng(seed);v=[]
    for _ in range(B):
        ix=rng.integers(0,n,n)
        if events[ix].sum()==0:continue
        try:x=fn(ix)
        except (AssertionError,ZeroDivisionError,FloatingPointError,np.linalg.LinAlgError):continue
        if np.isfinite(x):v.append(x)
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))],len(v)

# ------------------------------------------------------------------------------------------------------ analyses
def groups_of(L):
    """Leibovich risk group by upper bound (low <= 2 < intermediate <= 5 < high)."""
    g=sorted(SCFG['leibovich_groups'].items(),key=lambda kv:kv[1][1])
    return np.array([next((k for k,(a,b) in g if x<=b),g[-1][0]) for x in L])
def km_block(t,e,high,h_list):
    """KM per HE group; a horizon estimate is given only while someone in the group is still followed (else None)."""
    out={}
    for name,m in (('he_high',high),('he_low',~high)):
        out[name]=dict(n=int(m.sum()),events=int(e[m].sum()),**{f'at_risk_{int(h)}':int((t[m]>=h).sum()) for h in h_list},
                       **{f'rfs_{int(h)}':(km_at(t[m],e[m],h)[0] if (t[m]>=h).any() else None) for h in h_list},
                       curve=km_curve(t[m],e[m]) if m.any() else None)
    chi,p=C.logrank(t,e,high) if high.any() and (~high).any() and e.sum()>0 else (None,None);out['logrank']=dict(chi2=chi,p=p)
    return out

def cohort_block(cfg,name,t,e,L,z,lp0,lp1,r0,r1,thr,boot_seed):
    """All secondary analyses of one cohort. lp0/lp1, r0/r1: linear predictors and predicted risk by the horizon of the
    Leibovich-only and the Leibovich + HE models (Korea: cross-fitted; Scotland: frozen Korean models)."""
    h=SCFG['horizon_months'];B=SCFG['bootstrap_replicates'];T=np.arange(*SCFG['dca_thresholds']).round(4)
    out=dict(n=int(len(t)),events=int(e.sum()))
    out['td_auc']={}
    for m in SCFG['auc_months']:
        a0=td_auc(t,e,L,m);a1=td_auc(t,e,lp1,m)
        ci,nb=boot(len(t),e,lambda ix:td_auc(t[ix],e[ix],lp1[ix],m)-td_auc(t[ix],e[ix],L[ix],m),B,boot_seed)
        out['td_auc'][str(m)]=dict(leibovich=a0,fused=a1,he_alone=td_auc(t,e,z,m),delta=a1-a0,delta_ci95=ci,boot_used=nb,
                                   cases=int(((t<=m)&(e==1)).sum()),controls=int((t>m).sum()))
    tau=SCFG['uno_tau_months'];u0=uno_c(t,e,L,tau);u1=uno_c(t,e,lp1,tau)
    ci,_=boot(len(t),e,lambda ix:uno_c(t[ix],e[ix],lp1[ix],tau)-uno_c(t[ix],e[ix],L[ix],tau),B,boot_seed)
    out['uno_c']=dict(tau=tau,leibovich=u0,fused=u1,delta=u1-u0,delta_ci95=ci)
    g=SCFG['calibration_groups'][name]
    out['calibration']=dict(fused=calibration(t,e,r1,h,g),leibovich=calibration(t,e,r0,h,g))
    out['dca']=dict(thresholds=[float(x) for x in T],fused=net_benefit(t,e,r1,h,T),leibovich=net_benefit(t,e,r0,h,T),treat_all=treat_all(t,e,h,T))
    rep={}
    for pt in SCFG['dca_report']:
        i=int(np.argmin(np.abs(T-pt)));rep[str(pt)]=dict(fused=out['dca']['fused'][i],leibovich=out['dca']['leibovich'][i],treat_all=out['dca']['treat_all'][i])
    better=[float(x) for x,f,l,a in zip(T,out['dca']['fused'],out['dca']['leibovich'],out['dca']['treat_all']) if f>l and f>a and f>0]
    out['dca'].update(at=rep,thresholds_fused_best=better)
    cf=nri(t,e,r0,r1,h);ci,_=boot(len(t),e,lambda ix:nri(t[ix],e[ix],r0[ix],r1[ix],h)['total'],B,boot_seed)
    out['nri']=dict(category_free=dict(cf,total_ci95=ci),categorical=dict(nri(t,e,r0,r1,h,SCFG['nri_categories']),cuts=SCFG['nri_categories']))
    grp=groups_of(L);high=z>thr;hl=SCFG['auc_months'];out['km']=dict(threshold=thr,overall=km_block(t,e,high,hl),leibovich_groups={})
    for k in SCFG['leibovich_groups']:
        m=grp==k;out['km']['leibovich_groups'][k]=dict(n=int(m.sum()),events=int(e[m].sum()),**km_block(t[m],e[m],high[m],hl)) if m.any() else dict(n=0)
    return out

def encoder_block(cfg,e,cases_all,assign,scot,kr,sc):
    seeds=cfg['seeds'];folds=cfg['folds'];base=cfg['analysis']['baseline'];prim=C.primary(cases_all,cfg);h=SCFG['horizon_months']
    raw_k={p:float(np.mean([kr[(e,s)][p] for s in seeds])) for p in kr[(e,seeds[0])]}
    raw_s=[float(np.mean([sc[(e,s,f)][c['patient_id']] for s in seeds for f in folds])) for c in scot]
    zk,zs,_=E.standardise(cfg,cases_all,assign,raw_k,raw_s)
    tk=np.array([c['time'] for c in prim]);ek=np.array([c['event'] for c in prim]);Lk=np.array([c[base] for c in prim],float)
    zkk=np.array([zk[c['patient_id']] for c in prim]);fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'}
    fk=np.array([fold[c['patient_id']] for c in prim])
    ts=np.array([c['time'] for c in scot]);es=np.array([c['event'] for c in scot]);Ls=np.array([c[base] for c in scot],float);zs=np.asarray(zs)
    X0k=Lk[:,None];X1k=np.c_[Lk,zkk];X0s=Ls[:,None];X1s=np.c_[Ls,zs]
    lp0k=np.zeros(len(prim));lp1k=np.zeros(len(prim));r0k=np.zeros(len(prim));r1k=np.zeros(len(prim))
    for f in folds:  # Korea: cross-fitted coefficients and baseline hazards (other four folds)
        tr=fk!=f;te=~tr;m0=cox_model(X0k[tr],tk[tr],ek[tr]);m1=cox_model(X1k[tr],tk[tr],ek[tr])
        lp0k[te]=X0k[te]@m0[0];lp1k[te]=X1k[te]@m1[0];r0k[te]=abs_risk(m0,X0k[te],h);r1k[te]=abs_risk(m1,X1k[te],h)
    M0=cox_model(X0k,tk,ek);M1=cox_model(X1k,tk,ek)  # all 295 Korean cases: frozen for Scotland
    thr=float(np.median(zkk));seed=SCFG['bootstrap_seed']
    return dict(models=dict(leibovich_only=dict(coef=[float(x) for x in M0[0]]),fused=dict(coef=[float(x) for x in M1[0]]),
                            baseline_risk_by_horizon=dict(leibovich_only=float(1-math.exp(-step_at(M0[1],M0[2],h,one=0.))),
                                                          fused=float(1-math.exp(-step_at(M1[1],M1[2],h,one=0.))))),
                korea=cohort_block(cfg,'korea',tk,ek,Lk,zkk,lp0k,lp1k,r0k,r1k,thr,seed),
                scotland=cohort_block(cfg,'scotland',ts,es,Ls,zs,X0s@M0[0],X1s@M1[0],abs_risk(M0,X0s,h),abs_risk(M1,X1s,h),thr,seed))

# ------------------------------------------------------------------------------------------------------- report
def figures(res,out):
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except ImportError:return []
    out.mkdir(exist_ok=True);made=[];h=SCFG['horizon_months'];names=dict(korea='Korea (cross-fitted)',scotland='Scotland (external)')
    for e,r in res['encoders'].items():
        for coh in ('korea','scotland'):
            b=r[coh];tag=f'{coh}_{e}'
            fig,ax=plt.subplots(1,4,figsize=(18,4.2))
            for name,col in (('he_high','#c0392b'),('he_low','#2471a3')):
                c=b['km']['overall'][name]
                if c['curve']:ax[0].step(c['curve']['time'],c['curve']['surv'],where='post',color=col,label=f"HE {name[3:]} (n={c['n']})")
            ax[0].set(title=f"{names[coh]}: HE high vs low\nlog-rank p = {b['km']['overall']['logrank']['p']:.3g}" if b['km']['overall']['logrank']['p'] is not None else names[coh],
                      xlabel='months',ylabel='recurrence-free',ylim=(0,1.02));ax[0].legend()
            for k,ls in zip(SCFG['leibovich_groups'],('-','--',':')):
                g=b['km']['leibovich_groups'][k]
                for name,col in (('he_high','#c0392b'),('he_low','#2471a3')):
                    if g.get('n') and g[name]['curve']:ax[1].step(g[name]['curve']['time'],g[name]['curve']['surv'],where='post',color=col,ls=ls,label=f"{k} / HE {name[3:]} ({g[name]['n']})")
            ax[1].set(title='Within Leibovich risk groups',xlabel='months',ylim=(0,1.02));ax[1].legend(fontsize=7)
            for m,col in (('fused','#c0392b'),('leibovich','#7f8c8d')):
                gs=b['calibration'][m]['groups'];ax[2].errorbar([g['predicted'] for g in gs],[g['observed'] for g in gs],
                    yerr=[[g['observed']-g['observed_ci95'][0] for g in gs],[g['observed_ci95'][1]-g['observed'] for g in gs]],fmt='o-',color=col,label=m)
            lim=max(0.05,max(max(g['predicted'],g['observed']) for m in ('fused','leibovich') for g in b['calibration'][m]['groups'])*1.1)
            ax[2].plot([0,lim],[0,lim],'k:');ax[2].set(title=f'Calibration at {h} months',xlabel='predicted',ylabel='observed (KM)',xlim=(0,lim),ylim=(0,lim));ax[2].legend()
            d=b['dca'];ax[3].plot(d['thresholds'],d['fused'],color='#c0392b',label='Leibovich + HE');ax[3].plot(d['thresholds'],d['leibovich'],color='#7f8c8d',label='Leibovich')
            ax[3].plot(d['thresholds'],d['treat_all'],color='k',ls='--',label='treat all');ax[3].axhline(0,color='k',lw=.8)
            top=max(max(d['fused']),max(d['leibovich']),0.01);ax[3].set(title=f'Decision curve at {h} months',xlabel='threshold probability',ylabel='net benefit',ylim=(-0.05,top*1.2));ax[3].legend()
            fig.suptitle(f'{e}');fig.tight_layout();p=out/f'{tag}.png';fig.savefig(p,dpi=130);plt.close(fig);made.append(p.name)
    return made

def markdown(res):
    h=SCFG['horizon_months'];L=['# HE + 临床预测复发：预先登记的次要分析','',f"生成时间 {res['finished']}；次要分析代码签名 `{res['code_signature'][:16]}`；surv1 代码签名 `{res['surv1_code_signature'][:16]}`。",
       '韩国：交叉拟合（系数和基线风险都来自另外 4 折）；苏格兰：韩国 295 例拟合的模型原样冻结。主要时间点 %d 个月。'%h,'']
    for e,r in res['encoders'].items():
        L+=[f'## {e}','','| | 韩国 | 苏格兰 |','|---|---|---|']
        k=r['korea'];s=r['scotland']
        for m in SCFG['auc_months']:
            a,b=k['td_auc'][str(m)],s['td_auc'][str(m)]
            L.append(f"| {m} 个月 AUC：Leibovich → +HE（Δ，95% 区间） | {a['leibovich']:.3f} → {a['fused']:.3f}（{a['delta']:+.3f}，{a['delta_ci95'][0]:+.3f} 至 {a['delta_ci95'][1]:+.3f}） | {b['leibovich']:.3f} → {b['fused']:.3f}（{b['delta']:+.3f}，{b['delta_ci95'][0]:+.3f} 至 {b['delta_ci95'][1]:+.3f}） |")
        L.append(f"| {SCFG['horizon_months']} 个月 AUC：HE 单独 | {k['td_auc'][str(SCFG['horizon_months'])]['he_alone']:.3f} | {s['td_auc'][str(SCFG['horizon_months'])]['he_alone']:.3f} |")
        L.append(f"| Uno C（τ = {k['uno_c']['tau']} 个月）：Leibovich → +HE | {k['uno_c']['leibovich']:.3f} → {k['uno_c']['fused']:.3f} | {s['uno_c']['leibovich']:.3f} → {s['uno_c']['fused']:.3f} |")
        L.append(f"| Uno C（τ = {k['uno_c']['tau']} 个月）Δ | {k['uno_c']['delta']:+.3f}（{k['uno_c']['delta_ci95'][0]:+.3f} 至 {k['uno_c']['delta_ci95'][1]:+.3f}） | {s['uno_c']['delta']:+.3f}（{s['uno_c']['delta_ci95'][0]:+.3f} 至 {s['uno_c']['delta_ci95'][1]:+.3f}） |")
        for m,lab in (('fused','+HE'),('leibovich','Leibovich')):
            a,b=k['calibration'][m],s['calibration'][m]
            L.append(f"| 校准（{lab}）：斜率；O/E | {a['slope']:.2f}（{a['slope_ci95'][0]:.2f}–{a['slope_ci95'][1]:.2f}）；{a['o_e']:.2f} | {b['slope']:.2f}（{b['slope_ci95'][0]:.2f}–{b['slope_ci95'][1]:.2f}）；{b['o_e']:.2f} |")
        for pt,v in s['dca']['at'].items():
            w=k['dca']['at'][pt];L.append(f"| 决策曲线净获益，阈值 {pt}：+HE / Leibovich / 全部治疗 | {w['fused']:.3f} / {w['leibovich']:.3f} / {w['treat_all']:.3f} | {v['fused']:.3f} / {v['leibovich']:.3f} / {v['treat_all']:.3f} |")
        a,b=k['nri'],s['nri']
        L.append(f"| NRI（无分类）总计（95% 区间） | {a['category_free']['total']:+.2f}（{a['category_free']['total_ci95'][0]:+.2f} 至 {a['category_free']['total_ci95'][1]:+.2f}） | {b['category_free']['total']:+.2f}（{b['category_free']['total_ci95'][0]:+.2f} 至 {b['category_free']['total_ci95'][1]:+.2f}） |")
        L.append(f"| NRI（< {a['categorical']['cuts'][0]:.0%} / {a['categorical']['cuts'][0]:.0%}–{a['categorical']['cuts'][1]:.0%} / ≥ {a['categorical']['cuts'][1]:.0%}）：复发者 / 未复发者 | {a['categorical']['events']:+.2f} / {a['categorical']['nonevents']:+.2f} | {b['categorical']['events']:+.2f} / {b['categorical']['nonevents']:+.2f} |")
        L.append('')
        for coh,b in (('韩国',k),('苏格兰',s)):
            parts=[]
            for g,v in b['km']['leibovich_groups'].items():
                if not v.get('n'):continue
                hh,ll=v['he_high'],v['he_low'];p=v['logrank']['p']
                f=lambda x:'—' if x is None else '%.2f'%x
                parts.append(f"{dict(low='低危',intermediate='中危',high='高危')[g]} {v['n']} 例 {v['events']} 次复发：HE 高 {hh['n']} 例 {hh['events']} 次、{h} 个月无复发 {f(hh[f'rfs_{h}'])}（仍随访 {hh[f'at_risk_{h}']}），"
                             f"HE 低 {ll['n']} 例 {ll['events']} 次、{f(ll[f'rfs_{h}'])}（仍随访 {ll[f'at_risk_{h}']}），log-rank p = {'—' if p is None else '%.3g'%p}")
            o=b['km']['overall'];po=o['logrank']['p']
            L.append(f"- {coh}，全队列按 HE 高/低：HE 高 {o['he_high']['n']} 例 {o['he_high']['events']} 次复发，HE 低 {o['he_low']['n']} 例 {o['he_low']['events']} 次，log-rank p = {'—' if po is None else '%.3g'%po}")
            L.append(f"- {coh}，Leibovich 分组内按 HE 高/低：" + '；'.join(parts))
        L+=['',f"- 苏格兰决策曲线上 +HE 同时优于 Leibovich 和全部治疗的阈值：{', '.join('%.2f'%x for x in s['dca']['thresholds_fused_best']) or '无'}",'']
    if res['cox_warnings']:L+=[f"注意：{len(res['cox_warnings'])} 个 Cox 拟合未收敛或系数过大（见 SUPP_RESULTS.json cox_warnings）。",'']
    L+=['以上均为预先登记的次要分析，不改变主分析的判定。']
    return '\n'.join(L)+'\n'

def run():
    problems=[];ok,bad=lock_status()
    if not ok:problems.append(f'secondary-analysis code is not the locked version: {bad[:5]}')
    if not (SURV/'LOCK.json').exists() or sha(SURV/'LOCK.json')!=SCFG['surv1_lock_sha256']:problems.append('surv1 LOCK.json is not the one this plan was written for')
    ok2,bad2=C.lock_status()
    if not ok2:problems.append(f'surv1 code is not its locked version: {bad2[:5]}')
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    chk=C.read_json(C.CHECK_PASS) or {}
    if chk.get('state')!='PASS' or chk.get('code_signature')!=C.signature():problems.append('surv1 CHECK_PASS.json is not a PASS for its code')
    kr,sc,more,plats=E.collect(cfg,cases,assign,scot);problems+=more
    if plats and json.dumps(chk.get('platform'),sort_keys=True) not in plats:problems.append('surv1 runs and its technical check ran on different platforms')
    if problems:
        print('REFUSED',len(problems),'problem(s):',flush=True)
        for p in problems[:40]:print(' ',p,flush=True)
        return 1
    res=dict(encoders={e:encoder_block(cfg,e,cases,assign,scot,kr,sc) for e in cfg['encoders']},code_signature=signature(),
             surv1_code_signature=C.signature(),platform=[json.loads(p) for p in plats],finished=now(),config=SCFG)
    res['cox_warnings']=list(C.COX_WARNINGS);res['figures']=[]
    C.save_json(HERE/'SUPP_RESULTS.json',res);(HERE/'SUPP_RESULTS.md').write_text(markdown(res))  # results first
    try:
        res['figures']=figures(res,HERE/'figures')
        if not res['figures']:res['figure_warning']='matplotlib not importable: no figures'
    except Exception as ex:res['figure_warning']=f'figures failed: {ex!r}'[:500]
    if 'figure_warning' in res:print('WARNING',res['figure_warning'],flush=True)
    C.save_json(HERE/'SUPP_RESULTS.json',res)
    s=res['encoders'][cfg['primary_encoder']]['scotland']['td_auc'][str(SCFG['horizon_months'])]
    print('SUPP DONE scotland %d-month AUC Leibovich %.3f -> +HE %.3f'%(SCFG['horizon_months'],s['leibovich'],s['fused']),flush=True)
    return 0

def main(argv):
    cmd=argv[0] if argv else 'run'
    if cmd=='run':return run()
    if cmd=='lock-verify':ok,bad=lock_status();print('LOCK','OK' if ok else 'BROKEN',bad);return 0 if ok else 1
    if cmd=='lock-write':
        if (HERE/'LOCK.json').exists():print('LOCK.json exists already; a change needs a new version directory');return 1
        if sha(SURV/'LOCK.json')!=SCFG['surv1_lock_sha256']:print('REFUSED: surv1 LOCK.json differs from config');return 1
        if (SURV/'RESULTS.json').exists():print('REFUSED: surv1 results exist already: this plan can no longer be pre-registered');return 1
        lk=dict(locked_at=now(),files={n:sha(HERE/n) for n in LOCKED},code_signature=signature(),surv1_lock_sha256=SCFG['surv1_lock_sha256'])
        C.save_json(HERE/'LOCK.json',lk);print('LOCKED',lk['code_signature'],len(lk['files']),'files');return 0
    print(__doc__);return 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
