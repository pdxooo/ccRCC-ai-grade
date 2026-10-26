"""Local tests of the secondary-analysis plan on surv1's synthetic world (never deployed as data; nothing here reads real
features or outcomes). Builds the world with surv1's own tests/test_local.py (50 CPU runs + evaluate), lays the surv1
copy and a copy of this directory side by side, and checks gates, statistics (against independent implementations)
and the report. Usage: python tests/test_supp.py --surv1-src DIR [--workdir DIR]"""
import argparse,json,hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1];CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)
def run(cwd,args,env,expect=None):
    r=subprocess.run([sys.executable,'-u',*args],cwd=cwd,env=env,capture_output=True,text=True)
    if expect is not None and r.returncode!=expect:
        print(r.stdout[-3000:]);print(r.stderr[-3000:]);raise AssertionError(f'{args} exit {r.returncode}, expected {expect}')
    return r

# independent reference implementations (loops, no shared code with supp.py)
def ref_km(t,e,h):
    s=1.
    for u in sorted(set(t[e==1])):
        if u>h:break
        n=(t>=u).sum();d=((t==u)&(e==1)).sum();s*=1-d/n
    return s
def ref_G_before(t,e,x):
    s=1.
    for u in sorted(set(t[e==0])):
        if u>=x:break
        n=(t>=u).sum();d=((t==u)&(e==0)).sum();s*=1-d/n
    return s
def ref_auc(t,e,r,h):
    num=den=0.
    for i in range(len(t)):
        if not (e[i]==1 and t[i]<=h):continue
        w=1/ref_G_before(t,e,t[i])
        for j in range(len(t)):
            if t[j]>h:den+=w;num+=w*(1. if r[i]>r[j] else .5 if r[i]==r[j] else 0.)
    return num/den

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--surv1-src',required=True);ap.add_argument('--workdir');a=ap.parse_args()
    work=Path(a.workdir or tempfile.mkdtemp(prefix='supp_test_'));shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True);t0=time.monotonic()
    s1=work/'surv1_src';shutil.copytree(Path(a.surv1_src).resolve(),s1,ignore=shutil.ignore_patterns('__pycache__','runs','LOCK.json'))
    for q in [s1,*s1.rglob('*')]:q.chmod(q.stat().st_mode|0o200)  # the locked copy is read-only; its test writes inputs
    w1=work/'surv1_world'
    r=subprocess.run([sys.executable,'-u',str(s1/'tests/test_local.py'),'--workdir',str(w1)],capture_output=True,text=True)
    assert r.returncode==0 and 'ALL_LOCAL_TESTS_PASSED' in r.stdout,r.stdout[-2000:]+r.stderr[-2000:]
    lay=work/'code';S=lay/'surv_he_clin_v1_20261022';D=lay/'surv_he_clin_supp_v1_20261022'
    shutil.copytree(w1/'code',S,ignore=shutil.ignore_patterns('__pycache__'));res1=json.loads((S/'RESULTS.json').read_text())
    (S/'RESULTS.json').unlink();(S/'RESULTS.md').unlink()
    D.mkdir(parents=True)
    for n in ['supp.py','PROTOCOL.md','run_when_ready.sh']:shutil.copy2(SRC/n,D/n)
    (D/'tests').mkdir();shutil.copy2(SRC/'tests/test_supp.py',D/'tests/test_supp.py')
    cfg=json.loads((SRC/'config.json').read_text());cfg['surv1_lock_sha256']=sha(S/'LOCK.json');cfg['bootstrap_replicates']=200
    (D/'config.json').write_text(json.dumps(cfg,indent=2))
    P=w1/'P';env=dict(os.environ,HE_PATH_PREFIX=str(P/'hpcroot'),HE_PROJECT_DIR=str(P),OMP_NUM_THREADS='1')
    print(f'world built in {time.monotonic()-t0:.0f}s',flush=True)
    # ---- unit checks of the statistics against independent implementations
    run(D,['-c',r"""
import supp as X,numpy as np
rng=np.random.default_rng(3);n=150;t=np.round(rng.exponential(30,n),1);c=rng.uniform(10,90,n);e=(t<=c).astype(int);t=np.minimum(t,c);r=rng.normal(0,1,n)+0.02*(80-t)*e
assert abs(X.km_at(t,e,36)[0]-(lambda:__import__('tests.test_supp',fromlist=['x']).ref_km(t,e,36))())<1e-12
import tests.test_supp as T
for h in (20,36,60):assert abs(X.td_auc(t,e,r,h)-T.ref_auc(t,e,r,h))<1e-12
tt=rng.exponential(30,400);ee=np.ones(400,int);rr=rng.normal(0,1,400)  # no censoring: plain AUC / Harrell C
cs=(tt<=36);assert abs(X.td_auc(tt,ee,rr,36)-np.mean([(rr[i]>rr[~cs]).mean() for i in np.flatnonzero(cs)]))<1e-12
import common as C;assert abs(X.uno_c(tt,ee,rr,1e9)-C.harrell_c(tt,ee,rr))<1e-12
xx=rng.normal(0,1,(400,1));m=X.cox_model(xx,tt,ee);assert abs(m[0][0])<0.2 and abs(X.abs_risk(m,np.zeros((1,1)),36)[0]-(1-X.km_at(tt,ee,36)[0]))<0.03  # covariate unrelated to outcome
z=X.nri(t,e,r,r,36);assert z['total']==0 and z['up']==0
th=np.array([0.1,0.2]);risk=np.full(n,0.05);assert X.net_benefit(t,e,risk,36,th)==[0.,0.]
s=T.ref_km(t,e,36);assert np.allclose(X.treat_all(t,e,36,th),[(1-s)-s*p/(1-p) for p in th])
assert np.allclose(X.net_benefit(t,e,np.ones(n),36,th),X.treat_all(t,e,36,th))  # everyone above threshold = treat all
rk=np.clip(0.05+0.4*rng.random(n),0.01,0.9);cal=X.calibration(t,e,rk,36,3);assert len(cal['groups'])==3 and abs(cal['o_e']-(1-s)/rk.mean())<1e-12 and sum(g['n'] for g in cal['groups'])==n
print('UNIT_OK')"""],env,0)
    ok('statistics: Uno AUC = independent IPCW loop; no censoring -> plain AUC and Harrell C; Breslow risk ~ 1 - KM; NRI(model, itself) = 0; net benefit edge cases = treat-none / treat-all; O/E')
    # ---- gates
    r=run(D,['supp.py','run'],env);assert r.returncode==1 and 'not the locked version' in r.stdout
    (S/'RESULTS.json').write_text('{}');r=run(D,['supp.py','lock-write'],env);assert r.returncode==1 and 'can no longer be pre-registered' in r.stdout and not (D/'LOCK.json').exists();(S/'RESULTS.json').unlink()
    run(D,['supp.py','lock-write'],env,0);r=run(D,['supp.py','lock-write'],env);assert r.returncode==1 and 'exists already' in r.stdout
    run(D,['supp.py','lock-verify'],env,0)
    p=S/'runs/phikon-v2/seed2/fold3/predictions.json';b=p.read_bytes();p.write_bytes(b.replace(b'"risk": -',b'"risk": -0',1))
    r=run(D,['supp.py','run'],env);assert r.returncode==1 and 'artifacts changed' in r.stdout;p.write_bytes(b)
    lk=S/'LOCK.json';lb=lk.read_bytes();lk.write_bytes(lb+b' ');r=run(D,['supp.py','run'],env);assert r.returncode==1 and 'not the one this plan' in r.stdout;lk.write_bytes(lb)
    ok('gates: refuses unlocked; lock-write refused once surv1 RESULTS.json exists and when already locked; refuses a changed surv1 run artifact and a different surv1 LOCK.json')
    # ---- full run (via run_when_ready.sh, as on the A6000)
    (P/'env/he310/bin').mkdir(parents=True,exist_ok=True);py=P/'env/he310/bin/python'
    py.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n');py.chmod(0o755)  # a wrapper, not a symlink: keeps the venv
    (P/'logs').mkdir(exist_ok=True);(S/'RESULTS.json').write_text(json.dumps(res1));(S/'RESULTS.md').write_text('x')
    r=subprocess.run(['bash',str(D/'run_when_ready.sh')],env=env,capture_output=True,text=True,timeout=900)
    log=(P/'logs/surv1_supp.log').read_text();assert r.returncode==0 and 'SUPP DONE' in log,log[-3000:]
    R=json.loads((D/'SUPP_RESULTS.json').read_text());md=(D/'SUPP_RESULTS.md').read_text()
    # independent recomputation of the Scottish frozen-model 36-month AUC and HE_z (same as surv1's evaluate)
    world=json.loads((S/'config.json').read_text());seeds=world['seeds'];folds=world['folds']
    import csv
    surv={x['patient_id']:x for x in csv.DictReader(open(S/'inputs/scotland_surv.csv'))};ids=sorted(surv)
    risk={p:[] for p in ids}
    for s in seeds:
        for f in folds:
            for x in json.loads((S/f'runs/phikon-v2/seed{s}/fold{f}/predictions.json').read_text()):
                if x['cohort']=='scotland':risk[x['patient_id']].append(x['risk'])
    raw=np.array([np.mean(risk[p]) for p in ids]);zs=(raw-raw.mean())/raw.std()
    ts=np.array([min(float(surv[p]['dfs_months']),84.) for p in ids]);es=np.array([int(surv[p]['recurrence']) if float(surv[p]['dfs_months'])<=84 else 0 for p in ids])
    Ls=np.array([float(surv[p]['leibovich_fuhrman']) for p in ids]);E=R['encoders']['phikon-v2'];b=np.array(E['models']['fused']['coef'])
    sc=E['scotland']['td_auc']['36'];assert abs(ref_auc(ts,es,np.c_[Ls,zs]@b,36)-sc['fused'])<1e-12 and abs(ref_auc(ts,es,Ls,36)-sc['leibovich'])<1e-12
    assert abs(ref_auc(ts,es,zs,36)-sc['he_alone'])<1e-12
    num=den=0
    for i in np.flatnonzero(es==1):m=(ts>ts[i])|((ts==ts[i])&(es==0));den+=m.sum();num+=(zs[i]>zs[m]).sum()+.5*(zs[i]==zs[m]).sum()
    assert abs(num/den-res1['encoders']['phikon-v2']['main']['scotland']['c_he_alone'])<1e-12  # HE_z identical to surv1's
    assert sc['fused']>sc['leibovich'] and E['scotland']['nri']['category_free']['total']>0  # planted HE signal (positive control)
    for coh,g in (('korea',5),('scotland',3)):
        assert len(E[coh]['calibration']['fused']['groups'])==g and len(E[coh]['dca']['fused'])==len(E[coh]['dca']['thresholds'])==49
        assert set(E[coh]['km']['leibovich_groups'])=={'low','intermediate','high'} and E[coh]['n']==(295 if coh=='korea' else 120)
    assert '36 个月 AUC' in md and 'NRI' in md and '仍随访' in md and R['surv1_code_signature']==res1['code_signature']
    for coh in ('korea','scotland'):  # horizon KM only where someone is still followed
        for v in [E[coh]['km']['overall'],*[x for x in E[coh]['km']['leibovich_groups'].values() if x.get('n')]]:
            for g in ('he_high','he_low'):
                for hh in (36,60):assert (v[g][f'rfs_{hh}'] is None)==(v[g][f'at_risk_{hh}']==0)
    figs=R['figures'];have_mpl=True
    try:import matplotlib  # noqa: F401
    except ImportError:have_mpl=False
    assert (len(figs)==4 and all((D/'figures'/f).stat().st_size>10000 for f in figs)) if have_mpl else figs==[]
    first=json.loads((D/'SUPP_RESULTS.json').read_text());run(D,['supp.py','run'],env,0);second=json.loads((D/'SUPP_RESULTS.json').read_text())
    first.pop('finished');second.pop('finished');assert first==second
    ok(f"run_when_ready.sh -> supp.py: Scottish 36-month AUC (Leibovich {sc['leibovich']:.3f} -> +HE {sc['fused']:.3f}) = independent recomputation; HE_z = surv1's; "
       f"report complete; figures {'4 written' if have_mpl else 'skipped (no matplotlib here)'}; rerun identical")
    print(f'ALL_SUPP_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; work dir {work}')
if __name__=='__main__':main()
