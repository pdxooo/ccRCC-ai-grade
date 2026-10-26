"""Local CPU tests on a synthetic Korean + Scottish mirror (never deployed; nothing here reads real features or outcomes).
The code is copied byte for byte into <work>/code except config.json (device cpu, the fake input files' sha256, small
shapes) and the four inputs/*.csv (the fake world's outcomes and IF values). The fake world plants a recurrence-risk
signal in the HE features beyond the Leibovich score (positive control). Usage: python tests/test_local.py [--workdir DIR]"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1];DIMS={'phikon-v2':1024,'uni':1024};CHECKS=[]
GRADES={1:111,2:129,3:44,4:16}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wjson(p,o):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,indent=2))
def rjson(p):return json.loads(Path(p).read_text())
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)
def wcsv(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def outcomes(rng,n,risk,leib):
    """Exponential recurrence times with hazard from (Leibovich, hidden risk); uniform censoring."""
    lam=0.004*np.exp(0.25*(leib-4)+0.9*risk);t=rng.exponential(1/lam);c=rng.uniform(20,150,n)
    return np.minimum(t,c),(t<=c).astype(int)

def build_world(work):
    rng=np.random.default_rng(5);P=work/'P';H=P/'hpcroot'
    K=H/'work/data/user/projects/korean_ccrcc_SBIAD2452';S=H/'work/data/user/projects/scottish_ccrcc_SBIAD2454'
    pool=K/'qc/primary_technical_pool_v1_20261010';splits=K/'bib_phase1/first_round_v1_20261010';full=K/'features/nine_encoder_full_v1_20261010'
    spool=S/'qc/external_primary_technical_pool_v1_20261013';sfull=S/'features/nine_encoder_full_v1_20261021'
    for d in (pool,splits,full,spool,sfull,K/'qc_deps',P/'transfer'):d.mkdir(parents=True,exist_ok=True)
    direction={e:rng.normal(0,1,d).astype(np.float32) for e,d in DIMS.items()}
    def feats(folder,n,risk):
        folder.mkdir(parents=True,exist_ok=True);rec=dict(state='complete',n=n,features={})
        for e,d in DIMS.items():
            a=(rng.normal(0,1,(n,d)).astype(np.float32)+0.04*risk*direction[e]).astype(np.float16);np.save(folder/f'{e}.npy',a)
            (folder/f'{e}.npy.verified').touch();rec['features'][e]=dict(sha256=sha(folder/f'{e}.npy'))
        wjson(folder/'complete.json',rec)
    def cohort(n,ids,stems,grades):
        risk=rng.normal(0,1,n);leib=np.clip(np.round(4+1.5*rng.normal(0,1,n)+0.8*risk*0+0.6*(grades-2)),0,11)
        t,e=outcomes(rng,n,risk,leib);return risk,leib,t,e
    # Korea
    grades=np.concatenate([[g]*m for g,m in GRADES.items()]);rng.shuffle(grades)
    pts=[dict(patient_id=f'korea:KR{i+1}',slide_filename=f'HE_KR{i+1}.czi',ISUP_grade=str(int(grades[i])),primary_count=str(int(rng.integers(20,50)))) for i in range(300)]
    order=sorted(range(300),key=lambda i:(pts[i]['ISUP_grade'],rng.random()));group={pts[i]['patient_id']:r%5 for r,i in enumerate(order)}
    rows=[dict(fold=str(f),patient_id=p['patient_id'],slide_filename=p['slide_filename'],ISUP_grade=p['ISUP_grade'],
               role='test' if group[p['patient_id']]==f else 'val' if group[p['patient_id']]==(f+1)%5 else 'train') for f in range(5) for p in pts]
    wcsv(pool/'patient_manifest.csv',pts)
    with (splits/'patient_splits.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['fold','patient_id','slide_filename','ISUP_grade','role']);w.writeheader();w.writerows(rows)
    wjson(pool/'LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv')))
    wjson(splits/'SPLITS_LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv'),qc_lock_sha256=sha(pool/'LOCK.json'),split_sha256=sha(splits/'patient_splits.csv')))
    rk,lk,tk,ek=cohort(300,None,None,grades.astype(float));tk[:5]=0.;ek[:5]=1  # five DFS-0 recurrences
    for i,p in enumerate(pts):feats(full/f'HE_KR{i+1}',int(p['primary_count']),rk[i])
    ksurv=[dict(patient_id=p['patient_id'],recurrence=int(ek[i]),dfs_months=round(float(tk[i]),3),leibovich_fuhrman=int(lk[i]),leibovich_isup=int(max(0,lk[i]-1)),
                isup=int(p['ISUP_grade']),fuhrman=min(4,int(p['ISUP_grade'])+1),t_stage='1a',size_cm=4.0,necrosis=0) for i,p in enumerate(pts)]
    # Scotland
    sg=rng.choice([1,2,3,4],120,p=[.3,.42,.26,.02])
    spts=[dict(patient_id=f'scotland:{1000+i}',clinical_id=str(1000+i),slide_filename=f'iCAIRD -{1000+i}-HE-St Andrews-11_06_2021.czi',ISUP_grade=str(int(sg[i])),primary_count=str(int(rng.integers(20,50)))) for i in range(120)]
    wcsv(spool/'patient_manifest.csv',spts);wjson(spool/'LOCK.json',dict(patient_manifest_sha256=sha(spool/'patient_manifest.csv')))
    rs,ls,ts,es=cohort(120,None,None,sg.astype(float))
    for i,p in enumerate(spts):feats(sfull/Path(p['slide_filename']).stem,int(p['primary_count']),rs[i])
    ssurv=[dict(patient_id=p['patient_id'],recurrence=int(es[i]),dfs_months=round(float(ts[i]),3),leibovich_fuhrman=int(ls[i]),leibovich_isup=int(max(0,ls[i]-1)),
                isup=int(p['ISUP_grade']),fuhrman=min(4,int(p['ISUP_grade'])+1),t_stage='1a',size_cm=4.0,necrosis=0) for i,p in enumerate(spts)]
    kif=[dict(patient_id=pts[i]['patient_id'],p21_mcm2_percentage=float(np.exp(-0.5*rk[i]+rng.normal())),avg_nucleus_area_m=50.0) for i in range(10,81)]
    sif=[dict(patient_id=spts[i]['patient_id'],p21_mcm2_percentage=float(np.exp(-0.5*rs[i]+rng.normal())),avg_nucleus_area_m=50.0) for i in range(0,88)]
    for e in DIMS:(P/'transfer'/f'verified_feat_{e}').touch();(P/'transfer'/f'verified_scotland_feat_{e}').touch()
    return dict(P=P,ksurv=ksurv,ssurv=ssurv,kif=kif,sif=sif,group=group)

def copy_code(w,dest):
    for n in ['PROTOCOL.md','common.py','train.py','check.py','evaluate.py','make_labels.py','make_campaign.py','lock.py','CAMPAIGN.json']:
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/n,dest/n);assert sha(dest/n)==sha(SRC/n)
    shutil.copytree(SRC/'inputs',dest/'inputs');(dest/'tests').mkdir();shutil.copy2(SRC/'tests/test_local.py',dest/'tests/test_local.py')
    for n,rows in (('korea_surv.csv',w['ksurv']),('scotland_surv.csv',w['ssurv']),('korea_if.csv',w['kif']),('scotland_if.csv',w['sif'])):wcsv(dest/'inputs'/n,rows)
    cfg=json.loads((SRC/'config.json').read_text());assert cfg['device']=='cuda'
    cfg.update(device='cpu',train_patches=16,eval_chunk=64,epochs=6,learning_rate=0.001)
    for n in cfg['labels_sha256']:cfg['labels_sha256'][n]=sha(dest/'inputs'/n)
    cfg['check'].update(chunk_sizes=[64,0,7]);cfg['analysis']['bootstrap_replicates']=200
    (dest/'config.json').write_text(json.dumps(cfg,indent=2));return dest

def run(code,args,env,expect=None):
    r=subprocess.run([sys.executable,'-u',*args],cwd=code,env=env,capture_output=True,text=True)
    with (code.parent/'logs.txt').open('a') as f:f.write(f'$ {" ".join(args)}\n{r.stdout}\n{r.stderr}\nEXIT {r.returncode}\n')
    if expect is not None and r.returncode!=expect:
        print(r.stdout[-3000:]);print(r.stderr[-3000:]);raise AssertionError(f'{args} exit {r.returncode}, expected {expect}')
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir');ap.add_argument('--jobs',type=int,default=8);a=ap.parse_args()
    work=Path(a.workdir or tempfile.mkdtemp(prefix='surv1_test_'));shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True)
    t0=time.monotonic();w=build_world(work);code=copy_code(w,work/'code')
    env=dict(os.environ,HE_PATH_PREFIX=str(w['P']/'hpcroot'),HE_PROJECT_DIR=str(w['P']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    py=lambda s:run(code,['-c',s],env,0).stdout
    out=py("""import common as C,json,numpy as np
cfg=C.config();k,assign=C.load_korea(cfg);s=C.load_scotland(cfg);st=C.label_structure(k,assign,s,cfg);chk=C.check_cases(k,assign,cfg)
r=C.roles(k,assign,0,cfg);cuts=C.time_bins(r['train'],cfg)
assert all(c['time']<=84 for c in k) and not any(c['zero_dfs'] for c in r['train']) and any(c['zero_dfs'] for f in range(5) for c in C.roles(k,assign,f,cfg)['test'])
assert [C.bin_index(t,[1,2,3]) for t in (0.5,1,2.5,3,9)]==[0,1,2,3,3]
rng=np.random.default_rng(0);X=rng.normal(0,1,(3000,2));t=rng.exponential(1/np.exp(X@np.array([.7,-.5])));c=rng.uniform(0,2,3000);e=(t<=c).astype(int);tt=np.minimum(t,c)
b,se,_=C.cox_fit(X,tt,e);assert np.abs(b-np.array([.7,-.5])).max()<0.08 and (se>0).all() and se.max()<0.05,(b,se)
y=np.array([1,2,3,4.]);ev=np.array([1,1,0,1]);assert C.harrell_c(y,ev,[4,3,2,1])==1.0 and C.harrell_c(y,ev,[1,2,3,4])==0.0
assert C.harrell_c([2,2,5],[1,0,0],[3,1,2])==1.0 and C.harrell_c([2,2,5],[1,0,0],[1,3,2])==0.0  # recurrence vs censoring at the same time: comparable
assert C.harrell_c([2,2,5],[1,1,0],[2,1,0])==1.0  # two recurrences at the same time are not a comparable pair
assert abs(C.spearman([1,2,2,3],[4,3,3,1])-C.spearman([1,2,2,3][::-1],[4,3,3,1][::-1]))<1e-12 and C.spearman([1,2,3],[1,2,3])==1.0
g=rng.random(200)<.5;chi,p=C.logrank(rng.exponential(1,200),np.ones(200,int),g);assert p>0.001
print(json.dumps(dict(st=st,check=[c['event'] for c in chk],cuts=cuts)))""")
    d=json.loads(out);assert d['st']['korea_primary']==295 and d['check']==[1,1,1,1,0,0,0,0] and len(d['cuts'])==3
    ok('inputs: 300 Korean (295 primary, DFS-0 cases only in test folds), 120 Scottish; 84-month censoring; quartile time bins; Cox fit recovers known coefficients; Harrell C and log-rank sanity')
    # ---- technical check (waits for the Scottish features)
    (w['P']/'transfer/verified_scotland_feat_uni').unlink();r=run(code,['check.py','--device','cpu'],env);assert r.returncode==3,r.stdout[-300:]
    (w['P']/'transfer/verified_scotland_feat_uni').touch()
    r=run(code,['check.py','--device','cpu'],env,0);chk=rjson(code/'CHECK_PASS.json')
    assert chk['state']=='PASS' and len(chk['results'])==2 and all(v['passed'] for v in chk['results'].values())
    assert 'c_index' not in json.dumps(chk).lower() and 'auc' not in json.dumps(chk).lower()
    ok('check: waits (3) while Scottish features are not verified; 2 encoders: uninterrupted vs paused + fresh-process resume bitwise identical; chunks agree; outcomes = database table; no metric recorded')
    A=['train.py','--encoder','phikon-v2','--seed-index','1','--fold','2','--device','cpu']
    r=run(code,A,env);assert r.returncode==78 and 'LOCK.json' in r.stdout
    run(code,['lock.py','write'],env,0)
    r=run(code,A,env);assert r.returncode==78 and 'cross-card' in r.stdout
    c=dict(rjson(code/'CHECK_PASS.json'),cross_card_resume_proved=True);wjson(code/'CHECK_PASS.json',c)
    ok('gates: no LOCK.json -> 78; no cross-card proof -> 78')
    r=run(code,A,env,0);out=code/'runs/phikon-v2/seed1/fold2';P=rjson(out/'predictions.json')
    assert sum(x['cohort']=='scotland' for x in P)==120 and sum(x['cohort']=='korea' for x in P)==60 and all(np.isfinite(x['risk']) for x in P)
    alt=work/'alt';run(code,A+['--outroot',str(alt),'--stop-after-epochs','2'],env,75);run(code,A+['--outroot',str(alt)],env,0)
    assert (alt/'phikon-v2/seed1/fold2/predictions.json').read_bytes()==(out/'predictions.json').read_bytes()
    ok('formal run: predicts its 60 Korean test cases (DFS-0 included) and all 120 Scottish cases; paused + resumed == uninterrupted (bitwise)')
    camp=rjson(code/'CAMPAIGN.json');assert len(camp['runs'])==50
    r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'REFUSED' in r.stdout
    def one(j):return run(code,[x if x!='cuda' else 'cpu' for x in j['argv']]+['--threads','1'],env).returncode
    with ThreadPoolExecutor(a.jobs) as ex:rcs=list(ex.map(one,camp['runs']))
    assert rcs==[0]*50,rcs
    r=run(code,['evaluate.py'],env,0);res=rjson(code/'RESULTS.json');m=res['encoders']['phikon-v2']['main']
    # independent recomputation of the Korean HE-alone C index (seed-averaged OOF risk, primary cases)
    ks={x['patient_id']:x for x in w['ksurv']}
    risk={};[risk.setdefault(x['patient_id'],[]).append(x['risk']) for s in range(5) for f in range(5) for x in rjson(code/f'runs/phikon-v2/seed{s}/fold{f}/predictions.json') if x['cohort']=='korea']
    prim=[p for p in risk if ks[p]['dfs_months']!=0];tt=np.array([min(ks[p]['dfs_months'],84) for p in prim]);ee=np.array([ks[p]['recurrence'] if ks[p]['dfs_months']<=84 else 0 for p in prim])
    raw={p:np.mean(risk[p]) for p in prim};z={}
    for f in range(5):  # within-fold standardisation with the fold's primary test cases
        v=np.array([raw[p] for p in prim if w['group'][p]==f]);z.update({p:(raw[p]-v.mean())/v.std() for p in prim if w['group'][p]==f})
    rr=np.array([z[p] for p in prim]);num=den=0
    for i in np.flatnonzero(ee==1):mm=(tt>tt[i])|((tt==tt[i])&(ee==0));den+=mm.sum();num+=(rr[i]>rr[mm]).sum()+.5*(rr[i]==rr[mm]).sum()
    assert abs(num/den-m['korea']['c_he_alone'])<1e-12
    E=res['encoders']['phikon-v2'];assert len(E['scale']['per_fold_scotland_delta_c'])==5 and 'scotland' in E['sensitivity_with_zero_dfs'] and res['cox_warnings']==[]
    assert all(x>0 for x in E['scale']['korea_fold_sd']) and E['if_exploratory']['korea']['n']==71
    assert m['korea']['c_he_alone']>0.6 and m['scotland']['c_he_alone']>0.6,(m['korea']['c_he_alone'],m['scotland']['c_he_alone'])
    assert (code/'RESULTS.md').read_text().count('ΔC')>=2 and res['encoders']['uni']['if_exploratory']['scotland']['n']==88
    ok(f"campaign: 50 runs; evaluate refuses before they exist; Korean HE C matches an independent recomputation; planted HE signal found (C {m['korea']['c_he_alone']:.2f} / {m['scotland']['c_he_alone']:.2f}); verdict {res['decision']['verdict']}")
    py("""import evaluate as E,common as C
cfg=C.config()
def r(d,lo,hr):return dict(scotland=dict(delta_c=d,hr_he_ci95=[lo,lo+1],hr_he_per_sd=hr))
for x,want in [(r(.02,1.01,1.3),'POSITIVE'),(r(.05,0.95,1.3),'INCONCLUSIVE'),(r(.019,1.2,1.5),'INCONCLUSIVE'),(r(0,1.2,1.5),'NEGATIVE'),(r(.03,0.5,0.9),'NEGATIVE')]:
    assert E.decide(x,cfg)==want,(x,E.decide(x,cfg),want)""")
    ok('decision rule: positive needs Scottish dC >= 0.02 and HR lower bound > 1; negative if dC <= 0 or HR <= 1; otherwise inconclusive')
    print(f'ALL_LOCAL_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; work dir {work}')
if __name__=='__main__':main()
