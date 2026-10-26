"""Local CPU tests on a synthetic Korean + Scottish mirror (never deployed; nothing here reads real features or outcomes).
The code is copied byte for byte into <work>/code except config.json (device cpu, the fake input files' sha256, small
shapes and feature widths) and the two inputs/*_surv.csv (the fake world's grades and outcomes). The fake world has a
latent continuous grade: ISUP is its noisy discretisation, the features carry it and recurrence depends on it (positive
control for both decisions). Usage: python tests/test_local.py [--workdir DIR] [--jobs N]"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1];CHECKS=[]
ENCS=['phikon-v2','uni','uni2-h','virchow2','gigapath','conch','resnet50','convnext_tiny','vit_b_16'];DIM=8
KGRADES=[111,129,44,16];SGRADES=[36,50,32,2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wjson(p,o):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,indent=2))
def rjson(p):return json.loads(Path(p).read_text())
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)
def wcsv(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build_world(work):
    rng=np.random.default_rng(11);P=work/'P';H=P/'hpcroot'
    K=H/'work/data/user/projects/korean_ccrcc_SBIAD2452';S=H/'work/data/user/projects/scottish_ccrcc_SBIAD2454'
    pool=K/'qc/primary_technical_pool_v1_20261010';splits=K/'bib_phase1/first_round_v1_20261010';full=K/'features/nine_encoder_full_v1_20261010'
    spool=S/'qc/external_primary_technical_pool_v1_20261013';sfull=S/'features/nine_encoder_full_v1_20261021'
    for d in (pool,splits,full,spool,sfull,K/'qc_deps',P/'transfer'):d.mkdir(parents=True,exist_ok=True)
    direction={e:rng.normal(0,1,DIM).astype(np.float32) for e in ENCS}
    def feats(folder,n,latent):
        folder.mkdir(parents=True,exist_ok=True);rec=dict(state='complete',n=n,features={})
        for e in ENCS:
            a=(rng.normal(0,1,(n,DIM)).astype(np.float32)+0.5*latent*direction[e]).astype(np.float16);np.save(folder/f'{e}.npy',a)
            (folder/f'{e}.npy.verified').touch();rec['features'][e]=dict(sha256=sha(folder/f'{e}.npy'))
        wjson(folder/'complete.json',rec)
    def cohort(n,counts):
        latent=rng.normal(0,1,n);noisy=latent+0.6*rng.normal(0,1,n);order=np.argsort(noisy);isup=np.empty(n,int);i=0
        for g,m in enumerate(counts,1):isup[order[i:i+m]]=g;i+=m
        leib=np.clip(np.round(3+1.2*rng.normal(0,1,n)+0.8*(isup-2)),0,11)
        lam=0.006*np.exp(0.15*(leib-3)+0.9*latent);t=rng.exponential(1/lam);c=rng.uniform(20,150,n)
        return latent,isup,leib,np.minimum(t,c),(t<=c).astype(int)
    lk,gk,Lk,tk,ek=cohort(300,KGRADES);tk[:5]=0.;ek[:5]=1  # five DFS-0 recurrences
    pts=[dict(patient_id=f'korea:KR{i+1}',slide_filename=f'HE_KR{i+1}.czi',ISUP_grade=str(int(gk[i])),primary_count=str(int(rng.integers(20,50)))) for i in range(300)]
    order=sorted(range(300),key=lambda i:(pts[i]['ISUP_grade'],rng.random()));group={pts[i]['patient_id']:r%5 for r,i in enumerate(order)}
    rows=[dict(fold=str(f),patient_id=p['patient_id'],slide_filename=p['slide_filename'],ISUP_grade=p['ISUP_grade'],
               role='test' if group[p['patient_id']]==f else 'val' if group[p['patient_id']]==(f+1)%5 else 'train') for f in range(5) for p in pts]
    wcsv(pool/'patient_manifest.csv',pts)
    with (splits/'patient_splits.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['fold','patient_id','slide_filename','ISUP_grade','role']);w.writeheader();w.writerows(rows)
    wjson(pool/'LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv')))
    wjson(splits/'SPLITS_LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv'),qc_lock_sha256=sha(pool/'LOCK.json'),split_sha256=sha(splits/'patient_splits.csv')))
    for i,p in enumerate(pts):feats(full/f'HE_KR{i+1}',int(p['primary_count']),lk[i])
    def surv_rows(ids,g,L,t,e):
        return [dict(patient_id=p,recurrence=int(e[i]),dfs_months=round(float(t[i]),3),leibovich_fuhrman=int(L[i]),leibovich_isup=int(max(0,L[i]-1)),
                     isup=int(g[i]),fuhrman=int(min(4,g[i]+(i%2))),t_stage='1a',size_cm=4.0,necrosis=0) for i,p in enumerate(ids)]
    ksurv=surv_rows([p['patient_id'] for p in pts],gk,Lk,tk,ek)
    ls,gs,Ls,ts,es=cohort(120,SGRADES)
    spts=[dict(patient_id=f'scotland:{1000+i}',clinical_id=str(1000+i),slide_filename=f'iCAIRD -{1000+i}-HE-St Andrews-11_06_2021.czi',ISUP_grade=str(int(gs[i])),primary_count=str(int(rng.integers(20,50)))) for i in range(120)]
    wcsv(spool/'patient_manifest.csv',spts);wjson(spool/'LOCK.json',dict(patient_manifest_sha256=sha(spool/'patient_manifest.csv')))
    for i,p in enumerate(spts):feats(sfull/Path(p['slide_filename']).stem,int(p['primary_count']),ls[i])
    ssurv=surv_rows([p['patient_id'] for p in spts],gs,Ls,ts,es)
    for e in ENCS:(P/'transfer'/f'verified_feat_{e}').touch();(P/'transfer'/f'verified_scotland_feat_{e}').touch()
    return dict(P=P,ksurv=ksurv,ssurv=ssurv)

def copy_code(w,dest):
    for n in ['PROTOCOL.md','common.py','train.py','check.py','evaluate.py','make_campaign.py','lock.py','CAMPAIGN.json']:
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/n,dest/n);assert sha(dest/n)==sha(SRC/n)
    shutil.copytree(SRC/'inputs',dest/'inputs');(dest/'tests').mkdir();shutil.copy2(SRC/'tests/test_local.py',dest/'tests/test_local.py')
    for q in [dest,*dest.rglob('*')]:q.chmod(q.stat().st_mode|0o200)
    for n,rows in (('korea_surv.csv',w['ksurv']),('scotland_surv.csv',w['ssurv'])):wcsv(dest/'inputs'/n,rows)
    cfg=json.loads((SRC/'config.json').read_text());assert cfg['device']=='cuda' and list(cfg['encoders'])==ENCS
    cfg.update(device='cpu',train_patches=16,eval_chunk=64,epochs=6,learning_rate=0.003,encoders={e:DIM for e in ENCS})
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
    work=Path(a.workdir or tempfile.mkdtemp(prefix='gx1_test_'));shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True)
    t0=time.monotonic();w=build_world(work);code=copy_code(w,work/'code')
    env=dict(os.environ,HE_PATH_PREFIX=str(w['P']/'hpcroot'),HE_PROJECT_DIR=str(w['P']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    out=run(code,['-c',"""import common as C,json,numpy as np
cfg=C.config();k,assign=C.load_korea(cfg);s=C.load_scotland(cfg);st=C.label_structure(k,assign,s,cfg);chk=C.check_cases(k,assign,cfg)
r=C.roles(k,assign,0,cfg);assert any(c['zero_dfs'] for f in range(5) for c in C.roles(k,assign,f,cfg)['train']) and len(C.primary(k,cfg))==295
P=np.eye(4)[[0,1,2,3,1]];g=[1,2,3,4,2];assert C.highlow_auc(g,P)==1.0 and C.g12_auc(g,P)==1.0 and C.grade_metrics(g,P)['qwk']==1.0
assert np.allclose(C.grade_score([[.1,.2,.3,.4]]),[3.0]) and C.highlow_auc([1,3],[[.9,.1,0,0],[.2,.2,.3,.3]])==1.0
assert C.harrell_c_within([1,2,3,4],[1,1,1,0],[2,1,4,3],[1,1,2,2])==1.0 and C.harrell_c_within([1,2,3,4],[1,1,1,0],[1,2,3,4],[1,1,2,2])==0.0
print(json.dumps(dict(st=st,check=[c['isup'] for c in chk])))"""],env,0).stdout
    d=json.loads(out);assert d['st']['korea_isup']=={'1':111,'2':129,'3':44,'4':16} and d['st']['scotland_isup']=={'1':36,'2':50,'3':32,'4':2} and d['check']==[1,1,2,2,3,3,4,4]
    ok('inputs: 300 Korean (111/129/44/16; all graded, DFS-0 cases train too; 295 in the survival set), 120 Scottish; check cases 2 per grade; grading metrics and within-grade C sanity')
    T=w['P']/'transfer'
    for m in ('verified_feat_conch','verified_scotland_feat_vit_b_16'):
        (T/m).unlink();r=run(code,['check.py','--device','cpu'],env);assert r.returncode==3,(m,r.stdout[-300:]);(T/m).touch()
    r=run(code,['check.py','--device','cpu'],env,0);chk=rjson(code/'CHECK_PASS.json')
    assert chk['state']=='PASS' and len(chk['results'])==9 and all(v['passed'] for v in chk['results'].values())
    assert 'auc' not in json.dumps(chk).lower() and 'c_index' not in json.dumps(chk).lower()
    ok('check: waits (3) while any encoder lacks Korean or Scottish features; 9 encoders: uninterrupted vs paused + fresh-process resume bitwise identical; chunks agree; grades = database table; no metric recorded')
    A=['train.py','--encoder','gigapath','--seed-index','1','--fold','2','--device','cpu']
    r=run(code,A,env);assert r.returncode==78 and 'LOCK.json' in r.stdout
    run(code,['lock.py','write'],env,0)
    r=run(code,A,env);assert r.returncode==78 and 'cross-card' in r.stdout
    c=dict(rjson(code/'CHECK_PASS.json'),cross_card_resume_proved=True);wjson(code/'CHECK_PASS.json',c)
    ok('gates: no LOCK.json -> 78; no cross-card proof -> 78')
    r=run(code,A,env,0);out=code/'runs/gigapath/seed1/fold2';P=rjson(out/'predictions.json')
    assert sum(x['cohort']=='scotland' for x in P)==120 and sum(x['cohort']=='korea' for x in P)==60
    assert all(abs(sum(x['probs'])-1)<1e-9 and abs(x['grade_score']-sum((k+1)*p for k,p in enumerate(x['probs'])))<1e-9 for x in P)
    alt=work/'alt';run(code,A+['--outroot',str(alt),'--stop-after-epochs','2'],env,75);run(code,A+['--outroot',str(alt)],env,0)
    assert (alt/'gigapath/seed1/fold2/predictions.json').read_bytes()==(out/'predictions.json').read_bytes()
    md=rjson(out/'metadata.json');assert sum(md['class_counts'])==240 and len(md['labels_used'])==240
    ok('formal run: predicts its 60 Korean test cases and all 120 Scottish cases (probabilities sum to 1; grade score = sum k P(k)); paused + resumed == uninterrupted (bitwise)')
    camp=rjson(code/'CAMPAIGN.json');assert len(camp['runs'])==225
    r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'REFUSED' in r.stdout
    def one(j):return run(code,[x if x!='cuda' else 'cpu' for x in j['argv']]+['--threads','1'],env).returncode
    with ThreadPoolExecutor(a.jobs) as ex:rcs=list(ex.map(one,camp['runs']))
    assert rcs==[0]*225,rcs
    r=run(code,['evaluate.py'],env,0);res=rjson(code/'RESULTS.json');E=res['encoders']['phikon-v2']
    # independent recomputation: Scottish high/low AUC and AI-grade C of the primary encoder from the 25 models' predictions
    sv={x['patient_id']:x for x in w['ssurv']};ids=sorted(sv);pr={p:[] for p in ids}
    for s in range(5):
        for f in range(5):
            for x in rjson(code/f'runs/phikon-v2/seed{s}/fold{f}/predictions.json'):
                if x['cohort']=='scotland':pr[x['patient_id']].append(x['probs'])
    Pm=np.array([np.mean(pr[p],0) for p in ids]);hi=np.array([int(sv[p]['isup'])>=3 for p in ids]);sc=Pm[:,2]+Pm[:,3]
    aucv=np.mean([(sc[i]>sc[~hi]).mean()+.5*(sc[i]==sc[~hi]).mean() for i in np.flatnonzero(hi)])
    assert abs(aucv-E['scotland']['grading']['highlow_auc'])<1e-12
    g=Pm@np.arange(1,5);t=np.array([min(float(sv[p]['dfs_months']),84) for p in ids]);e=np.array([int(sv[p]['recurrence']) if float(sv[p]['dfs_months'])<=84 else 0 for p in ids])
    num=den=0
    for i in np.flatnonzero(e==1):m=(t>t[i])|((t==t[i])&(e==0));den+=m.sum();num+=(g[i]>g[m]).sum()+.5*(g[i]==g[m]).sum()
    assert abs(num/den-E['scotland']['prognosis']['c_ai_grade'])<1e-12
    assert len(res['encoders'])==9 and res['benchmark']['selected_by_korea'] in res['encoders'] and res['cox_warnings']==[] and all(len(v)==5 for v in E['per_seed'].values())
    assert E['korea']['prognosis']['n']==295 and E['korea']['grading']['n']==300 and (code/'RESULTS.md').read_text().count('|')>50
    D=res['decision'];k=E['scotland']
    assert k['grading']['highlow_auc']>0.8 and k['prognosis']['delta_c_vs_isup']>0,(k['grading']['highlow_auc'],k['prognosis']['delta_c_vs_isup'])
    ok(f"campaign: 225 runs; evaluate refuses before they exist; Scottish high/low AUC and AI-grade C = independent recomputation; planted signal found "
       f"(AUC {k['grading']['highlow_auc']:.2f}, dC {k['prognosis']['delta_c_vs_isup']:+.3f}); decisions {D['grading']} / {D['ai_grade']}")
    run(code,['-c',"""import evaluate as E,common as C
cfg=C.config()
def r(auc,lo,dc,hlo,hr):return dict(scotland=dict(grading=dict(highlow_auc=auc,highlow_auc_ci95=[lo,1]),prognosis=dict(delta_c_vs_isup=dc,cox_isup_plus_ai=dict(hr_ai_per_grade=hr,hr_ai_ci95=[hlo,hlo+1]))))
for x,want in [(r(.80,.71,.02,1.01,1.5),('SUCCESS','POSITIVE')),(r(.85,.70,.05,.9,1.3),('INCONCLUSIVE','INCONCLUSIVE')),(r(.69,.6,0,1.2,1.5),('FAIL','NEGATIVE')),
               (r(.75,.66,.03,.5,.95),('INCONCLUSIVE','NEGATIVE')),(r(.79,.75,.019,1.1,1.6),('INCONCLUSIVE','INCONCLUSIVE'))]:
    got=E.decide(x,cfg);assert (got['grading'],got['ai_grade'])==want and got['verdict']==f'grading={want[0]}; ai_grade={want[1]}',(x,got,want)
import numpy as np
x=r(.9,.85,.05,1.2,1.5);x['scotland']['prognosis']['cox_isup_plus_ai']=dict(ok=False,hr_ai_per_grade=None,hr_ai_ci95=None);assert E.decide(x,cfg)['ai_grade']=='INCONCLUSIVE'
x['scotland']['prognosis']['delta_c_vs_isup']=0.;assert E.decide(x,cfg)['ai_grade']=='NEGATIVE'
t=np.arange(1,41.);e=np.ones(40,int);g=np.repeat([1.,2.,3.,4.],10)
b,se,ll,ok=E.cox_safe(np.c_[g,g],t,e,'collinear');assert not ok and C.COX_WARNINGS[-1]['fit']=='collinear'  # exact collinearity: flagged, no crash
h=E.hr_of(b,se,1,ok);assert h['ok'] is False and h['hr_ai_per_grade'] is None"""],env,0)
    ok('decision rules: grading SUCCESS needs AUC >= 0.80 and CI lower > 0.70, FAIL below 0.70; AI grade POSITIVE needs dC >= 0.02 and HR lower bound > 1, NEGATIVE if dC <= 0 or HR <= 1; a failed Cox fit is flagged, never positive, never a crash')
    print(f'ALL_LOCAL_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; work dir {work}')
if __name__=='__main__':main()
