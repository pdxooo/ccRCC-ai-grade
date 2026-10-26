"""Local CPU tests on a synthetic mirror (never deployed; nothing here reads real features or labels).
The code is copied byte for byte into <work>/code except config.json (device cpu, the fake label file's sha256, small
shapes) and inputs/korea_labels.csv (the fake world's labels). Usage: python tests/test_local.py [--workdir DIR] [--jobs 8]"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1];HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
GRADES={1:111,2:129,3:44,4:16};DIMS={'phikon-v2':1024,'uni':1024};CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wjson(p,o):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,indent=2))
def rjson(p):return json.loads(Path(p).read_text())
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)

def build_world(work):
    rng=np.random.default_rng(11);P=work/'P';root=P/'hpcroot'/HPC_ROOT.lstrip('/')
    pool=root/'qc/primary_technical_pool_v1_20261010';splits=root/'bib_phase1/first_round_v1_20261010';full=root/'features/nine_encoder_full_v1_20261010'
    for d in (pool,splits,full,root/'qc_deps',P/'transfer'):d.mkdir(parents=True,exist_ok=True)
    grades=np.concatenate([[g]*n for g,n in GRADES.items()]);rng.shuffle(grades)
    pts=[dict(patient_id=f'korea:KR{i+1}',slide_filename=f'HE_KR{i+1}.czi',ISUP_grade=str(int(grades[i])),primary_count=str(int(rng.integers(20,60)))) for i in range(300)]
    order=sorted(range(300),key=lambda i:(pts[i]['ISUP_grade'],rng.random()));group={pts[i]['patient_id']:r%5 for r,i in enumerate(order)}
    rows=[]
    for fold in range(5):
        for p in pts:
            g=group[p['patient_id']];role='test' if g==fold else 'val' if g==(fold+1)%5 else 'train'
            rows.append(dict(fold=str(fold),patient_id=p['patient_id'],slide_filename=p['slide_filename'],ISUP_grade=p['ISUP_grade'],role=role))
    with (pool/'patient_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient_id','slide_filename','ISUP_grade','primary_count']);w.writeheader();w.writerows(pts)
    with (splits/'patient_splits.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['fold','patient_id','slide_filename','ISUP_grade','role']);w.writeheader();w.writerows(rows)
    wjson(pool/'LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv')))
    wjson(splits/'SPLITS_LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv'),qc_lock_sha256=sha(pool/'LOCK.json'),split_sha256=sha(splits/'patient_splits.csv')))
    # Fuhrman correlated with ISUP like the real table (ISUP 1: mostly 2-3, ISUP 4: 3-4)
    fu={p['patient_id']:int(rng.choice([1,2,3,4],p={'1':[.02,.6,.33,.05],'2':[0,.42,.55,.03],'3':[0,.05,.85,.1],'4':[0,0,.5,.5]}[p['ISUP_grade']])) for p in pts}
    mu={(e,g):rng.normal(0,1,d).astype(np.float32)*.03 for e,d in DIMS.items() for g in range(1,5)}
    for p in pts:
        stem=Path(p['slide_filename']).stem;n=int(p['primary_count']);(full/stem).mkdir(parents=True,exist_ok=True);rec=dict(state='complete',n=n,patient_id=p['patient_id'],features={})
        for e,d in DIMS.items():
            a=(rng.normal(0,1,(n,d)).astype(np.float32)+mu[(e,int(p['ISUP_grade']))]).astype(np.float16);np.save(full/stem/f'{e}.npy',a)
            (full/stem/f'{e}.npy.verified').touch();rec['features'][e]=dict(sha256=sha(full/stem/f'{e}.npy'))
        wjson(full/stem/'complete.json',rec)
    for e in DIMS:(P/'transfer'/f'verified_feat_{e}').touch()
    return dict(P=P,root=root,full=full,pts=pts,fu=fu)

def copy_code(w,dest):
    for n in ['PROTOCOL.md','common.py','train.py','check.py','evaluate.py','make_labels.py','make_campaign.py','lock.py','CAMPAIGN.json']:
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/n,dest/n);assert sha(dest/n)==sha(SRC/n)
    shutil.copytree(SRC/'inputs',dest/'inputs');(dest/'tests').mkdir();shutil.copy2(SRC/'tests/test_local.py',dest/'tests/test_local.py')
    with (dest/'inputs/korea_labels.csv').open('w',newline='') as f:  # the fake world's labels (the real file names real patients)
        wr=csv.DictWriter(f,fieldnames=['patient_id','ISUP_grade','Fuhrman_grade']);wr.writeheader()
        wr.writerows([dict(patient_id=p['patient_id'],ISUP_grade=p['ISUP_grade'],Fuhrman_grade=w['fu'][p['patient_id']]) for p in w['pts']])
    cfg=json.loads((SRC/'config.json').read_text());assert cfg['device']=='cuda'
    cfg.update(device='cpu',labels_csv_sha256=sha(dest/'inputs/korea_labels.csv'),train_patches=16,eval_chunk=64,epochs=3)
    cfg['check'].update(chunk_sizes=[64,0,7]);cfg['analysis']['bootstrap_replicates']=200
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    return dest

def run(code,args,env,expect=None):
    r=subprocess.run([sys.executable,'-u',*args],cwd=code,env=env,capture_output=True,text=True)
    with (code.parent/'logs.txt').open('a') as f:f.write(f'$ {" ".join(args)}\n{r.stdout}\n{r.stderr}\nEXIT {r.returncode}\n')
    if expect is not None and r.returncode!=expect:
        print(r.stdout[-3000:]);print(r.stderr[-3000:]);raise AssertionError(f'{args} exit {r.returncode}, expected {expect}')
    return r

def same_dir_results(a,b):
    import torch
    A=torch.load(a/'last.pt',map_location='cpu',weights_only=False);B=torch.load(b/'last.pt',map_location='cpu',weights_only=False)
    assert A['model'].keys()==B['model'].keys() and all(torch.equal(A['model'][k],B['model'][k]) for k in A['model'])
    assert (a/'predictions.json').read_bytes()==(b/'predictions.json').read_bytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir');ap.add_argument('--jobs',type=int,default=8);a=ap.parse_args()
    work=Path(a.workdir or tempfile.mkdtemp(prefix='fx1_test_'));shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True)
    t0=time.monotonic();w=build_world(work);code=copy_code(w,work/'code')
    env=dict(os.environ,HE_PATH_PREFIX=str(w['P']/'hpcroot'),HE_PROJECT_DIR=str(w['P']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    py=lambda s:run(code,['-c',s],env,0).stdout
    out=py("""import common as C,json,numpy as np
cfg=C.config();cases,assign=C.load_cases(cfg);s=C.label_structure(cases,assign,cfg);k=C.check_cases(cases,assign,cfg)
r=C.roles(cases,assign,0,cfg);a=C.aux_labels(r['train'],'C',cfg,1,0);b=C.aux_labels(r['train'],'B',cfg,1,0)
moved=sum(a[p]!=b[p] for p in a)
print(json.dumps(dict(n=len(cases),check=[c['isup'] for c in k],moved=moved,A=C.aux_labels(r['train'],'A',cfg,1,0))))
assert C.auc([0,0,1,1],[.1,.4,.35,.8])==0.75
P=np.array([[.7,.2,.05,.05],[.2,.6,.1,.1],[.1,.1,.4,.4],[.25,.25,.25,.25]]);g=np.array([1,2,3,4])
assert C.g12_auc(g,P)==1.0 and C.highlow_auc(g,P)==1.0 and abs(C.grade_metrics(g,P)['accuracy']-0.75)<1e-12""")
    d=json.loads(out);assert d['n']==300 and d['check']==[1,1,2,2,3,3,4,4] and d['moved']>0 and d['A']=={}
    ok('locked inputs: 300 cases, database ISUP == manifest, 240/60 per fold, each tested once; arm C permutes aux labels within ISUP grade (counts kept, %d of 240 moved); 8 check cases (2 per grade); metrics on a hand example'%d['moved'])
    lab=code/'inputs/korea_labels.csv';orig=lab.read_bytes();rows=list(csv.DictReader(open(lab)));rows[3]['ISUP_grade']=str(5-int(rows[3]['ISUP_grade']))
    with lab.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    cfgp=code/'config.json';c0=cfgp.read_text();cj=json.loads(c0);cj['labels_csv_sha256']=sha(lab);cfgp.write_text(json.dumps(cj,indent=2))
    r=run(code,['check.py','--device','cpu'],env);assert r.returncode==1 and 'database ISUP' in r.stderr,r.stderr[-400:]
    lab.write_bytes(orig);cfgp.write_text(c0)
    ok('refusal: a label file whose ISUP disagrees with the manifest -> exit 1 before anything runs')
    # ---- technical check
    shutil.rmtree(code/'check_runs',ignore_errors=True);(code/'CHECK_PASS.json').unlink(missing_ok=True)
    r=run(code,['check.py','--device','cuda'],env);assert r.returncode==78
    r=run(code,['check.py','--device','cpu'],env,0);chk=rjson(code/'CHECK_PASS.json')
    assert chk['state']=='PASS' and len(chk['results'])==6 and all(v['passed'] for v in chk['results'].values())
    txt=json.dumps(chk).lower();assert 'auc' not in txt and 'accuracy' not in txt and 'kappa' not in txt
    ok('check: 2 encoders x arms A/B/C: uninterrupted vs paused + fresh-process resume bitwise identical; chunks 64 / whole / 7 agree; ISUP labels = database file; B aux = own Fuhrman, C aux = within-grade permutation, A none; no metric in the record')
    # ---- gates
    A=['train.py','--encoder','phikon-v2','--arm','B','--seed-index','1','--fold','2','--device','cpu']
    r=run(code,A,env);assert r.returncode==78 and 'LOCK.json' in r.stdout,r.stdout[-300:]
    run(code,['lock.py','write'],env,0)
    r=run(code,A,env);assert r.returncode==78 and 'cross-card' in r.stdout,r.stdout[-300:]
    c=dict(rjson(code/'CHECK_PASS.json'),cross_card_resume_proved=True);wjson(code/'CHECK_PASS.json',c)  # stands in for a two-GPU check
    wjson(code/'CHECK_PASS.json',dict(c,platform=dict(c['platform'],torch='0.0')));r=run(code,A,env);assert r.returncode==78 and 'another platform' in r.stdout
    wjson(code/'CHECK_PASS.json',c)
    ev=(code/'evaluate.py').read_bytes();(code/'evaluate.py').write_bytes(ev+b'\n');r=run(code,A,env);assert r.returncode==78 and 'locked' in r.stdout;(code/'evaluate.py').write_bytes(ev)
    ok('gates: no LOCK.json -> 78; check without cross-card proof -> 78; check on another platform -> 78; a locked file changed -> 78')
    # ---- one formal run, pause/resume, already complete
    r=run(code,A,env,0);out=code/'runs/phikon-v2/B/seed1/fold2';P=rjson(out/'predictions.json')
    assert len(P)==60 and all(x['fold']==2 and len(x['probabilities'])==4 for x in P) and 'auc' not in r.stdout.lower()
    md=rjson(out/'metadata.json');assert md['aux_labels_used']=={p:int(w['fu'][p]>=3) for p in md['labels_used']}
    alt=work/'alt';run(code,A+['--outroot',str(alt),'--stop-after-epochs','1'],env,75);run(code,A+['--outroot',str(alt)],env,0)
    same_dir_results(out,alt/'phikon-v2/B/seed1/fold2');run(code,A,env,0)
    ok('formal run: 60 test predictions with 4 ISUP probabilities, nothing about performance printed; B trains on each case\'s own Fuhrman 1-2 vs 3-4; paused + resumed == uninterrupted (bitwise); re-run -> ALREADY_COMPLETE')
    # ---- the whole campaign from CAMPAIGN.json, then the analysis
    camp=rjson(code/'CAMPAIGN.json');assert len(camp['runs'])==150
    r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'REFUSED' in r.stdout
    def one(j):return run(code,[x if x!='cuda' else 'cpu' for x in j['argv']]+['--threads','1'],env).returncode
    with ThreadPoolExecutor(a.jobs) as ex:rcs=list(ex.map(one,camp['runs']))
    assert rcs==[0]*150,rcs
    assert all((code/j['complete']).exists() for j in camp['runs'])
    r=run(code,['evaluate.py'],env,0);res=rjson(code/'RESULTS.json');assert res['decision']['verdict'] in ('POSITIVE','NEGATIVE','INCONCLUSIVE')
    cases=sorted(w['pts'],key=lambda p:p['patient_id']);ids=[p['patient_id'] for p in cases];g=np.array([int(p['ISUP_grade']) for p in cases])
    Pr=np.mean([[{x['patient_id']:x['probabilities'] for f in range(5) for x in rjson(code/f'runs/phikon-v2/B/seed{s}/fold{f}/predictions.json')}[p] for p in ids] for s in range(5)],0)
    m=g<=2;sc=Pr[m,1]/(Pr[m,0]+Pr[m,1]);yy=g[m]==2;ref=np.mean([(p>q)+.5*(p==q) for p in sc[yy] for q in sc[~yy]])
    assert abs(ref-res['encoders']['phikon-v2']['arms']['B']['g12_auc'])<1e-12
    assert (code/'RESULTS.md').read_text().count('Δ(B−A)')==2 and set(res['encoders']['uni']['aux_head_fuhrman_auc'])=={'B','C'}
    shutil.move(code/'runs/uni/C/seed3/fold1/complete.json',work/'moved.json');r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'uni/C/seed3/fold1' in r.stdout
    shutil.move(work/'moved.json',code/'runs/uni/C/seed3/fold1/complete.json')
    ok(f"campaign: all 150 CAMPAIGN.json runs complete; evaluate refuses with any run missing; seed-averaged G1/G2 AUC matches an independent recomputation; verdict {res['decision']['verdict']}")
    # ---- decision rule (PROTOCOL.md section 6) on hand-made numbers
    py("""import evaluate as E,common as C
cfg=C.config()
def st(d,ds,bc,bcs):return {'phikon-v2':dict(delta_B_minus_A=d,seed_delta_B_minus_A=ds,delta_B_minus_C=bc,seed_delta_B_minus_C=bcs),'uni':dict(delta_B_minus_A=-1,seed_delta_B_minus_A=[-1]*5,delta_B_minus_C=-1,seed_delta_B_minus_C=[-1]*5)}
cases=[(st(.02,[.01,.02,.03,.01,-.01],.01,[.01,.01,.02,.0,.03]),'POSITIVE'),(st(.02,[.01,.02,.03,.01,-.01],.001,[.01]*5),'INCONCLUSIVE'),
       (st(.03,[.01]*5,.02,[.01,.01,-.01,-.01,.03]),'INCONCLUSIVE'),(st(.05,[.01,.02,.03,-.01,-.01],.01,[.01]*5),'INCONCLUSIVE'),
       (st(.05,[.01]*5,0.0,[.01]*5),'NEGATIVE'),(st(0.0,[.01]*5,.01,[.01]*5),'NEGATIVE'),(st(.019,[.01]*5,.01,[.01]*5),'INCONCLUSIVE'),
       (st(-.01,[.01]*5,.01,[.01]*5),'NEGATIVE')]
for s,want in cases:
    got=E.decide(s,cfg)[0];assert got==want,(s,got,want)""")
    ok('decision rule: positive needs B-A >= 0.02 with >= 4 of 5 seeds positive AND B-C >= 0.01 with >= 4 of 5 seeds positive on the primary encoder; negative if B-A <= 0 or B <= C; otherwise inconclusive')
    print(f'ALL_LOCAL_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; work dir {work}')
if __name__=='__main__':main()
