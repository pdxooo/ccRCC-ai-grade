"""Local CPU tests on a synthetic mirror (never deployed; nothing here reads real stage-1 features).
The code is copied byte for byte into <work>/code except config.json: device -> cpu, the fake world's case-list sha256,
and small shapes (train_locations, eval_chunk, epochs, check chunk sizes) so 120 real runs finish in minutes.
Usage: python tests/test_local.py [--workdir DIR] [--jobs 8]"""
import argparse,csv,hashlib,json,os,shutil,signal,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1];HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
GRADES={1:111,2:129,3:44,4:16};DIMS={'uni':1024,'virchow2':2560};CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wjson(p,o):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,indent=2))
def rjson(p):return json.loads(Path(p).read_text())
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)

def build_world(work):
    rng=np.random.default_rng(7);P=work/'P';root=P/'hpcroot'/HPC_ROOT.lstrip('/')
    pool=root/'qc/primary_technical_pool_v1_20261010';splits=root/'bib_phase1/first_round_v1_20261010'
    full=root/'features/nine_encoder_full_v1_20261010';hr=root/'features/highres_pilot_v1_20261021_stage1_a6000'
    for d in (pool,splits,full,hr,root/'qc_deps',P/'transfer/lists'):d.mkdir(parents=True,exist_ok=True)
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
    # stage-1 lock exactly as select_stage1.py: 12 G1 + 12 G2 per outer test fold
    test=[r for r in rows if r['role']=='test'];lk={p['patient_id']:p for p in pts};picked=[]
    for fold in range(5):
        for g in ('1','2'):
            pool_ids=sorted(r['patient_id'] for r in test if int(r['fold'])==fold and lk[r['patient_id']]['ISUP_grade']==g)
            sel=sorted(np.random.default_rng([20261021,fold,int(g)]).choice(pool_ids,12,replace=False).tolist())
            picked+=[dict(patient_id=p,fold=fold,ISUP_grade=int(g),slide_stem=lk[p]['slide_filename'].rsplit('.',1)[0]) for p in sel]
    lock=dict(seed=20261021,manifest_sha256=sha(pool/'patient_manifest.csv'),splits_sha256=sha(splits/'patient_splits.csv'),cases=picked)
    lock['list_sha256']=hashlib.sha256(json.dumps(picked,sort_keys=True).encode()).hexdigest()
    wjson(P/'transfer/lists/STAGE1_CASES_LOCK.json',lock);(P/'transfer/lists/stage1_cases.txt').write_text('\n'.join(c['slide_stem'] for c in picked)+'\n')
    mu={(e,y):rng.normal(0,1,d).astype(np.float32)*.02 for e,d in DIMS.items() for y in (0,1)}  # weak signal: AUCs well below 1
    for c in picked:
        y=c['ISUP_grade']-1;stem=c['slide_stem'];n=int(lk[c['patient_id']]['primary_count']);m=min(n,40)
        (full/stem).mkdir(parents=True,exist_ok=True);(hr/stem).mkdir(parents=True,exist_ok=True)
        frec=dict(state='complete',n=n,patient_id=c['patient_id'],features={});files={};stored={}
        idx=np.sort(rng.choice(n,m,replace=False))
        for e,d in DIMS.items():
            a=(rng.normal(0,1,(n,d)).astype(np.float32)+mu[(e,y)]).astype(np.float16);np.save(full/stem/f'{e}.npy',a)
            frec['features'][e]=dict(sha256=sha(full/stem/f'{e}.npy'));stored[e]=frec['features'][e]['sha256']
            L=a[idx];H=(rng.normal(0,1,(m,4,d)).astype(np.float32)+mu[(e,y)]*1.5).astype(np.float16);Hd=(np.repeat(L[:,None].astype(np.float32),4,1)+rng.normal(0,.1,(m,4,d))).astype(np.float16)
            for arm,x in (('L',L),('H',H),('Hd',Hd)):np.save(hr/stem/f'{e}_{arm}.npy',x);files[f'{e}_{arm}.npy']=sha(hr/stem/f'{e}_{arm}.npy')
        wjson(full/stem/'complete.json',frec)
        np.save(hr/stem/'location_indices.npy',idx);files['location_indices.npy']=sha(hr/stem/'location_indices.npy')
        wjson(hr/stem/'complete.json',dict(state='complete',status='PASS',patient_id=c['patient_id'],slide_stem=stem,n_locations=m,primary_count=n,
                                          files=files,stored_feature_sha256=stored,quarantined=True))
    conds=['stage0_automated_checks_passed_on_a6000: x','stage0_visual_review_signed_off: x','stage0_platform_parity_passed: x','stage0_training_check_passed: x']
    wjson(hr/'QUARANTINE.json',dict(state='QUARANTINED',release_conditions=conds,list_sha256=lock['list_sha256']))
    return dict(P=P,root=root,hr=hr,full=full,lock=lock,conds=[c.split(':')[0] for c in conds])

def copy_code(w,dest):
    for n in ['PROTOCOL.md','common.py','train.py','check76.py','evaluate.py','make_prior_errors.py','lock.py']:
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/n,dest/n);assert sha(dest/n)==sha(SRC/n)
    shutil.copytree(SRC/'inputs',dest/'inputs');(dest/'tests').mkdir();shutil.copy2(SRC/'tests/test_local.py',dest/'tests/test_local.py')
    # the real prior-error list names real patients; the fake world's grades are random, so give it its own G2-only list
    g2=[c['patient_id'] for c in w['lock']['cases'] if c['ISUP_grade']==2]
    pj=json.loads((dest/'inputs/prior_all_wrong_isup2_v1.json').read_text());pj.update(isup2_all_wrong=g2[:20],isup2_all_called_G1=g2[:12])
    (dest/'inputs/prior_all_wrong_isup2_v1.json').write_text(json.dumps(pj,indent=1,sort_keys=True))
    cfg=json.loads((SRC/'config.json').read_text());assert cfg['device']=='cuda'
    cfg.update(device='cpu',stage1_list_sha256=w['lock']['list_sha256'],max_locations=40,train_locations=16,eval_chunk=64,epochs=3)
    cfg['check76'].update(chunk_sizes=[64,0,7]);cfg['analysis']['bootstrap_replicates']=300
    (dest/'config.json').write_text(json.dumps(cfg,indent=2))
    return dest

def run(code,args,env,expect=None,tag=''):
    r=subprocess.run([sys.executable,'-u',*args],cwd=code,env=env,capture_output=True,text=True)
    with (code.parent/'logs.txt').open('a') as f:f.write(f'$ {" ".join(args)}\n{r.stdout}\n{r.stderr}\nEXIT {r.returncode}\n')
    if expect is not None and r.returncode!=expect:
        print(r.stdout[-3000:]);print(r.stderr[-3000:]);raise AssertionError(f'{args} exit {r.returncode}, expected {expect}')
    return r

def release(w,conds=None):
    conds=w['conds'] if conds is None else conds
    wjson(w['hr']/'RELEASE.json',dict(conditions={c:dict(met=True,by='test',date='2026-10-21') for c in conds}))

def same_dir_results(a,b):
    import torch
    A=torch.load(a/'last.pt',map_location='cpu',weights_only=False);B=torch.load(b/'last.pt',map_location='cpu',weights_only=False)
    assert A['model'].keys()==B['model'].keys() and all(torch.equal(A['model'][k],B['model'][k]) for k in A['model'])
    assert (a/'predictions.json').read_bytes()==(b/'predictions.json').read_bytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir');ap.add_argument('--jobs',type=int,default=8);a=ap.parse_args()
    work=Path(a.workdir or tempfile.mkdtemp(prefix='hr_stage1_test_'));shutil.rmtree(work,ignore_errors=True);work.mkdir(parents=True)
    t0=time.monotonic();w=build_world(work);code=copy_code(w,work/'code')
    env=dict(os.environ,HE_PATH_PREFIX=str(w['P']/'hpcroot'),HE_PROJECT_DIR=str(w['P']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    py=lambda s:run(code,['-c',s],env,0).stdout
    # ---- unit checks through the real modules
    out=py("""import common as C,json
cfg=C.config();cases,order=C.load_cases(cfg);s=C.label_structure(cases,cfg);k=C.check_cases(cases,order,cfg)
print(json.dumps(dict(n=len(cases),folds=s['folds'],check=[(c['stem'],c['y']) for c in k],first=order[:12])))
import numpy as np
assert C.auc([0,0,1,1],[.1,.4,.35,.8])==0.75 and C.auc([0,1],[.5,.5])==0.5
y=np.r_[np.zeros(30),np.ones(40)];s=np.random.default_rng(1).random(70)
assert abs(C.auc(y,np.round(s,1))-np.mean([(p>q)+.5*(p==q) for p in np.round(s,1)[y==1] for q in np.round(s,1)[y==0]]))<1e-12
a=C.sample_locations(40,16,cfg,1,2,3,'korea:KR5');b=C.sample_locations(40,16,cfg,1,2,3,'korea:KR5');c=C.sample_locations(40,16,cfg,1,2,4,'korea:KR5')
assert (a==b).all() and not (a==c).all() and len(set(a))==16 and (np.diff(a)>0).all() and (C.sample_locations(10,16,cfg,0,0,0,'x')==np.arange(10)).all()""")
    d=json.loads(out);assert d['n']==120 and all(v=={'train':[48,48],'test':[12,12]} for v in d['folds'].values())
    assert [y for _,y in d['check']].count(0)==4 and [y for _,y in d['check']].count(1)==4
    ok('locked inputs: 120 cases, 48/48 train and 12/12 test per fold, each tested once; 7.6 cases = first 4 G1 + first 4 G2 of stage1_cases.txt; AUC with ties; shared location sampling')
    py("""import train as T,common as C,numpy as np,torch
cfg=C.config();cases,_=C.load_cases(cfg);c=cases[0];b=T.Bags(cfg,'uni','H',torch.device('cpu'));b.plan([c]);b.preload([c])
H=np.load(C.feature_path(c,'uni','H'));locs=np.array([0,3,5])
assert torch.equal(b.tensor(c['patient_id'],locs),torch.from_numpy(H[locs].reshape(-1,1024).astype(np.float32)))
l=T.Bags(cfg,'uni','L',torch.device('cpu'));l.plan([c]);l.preload([c]);assert l.locations[c['patient_id']]==b.locations[c['patient_id']]
f=T.Bags(cfg,'virchow2','Lfull',torch.device('cpu'));f.plan([c]);f.preload([c]);assert f.arrays[c['patient_id']].shape==(c['primary_count'],2560)""")
    ok('Bags: H/Hd location i -> instances 4i..4i+3 (location-major); L and H have the same locations; Lfull uses the full-slide array')
    # ---- integrity refusals
    lockf=w['P']/'transfer/lists/STAGE1_CASES_LOCK.json';orig=lockf.read_text();bad=json.loads(orig);bad['cases'][0]['ISUP_grade']=3-bad['cases'][0]['ISUP_grade']
    bad['list_sha256']=hashlib.sha256(json.dumps(bad['cases'],sort_keys=True).encode()).hexdigest();lockf.write_text(json.dumps(bad))
    r=run(code,['check76.py','--device','cpu'],env);assert r.returncode==1 and 'not the locked one' in r.stderr,r.stderr[-500:];lockf.write_text(orig)
    first=w['lock']['cases'][0]['slide_stem']  # first line of stage1_cases.txt, always a 7.6 check case
    rec=w['hr']/first/'complete.json';o2=rec.read_text();x=json.loads(o2);x['status']='FAIL';rec.write_text(json.dumps(x))
    r=run(code,['train.py','--check','--encoder','uni','--arm','L','--seed-index','0','--device','cpu'],env);rec.write_text(o2)
    assert r.returncode==1 and 'status FAIL' in r.stderr,(r.returncode,r.stderr[-400:])
    ok('refusals: a changed case list -> exit 1 before anything runs; a stage-1 case recorded FAIL (L parity) -> exit 1')
    # ---- protocol 7.6 check
    r=run(code,['check76.py','--device','cuda'],env);assert r.returncode==78  # not the (test) platform
    r=run(code,['check76.py','--device','cpu'],env,0);chk=rjson(code/'CHECK76_PASS.json')
    assert chk['state']=='PASS' and len(chk['results'])==8 and all(v['passed'] for v in chk['results'].values())
    txt=json.dumps(chk).lower();assert 'auc' not in txt and 'accuracy' not in txt
    assert all(v['chunk_max_abs_diff'][k]<=1e-5 for v in chk['results'].values() for k in v['chunk_max_abs_diff'])
    ok('check76: 2 encoders x 4 arms: uninterrupted vs paused (75) + fresh-process resume bitwise identical (model, optimizer, RNG, history, predictions); chunks 64 / whole / 7 agree; labels used = lock; no metric in the record')
    # ---- formal gates
    A=['train.py','--encoder','uni','--arm','H','--seed-index','1','--fold','2','--device','cpu']
    r=run(code,A,env);assert r.returncode==78 and 'LOCK.json' in r.stdout,r.stdout[-400:]
    run(code,['lock.py','write'],env,0);lk=rjson(code/'LOCK.json');assert lk['evidence']['training_outputs_existing']==[] and lk['evidence']['stage1_cases_extracted']==120
    r=run(code,A,env);assert r.returncode==78 and 'QUARANTINED' in r.stdout,r.stdout[-400:]
    release(w,w['conds'][:3]);r=run(code,A,env);assert r.returncode==78 and 'stage0_training_check_passed' in r.stdout
    qf=w['hr']/'QUARANTINE.json';q0=qf.read_text();q=json.loads(q0);q['release_conditions']=[];qf.write_text(json.dumps(q))
    (w['hr']/'RELEASE.json').unlink();r=run(code,A,env);assert r.returncode==78 and 'expected' in r.stdout,r.stdout[-300:]
    q=json.loads(q0);q['release_conditions']=q['release_conditions'][:1];qf.write_text(json.dumps(q));release(w,w['conds'][:1])
    r=run(code,A,env);assert r.returncode==78 and 'expected' in r.stdout;qf.write_text(q0)
    wjson(w['hr']/'RELEASE.json',dict(conditions=[dict(name=n,met=True) for n in w['conds']]));r=run(code,A,env);assert r.returncode==78 and 'malformed' in r.stdout
    release(w);c=rjson(code/'CHECK76_PASS.json');assert c['cross_card_resume_proved'] is False  # one CPU device here
    r=run(code,A,env);assert r.returncode==78 and 'cross-card' in r.stdout,r.stdout[-400:]
    c=dict(c,cross_card_resume_proved=True);wjson(code/'CHECK76_PASS.json',c)  # stands in for a check run with both GPUs visible
    c2=dict(c,platform=dict(c['platform'],torch='0.0'));wjson(code/'CHECK76_PASS.json',c2)
    r=run(code,A,env);assert r.returncode==78 and 'another platform' in r.stdout,r.stdout[-400:];wjson(code/'CHECK76_PASS.json',c)
    ev=(code/'evaluate.py').read_bytes();(code/'evaluate.py').write_bytes(ev+b'\n');r=run(code,A,env);assert r.returncode==78 and 'locked' in r.stdout;(code/'evaluate.py').write_bytes(ev)
    r=run(code,['lock.py','write'],env);assert r.returncode==1
    ok('formal gates: no LOCK.json -> 78; locked but QUARANTINED -> 78; 3 of 4 release conditions -> 78; an emptied or shortened QUARANTINE.json -> 78; malformed RELEASE.json -> 78; 7.6 PASS without cross-card proof -> 78; 7.6 PASS on another platform -> 78; a locked file changed -> 78; LOCK.json is written once')
    # ---- formal run, pause/resume bitwise, already complete
    r=run(code,A,env,0);out=code/'runs/uni/H/seed1/fold2';P=rjson(out/'predictions.json')
    assert len(P)==24 and all(x['fold']==2 for x in P) and 'auc' not in r.stdout.lower()
    alt=work/'alt';r1=run(code,A+['--outroot',str(alt),'--stop-after-epochs','1'],env,75);r2=run(code,A+['--outroot',str(alt)],env,0)
    same_dir_results(out,alt/'uni/H/seed1/fold2');run(code,A,env,0)
    # a crash after the last checkpoint but before history.json / predictions / receipt: the rerun repairs history.json
    d=alt/'uni/H/seed1/fold2';(d/'complete.json').unlink();(d/'predictions.json').unlink();h=rjson(d/'history.json');wjson(d/'history.json',h[:1])
    run(code,A+['--outroot',str(alt)],env,0);assert len(rjson(d/'history.json'))==3;same_dir_results(out,d)
    first=w['lock']['cases'][0]['slide_stem'];rec=w['hr']/first/'complete.json';o3=rec.read_text();rec.write_text(o3[:40])
    r=run(code,['train.py','--check','--encoder','uni','--arm','L','--seed-index','0','--device','cpu','--outroot',str(work/'x')],env);rec.write_text(o3)
    assert r.returncode==1 and 'unreadable extraction record' in r.stdout,(r.returncode,r.stdout[-300:])
    ok('formal run: 24 test predictions of its fold, nothing about performance printed; paused at epoch 1 + resumed == uninterrupted (bitwise); re-run -> ALREADY_COMPLETE; a stale history.json after a late crash is repaired; a corrupt extraction record -> exit 1 (not "not extracted")')
    # ---- the whole campaign (real runs), then the pre-registered analysis
    cfg=rjson(code/'config.json');jobs=[(e,a_,s,f) for e in cfg['encoders'] for a_ in cfg['decision_arms']+cfg['reference_arms'] for s in cfg['seeds'] for f in cfg['folds']]
    r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'REFUSED' in r.stdout
    def one(j):e,a_,s,f=j;return run(code,['train.py','--encoder',e,'--arm',a_,'--seed-index',str(s),'--fold',str(f),'--device','cpu','--threads','1'],env).returncode
    with ThreadPoolExecutor(a.jobs) as ex:rcs=list(ex.map(one,jobs))
    assert rcs==[0]*120,rcs
    r=run(code,['evaluate.py'],env,0);res=rjson(code/'RESULTS.json');assert res['decision']['verdict'] in ('ADVANCE_TO_STAGE2','NEGATIVE_STOP','INCONCLUSIVE_USER_DECIDES')
    # independent recomputation of one number
    import itertools
    cases=sorted(w['lock']['cases'],key=lambda c:c['patient_id']);ids=[c['patient_id'] for c in cases];y=np.array([c['ISUP_grade']-1 for c in cases])
    pv=np.mean([[{x['patient_id']:x['p_G2'] for f in range(5) for x in rjson(code/f'runs/virchow2/H/seed{s}/fold{f}/predictions.json')}[p] for p in ids] for s in range(3)],0)
    pos=pv[y==1];neg=pv[y==0];ref=np.mean([(p>q)+.5*(p==q) for p in pos for q in neg])
    assert abs(ref-res['encoders']['virchow2']['arms']['H']['auc_seed_averaged'])<1e-12
    assert res['prior_all_wrong_G2_in_sample']==20 and res['prior_all_called_G1_in_sample']==12 and res['prior_list_label_mismatch']==[]
    assert (code/'RESULTS.md').read_text().count('Δ(H−L)')==2
    aucs=[v['auc_seed_averaged'] for r in res['encoders'].values() for v in r['arms'].values()];cis=[v['auc_ci95'] for r in res['encoders'].values() for v in r['arms'].values()]
    assert max(aucs)<1 and all(lo<hi for lo,hi in cis),(aucs,cis)
    # positive control: the synthetic world plants a stronger class signal in H only (1.5x), none extra in Hd
    assert res['decision']['verdict']=='ADVANCE_TO_STAGE2' and all(r['delta_H_minus_L']>0 and r['delta_H_minus_Hd']>0 for r in res['encoders'].values())
    shutil.move(code/'runs/uni/L/seed0/fold3/complete.json',work/'moved.json');r=run(code,['evaluate.py'],env);assert r.returncode==1 and 'uni/L/seed0/fold3' in r.stdout
    shutil.move(work/'moved.json',code/'runs/uni/L/seed0/fold3/complete.json')
    ok(f"campaign: 120 real runs; evaluate refuses with any run missing; seed-averaged AUC matches an independent recomputation; planted H-only signal found (positive control): {res['decision']['verdict']}")
    # ---- decision rule (PROTOCOL.md section 6) on hand-made numbers
    py("""import evaluate as E,common as C
cfg=C.config()
def st(d,ds,h):return dict(delta_H_minus_L=d,seed_delta_H_minus_L=ds,delta_H_minus_Hd=h)
cases=[({'uni':st(.03,[.01,.02,.05],.01),'virchow2':st(-.1,[-.1,-.1,-.1],-.1)},'ADVANCE_TO_STAGE2'),
       ({'uni':st(.05,[.01,.02,.05],-.01),'virchow2':st(.0,[.0,.0,.0],.0)},'INCONCLUSIVE_USER_DECIDES'),
       ({'uni':st(.05,[.01,-.02,.05],.02),'virchow2':st(-.01,[-.01,.0,-.02],.0)},'INCONCLUSIVE_USER_DECIDES'),
       ({'uni':st(.02,[.01,-.02,.05],.02),'virchow2':st(-.01,[-.01,.0,-.02],.0)},'NEGATIVE_STOP'),
       ({'uni':st(.08,[.10,.09,-.001],.02),'virchow2':st(.08,[.10,.09,-.001],.02)},'INCONCLUSIVE_USER_DECIDES'),
       ({'uni':st(.029,[.01,.02,.05],.05),'virchow2':st(-.01,[-.02,-.01,-.03],.0)},'INCONCLUSIVE_USER_DECIDES'),
       ({'uni':st(0,[.01,.02,.05],.05),'virchow2':st(-.01,[.02,-.01,-.03],.0)},'NEGATIVE_STOP'),
       ({'uni':st(.10,[.01,.0,.05],.05),'virchow2':st(.04,[.01,.02,.01],.0)},'INCONCLUSIVE_USER_DECIDES')]
for s,want in cases:
    got=E.decide(s,cfg)[0];assert got==want,(s,got,want)""")
    ok('decision rule: advance needs d >= 0.03 AND all seed deltas > 0 AND H > Hd for one encoder; negative needs both encoders d < 0.03 AND (d <= 0 or seed directions mixed); otherwise inconclusive')
    print(f'ALL_LOCAL_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; work dir {work}')
if __name__=='__main__':main()
