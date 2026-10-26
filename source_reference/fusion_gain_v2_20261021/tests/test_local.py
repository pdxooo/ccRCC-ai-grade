"""Local CPU test of fusion_gain_v2 on a synthetic mirror (no real data, no GPU, nothing remote).
Builds <work>/P/hpcroot/<HPC absolute path> with a fake 300-patient manifest, splits, LOCK files, float16 features of
the four member encoders, complete.json, DATA_READY.json and transfer markers (<work>/P/transfer). The code is copied
to <work>/code/fusion_gain_v2_20261021 byte for byte, except config.json 'device' -> 'cpu'; the real CLIs then run with
HE_PATH_PREFIX=<work>/P/hpcroot. Failures are injected only through `python -c` monkeypatches of the imported modules,
so the code files (and their signature) are never changed. Nothing is written into the component directory.
Usage: python tests/test_local.py [--workdir DIR] [--keep] [--threads 2]"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,tempfile,time,fcntl,signal
from pathlib import Path
import numpy as np
import torch

SRC=Path(__file__).resolve().parent.parent
CODE_FILES=['common.py','train.py','smoke.py','evaluate.py','ready.py','config.json','run_arm.sh','run_smoke.sh']
HPC_ROOT='/data/cluster/projects/korean_ccrcc_SBIAD2452'
DIMS={'phikon-v2':1024,'uni':1024,'gigapath':1536,'uni2-h':1536}
GRADES={1:111,2:130,3:43,4:16}
CHECKS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ok(msg):CHECKS.append(msg);print('CHECK ok:',msg,flush=True)
def wjson(p,o):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(o,indent=2))
def rjson(p):return json.loads(Path(p).read_text())

def build_mirror(work,threads):
    rng=np.random.default_rng(20261021);P=work/'P';root=P/'hpcroot'/HPC_ROOT.lstrip('/')
    pool=root/'qc/primary_technical_pool_v1_20261010';splits=root/'bib_phase1/first_round_v1_20261010';feats=root/'features/nine_encoder_full_v1_20261010'
    for d in (pool,splits,feats,root/'qc_deps',P/'transfer'):d.mkdir(parents=True,exist_ok=True)
    grades=np.concatenate([[g]*n for g,n in GRADES.items()]);rng.shuffle(grades)
    patients=[dict(patient_id=f'korea:KR{i+1}',slide_filename=f'HE_KR{i+1}.czi',ISUP_grade=str(int(grades[i]))) for i in range(300)]
    # stratified groups of exactly 60: sort by grade, deal round robin
    order=sorted(range(300),key=lambda i:(patients[i]['ISUP_grade'],rng.random()))
    group={patients[i]['patient_id']:r%5 for r,i in enumerate(order)}
    rows=[]
    for fold in range(5):
        g=[p for p in sorted(patients,key=lambda p:p['ISUP_grade']) if group[p['patient_id']]==(fold+1)%5]
        hi=[p for p in g if p['ISUP_grade'] in '34'];val={p['patient_id'] for p in hi+[p for p in g if p['ISUP_grade'] in '12'][:49-len(hi)]}
        for p in patients:
            role='test' if group[p['patient_id']]==fold else 'val' if p['patient_id'] in val else 'train'
            rows.append(dict(fold=str(fold),patient_id=p['patient_id'],slide_filename=p['slide_filename'],ISUP_grade=p['ISUP_grade'],role=role))
    # smoke patients (same rule as train.py --smoke) get 180 patches so patch subsampling (128 train / 256 eval) is exercised
    def smoke_set():
        out=set();lookup={p['patient_id']:p for p in patients}
        for role in ('train','val'):
            ids=sorted(r['patient_id'] for r in rows if r['fold']=='0' and r['role']==role)
            for g in '1234':out|=set([x for x in ids if lookup[x]['ISUP_grade']==g][:2])
        return out
    big=smoke_set()
    for p in patients:p['primary_count']=str(180 if p['patient_id'] in big else int(rng.integers(4,41)))
    with (pool/'patient_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient_id','slide_filename','ISUP_grade','primary_count']);w.writeheader();w.writerows(patients)
    with (splits/'patient_splits.csv').open('w',newline='',encoding='utf-8-sig') as f:  # the real file carries a BOM
        w=csv.DictWriter(f,fieldnames=['fold','patient_id','slide_filename','ISUP_grade','role']);w.writeheader();w.writerows(rows)
    wjson(pool/'LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv')))
    wjson(splits/'SPLITS_LOCK.json',dict(patient_manifest_sha256=sha(pool/'patient_manifest.csv'),qc_lock_sha256=sha(pool/'LOCK.json'),split_sha256=sha(splits/'patient_splits.csv')))
    means={(e,g):rng.normal(0,1,d).astype(np.float32)*0.6 for e,d in DIMS.items() for g in range(1,5)}
    completion={}
    for p in patients:
        stem=Path(p['slide_filename']).stem;folder=feats/stem;folder.mkdir(exist_ok=True);n=int(p['primary_count']);rec=dict(state='complete',n=n,patient_id=p['patient_id'],slide_filename=p['slide_filename'],features={})
        for e,d in DIMS.items():
            a=(rng.normal(0,1,(n,d)).astype(np.float32)+means[(e,int(p['ISUP_grade']))]).astype(np.float16);np.save(folder/(e+'.npy'),a)
            rec['features'][e]=dict(shape=[n,d],dtype='float16',sha256=sha(folder/(e+'.npy')))
        wjson(folder/'complete.json',rec);completion[p['patient_id']]=sha(folder/'complete.json')
    wjson(root/'bib_phase1/mil_first_round_v2_20261015/DATA_READY.json',dict(state='PASS',patients=300,feature_files=1200,completion_sha256=completion))
    return dict(P=P,prefix=P/'hpcroot',root=root,feats=feats,patients=patients,rows=rows,smoke=big)

def mark(m,enc,on=True):
    """Simulate the transfer lane: per-file .verified markers + batch marker verified_feat_<enc>."""
    for p in m['patients']:
        f=m['feats']/Path(p['slide_filename']).stem/(enc+'.npy.verified')
        f.touch() if on else f.unlink(missing_ok=True)
    b=m['P']/'transfer'/f'verified_feat_{enc}';b.touch() if on else b.unlink(missing_ok=True)

def copy_code(dest,device_patch=True):
    dest.mkdir(parents=True,exist_ok=True)
    for n in CODE_FILES:shutil.copy2(SRC/n,dest/n)
    if device_patch:
        cfg=(SRC/'config.json').read_text();assert cfg.count('"device": "cuda"')==1
        (dest/'config.json').write_text(cfg.replace('"device": "cuda"','"device": "cpu"'))
    for n in CODE_FILES:
        if n!='config.json':assert sha(dest/n)==sha(SRC/n)
    return dest

def logfile(code,tag):p=code.parent/'logs'/tag;p.parent.mkdir(exist_ok=True);return p
def run(code,args,env,expect=None,quiet=True,tag='',exe=None):
    t=time.monotonic();r=subprocess.run([*(exe or [sys.executable,'-u']),*args],cwd=code,env=env,capture_output=True,text=True)
    with logfile(code,tag or args[0].replace('.py','')).open('a') as f:f.write(f'$ {" ".join(args)}\n{r.stdout}\n{r.stderr}\nEXIT {r.returncode}\n')
    if expect is not None and r.returncode!=expect:
        print(r.stdout[-4000:]);print(r.stderr[-4000:]);raise AssertionError(f'{args} exit {r.returncode}, expected {expect}')
    if not quiet:print(r.stdout[-2000:])
    print(f'  ran {" ".join(args)[:150]} -> {r.returncode} ({time.monotonic()-t:.1f}s)',flush=True)
    return r
def bg(code,args,env,tag,exe=None,session=False):
    """Start a background process; its output goes to logs/<tag>."""
    f=logfile(code,tag).open('a');f.write(f'$ {" ".join(args)}\n');f.flush()
    p=subprocess.Popen([*(exe or [sys.executable,'-u']),*args],cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=session);p.log=logfile(code,tag);return p
def finish(p,timeout=600):
    rc=p.wait(timeout=timeout);time.sleep(0.05);return rc,p.log.read_text()
def wait_for(pred,timeout=120,what=''):
    t=time.monotonic()
    while not pred():
        if time.monotonic()-t>timeout:raise AssertionError(f'timeout waiting for {what}')
        time.sleep(0.02)
def lock_free(path):
    with open(path,'a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return True
        except BlockingIOError:return False
def no_orphans(root):
    locks=list(Path(root).rglob('run.lock'));assert locks,f'no run.lock under {root}'
    busy=[str(l) for l in locks if not lock_free(l)];assert not busy,f'still held (orphaned train.py?): {busy}'

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def same_state(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def assert_same_run(A,B):
    LA=load(A/'last.pt');LB=load(B/'last.pt')
    assert same_state(load(A/'best.pt')['model'],load(B/'best.pt')['model']) and same_state(LA['model'],LB['model'])
    assert LA['best_epoch']==LB['best_epoch'] and LA['best_key']==LB['best_key'] and LA['next_epoch']==LB['next_epoch']
    assert [{k:v for k,v in h.items() if k!='seconds'} for h in LA['history']]==[{k:v for k,v in h.items() if k!='seconds'} for h in LB['history']]
    for n in ('val_predictions.json','test_predictions.json'):assert (A/n).read_bytes()==(B/n).read_bytes(),n
    return LA

def check_predictions(path,expected_ids,role,fold=0):
    recs=rjson(path);assert [r['patient_id'] for r in recs]==list(expected_ids),'prediction patients/order'
    for r in recs:
        assert set(r)=={'patient_id','fold','role','ISUP_grade','probabilities'} and r['role']==role and r['fold']==fold
        assert isinstance(r['ISUP_grade'],int) and 1<=r['ISUP_grade']<=4
        p=np.array(r['probabilities']);assert p.shape==(4,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-5
    return recs

def inject(module,patch,args):
    """python -c: import the real module, apply a monkeypatch (a statement on M), run its CLI main()."""
    return ['-c',f'import sys,torch,{module} as M\n{patch}\nsys.exit(M.main(sys.argv[1:]))',*args]
RAISE=lambda exc:f"(_ for _ in ()).throw({exc})"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir');ap.add_argument('--keep',action='store_true');ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();work=Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix='fg2_local_'))
    if work.exists() and any(work.iterdir()):shutil.rmtree(work)
    work.mkdir(parents=True,exist_ok=True);t0=time.monotonic();print('WORKDIR',work,flush=True)
    m=build_mirror(work,a.threads);ok(f'synthetic mirror built ({sum(int(p["primary_count"]) for p in m["patients"])} patches, 300 patients, BOM splits)')
    code=copy_code(work/'code'/'fusion_gain_v2_20261021')
    base={k:v for k,v in os.environ.items() if k not in ('HE_PATH_PREFIX','HE_TRANSFER_DIR','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','CUDA_DEVICE_ORDER')}
    base.update(OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads));env=dict(base,HE_PATH_PREFIX=str(m['prefix']))
    T=str(a.threads);SM=['--device','cpu','--threads',T]

    # 1. path prefix + forced environment
    probe=['-c','import common as c;print(c.ROOT);print(c.FEATURES);print(c.DATA_READY);print(c.sys.path[0])']
    out=run(code,probe,dict(base),tag='probe').stdout.split();assert out[0]==HPC_ROOT and out[1]==HPC_ROOT+'/features/nine_encoder_full_v1_20261010' and out[2]==HPC_ROOT+'/bib_phase1/mil_first_round_v2_20261015/DATA_READY.json' and out[3]==HPC_ROOT+'/qc_deps'
    out2=run(code,probe,dict(base,HE_PATH_PREFIX=''),tag='probe').stdout.split();assert out2==out
    out3=run(code,probe,env,tag='probe').stdout.split();assert out3[0]==str(m['root']) and out3[2]==str(m['root']/'bib_phase1/mil_first_round_v2_20261015/DATA_READY.json')
    out4=run(code,probe,dict(base,HE_PATH_PREFIX=str(m['prefix'])+'/'),tag='probe').stdout.split();assert out4==out3
    assert run(code,probe,dict(base,HE_PATH_PREFIX='relative/x'),tag='probe').returncode!=0
    envp=['-c','import os,common as c;print(os.environ["CUBLAS_WORKSPACE_CONFIG"],os.environ["CUDA_DEVICE_ORDER"],c.INHERITED_CUBLAS_WORKSPACE_CONFIG,os.environ["OPENBLAS_NUM_THREADS"])']
    e=run(code,envp,dict({k:v for k,v in base.items() if k!='OPENBLAS_NUM_THREADS'},CUBLAS_WORKSPACE_CONFIG=':16:8',CUDA_DEVICE_ORDER='FASTEST_FIRST'),tag='probe').stdout.split()
    assert e==[':4096:8','PCI_BUS_ID',':16:8','4'],e
    ok('HE_PATH_PREFIX: unset/empty -> original HPC paths; set -> prefixed paths; relative rejected. CUBLAS_WORKSPACE_CONFIG forced to :4096:8 (inherited value kept for metadata), CUDA_DEVICE_ORDER=PCI_BUS_ID, OPENBLAS threads default 4')

    # 2. readiness helper
    mark(m,'phikon-v2')
    run(code,['ready.py','single:phikon-v2'],env,0);run(code,['ready.py','mdrop4'],env,1);run(code,['ready.py','single:uni'],env,1);run(code,['ready.py','bogus'],env,2)
    stem=Path(m['patients'][17]['slide_filename']).stem;mk=m['feats']/stem/'phikon-v2.npy.verified';mk.unlink()
    r=run(code,['ready.py','single:phikon-v2'],env,1);assert str(mk) in r.stdout;mk.touch()
    (m['P']/'transfer/verified_feat_phikon-v2').unlink();run(code,['ready.py','single:phikon-v2'],env,1);(m['P']/'transfer/verified_feat_phikon-v2').touch()
    run(code,['ready.py','single:phikon-v2'],env,0)
    run(code,['ready.py','single:phikon-v2'],dict(env,HE_TRANSFER_DIR=str(work/'nowhere')),1)
    ok('ready.py: 0 only with batch marker + all 300 per-file .verified markers; 1 otherwise; 2 for unknown arm; HE_TRANSFER_DIR override')

    # 3. gates before smoke
    r=run(code,['train.py','--arm','single:phikon-v2','--seed-index','0','--fold','0',*SM],env,78);assert 'SMOKE_PASS.json missing' in r.stdout
    orig=copy_code(work/'code_orig'/'fusion_gain_v2_20261021',device_patch=False)
    run(orig,['smoke.py','--device','cpu','--threads',T],env,78,tag='smoke_orig');run(orig,['train.py','--arm','single:phikon-v2','--seed-index','0','--fold','0','--device','cpu'],env,78,tag='train_orig')
    ok('formal training refused without SMOKE_PASS (78); with the real config (device cuda) CPU smoke (78) and CPU formal runs (78) are refused')

    # 4. smoke, auto: single passes (reduced + formal shape, bitwise), mdrop4 deferred (only phikon-v2 verified)
    run(code,['smoke.py',*SM],env,0,tag='smoke_auto')
    sp=rjson(code/'SMOKE_PASS.json');s1=sp['parts']['single']
    assert sp['state']=='PASS' and s1['state']=='PASS' and sp['parts']['mdrop4']['state']=='DEFERRED' and sp['parts_summary']=={'single':'PASS','mdrop4':'DEFERRED'}
    assert s1['covers']==['single:phikon-v2','single:uni','single:gigapath','single:uni2-h'] and s1['unit']['chunk_equivalence'] and s1['attempt']==1
    assert s1['resume_bitwise_identical'] and s1['formal_shape_bitwise_identical'] and s1['cross_card_resume'] is False and s1['resume_device']=='cpu'
    assert set(s1['unit']['real_dimensions'])=={'single:phikon-v2','single:uni','single:gigapath','single:uni2-h','mdrop4'} and s1['unit']['real_dimensions']['mdrop4']['dims']==[1024,1024,1536,1536]
    tm=rjson(code/'TIMING.json');assert isinstance(tm['single:phikon-v2'],dict) and tm['single:phikon-v2']['seconds_per_epoch']>0 and tm['mdrop4']=='deferred'
    lookup={p['patient_id']:p for p in m['patients']};att1=code/'smoke_runs'/'single_phikon-v2'/'attempt_001'
    roles={r:sorted(x['patient_id'] for x in m['rows'] if x['fold']=='0' and x['role']==r) for r in ('train','val','test')}
    sel=[p for g in '1234' for p in [x for x in roles['val'] if lookup[x]['ISUP_grade']==g][:2]]
    for shape,vids in (('reduced',sel),('formal',roles['val'])):
        A=att1/f'{shape}_full/single_phikon-v2/seed0/fold0';B=att1/f'{shape}_resume/single_phikon-v2/seed0/fold0'
        assert same_state(load(A/'best.pt')['model'],load(B/'best.pt')['model']) and same_state(load(A/'last.pt')['model'],load(B/'last.pt')['model'])
        assert (A/'val_predictions.json').read_bytes()==(B/'val_predictions.json').read_bytes()
        check_predictions(A/'val_predictions.json',vids,'val');assert not list(A.glob('test_predictions.json'))
        rc=rjson(A/'complete.json');assert all(sha(A/n)==d for n,d in rc['artifacts'].items()) and rc['smoke'] and rc['epochs']==2
        md=rjson(A/'metadata.json');assert md['contract']['smoke_shape']==shape and (md['contract']['config']['train_patches']==(128 if shape=='reduced' else 2048))
    assert rjson(att1/'attempt.json')['state']=='PASS'
    ok('smoke auto: unit checks (toy + all 5 arms at real dimensions) + single:phikon-v2 reduced-shape AND formal-shape full vs paused/resumed bitwise identical, in smoke_runs/<arm>/attempt_001; mdrop4 DEFERRED; TIMING from the formal-shape leg')

    # 4b. a failing re-run revokes the PASS; evidence is kept; transient errors never revoke; a new pass restores
    formal0=['train.py','--arm','single:phikon-v2','--seed-index','0','--fold','0',*SM]
    run(code,inject('smoke','M.compare=lambda *a,**k:'+RAISE("AssertionError('injected bitwise mismatch')"),['--part','single',*SM]),env,1,tag='smoke_fail')
    sp=rjson(code/'SMOKE_PASS.json');assert sp['parts']['single']['state']=='FAIL' and sp['parts']['single']['attempt']==2 and 'injected' in sp['parts']['single']['error']
    assert (code/'SMOKE_FAIL_single.json').exists() and (att1/'reduced_full/single_phikon-v2/seed0/fold0/last.pt').exists() and (att1/'formal_resume/single_phikon-v2/seed0/fold0/complete.json').exists()
    r=run(code,formal0+['--outroot',str(work/'revoked')],env,78,tag='train_revoked');assert 'has not passed (state FAIL)' in r.stdout and not (work/'revoked').exists()
    run(code,inject('smoke','M.real_part=lambda *a,**k:'+RAISE("torch.cuda.OutOfMemoryError('CUDA out of memory (injected)')"),['--part','single',*SM]),env,74,tag='smoke_transient')
    sp=rjson(code/'SMOKE_PASS.json');assert sp['parts']['single']['state']=='FAIL' and sp['attempts'][-1]['state']=='INCOMPLETE' and not sp['attempts'][-1]['applied']
    run(code,['smoke.py','--part','single',*SM],env,0,tag='smoke_repass')
    sp=rjson(code/'SMOKE_PASS.json');assert sp['parts']['single']['state']=='PASS' and sp['parts']['single']['attempt']==4 and not (code/'SMOKE_FAIL_single.json').exists()
    run(code,inject('smoke','M.real_part=lambda *a,**k:'+RAISE("M.Transient('leg timed out (injected)')"),['--part','single',*SM]),env,74,tag='smoke_transient')
    sp=rjson(code/'SMOKE_PASS.json');assert sp['parts']['single']['state']=='PASS' and sp['parts']['single']['attempt']==4 and [x['state'] for x in sp['attempts']]==['PASS','FAIL','INCOMPLETE','PASS','INCOMPLETE']
    assert rjson(code/'smoke_runs/single_phikon-v2/attempt_002/attempt.json')['state']=='FAIL' and all((code/f'smoke_runs/single_phikon-v2/attempt_00{i}').is_dir() for i in range(1,6))
    ok('smoke re-run: a failed check records parts.single=FAIL (revokes the earlier PASS: formal run refused 78), keeps attempt_001 evidence; transient errors (74) are INCOMPLETE and never replace PASS/FAIL; a later pass restores PASS and clears SMOKE_FAIL_single.json')

    # 4c. one smoke at a time; a stop signal to smoke.py stops its train.py leg (no orphan) and leaves the PASS alone
    with open(code/'smoke.run.lock','w') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX);r=run(code,['smoke.py','--part','single',*SM],env,5,tag='smoke_locked');assert 'SMOKE_LOCKED' in r.stdout
    p=bg(code,['smoke.py','--part','single',*SM],env,'smoke_signal');att6=code/'smoke_runs/single_phikon-v2/attempt_006'
    wait_for(lambda:(att6/'reduced_full/single_phikon-v2/seed0/fold0/run.lock').exists(),what='smoke leg start');time.sleep(0.3);p.send_signal(signal.SIGTERM)
    rc,log=finish(p);assert rc==75,(rc,log[-2000:]);no_orphans(att6)
    sp=rjson(code/'SMOKE_PASS.json');assert sp['parts']['single']['attempt']==4 and sp['parts']['single']['state']=='PASS' and rjson(att6/'attempt.json')['kind']=='interrupted'
    ok('smoke.py: a second concurrent smoke exits 5 (smoke.run.lock); SIGTERM to smoke.py is forwarded to its train.py leg, which stops (no orphan holding run.lock); exit 75, attempt INCOMPLETE, PASS kept')

    # 5. mdrop4 gates, then smoke --part mdrop4 appends without touching the single part
    sp=rjson(code/'SMOKE_PASS.json')
    run(code,['train.py','--arm','mdrop4','--seed-index','0','--fold','0',*SM],env,77,tag='train_mdrop4_gate')
    for enc in ('uni','gigapath','uni2-h'):mark(m,enc)
    run(code,['ready.py','all'],env,0)
    run(code,['train.py','--arm','mdrop4','--seed-index','0','--fold','0',*SM],env,78,tag='train_mdrop4_gate')
    run(code,['smoke.py','--part','mdrop4',*SM],env,0,tag='smoke_mdrop4')
    sp2=rjson(code/'SMOKE_PASS.json');assert sp2['parts']['single']==sp['parts']['single'] and sp2['parts']['mdrop4']['state']=='PASS' and sp2['parts']['mdrop4']['covers']==['mdrop4'] and sp2['parts']['mdrop4']['formal_shape_bitwise_identical']
    tm2=rjson(code/'TIMING.json');assert isinstance(tm2['mdrop4'],dict) and tm2['mdrop4']['seconds_per_epoch']>0
    ok('mdrop4: formal refused while data unverified (77) and while its smoke part is DEFERRED (78); smoke --part mdrop4 PASS (reduced + formal shape) appended, single part unchanged')

    # 6. formal run (patched config only differs in device): uninterrupted vs paused+resumed, bitwise
    fa=work/'formalA';fb=work/'formalB'
    run(code,formal0+['--outroot',str(fa)],env,0,tag='formalA')
    run(code,formal0+['--outroot',str(fb),'--stop-after-epochs','3'],env,75,tag='formalB');run(code,formal0+['--outroot',str(fb)],env,0,tag='formalB')
    A=fa/'single_phikon-v2/seed0/fold0';B=fb/'single_phikon-v2/seed0/fold0';LA=assert_same_run(A,B)
    check_predictions(A/'test_predictions.json',roles['test'],'test');check_predictions(A/'val_predictions.json',roles['val'],'val')
    assert len(roles['test'])==60 and len(roles['val'])==49 and len(roles['train'])==191
    rc=rjson(A/'complete.json');assert set(rc['artifacts'])=={'best.pt','last.pt','history.json','metadata.json','val_predictions.json','test_predictions.json'} and all(sha(A/n)==d for n,d in rc['artifacts'].items())
    assert rc['test']['n']==60 and rc['validation']['n']==49 and not rc['smoke'] and rc['epochs']==LA['next_epoch']>=30 and rc['physical_gpus']==[] and rc['cross_card_resume'] is False
    md=rjson(A/'metadata.json');assert md['platform']['device_type']=='cpu' and md['device']=='cpu' and md['he_path_prefix']==str(m['prefix']) and md['invocation']['cublas_workspace_config']==':4096:8' and md['invocation']['ram_margin_gib']==128
    assert rjson(A/'history.json')==LA['history'] and not (B/'paused.json').exists()
    res=rjson(A/'resources.json');assert (A/'resources.json').exists() and 'resources.json' not in rc['artifacts'] and all('host' in v and 'cuda_visible_devices' in v for v in res.values())
    run(code,formal0+['--outroot',str(fa)],env,0,tag='formalA');ok(f'formal single:phikon-v2 fold0: uninterrupted vs paused@3+resumed bitwise identical ({rc["epochs"]} epochs, best {rc["best_epoch"]}); 60 test / 49 val predictions well-formed; receipt hashes verified; history.json == checkpoint history; paused.json cleared; per-epoch card provenance; re-run -> ALREADY_COMPLETE')
    run(code,['train.py','--arm','mdrop4','--seed-index','1','--fold','2',*SM,'--outroot',str(work/'formalM'),'--stop-after-epochs','1'],env,75,tag='formal_mdrop4')
    ok('formal mdrop4 passes all gates after its smoke part (runs, pauses with 75)')

    # 6b. signals to train.py: first SIGTERM = pause at epoch end, SIGINT = stop at once; both resume bitwise identically
    for tag,sig,marker in (('sigterm',signal.SIGTERM,'PAUSED'),('sigint',signal.SIGINT,'STOPPED_NOW')):
        d=work/f'formal_{tag}';run_dir=d/'single_phikon-v2/seed0/fold0'
        p=bg(code,formal0+['--outroot',str(d)],env,f'formal_{tag}');wait_for(lambda:(run_dir/'last.pt').exists(),what='first epoch');p.send_signal(sig)
        rc_,log=finish(p);assert rc_==75 and marker in log,(rc_,log[-1500:]);no_orphans(d)
        if sig==signal.SIGTERM:assert rjson(run_dir/'paused.json')['signals']==['SIGTERM']
        run(code,formal0+['--outroot',str(d)],env,0,tag=f'formal_{tag}');assert_same_run(A,run_dir)
    p=bg(code,formal0+['--outroot',str(work/'formal_early')],env,'formal_early');time.sleep(0.5);p.send_signal(signal.SIGTERM);rc_,log=finish(p);assert rc_==75,(rc_,log[-1500:])
    ok('train.py signals: SIGTERM -> pause at epoch end (75, paused.json), SIGINT -> immediate stop (75); both resume to results bitwise identical to the uninterrupted run; SIGTERM during start-up -> 75')

    # 7. operational guards: RAM (76), run lock (79), platform-mismatch resume (78), transient GPU errors (74)
    g=inject('train','M.mem_available=lambda:1',['--arm','single:phikon-v2','--seed-index','0','--fold','0','--smoke','--device','cpu','--outroot',str(work/'ramtest')])
    run(code,g,env,76,tag='ram');ok('insufficient MemAvailable -> exit 76 before preloading')
    lk=work/'locktest/single_phikon-v2/seed0/fold0';lk.mkdir(parents=True);fh=(lk/'run.lock').open('w');fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(code,['train.py','--arm','single:phikon-v2','--seed-index','0','--fold','0','--smoke','--device','cpu','--outroot',str(work/'locktest')],env,79,tag='lock');fh.close();ok('concurrent process on the same run dir -> exit 79')
    pm=['train.py','--arm','single:phikon-v2','--seed-index','0','--fold','0','--smoke','--device','cpu','--outroot',str(work/'plat')]
    run(code,pm+['--stop-after-epochs','1'],env,75,tag='plat');pt=work/'plat/single_phikon-v2/seed0/fold0/last.pt';s=load(pt);s['platform']=dict(s['platform'],torch='other');torch.save(s,pt)
    r=run(code,pm,env,78,tag='plat');assert 'resume refused' in r.stdout;ok('resume on a different platform is refused (78)')
    sm=['--arm','single:phikon-v2','--seed-index','0','--fold','0','--smoke','--device','cpu']
    run(code,inject('train','M.setup_device=lambda n:'+RAISE("RuntimeError('CUDA error: all CUDA-capable devices are busy or unavailable')"),[*sm,'--outroot',str(work/'busy')]),env,74,tag='transient')
    run(code,inject('train','M.BagStore.tensors=lambda *a:'+RAISE("torch.cuda.OutOfMemoryError('CUDA out of memory (injected)')"),[*sm,'--outroot',str(work/'oom')]),env,74,tag='transient')
    run(code,inject('train','M.BagStore.tensors=lambda *a:'+RAISE("RuntimeError('some integrity problem')"),[*sm,'--outroot',str(work/'integrity')]),env,1,tag='transient')
    ok('busy/unavailable GPU and CUDA OOM -> exit 74 (transient); other errors stay 1')
    gi=('import json,torch,train as T\n'
        'class U:\n bytes=list(range(16))\n'
        'class V:\n def __str__(self):return "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"\n'
        'class W:\n def __str__(self):return "<torch._C._CUuuid object at 0x1>"\n'
        'out=[]\n'
        'for u in (U(),V(),W()):\n'
        ' T.torch.cuda.get_device_properties=lambda d,u=u:type("P",(),dict(name="NVIDIA RTX A6000",uuid=u))()\n'
        ' out.append(T.gpu_identity(torch.device("cuda",0))["gpu_uuid"])\n'
        'print(json.dumps(out))')
    r=run(code,['-c',gi],dict(env,CUDA_VISIBLE_DEVICES='1'),0,tag='gpu_identity')
    assert json.loads(r.stdout.strip().splitlines()[-1])==['GPU-00010203-0405-0607-0809-0a0b0c0d0e0f','GPU-0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0',None],r.stdout
    ok('gpu_identity (mocked CUDA properties): UUID from bytes or a uuid string, nvidia-smi format; an unparsable value is recorded as None, never as a per-process object string')

    # 8. run_arm.sh: signals are forwarded and waited for; no further fold after a signal; then completes
    arm_env=dict(env,HE_PY=sys.executable);sh=['bash'];ra=['run_arm.sh','single:phikon-v2','1','0',T,'--device','cpu']
    seed1=code/'runs/single_phikon-v2/seed1'
    p=bg(code,ra,arm_env,'run_arm',exe=sh,session=True);wait_for(lambda:(seed1/'fold0/last.pt').exists(),what='fold0 epoch')
    p.send_signal(signal.SIGUSR1);rc_,log=finish(p);assert rc_==75 and 'STOPPED arm=single:phikon-v2 seed=1 fold=0 code=75' in log,(rc_,log[-1500:])
    assert (seed1/'fold0/paused.json').exists() and not (seed1/'fold1').exists();no_orphans(seed1)
    p=bg(code,ra,arm_env,'run_arm',exe=sh,session=True);wait_for(lambda:not (seed1/'fold0/paused.json').exists(),what='fold0 resumed')
    p.send_signal(signal.SIGTERM);rc_,log=finish(p);assert rc_==75 and not (seed1/'fold1').exists(),(rc_,log[-1500:]);no_orphans(seed1)
    p=bg(code,ra,arm_env,'run_arm',exe=sh,session=True);wait_for(lambda:not (seed1/'fold0/paused.json').exists(),what='fold0 resumed')
    os.killpg(p.pid,signal.SIGTERM);rc_,log=finish(p);assert rc_==75 and 'PAUSED' in log.split('RUN_ARM_START')[-1],(rc_,log[-1500:]);no_orphans(seed1)
    r=run(code,ra,arm_env,0,tag='run_arm',exe=sh);assert 'ARM_SEED_COMPLETE single:phikon-v2 1' in r.stdout and all((seed1/f'fold{k}/complete.json').exists() for k in range(5))
    ok('run_arm.sh: SIGUSR1 / SIGTERM to the script and SIGTERM to its process group reach train.py, which pauses (75, resumable); the script waits, exits 75 and starts no further fold; no orphan; then all 5 folds complete')

    # 9. evaluate.py: one campaign only (fabricated receipts for all 5 arms x 4 seeds x 5 folds)
    ev=copy_code(work/'code_eval'/'fusion_gain_v2_20261021');cfg=rjson(ev/'config.json')
    sig=run(ev,['-c','import common;print(common.signature())'],env,0,tag='evaluate').stdout.strip()
    plat=dict(device_type='cpu',torch='x',cuda=None)
    def fabricate(mutate=lambda arm,s,k,md,done:None,smoke_state='PASS'):
        shutil.rmtree(ev/'runs',ignore_errors=True);rng=np.random.default_rng(5)
        for arm in cfg['arms']:
            for s in cfg['seeds']:
                for k in range(5):
                    d=ev/'runs'/arm.replace(':','_')/f'seed{s}'/f'fold{k}';d.mkdir(parents=True)
                    recs=[dict(patient_id=x['patient_id'],fold=k,role='test',ISUP_grade=int(x['ISUP_grade']),probabilities=(lambda v:(v/v.sum()).tolist())(rng.dirichlet(np.ones(4))+np.eye(4)[int(x['ISUP_grade'])-1]*0.5)) for x in m['rows'] if x['fold']==str(k) and x['role']=='test']
                    md=dict(fingerprint=f'f-{arm}-{s}-{k}',platform=dict(plat),contract=dict(code=sig,smoke=False,arm=arm,seed_index=s,fold=k,split_sha256='S',pool_sha256='Q'))
                    done=dict(state='COMPLETE',smoke=False,fingerprint=md['fingerprint'],physical_gpus=['GPU-a'],cross_card_resume=False)
                    mutate(arm,s,k,md,done);wjson(d/'test_predictions.json',recs);wjson(d/'metadata.json',md)
                    wjson(d/'complete.json',dict(done,artifacts={n:sha(d/n) for n in ('test_predictions.json','metadata.json')}))
        wjson(ev/'SMOKE_PASS.json',dict(state='PASS',code_signature=sig,parts={p:dict(state=smoke_state if p=='single' else 'PASS',code_signature=sig,platform=plat,attempt=1) for p in ('single','mdrop4')}))
    fabricate();run(ev,['evaluate.py'],env,0,tag='evaluate');res=rjson(ev/'RESULTS.json')
    assert set(res['preregistered'])=={'P1_encoder_vs_seed_diversity','P2_full_late_fusion','P3_learned_fusion'} and 'mdrop4_4seeds' in res['per_set'] and len(res['per_run'])==20
    assert res['campaign']['runs']==100 and res['campaign']['code_signature']==sig and res['campaign']['physical_gpus']==['GPU-a'] and res['campaign']['cross_card_resume_proven_by_smoke'] is False
    first=lambda f:(lambda arm,s,k,md,done:f(md,done) if (arm,s,k)==('mdrop4',3,4) else None)
    bad={'other code':(first(lambda md,done:md['contract'].update(code='0'*64)),'PASS','not one campaign'),
         'mixed platform':(first(lambda md,done:md['platform'].update(torch='2.7.0')),'PASS','runs disagree on platform'),
         'smoke receipt':(first(lambda md,done:done.update(smoke=True)),'PASS','not a COMPLETE formal run'),
         'paused run':(first(lambda md,done:done.update(state='PAUSED')),'PASS','not a COMPLETE formal run'),
         'wrong fold':(first(lambda md,done:md['contract'].update(fold=0)),'PASS','contract is not the formal run'),
         'revoked smoke':(lambda *x:None,'FAIL','smoke part single is FAIL')}
    for name,(mut,state,msg) in bad.items():
        fabricate(mut,state);(ev/'RESULTS.json').unlink(missing_ok=True);r=run(ev,['evaluate.py'],env,1,tag='evaluate')
        assert msg in r.stderr and not (ev/'RESULTS.json').exists(),(name,r.stderr[-800:])
    fabricate(first(lambda md,done:done.update(cross_card_resume=True,physical_gpus=['GPU-a','GPU-b'])));r=run(ev,['evaluate.py'],env,0,tag='evaluate')
    assert 'WARNING 1 run(s) resumed on the other card' in r.stdout and rjson(ev/'RESULTS.json')['campaign']['cross_card_resumes']==['mdrop4|seed3|fold4']
    ok('evaluate.py pools only one campaign: refuses other code signature, mixed platform, smoke/non-COMPLETE receipts, wrong contract, revoked smoke part; writes RESULTS.json (P1-P3 + campaign block) otherwise; warns about unproven cross-card resumes')
    print(f'ALL_LOCAL_TESTS_PASSED {len(CHECKS)} checks in {time.monotonic()-t0:.0f}s; logs in {work/"code/logs"}',flush=True)
    if not a.keep:shutil.rmtree(work)
if __name__=='__main__':main()
