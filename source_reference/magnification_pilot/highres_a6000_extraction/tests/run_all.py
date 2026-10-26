"""Local CPU test suite for the A6000 high-res pilot (no GPU, no CZI, no real weights).
Usage: run_all.py <scratch_dir>     (creates fresh fake worlds under it; prints one line per check; exit 1 on any failure)"""
import os,sys,json,time,shutil,hashlib,subprocess,fcntl,signal
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent;PY=sys.executable
K='/data/cluster/projects/korean_ccrcc_SBIAD2452'
results=[]
def check(name,ok,detail=''):
    results.append((name,bool(ok)));print(('PASS' if ok else 'FAIL'),name,('| '+str(detail)) if detail else '',flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class World:
    def __init__(self,path,variant='fake'):
        self.path=Path(path);subprocess.run([PY,str(TESTS/'make_world.py'),str(path),variant],check=True)
        self.pfx=self.path/'hpcroot';self.root=Path(str(self.pfx)+K);self.project=self.path/'project'
        self.out1=self.root/'features/highres_pilot_v1_20261021_stage1_a6000';self.out0=self.root/'features/highres_pilot_v1_20261021_a6000'
        self.run_lock=self.root/'features/highres_pilot_v1_20261021_a6000.lock'
        self.stored=self.root/'features/nine_encoder_full_v1_20261010';self.log=self.path/'logs/reader.jsonl'
    def env(self,**extra):
        e=dict(os.environ,HE_PATH_PREFIX=str(self.pfx),HE_PROJECT_DIR=str(self.project),FAKE_READER_LOG=str(self.log));e.update(extra);return e
    def run(self,mode,*args,**extra):
        t=time.monotonic();p=subprocess.run([PY,str(TESTS/'harness.py'),mode,str(self.path),'--device','cpu','--workers','2',*args],env=self.env(**extra),capture_output=True,text=True,timeout=600)
        return p.returncode,p.stdout+p.stderr,time.monotonic()-t
    def mark(self,stem=None,gate=True):
        if gate:(self.project/'transfer/verified_stage0').touch()
        if stem:
            for f in [self.root/'data/he'/f'{stem}.czi',self.stored/stem/'uni.npy',self.stored/stem/'virchow2.npy']:f.with_name(f.name+'.verified').touch()
    def unmark(self):
        (self.project/'transfer/verified_stage0').unlink(missing_ok=True)
        for f in self.root.rglob('*.verified'):f.unlink()
    def events(self,out):return [json.loads(l) for l in out.splitlines() if l.startswith('{"event"')]
    def complete(self,stem):return json.loads((self.out1/stem/'complete.json').read_text())
    def report0(self):return json.loads((self.out0/'STAGE0_REPORT.json').read_text())
    def summary(self,tag='shard0of1'):return json.loads((self.out1/'shards'/f'{tag}.json').read_text())

def sigterm_run(w,mode):
    """Start `mode` with a slow fake decode, SIGTERM it once a decode worker has opened a slide; returns (rc, worker pids alive)."""
    n0=len(w.log.read_text().splitlines()) if w.log.exists() else 0
    p=subprocess.Popen([PY,str(TESTS/'harness.py'),mode,str(w.path),'--device','cpu','--workers','2'],env=w.env(FAKE_DECODE_SLEEP='3'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    t=time.monotonic()
    while time.monotonic()-t<60 and not (w.log.exists() and len(w.log.read_text().splitlines())>n0):time.sleep(.2)
    pids={json.loads(l)['pid'] for l in w.log.read_text().splitlines()[n0:]};p.send_signal(signal.SIGTERM)
    try:rc=p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();rc=None
    time.sleep(.5);alive=[]
    for q in pids:
        try:os.kill(q,0);alive.append(q)
        except ProcessLookupError:pass
    return rc,pids,alive

def test_static(scratch):
    p=subprocess.run([PY,str(COMP/'verify_no_work_paths.py')],capture_output=True,text=True)
    check('verify_no_work_paths: runtime files clean',p.returncode==0,p.stdout.strip().splitlines()[-1])
    p=subprocess.run([PY,str(COMP/'verify_no_work_paths.py'),*map(str,sorted((COMP/'originals').glob('*.py')))],capture_output=True,text=True)
    check('verify_no_work_paths: flags the unpatched originals (6 /work literals + 3 unprefixed _path lookups)',p.returncode==1 and 'violations=9' in p.stdout,p.stdout.strip().splitlines()[-1])
    probe=scratch/'paths_probe.py'
    probe.write_text("p=row['primary_path']\nnp.load(p)\nok=hpc(row['primary_path'])\nw=info.get('weight_path')\n"
                     "x=info.get('weight_path')  # prefixed: built from the prefixed ASSETS\nif row['primary_path']==q:pass\n")
    p=subprocess.run([PY,str(COMP/'verify_no_work_paths.py'),str(probe)],capture_output=True,text=True)
    check('verify_no_work_paths: _path lookups via assignment / comparison / .get() are flagged, hpc() and waivers are not',
          p.returncode==1 and 'unwrapped_data_paths=3' in p.stdout and all(f'line {n}:' in p.stdout for n in (1,4,6)),p.stdout.strip().splitlines()[0][-60:])
    p=subprocess.run([PY,str(TESTS/'test_prefix_identity.py')],capture_output=True,text=True)
    check('prefix byte-identity (unset, empty, prefix, prefix/) + patched text minus prefix == original',p.returncode==0,' / '.join(l for l in p.stdout.splitlines() if l.startswith(('PASS','FAIL','MISMATCH')))[:400])
    code="import sys;sys.path.insert(0,sys.argv[1]);import stage0_extract as s0;print(s0.GATED,len(s0.missing_markers([s0.ROOT/'x.czi'])),s0.OUT.name,s0.RUN_LOCK.parent==s0.OUT.parent)"
    got={}
    for prefix in (None,'','/x/hpcroot'):
        env=dict(os.environ);env.pop('HE_PATH_PREFIX',None)
        if prefix is not None:env['HE_PATH_PREFIX']=prefix
        got[prefix]=subprocess.run([PY,'-c',code,str(COMP)],env=env,capture_output=True,text=True).stdout.strip()
    crash={}
    for prefix in ('relative/hpcroot',str(scratch/'no_such_mirror')):
        p=subprocess.run([PY,str(COMP/'stage0_extract.py')],env=dict(os.environ,HE_PATH_PREFIX=prefix,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        crash[prefix]=(p.returncode,p.stderr.strip().splitlines()[-1][:90] if p.stderr.strip() else '')
    check('stage0 as a script: configuration crash (relative / missing HE_PATH_PREFIX) exits 2, never 1 (= FAIL)',all(rc==2 for rc,_ in crash.values()),crash)
    check('transfer gate only with a prefix (HPC: no markers); OUT keeps the _a6000 suffix; run lock outside OUT',
          got[None]==got['']=='False 0 highres_pilot_v1_20261021_a6000 True' and got['/x/hpcroot']=='True 1 highres_pilot_v1_20261021_a6000 True',got)

def test_layout(scratch):
    w=World(scratch/'w_real','real')
    p=subprocess.run([PY,str(TESTS/'test_import_layout.py')],env=w.env(),capture_output=True,text=True)
    check('import layout: patched czi_pilot_v2 (main+spawn worker), unchanged mirror full_patch_qc sha pinned',p.returncode==0 and 'LAYOUT_OK' in p.stdout,(p.stdout+p.stderr)[-300:] if p.returncode else '')
    check('decode-pool initializer: workers re-check module path + sha256 (verified layout accepted, wrong sha breaks the pool)',p.returncode==0 and 'WORKER_CHECK_OK' in p.stdout,(p.stdout+p.stderr)[-300:] if p.returncode else '')
    p=subprocess.run([PY,str(TESTS/'test_import_layout.py'),'--bypass-shim'],env=w.env(),capture_output=True,text=True)
    check('import layout: bypassing the shim is detected',p.returncode==0 and 'BYPASS_DETECTED' in p.stdout,p.stdout.strip()[:120])

def test_runner(scratch):
    p=subprocess.run([PY,str(TESTS/'test_runner.py'),str(scratch/'runner')],capture_output=True,text=True,timeout=600)
    lines=[l for l in p.stdout.splitlines() if l.startswith(('PASS ','FAIL '))]
    for l in lines:
        ok,rest=l.split(' ',1);name,_,detail=rest.partition(' | ');check(name.strip(),ok=='PASS',detail)
    check('runner tests ran to completion',p.returncode==0 and len(lines)==9,(p.stdout+p.stderr)[-300:] if p.returncode else f'{len(lines)} checks')

def original_cpu_copy(w):
    """The HPC original stage0_extract.py with only device/path/test-size substitutions, to compare outputs byte for byte."""
    t=(COMP/'originals/stage0_extract.py').read_text()
    subs=[("Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')",f"Path('{w.root}')"),
          ("Path('/data/cluster/projects/gastritis_uni2h/outputs')",f"Path('{w.pfx}/data/cluster/projects/gastritis_uni2h/outputs')"),
          ("assert torch.cuda.is_available(),'GPU is required'","pass"),(".cuda()",".cpu()"),("device='cuda'","device='cpu'"),
          (".to('cuda',dtype=torch.float32)",".to('cpu',dtype=torch.float32)"),("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cpu',dtype=torch.float16)"),
          ("torch.cuda.synchronize()","None"),("module('pathology_adapters',OLD/'encoder-validation-20261008/validate.py')",f"module('pathology_adapters','{TESTS/'fake_validate.py'}')"),
          ("LOCATIONS=2048","LOCATIONS=24"),("BATCH=32","BATCH=8"),("row['primary_path']",f"('{w.pfx}'+row['primary_path'])")]
    for a,b in subs:
        assert a in t,a;t=t.replace(a,b)
    d=w.path/'orig_cpu';d.mkdir();(d/'stage0_extract.py').write_text(t);return d/'stage0_extract.py'

def test_stage0(scratch):
    w=World(scratch/'w_stage0')
    code,out,_=w.run('stage0');check('stage0: refuses to start without transfer/verified_stage0 (exit 3)',code==3 and 'verified_stage0' in out and not w.out0.exists(),out.strip()[-160:])
    w.mark(gate=True);code,out,_=w.run('stage0')
    check('stage0: waits while per-case .verified markers are missing (exit 3, checked before any CUDA use)',code==3 and 'unverified' in out and not w.out0.exists(),out.strip()[:160])
    for i in range(901,906):w.mark(f'HE_KR{i}')
    lockf=w.root/'qc/primary_technical_pool_v1_20261010/LOCK.json';good=lockf.read_text();d=json.loads(good);d['patient_manifest_sha256']='0'*64;lockf.write_text(json.dumps(d))
    code,out,_=w.run('stage0');rep=w.report0() if (w.out0/'STAGE0_REPORT.json').exists() else {}
    check('stage0: manifest sha256 != pool LOCK.json -> FAIL before extraction (exit 1)',code==1 and rep.get('state')=='FAIL' and 'patient_manifest_sha256' in rep.get('error','') and not any(w.out0.glob('HE_*/uni_L.npy')),rep.get('error','')[:100])
    lockf.write_text(good);shutil.rmtree(w.out0)
    smoke=w.root/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json';good=smoke.read_text()
    bad=json.loads(good);bad['models']['virchow2']['weight_sha256']='0'*64;smoke.write_text(json.dumps(bad))
    code,out,_=w.run('stage0');rep=w.report0()
    check('stage0: weight sha256 != smoke summary -> FAIL before extraction (exit 1)',code==1 and rep['state']=='FAIL' and 'virchow2' in rep['error'] and not any(w.out0.glob('HE_*/uni_L.npy')),rep.get('error','')[:120])
    smoke.write_text(good);shutil.rmtree(w.out0)
    holder=w.run_lock.open('a');fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code,out,_=w.run('stage0');holder.close()
    check('stage0: run lock held by another stage-0 process -> exit 5 (not 1), nothing written',code==5 and 'BUSY' in out and 'Traceback' not in out and not w.out0.exists(),out.strip()[-100:])
    code,out,_=w.run('stage0',FAKE_GPU_FREE_GIB='1.5');ev=w.events(out)
    check('stage0: shared GPU with < --min-free-gib free -> exit 3, no model load, nothing written',code==3 and any(e['event']=='gpu_busy' for e in ev) and not w.out0.exists(),ev[-1:] )
    code,out,_=w.run('stage0',FAKE_OOM_AT_LOAD='1');rep=w.report0()
    check('stage0: CUDA OOM while loading models -> exit 3, report state waiting_gpu_out_of_memory (not FAIL)',code==3 and rep['state']=='waiting_gpu_out_of_memory' and rep['exit_code']==3,rep['state'])
    code,out,_=w.run('stage0',FAKE_CRASH='1');rep=w.report0()
    check('stage0: crash during extraction -> exit 2, report state error + traceback (not FAIL, not left as running)',code==2 and rep['state']=='error' and 'simulated crash' in rep['traceback'],rep.get('error','')[:80])
    code,out,dt=w.run('stage0',FAKE_DECODE_SLEEP='5',FAKE_DECODE_TIMEOUT='1');rep=w.report0()
    check('stage0: hung CZI decode -> DecodeTimeout, workers killed, exit 2 without hanging',code==2 and rep['state']=='error' and 'DecodeTimeout' in rep['error'] and dt<30,f'{dt:.1f}s {rep.get("error","")[:80]}')
    rc,pids,alive=sigterm_run(w,'stage0');rep=w.report0()
    check('stage0: SIGTERM -> decode workers killed, report state interrupted (not left as running), exit 143',rc==143 and pids and not alive and rep['state']=='interrupted',f'rc={rc} workers={sorted(pids)} alive={alive}')
    shutil.rmtree(w.out0)
    code,out,dt=w.run('stage0');rep=w.report0()
    check('stage0 (patched, CPU): PASS_PENDING_HUMAN_VISUAL_REVIEW (exit 0)',code==0 and rep['state']=='PASS_PENDING_HUMAN_VISUAL_REVIEW',f"{dt:.1f}s cases={rep['cases']} L={ {k:round(v['min_cosine'],6) for k,v in rep['checks']['L_reproduces_stored'].items()} }")
    check('stage0: per-case progress lines printed',len([e for e in w.events(out) if e['event']=='case'])==4)
    plat=rep['platform'];need={'gpu','torch','torch_cuda','cudnn','torchvision','timm','pillow','numpy','numpy_path','imagecodecs','czifile','HE_PATH_PREFIX','hostname','device'}
    check('stage0: platform (library versions) + modules + weight verification + manifest sha recorded',need<=set(plat) and plat['HE_PATH_PREFIX']==str(w.pfx) and set(rep['modules'])=={'czi_pilot_v2.py','full_patch_qc.py','validate.py'}
          and all(rep['encoders'][n]['weight_sha256_matches_smoke_summary'] for n in ('uni','virchow2')) and rep['manifest_sha256']==json.loads(lockf.read_text())['patient_manifest_sha256'],
          {k:plat[k] for k in ('gpu','torch','pillow','numpy','device')})
    mt={str(f):f.stat().st_mtime_ns for f in w.out0.rglob('*') if f.is_file()}
    code,out,_=w.run('stage0')
    check('stage0: PASS already recorded for this code -> exit 0, nothing recomputed or touched',code==0 and 'already recorded' in out and mt=={str(f):f.stat().st_mtime_ns for f in w.out0.rglob('*') if f.is_file()})
    f=next(w.out0.glob('HE_*/uni_H.npy'));raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(bytes(raw))
    code,out,_=w.run('stage0')
    check('stage0: an output no longer matching its complete.json -> recomputed (deterministic) -> PASS',code==0 and 'already recorded' not in out and w.report0()['state']=='PASS_PENDING_HUMAN_VISUAL_REVIEW'
          and all(sha(c.parent/n)==h for c in w.out0.glob('HE_*/complete.json') for n,h in json.loads(c.read_text())['files'].items()))
    rep=w.report0()
    # byte-for-byte comparison with the unmodified HPC algorithm (CPU substitutions only)
    orig=original_cpu_copy(w)
    p=subprocess.run([PY,str(orig)],env=w.env(),capture_output=True,text=True);orig_out=w.root/'features/highres_pilot_v1_20261021'
    check('stage0 original (HPC code, CPU substitutions) runs',p.returncode==0,(p.stdout+p.stderr).strip()[-200:])
    ro=json.loads((orig_out/'STAGE0_REPORT.json').read_text());diffs=[]
    for k in ('cases','state'):
        if ro[k]!=rep[k]:diffs.append(k)
    for k in ('L_reproduces_stored','geometry','subtiles_union_exact','Hd_equals_L_quadrants'):
        if ro['checks'][k]!=rep['checks'][k]:diffs.append('checks.'+k)
    nfiles=0
    for d in sorted(orig_out.glob('HE_*')):
        for f in sorted(d.iterdir()):
            g=w.out0/d.name/f.name
            if f.name=='complete.json':
                a,b=json.loads(f.read_text()),json.loads(g.read_text());a.pop('code_sha256');b.pop('code_sha256')
                same=a==b
            else:same=g.exists() and f.read_bytes()==g.read_bytes()
            nfiles+=1
            if not same:diffs.append(f'{d.name}/{f.name}')
    check('stage0 patched == HPC original: all npy/png bytes, complete.json, checks identical',not diffs and nfiles>=4*11,f'{nfiles} files compared; diffs={diffs[:5]}')
    # stage1 imports the stage0 functions: overlapping cases must be byte-identical to stage0's outputs
    code,out,_=w.run('stage1');ev=w.events(out)
    same=[];diff=[]
    for c in [d.name for d in w.out1.glob('HE_*')]:
        if (w.out0/c).exists():
            for f in ['location_indices.npy','coords.npy']+[f'{e}_{a}.npy' for e in ('uni','virchow2') for a in ('L','H','Hd')]:
                (same if (w.out0/c/f).read_bytes()==(w.out1/c/f).read_bytes() else diff).append(f'{c}/{f}')
    check('stage1 outputs == stage0 outputs for the overlapping case(s) (same imported functions)',code==0 and same and not diff,f'{len(same)} files identical, diffs={diff[:3]}')

def test_stage1(scratch):
    import numpy as np
    w=World(scratch/'w_stage1');A,B='HE_KR905','HE_KR903'  # lock order: A (index 0) -> shard 0/2, B (index 1) -> shard 1/2
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: no stage-0 report -> exit 3 waiting_for_stage0; no GPU, no model load, nothing extracted',
          code==3 and s['state']=='waiting_for_stage0' and s['platform']['gpu'] is None and 'weights' not in s and not list(w.out1.glob('HE_*/complete.json')),s['stage0_gate'])
    q=json.loads((w.out1/'QUARANTINE.json').read_text()) if (w.out1/'QUARANTINE.json').exists() else {}
    check('stage1: QUARANTINE.json lists the 4 release conditions (stage-0 PASS, visual 7.4, A800 parity, training check 7.6)',
          q.get('state')=='QUARANTINED' and len(q.get('release_conditions',[]))==4 and all(k in json.dumps(q) for k in ('stage0_automated','visual','parity','7.6','RELEASE.json')))
    w.out0.mkdir(parents=True);(w.out0/'STAGE0_REPORT.json').write_text(json.dumps(dict(state='FAIL',error='simulated: L parity 0.981')))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: stage-0 report FAIL -> exit 1 blocked_by_stage0, nothing extracted',code==1 and s['state']=='blocked_by_stage0' and 'FAILED' in s['stage0_gate']['reason'] and not list(w.out1.glob('HE_*/complete.json')),s['stage0_gate']['reason'][:80])
    (w.out0/'STAGE0_REPORT.json').write_text(json.dumps(dict(state='running')))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: stage 0 still running -> exit 3 waiting_for_stage0',code==3 and s['state']=='waiting_for_stage0')
    shutil.rmtree(w.out0);w.mark(gate=True)
    for i in range(901,905):w.mark(f'HE_KR{i}',gate=False)
    code,out,_=w.run('stage0');w.unmark()
    check('stage1 world: stage 0 PASS (its markers removed again afterwards)',code==0 and w.report0()['state']=='PASS_PENDING_HUMAN_VISUAL_REVIEW',out.strip()[-120:])
    rep0=(w.out0/'STAGE0_REPORT.json').read_text()
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: stage 0 PASS, no markers -> exit 3 (waiting), nothing extracted',code==3 and s['state']=='waiting' and s['stage0_gate']['code']==0 and not list(w.out1.glob('HE_*/complete.json')),w.events(out)[-1])
    check('stage1: summary says what each case waits for',s['waiting_for'].get(A,[])[:2]==['P/transfer/verified_stage0',f'K/data/he/{A}.czi'] and f'K/features/nine_encoder_full_v1_20261010/{A}/uni.npy' in s['waiting_for'][A],s['waiting_for'].get(A))
    w.mark(A,gate=False);code,out,_=w.run('stage1')
    check('stage1: markers present but transfer/verified_stage0 missing (weights) -> exit 3',code==3 and not (w.out1/A/'complete.json').exists())
    w.mark(A);code,out,dt=w.run('stage1');ev=w.events(out)
    check('stage1: A ready, B not -> A extracted, exit 3',code==3 and w.complete(A)['status']=='PASS' and not (w.out1/B).exists(),f'{dt:.1f}s {[e for e in ev if e["event"]=="case"]}')
    c=w.complete(A)
    sys.path.insert(0,str(COMP));os.environ.update(HE_PATH_PREFIX=str(w.pfx),HE_PROJECT_DIR=str(w.project))
    import stage0_extract as s0
    s0.LOCATIONS=24
    man={l.split(',')[0]:l.split(',') for l in (w.root/'qc/primary_technical_pool_v1_20261010/patient_manifest.csv').read_text().splitlines()[1:]}
    n=int(man['korea:KR905'][6]);idx=s0.location_indices(n,'korea:KR905')
    with np.load(w.root/'qc/primary_technical_pool_v1_20261010/primary'/f'{A}.npz') as z:coords=z['coords']
    shapes={f'{e}_{a}':np.load(w.out1/A/f'{e}_{a}.npy').shape for e in ('uni','virchow2') for a in ('L','H','Hd')}
    dtypes={np.load(w.out1/A/f'{e}_{a}.npy').dtype for e in ('uni','virchow2') for a in ('L','H','Hd')}
    m=len(idx)
    ok=(np.array_equal(np.load(w.out1/A/'location_indices.npy'),idx) and np.array_equal(np.load(w.out1/A/'coords.npy'),coords[idx]) and m==24 and n==37
        and shapes=={'uni_L':(m,1024),'uni_H':(m,4,1024),'uni_Hd':(m,4,1024),'virchow2_L':(m,2560),'virchow2_H':(m,4,2560),'virchow2_Hd':(m,4,2560)} and dtypes=={np.dtype('float16')}
        and all(sha(w.out1/A/f)==h for f,h in c['files'].items()) and len(c['files'])==8)
    check('stage1 A: location_indices = stage-0 seed rule, coords = primary[idx], shapes (n,d)/(n,4,d) float16, file sha256s',ok,f'n={n} m={m} shapes={shapes}')
    check('stage1 A: L vs stored cosine recorded (min/mean) and no label fields in complete.json',all(k in c['L_reproduces_stored'][e] for e in ('uni','virchow2') for k in ('min_cosine','mean_cosine'))
          and 'ISUP' not in json.dumps(c) and 'grade' not in json.dumps(c).lower(),{e:round(v['min_cosine'],6) for e,v in c['L_reproduces_stored'].items()})
    pin=json.loads((w.out1/'RECIPE.json').read_text());runs=[json.loads(l) for l in (w.out1/'shards/shard0of1.runs.jsonl').read_text().splitlines()]
    check('stage1 A: provenance: library versions in complete.json and recipe, platform, stage-0 report sha; RECIPE.json pinned; per-run history',
          {'torch','pillow','numpy','numpy_path','timm','torchvision','imagecodecs','czifile','cudnn'}<=set(c['environment']) and c['environment']==w.summary()['recipe']['environment']
          and {'hostname','device','torch'}<=set(c['platform']) and c['stage0_report']['sha256']==sha(w.out0/'STAGE0_REPORT.json') and pin['recipe_sha256']==c['recipe_sha256']
          and [r['event'] for r in runs].count('start')==[r['event'] for r in runs].count('exit')>=4,{k:c['environment'][k] for k in ('torch','pillow','numpy')})
    logs=[json.loads(l) for l in w.log.read_text().splitlines()]
    check('stage1: every decode worker imported the patched czi_pilot_v2 and the prefixed ROOT',logs and all(l['czi_pilot_v2']==str((COMP/'czi_pilot_v2.py').resolve()) and l['ROOT']==str(w.root) for l in logs),f'{len(logs)} worker reader opens, pids={sorted({l["pid"] for l in logs})}')
    f=w.out1/A/'uni_H.npy';raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(bytes(raw))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: a bad case while others still wait for data -> exit 3 waiting_with_errors (lane keeps going), error recorded',code==3 and s['state']=='waiting_with_errors' and s['cases'][A]=='ERROR' and 'sha256 differs' in s['errors'][A],s['cases'])
    raw[-1]^=1;f.write_bytes(bytes(raw))
    code,out,_=w.run('stage1','--shard','1','--nshards','2');s=w.summary('shard1of2')
    check('stage1 shard 1/2: assigned only B; B not ready -> exit 3',code==3 and s['assigned']==[B],s['assigned'])
    code,out,dt=w.run('stage1','--shard','1','--nshards','2','--wait','3','--poll','1')
    check('stage1 --wait 3 --poll 1: polls then exits 3',code==3 and dt>=3,f'{dt:.1f}s')
    w.mark(B);code,out,_=w.run('stage1','--shard','1','--nshards','2');ev=w.events(out)
    check('stage1 shard 1/2: B extracted -> exit 0',code==0 and w.complete(B)['status']=='PASS' and [e['case'] for e in ev if e['event']=='case']==[B])
    mt={f.name:f.stat().st_mtime_ns for f in (w.out1/A).iterdir()}
    code,out,_=w.run('stage1','--shard','0','--nshards','2');ev=w.events(out)
    check('stage1 shard 0/2: A skipped after hash verification -> exit 0, files untouched',code==0 and not [e for e in ev if e['event']=='case'] and mt=={f.name:f.stat().st_mtime_ns for f in (w.out1/A).iterdir()})
    code,out,_=w.run('stage1');check('stage1 shard 0/1: everything complete -> exit 0, nothing recomputed',code==0 and not [e for e in w.events(out) if e['event']=='case'])
    f=w.out1/A/'uni_H.npy';raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(bytes(raw))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: corrupted output file and nothing left to wait for -> exit 1',code==1 and s['state']=='error' and 'sha256 differs' in json.dumps(s['errors']),s['cases'])
    raw[-1]^=1;f.write_bytes(bytes(raw));code,out,_=w.run('stage1');check('stage1: restored file -> exit 0',code==0)
    holder=(w.out1/'locks/shard0of1.lock').open('a');fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code,out,_=w.run('stage1');holder.close();check('stage1: shard lock held by another process -> exit 5',code==5,out.strip()[-60:])
    code,out,_=w.run('stage1',FAKE_LOCATIONS='20');s=w.summary()
    check('stage1: recipe change (LOCATIONS) -> RECIPE.json pin refuses to mix, exit 1, nothing touched',code==1 and 'refusing to mix' in json.dumps(s['errors']) and not [e for e in w.events(out) if e['event']=='case'])
    rp=w.out0/'STAGE0_REPORT.json';r=json.loads(rep0);r['code_sha256']='0'*64;rp.write_text(json.dumps(r))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: stage 0 passed under other stage-0 code -> exit 1 blocked_by_stage0',code==1 and s['state']=='blocked_by_stage0' and 'changed since stage 0 passed' in s['stage0_gate']['reason'],s['stage0_gate']['reason'][:90])
    r=json.loads(rep0);r['platform']['pillow']='0.0.1';rp.write_text(json.dumps(r))
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: stage 0 passed under other library versions -> exit 1 blocked_by_stage0',code==1 and 'library versions differ' in s['stage0_gate']['reason'] and 'pillow' in s['stage0_gate']['reason'],s['stage0_gate']['reason'][:90])
    rp.write_text(rep0)
    before=w.complete(A)['files'];shutil.move(str(w.out1/A),str(w.path/'A_first_run'))
    code,out,_=w.run('stage1');after=w.complete(A)['files']
    check('stage1: re-extraction of A is bit-identical (deterministic)',code==0 and before==after,f'{len(after)} file hashes equal')
    # FAIL flag: stored virchow2 features of B no longer match -> case FAIL, exit 4
    st=w.stored/B/'virchow2.npy';a=np.load(st).astype(np.float32);rng=np.random.default_rng(0)
    row=int(s0.location_indices(40,'korea:KR903')[5]);a[row]=rng.standard_normal(a.shape[1]).astype(np.float32);np.save(st,a.astype(np.float16))  # a sampled row
    ref=json.loads((w.stored/B/'complete.json').read_text());ref['features']['virchow2']['sha256']=sha(st);(w.stored/B/'complete.json').write_text(json.dumps(ref))
    shutil.rmtree(w.out1/B);code,out,_=w.run('stage1');cb=w.complete(B)
    check('stage1: stored-L mismatch -> case FAIL (min cos < 0.999) recorded, exit 4',code==4 and cb['status']=='FAIL' and cb['L_reproduces_stored']['virchow2']['min_cosine']<.999 and cb['L_reproduces_stored']['uni']['pass_'],
          {e:round(v['min_cosine'],4) for e,v in cb['L_reproduces_stored'].items()})
    wf=Path(str(w.pfx)+'/data/cluster/software/uni2h/models/uni/pytorch_model.bin');wgood=wf.read_bytes();wf.write_bytes(wgood+b'x')
    shutil.rmtree(w.out1/A);code,out,_=w.run('stage1');s=w.summary()
    check('stage1: weight file sha256 != smoke summary -> WeightMismatch, nothing extracted, exit 1',code==1 and 'WeightMismatch' in json.dumps(s['errors']) and not (w.out1/A/'complete.json').exists())
    wf.write_bytes(wgood)
    smoke=w.root/'features/nine_encoder_smoke_v1_20261010/summary_shard0.json';good=smoke.read_text();bad=json.loads(good);bad['models']['uni']['weight_sha256']='f'*64;smoke.write_text(json.dumps(bad))
    code,out,_=w.run('stage1')
    check('stage1: weight reference (smoke summary) changed -> refused (recipe pin / stage-0 gate), nothing extracted, exit 1',code==1 and not (w.out1/A/'complete.json').exists())
    smoke.write_text(good)
    lock=w.project/'transfer/lists/STAGE1_CASES_LOCK.json';good=lock.read_text();d=json.loads(good);d['cases'][0]['fold']=4;lock.write_text(json.dumps(d))
    code,out,_=w.run('stage1');check('stage1: edited case lock -> list_sha256 assertion, exit 1',code==1 and 'list sha256' in out)
    lock.write_text(good)
    code,out,_=w.run('stage1',FAKE_GPU_FREE_GIB='1.5');ev=w.events(out)
    check('stage1: shared GPU with < --min-free-gib free -> no model load, exit 3',code==3 and any(e['event']=='gpu_busy' for e in ev) and not (w.out1/A/'complete.json').exists(),[e for e in ev if e['event']=='gpu_busy'])
    code,out,_=w.run('stage1',FAKE_OOM_AT_LOAD='1');ev=w.events(out);s=w.summary()
    check('stage1: CUDA OOM while loading models -> exit 3 waiting (not 1), nothing extracted',code==3 and s['state']=='waiting' and any(e['event']=='gpu_oom_at_load' for e in ev) and not s['errors'] and not (w.out1/A/'complete.json').exists(),ev[-1])
    code,out,_=w.run('stage1',FAKE_OOM='1');s=w.summary()
    check('stage1: CUDA OOM during a case -> case left incomplete, exit 3 (retry later)',code==3 and s['cases'][A]=='GPU_OOM_RETRY' and json.loads((w.out1/A/'error.json').read_text())['state']=='gpu_out_of_memory_retry_later',s['cases'])
    code,out,_=w.run('stage1',FAKE_OOM='cublas');s=w.summary()
    check('stage1: cuBLAS allocation failure (plain RuntimeError) -> treated as OOM, exit 3',code==3 and s['cases'][A]=='GPU_OOM_RETRY',s['cases'])
    code,out,_=w.run('stage1');s=w.summary()
    check('stage1: retry after OOM completes A (B still FAIL) -> exit 4, error.json cleared',code==4 and w.complete(A)['status']=='PASS' and not (w.out1/A/'error.json').exists(),s['cases'])
    shutil.rmtree(w.out1/A);code,out,dt=w.run('stage1',FAKE_DECODE_SLEEP='5',FAKE_DECODE_TIMEOUT='1');s=w.summary()
    check('stage1: hung CZI decode -> DecodeTimeout recorded for the case, workers killed, no hang, exit 1 (nothing left to wait for)',
          code==1 and s['cases'][A]=='ERROR' and 'DecodeTimeout' in s['errors'][A] and dt<30,f'{dt:.1f}s')
    code,out,_=w.run('stage1');check('stage1: next run re-extracts A -> exit 4 (B FAIL)',code==4 and w.complete(A)['status']=='PASS')
    shutil.rmtree(w.out1/A);rc,pids,alive=sigterm_run(w,'stage1');s=w.summary()
    runs=[json.loads(l)['event'] for l in (w.out1/'shards/shard0of1.runs.jsonl').read_text().splitlines()]
    check('stage1: SIGTERM -> decode workers killed, summary state interrupted, case left incomplete, exit 143',
          rc==143 and pids and not alive and s['state']=='interrupted' and runs[-1]=='interrupted' and not (w.out1/A/'complete.json').exists(),f'rc={rc} workers={sorted(pids)} alive={alive}')
    code,out,_=w.run('stage1');check('stage1: after SIGTERM the next run completes A -> exit 4 (B FAIL)',code==4 and w.complete(A)['status']=='PASS')
    (w.out1/'QUARANTINE.json').write_text(json.dumps(dict(state='QUARANTINED',release_conditions=['old a','old b'])))
    code,out,_=w.run('stage1');q=json.loads((w.out1/'QUARANTINE.json').read_text())
    check('stage1: an outdated QUARANTINE.json is refreshed',len(q['release_conditions'])==4)

def test_partition(scratch):
    sys.path.insert(0,str(COMP))
    import stage1_extract as s1
    lock=json.loads((COMP.parent/'STAGE1_CASES_LOCK.json').read_text());cases=lock['cases']
    check('real lock: list_sha256 recomputes to the pinned constant',s1.list_sha256(cases)==s1.LIST_SHA256==lock['list_sha256'],s1.LIST_SHA256[:16])
    txt=[l.strip() for l in (COMP.parent/'stage1_cases.txt').read_text().splitlines() if l.strip()]
    check('real lock: stage1_cases.txt (transfer order) == lock order',txt==[c['slide_stem'] for c in cases],len(txt))
    parts=[s1.partition(cases,i,2) for i in range(2)];ids=[c['slide_stem'] for p in parts for c in p]
    per_batch=[[sum(1 for c in p if c['slide_stem'] in txt[b*12:(b+1)*12]) for p in parts] for b in range(10)]
    check('real lock: shards 0/2 + 1/2 disjoint, cover all 120, 6+6 cases from every 12-case transfer batch',sorted(ids)==sorted(txt) and len(set(ids))==120 and all(x==[6,6] for x in per_batch),per_batch[:3])
    check('partition is deterministic',[c['slide_stem'] for c in s1.partition(cases,1,2)]==[c['slide_stem'] for c in parts[1]])

if __name__=='__main__':
    scratch=Path(sys.argv[1]).resolve()
    if scratch.exists():shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    test_static(scratch);test_layout(scratch);test_runner(scratch);test_partition(scratch);test_stage0(scratch);test_stage1(scratch)
    bad=[n for n,ok in results if not ok]
    print(f'SUMMARY {len(results)-len(bad)}/{len(results)} passed'+(f'; FAILED: {bad}' if bad else ''))
    sys.exit(1 if bad else 0)
