"""High-resolution pilot, stage 1 extraction on lab-a6000: the 120 locked cases. No training; labels are never read
(grade counts are logged from the lock file only).

Every scientific step is imported from the patched stage-0 module (location sampling with the stage-0 seed rule,
2x2 sub-tiling, L/H/Hd inputs, spawn decode workers, UNI bilinear / Virchow2 bicubic, fp16 autocast embedding,
float16 storage). This file only schedules cases, gates them, verifies hashes, writes atomically and records provenance.

Stage-0 gate (protocol section 7): nothing is extracted (and no GPU is touched) unless the A6000 stage-0 report
features/highres_pilot_v1_20261021_a6000/STAGE0_REPORT.json says PASS_PENDING_HUMAN_VISUAL_REVIEW for the current
stage0_extract.py sha256, the same helper-module sha256s, the same weight reference and the same library versions.
The human visual review (7.4) and the training check (7.6) gate the USE of the features (QUARANTINE.json), not the
extraction; see CHANGES.md (deviation requiring the user's consent; strict alternative: scheduler
stage1_requires_approval=true).

Per case (OUT/<slide_stem>/): <enc>_L.npy (n,d), <enc>_H.npy (n,4,d), <enc>_Hd.npy (n,4,d), location_indices.npy,
coords.npy, complete.json (sha256 of every file, recipe fingerprint incl. library versions, platform, L-vs-stored
cosine min/mean, PASS/FAIL). A case is processed only when its CZI and its stored uni.npy / virchow2.npy carry
.verified markers (and transfer/verified_stage0 exists: weights). Cases with a valid complete.json are skipped after
re-hashing. All cases of all shards must share one recipe, pinned in OUT/RECIPE.json by the first extraction.

QUARANTINED: see OUT/QUARANTINE.json. Nothing here may be used downstream before the release conditions are met and
recorded by a human in OUT/RELEASE.json (nothing in this file reads RELEASE.json: every consumer must check it).

Exit codes: 0 all assigned cases complete and PASS | 3 waiting: stage 0 has not passed yet, some assigned cases' inputs
are not verified yet (also when other cases errored meanwhile: see the shard summary), or the shared GPU has
< --min-free-gib free / ran out of memory (retry later) | 4 all assigned cases complete but >=1 FAIL (L parity < 0.999;
needs review) | 5 another process holds this shard's lock | 1 error: stage 0 FAILED or no longer matches, recipe
mismatch, weight mismatch, or case errors with nothing left to wait for (see the shard summary / case error.json).
"""
import os,sys,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stage0_extract as s0  # first: thread env vars and the stage-0 sys.path layout (qc_deps numpy, patched czi_pilot_v2)
import csv,json,hashlib,time,argparse,fcntl,shutil,collections,traceback
import numpy as np

OUT=s0.ROOT/'features/highres_pilot_v1_20261021_stage1_a6000'
LOCK_FILE=s0.TRANSFER/'lists/STAGE1_CASES_LOCK.json'
CASE_LIST=s0.TRANSFER/'lists/stage1_cases.txt'  # transfer order; must hold the same cases as the lock
STAGE0_REPORT=s0.OUT/'STAGE0_REPORT.json'
LIST_SHA256='264688b81e670a4b381df8a7b4caf43de64d2926584bfd621994541f94a66584'
N_CASES=120
MIN_COSINE=.999
MIN_FREE_BYTES=20*1024**3
KEEP=('patient_id','slide_filename','source_sha256','primary_count','mpp_x','mpp_y','primary_path','primary_sha256')  # no label columns
EXIT_OK,EXIT_ERROR,EXIT_WAITING,EXIT_FAIL,EXIT_BUSY=0,1,3,4,5
QUARANTINE=dict(
    state='QUARANTINED',
    rule='Features in this directory must NOT be used by any downstream training, evaluation or analysis until ALL release '
         'conditions below are met and recorded by a human in RELEASE.json next to this file. Extraction may proceed meanwhile '
         '(it starts only after the automated stage-0 checks passed; see highres_a6000/CHANGES.md). The extraction code never '
         'reads RELEASE.json: every downstream consumer must check it.',
    release_conditions=[
        'stage0_automated_checks_passed_on_a6000: features/highres_pilot_v1_20261021_a6000/STAGE0_REPORT.json state '
        'PASS_PENDING_HUMAN_VISUAL_REVIEW for the stage0_module_sha256 recorded in every complete.json (protocol sections '
        '7.1-7.3 and 7.5; enforced by stage1_extract.py before any extraction)',
        'stage0_visual_review_signed_off: the user or the designated pathologist has reviewed the stage-0 L-vs-H panels '
        '(features/highres_pilot_v1_20261021*/<stem>/visual_*.png) and signed off that H shows finer nuclear detail (protocol '
        'section 7.4; no visible difference => stop, do not use these features)',
        'stage0_platform_parity_passed: A6000 stage-0 outputs (features/highres_pilot_v1_20261021_a6000) match the A800/HPC stage-0 '
        'outputs (features/highres_pilot_v1_20261021) for L, H and Hd of both encoders: identical location_indices.npy and coords.npy, '
        'per-row cosine >= 0.999',
        'stage0_training_check_passed: the protocol section 7.6 training check (8 cases, 2 epochs: resume consistency, '
        'chunked == whole-bag results, label mapping) has passed before any stage-1 training'],
    produced_by='highres_a6000/stage1_extract.py',list_sha256=LIST_SHA256)
gpu_free_gib=s0.gpu_free_gib  # module attribute: tests simulate a busy shared card through it

class RecipeMismatch(RuntimeError):pass  # outputs made under another recipe exist: stop the shard, never mix

def write_npy(p,a):
    """np.save bytes, written to a temporary name and renamed (a crash never leaves a truncated .npy under the final name)."""
    tmp=p.with_name(p.name+'.tmp')
    with open(tmp,'wb') as f:np.save(f,a)
    os.replace(tmp,p)

def save_unique(p,r):
    tmp=p.with_name(f'{p.name}.{os.getpid()}.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False));os.replace(tmp,p)

def list_sha256(cases):return hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()

def load_lock():
    lock=json.loads(LOCK_FILE.read_text());cases=lock['cases']
    got=list_sha256(cases)
    assert got==LIST_SHA256 and lock['list_sha256']==LIST_SHA256,f'stage-1 case list sha256 {got} / {lock["list_sha256"]} != locked {LIST_SHA256}'
    assert len(cases)==N_CASES and len({c['patient_id'] for c in cases})==N_CASES and len({c['slide_stem'] for c in cases})==N_CASES
    return lock,cases

def transfer_order(cases):
    if not CASE_LIST.exists():return None
    order=[l.strip() for l in CASE_LIST.read_text().splitlines() if l.strip()]
    assert sorted(order)==sorted(c['slide_stem'] for c in cases),f'{CASE_LIST} does not hold the locked stage-1 cases'
    return order==[c['slide_stem'] for c in cases]

def manifest_rows(lock,cases):
    f=s0.POOL/'patient_manifest.csv';got=s0.sha(f)
    assert got==lock['manifest_sha256'],f'{f} sha256 {got} != lock manifest_sha256'
    want={c['patient_id'] for c in cases};rows={}
    for r in csv.DictReader(f.open(encoding='utf-8-sig')):
        if r['patient_id'] in want:rows[r['patient_id']]={k:r[k] for k in KEEP}  # label columns dropped unread
    assert set(rows)==want,sorted(want-set(rows))
    for c in cases:assert Path(rows[c['patient_id']]['slide_filename']).stem==c['slide_stem'],c
    return rows

def partition(cases,shard,nshards):
    """Deterministic round-robin over the lock order: each 12-case transfer batch feeds every shard."""
    return [c for k,c in enumerate(cases) if k%nshards==shard]

def recipe(layout,env):
    """Everything that determines the feature values; outputs made under another recipe are never mixed in."""
    weights={nm:json.loads(s0.SMOKE.read_text())['models'][nm]['weight_sha256'] for nm in s0.ENCODERS}  # load_models enforces equality
    r=dict(locations=s0.LOCATIONS,seed=s0.SEED,seed_rule="np.random.default_rng(stable_seed('highres_v1',SEED,patient_id)).choice(n,LOCATIONS) sorted; all if n<=LOCATIONS",
           batch=s0.BATCH,encoders={nm:dict(dim=d,interpolation=i,weight_sha256=weights[nm]) for nm,(d,i) in s0.ENCODERS.items()},
           inputs='L full 128um->224; H 2x2 native halves->224; Hd 2x2 quadrants of L->224',precision='fp16 autocast, float16 storage',
           stage0_module_sha256=s0.sha(s0.__file__),modules={k:v['sha256'] for k,v in layout.items()},environment=env)
    return r,hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()

def check_recipe_pin(rec_sha):
    f=OUT/'RECIPE.json'
    if f.exists() and json.loads(f.read_text())['recipe_sha256']!=rec_sha:
        raise RecipeMismatch(f'{f} pins recipe {json.loads(f.read_text())["recipe_sha256"]}, current {rec_sha} (code, weights, modules or '
                             f'library versions changed); refusing to mix: move the whole output directory away to re-extract')

def pin_recipe(rec,rec_sha):
    """The first extraction of any shard pins the recipe (atomic: link fails if another process pinned first)."""
    f=OUT/'RECIPE.json'
    if not f.exists():
        tmp=f.with_name(f'{f.name}.{os.getpid()}.tmp');tmp.write_text(json.dumps(dict(recipe_sha256=rec_sha,recipe=rec,pinned_unix=time.time()),indent=2))
        try:os.link(tmp,f)
        except FileExistsError:pass
        finally:tmp.unlink()
    check_recipe_pin(rec_sha)

def complete_record(dst,recipe_sha):
    """None if the case has no complete.json; the record if complete and every file re-hashes; raises otherwise."""
    f=dst/'complete.json'
    if not f.exists():return None
    c=json.loads(f.read_text())
    if c.get('state')!='complete':return None
    if c.get('recipe_sha256')!=recipe_sha:
        raise RecipeMismatch(f'{dst}: extracted under recipe {c.get("recipe_sha256")}, current {recipe_sha}; refusing to mix (move the old output away to re-extract)')
    for name,h in c['files'].items():
        if s0.sha(dst/name)!=h:raise RuntimeError(f'{dst/name}: sha256 differs from complete.json')
    return c

def stage0_gate(layout,env):
    """Protocol section 7: stage 1 starts only after the A6000 stage-0 automated checks passed with this exact stage-0
    code, helper modules, weight reference and library versions. Returns (exit code, reason): 0 passed; 3 stage 0 has
    not passed yet (not run, running, waiting, error); 1 stage 0 FAILED or no longer matches (a human must act)."""
    f=STAGE0_REPORT
    try:r=json.loads(f.read_text())
    except FileNotFoundError:return EXIT_WAITING,f'{f} does not exist: stage 0 has not run'
    except (OSError,ValueError) as e:return EXIT_WAITING,f'{f} unreadable: {e!r}'
    state=r.get('state')
    if state=='FAIL':return EXIT_ERROR,f'stage 0 FAILED ({str(r.get("error") or "automated checks")[:300]}); stage 1 must not start (protocol section 7)'
    if state!=s0.PASS_STATE:return EXIT_WAITING,f'stage 0 state {state!r}, not {s0.PASS_STATE}'
    problems=[]
    if r.get('code_sha256')!=s0.sha(s0.__file__):
        problems.append(f'stage0_extract.py changed since stage 0 passed (report {str(r.get("code_sha256"))[:12]}, current {s0.sha(s0.__file__)[:12]}): re-run stage 0')
    checks=r.get('checks') or {};L=checks.get('L_reproduces_stored') or {}
    if not (set(L)==set(s0.ENCODERS) and all(v.get('pass_') is True for v in L.values()) and checks.get('subtiles_union_exact') is True
            and checks.get('Hd_equals_L_quadrants') is True):problems.append('report checks are not all passed')
    if {k:(v or {}).get('sha256') for k,v in (r.get('modules') or {}).items()}!={k:v['sha256'] for k,v in layout.items()}:
        problems.append('helper module sha256s differ from the stage-0 run')
    if (r.get('weights_reference') or {}).get('sha256')!=s0.sha(s0.SMOKE):problems.append(f'weight reference {s0.SMOKE} changed since stage 0')
    plat=r.get('platform') or {};diff={k:[plat.get(k),v] for k,v in env.items() if plat.get(k)!=v}
    if diff:problems.append(f'library versions differ from the stage-0 run [stage0, now]: {json.dumps(diff)}')
    if problems:return EXIT_ERROR,'; '.join(problems)
    return EXIT_OK,f'stage 0 {s0.PASS_STATE} for this code ({f}, sha256 {s0.sha(f)})'

def missing_inputs(row):
    """What a case still waits for: the weights batch marker and the .verified markers of its CZI and stored features."""
    if not s0.GATED:return []
    gate=s0.TRANSFER/'verified_stage0'
    return ([] if gate.exists() else [str(gate)])+s0.missing_markers(s0.case_inputs(row))
def is_ready(row):return not missing_inputs(row)
def short(p):
    p=Path(p)
    for base,label in ((s0.ROOT,'K'),(s0.PROJECT,'P')):
        try:return f'{label}/{p.relative_to(base)}'
        except ValueError:pass
    return str(p)

def write_quarantine():
    f=OUT/'QUARANTINE.json'
    try:cur=json.loads(f.read_text())
    except (OSError,ValueError):cur=None
    if cur!=QUARANTINE:save_unique(f,QUARANTINE)  # refreshed whenever the conditions in this code change

def process_case(c,row,pool,embed,totals,recipe_sha,prov):
    started=time.monotonic();stem=c['slide_stem'];patient=c['patient_id'];dst=OUT/stem;dst.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(OUT).free>MIN_FREE_BYTES,'Keep 20 GiB free'
    (dst/'error.json').unlink(missing_ok=True)
    primary=s0.hpc(row['primary_path']);assert s0.sha(primary)==row['primary_sha256']
    with np.load(primary,allow_pickle=False) as z:coords=z['coords'].copy()
    n=int(row['primary_count']);assert len(coords)==n
    slide=s0.ROOT/'data/he'/row['slide_filename']
    receipt=json.loads(slide.with_name(slide.name+'.receipt.json').read_text());assert receipt['sha256_local']==row['source_sha256']
    ref=json.loads((s0.STORED/stem/'complete.json').read_text())
    assert ref['n']==n and ref['primary_coords_sha256']==row['primary_sha256'] and ref['source_sha256']==row['source_sha256'],'stored features belong to other coordinates'
    stored={}
    for nm,(d,_) in s0.ENCODERS.items():
        f=s0.STORED/stem/f'{nm}.npy';stored[nm]=s0.sha(f);assert stored[nm]==ref['features'][nm]['sha256'],f'{f} sha256 != stored complete.json'
        assert np.load(f,mmap_mode='r').shape==(n,d)
    idx=s0.location_indices(n,patient);sel=coords[idx];m=len(idx)
    feats,sizes,hd=s0.extract_locations(pool,str(slide),sel,embed,totals,hd_checked=False)
    assert hd and len(sizes)==m
    files={}
    for name,a in [('location_indices.npy',idx),('coords.npy',sel)]+[(f'{nm}_{arm}.npy',a) for nm,parts in feats.items() for arm,a in parts.items()]:
        assert a.dtype!=np.float16 or np.isfinite(a).all()
        write_npy(dst/name,a);files[name]=s0.sha(dst/name)
    parity={}
    for nm,parts in feats.items():
        cos=s0.cosine_to_stored(s0.STORED/stem/f'{nm}.npy',idx,parts['L'])
        parity[nm]=dict(min_cosine=float(cos.min()),mean_cosine=float(cos.mean()),pass_=bool(cos.min()>=MIN_COSINE))
    status='PASS' if all(v['pass_'] for v in parity.values()) else 'FAIL'
    record=dict(state='complete',status=status,patient_id=patient,slide_stem=stem,n_locations=m,primary_count=n,files=files,
                L_reproduces_stored=parity,Hd_equals_L_quadrants_first_location=hd,geometry=s0.geometry_record(patient,m,sizes,row),
                source_sha256=row['source_sha256'],primary_coords_sha256=row['primary_sha256'],stored_feature_sha256=stored,
                recipe_sha256=recipe_sha,code_sha256=s0.sha(__file__),stage0_module_sha256=s0.sha(s0.__file__),gpu=prov['platform'].get('gpu'),
                platform=prov['platform'],environment=prov['environment'],stage0_report=prov.get('stage0_report'),
                seconds=time.monotonic()-started,quarantined=True)
    s0.save(dst/'complete.json',record)
    return record

def main(argv=None):
    ap=argparse.ArgumentParser(description='High-res pilot stage 1 on lab-a6000 (120 locked cases).')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--workers',type=int,default=s0.WORKERS,help='CZI decode processes (default %(default)s)')
    ap.add_argument('--wait',type=float,default=0,help='keep polling up to this many idle seconds for newly verified cases before exiting 3 (default 0: exit at once)')
    ap.add_argument('--poll',type=float,default=60,help='seconds between polls when --wait > 0')
    ap.add_argument('--min-free-gib',type=float,default=s0.MIN_FREE_GIB,help='shared GPU: exit 3 (retry later) instead of loading models when less is free (default %(default)s)')
    args=ap.parse_args(argv)
    assert 0<=args.shard<args.nshards,'need 0 <= --shard < --nshards'
    assert not s0.PFX or s0.PFX.startswith('/'),'HE_PATH_PREFIX must be an absolute path'
    lock,cases=load_lock();order_matches=transfer_order(cases);rows=manifest_rows(lock,cases)
    mine=partition(cases,args.shard,args.nshards)
    grade_counts=dict(sorted(collections.Counter(f"fold{c['fold']}_ISUP{c['ISUP_grade']}" for c in cases).items()))  # counts from the lock only
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'locks').mkdir(exist_ok=True);(OUT/'shards').mkdir(exist_ok=True)
    write_quarantine()
    tag=f'shard{args.shard}of{args.nshards}'
    shard_lock=(OUT/'locks'/f'{tag}.lock').open('a')
    try:fcntl.flock(shard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print(f'BUSY another process holds {tag}',flush=True);return EXIT_BUSY
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))  # unwind: decode workers are killed in the handlers below
    layout=s0.check_import_layout();env=s0.env_fingerprint();rec,rec_sha=recipe(layout,env)
    prov=dict(environment=env,platform=s0.platform_info(args.device,query_gpu=False))
    summary=dict(state='running',shard=args.shard,nshards=args.nshards,assigned=[c['slide_stem'] for c in mine],list_sha256=LIST_SHA256,
                 lock_file=str(LOCK_FILE),transfer_order_matches_lock=order_matches,grade_counts_from_lock=grade_counts,
                 recipe=rec,recipe_sha256=rec_sha,code_sha256=s0.sha(__file__),modules=layout,workers=args.workers,
                 platform=prov['platform'],scheduler_job=os.getenv('HE_SCHED_JOB'),started_unix=time.time(),cases={},timing={})
    summary_file=OUT/'shards'/f'{tag}.json';history=OUT/'shards'/f'{tag}.runs.jsonl'
    def log_run(**kw):
        with history.open('a') as h:h.write(json.dumps(dict(kw,pid=os.getpid(),unix=time.time()))+'\n')
    log_run(event='start',recipe_sha256=rec_sha,device=args.device,scheduler_job=os.getenv('HE_SCHED_JOB'))
    print(json.dumps(dict(event='start',shard=tag,assigned=len(mine),grade_counts_from_lock=grade_counts,recipe_sha256=rec_sha)),flush=True)
    done={};errors={};busy=set();oom=set();waiting_for={};gate=dict(code=None,reason=None)
    pool=None;embed=None;idle_since=None;gpu_busy=False;pinned=False
    totals=dict(locations=0,decode_wait=0.,gpu=0.);t0=time.monotonic()
    def status_of(stem):
        if stem in errors:return 'ERROR'
        if stem in done:return done[stem]['status']
        return 'BUSY_ELSEWHERE' if stem in busy else 'GPU_OOM_RETRY' if stem in oom else 'WAITING'
    def write_summary(state):
        summary.update(state=state,cases={c['slide_stem']:status_of(c['slide_stem']) for c in mine},errors=errors,stage0_gate=dict(gate),
                       waiting_for={s:v for s,v in waiting_for.items() if s not in done and s not in errors},updated_unix=time.time(),
                       timing=dict(**totals,wall_seconds=time.monotonic()-t0,locations_per_second=totals['locations']/max(1e-9,time.monotonic()-t0)))
        s0.save(summary_file,summary)
    try:
        check_recipe_pin(rec_sha)
        while True:
            ready=[];waiting=[];busy.clear();progress=False;waiting_for.clear()
            for c in mine:
                stem=c['slide_stem']
                if stem in done or stem in errors:continue
                try:prev=complete_record(OUT/stem,rec_sha)
                except RecipeMismatch:raise
                except Exception as e:errors[stem]=str(e);print(json.dumps(dict(event='invalid_complete',case=stem,error=str(e))),flush=True);continue
                if prev:done[stem]=prev;continue
                miss=missing_inputs(rows[c['patient_id']])
                if miss:waiting.append(c);waiting_for[stem]=[short(m) for m in miss]
                else:ready.append(c)
            code0,why0=stage0_gate(layout,env)
            if code0!=gate['code']:print(json.dumps(dict(event='stage0_gate',code=code0,reason=why0)),flush=True)
            gate.update(code=code0,reason=why0)
            write_summary('running')
            if code0!=EXIT_OK:break  # protocol section 7: no extraction (and no GPU use) before stage 0 passed
            prov['stage0_report']=dict(path=str(STAGE0_REPORT),sha256=s0.sha(STAGE0_REPORT))
            if ready and embed is None:
                try:
                    import torch
                    torch.set_num_threads(2);torch.manual_seed(s0.SEED)
                    s0.require_gpu(args.device)
                    free=gpu_free_gib(args.device)
                    if free is not None and free<args.min_free_gib:
                        print(json.dumps(dict(event='gpu_busy',device=args.device,free_gib=round(free,1),need_gib=args.min_free_gib)),flush=True);gpu_busy=True;break
                    summary['platform']=prov['platform']=s0.platform_info(args.device)
                    pathology=s0.load_pathology();models,infos=s0.load_models(pathology,args.device)  # raises on weight sha256 mismatch
                    summary['weights']=dict(infos,reference=dict(path=str(s0.SMOKE),sha256=s0.sha(s0.SMOKE)))
                    embed=s0.embedder(pathology,models,args.device)
                except Exception as e:
                    if not s0.is_gpu_oom(e):raise
                    print(json.dumps(dict(event='gpu_oom_at_load',device=args.device,error=str(e)[:300])),flush=True);gpu_busy=True;break
            if ready and not pinned:pin_recipe(rec,rec_sha);pinned=True
            for c in ready:
                stem=c['slide_stem'];case_lock=(OUT/'locks'/f'{stem}.lock').open('a')
                try:
                    try:fcntl.flock(case_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:busy.add(stem);continue  # another process is extracting it
                    prev=complete_record(OUT/stem,rec_sha)  # it may have been finished elsewhere meanwhile
                    if prev:done[stem]=prev;progress=True;continue
                    if pool is None:pool=s0.decode_pool(args.workers,layout)
                    record=process_case(c,rows[c['patient_id']],pool,embed,totals,rec_sha,prov);done[stem]=record;progress=True
                    print(json.dumps(dict(event='case',case=stem,status=record['status'],seconds=round(record['seconds'],1),
                                          L_min_cosine={k:v['min_cosine'] for k,v in record['L_reproduces_stored'].items()})),flush=True)
                except RecipeMismatch:raise
                except Exception as e:
                    s0.close_pool(pool,kill=True);pool=None  # its decodes may still run or hang: never reuse those workers
                    if s0.is_gpu_oom(e):  # shared GPU: leave the case incomplete, stop, exit 3 (retry later)
                        oom.add(stem);gpu_busy=True;s0.save(OUT/stem/'error.json',dict(state='gpu_out_of_memory_retry_later',case=stem,traceback=traceback.format_exc()))
                        print(json.dumps(dict(event='gpu_oom',case=stem)),flush=True);break
                    errors[stem]=traceback.format_exc();s0.save(OUT/stem/'error.json',dict(state='error',case=stem,traceback=errors[stem]))
                    print(json.dumps(dict(event='error',case=stem,error=repr(e)[:500])),flush=True)
                finally:case_lock.close()
                write_summary('running')
            if gpu_busy:break
            if progress:idle_since=None;continue  # rescan: more cases may have been verified meanwhile
            if (waiting or busy) and args.wait>0:
                idle_since=idle_since or time.monotonic()
                if time.monotonic()-idle_since<args.wait:time.sleep(args.poll);continue
            break
    except Exception as e:
        s0.close_pool(pool,kill=True);pool=None
        if s0.is_gpu_oom(e):gpu_busy=True;print(json.dumps(dict(event='gpu_oom',device=args.device,error=str(e)[:300])),flush=True)
        else:errors['__run__']=traceback.format_exc();print(errors['__run__'],flush=True)
    except BaseException:  # SIGTERM (SystemExit) or Ctrl-C: kill the decode workers, record, exit
        s0.close_pool(pool,kill=True);pool=None;write_summary('interrupted');log_run(event='interrupted');raise
    finally:
        s0.close_pool(pool)
    remaining=[c['slide_stem'] for c in mine if c['slide_stem'] not in done and c['slide_stem'] not in errors]
    failed=[s for s,r in done.items() if r['status']!='PASS']
    if '__run__' in errors:state,code='error',EXIT_ERROR
    elif gate['code'] not in (None,EXIT_OK):
        state,code=('waiting_for_stage0',EXIT_WAITING) if gate['code']==EXIT_WAITING else ('blocked_by_stage0',EXIT_ERROR)
    elif remaining:state,code=('waiting_with_errors' if errors else 'waiting'),EXIT_WAITING  # errors stay recorded; later batches still run
    elif errors:state,code='error',EXIT_ERROR
    elif failed:state,code='complete_with_FAIL',EXIT_FAIL
    else:state,code='complete',EXIT_OK
    write_summary(state);log_run(event='exit',state=state,code=code,complete=len(done),waiting=len(remaining),errors=sorted(errors))
    print(json.dumps(dict(event='exit',shard=tag,state=state,code=code,complete=len(done),waiting=len(remaining),fail=failed,errors=sorted(errors),
                          stage0_gate=gate['reason'] if gate['code'] else None)),flush=True)
    return code

if __name__=='__main__':sys.exit(main())
