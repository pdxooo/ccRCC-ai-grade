"""Technical gate for fusion_gain_v2 (GPU port of v1). Uses training/validation cases of fold 0 only; never opens test
predictions. Parts: 'single' = checks with single:phikon-v2 (covers every single:* arm: same code path);
                    'mdrop4' = checks with mdrop4 (needs all four encoders verified).
One attempt of a part runs in smoke_runs/<arm>/attempt_NNN/ (attempts are numbered and never deleted):
  1. unit checks: toy shapes as in v1, plus every arm at its real input dimensions on the device with synthetic features
     (same seed -> bitwise identical training; checkpoint/restore mid-run -> bitwise identical; chunked evaluation).
  2. reduced shape (8+8 patients, 128 patches, 2 epochs): full run vs (1 epoch, exit 75, fresh-process resume on the
     resume device); 3. formal shape (all fold-0 train/val patients, 2048 train patches, eval chunk 4096, every val patch,
     2 epochs): the same comparison. Models, optimizer, RNG (incl. CUDA), selection, history and validation predictions
     must be bitwise identical. TIMING.json is taken from the formal-shape full run.
  The resume device is --resume-device, else the other visible GPU when two are visible (cross-card proof), else --device.
SMOKE_PASS.json parts.<part>.state: PASS; FAIL (a check failed: revokes an earlier PASS of the same code, formal runs of
the part are refused until a later attempt passes); INCOMPLETE (GPU busy/out of memory, not enough RAM, a leg timed out,
or a stop signal: proves nothing, so it never replaces PASS or FAIL); DEFERRED (auto mode, mdrop4 data not verified yet);
NOT_RUN. The top-level 'state' only says the file belongs to 'code_signature' (kept for the scheduler); see parts_summary.
CLI: python smoke.py [--part auto|single|mdrop4] [--device cuda] [--resume-device DEV] [--threads 4] [--leg-timeout-seconds S]
  auto (default): single, then mdrop4 if its data are verified, otherwise mdrop4 is recorded as DEFERRED.
  --part mdrop4: run later; adds parts.mdrop4 (and TIMING mdrop4) without touching parts.single, provided the code
                 signature is unchanged (a changed signature starts a fresh SMOKE_PASS.json).
Only one smoke.py runs at a time (smoke.run.lock). SIGTERM/SIGUSR1/SIGINT stop the running train.py leg at once (it is
sent SIGINT), the attempt is recorded INCOMPLETE and smoke.py exits 75.
Exit: 0 requested parts passed (auto may defer mdrop4); 1 a check failed (part FAIL, SMOKE_FAIL_<part>.json); 2 usage;
3 data for a requested part not verified; 4 parts passed but TIMING.json could not be written; 5 another smoke.py holds
smoke.run.lock (nothing done); 74 transient (GPU busy/out of memory, RAM, leg timeout; part INCOMPLETE); 75 stopped by a
signal (part INCOMPLETE); 78 device is not the configured platform."""
from common import *
import argparse,subprocess,time,fcntl,datetime,traceback,copy,signal,torch
from torch import nn
import torch.nn.functional as F
import train as T
import ready as R

SMOKE_ARM={'single':'single:phikon-v2','mdrop4':'mdrop4'}
STATES=('PASS','FAIL','INCOMPLETE','DEFERRED','NOT_RUN')
TRANSIENT_CODES={74,76,79,-9}  # train.py: GPU busy/OOM, RAM, run dir locked; -9: SIGKILL (host OOM killer). Says nothing about the code.
STATE_NOTE='Top-level state only says this file belongs to code_signature and root; formal runs are gated per part (parts.<part>.state must be PASS).'
def covers(part,cfg):return [a for a in cfg['arms'] if T.smoke_part(a,cfg)==part]
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

class Interrupted(Exception):
    """A stop signal reached smoke.py: the attempt proves nothing either way."""
class Transient(Exception):
    """GPU busy or out of memory, not enough RAM, or a leg timed out."""
SIG=dict(child=None,signals=[])
def on_signal(signum,frame):
    SIG['signals'].append(signal.Signals(signum).name);child=SIG['child']
    if child is not None:
        try:child.send_signal(signal.SIGINT)  # train.py: SIGINT = stop at once, exit 75 (resumable)
        except OSError:pass

def unit_checks(device,threads,cfg):
    T.seed_all(11,threads)
    for model in (T.GatedMIL([7,11],16,8,0),T.ModalityDropoutFusion([7,11,5],16,8,0,16)):
        model.to(device).eval();k=len(model.branches) if hasattr(model,'branches') else 2
        dims=[7,11] if k==2 else [7,11,5];x=[torch.randn(19,d).to(device) for d in dims]
        a=model(x);b=model.predict_chunks([[t[:5] for t in x],[t[5:12] for t in x],[t[12:] for t in x]])
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-5),'chunk equivalence'
        ix=torch.randperm(19).to(device);assert torch.allclose(a,model([t[ix] for t in x]),atol=1e-6,rtol=1e-5),'permutation invariance'
        model.train();F.cross_entropy(model(x),torch.tensor([2],device=device)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'gradients'
    m=T.ModalityDropoutFusion([7,11,5],16,8,0,16).to(device);x=[torch.randn(9,d).to(device) for d in (7,11,5)]
    F.cross_entropy(m(x,[True,False,True]),torch.tensor([1],device=device)).backward()
    assert m.branches[1][0].weight.grad is None and m.branches[0][0].weight.grad.abs().sum()>0,'dropped branch gets no gradient'
    masks=[T.branch_mask(4,.25,1,0,e,f'p{i}') for e in range(20) for i in range(50)]
    assert all(any(mk) for mk in masks) and T.branch_mask(4,.25,1,0,3,'p1')==T.branch_mask(4,.25,1,0,3,'p1')
    kept=np.mean([sum(mk) for mk in masks]);assert 2.7<kept<3.3,kept
    real=real_dim_checks(device,threads,cfg)
    return dict(chunk_equivalence=True,permutation_invariance=True,gradients=True,dropped_branch_no_grad=True,mask_deterministic_nonempty=True,mean_branches_kept=float(kept),real_dimensions=real,device=str(device))

def real_dim_checks(device,threads,cfg):
    """Every configured arm (mdrop4 included, whatever part runs) at its real input dimensions on `device`, with synthetic
    features: the same seed gives bitwise identical parameters, a checkpoint/restore after step 2 gives bitwise identical
    parameters, chunked evaluation (eval_chunk) is repeatable and matches one-pass evaluation."""
    out={};chunk=cfg['eval_chunk']
    for arm,spec in cfg['arms'].items():
        dims=[DIMS[m] for m in spec['members']];dropping=spec['fusion']=='modality_dropout_mean'
        g=torch.Generator().manual_seed(20261021)
        bags=[[torch.randn(cfg['train_patches'],d,generator=g).half() for d in dims] for _ in range(3)]
        big=[torch.randn(2*chunk+808,d,generator=g).half() for d in dims]
        def train(steps,resume=None):
            T.seed_all(7,threads);model=T.make_model(arm,cfg).to(device);first=0;snap=None
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);T.restore_rng(resume['rng'],device);first=resume['step']
            for step in range(first,steps):
                model.train();x=[b.to(device).float() for b in bags[step%3]]
                mask=T.branch_mask(len(dims),spec['branch_drop'],7,0,step,'synthetic') if dropping else None
                loss=F.cross_entropy(model(x,mask),torch.tensor([step%4],device=device),reduction='sum')
                assert bool(torch.isfinite(loss)),f'{arm}: non-finite loss'
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
                if step+1==2:snap=dict(model=T.cpu_state(model),optimizer=copy.deepcopy(opt.state_dict()),rng=T.rng_state(device),step=2)
            return model,snap
        m1,snap=train(4);m2,_=train(4);m3,_=train(4,snap)
        h=[T.param_hash(m) for m in (m1,m2,m3)]
        assert h[0]==h[1],f'{arm}: same seed gave different parameters at real dimensions (non-deterministic kernels)'
        assert h[0]==h[2],f'{arm}: checkpoint/restore after step 2 changed the parameters at real dimensions'
        m1.eval();xb=[t.to(device).float() for t in big];n=len(big[0])
        with torch.no_grad():one=m1(xb)
        c1=m1.predict_chunks([[t[i:i+chunk] for t in xb] for i in range(0,n,chunk)]);c2=m1.predict_chunks([[t[i:i+chunk] for t in xb] for i in range(0,n,chunk)])
        assert torch.equal(c1,c2),f'{arm}: chunked evaluation is not repeatable'
        assert torch.allclose(one,c1,atol=1e-5,rtol=1e-4),f'{arm}: chunked vs one-pass evaluation differ'
        out[arm]=dict(dims=dims,train_patches=cfg['train_patches'],eval_patches=n,steps=4,param_sha256=h[0],max_abs_chunk_vs_onepass=float((one-c1).abs().max()))
        del m1,m2,m3,snap,xb
    if device.type=='cuda':torch.cuda.empty_cache()
    return out

def params(path):return torch.load(path,map_location='cpu',weights_only=False)
def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def equal(a,b):
    """Recursive bitwise equality for checkpoint contents (tensors, numpy arrays, dicts, lists, tuples, scalars)."""
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def strip(history):return [{k:v for k,v in h.items() if k!='seconds'} for h in history]

def train_cmd(arm,outroot,device,threads,extra=()):
    return [sys.executable,'-u',str(HERE/'train.py'),'--arm',arm,'--seed-index','0','--fold','0','--device',device,'--threads',str(threads),'--outroot',str(outroot),*extra]
def call(cmd,expect,timeout):
    """Run one train.py leg as a child; signals to smoke.py are forwarded to it (on_signal)."""
    if SIG['signals']:raise Interrupted(f'stop signal {SIG["signals"]} before {" ".join(cmd[3:])}')
    print('RUN',' '.join(cmd),flush=True);child=subprocess.Popen(cmd);SIG['child']=child
    try:
        if SIG['signals']:child.send_signal(signal.SIGINT)  # a signal arrived while the child was being created
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise Transient(f'leg timed out after {timeout:.0f}s: {" ".join(cmd[3:])}')
    finally:SIG['child']=None
    if SIG['signals']:raise Interrupted(f'stop signal {SIG["signals"]} during {" ".join(cmd[3:])} (child exit {code})')
    if code in TRANSIENT_CODES:raise Transient(f'{" ".join(cmd[3:])} returned {code} (GPU busy/OOM, RAM, lock or OOM kill)')
    assert code==expect,f'{" ".join(cmd[3:])} returned {code}, expected {expect}'

def leg(root,arm):return root/arm.replace(':','_')/'seed0'/'fold0'
def compare(a,b,device,what):
    """Bitwise comparison of an uninterrupted 2-epoch run (a) with a paused (1 epoch) and resumed run (b)."""
    A=params(a/'last.pt');B=params(b/'last.pt')
    assert same(params(a/'best.pt')['model'],params(b/'best.pt')['model']),f'{what}: resume changed best model'
    assert same(A['model'],B['model']),f'{what}: resume changed last model'
    assert equal(A['optimizer'],B['optimizer']),f'{what}: resume changed optimizer state'
    assert equal(A['rng'],B['rng']),f'{what}: resume changed RNG state (incl. CUDA generator)'
    assert A['next_epoch']==B['next_epoch']==2 and A['best_epoch']==B['best_epoch'] and A['best_key']==B['best_key'],f'{what}: resume changed selection'
    assert strip(A['history'])==strip(B['history']),f'{what}: resume changed training/validation history'
    assert json.loads((a/'val_predictions.json').read_text())==json.loads((b/'val_predictions.json').read_text()),f'{what}: resume changed predictions'
    receipts={k:json.loads((r/'complete.json').read_text()) for k,r in (('full',a),('resume',b))}
    assert all(x['state']=='COMPLETE' and x['smoke'] for x in receipts.values()) and receipts['full']['parameters_changed_this_invocation'],f'{what}: receipts'
    assert not (a/'test_predictions.json').exists() and not (b/'test_predictions.json').exists(),'smoke must not touch test patients'
    if device.type=='cuda':assert 'cuda' in A['rng'] and 'cuda' in B['rng'],'CUDA RNG state missing from checkpoint'
    return {k:dict(fingerprint=v['fingerprint'],best_epoch=v['best_epoch'],epochs=v['epochs']) for k,v in receipts.items()}
def epoch_cards(run):
    res=json.loads((run/'resources.json').read_text());return [res[k].get('gpu_uuid') for k in sorted(res,key=int)]

def real_part(part,args,device,resume_device,adir):
    """Reduced-shape and formal-shape legs; returns (check record, timing record)."""
    arm=SMOKE_ARM[part];to=args.leg_timeout_seconds
    def legs(tag,extra):
        full=adir/f'{tag}_full';res=adir/f'{tag}_resume'
        t=time.monotonic();call(train_cmd(arm,full,args.device,args.threads,extra),0,to);wall=time.monotonic()-t
        call(train_cmd(arm,res,args.device,args.threads,[*extra,'--stop-after-epochs','1']),75,to)
        call(train_cmd(arm,res,resume_device,args.threads,extra),0,to)
        return leg(full,arm),leg(res,arm),wall
    a,b,_=legs('reduced',['--smoke']);reduced=compare(a,b,device,'reduced shape')
    fa,fb,wall=legs('formal',['--smoke','--smoke-shape','formal']);formal=compare(fa,fb,device,'formal shape')
    cards=epoch_cards(fb)  # [card of epoch 0 (first process), card of epoch 1 (resumed process)]
    checks=dict(resume_bitwise_identical=True,formal_shape_bitwise_identical=True,
                checked=['unit_toy','unit_real_dimensions_all_arms','best_model','last_model','optimizer','rng','selection','history','val_predictions','no_test_predictions','reduced_and_formal_shape'],
                resume_cards=cards,cross_card_resume=len(cards)==2 and None not in cards and cards[0]!=cards[1],receipts=dict(reduced=reduced,formal=formal))
    return checks,timing_from(fa,wall,args,device)
def timing_from(run,wall,args,device):
    h=json.loads((run/'history.json').read_text());res=json.loads((run/'resources.json').read_text())
    peak=[r['gpu_max_allocated_bytes'] for r in res.values() if r['gpu_max_allocated_bytes'] is not None]
    return dict(seconds_per_epoch=float(np.mean([e['seconds'] for e in h])),preload_and_overhead_seconds=wall-sum(e['seconds'] for e in h),gpu_max_allocated_bytes=max(peak) if peak else None,
                gpu_max_reserved_bytes=max([r['gpu_max_reserved_bytes'] or 0 for r in res.values()]) or None,max_rss_bytes=max(r['max_rss_bytes'] for r in res.values()),
                device=args.device,threads=args.threads,platform=T.platform(device),source='formal-shape full leg of the smoke (fold 0 train/val, 2 epochs; overhead includes preload and final validation)',finished=now())

def locked_update(name,mutate):
    """Read-modify-write of HERE/<name> under an exclusive lock."""
    with (HERE/(name+'.lock')).open('w') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX);path=HERE/name
        new=mutate(json.loads(path.read_text()) if path.exists() else None);save_json(path,new);return new
def record_part(part,result,cfg):
    """PASS and FAIL always replace the part's record (FAIL revokes a PASS); INCOMPLETE and DEFERRED never replace PASS or FAIL."""
    sig=signature()
    def mutate(cur):
        if cur is None or cur.get('code_signature')!=sig or cur.get('root')!=str(ROOT):
            cur=dict(state='PASS',state_note=STATE_NOTE,version=cfg['version'],code_signature=sig,root=str(ROOT),parts={p:dict(state='NOT_RUN',covers=covers(p,cfg)) for p in SMOKE_ARM},attempts=[])
        old=cur['parts'][part]['state'];applied=result['state'] in ('PASS','FAIL') or old in ('NOT_RUN','DEFERRED','INCOMPLETE')
        if applied:cur['parts'][part]=result
        if 'attempt' in result:cur.setdefault('attempts',[]).append(dict(part=part,attempt=result['attempt'],state=result['state'],applied=applied,error=result.get('error'),evidence=result.get('evidence'),finished=result.get('finished')))
        cur['parts_summary']={p:v['state'] for p,v in cur['parts'].items()};cur['state_note']=STATE_NOTE;cur['updated']=now()
        assert cur['state']=='PASS' and all(v['state'] in STATES for v in cur['parts'].values())
        return cur
    new=locked_update('SMOKE_PASS.json',mutate)
    if result['state']=='PASS':(HERE/f'SMOKE_FAIL_{part}.json').unlink(missing_ok=True)  # the failure stays in its attempt directory
    return new
def new_attempt(arm):
    base=HERE/'smoke_runs'/arm.replace(':','_');base.mkdir(parents=True,exist_ok=True)
    n=1+max([int(p.name[8:]) for p in base.glob('attempt_*') if p.name[8:].isdigit()],default=0)
    d=base/f'attempt_{n:03d}';d.mkdir();return n,d
def other_card(device):
    if device.type!='cuda' or torch.cuda.device_count()<2:return None
    return f'cuda:{1 if device.index==0 else 0}'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part',choices=['auto','single','mdrop4'],default='auto');p.add_argument('--device',default='cuda')
    p.add_argument('--resume-device',help='device for the resume legs (default: the other visible GPU if two are visible, else --device)')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--leg-timeout-seconds',type=float,default=4*3600,help='per train.py leg; a timeout makes the attempt INCOMPLETE (exit 74)')
    args=p.parse_args(argv)
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,on_signal)
    lock=(HERE/'smoke.run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('SMOKE_LOCKED another smoke.py holds',HERE/'smoke.run.lock','- nothing done',flush=True);return 5
    cfg=config()
    if torch.device(args.device).type!=cfg['device']:print('REFUSED smoke on',args.device,'but the configured platform is',cfg['device'],flush=True);return 78
    try:device=T.setup_device(args.device)
    except Exception as e:
        if not T.is_transient(e):raise
        print('TRANSIENT_GPU_FAILURE',repr(e),flush=True);return 74
    resume_device=args.resume_device or other_card(device) or args.device
    parts=['single','mdrop4'] if args.part=='auto' else [args.part];timings={}
    for part in parts:
        if SIG['signals']:print('SMOKE_STOPPED by',SIG['signals'],'before part',part,flush=True);return 75
        arm=SMOKE_ARM[part];status=R.arm_status(arm,cfg)
        if not status['ready']:
            if args.part=='auto' and part=='mdrop4':
                record_part(part,dict(state='DEFERRED',covers=covers(part,cfg),reason='data not verified: '+json.dumps(status['missing'][:10]),recorded=now()),cfg)
                timings[arm]='deferred';print('SMOKE_PART_DEFERRED',part,flush=True);continue
            print('NOT_READY',part,json.dumps(status['missing'][:10]),flush=True);return 3
        n,adir=new_attempt(arm);print('SMOKE_ATTEMPT',part,n,adir,'device',args.device,'resume_device',resume_device,flush=True)
        base=dict(part=part,smoke_arm=arm,covers=covers(part,cfg),attempt=n,evidence=str(adir),device=args.device,resume_device=resume_device,platform=T.platform(device),
                  gpu=T.gpu_identity(device),root=str(ROOT),he_path_prefix=PREFIX,code_signature=signature(),started=now())
        try:
            unit=unit_checks(device,args.threads,cfg);checks,timing=real_part(part,args,device,resume_device,adir)
        except Exception as e:
            kind='interrupted' if isinstance(e,Interrupted) else 'transient' if isinstance(e,Transient) or T.is_transient(e) else 'check_failed'
            rec=dict(base,state='FAIL' if kind=='check_failed' else 'INCOMPLETE',kind=kind,error=repr(e),traceback=traceback.format_exc(),signals=SIG['signals'],finished=now())
            save_json(adir/'attempt.json',rec)
            if kind=='check_failed':save_json(HERE/f'SMOKE_FAIL_{part}.json',rec)
            cur=record_part(part,rec,cfg)
            print('SMOKE_PART_FAILED' if kind=='check_failed' else 'SMOKE_PART_INCOMPLETE',part,kind,repr(e),'-> parts.'+part,cur['parts'][part]['state'],flush=True)
            return {'check_failed':1,'transient':74,'interrupted':75}[kind]
        result=dict(base,state='PASS',**checks,unit=unit,data_status=R.arm_status(arm,cfg)['encoders'],finished=now())
        save_json(adir/'attempt.json',result);record_part(part,result,cfg);timings[arm]=timing
        print('SMOKE_PASS',part,'attempt',n,'cross_card_resume',checks['cross_card_resume'],flush=True);print('TIMING',arm,json.dumps(timing),flush=True)
    if not timings:return 0
    try:
        def mutate(cur):
            cur=cur if cur and cur.get('code_signature')==signature() else dict(code_signature=signature())
            for arm,v in timings.items():
                if v!='deferred' or not isinstance(cur.get(arm),dict):cur[arm]=v
            return cur
        locked_update('TIMING.json',mutate);print('TIMING_DONE',flush=True)
    except Exception as e:print('TIMING_WRITE_FAILED',repr(e),flush=True);return 4
    return 0
if __name__=='__main__':raise SystemExit(main())
