"""Technical check before the formal runs (PROTOCOL.md section 4). No performance number is computed: predictions are only
compared between legs and between chunk sizes, never with labels.
For every encoder x arm (A, B, C), on the 8 pre-defined check cases (2 per ISUP grade, the first by patient_id in fold 0's
training set):
  full leg:   train.py --check, 2 epochs uninterrupted;
  resume leg: the same with --stop-after-epochs 1 (exit 75), then a fresh process resumes epoch 2, on the other visible GPU
              when two are visible (--resume-device);
  equal:      model, optimizer, RNG states, history (without timings) and predictions bitwise identical between legs;
  chunks:     the 4 ISUP probabilities with eval chunk 4096, whole bag and 777 differ by at most the configured tolerance;
  labels:     the ISUP labels used equal the database label file read independently; arm B's auxiliary labels are each
              case's own Fuhrman 1-2 vs 3-4; arm C's are a permutation within each ISUP grade; arm A has none.
Writes CHECK_PASS.json (state PASS or FAIL, code signature, platform). Attempts live in check_runs/attempt_NNN/.
Only a failed comparison (or a leg exiting 1/2) records FAIL. A leg that was killed (e.g. by the OOM killer), timed out,
hit a transient GPU/RAM/lock problem or stopped on a signal proves nothing: nothing is recorded and the check exits 74/75.
Exit: 0 PASS; 1 FAIL; 2 usage; 3 features of the check cases not verified yet; 5 another check.py holds the lock;
74 transient (GPU busy/OOM, RAM, killed, leg timeout); 75 stopped by a signal; 78 device is not the configured platform."""
import signal,subprocess,argparse,fcntl,time
from common import *
SIG=dict(child=None,signals=[])
def on_signal(signum,frame):
    SIG['signals'].append(signal.Signals(signum).name);c=SIG['child']
    if c is not None:
        try:c.send_signal(signal.SIGINT)  # train.py: stop at once, exit 75
        except OSError:pass
class Interrupted(Exception):pass
class Transient(Exception):pass
class LegFailed(Exception):pass
class NotReady(Exception):pass

def leg(args,env,log,expect,timeout):
    if SIG['signals']:raise Interrupted(SIG['signals'])
    with open(log,'a') as f:
        f.write('$ %s\n'%' '.join(args));f.flush()
        p=subprocess.Popen([sys.executable,'-u','train.py']+args,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT);SIG['child']=p
        if SIG['signals']:p.send_signal(signal.SIGINT)  # a signal that arrived between Popen and SIG['child']
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)
            try:p.wait(timeout=120)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            SIG['child']=None;raise Transient(f'train.py {args} timed out after {timeout:.0f}s')
        SIG['child']=None
    if SIG['signals']:raise Interrupted(SIG['signals'])
    if rc==77:raise NotReady(f'train.py {args} exit 77')
    if rc in (74,76,79) or rc<0 or rc>=128:raise Transient(f'train.py {args} exit {rc}')  # killed / OOM killer / busy
    if rc==75 and expect!=75:raise Transient(f'train.py {args} exit 75 (stopped by a signal from outside)')
    if rc!=expect:raise LegFailed(f'train.py {args} exit {rc}, expected {expect} (see {log})')

def load_pt(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def same_tensors(a,b):
    import torch
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tensors(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(same_tensors(x,y) for x,y in zip(a,b))
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    return a==b
def strip_history(h):return [{k:v for k,v in e.items() if k not in ('seconds','gpu_peak')} for e in h]

def compare(full,resume,cfg,cases,arm):
    A=load_pt(full/'last.pt');B=load_pt(resume/'last.pt');r={}
    r['model_bitwise']=same_tensors(A['model'],B['model'])
    r['optimizer_bitwise']=same_tensors(A['optimizer'],B['optimizer'])
    rs=dict(A['rng']);rb=dict(B['rng'])
    r['rng_bitwise']=rs.keys()==rb.keys() and all(same_tensors(rs[k],rb[k]) if k!='numpy' else (rs[k][0]==rb[k][0] and np.array_equal(rs[k][1],rb[k][1]) and tuple(rs[k][2:])==tuple(rb[k][2:])) for k in rs)
    r['history_identical']=strip_history(A['history'])==strip_history(B['history'])
    r['epochs']=[A['next_epoch'],B['next_epoch']]
    receipts=[read_json(d/'complete.json') or {} for d in (full,resume)]
    cuda=A['platform']['device_type']=='cuda'
    r['legs_as_designed']=bool([i['start_epoch'] for i in A['invocations']]==[0] and [i['start_epoch'] for i in B['invocations']]==[0,1]
                               and all(x.get('state')=='COMPLETE' and x.get('check') is True for x in receipts)
                               and full.resolve()!=resume.resolve() and (not cuda or ('cuda' in rs and 'cuda' in rb)))
    PA=json.loads((full/'predictions.json').read_text());PB=json.loads((resume/'predictions.json').read_text())
    r['predictions_identical']=PA==PB
    ids=[c['patient_id'] for c in cases]
    ref={x['patient_id']:np.array(x['probabilities']) for x in PA['0']};assert sorted(ref)==sorted(ids)
    r['chunk_max_abs_diff']={ch:max(float(np.abs(np.array(x['probabilities'])-ref[x['patient_id']]).max()) for x in PA[ch]) for ch in PA}
    r['chunks_ok']=all(v<=cfg['check']['chunk_tolerance'] for v in r['chunk_max_abs_diff'].values()) and \
        set(PA)=={str(c) for c in cfg['check']['chunk_sizes']}
    MA=json.loads((full/'metadata.json').read_text());MB=json.loads((resume/'metadata.json').read_text())
    raw={x['patient_id']:x for x in read_csv(LABELS)}  # the database label file, read independently of load_cases()
    lock={c['patient_id']:int(raw[c['patient_id']]['ISUP_grade'])-1 for c in cases}
    own={c['patient_id']:int(int(raw[c['patient_id']]['Fuhrman_grade'])>=3) for c in cases}
    r['labels_used_match_lock']=MA['labels_used']==lock==MB['labels_used']
    used=MA['aux_labels_used'];r['aux_labels_as_designed']=used==MB['aux_labels_used'] and (
        used=={} if arm=='A' else used==own if arm=='B' else
        sorted(used)==sorted(own) and all(sorted(used[p] for p in grp)==sorted(own[p] for p in grp)
                                          for grp in [[p for p in own if lock[p]==g] for g in range(4)]))
    r['resume_devices']=[i['gpu'].get('gpu_uuid') or i['gpu'].get('device') for i in B['invocations']]
    r['cross_card_resume']=len(set(map(str,r['resume_devices'])))>1
    r['platform']=MA['platform']
    r['passed']=all(r[k] for k in ('model_bitwise','optimizer_bitwise','rng_bitwise','history_identical','predictions_identical','chunks_ok',
                                    'labels_used_match_lock','aux_labels_as_designed','legs_as_designed')) and r['epochs']==[cfg['check']['epochs']]*2
    return r

def other_card(device):
    try:
        import torch
        if torch.device(device).type!='cuda' or torch.cuda.device_count()<2:return device
        idx=torch.device(device).index or 0;return f'cuda:{1 if idx==0 else 0}'
    except Exception:return device

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device',default='cuda');p.add_argument('--resume-device',help='device of the resume legs (default: the other visible GPU if two are visible)')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--leg-timeout-seconds',type=float,default=2*3600)
    args=p.parse_args(argv)
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,on_signal)
    lk=(HERE/'check.run.lock').open('w')
    try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('CHECK_LOCKED another check.py is running',flush=True);return 5
    cfg=config()
    if args.device.split(':')[0]!=cfg['device']:print('REFUSED check on',args.device,'but the configured platform is',cfg['device'],flush=True);return 78
    cases,assign=load_cases(cfg);picked=check_cases(cases,assign,cfg)
    structure=label_structure(cases,assign,cfg)
    for enc in cfg['encoders']:
        miss=missing_data(picked,enc)
        if miss:print('NOT_READY',enc,miss,flush=True);return 3
    resume_device=args.resume_device or other_card(args.device)
    base=HERE/'check_runs';base.mkdir(exist_ok=True)
    n=1+max([int(d.name[8:]) for d in base.glob('attempt_*') if d.name[8:].isdigit()],default=0);att=base/f'attempt_{n:03d}';att.mkdir()
    env=dict(os.environ);sig=signature()
    print('CHECK_ATTEMPT',n,att,'cases',[c['patient_id'] for c in picked],'device',args.device,'resume_device',resume_device,flush=True)
    results={};state='PASS';error=None
    try:
        for enc in cfg['encoders']:
            for arm in cfg['arms']:
                d=att/f'{enc}_{arm}';log=d/'legs.log';d.mkdir()
                common_args=['--check','--encoder',enc,'--arm',arm,'--seed-index','0','--threads',str(args.threads)]
                T=args.leg_timeout_seconds
                leg(common_args+['--device',args.device,'--outroot',str(d/'full')],env,log,0,T)
                leg(common_args+['--device',args.device,'--outroot',str(d/'resume'),'--stop-after-epochs','1'],env,log,75,T)
                leg(common_args+['--device',resume_device,'--outroot',str(d/'resume')],env,log,0,T)
                leaf=lambda root:root/enc/arm/'seed0'/'check'
                r=compare(leaf(d/'full'),leaf(d/'resume'),cfg,picked,arm);results[f'{enc}:{arm}']=r
                print('CHECK',enc,arm,'PASS' if r['passed'] else 'FAIL',json.dumps({k:v for k,v in r.items() if k!='platform'}),flush=True)
                if not r['passed']:state='FAIL'
    except Interrupted as e:print('CHECK_STOPPED by',e.args[0],flush=True);return 75
    except Transient as e:print('CHECK_TRANSIENT',e,'(nothing recorded)',flush=True);return 74
    except NotReady as e:print('NOT_READY',e,flush=True);return 3
    except LegFailed as e:state='FAIL';error=str(e);print('CHECK_LEG_FAILED',e,flush=True)
    if sig!=signature():print('CHECK code changed while the check ran: not recorded',flush=True);return 1
    plats=[r['platform'] for r in results.values()]
    if plats and any(x!=plats[0] for x in plats):state='FAIL';error=(error or '')+' legs ran on different platforms'
    rec=dict(state=state,code_signature=sig,platform=plats[0] if plats else None,attempt=n,attempt_dir=str(att),finished=now(),
             cases=[c['patient_id'] for c in picked],labels={c['patient_id']:c['y'] for c in picked},label_structure=structure,
             cross_card_resume_proved=state=='PASS' and all(r['cross_card_resume'] for r in results.values()),
             results=results,error=error,note='no performance metric is computed by this check')
    save_json(att/'CHECK_RESULT.json',rec)
    save_json(CHECK_PASS,rec)  # the latest attempt decides: a FAIL revokes an earlier PASS
    print('CHECK',state,'cross_card_resume_proved',rec['cross_card_resume_proved'],flush=True)
    return 0 if state=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
