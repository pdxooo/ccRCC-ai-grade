"""High-res stage 1: G1 vs G2 gated ABMIL on frozen L / H / Hd / Lfull features (PROTOCOL.md section 2).
One run = encoder x arm x seed index x fold: train 40 fixed epochs on the other four folds' 96 cases, keep the last
epoch, predict P(G2) for the fold's 24 test cases. No test metric is computed or printed here.
--check: protocol 7.6 mode (driven by check76.py): train on the 8 check cases and predict them with several chunk sizes.
Exit codes: 0 complete (or already complete); 1 integrity or check failure; 2 usage; 74 transient GPU problem (retry later);
75 paused or stopped by a signal (resumable, bitwise); 76 not enough available RAM; 77 inputs not extracted yet;
78 refused: LOCK.json, RELEASE.json or CHECK76_PASS.json not satisfied, or another platform; 79 run dir locked.
Signals: first SIGTERM/SIGUSR1 pauses at the end of the current epoch (at once before training starts); a second SIGTERM
or any SIGINT stops at once; both exit 75 and resume from the last epoch checkpoint."""
import signal
class StopNow(BaseException):
    """Raised by the signal handler for an immediate stop."""
STOP=dict(requested=False,terms=0,signals=[],armed=False)
def on_signal(signum,frame):
    STOP['requested']=True;STOP['signals'].append(signal.Signals(signum).name)
    if signum==signal.SIGTERM:STOP['terms']+=1
    if STOP['armed'] and (signum==signal.SIGINT or STOP['terms']>=2):raise StopNow(signal.Signals(signum).name)
def install_signal_handlers():
    for s in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(s,on_signal)
if __name__=='__main__':install_signal_handlers()  # before the slow imports: a signal now only sets the flag
from common import *
import argparse,time,random,fcntl,socket,resource,re,traceback
import torch
from torch import nn
import torch.nn.functional as F

class GatedMIL(nn.Module):
    """Same head as the earlier 4-grade models (mil_first_round_v2), with 2 output classes."""
    def __init__(self,dim,hidden=256,attention_hidden=128,dropout=.25,classes=2):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.av=nn.Linear(hidden,attention_hidden);self.au=nn.Linear(hidden,attention_hidden)
        self.score=nn.Linear(attention_hidden,1);self.classifier=nn.Linear(hidden,classes)
    def encode(self,x):
        h=self.project(F.normalize(x.float(),p=2,dim=1,eps=1e-8))
        return h,self.score(torch.tanh(self.av(h))*torch.sigmoid(self.au(h))).squeeze(1)
    def forward(self,x):
        h,a=self.encode(x);return self.classifier((torch.softmax(a,dim=0)[:,None]*h).sum(0,keepdim=True))
    @torch.no_grad()
    def predict_chunks(self,chunks):
        """Exact whole-case attention: chunked encoding, one softmax over all instances."""
        assert not self.training
        hs=[];scores=[]
        for x in chunks:
            h,a=self.encode(x);hs.append(h);scores.append(a)
        h=torch.cat(hs,0);a=torch.cat(scores,0)
        return self.classifier((torch.softmax(a,dim=0)[:,None]*h).sum(0,keepdim=True))

class Bags:
    """float16 features of one encoder x arm, one array per case, flattened to instances: L / Lfull (n, d); H / Hd
    (n, 4, d) -> (4n, d), location-major, so location i owns instances 4i..4i+3."""
    def __init__(self,cfg,enc,arm,device):
        self.cfg=cfg;self.enc=enc;self.arm=arm;self.device=device;self.arrays={};self.locations={};self.sha={}
        self.k=cfg['subtiles'] if arm in ('H','Hd') else 1
    def plan(self,cases):
        """(path, sha256, shape) per case from the extraction records; bytes to load."""
        self.expected={c['patient_id']:expected_feature(c,self.enc,self.arm,self.cfg) for c in cases}
        return sum(int(np.prod(s))*2 for _,_,s in self.expected.values())
    def preload(self,cases,stop=lambda:False):
        for i,c in enumerate(cases):
            if stop():return None
            path,digest,shape=self.expected[c['patient_id']]
            got=sha(path);assert got==digest,f'{path}: sha256 {got[:12]} != record {digest[:12]}'
            a=np.load(path,allow_pickle=False);assert a.dtype==np.float16 and a.shape==tuple(shape),f'{path}: {a.dtype} {a.shape} != {shape}'
            assert np.isfinite(a).all(),f'{path}: non-finite values'
            self.locations[c['patient_id']]=a.shape[0];self.arrays[c['patient_id']]=a.reshape(-1,a.shape[-1]);self.sha[c['patient_id']]=got
        return len(cases)
    def instances(self,patient,locs):
        if self.k==1:return locs
        return (locs[:,None]*self.k+np.arange(self.k)[None,:]).reshape(-1)
    def tensor(self,patient,locs):
        a=self.arrays[patient][self.instances(patient,locs)]
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=self.device,dtype=torch.float32)
    def chunks(self,patient,chunk):
        a=self.arrays[patient];step=len(a) if not chunk else chunk
        for i in range(0,len(a),step):yield torch.from_numpy(np.ascontiguousarray(a[i:i+step])).to(device=self.device,dtype=torch.float32)

# ------------------------------------------------------------------------------------------- device & state
class TransientGPU(RuntimeError):
    """CUDA unavailable, busy or out of memory: not an integrity problem; retry later (exit 74)."""
TRANSIENT_MARKERS=('out of memory','busy or unavailable','CUBLAS_STATUS_ALLOC_FAILED','CUBLAS_STATUS_NOT_INITIALIZED',
                   'CUDNN_STATUS_ALLOC_FAILED','CUDNN_STATUS_NOT_INITIALIZED','no CUDA-capable device','CUDA driver initialization failed')
def is_transient(e):
    if isinstance(e,TransientGPU) or isinstance(e,getattr(torch.cuda,'OutOfMemoryError',())):return True
    return isinstance(e,RuntimeError) and any(m in str(e) for m in TRANSIENT_MARKERS)
def setup_device(name):
    device=torch.device(name)
    if device.type=='cuda':
        try:
            if not torch.cuda.is_available():raise TransientGPU('CUDA requested but not available')
            if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
            torch.cuda.set_device(device);torch.zeros(1,device=device)  # a busy or full GPU fails here
        except TransientGPU:raise
        except RuntimeError as e:raise TransientGPU(f'CUDA initialisation failed on {device}: {e}') from e
    return device
def platform(device):
    p=dict(device_type=device.type,torch=torch.__version__,cuda=torch.version.cuda)
    if device.type=='cuda':p.update(gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),cudnn=torch.backends.cudnn.version())
    return p
def gpu_identity(device):
    """The physical card (not part of platform(): both A6000s are one platform)."""
    ident=dict(device=str(device),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),host=socket.gethostname(),pid=os.getpid())
    if device.type=='cuda':
        pr=torch.cuda.get_device_properties(device);u=getattr(pr,'uuid',None);raw=getattr(u,'bytes',None);uuid=None
        if raw is not None and len(raw)==16:h=bytes(bytearray(raw)).hex();uuid=f'GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'
        elif u is not None and re.fullmatch(r'(GPU-)?[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(u)):uuid=str(u) if str(u).startswith('GPU-') else 'GPU-'+str(u)
        ident.update(gpu_name=pr.name,gpu_uuid=uuid)
    return ident
def seed_all(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def rng_state(device):
    s=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if device.type=='cuda':s['cuda']=torch.cuda.get_rng_state(device)
    return s
def restore_rng(s,device):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(s['cuda'],device)
    else:assert 'cuda' not in s,'checkpoint holds a CUDA RNG state but the device is not CUDA'
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save_pt(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def mem_available():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    except OSError:return None
    return None
def raise_oom_score(value):
    try:
        with open('/proc/self/oom_score_adj','w') as f:f.write(str(int(value)))
        with open('/proc/self/oom_score_adj') as f:return int(f.read())
    except OSError:return None
def predict(model,bags,cases,chunk):
    model.eval();out=[]
    with torch.no_grad():
        for c in cases:
            p=model.predict_chunks(bags.chunks(c['patient_id'],chunk)).softmax(1)[0].cpu().numpy().astype(np.float64)
            assert np.isfinite(p).all() and abs(p.sum()-1)<1e-5
            out.append(dict(patient_id=c['patient_id'],fold=c['fold'],ISUP_grade=c['grade'],y=c['y'],p_G2=float(p[1]),
                            instances=int(bags.arrays[c['patient_id']].shape[0])))
    return out

# ---------------------------------------------------------------------------------------------------- gates
def formal_gate(cfg,device_type):
    """Everything a formal run needs before any CUDA context: (code, reason) or None."""
    if device_type!=cfg['device']:return 78,f'device type {device_type} is not the configured platform {cfg["device"]}'
    ok,bad=lock_status()
    if not ok:return 78,f'code is not the locked version (LOCK.json): {bad[:5]}'
    ok,missing=release_status()
    if not ok:return 78,f'stage-1 features are QUARANTINED: release conditions not recorded in RELEASE.json: {missing}'
    chk=read_json(CHECK_PASS) or {}
    if chk.get('state')!='PASS' or chk.get('code_signature')!=signature():return 78,f'CHECK76_PASS.json is not a PASS for this code (state {chk.get("state")})'
    return None
def pause(out,next_epoch,fid,reason):
    save_json(out/'paused.json',dict(next_epoch=next_epoch,fingerprint=fid,reason=reason,signals=STOP['signals'],time=now()))
    print('PAUSED',out,'next_epoch',next_epoch,reason,flush=True);return 75

def run(args):
    cfg=config()
    if args.encoder not in cfg['encoders'] or args.arm not in cfg['decision_arms']+cfg['reference_arms'] or args.seed_index not in cfg['seeds']:
        print('USAGE encoder/arm/seed not in config',flush=True);return 2
    cases,order=load_cases(cfg)
    if args.check:
        train=test=check_cases(cases,order,cfg);fold=-1;epochs=cfg['check76']['epochs'];chunks=cfg['check76']['chunk_sizes']
    else:
        if args.fold not in cfg['folds']:print('USAGE fold not in config',flush=True);return 2
        r=roles(cases,args.fold);train,test=r['train'],r['test'];fold=args.fold;epochs=cfg['epochs'];chunks=[cfg['eval_chunk']]
        refused=formal_gate(cfg,torch.device(args.device).type)
        if refused:print('REFUSED code',refused[0],refused[1],flush=True);return refused[0]
    try:
        miss=missing_data(train+test,args.encoder,args.arm)
    except AssertionError as e:print('INTEGRITY',e,flush=True);return 1
    if miss:print('NOT_READY',len(miss),'cases not extracted:',miss[:10],flush=True);return 77
    if STOP['requested']:print('STOPPED before start',STOP['signals'],flush=True);return 75
    out=(Path(args.outroot) if args.outroot else RUNS)/args.encoder/args.arm/f'seed{args.seed_index}'/('check' if args.check else f'fold{fold}')
    out.mkdir(parents=True,exist_ok=True)
    lockfile=(out/'run.lock').open('w')
    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('LOCKED',out,flush=True);return 79
    fingerprint=dict(code=signature(),config=cfg,encoder=args.encoder,arm=args.arm,seed_index=args.seed_index,fold=fold,check=args.check,
                     epochs=epochs,chunks=chunks,list_sha256=cfg['stage1_list_sha256'],split_sha256=sha(SPLITS/'patient_splits.csv'),
                     train=[c['patient_id'] for c in train],test=[c['patient_id'] for c in test],
                     labels={c['patient_id']:c['y'] for c in train+test},records={c['patient_id']:record_digest(c,args.arm) for c in train+test})
    fid=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    names=['last.pt','history.json','metadata.json','predictions.json']
    if (out/'complete.json').exists():
        receipt=json.loads((out/'complete.json').read_text());assert receipt['fingerprint']==fid,'complete.json belongs to another fingerprint'
        for n,h in receipt['artifacts'].items():assert sha(out/n)==h,f'{n} changed after completion'
        print('ALREADY_COMPLETE',out,flush=True);return 0
    STOP['armed']=True  # from here a SIGINT / second SIGTERM stops at once (exit 75; resume from the last epoch checkpoint)
    device=setup_device(args.device);plat=platform(device);ident=gpu_identity(device)
    if not args.check:
        chk=read_json(CHECK_PASS) or {}
        if plat!=chk.get('platform'):print('REFUSED code 78 the 7.6 check passed on another platform:',chk.get('platform'),'this is',plat,flush=True);return 78
        if chk.get('cross_card_resume_proved') is not True:print('REFUSED code 78 the 7.6 check did not prove cross-card resume (run it with both GPUs visible)',flush=True);return 78
    oom_adj=raise_oom_score(args.oom_score_adj) if args.oom_score_adj is not None else None
    seed=model_seed(cfg,args.seed_index,max(fold,0));seed_all(seed,args.threads)
    model=GatedMIL(cfg['encoders'][args.encoder],cfg['hidden'],cfg['attention_hidden'],cfg['dropout']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ys=np.array([c['y'] for c in train]);counts=np.bincount(ys,minlength=2);assert (counts>0).all()
    weights=torch.tensor(len(ys)/(2*counts),dtype=torch.float32,device=device)
    start_epoch=0;history=[];invocations=[]
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['fingerprint']==fid,'last.pt belongs to another fingerprint'
        if state['platform']!=plat:print('REFUSED code 78 resume on another platform:',state['platform'],'this is',plat,flush=True);return 78
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['next_epoch'];history=state['history'];invocations=state['invocations'];restore_rng(state['rng'],device)
    invocations=invocations+[dict(started=now(),start_epoch=start_epoch,gpu=ident,argv=sys.argv[1:])]
    bags=Bags(cfg,args.encoder,args.arm,device);need=bags.plan(train+test);avail=mem_available()
    print('CACHE_BYTES',need,'MEM_AVAILABLE',avail,'MARGIN_GIB',args.ram_margin_gib,flush=True)
    if avail is not None and need+args.ram_margin_gib*1024**3>avail:print('INSUFFICIENT_RAM',out,flush=True);return 76
    if bags.preload(train+test,stop=lambda:STOP['requested']) is None:return pause(out,start_epoch,fid,'stop requested during preload')
    metadata=dict(fingerprint=fid,contract=fingerprint,platform=plat,seed=seed,class_counts=counts.tolist(),class_weights=weights.cpu().tolist(),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),feature_sha256=bags.sha,
                  locations={p:n for p,n in bags.locations.items()},labels_used={c['patient_id']:int(c['y']) for c in train},
                  invocations=invocations,threads=args.threads,he_path_prefix=PREFIX,oom_score_adj=oom_adj,
                  cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),inherited_cublas_workspace_config=INHERITED_CUBLAS_WORKSPACE_CONFIG,
                  scheduler_job=os.environ.get('HE_SCHED_JOB'))
    save_json(out/'metadata.json',metadata)
    t0=time.monotonic();A=cfg['accumulate_patients'];K=cfg['train_locations']
    for epoch in range(start_epoch,epochs):
        if STOP['requested']:return pause(out,epoch,fid,'signal')
        model.train();optimizer.zero_grad(set_to_none=True);t=time.monotonic();loss_sum=0.
        perm=train_order(len(train),cfg,args.seed_index,fold,epoch)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for i,j in enumerate(perm):
            c=train[int(j)];y=c['y']
            locs=sample_locations(bags.locations[c['patient_id']],K,cfg,args.seed_index,fold,epoch,c['patient_id'])
            logits=model(bags.tensor(c['patient_id'],locs))
            loss=F.cross_entropy(logits,torch.tensor([y],device=device),reduction='sum')*weights[y]
            assert bool(torch.isfinite(loss)),'non-finite training loss'
            g0=(i//A)*A;gs=min(A,len(perm)-g0);(loss/gs).backward();loss_sum+=float(loss.detach())
            if (i+1)%A==0 or i+1==len(perm):
                nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        history.append(dict(epoch=epoch,training_loss=loss_sum/len(perm),seconds=time.monotonic()-t,
                            gpu_peak=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None))
        save_pt(out/'last.pt',dict(fingerprint=fid,model=cpu_state(model),optimizer=optimizer.state_dict(),next_epoch=epoch+1,
                                   history=history,invocations=invocations,rng=rng_state(device),platform=plat))
        save_json(out/'history.json',history)
        print('EPOCH',args.encoder,args.arm,args.seed_index,fold,epoch,'train_loss',round(history[-1]['training_loss'],5),
              'seconds',round(history[-1]['seconds'],2),flush=True)
        if epoch+1<epochs:
            if STOP['requested']:return pause(out,epoch+1,fid,'signal')
            if args.stop_after_epochs and epoch+1>=args.stop_after_epochs:return pause(out,epoch+1,fid,'--stop-after-epochs')
            if time.monotonic()-t0>args.max_runtime_seconds:return pause(out,epoch+1,fid,'max runtime')
    assert len(history)==epochs and [h['epoch'] for h in history]==list(range(epochs))
    save_json(out/'history.json',history)  # a crash after the last checkpoint must not leave a shorter history behind
    preds={str(ch):predict(model,bags,test,ch) for ch in chunks}
    STOP['armed']=False
    save_json(out/'predictions.json',preds if args.check else preds[str(cfg['eval_chunk'])])
    (out/'paused.json').unlink(missing_ok=True)
    gpus=sorted({i['gpu'].get('gpu_uuid') or i['gpu'].get('cuda_visible_devices') for i in invocations},key=str)
    receipt=dict(state='COMPLETE',check=args.check,fingerprint=fid,encoder=args.encoder,arm=args.arm,seed_index=args.seed_index,fold=fold,
                 epochs=len(history),platform=plat,physical_gpus=gpus,cross_card_resume=len(gpus)>1,finished=now(),
                 artifacts={n:sha(out/n) for n in names})
    save_json(out/'complete.json',receipt);print('COMPLETE',args.encoder,args.arm,args.seed_index,fold,len(history),flush=True);return 0

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--encoder',required=True);p.add_argument('--arm',required=True);p.add_argument('--seed-index',type=int,required=True)
    p.add_argument('--fold',type=int,default=-1);p.add_argument('--check',action='store_true',help='protocol 7.6 mode (check76.py)')
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--outroot')
    p.add_argument('--stop-after-epochs',type=int);p.add_argument('--max-runtime-seconds',type=float,default=22*3600)
    p.add_argument('--ram-margin-gib',type=float,default=64.0,help='refuse to preload (exit 76) unless MemAvailable exceeds the cache by this much')
    p.add_argument('--oom-score-adj',type=int,default=500,help='raise our own OOM score so the kernel kills this (resumable) run first')
    args=p.parse_args(argv)
    try:return run(args)
    except StopNow as e:print('STOPPED_NOW signal',e.args[0],'(resumable from the last epoch checkpoint)',flush=True);return 75
    except Exception as e:
        if is_transient(e):traceback.print_exc();print('TRANSIENT_GPU_FAILURE',repr(e)[:500],'(exit 74; retry later)',flush=True);return 74
        raise
    finally:  # a late signal during interpreter teardown must not turn the exit code into -15
        for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,signal.SIG_IGN)
if __name__=='__main__':raise SystemExit(main())
