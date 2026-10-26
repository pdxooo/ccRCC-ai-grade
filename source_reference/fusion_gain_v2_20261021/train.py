"""Fusion-gain round v2: GPU port of fusion_gain_v1 (frozen-feature gated ABMIL). Same splits, sampling and head as the
first round; selection by validation cross-entropy; 4 seeds; adds a modality-dropout mean-fusion arm.
Exit codes: 0 complete (or already complete); 1 integrity or check failure (do not retry blindly); 2 usage error;
74 transient GPU problem (CUDA unavailable/busy, out of GPU memory; resumable, retry later); 75 paused or stopped by a
signal (resumable); 76 deferred: not enough available CPU RAM; 77 deferred: arm data not verified (formal only);
78 refused: no passing smoke part, wrong platform, or a resume on another platform; 79 run dir locked by another process.
Signals: the first SIGTERM or SIGUSR1 pauses at the next safe point (end of the current epoch; at once while the cache is
preloading or before training starts) and exits 75. A second SIGTERM, or any SIGINT, stops at once and exits 75; the run
then resumes from the last epoch checkpoint (last.pt), bitwise identically. SIGUSR1 never stops at once."""
import signal
class StopNow(BaseException):
    """Raised by the signal handler for an immediate stop (second SIGTERM or any SIGINT)."""
STOP=dict(requested=False,terms=0,signals=[],armed=False)
def on_signal(signum,frame):
    STOP['requested']=True;STOP['signals'].append(signal.Signals(signum).name)
    if signum==signal.SIGTERM:STOP['terms']+=1
    if STOP['armed'] and (signum==signal.SIGINT or STOP['terms']>=2):raise StopNow(signal.Signals(signum).name)
def install_signal_handlers():
    for s in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(s,on_signal)
if __name__=='__main__':install_signal_handlers()  # before the slow imports below: a signal now only sets the flag
from common import *
import argparse,time,random,fcntl,socket,resource,traceback,datetime,re
import torch
from torch import nn
import torch.nn.functional as F
import ready as R

class GatedMIL(nn.Module):
    def __init__(self,dimensions,hidden=256,attention_hidden=128,dropout=.25):
        super().__init__();self.dimensions=list(dimensions)
        self.project=nn.Sequential(nn.Linear(sum(dimensions),hidden),nn.ReLU(),nn.Dropout(dropout))
        self.av=nn.Linear(hidden,attention_hidden);self.au=nn.Linear(hidden,attention_hidden)
        self.score=nn.Linear(attention_hidden,1);self.classifier=nn.Linear(hidden,4)
    def encode(self,blocks,mask=None):
        x=torch.cat([F.normalize(b.float(),p=2,dim=1,eps=1e-8) for b in blocks],dim=1)
        h=self.project(x);a=self.score(torch.tanh(self.av(h))*torch.sigmoid(self.au(h))).squeeze(1)
        return h,a
    def forward(self,blocks,mask=None):
        h,a=self.encode(blocks,mask);return self.classifier((torch.softmax(a,dim=0)[:,None]*h).sum(0,keepdim=True))
    @torch.no_grad()
    def predict_chunks(self,chunks):
        assert not self.training
        hs=[];scores=[]
        for blocks in chunks:
            h,a=self.encode(blocks);hs.append(h);scores.append(a)
        h=torch.cat(hs,0);a=torch.cat(scores,0)
        return self.classifier((torch.softmax(a,dim=0)[:,None]*h).sum(0,keepdim=True))

class ModalityDropoutFusion(GatedMIL):
    """Per-encoder projection to branch_dim, mean over the encoders kept for this bag, then the same gated ABMIL.
    Training drops whole encoders per bag (mask); evaluation always uses all encoders."""
    def __init__(self,dimensions,hidden,attention_hidden,dropout,branch_dim=128):
        super().__init__([branch_dim],hidden,attention_hidden,dropout)
        self.branches=nn.ModuleList([nn.Sequential(nn.Linear(d,branch_dim),nn.ReLU(),nn.Dropout(dropout)) for d in dimensions])
    def encode(self,blocks,mask=None):
        assert len(blocks)==len(self.branches) and len({b.shape[0] for b in blocks})==1
        keep=range(len(blocks)) if mask is None else [i for i,m in enumerate(mask) if m]
        assert len(keep)>0
        z=torch.stack([self.branches[i](F.normalize(blocks[i].float(),p=2,dim=1,eps=1e-8)) for i in keep],dim=1).mean(1)
        h=self.project(z);a=self.score(torch.tanh(self.av(h))*torch.sigmoid(self.au(h))).squeeze(1)
        return h,a

def branch_mask(k,p,seed,fold,epoch,patient):
    rng=np.random.default_rng(stable_seed(seed,fold,epoch,patient,'branch_mask'))
    m=rng.random(k)>=p
    if not m.any():m[rng.integers(k)]=True
    return m.tolist()

def make_model(arm,cfg):
    spec=cfg['arms'][arm];dims=[DIMS[x] for x in spec['members']]
    if spec['fusion']=='none':return GatedMIL(dims,cfg['hidden'],cfg['attention_hidden'],cfg['dropout'])
    assert spec['fusion']=='modality_dropout_mean'
    return ModalityDropoutFusion(dims,cfg['hidden'],cfg['attention_hidden'],cfg['dropout'],spec['branch_dim'])

def max_rss_bytes():
    r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;return int(r if sys.platform=='darwin' else r*1024)
def mem_available():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    except OSError:pass
    return None
def raise_oom_score(value):
    """Make this (resumable) process the kernel OOM killer's preferred victim on the shared box. Raising needs no privilege."""
    try:
        with open('/proc/self/oom_score_adj','w') as f:f.write(str(int(value)))
        with open('/proc/self/oom_score_adj') as f:return int(f.read())
    except (OSError,ValueError):return None

class BagStore:
    """Per-process CPU-RAM cache of float16 features (no sharing between processes); batches are moved to `device`."""
    def __init__(self,lookup,encoders,device='cpu'):self.lookup=lookup;self.encoders=encoders;self.device=torch.device(device);self.cache={}
    def path(self,patient):return FEATURES/Path(self.lookup[patient]['slide_filename']).stem
    def load(self,patient):
        if patient not in self.cache:
            row=self.lookup[patient];folder=self.path(patient)
            assert (folder/'complete.json').exists(),f'Incomplete features: {patient}'
            arrays=[]
            for name in self.encoders:
                a=np.load(folder/(name+'.npy'),allow_pickle=False)
                assert a.shape==(int(row['primary_count']),DIMS[name]) and a.dtype==np.float16
                arrays.append(a)
            self.cache[patient]=arrays
        return self.cache[patient]
    def need(self,patients):return sum(int(self.lookup[p]['primary_count']) for p in patients)*sum(DIMS[x] for x in self.encoders)*2
    def preload(self,patients,stop=lambda:False):
        """Returns the bytes loaded, or None if stop() became true (the cache is then dropped)."""
        need=self.need(patients)
        allocation=os.environ.get('SLURM_MEM_PER_NODE')
        if allocation and allocation.isdigit():assert need+8*1024**3<int(allocation)*1024**2,'Insufficient requested CPU RAM'
        print('CACHE_BYTES',need,'GiB',round(need/1024**3,2),'patients',len(patients),'encoders',','.join(self.encoders),flush=True)
        t=time.monotonic()
        for i,p in enumerate(patients):
            if stop():self.cache.clear();print('CACHE_ABORTED after',i,'patients (stop requested)',flush=True);return None
            self.load(p)
            if (i+1)%25==0:print('CACHE_PATIENTS',i+1,'/',len(patients),flush=True)
        loaded=sum(a.nbytes for p in patients for a in self.cache[p]);assert loaded==need
        print('CACHE_LOADED bytes',loaded,'GiB',round(loaded/1024**3,2),'seconds',round(time.monotonic()-t,1),'max_rss_bytes',max_rss_bytes(),flush=True)
        return loaded
    def tensors(self,patient,indices):
        # float16 -> device -> float32 (exact conversion; identical values to v1's CPU .float()).
        return [torch.from_numpy(np.ascontiguousarray(a[indices])).to(self.device).float() for a in self.load(patient)]
    def chunks(self,patient,chunk,limit=None):
        n=int(self.lookup[patient]['primary_count'])
        indices=patch_indices(n,limit,20261014,0,-1,patient)
        for i in range(0,len(indices),chunk):yield self.tensors(patient,indices[i:i+chunk])

class TransientGPU(Exception):
    """CUDA unavailable, busy or out of memory: not an integrity problem; the run can be retried later (exit 74)."""
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
            torch.cuda.set_device(device);torch.zeros(1,device=device)  # create the context now: a busy or full GPU fails here
        except TransientGPU:raise
        except RuntimeError as e:raise TransientGPU(f'CUDA initialisation failed on {device}: {e}') from e
    return device
def platform(device):
    p=dict(device_type=device.type,torch=torch.__version__,cuda=torch.version.cuda)
    if device.type=='cuda':p.update(gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),cudnn=torch.backends.cudnn.version())
    return p
def gpu_identity(device):
    """Which physical card this process uses (platform() deliberately excludes it: both A6000s are one platform)."""
    ident=dict(device=str(device),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'),host=socket.gethostname(),pid=os.getpid())
    if device.type=='cuda':
        pr=torch.cuda.get_device_properties(device);u=getattr(pr,'uuid',None);raw=getattr(u,'bytes',None);uuid=None
        if raw is not None and len(raw)==16:h=bytes(bytearray(raw)).hex();uuid=f'GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'  # nvidia-smi format
        elif u is not None and re.fullmatch(r'(GPU-)?[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}',str(u)):uuid=str(u) if str(u).startswith('GPU-') else 'GPU-'+str(u)
        ident.update(gpu_name=pr.name,gpu_uuid=uuid,pci_domain_id=getattr(pr,'pci_domain_id',None),pci_bus_id=getattr(pr,'pci_bus_id',None),pci_device_id=getattr(pr,'pci_device_id',None))
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
    else:assert 'cuda' not in s,'checkpoint holds CUDA RNG state but device is not CUDA'
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def param_hash(model):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
def save_pt(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def evaluate(model,store,patients,cfg,role,fold,limit=None):
    model.eval();records=[]
    with torch.no_grad():
        for patient in patients:
            logits=model.predict_chunks(store.chunks(patient,cfg['eval_chunk'],limit))
            probs=logits.softmax(1)[0].cpu().numpy();assert np.isfinite(probs).all()
            records.append(dict(patient_id=patient,fold=fold,role=role,ISUP_grade=int(store.lookup[patient]['ISUP_grade']),probabilities=probs.tolist()))
    return records,metrics([r['ISUP_grade']-1 for r in records],[r['probabilities'] for r in records])

def roles_for(assign,fold):
    return {r:sorted(x['patient_id'] for x in assign if int(x['fold'])==fold and x['role']==r) for r in ('train','val','test')}

def smoke_part(arm,cfg):
    """Smoke part that must have passed before formal training of `arm`: 'single' covers every single:* arm, 'mdrop4' the fusion arm."""
    return 'single' if cfg['arms'][arm]['fusion']=='none' else arm
def formal_gate(arm,cfg,device_type):
    """Checks that need no GPU. None if formal training of `arm` may start, else (exit_code, reason)."""
    if device_type!=cfg['device']:return 78,f'device type {device_type} is not the configured platform {cfg["device"]}'
    status=R.arm_status(arm,cfg)
    if not status['ready']:return 77,'data not verified: '+json.dumps(status['missing'][:10])
    path=HERE/'SMOKE_PASS.json'
    if not path.exists():return 78,'SMOKE_PASS.json missing'
    smoke=json.loads(path.read_text());part=smoke_part(arm,cfg);rec=smoke.get('parts',{}).get(part,{})
    if smoke.get('state')!='PASS' or smoke.get('code_signature')!=signature():return 78,'SMOKE_PASS.json does not belong to this code signature'
    if rec.get('state')!='PASS' or arm not in rec.get('covers',[]):return 78,f'smoke part {part} has not passed (state {rec.get("state")})'
    if rec.get('code_signature')!=signature() or rec.get('root')!=str(ROOT):return 78,f'smoke part {part} ran with other code or another data root'
    return None
def platform_gate(arm,cfg,plat):
    rec=json.loads((HERE/'SMOKE_PASS.json').read_text())['parts'][smoke_part(arm,cfg)]
    if rec.get('platform')!=plat:return 78,f'smoke part {smoke_part(arm,cfg)} ran on {rec.get("platform")}, this is {plat}'
    return None
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def pause(out,next_epoch,fid,reason):
    save_json(out/'paused.json',dict(next_epoch=next_epoch,fingerprint=fid,reason=reason,signals=STOP['signals'],time=now()))
    print('PAUSED',out,'next_epoch',next_epoch,reason,flush=True);return 75

def run(args):
    cfg=config();lookup,assign=load_inputs();assert args.arm in cfg['arms'] and 0<=args.fold<5 and args.seed_index in cfg['seeds']
    spec=cfg['arms'][args.arm];encoders=spec['members'];roles=roles_for(assign,args.fold);eval_limit=None
    if args.smoke:
        if args.smoke_shape=='reduced':
            for role in ('train','val'):
                roles[role]=[p for grade in range(1,5) for p in [x for x in roles[role] if int(lookup[x]['ISUP_grade'])==grade][:2]]
                assert len(roles[role])==8
            cfg=dict(cfg,max_epochs=args.smoke_epochs,min_epochs=args.smoke_epochs,train_patches=128,eval_chunk=128);eval_limit=256
        else:  # formal shapes (all fold-0 train/val patients, 2048 train patches, eval chunk 4096, every val patch); epochs only
            cfg=dict(cfg,max_epochs=args.smoke_epochs,min_epochs=args.smoke_epochs)
        roles['test']=[]
    else:
        refused=formal_gate(args.arm,cfg,torch.device(args.device).type)  # before any CUDA context is created
        if refused:print('REFUSED',args.arm,'code',refused[0],refused[1],flush=True);return refused[0]
    if STOP['requested']:print('STOPPED before start',STOP['signals'],flush=True);return 75
    device=setup_device(args.device);plat=platform(device);ident=gpu_identity(device)
    if not args.smoke:
        refused=platform_gate(args.arm,cfg,plat)
        if refused:print('REFUSED',args.arm,'code',refused[0],refused[1],flush=True);return refused[0]
        ready=json.loads(DATA_READY.read_text())
        assert ready['state']=='PASS' and ready['patients']==300
        for p in lookup:assert sha(FEATURES/Path(lookup[p]['slide_filename']).stem/'complete.json')==ready['completion_sha256'][p]
    seed=arm_seed(cfg,args.seed_index,args.fold)
    runroot=Path(args.outroot) if args.outroot else HERE/('smoke_runs' if args.smoke else 'runs')
    out=runroot/args.arm.replace(':','_')/f'seed{args.seed_index}'/f'fold{args.fold}';out.mkdir(parents=True,exist_ok=True)
    lockfile=(out/'run.lock').open('w')
    try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('LOCKED',out,flush=True);return 79
    fingerprint=dict(code=signature(),config=cfg,arm=args.arm,seed_index=args.seed_index,fold=args.fold,split_sha256=sha(SPLITS/'patient_splits.csv'),pool_sha256=sha(POOL/'LOCK.json'),smoke=args.smoke,patients=roles)
    if args.smoke:fingerprint['smoke_shape']=args.smoke_shape
    fid=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    if (out/'complete.json').exists():
        receipt=json.loads((out/'complete.json').read_text());assert receipt['fingerprint']==fid
        for name,digest in receipt['artifacts'].items():assert sha(out/name)==digest
        print('ALREADY_COMPLETE',out,flush=True);return 0
    oom_adj=raise_oom_score(args.oom_score_adj) if args.oom_score_adj is not None else None
    seed_all(seed,args.threads)
    model=make_model(args.arm,cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    train_labels=np.array([int(lookup[p]['ISUP_grade'])-1 for p in roles['train']]);counts=np.bincount(train_labels,minlength=4)
    assert (counts>0).all();weights=torch.tensor(len(train_labels)/(4*counts),dtype=torch.float32,device=device)
    start_epoch=0;history=[];best_key=float('inf');best_model=None;best_epoch=-1;bad=0;finished_training=False
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['fingerprint']==fid
        if state['platform']!=plat:print('REFUSED','code',78,f'resume refused: run started on {state["platform"]}, this is {plat}',flush=True);return 78
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['next_epoch'];history=state['history'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch'];bad=state['bad'];finished_training=state['finished_training'];restore_rng(state['rng'],device)
    init_hash=param_hash(model)
    store=BagStore(lookup,encoders,device);t0=time.monotonic()
    if not finished_training:
        need=store.need(roles['train']+roles['val']);avail=mem_available()
        print('MEM_AVAILABLE',avail,'NEED',need,'MARGIN_GIB',args.ram_margin_gib,flush=True)
        if avail is not None and need+args.ram_margin_gib*1024**3>avail:print('INSUFFICIENT_RAM',out,flush=True);return 76
        if store.preload(roles['train']+roles['val'],stop=lambda:STOP['requested']) is None:return pause(out,start_epoch,fid,'stop requested during preload')
    metadata=dict(fingerprint=fid,contract=fingerprint,class_counts=counts.tolist(),class_weights=weights.cpu().tolist(),trainable_parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,numpy=np.__version__,threads=args.threads,job_id=os.environ.get('SLURM_JOB_ID'),seed=seed,
                  device=str(device),platform=plat,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),host=socket.gethostname(),he_path_prefix=PREFIX,
                  invocation=dict(started=now(),start_epoch=start_epoch,gpu=ident,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),inherited_cublas_workspace_config=INHERITED_CUBLAS_WORKSPACE_CONFIG,
                                  ram_margin_gib=args.ram_margin_gib,oom_score_adj=oom_adj,scheduler_job=os.environ.get('HE_SCHED_JOB')))
    save_json(out/'metadata.json',metadata)
    k=len(encoders);dropping=spec['fusion']=='modality_dropout_mean';resources=json.loads((out/'resources.json').read_text()) if (out/'resources.json').exists() else {}
    for epoch in range(start_epoch,cfg['max_epochs']):
        if finished_training:break
        if STOP['requested']:return pause(out,epoch,fid,'stop requested before the epoch started')
        if epoch==start_epoch:(out/'paused.json').unlink(missing_ok=True)  # training resumed: the pause note is history
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        model.train();optimizer.zero_grad(set_to_none=True);epoch_start=time.monotonic();loss_sum=0.
        order=np.random.default_rng(stable_seed(seed,args.fold,epoch,'order')).permutation(len(roles['train']))
        for i,index in enumerate(order):
            patient=roles['train'][int(index)];n=int(lookup[patient]['primary_count']);y=int(lookup[patient]['ISUP_grade'])-1
            indices=patch_indices(n,cfg['train_patches'],seed,args.fold,epoch,patient)
            mask=branch_mask(k,spec['branch_drop'],seed,args.fold,epoch,patient) if dropping else None
            logits=model(store.tensors(patient,indices),mask);loss=F.cross_entropy(logits,torch.tensor([y],device=device),reduction='sum')*weights[y]
            assert bool(torch.isfinite(loss)),'Non-finite training loss'
            group_start=(i//cfg['accumulate_patients'])*cfg['accumulate_patients'];group_size=min(cfg['accumulate_patients'],len(order)-group_start)
            (loss/group_size).backward();loss_sum+=float(loss.detach())
            if (i+1)%cfg['accumulate_patients']==0 or i+1==len(order):
                nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,eval_limit)
        key=val['cross_entropy'];improved=key<best_key-1e-8
        if improved:best_key=key;best_epoch=epoch;best_model=cpu_state(model);bad=0
        else:bad+=1
        finished_training=epoch+1>=cfg['max_epochs'] or (epoch+1>=cfg['min_epochs'] and bad>=cfg['patience'])
        history.append(dict(epoch=epoch,training_loss=loss_sum/len(order),validation=val,seconds=time.monotonic()-epoch_start))
        state=dict(fingerprint=fid,model=cpu_state(model),optimizer=optimizer.state_dict(),next_epoch=epoch+1,history=history,best_key=best_key,best_model=best_model,best_epoch=best_epoch,bad=bad,rng=rng_state(device),finished_training=finished_training,platform=plat)
        save_pt(out/'last.pt',state);save_json(out/'history.json',history)
        gpu_peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        resources[str(epoch)]=dict(seconds=history[-1]['seconds'],gpu_max_allocated_bytes=gpu_peak,gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,max_rss_bytes=max_rss_bytes(),finished=now(),**ident)
        save_json(out/'resources.json',resources)  # operational log only (incl. which physical card ran the epoch); not an artifact and not hashed
        print('EPOCH',args.arm,args.seed_index,args.fold,epoch,'train_loss',round(history[-1]['training_loss'],5),'val_ce',round(val['cross_entropy'],5),'val_f1',round(val['macro_f1'],5),'seconds',round(history[-1]['seconds'],2),'device',device,'gpu',ident.get('gpu_uuid'),'gpu_peak',gpu_peak,flush=True)
        if not finished_training and (STOP['requested'] or time.monotonic()-t0>args.max_runtime_seconds or (args.stop_after_epochs and epoch+1>=args.stop_after_epochs)):
            return pause(out,epoch+1,fid,'stop requested' if STOP['requested'] else 'max runtime or --stop-after-epochs')
    assert best_model is not None
    save_json(out/'history.json',history)  # from the checkpoint state: a crash between last.pt and history.json cannot leave it stale
    final_hash=param_hash(model)
    model.load_state_dict(best_model);save_pt(out/'best.pt',dict(model=best_model,epoch=best_epoch,metadata=metadata))
    val_records,val=evaluate(model,store,roles['val'],cfg,'val',args.fold,eval_limit);save_json(out/'val_predictions.json',val_records)
    test=None;names=['best.pt','last.pt','history.json','metadata.json','val_predictions.json']
    if not args.smoke:
        store.cache.clear();test_records,test=evaluate(model,store,roles['test'],cfg,'test',args.fold,None);save_json(out/'test_predictions.json',test_records);names.append('test_predictions.json')
    cards=sorted({str(r.get('gpu_uuid')) for r in resources.values() if isinstance(r,dict) and r.get('gpu_uuid')})
    receipt=dict(state='COMPLETE',smoke=args.smoke,fingerprint=fid,arm=args.arm,seed_index=args.seed_index,fold=args.fold,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,parameters_changed_this_invocation=init_hash!=final_hash,seconds_this_invocation=time.monotonic()-t0,
                 physical_gpus=cards,cross_card_resume=len(cards)>1,artifacts={n:sha(out/n) for n in names})
    save_json(out/'complete.json',receipt);(out/'paused.json').unlink(missing_ok=True)
    print('COMPLETE',args.arm,args.seed_index,args.fold,len(history),flush=True);return 0

def parser():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed-index',type=int,required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-epochs',type=int,default=2);p.add_argument('--outroot');p.add_argument('--stop-after-epochs',type=int)
    p.add_argument('--smoke-shape',choices=['reduced','formal'],default='reduced',help='--smoke only: reduced (8+8 patients, 128 patches) or formal (fold train/val sets and formal patch/chunk sizes)')
    p.add_argument('--device',default='cuda',help="torch device, e.g. cuda, cuda:1, cpu (formal runs must match config['device'])")
    p.add_argument('--threads',type=int,default=int(os.environ.get('SLURM_CPUS_PER_TASK','4')));p.add_argument('--max-runtime-seconds',type=float,default=22*3600)
    p.add_argument('--ram-margin-gib',type=float,default=128.0,help='refuse to preload (exit 76) unless MemAvailable exceeds the cache by this much (shared box)')
    p.add_argument('--oom-score-adj',type=int,default=500,help="raise /proc/self/oom_score_adj to this value (Linux; raising needs no privilege), so that on a host OOM this resumable job is killed before other users' processes")
    return p
def main(argv=None):
    install_signal_handlers();STOP['armed']=True
    args=parser().parse_args(argv)
    if STOP['requested']:print('STOPPED before start',STOP['signals'],flush=True);return 75
    try:return run(args)
    except StopNow as e:print('STOPPED_NOW signal',e.args[0],'(resumable from the last epoch checkpoint)',flush=True);return 75
    except Exception as e:
        if not is_transient(e):raise
        traceback.print_exc();print('TRANSIENT_GPU_FAILURE',repr(e)[:500],'(exit 74; resumable, retry later)',flush=True);return 74
if __name__=='__main__':raise SystemExit(main())
