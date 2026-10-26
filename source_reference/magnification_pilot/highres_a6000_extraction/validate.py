"""Six frozen, local encoders on identical real HE crops, then fusion training smoke.

Technical validation only. Synthetic slide targets verify gradients; no clinical
performance is estimated. Each encoder retains its own official preprocessing.
Architectures: official model cards and Mahmoodlab/TRIDENT local-weight adapters.
"""
import argparse, gc, hashlib, importlib.metadata, json, os, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import timm
from torchvision import transforms as T

ROOT=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/projects/gastritis_uni2h')  # A6000: HE_PATH_PREFIX mirror root; '' => HPC original
RUN=ROOT/'outputs/encoder-validation-20261008'
ASSETS=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/software/uni2h/models')
BIN=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/bin')
NAMES=['uni','uni2-h','phikon-v2','conch','virchow2','gigapath']
EXPECTED={'uni':1024,'uni2-h':1536,'phikon-v2':1024,'conch':512,'virchow2':2560,'gigapath':1536}
MEAN=(.485,.456,.406); STD=(.229,.224,.225)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def save(name,obj):
    p=RUN/name; temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); os.replace(temp,p)

def standard(size=224,bicubic=False,crop=True):
    steps=[T.Resize(size,interpolation=T.InterpolationMode.BICUBIC if bicubic else T.InterpolationMode.BILINEAR)]
    if crop: steps.append(T.CenterCrop(224))
    return T.Compose(steps+[T.ToTensor(),T.Normalize(MEAN,STD)])

def build(name):
    path=ASSETS/name/'pytorch_model.bin'
    info={'strict_load':True,'feature_selection':'pooled output'}
    if name=='phikon-v2':
        from transformers import AutoModel, AutoImageProcessor
        proc=AutoImageProcessor.from_pretrained(ASSETS/name,local_files_only=True,use_fast=False)
        model,li=AutoModel.from_pretrained(ASSETS/name,local_files_only=True,output_loading_info=True)
        assert not any(li.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),li
        info.update(weight=str(ASSETS/name/'model.safetensors'),loading_info=li,preprocessing=proc.to_dict(),feature_selection='last_hidden_state[:,0,:]')
        return model,lambda imgs:proc(images=imgs,return_tensors='pt')['pixel_values'],info
    if name=='conch':
        from conch.open_clip_custom import create_model_from_pretrained
        from conch.open_clip_custom.factory import read_state_dict
        from conch.open_clip_custom.coca_model import resize_pos_embed
        path=BIN/'pytorch_model(2).bin'
        model,trans=create_model_from_pretrained('conch_ViT-B-16',checkpoint_path=str(path))
        sd=read_state_dict(str(path)); resize_pos_embed(sd,model)
        # Official release deliberately omits the multimodal caption decoder.
        # Remove only that unused branch, then strictly validate every remaining
        # tensor (including the entire vision and text encoders).
        assert not any(k.startswith('text_decoder.') for k in sd)
        model.text_decoder=nn.Identity()
        model.load_state_dict(sd,strict=True); del sd
        info['unused_caption_decoder']='omitted by official release; removed before strict encoder load'
        info['feature_selection']='encode_image(proj_contrast=False, normalize=False)'
    else:
        common=dict(pretrained=False,num_classes=0,dynamic_img_size=True,img_size=224)
        if name=='uni':
            model=timm.create_model('vit_large_patch16_224',init_values=1e-5,**common); trans=standard()
        elif name=='uni2-h':
            model=timm.create_model('vit_giant_patch14_224',patch_size=14,depth=24,num_heads=24,init_values=1e-5,
                embed_dim=1536,mlp_ratio=2.66667*2,no_embed_class=True,mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=nn.SiLU,reg_tokens=8,**common); trans=standard()
        elif name=='virchow2':
            path=BIN/'pytorch_model(3).bin'
            model=timm.create_model('vit_huge_patch14_224',init_values=1e-5,reg_tokens=4,mlp_ratio=5.3375,
                global_pool='',mlp_layer=timm.layers.SwiGLUPacked,act_layer=nn.SiLU,**common)
            trans=standard(bicubic=True,crop=False); info['feature_selection']='concat CLS + mean tokens[:,5:] (registers excluded)'
        elif name=='gigapath':
            path=BIN/'pytorch_model(4).bin'
            model=timm.create_model('vit_giant_patch14_dinov2',in_chans=3,patch_size=16,embed_dim=1536,
                depth=40,num_heads=24,mlp_ratio=5.33334,**common); trans=standard(256,True)
        else: raise KeyError(name)
        sd=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(sd,strict=True); del sd
    info.update(weight=str(path),preprocessing=repr(trans))
    return model,lambda imgs:torch.stack([trans(im) for im in imgs]),info

def forward(name,m,x):
    if name=='conch': return m.encode_image(x,proj_contrast=False,normalize=False)
    if name=='phikon-v2': return m(pixel_values=x).last_hidden_state[:,0,:]
    y=m(x)
    return torch.cat([y[:,0],y[:,5:].mean(1)],dim=-1) if name=='virchow2' else y

class Fusion(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.proj=nn.ModuleDict({n:nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU()) for n,d in dims.items()})
        self.gate=nn.Linear(64,1)
        self.attention=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
        self.heads=nn.ModuleList([nn.Linear(64,k) for k in [3,3,3,4]])
    def forward(self,features):
        z=torch.stack([self.proj[n](features[n]) for n in self.proj],dim=1)
        h=(z*self.gate(z).softmax(1)).sum(1)
        bags=h.reshape(4,4,64)
        pooled=(bags*self.attention(bags).softmax(1)).sum(1)
        return [head(pooled) for head in self.heads]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    previous=json.loads((RUN/'report.json').read_text()) if args.resume else {}
    if previous: save('report-before-resume-'+str(int(time.time()))+'.json',previous)
    torch.set_num_threads(2); torch.manual_seed(20261008)
    assert torch.cuda.is_available(),'GPU required, CPU fallback disabled'
    report={'state':'running','job_id':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
            'models':{},'fusion':None,'clinical_validation':False,'physical_scale_verified':False,
            'source_crop_pixels':224,'code_sha256':sha(__file__),
            'dependencies':{n:importlib.metadata.version(n) for n in ['torch','timm','conch','transformers']}}
    save('report.json',report)
    man=ROOT/'outputs/full-features790-20261003/input_manifest.json'
    source=json.loads(man.read_text()); report['manifest_sha256']=sha(man)
    rows=sorted([r for r in source['slides'] if r['cohort']=='antrum' and r['n']>=4],key=lambda r:r['source_bytes'])[:4]
    assert len(rows)==4
    images=[]; ids=[]; Image.MAX_IMAGE_PIXELS=750000000
    for r in rows:
        assert sha(r['index_file'])==r['index_file_sha256']
        with np.load(r['index_file'],allow_pickle=False) as ar:
            coords=ar['coords'][np.linspace(0,r['n']-1,4,dtype=int)]
        with Image.open(r['source']) as raw:
            assert raw.mode=='RGB'
            for x,y in coords:
                im=raw.crop((int(x),int(y),int(x)+224,int(y)+224)); images.append(im)
                ids.append({'slide':r['cohort']+'/'+r['slide_id'],'xy':[int(x),int(y)],'rgb_sha256':hashlib.sha256(im.tobytes()).hexdigest()})
    assert len(images)==16
    order_hash=hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest()
    save('patch_manifest.json',{'patches':ids,'order_sha256':order_hash,'source_crop_pixels':224})
    features={}
    for name in NAMES:
        prior=previous.get('models',{}).get(name,{})
        if prior.get('state')=='passed' and prior.get('order_sha256')==order_hash:
            with np.load(RUN/(name+'-features.npz'),allow_pickle=False) as ar:
                assert ar['patch_ids'].tolist()==[json.dumps(i,sort_keys=True) for i in ids]
                assert str(ar['order_sha256'])==order_hash
                arr=ar['features'].copy()
            assert arr.shape==(16,EXPECTED[name]) and np.isfinite(arr).all()
            features[name]=arr
            info=dict(prior); info['validated_by_job']=previous['job_id']; info['validation_code_sha256']=previous['code_sha256']
            info['feature_file_sha256']=sha(RUN/(name+'-features.npz'))
            report['models'][name]=info; save('report.json',report)
            print('REUSE_VERIFIED',name,arr.shape,flush=True); continue
        model=None; x=None; a=None; b=None
        started=time.monotonic(); print('START',name,flush=True)
        try:
            model,prep,info=build(name)
            info['weight_sha256']=sha(info['weight'])
            info['parameters']=sum(p.numel() for p in model.parameters())
            model=model.eval().requires_grad_(False).cuda()
            torch.cuda.reset_peak_memory_stats()
            x=prep(images).cuda(); info['input_tensor_shape']=list(x.shape)
            amp=torch.bfloat16 if name in ['uni2-h','phikon-v2'] else torch.float16
            # Official CONCH adapter uses float32.
            use_amp=name!='conch'
            with torch.inference_mode(),torch.autocast('cuda',enabled=use_amp,dtype=amp):
                a=torch.cat([forward(name,model,x[i:i+4]).float() for i in range(0,len(x),4)])
                b=forward(name,model,x[:4]).float()
            assert tuple(a.shape)==(16,EXPECTED[name]),a.shape
            assert torch.isfinite(a).all() and (a.norm(dim=1)>0).all()
            repeat_max=float((a[:4]-b).abs().max())
            assert torch.allclose(a[:4],b,rtol=1e-4,atol=1e-4),repeat_max
            assert a.std(dim=0).max()>1e-6,'Collapsed features'
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
            arr=a.cpu().numpy().copy(); features[name]=arr
            np.savez(RUN/(name+'-features.npz'),features=arr,patch_ids=np.array([json.dumps(i,sort_keys=True) for i in ids]),order_sha256=np.array(order_hash))
            info['feature_file_sha256']=sha(RUN/(name+'-features.npz'))
            info.update(state='passed',shape=list(arr.shape),storage_dtype=str(arr.dtype),finite=True,noncollapsed=True,
                repeat_max_abs_difference=repeat_max,models_frozen=True,order_sha256=order_hash,
                elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                precision=str(amp) if use_amp else 'torch.float32')
            report['models'][name]=info; print('PASS',name,arr.shape,flush=True)
        except Exception:
            report['models'][name]={'state':'failed','traceback':traceback.format_exc()}; print(traceback.format_exc(),flush=True)
        finally:
            del model,x,a,b
            gc.collect(); torch.cuda.empty_cache(); save('report.json',report)
    if len(features)==6:
        # Re-read files: verify persisted row identity before fusion, not just in-memory order.
        fs={}
        for name in NAMES:
            with np.load(RUN/(name+'-features.npz'),allow_pickle=False) as ar:
                assert str(ar['order_sha256'])==order_hash
                assert ar['patch_ids'].tolist()==[json.dumps(i,sort_keys=True) for i in ids]
                fs[name]=torch.tensor(ar['features'],device='cuda')
        fusion=Fusion({n:a.shape[1] for n,a in features.items()}).cuda().train()
        opt=torch.optim.AdamW(fusion.parameters(),lr=1e-3)
        before={n:p.detach().clone() for n,p in fusion.named_parameters()}
        logits=fusion(fs)
        # Synthetic targets solely test multi-task tensor/loss/gradient plumbing.
        targets=[torch.arange(4,device='cuda')%k for k in [3,3,3,4]]
        loss=sum(nn.functional.cross_entropy(z,y) for z,y in zip(logits,targets))
        assert torch.isfinite(loss); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
        grad={n:float(fusion.proj[n][1].weight.grad.norm()) for n in NAMES}
        assert all(v>0 for v in grad.values())
        opt.step()
        changes={n:float((p-before[n]).norm()) for n,p in fusion.named_parameters()}
        assert all(changes[f'proj.{n}.1.weight']>0 for n in NAMES)
        assert sum(changes.values())>0
        torch.save(fusion.state_dict(),RUN/'fusion-smoke.pt')
        report['fusion']={'state':'passed','loss':float(loss.detach()),'encoder_projection_gradient_norms':grad,
             'parameter_update_norm_sum':sum(changes.values()),'feature_dims':{n:int(a.shape[1]) for n,a in features.items()},
             'bags':4,'patches_per_bag':4,'target_kind':'synthetic technical test only',
             'trainable_parameters':sum(p.numel() for p in fusion.parameters()),'persisted_patch_order_verified':True,
             'frozen_encoders':True,'clinical_performance_estimated':False}
    report['state']='passed' if len(features)==6 and report.get('fusion',{}).get('state')=='passed' else 'failed'
    save('report.json',report); print('FINAL',report['state'],flush=True)
    if report['state']!='passed': raise RuntimeError('Not all six encoders passed; see report.json')

if __name__=='__main__': main()
