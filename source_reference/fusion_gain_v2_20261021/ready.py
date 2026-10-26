"""Data readiness of fusion_gain_v2 arms.
Mirror mode (HE_PATH_PREFIX or HE_TRANSFER_DIR set; lab-a6000): an arm is ready only if <transfer>/verified_feat_<enc>
exists for every member encoder AND, for all 300 manifest patients, <FEATURES>/<stem>/<enc>.npy and its sibling
<enc>.npy.verified marker exist. <transfer> = $HE_TRANSFER_DIR, else <parent of HE_PATH_PREFIX>/transfer.
Native mode (neither set; the HPC itself): ready if the 300 feature files of every member encoder exist.
CLI: python ready.py <arm>|all [<arm> ...] [--quiet]
     one JSON line per arm on stdout; exit 0 only if every named arm is ready, 1 otherwise, 2 on a bad arm name."""
from common import *
def transfer_dir():
    d=os.environ.get('HE_TRANSFER_DIR','')
    if d:return Path(d)
    return Path(PREFIX).parent/'transfer' if PREFIX else None
def arm_status(arm,cfg=None):
    cfg=cfg or config();members=cfg['arms'][arm]['members'];T=transfer_dir()
    out=dict(arm=arm,members=members,mode='mirror' if T else 'native',transfer_dir=str(T) if T else None,root=str(ROOT),encoders={},missing=[])
    manifest=POOL/'patient_manifest.csv'
    if not manifest.exists():
        out.update(ready=False);out['missing'].append(str(manifest));return out
    stems=sorted(Path(r['slide_filename']).stem for r in read_csv(manifest))
    for enc in members:
        files=[FEATURES/s/(enc+'.npy') for s in stems]
        present=sum(f.exists() for f in files);rec=dict(patients=len(stems),files=present)
        missing=[str(f) for f in files if not f.exists()]
        if T:
            batch=T/f'verified_feat_{enc}';rec['batch_marker']=batch.exists()
            if not rec['batch_marker']:out['missing'].append(str(batch))
            marks=[f.with_name(f.name+'.verified') for f in files];rec['verified']=sum(m.exists() for m in marks)
            missing+=[str(m) for m in marks if not m.exists()]
        out['encoders'][enc]=rec;out['missing']+=missing[:5]
        if len(missing)>5:out['missing'].append(f'... {len(missing)-5} more for {enc}')
    out['ready']=len(stems)==300 and not out['missing']
    return out
def is_ready(arm,cfg=None):return arm_status(arm,cfg)['ready']
def main(argv):
    quiet='--quiet' in argv;names=[a for a in argv if a!='--quiet'];cfg=config()
    if not names:print(__doc__);return 2
    if names==['all']:names=list(cfg['arms'])
    bad=[a for a in names if a not in cfg['arms']]
    if bad:print('UNKNOWN_ARM',bad,'choices',list(cfg['arms']),file=sys.stderr);return 2
    ok=True
    for arm in names:
        s=arm_status(arm,cfg);ok&=s['ready']
        if not quiet:print(json.dumps(s if not s['ready'] else {k:s[k] for k in ('arm','ready','mode','encoders')}),flush=True)
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
