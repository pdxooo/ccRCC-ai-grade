"""Prefix byte-identity: with HE_PATH_PREFIX unset or '' every path the patched modules build equals the HPC original
exactly; with a prefix it equals prefix + original. Also checks that the patched czi_pilot_v2.py / validate.py differ
from the originals only by the prefix expression (+ comment). Runs each case in a fresh interpreter with import stubs
for timm / torchvision / czifile (not installed locally)."""
import os,sys,json,subprocess,tempfile
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent;ORIG=COMP/'originals'
EXPR="os.environ.get('HE_PATH_PREFIX','').rstrip('/')+";COMMENT="  # A6000: HE_PATH_PREFIX mirror root; '' => HPC original"
PROBE=r'''
import sys,json,importlib.util
from pathlib import Path
def load(name,path):
    before=len(sys.path);spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,sys.path[:len(sys.path)-before]
O,N=Path(sys.argv[1]),Path(sys.argv[2]);r={}
for key,f in [('czi','czi_pilot_v2.py'),('validate','validate.py'),('stage0','stage0_extract.py')]:
    for side,d in [('orig',O),('new',N)]:
        m,added=load(f'{side}_{key}',d/f)
        attrs={a:str(getattr(m,a)) for a in ['ROOT','RUN','ASSETS','BIN','POOL','STORED','OUT','OLD'] if hasattr(m,a)}
        if key=='stage0':
            attrs['slide']=str(m.ROOT/'data/he'/'HE_KR1.czi')
            attrs['primary_path']=m.hpc('/data/cluster/projects/x.npz') if hasattr(m,'hpc') else '/data/cluster/projects/x.npz'
        r[f'{side}_{key}']=dict(attrs=attrs,sys_path_added=added)
print(json.dumps(r))
'''
def stubs(d):
    d=Path(d);(d/'timm').mkdir();(d/'timm/__init__.py').write_text('class _L:\n    SwiGLUPacked=None\nlayers=_L()\n')
    (d/'torchvision').mkdir();(d/'torchvision/__init__.py').write_text('');(d/'torchvision/transforms.py').write_text('')
    (d/'czifile.py').write_text('')
def run_case(prefix,stubdir):
    env=dict(os.environ);env.pop('HE_PATH_PREFIX',None);env['PYTHONPATH']=stubdir
    if prefix is not None:env['HE_PATH_PREFIX']=prefix
    out=subprocess.run([sys.executable,'-c',PROBE,str(ORIG),str(COMP)],env=env,capture_output=True,text=True)
    assert out.returncode==0,out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])
def main():
    fails=[]
    with tempfile.TemporaryDirectory() as t:
        stubs(t)
        for prefix in [None,'','/x/hpcroot','/x/hpcroot/']:
            r=run_case(prefix,t);p=(prefix or '').rstrip('/');n_checked=0
            for key in ('czi','validate','stage0'):
                o,n=r[f'orig_{key}'],r[f'new_{key}']
                for a,v in o['attrs'].items():
                    want=p+v
                    if key=='stage0' and a=='OUT':want=p+v+'_a6000'  # documented deviation: separate A6000 output directory
                    n_checked+=1
                    if n['attrs'][a]!=want:fails.append((prefix,key,a,n['attrs'][a],want))
                added=[x for x in n['sys_path_added'] if Path(x).resolve()!=COMP]  # the shim entry is the documented addition
                if added!=[p+x for x in o['sys_path_added']]:fails.append((prefix,key,'sys.path',added,o['sys_path_added']))
            status='PASS' if not fails else 'FAIL'
            print(f'{status} prefix={prefix!r}: {n_checked} paths + 3 sys.path insert lists compared; stage0 ROOT={r["new_stage0"]["attrs"]["ROOT"]} primary_path={r["new_stage0"]["attrs"]["primary_path"]}')
    for f in ('czi_pilot_v2.py','validate.py'):
        new=(COMP/f).read_text().replace(COMMENT,'').replace(EXPR,'');same=new==(ORIG/f).read_text()
        print(('PASS' if same else 'FAIL'),f'{f}: patched text minus prefix expression == original bytes')
        if not same:fails.append(f)
    for f in fails:print('MISMATCH',f)
    print('RESULT',('FAIL' if fails else 'PASS'));return 1 if fails else 0
if __name__=='__main__':sys.exit(main())
