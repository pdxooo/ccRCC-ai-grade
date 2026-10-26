"""Static check: no HPC absolute path literal (a string starting with /work and a slash) in the Python files the A6000
pilot uses, unless the literal is wrapped by the path prefix, i.e. it is
  * the right operand of `<expr mentioning HE_PATH_PREFIX> + <literal>` or `PFX + <literal>`, or
  * the argument of hpc(<literal>) / s0.hpc(<literal>).
Also flags data-path lookups whose key ends in '_path' (row['primary_path'], d.get('weight_path')) unless the lookup is
the direct argument of hpc() / s0.hpc(), in ANY context (call argument, assignment, comparison, ...): a value taken
out of a manifest/JSON row first and prefixed later must be restructured or explicitly waived with a trailing
`# prefixed:` comment that says where the prefix comes from. Not covered: keys not ending in '_path' (e.g. 'weight',
'index_file'), and HPC paths that reach the code without a '_path' key. Comments that mention HPC paths are listed
(informational only).

Usage: python verify_no_work_paths.py [extra.py ...]
Default files: stage0_extract.py, stage1_extract.py, czi_pilot_v2.py, validate.py, verify_no_work_paths.py in this
directory, plus ROOT/bib_phase1/full_patch_qc.py from the mirror ($HE_PATH_PREFIX; imported unchanged) when it exists.
Exit 0 if clean, 1 if any violation or unwrapped data path."""
import ast,io,os,re,sys,tokenize
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=re.compile(r'(^|[^\w.-])/work(/|$)')

def default_files():
    files=[HERE/n for n in ('stage0_extract.py','stage1_extract.py','czi_pilot_v2.py','validate.py','verify_no_work_paths.py')]
    mirror=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/projects/korean_ccrcc_SBIAD2452/bib_phase1/full_patch_qc.py')
    if os.environ.get('HE_PATH_PREFIX','').rstrip('/') and mirror.exists():files.append(mirror)
    return files

def is_prefix_expr(node,src):
    if isinstance(node,ast.Name) and node.id=='PFX':return True
    return 'HE_PATH_PREFIX' in (ast.get_source_segment(src,node) or '')

def is_hpc_call(node):
    f=node.func if isinstance(node,ast.Call) else None
    return (isinstance(f,ast.Name) and f.id=='hpc') or (isinstance(f,ast.Attribute) and f.attr=='hpc')

def check(path):
    src=Path(path).read_text();tree=ast.parse(src);parent={}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):parent[child]=node
    violations=[];warnings=[];notes=[]
    lines=src.splitlines()
    def waived(lineno):return '# prefixed:' in lines[lineno-1]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and WORK.search(node.value):
            lit,p=node,parent.get(node)
            if isinstance(p,ast.JoinedStr):lit,p=p,parent.get(p)
            ok=(isinstance(p,ast.BinOp) and isinstance(p.op,ast.Add) and p.right is lit and is_prefix_expr(p.left,src)) or \
               (isinstance(p,ast.Call) and is_hpc_call(p) and lit in p.args)
            if not ok:violations.append((node.lineno,repr(node.value)[:100]))
        key=None
        if isinstance(node,ast.Subscript):  # row['primary_path']
            key=node.slice.value if isinstance(node.slice,ast.Constant) else getattr(getattr(node.slice,'value',None),'value',None)
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get' and node.args and isinstance(node.args[0],ast.Constant):
            key=node.args[0].value  # d.get('weight_path', ...)
        if isinstance(key,str) and key.endswith('_path'):
            p=parent.get(node)
            if not (isinstance(p,ast.Call) and is_hpc_call(p) and node in p.args) and not waived(node.lineno):
                warnings.append((node.lineno,ast.get_source_segment(src,node)))
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type==tokenize.COMMENT and WORK.search(tok.string):notes.append((tok.start[0],tok.string.strip()[:100]))
    return violations,warnings,notes

def main(argv):
    files=[Path(a) for a in argv] or default_files();bad=0
    for f in files:
        v,w,n=check(f);bad+=len(v)+len(w)
        print(f'{"FAIL" if v or w else "OK  "} {f}  unwrapped_work_literals={len(v)} unwrapped_data_paths={len(w)} comments={len(n)}')
        for line,text in v:print(f'    VIOLATION line {line}: {text}')
        for line,text in w:print(f'    WARN      line {line}: data path not passed through hpc(): {text}')
        for line,text in n:print(f'    note      line {line}: comment {text}')
    print('RESULT',('FAIL' if bad else 'PASS'),f'files={len(files)} violations={bad}')
    return 1 if bad else 0

if __name__=='__main__':sys.exit(main(sys.argv[1:]))
