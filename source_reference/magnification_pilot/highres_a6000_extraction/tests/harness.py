"""Run the A6000 stage0/stage1 entry points on CPU inside a fake world (see make_world.py).
Test-only monkeypatches (main process only; decode workers use the fake ROOT/bib_phase1/full_patch_qc.py via the
real import layout and are checked by the real pool initializer): require_gpu -> no-op, load_pathology ->
tests/fake_validate.py, LOCATIONS -> $FAKE_LOCATIONS (24), BATCH -> $FAKE_BATCH (8), PINNED['full_patch_qc.py'] -> sha256
of the fake reader, stage1 LIST_SHA256/N_CASES -> the fake lock's expected values. Optional fault injection:
  $FAKE_GPU_FREE_GIB    busy shared GPU (stage 0 and stage 1)
  $FAKE_OOM_AT_LOAD     torch.cuda.OutOfMemoryError while the models load (stage 0 and stage 1)
  $FAKE_OOM             stage-1 case processing raises OutOfMemoryError ('1') or a cuBLAS alloc RuntimeError ('cublas')
  $FAKE_CRASH           stage-0 extraction raises a plain ValueError (a crash, not a FAIL)
  $FAKE_DECODE_TIMEOUT  stage0_extract.DECODE_TIMEOUT in seconds (with $FAKE_DECODE_SLEEP in the fake reader: hung decode)
Usage: HE_PATH_PREFIX=<world>/hpcroot HE_PROJECT_DIR=<world>/project harness.py stage0|stage1 <world> [args...]"""
import os,sys
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent

class OutOfMemoryError(RuntimeError):pass  # same class name as torch.cuda.OutOfMemoryError

if __name__=='__main__':
    mode,world,argv=sys.argv[1],Path(sys.argv[2]),sys.argv[3:]
    sys.path.insert(0,str(COMP));sys.path.insert(0,str(TESTS))
    import stage0_extract as s0
    import fake_validate
    s0.require_gpu=lambda device:None
    s0.load_pathology=lambda:fake_validate
    s0.LOCATIONS=int(os.environ.get('FAKE_LOCATIONS','24'));s0.BATCH=int(os.environ.get('FAKE_BATCH','8'))
    s0.PINNED['full_patch_qc.py']=s0.sha(s0.ROOT/'bib_phase1/full_patch_qc.py')
    if os.environ.get('FAKE_GPU_FREE_GIB'):s0.gpu_free_gib=lambda device:float(os.environ['FAKE_GPU_FREE_GIB'])
    if os.environ.get('FAKE_OOM_AT_LOAD'):
        def load_models(*a,**k):raise OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB (simulated, at model load)')
        s0.load_models=load_models
    if os.environ.get('FAKE_CRASH'):
        def extract_locations(*a,**k):raise ValueError('simulated crash during extraction')
        s0.extract_locations=extract_locations
    if os.environ.get('FAKE_DECODE_TIMEOUT'):s0.DECODE_TIMEOUT=float(os.environ['FAKE_DECODE_TIMEOUT'])
    if mode=='stage0':sys.exit(s0.main(argv))
    import stage1_extract as s1
    s1.LIST_SHA256=(world/'expected_list_sha256.txt').read_text().strip();s1.N_CASES=2
    s1.QUARANTINE['list_sha256']=s1.LIST_SHA256
    if os.environ.get('FAKE_GPU_FREE_GIB'):s1.gpu_free_gib=s0.gpu_free_gib  # simulate a busy shared GPU
    if os.environ.get('FAKE_OOM'):
        def process_case(*a,**k):
            if os.environ['FAKE_OOM']=='cublas':raise RuntimeError('CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)` (simulated)')
            raise OutOfMemoryError('CUDA out of memory (simulated)')
        s1.process_case=process_case
    sys.exit(s1.main(argv))
