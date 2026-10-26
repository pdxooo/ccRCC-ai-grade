"""Import-layout test against a fake mirror holding the UNCHANGED full_patch_qc.py and the ORIGINAL czi_pilot_v2.py
(make_world.py <w> real): the patched czi_pilot_v2 from this component must win in the main process and in spawn
workers, full_patch_qc must come from ROOT/bib_phase1 with its pinned sha256, and the check must fail loudly when
the shim is bypassed; the decode-pool initializer must accept the verified layout and break the pool otherwise. Usage: HE_PATH_PREFIX=<w>/hpcroot test_import_layout.py [--bypass-shim]"""
import os,sys,json,multiprocessing
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent
if __name__=='__main__':
    sys.path.insert(0,str(COMP));sys.path.insert(0,str(TESTS))
    import stage0_extract as s0
    if '--bypass-shim' in sys.argv:
        sys.path[:]=[p for p in sys.path if Path(p).resolve()!=COMP]  # simulate a layout without the shim
        try:s0.check_import_layout()
        except AssertionError as e:print('BYPASS_DETECTED',str(e)[:160]);sys.exit(0)
        print('BYPASS_NOT_DETECTED');sys.exit(1)
    layout=s0.check_import_layout()
    import czi_pilot_v2,full_patch_qc,layout_probe
    assert full_patch_qc.IndexedReader.__mro__[1] is czi_pilot_v2.Reader
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:child=pool.submit(layout_probe.probe,0).result()
    assert child['czi_pilot_v2']==str((COMP/'czi_pilot_v2.py').resolve()),child
    assert child['full_patch_qc']==str((s0.ROOT/'bib_phase1/full_patch_qc.py').resolve()),child
    assert child['czi_ROOT']==str(s0.ROOT),child
    # the pool initializer re-checks path + sha256 in every worker: accepted with the verified layout, refused otherwise
    pool=s0.decode_pool(2,layout)
    try:assert pool.submit(layout_probe.probe,0).result(timeout=120)['czi_pilot_v2']==child['czi_pilot_v2']
    finally:s0.close_pool(pool)
    from concurrent.futures.process import BrokenProcessPool
    bad={k:dict(v) for k,v in layout.items()};bad['full_patch_qc.py']['sha256']='0'*64
    pool=s0.decode_pool(1,bad)
    try:pool.submit(layout_probe.probe,0).result(timeout=120);print('WORKER_CHECK_NOT_ENFORCED');sys.exit(1)
    except BrokenProcessPool:pass
    finally:s0.close_pool(pool,kill=True)
    print('WORKER_CHECK_OK')
    print('LAYOUT_OK',json.dumps(dict(main=layout,worker=child),ensure_ascii=False))
