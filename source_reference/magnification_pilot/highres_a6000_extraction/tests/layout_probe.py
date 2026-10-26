"""Executed inside a spawned worker: report which helper modules the worker actually imports."""
import os,sys
def probe(_):
    import full_patch_qc,czi_pilot_v2
    return dict(czi_pilot_v2=os.path.realpath(czi_pilot_v2.__file__),full_patch_qc=os.path.realpath(full_patch_qc.__file__),
                czi_ROOT=str(czi_pilot_v2.ROOT),reader_base=full_patch_qc.IndexedReader.__mro__[1].__module__,sys_path_head=sys.path[:4])
