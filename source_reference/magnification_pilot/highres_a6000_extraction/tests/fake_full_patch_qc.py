"""TEST FAKE of full_patch_qc.py, installed as ROOT/bib_phase1/full_patch_qc.py in the fake mirror: IndexedReader returns
synthetic, position-consistent RGB instead of decoding a CZI. It imports czi_pilot_v2 exactly like the real module and
logs which czi_pilot_v2 file was imported (in every decode worker) to $FAKE_READER_LOG; $FAKE_DECODE_SLEEP delays every region()."""
import os,json,hashlib,time
from czi_pilot_v2 import ROOT,np
import czi_pilot_v2

class IndexedReader:
    def __init__(self,path):
        self.path=str(path)
        with open(self.path,'rb') as f:f.read(1)  # the slide path built by the caller must exist
        self.seed=int(hashlib.sha256(os.path.basename(self.path).encode()).hexdigest()[:8],16)
        log=os.environ.get('FAKE_READER_LOG')
        if log:
            with open(log,'a') as f:
                f.write(json.dumps(dict(pid=os.getpid(),czi_pilot_v2=os.path.realpath(czi_pilot_v2.__file__),ROOT=str(ROOT),slide=self.path))+'\n')
    def region(self,scene,x,y,w,h):
        if os.environ.get('FAKE_DECODE_SLEEP'):time.sleep(float(os.environ['FAKE_DECODE_SLEEP']))  # simulate a hung CZI read
        yy,xx=np.mgrid[y:y+h,x:x+w].astype(np.int64);s=self.seed
        base=((xx*73856093)^(yy*19349663)^(s+scene*83492791))&0xffff
        r=(120+60*np.sin(xx/7.0+s%13)+(base%40)).clip(0,255)
        g=(100+50*np.cos(yy/5.0+s%7)+((base>>4)%40)).clip(0,255)
        b=(140+40*np.sin((xx+yy)/11.0)+((base>>8)%40)).clip(0,255)
        return np.stack([r,g,b],-1).astype(np.uint8),np.ones((h,w),bool),[]
    def close(self):pass
