"""Bounded CZI reader / tissue-candidate pilot. Not final histologic QC."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, math, csv, json, hashlib, time, traceback, collections
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/projects/korean_ccrcc_SBIAD2452')  # A6000: HE_PATH_PREFIX mirror root; '' => HPC original
sys.path.insert(0,str(ROOT/'qc_deps'))
import numpy as np
import czifile
from PIL import Image, ImageDraw

def save_json(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def spatial_rgb(raw,axes):
    assert len(axes)==raw.ndim and all(axes.count(a)==1 for a in 'YX0')
    selected=[axes.index(a) for a in 'YX0'];other=[i for i in range(raw.ndim) if i not in selected]
    assert all(raw.shape[i]==1 for i in other),'Ambiguous nonspatial axes'
    out=np.transpose(raw,selected+other).reshape(tuple(raw.shape[i] for i in selected))
    assert out.dtype==np.uint8 and out.shape[-1]==3
    return out  # legacy czifile.data() already converts BGR to RGB.

class Reader:
    def __init__(self,path):
        self.cf=czifile.CziFile(str(path),multifile=False);self.entries=list(self.cf.subblock_directory)
        self.groups=collections.defaultdict(list);self.cache=collections.OrderedDict()
        for i,e in enumerate(self.entries):
            d={v.dimension:v for v in e.dimension_entries}
            if not all(k in d for k in ['X','Y']):continue
            x,y=d['X'],d['Y'];sx=x.size/x.stored_size;sy=y.size/y.stored_size
            assert abs(sx-sy)<1e-5 and abs(sx-round(sx))<1e-5,'Unsupported noninteger pyramid scaling'
            scene=d['S'].start if 'S' in d else 0
            self.groups[(scene,round(sx))].append((i,x.start,y.start,x.size,y.size))
        self.scenes=sorted(s for s,l in self.groups if l==1)
        assert self.scenes
        md=ET.fromstring(self.cf.metadata())
        self.mpp={e.attrib['Id']:float(e.findtext('Value'))*1e6 for e in md.findall('.//Scaling/Items/Distance') if e.attrib.get('Id') in ['X','Y']}
        assert set(self.mpp)=={'X','Y'} and abs(self.mpp['X']-self.mpp['Y'])<1e-8
    def decode(self,i):
        if i not in self.cache:
            e=self.entries[i];self.cache[i]=spatial_rgb(e.data_segment().data(resize=False),e.axes)
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(i);return self.cache[i]
    def bbox(self,scene):
        a=self.groups[(scene,1)]
        return (min(t[1] for t in a),min(t[2] for t in a),max(t[1]+t[3] for t in a),max(t[2]+t[4] for t in a))
    def overview(self,scene,target=27):
        levels=sorted(l for s,l in self.groups if s==scene)
        level=min(levels,key=lambda l:abs(math.log(l/target)))
        x0,y0,x1,y1=self.bbox(scene)
        # Pyramid tiles may have a non-zero phase relative to native origin.
        # Anchor the overview on its actual pyramid lattice, not coordinate zero.
        tiles=self.groups[(scene,level)]
        anchor_x=min(t[1] for t in tiles);anchor_y=min(t[2] for t in tiles)
        ox=anchor_x+math.floor((x0-anchor_x)/level)*level
        oy=anchor_y+math.floor((y0-anchor_y)/level)*level
        w=math.ceil((x1-ox)/level);h=math.ceil((y1-oy)/level)
        assert w*h<=60_000_000,'Overview memory bound'
        arr=np.full((h,w,3),255,np.uint8);covered=np.zeros((h,w),bool)
        for i,x,y,ew,eh in self.groups[(scene,level)]:
            assert (x-ox)%level==0 and (y-oy)%level==0,'Unaligned pyramid'
            tile=self.decode(i);tx=(x-ox)//level;ty=(y-oy)//level
            xa=max(tx,0);ya=max(ty,0);xb=min(tx+tile.shape[1],w);yb=min(ty+tile.shape[0],h)
            if xb>xa and yb>ya:
                arr[ya:yb,xa:xb]=tile[ya-ty:yb-ty,xa-tx:xb-tx];covered[ya:yb,xa:xb]=True
        return arr,covered,(ox,oy,level)
    def region(self,scene,x,y,w,h):
        assert w*h<=4_000_000
        arr=np.full((h,w,3),255,np.uint8);cover=np.zeros((h,w),bool);used=[]
        for i,tx,ty,tw,th in self.groups[(scene,1)]:
            xa=max(x,tx);ya=max(y,ty);xb=min(x+w,tx+tw);yb=min(y+h,ty+th)
            if xb<=xa or yb<=ya:continue
            tile=self.decode(i)
            assert tile.shape[:2]==(th,tw)
            arr[ya-y:yb-y,xa-x:xb-x]=tile[ya-ty:yb-ty,xa-tx:xb-tx]
            cover[ya-y:yb-y,xa-x:xb-x]=True;used.append(i)
        return arr,cover,used
    def check_reader(self,scene):
        # Independent direct-subblock equality, including the singleton-axis fix.
        a=self.groups[(scene,1)];t=a[len(a)//2];i,x,y,w,h=t
        raw=self.decode(i);cw=min(w,128);ch=min(h,128)
        got,cover,_=self.region(scene,x,y,cw,ch)
        assert cover.all() and np.array_equal(got,raw[:ch,:cw])
        # Exercise a native tile boundary and compare one crop with four subcrops.
        b=self.bbox(scene);xx=min(max(x+w-64,b[0]),b[2]-128);yy=min(max(y+h-64,b[1]),b[3]-128)
        whole,c,_=self.region(scene,xx,yy,128,128)
        for dy in [0,64]:
            for dx in [0,64]:
                quarter,q,_=self.region(scene,xx+dx,yy+dy,64,64)
                assert np.array_equal(whole[dy:dy+64,dx:dx+64],quarter)
                assert np.array_equal(c[dy:dy+64,dx:dx+64],q)
        return {'direct_subblock_equality':True,'boundary_partition_equality':True,'boundary_coverage':float(c.mean())}
    def close(self):self.cf.close()

def pilot(row,out):
    start=time.monotonic();path=ROOT/'data/he'/row['slide_filename'];st=path.stat()
    assert st.st_size==int(row['source_bytes'])
    receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text())
    assert receipt['sha256_local']==row['source_sha256']
    p=out/path.stem;p.mkdir(exist_ok=True)
    result={'slide_filename':path.name,'patient_id':row['patient_id'],'grade':int(row['ISUP_grade']),
            'status':'RUNNING','scope':'CZI reading and preliminary tissue candidates; not final blur/artifact or tumor QC','scenes':[],'patches':[]}
    r=Reader(path)
    try:
        result['mpp_um']=r.mpp;crop=round(128/r.mpp['X']);result['crop_native_pixels']=crop
        result['actual_field_um']=crop*r.mpp['X'];coords_all=[]
        for scene in r.scenes:
            check=r.check_reader(scene);rgb,covered,origin=r.overview(scene);ox,oy,scale=origin
            bright=rgb.mean(2);sat=rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)
            tissue=covered & (bright>20)&(bright<245)&(sat>8)
            # Intentionally permissive pilot only; pale tissue and artifacts need review.
            integral=np.pad(tissue.astype(np.int32),((1,0),(1,0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
            b=r.bbox(scene);grid=[];fractions=[];total=0
            for y in range(b[1],b[3]-crop+1,crop):
                for x in range(b[0],b[2]-crop+1,crop):
                    total+=1;xa=max(0,math.floor((x-ox)/scale));ya=max(0,math.floor((y-oy)/scale))
                    xb=min(rgb.shape[1],math.ceil((x+crop-ox)/scale));yb=min(rgb.shape[0],math.ceil((y+crop-oy)/scale))
                    f=float((integral[yb,xb]-integral[ya,xb]-integral[yb,xa]+integral[ya,xa])/max(1,(yb-ya)*(xb-xa)))
                    if f>=.20:grid.append([scene,x,y,crop,crop]);fractions.append(f)
            assert grid,'No preliminary tissue coordinates'
            coords_all.extend(grid)
            canvas=Image.fromarray(rgb);canvas.thumbnail((1400,1400));canvas.save(p/f'scene{scene}_overview.jpg',quality=90)
            tint=rgb.copy();tint[tissue]=(rgb[tissue].astype(np.float32)*.7+np.array([0,255,0])*.3).astype(np.uint8)
            overlay=Image.fromarray(tint);overlay.thumbnail((1400,1400));overlay.save(p/f'scene{scene}_candidate_mask.jpg',quality=90)
            Image.fromarray(tissue.astype(np.uint8)*255).save(p/f'scene{scene}_candidate_mask.png')
            indexes=np.linspace(0,len(grid)-1,min(16,len(grid)),dtype=int)
            sheet=Image.new('RGB',(4*256,math.ceil(len(indexes)/4)*280),'white');draw=ImageDraw.Draw(sheet)
            for j,index in enumerate(indexes):
                _,x,y,w,h=grid[index];tile,cov,used=r.region(scene,x,y,w,h)
                assert cov.all(),'Native sampled region has unscanned gaps'
                image=Image.fromarray(tile);name=f'scene{scene}_sample{j:02d}.png';image.save(p/name)
                preview=image.resize((256,256),Image.Resampling.LANCZOS)
                sheet.paste(preview,((j%4)*256,(j//4)*280));draw.text(((j%4)*256+2,(j//4)*280+256),f'{x},{y}',fill='black')
                result['patches'].append({'patient_id':row['patient_id'],'slide_filename':path.name,'scene':scene,'x':x,'y':y,'size_native':w,
                      'field_um':w*r.mpp['X'],'rgb_sha256':hashlib.sha256(tile.tobytes()).hexdigest(),'path':str(p/name),'source_subblocks':used})
            sheet.save(p/f'scene{scene}_sample_sheet.jpg',quality=93)
            result['scenes'].append({'scene':scene,'bbox':b,'overview_scale':scale,'origin':origin,'grid_total':total,
                                    'candidate_tiles':len(grid),'reader_tests':check,'tissue_threshold':.20,'mask_kind':'permissive_color_heuristic_pilot'})
        np.savez_compressed(p/'candidate_coords.npz',coords=np.array(coords_all,dtype=np.int64),columns=np.array(['scene','x','y','w','h']),
                            mpp_um=np.array([r.mpp['X'],r.mpp['Y']]),source_sha256=np.array(row['source_sha256']))
        assert path.stat().st_size==st.st_size and path.stat().st_mtime_ns==st.st_mtime_ns
        result.update(status='PASS_TECHNICAL_ONLY',candidate_tiles=len(coords_all),seconds=time.monotonic()-start)
    finally:r.close()
    save_json(p/'report.json',result);return result

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(args.manifest,encoding='utf-8-sig')));results=[]
    for row in rows:
        try:result=pilot(row,out)
        except Exception:result={'slide_filename':row['slide_filename'],'status':'FAIL','traceback':traceback.format_exc()}
        results.append(result)
        save_json(out/'summary.json',{'state':'running','completed':len(results),'requested':len(rows),'results':results})
        print(json.dumps({k:result.get(k) for k in ['slide_filename','status','candidate_tiles','seconds','traceback']}),flush=True)
    save_json(out/'summary.json',{'state':'complete','completed':len(results),'requested':len(rows),'results':results,
                                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    assert all(r['status']=='PASS_TECHNICAL_ONLY' for r in results),'Pilot failures require review'

if __name__=='__main__':main()
