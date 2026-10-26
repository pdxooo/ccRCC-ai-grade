"""Full candidate native-pixel QC. Flags are heuristics, not pathology judgements."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys, csv, json, time, hashlib, traceback, math, gzip, heapq, argparse, fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from czi_pilot_v2 import ROOT, Reader, spatial_rgb, save_json, np, Image, ImageDraw

VERSION='native_qc_v2_20261009'
POLICY={'field_um':128,'candidate_fraction':0.10,'blank_reject_white':0.995,
        'black_reject':0.95,'review_black':0.02,'review_tissue':0.10,
        'review_focus':15.0,'pen_score_informational_only':True,
        'note':'Fixed label-blind heuristic flags; focus flags do not prove blur. Blue/green score is informational, not a selection rule. No tumor detector.'}

class IndexedReader(Reader):
    def __init__(self,path):
        super().__init__(path);self.index=defaultdict(list)
        for (scene,level), entries in self.groups.items():
            if level!=1:continue
            for entry in entries:
                i,x,y,w,h=entry
                for gy in range(y//2048,(y+h-1)//2048+1):
                    for gx in range(x//2048,(x+w-1)//2048+1):self.index[(scene,gx,gy)].append(entry)
    def decode(self,i):
        if i not in self.cache:
            e=self.entries[i];self.cache[i]=spatial_rgb(e.data_segment().data(resize=False),e.axes)
            if len(self.cache)>32:self.cache.popitem(last=False)
        self.cache.move_to_end(i);return self.cache[i]
    def region(self,scene,x,y,w,h):
        arr=np.full((h,w,3),255,np.uint8);cover=np.zeros((h,w),bool);used=[];entries={}
        for gy in range(y//2048,(y+h-1)//2048+1):
            for gx in range(x//2048,(x+w-1)//2048+1):
                for e in self.index.get((scene,gx,gy),[]):entries[e[0]]=e
        for i,tx,ty,tw,th in sorted(entries.values()):
            xa=max(x,tx);ya=max(y,ty);xb=min(x+w,tx+tw);yb=min(y+h,ty+th)
            if xb<=xa or yb<=ya:continue
            tile=self.decode(i);assert tile.shape[:2]==(th,tw)
            arr[ya-y:yb-y,xa-x:xb-x]=tile[ya-ty:yb-ty,xa-tx:xb-tx]
            cover[ya-y:yb-y,xa-x:xb-x]=True;used.append(i)
        return arr,cover,used

def metrics(rgb,cover):
    # Fixed resolution: native sampling ~0.22 um/px. Focus calculated on native pixels.
    small=np.asarray(Image.fromarray(rgb).resize((192,192),Image.Resampling.BOX))
    a=small.astype(np.int16);lo=a.min(2);hi=a.max(2);mean=a.mean(2);chroma=hi-lo
    white=float(((lo>242)&(chroma<12)).mean());black=float((hi<25).mean())
    tissue=(mean>25)&(lo<235)&((chroma>5)|(mean<210))
    pen=((a[:,:,2]>a[:,:,0]*1.5)&(a[:,:,2]>a[:,:,1]*1.3)&(a[:,:,0]<120)) | ((a[:,:,1]>a[:,:,0]*1.5)&(a[:,:,1]>a[:,:,2]*1.3)&(a[:,:,0]<140))
    gray=(rgb[:,:,0].astype(np.float32)*.299+rgb[:,:,1]*.587+rgb[:,:,2]*.114)
    lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
    native_tissue=np.asarray(Image.fromarray(tissue).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.NEAREST))[1:-1,1:-1]
    focus=float(np.var(lap[native_tissue])) if native_tissue.any() else 0.0
    return {'coverage':float(cover.mean()),'white':white,'black':black,'tissue':float(tissue.mean()),'focus':focus,'pen':float(pen.mean())}

def decide(m):
    if m['coverage']<0.999:return 'HOLD','unscanned_gap'
    if m['black']>=POLICY['black_reject']:return 'REJECT','almost_all_black'
    if m['white']>=POLICY['blank_reject_white'] and m['tissue']<0.005:return 'REJECT','almost_all_blank'
    flags=[]
    if m['black']>POLICY['review_black']:flags.append('black_edge')
    if m['tissue']<POLICY['review_tissue']:flags.append('sparse_or_pale')
    if m['focus']<POLICY['review_focus']:flags.append('low_focus_score')
    return ('REVIEW',';'.join(flags)) if flags else ('PASS_AUTO','')

def integral(mask):return np.pad(mask.astype(np.int32),((1,0),(1,0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
def fraction(ii,x,y,crop,origin,shape):
    ox,oy,scale=origin;xa=max(0,math.floor((x-ox)/scale));ya=max(0,math.floor((y-oy)/scale))
    xb=min(shape[1],math.ceil((x+crop-ox)/scale));yb=min(shape[0],math.ceil((y+crop-oy)/scale))
    return float((ii[yb,xb]-ii[ya,xb]-ii[yb,xa]+ii[ya,xa])/max(1,(yb-ya)*(xb-xa)))

def sheet(items,path):
    if not items:return
    canvas=Image.new('RGB',(4*224,math.ceil(len(items)/4)*252),'white');draw=ImageDraw.Draw(canvas)
    for j,(rgb,label) in enumerate(items):
        xx=j%4*224;yy=j//4*252;canvas.paste(Image.fromarray(rgb).resize((224,224)),(xx,yy));draw.text((xx+2,yy+225),label[:34],fill='black')
    canvas.save(path,quality=88)

def run_slide(row,out,limit):
    started=time.monotonic();path=ROOT/'data/he'/row['slide_filename'];dst=Path(out)/path.stem;dst.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final=dst/'report.json'
    if final.exists():
        prior=json.loads(final.read_text())
        if prior.get('state')=='complete' and prior.get('code_sha256')==code_hash and prior.get('limit')==limit:return prior
    st=path.stat();assert st.st_size==int(row['source_bytes'])
    receipt=json.loads(path.with_name(path.name+'.receipt.json').read_text());assert receipt['sha256_local']==row['source_sha256']
    r=IndexedReader(path);counts=Counter();reasons=Counter();scenes=[];n=0;rejected_grid=0;random_examples=[];flag_examples=[];lowfocus=[];excluded_examples=[]
    rng=np.random.default_rng(int(hashlib.sha256(path.name.encode()).hexdigest()[:8],16))
    fields=['scene','x','y','w','h','candidate_fraction','coverage','white','black','tissue','focus','pen','decision','reason']
    try:
        crop=round(POLICY['field_um']/r.mpp['X'])
        with gzip.open(dst/'patch_qc.csv.gz.tmp','wt',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for scene in r.scenes:
                reader_check=r.check_reader(scene);rgb,covered,origin=r.overview(scene)
                a=rgb.astype(np.int16);mean=a.mean(2);lo=a.min(2);chroma=a.max(2)-lo
                mask=covered&(mean>15)&(lo<235)&((chroma>5)|(mean<210));ii=integral(mask)
                pic=Image.fromarray(rgb);pic.thumbnail((1200,1200));pic.save(dst/f'scene{scene}_overview.jpg',quality=86)
                tint=rgb.copy();tint[mask]=(tint[mask]*.65+np.array([0,255,0])*.35).astype(np.uint8)
                pic=Image.fromarray(tint);pic.thumbnail((1200,1200));pic.save(dst/f'scene{scene}_mask.jpg',quality=86)
                b=r.bbox(scene);grid=[];negative=[];total=0
                for y in range(b[1],b[3]-crop+1,crop):
                    for x in range(b[0],b[2]-crop+1,crop):
                        total+=1;f=fraction(ii,x,y,crop,origin,rgb.shape)
                        if f>=POLICY['candidate_fraction']:grid.append((x,y,f))
                        else:
                            rejected_grid+=1
                            # Prefer near-threshold exclusions for a deliberate false-negative audit.
                            if f>0:heapq.heappush(negative,(f,x,y))
                            if len(negative)>8:heapq.heappop(negative)
                # Preserve a spatial stripe order to reuse decoded native subblocks.
                grid.sort(key=lambda t:(t[1]//2048,t[0]//2048,t[1],t[0]))
                scenes.append({'scene':scene,'total_grid':total,'candidates':len(grid),'reader_check':reader_check,'overview_origin':origin})
                for f,x,y in sorted(negative,reverse=True):
                    tile,cov,_=r.region(scene,x,y,crop,crop);m=metrics(tile,cov)
                    excluded_examples.append((np.asarray(Image.fromarray(tile).resize((224,224))),f'excl s{scene} f={f:.2f} t={m["tissue"]:.2f}'))
                for x,y,f in grid:
                    if limit and n>=limit:break
                    tile,cov,_=r.region(scene,x,y,crop,crop);m=metrics(tile,cov);decision,reason=decide(m)
                    writer.writerow(dict(scene=scene,x=x,y=y,w=crop,h=crop,candidate_fraction=round(f,5),**{k:round(v,5) for k,v in m.items()},decision=decision,reason=reason))
                    n+=1;counts[decision]+=1;reasons.update(reason.split(';') if reason else [])
                    thumb=None
                    j=int(rng.integers(n))
                    if len(random_examples)<16 or j<16:
                        thumb=np.asarray(Image.fromarray(tile).resize((224,224)));entry=(thumb,f'{decision} s{scene} {x},{y}')
                        if len(random_examples)<16:random_examples.append(entry)
                        else:random_examples[j]=entry
                    if decision!='PASS_AUTO' and (len(flag_examples)<16 or int(rng.integers(max(1,n-counts['PASS_AUTO'])))<16):
                        if thumb is None:thumb=np.asarray(Image.fromarray(tile).resize((224,224)))
                        entry=(thumb,reason)
                        if len(flag_examples)<16:flag_examples.append(entry)
                        else:flag_examples[int(rng.integers(16))]=entry
                    if len(lowfocus)<16 or m['focus'] < -lowfocus[0][0]:
                        if thumb is None:thumb=np.asarray(Image.fromarray(tile).resize((224,224)))
                        heapq.heappush(lowfocus,(-m['focus'],n,thumb,f'focus={m["focus"]:.1f} {decision}'))
                        if len(lowfocus)>16:heapq.heappop(lowfocus)
                    if n%500==0:save_json(dst/'progress.json',{'state':'running','processed':n,'counts':dict(counts),'elapsed_seconds':time.monotonic()-started})
        assert n>0,'No candidate patches: slide requires review'
        (dst/'patch_qc.csv.gz.tmp').replace(dst/'patch_qc.csv.gz')
        sheet(random_examples,dst/'random.jpg');sheet(flag_examples,dst/'flagged.jpg');sheet([(a[2],a[3]) for a in sorted(lowfocus,key=lambda a:-a[0])],dst/'low_focus.jpg');sheet(excluded_examples,dst/'excluded_boundary.jpg')
        assert path.stat().st_size==st.st_size and path.stat().st_mtime_ns==st.st_mtime_ns
        result={'state':'complete','patient_id':row['patient_id'],'slide_filename':path.name,'source_sha256':row['source_sha256'],
                'version':VERSION,'code_sha256':code_hash,'policy':POLICY,'limit':limit,'mpp':r.mpp,'crop_native':crop,'scenes':scenes,
                'processed':n,'counts':dict(counts),'flag_reasons':dict(reasons),'lowres_excluded_grid':rejected_grid,
                'seconds':time.monotonic()-started,'scope':'All selected candidates measured if limit=0; PASS_AUTO is heuristic technical eligibility, not pathology approval.'}
        save_json(final,result);return result
    finally:r.close()

def main():
    # Check every decision branch before touching a slide.
    base=dict(coverage=1.,white=0.,black=0.,tissue=.8,focus=100.,pen=0.)
    assert decide(base)==('PASS_AUTO','')
    assert decide(dict(base,white=1.,tissue=0.))[0]=='REJECT'
    assert decide(dict(base,black=1.))[0]=='REJECT'
    assert decide(dict(base,coverage=.5))[0]=='HOLD'
    assert decide(dict(base,focus=0.))[0]=='REVIEW'
    assert decide(dict(base,tissue=.01))[0]=='REVIEW'
    assert decide(dict(base,pen=.5))[0]=='PASS_AUTO'  # H&E blue nuclei must not trigger pen rejection.
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=list(csv.DictReader(open(args.manifest,encoding='utf-8-sig')));results=[]
    save_json(out/'run_config.json',{'policy':POLICY,'args':vars(args),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_slide,row,str(out),args.limit):row for row in rows}
        for f in as_completed(futures):
            row=futures[f]
            try:result=f.result()
            except Exception:result={'state':'error','slide_filename':row['slide_filename'],'traceback':traceback.format_exc()}
            results.append(result);save_json(out/'summary.json',{'state':'running','requested':len(rows),'completed':len(results),'results':results})
            print(json.dumps(result),flush=True)
    save_json(out/'summary.json',{'state':'complete' if all(r['state']=='complete' for r in results) else 'completed_with_errors','requested':len(rows),'completed':len(results),'results':results})
    if any(r['state']!='complete' for r in results):sys.exit(1)

if __name__=='__main__':main()
