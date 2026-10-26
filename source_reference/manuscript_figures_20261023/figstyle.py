"""Shared style and helpers for the manuscript figures, following Nature Portfolio artwork conventions: 180 mm double-column
width, sans-serif text at 5.5-7 pt, 8 pt bold panel labels in the journal's a), b), c) form, 0.5 pt axes, italic P, and
colour-blind-safe palettes checked with the dataviz validator (Machado 2009 CVD simulation, OKLab distances). No study
module is imported here, so each figure script can import its own locked study modules. Vector PDF + 300 dpi PNG."""
from pathlib import Path
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
MM=1/25.4;W=180*MM;PT=1/72
INK='#222222';MUTED='#666666';FAINT='#BBBBBB';GRID='#EBEBEB'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','DejaVu Sans'],'font.size':6,
                     'axes.titlesize':6.5,'axes.titleweight':'normal','axes.titlepad':3.5,'axes.titlelocation':'left',
                     'axes.labelsize':6.5,'axes.labelpad':2,'xtick.labelsize':6,'ytick.labelsize':6,
                     'legend.fontsize':5.8,'legend.title_fontsize':5.8,'legend.frameon':False,'legend.handlelength':1.3,
                     'legend.handletextpad':0.4,'legend.borderaxespad':0.3,'legend.labelspacing':0.28,'legend.borderpad':0.2,
                     'axes.linewidth':0.5,'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
                     'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.minor.width':0.4,'ytick.minor.width':0.4,
                     'xtick.major.size':2,'ytick.major.size':2,'xtick.minor.size':1.2,'ytick.minor.size':1.2,
                     'xtick.major.pad':1.5,'ytick.major.pad':1.5,'lines.linewidth':0.9,'lines.markersize':3,'patch.linewidth':0.5,
                     'errorbar.capsize':1.2,'axes.spines.top':False,'axes.spines.right':False,
                     'mathtext.fontset':'custom','mathtext.rm':'Liberation Sans','mathtext.it':'Liberation Sans:italic',
                     'mathtext.bf':'Liberation Sans:bold','mathtext.default':'regular',
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'image.interpolation':'antialiased'})
KOR='#0072B2';SCO='#D55E00';HI='#C0392B';LO='#2C7FB8';PATH='#0072B2';IMG='#8C8C8C';MID='#7B7B7B'
# ISUP 1-4. Validated as a set (all pairs): worst CVD dE 16.6, worst normal-vision dE 16.5, every step >= 3.4:1 on white.
GRADE_C=['#24589E','#3B8FCB','#D46F2A','#A50F15']
ENC_NAME={'phikon-v2':'Phikon-v2','uni':'UNI','uni2-h':'UNI2-h','virchow2':'Virchow2','gigapath':'Prov-GigaPath','conch':'CONCH',
          'resnet50':'ResNet50','convnext_tiny':'ConvNeXt-T','vit_b_16':'ViT-B/16'}
ENC_C={'phikon-v2':'#0072B2','uni':'#009E73','uni2-h':'#56B4E9','virchow2':'#CC79A7','gigapath':'#E69F00','conch':'#D55E00',
       'resnet50':'#999999','convnext_tiny':'#666666','vit_b_16':'#BBBBBB'}
HALO=[pe.withStroke(linewidth=1.6,foreground='white')]
OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)

def label(ax,s,dx=-20,dy=3,x=None,y=None):
    """Panel label 'a)' (npj convention), 8 pt bold. Placed dx, dy points from the axes' top-left corner, or at axes
    fraction (x, y) when given."""
    kw=dict(fontsize=8,fontweight='bold',ha='left',va='bottom')
    if x is not None:return ax.text(x,1.04 if y is None else y,f'{s})',transform=ax.transAxes,**kw)
    return ax.annotate(f'{s})',xy=(0,1),xycoords='axes fraction',xytext=(dx,dy),textcoords='offset points',annotation_clip=False,**kw)
def flabel(fig,x_mm,y_mm,s):
    """Panel label at a position in mm from the figure's bottom-left corner."""
    w,h=fig.get_size_inches();fig.text(x_mm*MM/w,y_mm*MM/h,f'{s})',fontsize=8,fontweight='bold',ha='left',va='bottom')
def save(fig,name):fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png');plt.close(fig);print('WROTE',name,flush=True)
def pfmt(p):
    P='$\\mathit{P}$'
    if p is None:return f'{P} = n/a'
    if p<1e-3:m,e=f'{p:.1e}'.split('e');return f'{P} = {m} × 10$^{{{int(e)}}}$'
    return f'{P} = {p:.2g}' if p<0.1 else f'{P} = {p:.2f}'
def despine(ax,left=False,bottom=False):
    if left:ax.spines['left'].set_visible(False);ax.tick_params(left=False)
    if bottom:ax.spines['bottom'].set_visible(False);ax.tick_params(bottom=False)

def km_curve(t,e):
    t=np.asarray(t,float);e=np.asarray(e,int);xs=[0.];ys=[1.];s=1.
    for u in np.unique(t[e==1]):
        n=(t>=u).sum();d=((t==u)&(e==1)).sum();xs.append(u);ys.append(s);s*=1-d/n;xs.append(u);ys.append(s)
    xs.append(t.max());ys.append(s);return np.array(xs),np.array(ys)
def km_at(t,e,x):
    t=np.asarray(t,float);e=np.asarray(e,int);s=1.
    for u in np.unique(t[e==1]):
        if u>x:break
        s*=1-((t==u)&(e==1)).sum()/(t>=u).sum()
    return s
def km_panel(ax,groups,title=None,p=None,tmax=84,step=12,legend_loc='lower left',ylabel=True,legend_title='Recurrences/patients',
             risk_label='No. at risk',p_xy=(0.98,0.05),ymin=0.5):
    """groups: list of (name, short, colour, t, e). The y axis runs from ymin to 100% (in percent) so that the curves use
    the panel; no curve may fall below ymin. Censor ticks on the curves; a compact numbers-at-risk table is drawn under the
    x-axis label (callers leave ~16 mm below the axes for it)."""
    for name,short,col,t,e in groups:
        if len(t)==0:continue
        x,y=km_curve(t,e);assert y.min()>=ymin,(name,y.min(),ymin)
        ax.plot(x,y,color=col,lw=0.9,label=f'{name} ({int(np.sum(e))}/{len(t)})',solid_joinstyle='miter',solid_capstyle='butt',zorder=3)
        cens=np.asarray(t)[np.asarray(e)==0];ax.plot(cens,[km_at(t,e,c) for c in cens],'|',color=col,ms=2.0,mew=0.45,zorder=3)
    yt=np.round(np.arange(ymin,1.0001,0.1),2)
    ax.set(xlim=(0,tmax),ylim=(ymin,1.012),xticks=range(0,tmax+1,step),yticks=yt);ax.set_yticklabels([f'{v*100:.0f}' for v in yt])
    ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True)
    if ylabel:ax.set_ylabel('Recurrence-free survival (%)')
    if title:ax.set_title(title)
    ax.legend(loc=legend_loc,title=legend_title,alignment='left',handlelength=1.1)
    if p is not None:ax.text(*p_xy,pfmt(p),transform=ax.transAxes,ha='right',va='bottom',fontsize=6)
    ax.set_xlabel('Months since surgery')
    y0=-28;row=7
    ax.annotate(risk_label,xy=(0,0),xycoords=('data','axes fraction'),xytext=(-10,y0+1),textcoords='offset points',ha='right',va='center',
                fontsize=5.8,color=MUTED,annotation_clip=False)
    for i,(name,short,col,t,e) in enumerate(groups):
        yy=y0-(i+1)*row;t=np.asarray(t)
        ax.annotate(short,xy=(0,0),xycoords=('data','axes fraction'),xytext=(-10,yy),textcoords='offset points',ha='right',va='center',
                    fontsize=5.8,color=col,fontweight='bold',annotation_clip=False)
        for m in range(0,tmax+1,step):
            ax.annotate(str(int((t>=m).sum())),xy=(m,0),xycoords=('data','axes fraction'),xytext=(0,yy),textcoords='offset points',
                        ha='center',va='center',fontsize=5.8,annotation_clip=False)
def spread(vals,gap):
    """Label positions for values sorted in descending order: neighbours closer than gap are pushed apart symmetrically
    until every pair is at least gap apart, so that the labels keep the order of the values and stay near them."""
    y=np.array(vals,float)
    for _ in range(500):
        moved=False
        for i in range(len(y)-1):
            d=y[i]-y[i+1]
            if d<gap-1e-12:s=(gap-d)/2;y[i]+=s;y[i+1]-=s;moved=True
        if not moved:break
    return list(y)
def roc(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float);o=np.argsort(-s,kind='stable');y=y[o];s=s[o]
    tp=np.cumsum(y);fp=np.cumsum(~y);keep=np.r_[np.diff(s)!=0,True]
    return np.r_[0,fp[keep]/(~y).sum()],np.r_[0,tp[keep]/y.sum()]
def grid_image(xy,val):
    """Patch values on their tissue grid (scenes side by side), NaN background, one pixel per patch."""
    step=int(np.median(xy[:,3]));parts=[]
    for s in np.unique(xy[:,0]):
        k=xy[:,0]==s;gx=((xy[k,1]-xy[k,1].min())//step).astype(int);gy=((xy[k,2]-xy[k,2].min())//step).astype(int)
        a=np.full((gy.max()+1,gx.max()+1),np.nan);a[gy,gx]=val[k];parts.append(a)
    h=max(a.shape[0] for a in parts);out=[]
    for a in parts:out.append(np.vstack([a,np.full((h-a.shape[0],a.shape[1]),np.nan)]));out.append(np.full((h,3),np.nan))
    return np.hstack(out[:-1])
def coords_of(C,c):
    """Patch coordinates in feature order: the QC pool's primary/<stem>.npz, sha256-checked against complete.json."""
    pool=C.POOL if c['cohort']=='korea' else C.SPOOL;p=pool/'primary'/f"{c['stem']}.npz"
    rec=json.loads((C.FEATURE_DIR[c['cohort']]/c['stem']/'complete.json').read_text())
    if 'primary_coords_sha256' in rec:assert C.sha(p)==rec['primary_coords_sha256']
    with np.load(p,allow_pickle=False) as a:xy=a['coords'].copy()
    assert len(xy)==rec['n'];return xy

# ---------------------------------------------------------------------------- slide overviews with patch overlays
def slide_overview(reader,xy,max_side=1500,gap_px=24):
    """Low-magnification H&E of the scenes that hold patches (pyramid level nearest 27x), cropped to the patched tissue
    and placed side by side. Returns the RGB image, each patch's box (x0, y0, x1, y1) in its pixels, and µm per pixel."""
    from PIL import Image
    parts=[];boxes=np.zeros((len(xy),4));xoff=0
    for s in np.unique(xy[:,0]):
        k=np.flatnonzero(xy[:,0]==s);rgb,_,(ox,oy,lev)=reader.overview(int(s))
        x0=(xy[k,1]-ox)/lev;y0=(xy[k,2]-oy)/lev;x1=(xy[k,1]+xy[k,3]-ox)/lev;y1=(xy[k,2]+xy[k,4]-oy)/lev
        m=int(round(3*np.median(xy[k,3])/lev))
        cx0=max(0,int(x0.min())-m);cy0=max(0,int(y0.min())-m);cx1=min(rgb.shape[1],int(np.ceil(x1.max()))+m);cy1=min(rgb.shape[0],int(np.ceil(y1.max()))+m)
        parts.append(rgb[cy0:cy1,cx0:cx1]);boxes[k]=np.c_[x0-cx0+xoff,y0-cy0,x1-cx0+xoff,y1-cy0];xoff+=cx1-cx0+gap_px
    H=max(p.shape[0] for p in parts);Wd=xoff-gap_px;img=np.full((H,Wd,3),255,np.uint8);xo=0
    for p in parts:img[:p.shape[0],xo:xo+p.shape[1]]=p;xo+=p.shape[1]+gap_px
    img[(img<28).all(2)]=255  # scanner background outside the tissue (black on some scanners) shown as white
    f=min(1.0,max_side/max(img.shape[:2]))
    if f<1:img=np.asarray(Image.fromarray(img).resize((round(img.shape[1]*f),round(img.shape[0]*f)),Image.Resampling.LANCZOS));boxes=boxes*f
    return img,boxes,lev*reader.mpp['X']/f
def spaced_top(score,xy,n,min_sep_patches=8):
    """The n highest-scoring patches, taken greedily so that each lies at least min_sep_patches patch widths from those
    already taken (same scene), so that the marks on an overview do not pile up on one focus."""
    step=float(np.median(xy[:,3]));taken=[]
    for i in np.argsort(-score):
        if all(xy[i,0]!=xy[j,0] or np.hypot(xy[i,1]-xy[j,1],xy[i,2]-xy[j,2])>=min_sep_patches*step for j in taken):taken.append(i)
        if len(taken)==n:break
    return np.array(taken)
def scalebar(ax,umpp,mm=2,pad_px=12):
    """A scale bar of mm millimetres in the lower-left corner of an image axes."""
    L=mm*1000/umpp;h=ax.get_ylim()[0];ax.plot([pad_px,pad_px+L],[h-pad_px,h-pad_px],color=INK,lw=1.3,solid_capstyle='butt')
    ax.text(pad_px+L/2,h-pad_px-6,f'{mm} mm',ha='center',va='bottom',fontsize=5.5,path_effects=HALO)
def overlay(shape,boxes,vals,cmap,norm,alpha):
    """RGBA layer with each patch's box filled in cmap(norm(value))."""
    ov=np.zeros(tuple(shape[:2])+(4,));cols=plt.get_cmap(cmap)(norm(np.asarray(vals,float)))
    b=np.c_[np.floor(boxes[:,:2]),np.ceil(boxes[:,2:])].astype(int)
    for (x0,y0,x1,y1),c in zip(b,cols):ov[y0:y1,x0:x1,:3]=c[:3];ov[y0:y1,x0:x1,3]=alpha
    return ov
def grey(img,lift=0.35):
    g=(img[...,:3].astype(float)@[0.299,0.587,0.114])/255;g=lift+(1-lift)*g;return np.repeat(g[...,None],3,2)
def mark_patches(ax,boxes,idx,min_px=14,col='black'):
    """Numbered squares (1, 2, ...) at the given patches, at least min_px wide so they stay visible on the overview;
    each mark carries a halo in the opposite tone."""
    halo='white' if col=='black' else 'black'
    for n,i in enumerate(idx,1):
        x0,y0,x1,y1=boxes[i];cx,cy=(x0+x1)/2,(y0+y1)/2;h=max(x1-x0,min_px)/2
        ax.add_patch(plt.Rectangle((cx-h,cy-h),2*h,2*h,fill=False,ec=halo,lw=1.6));ax.add_patch(plt.Rectangle((cx-h,cy-h),2*h,2*h,fill=False,ec=col,lw=0.7))
        ax.text(cx+h+2,cy-h-1,str(n),fontsize=6,fontweight='bold',color=col,ha='left',va='bottom',
                path_effects=[pe.withStroke(linewidth=1.6,foreground=halo)])
def hbar(fig,ax,mappable,ticks,label_txt,frac=(0.15,-0.07,0.7,0.035)):
    ca=ax.inset_axes(list(frac));cb=fig.colorbar(mappable,cax=ca,orientation='horizontal');cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=5.5,length=1.5,width=0.4,pad=1);cb.outline.set_linewidth(0.4);cb.set_label(label_txt,fontsize=5.8,labelpad=1.5);return cb
def tile(ax,rgb,num=None,frame=None,caption=None,scalebar_um=None,field_um=128):
    """A native-resolution patch with an optional number badge, frame colour, caption underneath and scale bar."""
    from PIL import Image
    im=np.asarray(Image.fromarray(rgb).resize((448,448),Image.Resampling.BICUBIC));ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(1.0 if frame else 0.4);sp.set_color(frame or INK)
    if num is not None:ax.text(18,20,str(num),fontsize=6.2,fontweight='bold',ha='left',va='top',color='black',
                               bbox=dict(boxstyle='circle,pad=0.18',fc='white',ec='none',alpha=0.9))
    if caption:ax.text(0.5,-0.04,caption,transform=ax.transAxes,ha='center',va='top',fontsize=5.8)
    if scalebar_um:
        L=scalebar_um/field_um*448;ax.plot([448-16-L,448-16],[426,426],color='black',lw=1.3,solid_capstyle='butt')
        ax.text(448-16-L/2,414,f'{scalebar_um} µm',ha='center',va='bottom',fontsize=5.5,path_effects=HALO)
def example_row(fig,gs,reader,xy,ph,at,title,top_n=4):
    """One slide: H&E overview with the per-patch P(grade 3) + P(grade 4) overlay, the attention overlay on a grey
    overview, and the top_n patches by attention x (P(grade 3) + P(grade 4)), at least 8 patch widths
    (about 1 mm) apart, at native resolution and numbered on both maps. gs: a subgridspec of 1 x 3 (map, map, tiles)."""
    img,boxes,umpp=slide_overview(reader,xy);top=spaced_top(at*ph,xy,top_n)
    ax=fig.add_subplot(gs[0]);ax.imshow(img);ov=overlay(img.shape,boxes,ph,'RdYlBu_r',Normalize(0,1),0.7);ax.imshow(ov)
    mark_patches(ax,boxes,top);ax.set_axis_off();ax.set_title(title,fontsize=6.5,loc='left');scalebar(ax,umpp)
    sm=plt.cm.ScalarMappable(Normalize(0,1),'RdYlBu_r');hbar(fig,ax,sm,[0,0.5,1],'Patch P(grade 3) + P(grade 4)')
    ax2=fig.add_subplot(gs[1]);ax2.imshow(grey(img));la=np.log10(at/at.max()+1e-6)
    ov=overlay(img.shape,boxes,la,'magma',Normalize(-4,0),0.85);ax2.imshow(ov);mark_patches(ax2,boxes,top,col='white')
    ax2.set_axis_off();sm=plt.cm.ScalarMappable(Normalize(-4,0),'magma');hbar(fig,ax2,sm,[-4,-2,0],'Attention (log$_{10}$, relative to maximum)')
    sub=gs[2].subgridspec(2,2,wspace=0.06,hspace=0.16)
    for n,i in enumerate(top):
        s_,x,y,w,h=[int(v) for v in xy[i]];rgb,_,_=reader.region(s_,x,y,w,h);a=fig.add_subplot(sub[n//2,n%2])
        tile(a,rgb,num=n+1,frame=HI if ph[i]>0.5 else LO,caption=f'P(3) + P(4) = {ph[i]:.2f}',scalebar_um=50 if n==len(top)-1 else None)
    return ax,ax2
