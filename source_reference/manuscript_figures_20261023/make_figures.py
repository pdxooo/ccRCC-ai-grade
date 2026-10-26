"""Manuscript figures 1-4 and 6 (npj Precision Oncology draft). Presentation only: every number comes from the locked studies'
outputs (gx1 RESULTS.json and its 225 runs read through gx1's own gates, the exploratory EXPLAIN_RESULTS.json,
case_features.csv and patch maps) and is recomputed the same way where a curve needs per-case values. Writes
figures/Fig1-4,6.pdf (vector) and .png (300 dpi), 180 mm wide. Usage (lab-a6000): HE_PATH_PREFIX=... python make_figures.py"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
GX1=(HERE/'..'/'grade_ext_v1_20261022').resolve();EXPL=(HERE/'..'/'gx1_explain_v1_20261023').resolve()
sys.path.insert(0,str(HERE));sys.path.insert(1,str(GX1))
from figstyle import *
from matplotlib.patches import FancyBboxPatch,Polygon
import common as C
import evaluate as E
sys.path.insert(0,str(C.ROOT/'qc_deps'));sys.path.insert(0,str(C.ROOT/'bib_phase1'))

def load():
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    enc='phikon-v2';S=cfg['seeds'];F=cfg['folds']
    P={p:np.mean([kr[(enc,s)][p] for s in S],0) for p in kr[(enc,S[0])]}
    P.update({c['patient_id']:np.mean([sc[(enc,s,f)][c['patient_id']] for s in S for f in F],0) for c in scot})
    ai={p:float(C.grade_score(v[None])[0]) for p,v in P.items()}
    R=json.loads((GX1/'RESULTS.json').read_text());X=json.loads((EXPL/'EXPLAIN_RESULTS.json').read_text())
    feats={r['patient_id']:r for r in C.read_csv(EXPL/'case_features.csv')}
    IF={**{r['patient_id']:r for r in C.read_csv(EXPL/'inputs/korea_if.csv')},**{r['patient_id']:r for r in C.read_csv(EXPL/'inputs/scotland_if.csv')}}
    return dict(cfg=cfg,cases=cases,scot=scot,P=P,ai=ai,R=R,X=X,feats=feats,IF=IF,enc=enc,byid={c['patient_id']:c for c in cases+scot})
def reader_for(c):
    from full_patch_qc import IndexedReader
    return IndexedReader(str((C.ROOT if c['cohort']=='korea' else C.SROOT)/'data/he'/(c['stem']+'.czi')))

# ------------------------------------------------------------------------------------------------------- Fig 1
def fig1(D):
    H=96;fig=plt.figure(figsize=(W,H*MM));A=fig.add_axes([0,0,1,1]);A.set_xlim(0,180);A.set_ylim(0,H);A.axis('off')
    def axmm(x,y,w,h):return fig.add_axes([x/180,y/H,w/180,h/H])
    def arrow(x0,x1,y):A.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle='-|>',lw=0.6,color=INK,mutation_scale=6,shrinkA=0,shrinkB=0))
    def cap(x,t,y=53.0,**k):A.text(x,y,t,ha='center',va='top',fontsize=5.7,linespacing=1.25,**k)
    for s_,(x,y) in zip('abc',((2,91),(62,91),(2,29.5))):A.text(x,y,f'{s_})',fontsize=8,fontweight='bold',va='bottom')
    # a) cohorts
    for (coh,cs,col,tint,y0,head,l2,l3) in (('korea',D['cases'],KOR,'#E8F1F8',66,'Korea · development','300 patients · 48 recurrences in 295','Locked five-fold cross-validation'),
                                         ('scotland',D['scot'],SCO,'#FBEDE5',41,'Scotland · external validation','120 patients · 27 recurrences','Never used for training, selection or tuning')):
        A.add_patch(FancyBboxPatch((4,y0),53,22,boxstyle='round,pad=0,rounding_size=1.6',fc=tint,ec=col,lw=0.6))
        A.text(7,y0+18,head,fontsize=6.5,fontweight='bold',color=col,va='center');A.text(7,y0+13.6,l2,fontsize=6,va='center');A.text(7,y0+9.6,l3,fontsize=6,va='center',color=MUTED)
        n=np.array([sum(c['isup']==k for c in cs) for k in (1,2,3,4)]);x=7.0;L=47.0
        for k,v in enumerate(n):
            w=L*v/n.sum();A.add_patch(plt.Rectangle((x,y0+2.6),w,4.2,fc=GRADE_C[k],ec='white',lw=0.6))
            if w>4:A.text(x+w/2,y0+4.7,str(v),ha='center',va='center',fontsize=5.6,color='white',fontweight='bold')
            x+=w
    for k in range(4):A.add_patch(plt.Rectangle((7+k*12.4,36.4),2.6,2.2,fc=GRADE_C[k],ec='none'));A.text(10.3+k*12.4,37.5,f'ISUP {k+1}',fontsize=5.6,va='center')
    # b) pipeline, drawn with the Fig. 4 example slide and its own model outputs
    ex=json.loads((EXPL/'examples.json').read_text())['scotland'][0];c=D['byid'][ex['case']];yc=66
    m=np.load(EXPL/'maps'/f"{c['stem']}.npz");ph=m['p_high'];at=m['attention'];xy=coords_of(C,c);rd=reader_for(c)
    img,boxes,_=slide_overview(rd,xy,max_side=1000);top=spaced_top(at*ph,xy,3)
    ax=axmm(63,53.5,25,26);ax.imshow(img);ax.set_axis_off()
    for i in top:
        x0,y0_,x1,y1=boxes[i];cx,cy=(x0+x1)/2,(y0_+y1)/2;h=max(x1-x0,30)/2
        ax.add_patch(plt.Rectangle((cx-h,cy-h),2*h,2*h,fill=False,ec='white',lw=1.4));ax.add_patch(plt.Rectangle((cx-h,cy-h),2*h,2*h,fill=False,ec=INK,lw=0.6))
    cap(75.5,'H&E whole slide\n128 µm patches');arrow(89,92,yc)
    for j,i in enumerate(top[::-1]):
        s_,x,y,w,h=[int(v) for v in xy[i]];rgb,_,_=rd.region(s_,x,y,w,h);a=axmm(93+j*1.6,58.5+j*1.8,11,11);tile(a,rgb,frame=INK)
    rd.close();cap(100.4,'All patches\nof the slide');arrow(108.5,111.5,yc)
    A.add_patch(Polygon([(112,56),(112,76),(123,71),(123,61)],closed=True,fc='#E9EEF5',ec=INK,lw=0.5))
    A.text(117.3,yc,'Frozen\nencoder',ha='center',va='center',fontsize=5.8,linespacing=1.2);cap(117.5,'9 compared:\n6 pathology,\n3 ImageNet')
    arrow(124,127,yc)
    f=np.load(C.FEATURE_DIR['scotland']/c['stem']/'phikon-v2.npy',mmap_mode='r');v=np.asarray(f[np.sort(top)][:,:14],float);v=(v-v.mean(0))/(v.std(0)+1e-9)
    for j in range(3):
        a=axmm(128+j*2.7,58,1.9,16);a.imshow(v[j][:,None],cmap='PuBu',aspect='auto',vmin=-1.5,vmax=1.5);a.set_xticks([]);a.set_yticks([])
        for sp in a.spines.values():sp.set_linewidth(0.4)
    cap(131.5,'Patch\nfeatures h$_i$');arrow(137.5,140.5,yc)
    A.add_patch(FancyBboxPatch((141.5,56),14,20,boxstyle='round,pad=0,rounding_size=1.2',fc='#F3F3F3',ec=INK,lw=0.5))
    A.text(148.5,73.2,'Gated\nattention',ha='center',va='center',fontsize=5.8,linespacing=1.15)
    a=axmm(143.5,59.8,10,8.2);r=at[np.sort(top)]/at[np.sort(top)].max();a.bar(range(3),r,color=KOR,width=0.62);a.set_xticks(range(3));a.set_xticklabels(['a$_1$','a$_2$','a$_3$'],fontsize=5.2)
    a.set_yticks([]);a.set_ylim(0,1.15);a.spines['left'].set_visible(False);a.tick_params(length=0,pad=0.8);a.patch.set_alpha(0)
    cap(148.5,'Weighted mean\nΣ a$_i$h$_i$');arrow(156.5,159.5,yc)
    p=D['P'][c['patient_id']];a=axmm(162.5,64.5,15,10.5);a.bar(range(1,5),p,color=GRADE_C,width=0.66);a.set_xticks(range(1,5));a.set_ylim(0,1)
    a.set_yticks([0,0.5,1]);a.set_yticklabels(['0','0.5','1']);a.tick_params(labelsize=5.2,length=1.5,pad=1);a.set_title('Slide P(ISUP k)',fontsize=5.7,pad=2)
    a=axmm(162.5,56.5,15,3.2);g=D['ai'][c['patient_id']]
    a.imshow(np.linspace(0,1,256)[None],extent=(1,4,0,0.45),aspect='auto',cmap=matplotlib.colors.LinearSegmentedColormap.from_list('g',GRADE_C));a.set_xlim(1,4);a.set_ylim(0,1)
    a.set_yticks([]);a.spines['left'].set_visible(False);a.spines['bottom'].set_visible(False);a.set_xticks([1,2,3,4]);a.tick_params(labelsize=5.2,length=0,pad=0.6)
    a.plot([g],[0.72],marker='v',color=INK,ms=3.6,clip_on=False)
    cap(170,f'AI grade = Σ k·P(k)\n= {g:.2f}',y=53.0,fontweight='bold')
    A.text(63,42.5,f"Example: Scotland {c['patient_id'].split(':')[1]} (ISUP 2) with Phikon-v2; Scottish predictions are means of 25 models.",fontsize=5.5,color=MUTED,va='top')
    # c) pre-specification sequence
    steps=['Protocol and\ndecision rules','Independent\ncode review','SHA-256 lock of\ncode and inputs','Bitwise resume\ncheck on 2nd GPU','Formal runs','Evaluation by\nlocked rules']
    fills=['#EEF1F5']*4+['#E3EEF8','#FBF1DF'];n=2.6;g_=0.9;w=(174-5*(g_-n))/6;y=8.5;h=11;xs=[4+i*(w-n+g_) for i in range(6)]
    for i,(t,fc,x) in enumerate(zip(steps,fills,xs)):
        pts=[(x,y),(x+w-n,y),(x+w,y+h/2),(x+w-n,y+h),(x,y+h)]+([(x+n,y+h/2)] if i else [])
        A.add_patch(Polygon(pts,closed=True,fc=fc,ec=INK,lw=0.45));A.text(x+w/2+(n/2 if i else 0)-n/2,y+h/2,t,ha='center',va='center',fontsize=5.7,linespacing=1.15)
    def brace(x0,x1,t):A.plot([x0,x0,x1,x1],[21.2,22.4,22.4,21.2],color=INK,lw=0.5);A.text((x0+x1)/2,23.2,t,ha='center',va='bottom',fontsize=5.8)
    brace(xs[0]+0.5,xs[3]+w-1.0,'Fixed before the formal runs');brace(xs[4]+1.5,xs[5]+w-0.5,'Run and judged under the locked rules')
    A.text(4,4.2,'Applied to each formal study after exploratory development on the Korean cohort: grading (225 runs), recurrence (50 runs), and the secondary, exploratory and post hoc analyses.',
           fontsize=5.5,color=MUTED,va='center')
    save(fig,'Fig1')

# ------------------------------------------------------------------------------------------------------- Fig 2
def fig2(D):
    R=D['R'];e=D['enc'];fig=plt.figure(figsize=(W,122*MM))
    top=fig.add_gridspec(1,2,width_ratios=[1,1.95],wspace=0.42,left=0.075,right=0.985,top=0.955,bottom=0.565)
    bot=fig.add_gridspec(1,3,width_ratios=[1,1,0.06],wspace=0.3,left=0.075,right=0.535,top=0.42,bottom=0.075)
    rgt=fig.add_gridspec(1,2,width_ratios=[1,1.1],wspace=0.5,left=0.625,right=0.985,top=0.42,bottom=0.075)
    ax=fig.add_subplot(top[0]);label(ax,'a',dx=-24)
    for coh,cs,col,name in (('korea',D['cases'],KOR,'Korea, out-of-fold'),('scotland',D['scot'],SCO,'Scotland, external')):
        y=[c['isup']>=3 for c in cs];s=[D['P'][c['patient_id']][2]+D['P'][c['patient_id']][3] for c in cs];fx,ty=roc(y,s)
        g=R['encoders'][e][coh]['grading'];ax.plot(fx,ty,color=col,lw=0.9,label=f"{name}\nAUC {g['highlow_auc']:.3f} ({g['highlow_auc_ci95'][0]:.3f}–{g['highlow_auc_ci95'][1]:.3f})")
    ax.plot([0,1],[0,1],ls=(0,(1.5,1.5)),color=FAINT,lw=0.6);ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='1 − specificity',ylabel='Sensitivity',aspect='equal')
    ax.set_xticks([0,0.5,1]);ax.set_yticks([0,0.5,1]);ax.set_title('ISUP 3–4 versus 1–2 (Phikon-v2)');ax.legend(loc='lower right',handlelength=1.0)
    # b) encoder comparison as a horizontal dot plot
    ax=fig.add_subplot(top[1]);label(ax,'b',dx=-58)
    encs=sorted(R['encoders'],key=lambda k:(R['encoders'][k]['family']!='pathology',-R['encoders'][k]['scotland']['grading']['highlow_auc']))
    ypos=[];y=0
    for i,k in enumerate(encs):
        if i and R['encoders'][k]['family']!=R['encoders'][encs[i-1]]['family']:y+=0.9
        ypos.append(y);y+=1
    for k,y in zip(encs,ypos):
        r=R['encoders'][k];g=r['scotland']['grading'];col=PATH if r['family']=='pathology' else IMG
        ax.plot(g['highlow_auc_ci95'],[y,y],color=col,lw=0.9,solid_capstyle='butt');ax.plot(g['highlow_auc'],y,'o',color=col,ms=4,mec='white',mew=0.4,zorder=3)
        ax.plot(r['korea']['grading']['highlow_auc'],y,'D',mfc='white',mec=col,mew=0.7,ms=3.2,zorder=3)
    ax.axvline(0.80,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.text(0.797,ypos[-1]+0.85,'Pre-specified threshold',fontsize=5.5,color=MUTED,va='center',ha='right')
    ax.set_yticks(ypos);ax.set_yticklabels([ENC_NAME[k]+(' (primary)' if k=='phikon-v2' else '') for k in encs]);ax.invert_yaxis()
    for t,k in zip(ax.get_yticklabels(),encs):
        if k=='phikon-v2':t.set_fontweight('bold')
    ax.set(xlim=(0.62,0.97),ylim=(ypos[-1]+1.3,-1.2),xlabel='AUC, ISUP 3–4 versus 1–2');despine(ax,left=True)
    fam=R['benchmark']['family_mean_scotland_highlow_auc']
    ax.text(0.623,ypos[0]-0.8,f"Pathology foundation models · external mean {fam['pathology']:.3f}",fontsize=5.8,color=PATH,ha='left',va='center',fontweight='bold',bbox=dict(fc='white',ec='none',pad=0.6))
    ax.text(0.623,ypos[6]-0.68,f"ImageNet encoders · external mean {fam['imagenet']:.3f}",fontsize=5.8,color=IMG,ha='left',va='center',fontweight='bold',bbox=dict(fc='white',ec='none',pad=0.6))
    ax.plot([],[],'o',color=INK,ms=4,label='Scotland, external (95% CI)');ax.plot([],[],'D',mfc='white',mec=INK,mew=0.7,ms=3.2,ls='none',label='Korea, out-of-fold')
    ax.legend(loc='lower right',bbox_to_anchor=(1.0,0.995),ncol=2,handlelength=0.8,columnspacing=1.2,borderaxespad=0);ax.set_title('Nine frozen encoders')
    # c) confusion matrices with a shared colour bar
    for j,(coh,name) in enumerate((('korea','Korea, out-of-fold'),('scotland','Scotland, external'))):
        ax=fig.add_subplot(bot[j]);cm=np.array(R['encoders'][e][coh]['grading']['confusion'],float);rown=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(rown,cmap='Blues',vmin=0,vmax=1)
        for a in range(4):
            for b in range(4):ax.text(b,a,f'{int(cm[a,b])}',ha='center',va='center',fontsize=6,color='white' if rown[a,b]>0.55 else INK)
        ax.set(xticks=range(4),yticks=range(4),xticklabels=[1,2,3,4],yticklabels=[1,2,3,4],xlabel='Predicted ISUP grade')
        if j==0:ax.set_ylabel('Reference ISUP grade');label(ax,'c',dx=-24)
        g=R['encoders'][e][coh]['grading'];ax.set_title(f"{name}\nQWK {g['qwk']:.2f}",linespacing=1.3)
        for s in ax.spines.values():s.set_visible(False)
        ax.tick_params(length=0)
    ca=fig.add_subplot(bot[2]);cb=fig.colorbar(im,cax=ca);cb.set_label('Row proportion',fontsize=5.8,labelpad=2);cb.ax.tick_params(labelsize=5.5,length=1.5,width=0.4);cb.outline.set_linewidth(0.4)
    ca.set_position([ca.get_position().x0,ca.get_position().y0+0.03,ca.get_position().width,ca.get_position().height-0.06])
    # d) grade 1 versus 2
    ax=fig.add_subplot(rgt[0]);label(ax,'d',dx=-24)
    for coh,cs,col,name in (('korea',D['cases'],KOR,'Korea'),('scotland',D['scot'],SCO,'Scotland')):
        low=[c for c in cs if c['isup']<=2];y=[c['isup']==2 for c in low];s=[D['P'][c['patient_id']][1]/(D['P'][c['patient_id']][0]+D['P'][c['patient_id']][1]) for c in low]
        fx,ty=roc(y,s);g=R['encoders'][e][coh]['grading'];ax.plot(fx,ty,color=col,lw=0.9,label=f"{name} {g['g12_auc']:.3f}")
    ax.plot([0,1],[0,1],ls=(0,(1.5,1.5)),color=FAINT,lw=0.6);ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='1 − specificity',ylabel='Sensitivity',aspect='equal')
    ax.set_xticks([0,0.5,1]);ax.set_yticks([0,0.5,1]);ax.set_title('ISUP 2 versus 1');ax.legend(loc='lower right',handlelength=1.0,title='AUC',fontsize=5.6)
    # e) transfer from Korea to Scotland as a paired slope chart; labels spread so that they keep the order of the Scottish values
    ax=fig.add_subplot(rgt[1]);label(ax,'e',dx=-24)
    ends=sorted([(R['encoders'][k]['scotland']['grading']['highlow_auc'],k) for k in encs],reverse=True);yl=spread([y_ for y_,_ in ends],0.0118)
    for (y_,k),l_ in zip(ends,yl):
        r=R['encoders'][k];col=PATH if r['family']=='pathology' else IMG
        ax.plot([0,1],[r['korea']['grading']['highlow_auc'],y_],'-',color=col,lw=0.8,marker='o',ms=2.8,mec='white',mew=0.3,zorder=2)
        ax.plot([1.04,1.16],[y_,l_],color=FAINT,lw=0.4);ax.text(1.19,l_,ENC_NAME[k],fontsize=5.4,va='center',color=col,fontweight='bold' if k=='phikon-v2' else 'normal')
    ax.set(xlim=(-0.15,2.0),ylim=(0.735,0.93),ylabel='AUC, ISUP 3–4 versus 1–2');ax.set_xticks([0,1]);ax.set_xticklabels(['Korea','Scotland'])
    ax.spines['bottom'].set_bounds(0,1);ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True);ax.set_title('Transfer to Scotland')
    save(fig,'Fig2')

# ------------------------------------------------------------------------------------------------------- Fig 3
def fig3(D):
    R=D['R'];e=D['enc'];cfg=D['cfg'];X=D['X'];fig=plt.figure(figsize=(W,154*MM))
    top=fig.add_gridspec(1,2,wspace=0.3,left=0.085,right=0.985,top=0.965,bottom=0.71)
    for j,(coh,cs,name) in enumerate((('korea',D['cases'],'Korea · ISUP 1–2'),('scotland',D['scot'],'Scotland · ISUP 1–2'))):
        lowg=[c for c in C.primary(cs,cfg) if c['isup']<=2];g=np.array([D['ai'][c['patient_id']] for c in lowg]);thr=np.median(g)
        lg=R['encoders'][e][coh]['prognosis']['low_grade'];assert abs(thr-lg['threshold'])<1e-9
        hi=g>thr;t=np.array([c['time'] for c in lowg]);ev=np.array([c['event'] for c in lowg])
        ax=fig.add_subplot(top[j]);label(ax,'ab'[j],dx=-26)
        km_panel(ax,[('AI grade above median','High',HI,t[hi],ev[hi]),('AI grade below median','Low',LO,t[~hi],ev[~hi])],name,lg['logrank_p'],ylabel=(j==0))
    bot=fig.add_gridspec(1,2,width_ratios=[1.0,0.62],wspace=0.04,left=0.12,right=0.6,top=0.545,bottom=0.33)
    dgs=fig.add_gridspec(1,1,left=0.745,right=0.985,top=0.545,bottom=0.33)
    ax=fig.add_subplot(bot[0]);tx=fig.add_subplot(bot[1],sharey=ax);label(ax,'c',dx=-44)
    rows=[];y=0
    for adj,key in (('Adjusted for ISUP grade','cox_isup_plus_ai'),('Adjusted for Leibovich score','cox_leibovich_plus_ai')):
        rows.append(('head',adj,y));y+=1
        for coh,col in (('korea',KOR),('scotland',SCO)):rows.append((coh,key,y,col));y+=1
        y+=0.4
    for r in rows:
        if r[0]=='head':ax.text(0,r[2],r[1],transform=ax.get_yaxis_transform(),fontsize=6,fontweight='bold',va='center',ha='left',bbox=dict(fc='white',ec='none',pad=0.8),zorder=4);continue
        coh,key,yy,col=r;h=R['encoders'][e][coh]['prognosis'][key]
        ax.plot(h['hr_ai_ci95'],[yy,yy],color=col,lw=0.9,solid_capstyle='butt');ax.plot(h['hr_ai_per_grade'],yy,'s',color=col,ms=3.6,mec='white',mew=0.4,zorder=3)
        ax.text(-0.03,yy,'Korea' if coh=='korea' else 'Scotland',transform=ax.get_yaxis_transform(),fontsize=6,ha='right',va='center',color=col)
        tx.text(0.03,yy,f"{h['hr_ai_per_grade']:.2f} ({h['hr_ai_ci95'][0]:.2f}–{h['hr_ai_ci95'][1]:.2f})",fontsize=6,va='center')
        tx.text(0.66,yy,pfmt(h['p']).replace('$\\mathit{P}$ = ','').replace('$\\mathit{P}$ ',''),fontsize=6,va='center')
    tx.text(0.03,-0.9,'HR (95% CI)',fontsize=6,fontweight='bold',va='center');tx.text(0.66,-0.9,'$\\mathit{P}$',fontsize=6,fontweight='bold',va='center')
    tx.set_xlim(0,1);tx.axis('off');ax.set_ylim(y-0.2,-1.4)
    ax.axvline(1,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.set_xscale('log');ax.set_xlim(0.3,12);ax.set_xticks([0.5,1,2,5,10]);ax.set_xticklabels(['0.5','1','2','5','10'])
    ax.minorticks_off();ax.set_yticks([]);despine(ax,left=True);ax.set_xlabel('HR per AI-grade unit (log scale)');ax.set_title('AI grade in Cox models')
    ax=fig.add_subplot(dgs[0]);label(ax,'d',dx=-58)
    ms=[('ISUP grade','c_isup'),('Fuhrman grade','c_fuhrman'),('AI grade','c_ai_grade'),('Leibovich score','c_leibovich')]
    for i,(nm,k) in enumerate(ms):
        a=R['encoders'][e]['korea']['prognosis'][k];b=R['encoders'][e]['scotland']['prognosis'][k]
        ax.plot([a,b],[i,i],color=FAINT,lw=1.4,solid_capstyle='round',zorder=1)
        ax.plot(a,i,'o',color=KOR,ms=4.2,mec='white',mew=0.5,zorder=3);ax.plot(b,i,'o',color=SCO,ms=4.2,mec='white',mew=0.5,zorder=3)
    ax.set_yticks(range(len(ms)));ax.set_yticklabels([m for m,_ in ms]);ax.set_ylim(len(ms)-0.4,-0.9);despine(ax,left=True)
    for t,(nm,_) in zip(ax.get_yticklabels(),ms):
        if nm=='AI grade':t.set_fontweight('bold')
    ax.set(xlim=(0.55,0.87),xlabel="Harrell's C, recurrence");ax.set_xticks([0.6,0.7,0.8]);ax.grid(axis='x',color=GRID,lw=0.5);ax.set_axisbelow(True)
    ax.plot([],[],'o',color=KOR,ms=4,label='Korea');ax.plot([],[],'o',color=SCO,ms=4,label='Scotland');ax.legend(loc='upper left',ncol=2,handlelength=0.6,columnspacing=0.8,bbox_to_anchor=(0,1.02))
    ax.set_title('Discrimination of recurrence')
    # e) discrimination within ISUP grade; f) ISUP grade 2 alone (exploratory)
    egs=fig.add_gridspec(1,2,wspace=0.95,left=0.12,right=0.9,top=0.205,bottom=0.072)
    ax=fig.add_subplot(egs[0]);label(ax,'e',dx=-44)
    for i,(coh,col,nm) in enumerate((('korea',KOR,'Korea'),('scotland',SCO,'Scotland'))):
        p=R['encoders'][e][coh]['prognosis'];c_,ci=p['c_within_isup_grade'],p['c_within_isup_grade_ci95']
        ax.plot(ci,[i,i],color=col,lw=0.9,solid_capstyle='butt');ax.plot(c_,i,'o',color=col,ms=4,mec='white',mew=0.4,zorder=3)
        ax.text(-0.03,i,nm,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6,color=col)
        ax.text(1.02,i,f"{c_:.3f} ({ci[0]:.3f}–{ci[1]:.3f})",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=5.8)
    ax.axvline(0.5,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.set(xlim=(0.4,0.9),ylim=(1.6,-0.6),xlabel="Harrell's C within ISUP grade");ax.set_yticks([]);despine(ax,left=True)
    ax.set_title('AI grade within ISUP grade')
    ax=fig.add_subplot(egs[1]);label(ax,'f',dx=-44)
    for i,(coh,col,nm) in enumerate((('korea',KOR,'Korea'),('scotland',SCO,'Scotland'))):
        d=X['cohorts'][coh]['isup2'];hr,lo,hi_,pv=d['cox_ai'][0]
        ax.plot([lo,hi_],[i,i],color=col,lw=0.9,solid_capstyle='butt');ax.plot(hr,i,'s',color=col,ms=3.6,mec='white',mew=0.4,zorder=3)
        ax.text(-0.03,i,f"{nm}\n{d['events']}/{d['n']}",transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=5.8,color=col,linespacing=1.15)
        ax.text(1.02,i,f"{hr:.2f} ({lo:.2f}–{hi_:.2f})\n{pfmt(pv)}",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=5.6,linespacing=1.2)
    ax.axvline(1,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.set_xscale('log');ax.set(xlim=(0.3,15),ylim=(1.6,-0.6));ax.set_xticks([0.5,1,2,5,10]);ax.set_xticklabels(['0.5','1','2','5','10'])
    ax.minorticks_off();ax.set_yticks([]);despine(ax,left=True);ax.set_xlabel('HR per AI-grade unit (log scale)');ax.set_title('ISUP grade 2 only (exploratory)')
    save(fig,'Fig3')

# ------------------------------------------------------------------------------------------------------- Fig 4
def fig4(D):
    ex=json.loads((EXPL/'examples.json').read_text())['scotland'];fig=plt.figure(figsize=(W,198*MM))
    outer=fig.add_gridspec(2,1,hspace=0.25,left=0.045,right=0.99,top=0.965,bottom=0.335)
    for r,item in enumerate(ex):
        c=D['byid'][item['case']];m=np.load(EXPL/'maps'/f"{c['stem']}.npz");ph=m['p_high'];at=m['attention'];xy=coords_of(C,c);assert len(xy)==len(ph)
        tag='recurrence at %.0f months'%item['time'] if item['event'] else 'no recurrence, followed %.0f months'%item['time']
        gs=outer[r].subgridspec(1,3,width_ratios=[1,1,1.02],wspace=0.08);rd=reader_for(c)
        ax,_=example_row(fig,gs,rd,xy,ph,at,f"Scotland {item['case'].split(':')[1]} · ISUP grade {item['isup']} · AI grade {item['ai_grade']:.2f} · {tag}");rd.close()
        label(ax,'ab'[r],dx=-13,dy=3.5)
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,1.05,1.25],wspace=0.62,left=0.095,right=0.925,top=0.245,bottom=0.06)
    for j,(coh,cs,name) in enumerate((('korea',D['cases'],'Korea'),('scotland',D['scot'],'Scotland'))):
        ax=fig.add_subplot(gs[j]);vals=[[float(D['feats'][c['patient_id']]['hg_frac']) for c in cs if c['isup']==k] for k in (1,2,3,4)]
        bp=ax.boxplot(vals,positions=[1,2,3,4],widths=0.56,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=0.9),whiskerprops=dict(lw=0.5,color=INK),capprops=dict(lw=0.5,color=INK),boxprops=dict(lw=0.5,ec=INK))
        for patch,col in zip(bp['boxes'],GRADE_C):patch.set_facecolor(col);patch.set_alpha(0.28)
        rng=np.random.default_rng(0)
        for k,v in enumerate(vals,1):ax.plot(k+rng.uniform(-0.17,0.17,len(v)),v,'o',color=GRADE_C[k-1],ms=1.6,alpha=0.85,mew=0)
        ax.set(xticks=[1,2,3,4],ylim=(-0.02,1.19),yticks=[0,0.25,0.5,0.75,1],xlabel='ISUP grade');ax.set_xticklabels(['1','2','3','4']);ax.spines['left'].set_bounds(-0.02,1.0)
        if j==0:ax.set_ylabel('High-grade-like patches\n(fraction of slide)');label(ax,'c',dx=-30)
        ax.set_title(name);ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[2]);label(ax,'d',dx=-30)
    for coh,cs,col,nm in (('korea',D['cases'],KOR,'Korea'),('scotland',D['scot'],SCO,'Scotland')):
        ax.plot([D['ai'][c['patient_id']] for c in cs],[float(D['feats'][c['patient_id']]['hg_frac']) for c in cs],'o',color=col,ms=2.2,alpha=0.75,mew=0,label=f"{nm}, ρ = {D['X']['cohorts'][coh]['hg_frac_vs_ai_spearman']:.2f}")
    ax.set(xlim=(1,4),ylim=(-0.02,1.19),xticks=[1,2,3,4],yticks=[0,0.25,0.5,0.75,1],xlabel='AI grade');ax.set_yticklabels([]);ax.spines['left'].set_bounds(-0.02,1.0)
    ax.legend(loc='upper left',handletextpad=0.1,markerscale=1.5,fontsize=5.4,borderaxespad=0.1,frameon=True,facecolor='white',edgecolor='none',framealpha=1);ax.grid(color=GRID,lw=0.5);ax.set_axisbelow(True);ax.set_title('Against the AI grade')
    ax=fig.add_subplot(gs[3]);label(ax,'e',dx=-30);y=0;ticks=[]
    for grp,key in (('ISUP 1–2','hg_frac_low_grade'),('ISUP 2 only','hg_frac_isup2')):
        ax.text(0.02,y,grp,transform=ax.get_yaxis_transform(),fontsize=5.8,fontweight='bold',va='center',bbox=dict(fc='white',ec='none',pad=0.6),zorder=4);y+=1
        for coh,col,nm in (('korea',KOR,'Korea'),('scotland',SCO,'Scotland')):
            d=D['X']['cohorts'][coh][key];hr,lo,hi_,pv=d['cox_hg_frac_per_sd'][0]
            ax.plot([lo,hi_],[y,y],color=col,lw=0.9,solid_capstyle='butt');ax.plot(hr,y,'s',color=col,ms=3.4,mec='white',mew=0.4,zorder=3)
            ax.text(-0.03,y,nm,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=5.8,color=col)
            ax.text(1.02,y,f"C {d['c_hg_frac']:.2f}",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=5.6);y+=1
        y+=0.3
    ax.axvline(1,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.set_xscale('log');ax.set(xlim=(0.45,2.6),ylim=(y-0.1,-0.7));ax.set_xticks([0.5,1,2]);ax.set_xticklabels(['0.5','1','2'])
    ax.minorticks_off();ax.set_yticks([]);despine(ax,left=True);ax.set_xlabel('HR per s.d. (log scale)');ax.set_title('Recurrence, low-grade tumours')
    save(fig,'Fig4')

# ------------------------------------------------------------------------------------------------------- Fig 6
def fig6(D):
    X=D['X'];cfg=D['cfg'];IF=D['IF'];fig=plt.figure(figsize=(W,206*MM))
    r1=fig.add_gridspec(1,3,width_ratios=[1,1,1.25],wspace=0.5,left=0.085,right=0.93,top=0.955,bottom=0.73)
    r2=fig.add_gridspec(1,3,width_ratios=[1,1,1.12],wspace=0.42,left=0.085,right=0.985,top=0.645,bottom=0.43)
    r3=fig.add_gridspec(1,2,wspace=0.28,left=0.085,right=0.985,top=0.335,bottom=0.12)
    off=0.01  # P21 offset used in the locked Cox model (log(P21% + 0.01)); also the floor of the log axis
    for j,(coh,cs,name) in enumerate((('korea',D['cases'],'Korea'),('scotland',D['scot'],'Scotland'))):
        # a) nuclear area
        sub=[c for c in cs if c['patient_id'] in IF];ax=fig.add_subplot(r1[j]);n=X['cohorts'][coh]['nuclear']['avg_nucleus_area_m']
        for k in (1,2,3,4):
            s_=[c for c in sub if c['isup']==k]
            if s_:ax.plot([D['ai'][c['patient_id']] for c in s_],[float(IF[c['patient_id']]['avg_nucleus_area_m']) for c in s_],'o',color=GRADE_C[k-1],ms=3,mec='white',mew=0.3,label=f'ISUP {k}')
        ax.set(xlabel='AI grade',xlim=(1,3.6),ylim=(43,77),xticks=[1,2,3],yticks=[45,50,55,60,65,70])
        if j==0:ax.set_ylabel('Mean nuclear area (µm²)');label(ax,'a',dx=-26)
        ax.set_title(f'{name} (n = {len(sub)})');ax.grid(color=GRID,lw=0.5);ax.set_axisbelow(True)
        ax.text(0.03,0.97,f"ρ = {n['spearman_ai']:.2f}; within grade {n['within_isup_spearman_ai']:.2f}",transform=ax.transAxes,fontsize=5.8,va='top')
        if j==1:ax.legend(loc='lower right',handletextpad=0.1,borderaxespad=0.2,fontsize=5.4)
        # c) AI grade against P21+/MCM2-, quadrants at the cohort's median AI grade and the published 2% cut-off
        surv=[c for c in C.primary(cs,cfg) if c['patient_id'] in IF];d=X['cohorts'][coh]['if_biomarker'];thr=d['ai_threshold']
        g=np.array([D['ai'][c['patient_id']] for c in surv]);p21=np.array([float(IF[c['patient_id']]['p21_mcm2_percentage']) for c in surv])
        t=np.array([c['time'] for c in surv]);ev=np.array([c['event'] for c in surv]);both=(g>thr)&(p21<2.0)
        ax=fig.add_subplot(r2[j])
        ax.plot(g[ev==0],np.maximum(p21[ev==0],off),'o',mfc='white',mec=MID,mew=0.6,ms=3,label='No recurrence')
        ax.plot(g[ev==1],np.maximum(p21[ev==1],off),'o',color=HI,mec='white',mew=0.3,ms=3.4,label='Recurrence')
        ax.set_yscale('log');ax.axvline(thr,color=INK,lw=0.5,ls=(0,(3,2)));ax.axhline(2.0,color=INK,lw=0.5,ls=(0,(3,2)))
        ax.set(xlim=(1,3.6),ylim=(off*0.7,180),xlabel='AI grade',xticks=[1,2,3]);ax.set_yticks([0.01,0.1,1,10,100]);ax.set_yticklabels(['≤0.01','0.1','1','10','100'])
        for qx,qy,m in ((0.98,0.04,(g>thr)&(p21<2)),(0.98,0.96,(g>thr)&(p21>=2)),(0.02,0.04,(g<=thr)&(p21<2)),(0.02,0.96,(g<=thr)&(p21>=2))):
            ax.text(qx,qy,f"{int(ev[m].sum())}/{int(m.sum())}",transform=ax.transAxes,ha='right' if qx>0.5 else 'left',va='bottom' if qy<0.5 else 'top',fontsize=5.8,
                    fontweight='bold' if m is not None and np.array_equal(m,both) else 'normal',color=HI if np.array_equal(m,both) else INK)
        if j==0:ax.set_ylabel('P21+/MCM2− cells (%)');label(ax,'c',dx=-26)
        ax.set_title(f'{name}: ρ = {d["spearman_ai_p21"]:.2f}')
        if j==1:ax.legend(loc='upper left',bbox_to_anchor=(0.0,0.9),handletextpad=0.1,fontsize=5.4)
        # e) recurrence-free survival for the combination
        ax=fig.add_subplot(r3[j])
        if j==0:label(ax,'e',dx=-26)
        km_panel(ax,[('AI grade high and P21+/MCM2− < 2%','Both',HI,t[both],ev[both]),('All others','Others',LO,t[~both],ev[~both])],f'{name} · AI grade and P21+/MCM2−',
                 d['ai_high_and_p21_low_vs_rest']['p'],ylabel=(j==0),ymin=0.4,p_xy=(0.98,0.9))
    # b) Spearman correlations of the AI grade and ISUP grade with immunofluorescence morphology
    ax=fig.add_subplot(r1[2]);label(ax,'b',dx=-44)
    feats=[('avg_nucleus_area_m','Nuclear area'),('avg_nucleus_perimeter_m','Perimeter'),('avg_cell_area_m','Cell area'),('avg_nucleus_roundness','Roundness')]
    cols=[('spearman_ai','AI'),('within_isup_spearman_ai','AI within\ngrade'),('spearman_isup','ISUP')]
    M=np.array([[X['cohorts'][coh]['nuclear'][f][k] for coh in ('korea','scotland') for k,_ in cols] for f,_ in feats])
    im=ax.imshow(M,cmap='RdBu_r',vmin=-0.7,vmax=0.7,aspect='auto')
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):ax.text(k,i,f'{M[i,k]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if abs(M[i,k])>0.45 else INK)
    ax.set_yticks(range(len(feats)));ax.set_yticklabels([f for _,f in feats]);ax.set_xticks(range(6));ax.set_xticklabels([c for _,c in cols]*2,fontsize=5.2,linespacing=1.0)
    ax.axvline(2.5,color='white',lw=2);ax.tick_params(length=0)
    for s_ in ax.spines.values():s_.set_visible(False)
    for x0,nm in ((1,'Korea'),(4,'Scotland')):ax.text(x0,-0.6,nm,ha='center',va='bottom',fontsize=6,color=KOR if nm=='Korea' else SCO,fontweight='bold')
    ca=ax.inset_axes([1.05,0.12,0.045,0.76]);cb=fig.colorbar(im,cax=ca);cb.set_ticks([-0.5,0,0.5]);cb.ax.tick_params(labelsize=5.2,length=1.5,width=0.4);cb.outline.set_linewidth(0.4)
    cb.set_label('Spearman ρ',fontsize=5.6,labelpad=1.5);ax.set_title('Immunofluorescence morphology',pad=13)
    # d) Cox model with both markers
    ax=fig.add_subplot(r2[2]);label(ax,'d',dx=-44);y=0
    for coh,col,nm in (('korea',KOR,'Korea'),('scotland',SCO,'Scotland')):
        d=X['cohorts'][coh]['if_biomarker'];ax.text(0.02,y,f"{nm} ({d['events']}/{d['n']})",transform=ax.get_yaxis_transform(),fontsize=5.8,fontweight='bold',color=col,va='center',bbox=dict(fc='white',ec='none',pad=0.6),zorder=4);y+=1
        for (hr,lo,hi_,pv),lab in zip(d['cox_ai_plus_logp21'],('AI grade (per unit)','log P21+/MCM2−')):
            ax.plot([lo,hi_],[y,y],color=col,lw=0.9,solid_capstyle='butt');ax.plot(hr,y,'s',color=col,ms=3.4,mec='white',mew=0.4,zorder=3)
            ax.text(-0.03,y,lab,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=5.6)
            ax.text(hr,y+0.36,pfmt(pv),ha='center',va='top',fontsize=5.2,bbox=dict(fc='white',ec='none',pad=0.4));y+=1.35
        y+=0.2
    ax.axvline(1,color=MUTED,lw=0.5,ls=(0,(3,2)));ax.set_xscale('log');ax.set(xlim=(0.25,30),ylim=(y-0.1,-0.7));ax.set_xticks([0.3,1,3,10,30]);ax.set_xticklabels(['0.3','1','3','10','30'])
    ax.minorticks_off();ax.set_yticks([]);despine(ax,left=True);ax.set_xlabel('HR (log scale)');ax.set_title('Both markers in one Cox model')
    save(fig,'Fig6')

if __name__=='__main__':
    D=load();only=sys.argv[1:]
    for f in (fig1,fig2,fig3,fig4,fig6):
        if not only or f.__name__ in only:f(D)
    print('ALL FIGURES DONE',flush=True)
