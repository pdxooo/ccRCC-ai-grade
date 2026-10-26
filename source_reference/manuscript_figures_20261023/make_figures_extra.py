"""Fig. 5 and Supplementary Figs 1-7, 9 and 10 (grading study). Presentation only: gx1's 225 runs are read through gx1's own
gates (evaluate.collect); per-seed values, C indices and hazard ratios come from gx1 RESULTS.json; patch maps from the
exploratory analysis; the negative studies from their own RESULTS.json files. Descriptive curves added here (AI-grade
tertiles, grade distributions) carry no P values. Usage (lab-a6000): HE_PATH_PREFIX=... python make_figures_extra.py"""
import sys,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;CODE=(HERE/'..').resolve()
GX1=CODE/'grade_ext_v1_20261022';EXPL=CODE/'gx1_explain_v1_20261023'
sys.path.insert(0,str(HERE));sys.path.insert(1,str(GX1))
from figstyle import *
from matplotlib.patches import FancyBboxPatch
import common as C
import evaluate as E
sys.path.insert(0,str(C.ROOT/'qc_deps'));sys.path.insert(0,str(C.ROOT/'bib_phase1'))
PFX=str(C.ROOT).split('/work/home')[0]

def load():
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    S=cfg['seeds'];F=cfg['folds'];P={}
    for e in cfg['encoders']:
        P[e]={p:np.mean([kr[(e,s)][p] for s in S],0) for p in kr[(e,S[0])]}
        P[e].update({c['patient_id']:np.mean([sc[(e,s,f)][c['patient_id']] for s in S for f in F],0) for c in scot})
    ai={p:float(C.grade_score(v[None])[0]) for p,v in P['phikon-v2'].items()}
    R=json.loads((GX1/'RESULTS.json').read_text())
    return dict(cfg=cfg,cases=cases,scot=scot,P=P,ai=ai,R=R,byid={c['patient_id']:c for c in cases+scot})

# ---------------------------------------------------------------------------------------- Fig 5: morphology atlas
def fig5(D):
    from full_patch_qc import IndexedReader
    from PIL import Image
    pool=[c for c in D['cases']+D['scot'] if (C.ROOT if c['cohort']=='korea' else C.SROOT).joinpath('data/he',c['stem']+'.czi').exists()]
    rng=np.random.default_rng(20261023);order=list(rng.permutation(len(pool)))
    bins=[(1.0,1.5),(1.5,2.0),(2.0,2.5),(2.5,3.0),(3.0,4.01)];ncol=6;picked={b:[] for b in bins};maps={}
    for i in order:
        c=pool[i];m=np.load(EXPL/'maps'/f"{c['stem']}.npz");eg=m['egrade'];at=m['attention'];top=at>=np.quantile(at,0.8)
        for b in reversed(bins):  # scarce high-grade rows first
            if len(picked[b])>=ncol or any(x[0]['patient_id']==c['patient_id'] for v in picked.values() for x in v):continue
            k=np.flatnonzero(top&(eg>=b[0])&(eg<b[1]))
            if len(k):j=k[np.argmax(at[k])];picked[b].append((c,j,float(eg[j])));break
        if all(len(v)>=ncol for v in picked.values()):break
    BIN_C=['#24589E','#3B8FCB','#B4B4B4','#D46F2A','#A50F15']  # patch-grade bins; validated (all pairs: CVD dE >= 14.9, normal >= 16.5)
    fig=plt.figure(figsize=(W,214*MM));gA=fig.add_gridspec(len(bins),ncol,left=0.1,right=0.995,top=0.975,bottom=0.255,wspace=0.04,hspace=0.18)
    axs=np.array([[fig.add_subplot(gA[r,k]) for k in range(ncol)] for r in range(len(bins))])
    fig.text(0.008,0.978,'a)',fontsize=8,fontweight='bold',va='bottom')
    for r,b in enumerate(bins):
        axs[r,0].text(-0.12,0.5,f"Patch grade\n{b[0]:.1f}–{min(b[1],4):.1f}",transform=axs[r,0].transAxes,ha='right',va='center',fontsize=6.5,fontweight='bold')
        for k in range(ncol):
            ax=axs[r,k];ax.set_xticks([]);ax.set_yticks([])
            if k>=len(picked[b]):ax.axis('off');continue
            c,j,eg=picked[b][k];xy=coords_of(C,c);s_,x,y,w,h=[int(v) for v in xy[j]];rd=IndexedReader(str((C.ROOT if c['cohort']=='korea' else C.SROOT)/'data/he'/(c['stem']+'.czi')))
            rgb,_,_=rd.region(s_,x,y,w,h);rd.close();ax.imshow(np.asarray(Image.fromarray(rgb).resize((448,448),Image.Resampling.BICUBIC)))
            for sp in ax.spines.values():sp.set_visible(True);sp.set_color(BIN_C[r]);sp.set_linewidth(1.2)
            ax.set_title(f"{'KR' if c['cohort']=='korea' else 'SC'} {c['patient_id'].split(':')[1].replace('KR','')} · ISUP {c['isup']} · {eg:.2f}",fontsize=5.6)
            if r==len(bins)-1 and k==ncol-1:L=int(50/128*448);ax.plot([448-14-L,448-14],[428,428],color='black',lw=1.4);ax.text(448-14-L/2,416,'50 µm',ha='center',fontsize=5.6,path_effects=HALO)
    # b, c) composition of each slide's patches by patch grade, averaged within ISUP grade (all slides, unweighted)
    edges=[1.0,1.5,2.0,2.5,3.0,4.01];gB=fig.add_gridspec(1,2,left=0.125,right=0.84,top=0.19,bottom=0.055,wspace=0.32)
    for j,(cs,nm,col) in enumerate(((D['cases'],'Korea',KOR),(D['scot'],'Scotland',SCO))):
        ax=fig.add_subplot(gB[j]);fr={k:[] for k in (1,2,3,4)}
        for c in cs:
            eg=np.load(EXPL/'maps'/f"{c['stem']}.npz")['egrade'];fr[c['isup']].append(np.histogram(np.clip(eg,1,4),bins=edges)[0]/len(eg))
        for k in (1,2,3,4):
            m=np.mean(fr[k],0);left=0
            for b_,v in enumerate(m):
                ax.barh(k,v,left=left,height=0.68,color=BIN_C[b_],edgecolor='white',linewidth=0.6)
                if v>=0.07:ax.text(left+v/2,k,f'{v*100:.0f}',ha='center',va='center',fontsize=5.4,color='white' if b_ in (0,1,4) else INK)
                left+=v
            print(f'FIG5 {nm} ISUP{k} n={len(fr[k])} '+' '.join(f'{v*100:.1f}' for v in m),flush=True)
        ax.set_yticks([1,2,3,4]);ax.set_yticklabels([f'ISUP {k} ({len(fr[k])})' for k in (1,2,3,4)]);ax.invert_yaxis()
        ax.set(xlim=(0,1),xlabel='Fraction of the slide\'s patches');ax.set_xticks([0,0.25,0.5,0.75,1]);ax.set_xticklabels(['0','25%','50%','75%','100%'])
        despine(ax,left=True);ax.tick_params(axis='y',length=0);ax.set_title(nm,color=col,fontweight='bold')
        label(ax,'bc'[j],dx=-48)
    lg=[plt.Rectangle((0,0),1,1,fc=c_,ec='none') for c_ in BIN_C]
    fig.legend(lg,[f'{b[0]:.1f}–{min(b[1],4):.1f}' for b in bins],title='Patch grade',loc='center left',bbox_to_anchor=(0.855,0.122),handlelength=1.0,handleheight=0.9,fontsize=5.6,title_fontsize=5.8,alignment='left')
    save(fig,'Fig5')

# --------------------------------------------------------------------------------------- Supplementary figures
def s1(D):
    fig=plt.figure(figsize=(W,92*MM));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,180);ax.set_ylim(0,92);ax.axis('off')
    def box(x,y,w,h,t,fc='#F4F4F4',ec='#444444'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.5',fc=fc,ec=ec,lw=0.7));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=6.3,linespacing=1.3)
    def arr(x0,y0,x1,y1):ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=0.7,color='#444444',mutation_scale=7))
    for x0,col,lab,n,nsurv,excl,nif,feat in ((6,KOR,'Korea (S-BIAD2452)',300,295,'5 with a recorded\ndisease-free interval of 0',71,'9 encoders extracted on A800'),
                                             (96,SCO,'Scotland (S-BIAD2454)',120,120,'none excluded',88,'95 slides on A800, 25 on RTX A6000\n(parity: cosine ≥ 0.99965)')):
        box(x0,76,78,12,f'{lab}\n{n} patients, one H&E slide each',fc='#E8F1F8' if col==KOR else '#FBEDE5',ec=col)
        box(x0,57,78,12,f'Technical QC patch pool (128 µm)\n{feat}');arr(x0+39,75.6,x0+39,69.4)
        box(x0,36,36,13,f'Grading analyses\n{n} patients');box(x0+42,36,36,13,f'Recurrence analyses\n{nsurv} patients');arr(x0+18,56.6,x0+18,49.4);arr(x0+60,56.6,x0+60,49.4)
        box(x0+42,15,36,13,f'Excluded from recurrence:\n{excl}',fc='white');arr(x0+60,35.6,x0+60,28.4)
        box(x0,15,36,13,f'Immunofluorescence subset\n{nif} patients',fc='white');arr(x0+18,35.6,x0+18,28.4)
    ax.text(6,6,'Korea: 5-fold locked cross-validation for development. Scotland: external validation only. Recurrence censored at 84 months.',fontsize=6,color='#333333')
    save(fig,'SuppFig1')

def s2(D):
    R=D['R'];fig,axs=plt.subplots(1,2,figsize=(W,88*MM));plt.subplots_adjust(left=0.07,right=0.98,top=0.9,bottom=0.13,wspace=0.25)
    for ax,(coh,cs,nm) in zip(axs,(('korea',D['cases'],'Korea, out-of-fold'),('scotland',D['scot'],'Scotland, external'))):
        y=[c['isup']>=3 for c in cs]
        for e in sorted(R['encoders'],key=lambda k:-R['encoders'][k][coh]['grading']['highlow_auc']):
            s=[D['P'][e][c['patient_id']][2]+D['P'][e][c['patient_id']][3] for c in cs];fx,ty=roc(y,s)
            ax.plot(fx,ty,color=ENC_C[e],lw=1.0 if e=='phikon-v2' else 0.7,ls='-' if R['encoders'][e]['family']=='pathology' else '--',
                    label=f"{ENC_NAME[e]} {R['encoders'][e][coh]['grading']['highlow_auc']:.3f}")
        ax.plot([0,1],[0,1],ls=':',color='#999999',lw=0.6);ax.set(xlim=(0,1),ylim=(0,1.01),aspect='equal',xlabel='1 − specificity',ylabel='Sensitivity')
        ax.set_title(f'{nm}: ISUP 3–4 vs 1–2',loc='left');ax.legend(loc='lower right',frameon=False,fontsize=5.8,title='AUC',title_fontsize=6)
    label(axs[0],'a',x=-0.12);label(axs[1],'b',x=-0.12);save(fig,'SuppFig2')

def s3(D):
    R=D['R'];encs=sorted(R['encoders'],key=lambda k:(R['encoders'][k]['family']!='pathology',-R['encoders'][k]['scotland']['grading']['highlow_auc']))
    fig,axs=plt.subplots(1,3,figsize=(W,76*MM));plt.subplots_adjust(left=0.07,right=0.99,top=0.89,bottom=0.33,wspace=0.32)
    for ax,(key,ens,lab) in zip(axs,(('scotland_highlow_auc',lambda r:r['scotland']['grading']['highlow_auc'],'AUC, ISUP 3–4 vs 1–2'),
                                     ('scotland_g12_auc',lambda r:r['scotland']['grading']['g12_auc'],'AUC, grade 1 vs 2'),
                                     ('scotland_c_ai_grade',lambda r:r['scotland']['prognosis']['c_ai_grade'],"Harrell's C, AI grade"))):
        for i,e in enumerate(encs):
            r=R['encoders'][e];v=r['per_seed'][key];ax.plot(i+np.linspace(-0.18,0.18,len(v)),v,'o',color=ENC_C[e],ms=2.6,alpha=0.8,mew=0)
            ax.plot([i-0.28,i+0.28],[ens(r)]*2,color='black',lw=1.0)
        ax.set_xticks(range(len(encs)));ax.set_xticklabels([ENC_NAME[e] for e in encs],rotation=45,ha='right');ax.set_ylabel(lab);ax.set_title(f'Scotland: {lab}',loc='left',fontsize=6.8)
    for ax,s in zip(axs,'abc'):label(ax,s,x=-0.22)
    fig.text(0.07,0.01,'Dots: the five seeds (each the mean of its five fold models); bar: the 25-model ensemble used in the main analysis.',fontsize=6,color='#333333')
    save(fig,'SuppFig3')

def s4(D):
    fig,axs=plt.subplots(1,2,figsize=(W,70*MM));plt.subplots_adjust(left=0.08,right=0.98,top=0.88,bottom=0.2,wspace=0.25)
    for ax,(cs,nm) in zip(axs,((D['cases'],'Korea, out-of-fold'),(D['scot'],'Scotland, external'))):
        vals=[[D['ai'][c['patient_id']] for c in cs if c['isup']==k] for k in (1,2,3,4)]
        bp=ax.boxplot(vals,positions=[1,2,3,4],widths=0.5,showfliers=False,patch_artist=True,medianprops=dict(color='black',lw=0.9),whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6),boxprops=dict(lw=0.6))
        for p_,col in zip(bp['boxes'],GRADE_C):p_.set_facecolor(col);p_.set_alpha(0.35)
        rng=np.random.default_rng(1)
        for k,v in enumerate(vals,1):ax.plot(k+rng.uniform(-0.15,0.15,len(v)),v,'o',color=GRADE_C[k-1],ms=1.8,alpha=0.7,mew=0)
        ax.set(xticks=[1,2,3,4],ylim=(1,4),ylabel='AI grade');ax.set_xticklabels([f'ISUP {k}\n(n = {len(v)})' for k,v in enumerate(vals,1)]);ax.set_title(nm,loc='left')
        for g in (1.5,2.5,3.5):ax.axhline(g,color='#EEEEEE',lw=0.6,zorder=0)
    label(axs[0],'a',x=-0.12);label(axs[1],'b',x=-0.12);save(fig,'SuppFig4')

def s5(D):
    cfg=D['cfg'];fig=plt.figure(figsize=(W,156*MM));gs=fig.add_gridspec(2,2,hspace=0.8,wspace=0.28,left=0.1,right=0.985,top=0.955,bottom=0.165)
    tcols=[LO,'#7F7F7F',HI]
    for j,(cs,nm) in enumerate(((D['cases'],'Korea'),(D['scot'],'Scotland'))):
        surv=C.primary(cs,cfg);g=np.array([D['ai'][c['patient_id']] for c in surv]);t=np.array([c['time'] for c in surv]);ev=np.array([c['event'] for c in surv])
        q=np.quantile(g,[1/3,2/3]);grp=np.digitize(g,q)
        ax=fig.add_subplot(gs[0,j]);label(ax,'ab'[j],dx=-26)
        km_panel(ax,[(f'AI grade tertile {k+1}',f'T{k+1}',tcols[k],t[grp==k],ev[grp==k]) for k in (2,1,0)],f'{nm} · AI grade tertiles (all grades)',ylabel=(j==0),ymin=0.4)
        isup=np.array([min(c['isup'],3) for c in surv])
        ax=fig.add_subplot(gs[1,j]);label(ax,'cd'[j],dx=-26)
        km_panel(ax,[(lab,sh,col,t[isup==k],ev[isup==k]) for k,lab,sh,col in ((3,'ISUP 3–4','3–4',HI),(2,'ISUP 2','2','#7F7F7F'),(1,'ISUP 1','1',LO))],f'{nm} · pathologist ISUP grade',ylabel=(j==0),ymin=0.4)
    fig.text(0.1,0.008,'Descriptive curves; tertile cut-points from each cohort’s own AI-grade distribution; no tests.',fontsize=5.8,color=MUTED)
    save(fig,'SuppFig5')

def s6(D):
    R=D['R'];encs=sorted(R['encoders'],key=lambda k:(R['encoders'][k]['family']!='pathology',-R['encoders'][k]['scotland']['grading']['highlow_auc']))
    fig,axs=plt.subplots(1,2,figsize=(W,78*MM));plt.subplots_adjust(left=0.08,right=0.98,top=0.88,bottom=0.24,wspace=0.3)
    ax=axs[0]
    for coh,col,dx in (('korea',KOR,-0.12),('scotland',SCO,0.12)):
        ax.plot([i+dx for i in range(len(encs))],[R['encoders'][e][coh]['prognosis']['c_ai_grade'] for e in encs],'o',color=col,ms=3.5,label=f"{'Korea' if coh=='korea' else 'Scotland'}: AI grade")
        ax.axhline(R['encoders'][encs[0]][coh]['prognosis']['c_isup'],color=col,ls='--',lw=0.7,label=f"{'Korea' if coh=='korea' else 'Scotland'}: ISUP grade")
    ax.set_xticks(range(len(encs)));ax.set_xticklabels([ENC_NAME[e] for e in encs],rotation=45,ha='right');ax.set(ylabel="Harrell's C, recurrence",ylim=(0.55,0.85))
    ax.legend(frameon=False,fontsize=5.8,loc='upper right',ncol=2);ax.set_title('AI grade by encoder',loc='left');label(ax,'a',x=-0.12)
    ax=axs[1]
    for i,e in enumerate(encs):
        for coh,col,dy in (('korea',KOR,-0.15),('scotland',SCO,0.15)):
            h=R['encoders'][e][coh]['prognosis']['cox_isup_plus_ai']
            if h['ok']:ax.errorbar(h['hr_ai_per_grade'],len(encs)-1-i+dy,xerr=[[h['hr_ai_per_grade']-h['hr_ai_ci95'][0]],[h['hr_ai_ci95'][1]-h['hr_ai_per_grade']]],fmt='s',color=col,ms=3,capsize=0,lw=0.7)
    ax.axvline(1,color='#666666',ls='--',lw=0.6);ax.set_xscale('log');ax.set_xticks([0.5,1,2,5,10,20,50,100]);ax.set_xticklabels(['0.5','1','2','5','10','20','50','100']);ax.minorticks_off()
    ax.set_yticks(range(len(encs)));ax.set_yticklabels([ENC_NAME[e] for e in encs][::-1]);ax.set_xlabel('HR per AI-grade unit, adjusted for ISUP grade')
    ax.plot([],[],'s',color=KOR,ms=3,label='Korea');ax.plot([],[],'s',color=SCO,ms=3,label='Scotland');ax.legend(frameon=False,fontsize=5.8,loc='lower right')
    ax.set_title('AI grade beyond ISUP, by encoder',loc='left');label(ax,'b',x=-0.25);save(fig,'SuppFig6')

def s10(D):  # cross-platform parity
    stem='iCAIRD -1076-HE-St Andrews-10_06_2021';a6=Path(os.environ['HE_PROJECT_DIR'])/'scratch/parity_a6000'/stem;hpc=C.SFULL/stem
    encs=['phikon-v2','uni','uni2-h','virchow2','gigapath','conch','resnet50','convnext_tiny','vit_b_16'];vals=[]
    for e in encs:
        x=np.load(a6/f'{e}.npy').astype(np.float64);y=np.load(hpc/f'{e}.npy').astype(np.float64)
        cos=(x*y).sum(1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1);vals.append(np.clip(1-cos,1e-9,None))
    fig,ax=plt.subplots(figsize=(W*0.75,70*MM));plt.subplots_adjust(left=0.12,right=0.98,top=0.88,bottom=0.25)
    bp=ax.boxplot(vals,widths=0.55,showfliers=True,flierprops=dict(marker='.',ms=1.5,alpha=0.4),medianprops=dict(color='black'),patch_artist=True)
    for p_,e in zip(bp['boxes'],encs):p_.set_facecolor(ENC_C[e]);p_.set_alpha(0.45)
    ax.set_yscale('log');ax.axhline(1e-3,color=HI,ls='--',lw=0.7);ax.text(9.4,1.15e-3,'acceptance: cosine ≥ 0.999',ha='right',fontsize=6,color=HI)
    ax.set_xticks(range(1,len(encs)+1));ax.set_xticklabels([ENC_NAME[e] for e in encs],rotation=45,ha='right');ax.set_ylabel('1 − cosine similarity per patch')
    ax.set_title(f'Cross-platform parity, Scotland 1076 ({len(vals[0]):,} patches): RTX A6000 vs A800',loc='left')
    fig.text(0.12,0.01,'Values below 10$^{-9}$ (including identical vectors) are drawn at 10$^{-9}$.',fontsize=6,color='#333333');save(fig,'SuppFig10')

def s9(D):  # negative studies
    fx=json.loads((CODE/'fuhrman_aux_v1_20261022/RESULTS.json').read_text());hr=json.loads((CODE/'highres_stage1_train_v1_20261021/RESULTS.json').read_text())
    fu=json.loads((CODE/'fusion_gain_v2_20261021/RESULTS.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(W,68*MM),gridspec_kw=dict(width_ratios=[1.1,1.1,0.95]));plt.subplots_adjust(left=0.07,right=0.985,top=0.89,bottom=0.2,wspace=0.7)
    ax=axs[0];armc={'A':MID,'B':HI,'C':LO}
    for k,e in enumerate(('phikon-v2','uni')):
        for j,a in enumerate(('A','B','C')):
            r=fx['encoders'][e]['arms'][a];x=k*3.6+j;v=r['g12_auc_per_seed'];ax.plot(x+np.linspace(-0.2,0.2,len(v)),v,'o',color=armc[a],ms=2.5,mew=0)
            ax.plot([x-0.3,x+0.3],[r['g12_auc']]*2,color='black',lw=1)
    ax.set_xticks([0,1,2,3.6,4.6,5.6]);ax.set_xticklabels(['A','B','C']*2);ax.set(ylabel='AUC, grade 1 vs 2 (Korea, out-of-fold)')
    tr=ax.get_xaxis_transform();ax.text(1,-0.17,'Phikon-v2',ha='center',va='top',fontsize=6.5,fontweight='bold',transform=tr);ax.text(4.6,-0.17,'UNI',ha='center',va='top',fontsize=6.5,fontweight='bold',transform=tr)
    ax.set_title('Auxiliary Fuhrman supervision',loc='left',fontsize=6.8);label(ax,'a',x=-0.25)
    ax=axs[1];armh={'L':MID,'H':HI,'Hd':'#E69F00','Lfull':LO}
    for k,e in enumerate(('uni','virchow2')):
        for j,a in enumerate(('L','H','Hd','Lfull')):
            r=hr['encoders'][e]['arms'][a];x=k*4.8+j;v=r['auc_per_seed'];ax.plot(x+np.linspace(-0.2,0.2,len(v)),v,'o',color=armh[a],ms=2.5,mew=0)
            ax.plot([x-0.3,x+0.3],[r['auc_seed_averaged']]*2,color='black',lw=1)
    ax.set_xticks([0,1,2,3,4.8,5.8,6.8,7.8]);ax.set_xticklabels(['L','H','Hd','Lfull']*2,fontsize=6);ax.set(ylabel=f"AUC, grade 1 vs 2 ({hr['n_cases']} Korean cases)")
    tr=ax.get_xaxis_transform();ax.text(1.5,-0.17,'UNI',ha='center',va='top',fontsize=6.5,fontweight='bold',transform=tr);ax.text(6.3,-0.17,'Virchow2',ha='center',va='top',fontsize=6.5,fontweight='bold',transform=tr)
    ax.set_title('Patch magnification',loc='left',fontsize=6.8);label(ax,'b',x=-0.25)
    ax=axs[2];P=fu['preregistered'];rows=[('P1: 4 encoders vs\nPhikon-v2 × 4 seeds','P1_encoder_vs_seed_diversity'),('P2: 16-model late\nfusion vs Phikon-v2','P2_full_late_fusion'),('P3: learned fusion\nvs Phikon-v2','P3_learned_fusion')]
    for i,(lab,k) in enumerate(rows):
        d=P[k]['auc_high_low'];y=len(rows)-1-i;ax.errorbar(d['diff'],y,xerr=[[d['diff']-d['ci95'][0]],[d['ci95'][1]-d['diff']]],fmt='s',color=MID,ms=3.5,capsize=1.5,lw=0.8)
    ax.axvline(0,color='#666666',ls='--',lw=0.6);ax.set_yticks(range(len(rows)));ax.set_yticklabels([r_[0] for r_ in rows][::-1],fontsize=6)
    ax.set_xlabel('Δ AUC, ISUP 3–4 vs 1–2 (Korea)');ax.set_title('Encoder fusion',loc='left',fontsize=6.5);label(ax,'c',x=-0.6)
    save(fig,'SuppFig9')

def s7(D):  # Korean examples, in the Fig. 4 layout
    from full_patch_qc import IndexedReader
    ex=json.loads((EXPL/'examples.json').read_text())['korea'];fig=plt.figure(figsize=(W,136*MM))
    outer=fig.add_gridspec(2,1,hspace=0.24,left=0.045,right=0.99,top=0.955,bottom=0.085)
    for r,item in enumerate(ex):
        c=D['byid'][item['case']];m=np.load(EXPL/'maps'/f"{c['stem']}.npz");ph=m['p_high'];at=m['attention'];xy=coords_of(C,c)
        tag='recurrence at %.0f months'%item['time'] if item['event'] else 'no recurrence, followed %.0f months'%item['time']
        gs=outer[r].subgridspec(1,3,width_ratios=[1,1,1.02],wspace=0.08);rd=IndexedReader(str(C.ROOT/'data/he'/(c['stem']+'.czi')))
        ax,_=example_row(fig,gs,rd,xy,ph,at,f"Korea {item['case'].split(':')[1]} · ISUP grade {item['isup']} · AI grade {item['ai_grade']:.2f} · {tag}");rd.close()
        label(ax,'ab'[r],dx=-13,dy=3.5)
    save(fig,'SuppFig7')

if __name__=='__main__':
    D=load()
    for f in (fig5,s1,s2,s3,s4,s5,s6,s7,s9,s10):f(D)
    print('EXTRA FIGURES DONE',flush=True)
