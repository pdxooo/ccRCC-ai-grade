"""Fig. 7 and Supplementary Fig. 8: the H&E-only recurrence model (surv1) and its pre-specified secondary analyses.
Presentation only: surv1's 50 runs are read through surv1's own gates (evaluate.collect) and HE_z is surv1's
(evaluate.standardise); C indices, P values, calibration and decision curves come from surv1 RESULTS.json and the
secondary SUPP_RESULTS.json. Usage (lab-a6000): HE_PATH_PREFIX=... python make_figures_surv.py"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
S1=(HERE/'..'/'surv_he_clin_v1_20261022').resolve();SUPP=(HERE/'..'/'surv_he_clin_supp_v1_20261022').resolve()
sys.path.insert(0,str(HERE));sys.path.insert(1,str(S1))
from figstyle import *
import common as C
import evaluate as E

def load():
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    R=json.loads((S1/'RESULTS.json').read_text());SR=json.loads((SUPP/'SUPP_RESULTS.json').read_text());Z={}
    for e in cfg['encoders']:
        raw_k={p:float(np.mean([kr[(e,s)][p] for s in cfg['seeds']])) for p in kr[(e,cfg['seeds'][0])]}
        raw_s=[float(np.mean([sc[(e,s,f)][c['patient_id']] for s in cfg['seeds'] for f in cfg['folds']])) for c in scot]
        zk,zs,_=E.standardise(cfg,cases,assign,raw_k,raw_s);Z[e]=(zk,dict(zip([c['patient_id'] for c in scot],zs)))
    return dict(cfg=cfg,prim=C.primary(cases,cfg),scot=scot,R=R,SR=SR,Z=Z)

def calib_panel(ax,cal,title):
    """Observed (Kaplan-Meier, 95% CI) against predicted 36-month risk by risk group; the two models are offset sideways by
    a small fixed amount so that their intervals do not overlap, and no line joins the groups."""
    lim=max(0.05,max(max(q['predicted'],q['observed_ci95'][1]) for m in ('fused','leibovich') for q in cal[m]['groups'])*1.06)
    lim=float(np.ceil(lim*10)/10);dx=lim*0.014
    ax.plot([0,lim],[0,lim],ls=(0,(2,2)),color=FAINT,lw=0.6,zorder=1)
    for m,col,lab,sx,mfc in (('leibovich',MID,'Leibovich',-1,'white'),('fused',HI,'Leibovich + H&E',1,HI)):
        g=cal[m]['groups'];x=[q['predicted']+sx*dx for q in g];y=[q['observed'] for q in g]
        ax.errorbar(x,y,yerr=[[q['observed']-q['observed_ci95'][0] for q in g],[q['observed_ci95'][1]-q['observed'] for q in g]],fmt='o',color=col,mfc=mfc,mec=col,
                    ms=3.2,mew=0.8,elinewidth=0.7,capsize=0,zorder=3,label=f"{lab}: slope {cal[m]['slope']:.2f}, O/E {cal[m]['o_e']:.2f}")
    t=[v for v in (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8) if v<=lim+1e-9]
    ax.set(xlim=(0,lim),ylim=(0,lim),xticks=t,yticks=t,aspect='equal',xlabel='Predicted recurrence by 36 months',ylabel='Observed recurrence (Kaplan–Meier)')
    ax.grid(color=GRID,lw=0.5);ax.set_axisbelow(True);ax.set_title(title,loc='left');ax.set_anchor('N')
    ax.legend(loc='upper left',frameon=False,fontsize=5.4,handletextpad=0.2,borderaxespad=0.3)
def dca_panel(ax,d,title,xmax=0.4):
    """Net benefit for recurrence by 36 months over threshold probabilities up to xmax."""
    T=np.asarray(d['thresholds']);k=T<=xmax+1e-9
    ax.axhline(0,color=INK,lw=0.5,label='Treat none',zorder=1)
    ax.plot(T[k],np.asarray(d['treat_all'])[k],color='#555555',lw=0.7,ls=(0,(3,2)),label='Treat all',zorder=2)
    ax.plot(T[k],np.asarray(d['leibovich'])[k],color=MID,lw=0.9,label='Leibovich',zorder=3)
    ax.plot(T[k],np.asarray(d['fused'])[k],color=HI,lw=0.9,label='Leibovich + H&E',zorder=4)
    top=max(np.max(np.asarray(d['fused'])[k]),np.max(np.asarray(d['leibovich'])[k]),0.02)
    ax.set(xlim=(0,xmax),ylim=(-0.02,top*1.18),xlabel='Threshold probability (36 months)',ylabel='Net benefit')
    ax.set_xticks(np.round(np.arange(0,xmax+1e-9,0.1),1));ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True)
    h,l=ax.get_legend_handles_labels();o=[3,2,1,0];ax.legend([h[i] for i in o],[l[i] for i in o],loc='upper right',frameon=False,fontsize=5.6,handlelength=1.4)
    ax.set_title(title,loc='left')

def fig7(D):
    R=D['R'];SR=D['SR'];fig=plt.figure(figsize=(W,142*MM))
    gs=fig.add_gridspec(2,3,hspace=0.56,wspace=0.42,left=0.08,right=0.985,top=0.955,bottom=0.08)
    for j,(e,nm) in enumerate((('phikon-v2','Phikon-v2'),('uni','UNI'))):
        zk,zs=D['Z'][e];thr=float(np.median([zk[c['patient_id']] for c in D['prim']]))
        hi=np.array([zs[c['patient_id']]>thr for c in D['scot']]);t=np.array([c['time'] for c in D['scot']]);ev=np.array([c['event'] for c in D['scot']])
        lr=R['encoders'][e]['main']['scotland']['logrank_split_at_korean_median'];assert int(hi.sum())==lr['high_n']
        ax=fig.add_subplot(gs[0,j]);label(ax,'ab'[j],dx=-24)
        km_panel(ax,[('High H&E risk','High',HI,t[hi],ev[hi]),('Low H&E risk','Low',LO,t[~hi],ev[~hi])],f"Scotland · {nm}",lr['p'],ylabel=(j==0),p_xy=(0.98,0.04))
    ax=fig.add_subplot(gs[0,2]);label(ax,'c',dx=-24)
    ms=[('Leibovich','c_baseline','c_baseline'),('Leibovich\n+ H&E','c_fused','c_fused_frozen'),('H&E\nalone','c_he_alone','c_he_alone')]
    for k,(e,mk) in enumerate((('phikon-v2','o'),('uni','s'))):
        for coh,col,dx,key in (('korea',KOR,-0.18,1),('scotland',SCO,0.06,2)):
            m=R['encoders'][e]['main'][coh];ax.plot([i+dx+0.12*k for i in range(3)],[m[x[key]] for x in ms],mk,color=col,ms=3.8,mfc=col if k==0 else 'white',ls='none',
                                                    label=f"{'Korea' if coh=='korea' else 'Scotland'}, {ENC_NAME[e]}")
    ax.set_xticks(range(3));ax.set_xticklabels([x[0] for x in ms]);ax.set(ylim=(0.70,0.88),xlim=(-0.5,2.5),ylabel="Harrell's C, recurrence")
    ax.legend(loc='lower left',handletextpad=0.2,fontsize=5.6);ax.set_title('Discrimination of recurrence');ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True)
    s=R['encoders']['phikon-v2']['main']['scotland'];ax.text(0.98,0.97,f"Scotland ΔC (Phikon-v2)\n{s['delta_c']:+.3f} ({s['delta_c_ci95'][0]:+.3f} to {s['delta_c_ci95'][1]:+.3f})",transform=ax.transAxes,ha='right',va='top',fontsize=5.8)
    sp=SR['encoders']['phikon-v2']['scotland']
    ax=fig.add_subplot(gs[1,0]);label(ax,'d',dx=-24);calib_panel(ax,sp['calibration'],'Calibration · Scotland, Phikon-v2')
    ax=fig.add_subplot(gs[1,1]);label(ax,'e',dx=-24);dca_panel(ax,sp['dca'],'Decision curve · Scotland, Phikon-v2')
    ax=fig.add_subplot(gs[1,2]);label(ax,'f',dx=-24)
    xs=[0,1,2.4,3.4];k_=0
    for m in ('36','60'):
        for coh,col in (('korea',KOR),('scotland',SCO)):
            a=SR['encoders']['phikon-v2'][coh]['td_auc'][m];x=xs[k_];k_+=1
            ax.plot([x,x],[a['leibovich'],a['fused']],color=FAINT,lw=1.6,solid_capstyle='round',zorder=1)
            ax.plot(x,a['leibovich'],'o',color=MID,ms=4.2,mec='white',mew=0.5,zorder=3);ax.plot(x,a['fused'],'o',color=col,ms=4.2,mec='white',mew=0.5,zorder=3)
            ax.text(x+0.14,(a['leibovich']+a['fused'])/2,f"{a['delta']:+.3f}",ha='left',va='center',fontsize=5.6)
    ax.set_xticks(xs);ax.set_xticklabels(['Korea','Scotland','Korea','Scotland']);ax.set(xlim=(-0.5,4.0),ylim=(0.83,0.89),ylabel='Time-dependent AUC')
    tr=ax.get_xaxis_transform();ax.text(0.5,-0.085,'36 months',ha='center',va='top',transform=tr,fontsize=6);ax.text(2.9,-0.085,'60 months',ha='center',va='top',transform=tr,fontsize=6)
    ax.grid(axis='y',color=GRID,lw=0.5);ax.set_axisbelow(True)
    ax.plot([],[],'o',color=MID,ms=4,label='Leibovich');ax.plot([],[],'o',color=KOR,ms=4,label='+ H&E, Korea');ax.plot([],[],'o',color=SCO,ms=4,label='+ H&E, Scotland')
    ax.legend(loc='upper right',handlelength=0.6)
    ax.set_title('Time-dependent AUC · Phikon-v2')
    save(fig,'Fig7')

def suppfig8(D):
    SR=D['SR'];fig=plt.figure(figsize=(W,175*MM));gs=fig.add_gridspec(3,3,hspace=0.55,wspace=0.55,left=0.08,right=0.98,top=0.95,bottom=0.07)
    combos=[('phikon-v2','korea','Korea, Phikon-v2'),('uni','korea','Korea, UNI'),('uni','scotland','Scotland, UNI')]
    for r,(e,coh,nm) in enumerate(combos):
        b=SR['encoders'][e][coh]
        ax=fig.add_subplot(gs[r,0]);label(ax,'abc'[r]);calib_panel(ax,b['calibration'],f'Calibration, {nm}')
        ax=fig.add_subplot(gs[r,1]);dca_panel(ax,b['dca'],f'Decision curve, {nm}')
        ax=fig.add_subplot(gs[r,2]);u=b['uno_c'];a36=b['td_auc']['36'];a60=b['td_auc']['60']
        rows=[('Uno C (60 mo)',u['leibovich'],u['fused'],u['delta'],u['delta_ci95']),('AUC 36 mo',a36['leibovich'],a36['fused'],a36['delta'],a36['delta_ci95']),
              ('AUC 60 mo',a60['leibovich'],a60['fused'],a60['delta'],a60['delta_ci95'])]
        for i,(lab,l0,l1,dl,ci) in enumerate(rows):
            y=2-i;ax.plot([l0,l1],[y,y],color='#CCCCCC',lw=1.5);ax.plot(l0,y,'o',color=MID,ms=3.5);ax.plot(l1,y,'o',color=HI,ms=3.5)
            ax.text(0.995,y+0.22,f"Δ {dl:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})",transform=ax.get_yaxis_transform(),ha='right',fontsize=5.6)
        ax.set_yticks([2,1,0]);ax.set_yticklabels([r_[0] for r_ in rows]);ax.set(xlim=(0.65,0.95),ylim=(-0.6,2.7),xlabel='Value (grey: Leibovich; red: + H&E)')
        ax.set_title(f'Discrimination, {nm}',loc='left')
    save(fig,'SuppFig8')

if __name__=='__main__':
    D=load();fig7(D);suppfig8(D);print('SURV FIGURES DONE',flush=True)
