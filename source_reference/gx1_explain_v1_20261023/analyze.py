"""Exploratory analyses that follow the locked gx1 study (PROTOCOL.md): ISUP grade 2 focus, patch-level interpretation,
nuclear morphology and the P21+/MCM2- immunofluorescence biomarker. Reads gx1's 225 runs through gx1's own gates
(evaluate.collect), case_features.csv from patch_maps.py and the immunofluorescence tables (inputs/, the same files as
surv1's). Primary encoder only (Phikon-v2). Writes EXPLAIN_RESULTS.json, EXPLAIN_RESULTS.md, figures/*.png.
Usage (lab-a6000): HE_PATH_PREFIX=... python analyze.py | python analyze.py lock-write | python analyze.py lock-verify"""
import sys,json,csv,math,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
GX1=(HERE/CFG['gx1_dir']).resolve();sys.path.insert(0,str(GX1))
import common as C  # gx1's locked modules
import evaluate as E
import numpy as np
LOCKED=['PROTOCOL.md','config.json','patch_maps.py','analyze.py','tiles.py','inputs/korea_if.csv','inputs/scotland_if.csv',
        'case_features.csv','PATCH_MAPS_DONE.json']  # the last two: patch_maps.py output (no outcome read), frozen at lock time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock_status():
    lk=C.read_json(HERE/'LOCK.json')
    if not lk:return False,['LOCK.json missing']
    bad=[n for n,h in lk['files'].items() if not (HERE/n).exists() or sha(HERE/n)!=h];return not bad,bad

def cox1(x,t,e,tag):
    """Univariable or multivariable Cox: list of (HR, lo, hi, p) per column, or None if the fit is unusable."""
    b,se,ll,ok=E.cox_safe(np.asarray(x,float).reshape(len(t),-1),t,e,tag)
    if not ok:return None
    return [[float(np.exp(b[j])),float(np.exp(b[j]-1.96*se[j])),float(np.exp(b[j]+1.96*se[j])),float(2*C.norm_sf(abs(b[j]/se[j])))] for j in range(len(b))]
def km(t,e):
    t=np.asarray(t,float);e=np.asarray(e,int);xs=[0.];ys=[1.];s=1.
    for u in np.unique(t[e==1]):
        n=(t>=u).sum();d=((t==u)&(e==1)).sum();s*=1-d/n;xs.append(float(u));ys.append(float(s))
    return xs,ys
def split_block(t,e,score,thr,ge=False):
    t=np.asarray(t,float);e=np.asarray(e,int);hi=np.asarray(score)>=thr if ge else np.asarray(score)>thr
    out=dict(threshold=float(thr),high_n=int(hi.sum()),high_events=int(e[hi].sum()),low_n=int((~hi).sum()),low_events=int(e[~hi].sum()))
    if hi.any() and (~hi).any() and e.sum()>0:chi,p=C.logrank(t,e,hi);out.update(logrank_chi2=chi,logrank_p=p)
    return out
def partial_spearman(x,y,strata):
    """Within-stratum rank correlation: ranks normalised to (rank - 0.5) / n inside each stratum (strata with >= 3
    cases), centred, then pooled, so each case weighs the same whatever its stratum size. Returns (rho, used strata)."""
    x=np.asarray(x,float);y=np.asarray(y,float);g=np.asarray(strata);rx=np.zeros(len(x));ry=np.zeros(len(y));used={}
    for v in np.unique(g):
        m=g==v
        if m.sum()<3:continue
        n=m.sum();rx[m]=(C.avg_rank(x[m])-.5)/n;ry[m]=(C.avg_rank(y[m])-.5)/n;rx[m]-=rx[m].mean();ry[m]-=ry[m].mean();used[int(v)]=int(n)
    den=math.sqrt((rx*rx).sum()*(ry*ry).sum())
    return (float((rx*ry).sum()/den) if den>0 else None),used
def finite(o):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(o,dict):return {k:finite(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [finite(v) for v in o]
    if isinstance(o,(float,np.floating)):return float(o) if np.isfinite(o) else None
    return o

def run():
    ok,bad=lock_status();assert ok,f'not the locked analysis code or patch-map output: {bad[:5]}'
    ok,bad=C.lock_status();assert ok,f'gx1 is not its locked version: {bad[:5]}'
    for n in ('korea_if.csv','scotland_if.csv'):assert sha(HERE/'inputs'/n)==CFG['if_sha256'][n],n
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,_=E.collect(cfg,cases,assign,scot);assert not problems,problems[:5]
    enc=CFG['encoder'];seeds=cfg['seeds'];folds=cfg['folds']
    Pk={p:np.mean([kr[(enc,s)][p] for s in seeds],0) for p in kr[(enc,seeds[0])]}
    Ps={c['patient_id']:np.mean([sc[(enc,s,f)][c['patient_id']] for s in seeds for f in folds],0) for c in scot}
    ai={**{p:float(C.grade_score(v[None])[0]) for p,v in Pk.items()},**{p:float(C.grade_score(v[None])[0]) for p,v in Ps.items()}}
    feats={r['patient_id']:r for r in C.read_csv(HERE/'case_features.csv')}
    IF={**{r['patient_id']:r for r in C.read_csv(HERE/'inputs/korea_if.csv')},**{r['patient_id']:r for r in C.read_csv(HERE/'inputs/scotland_if.csv')}}
    res=dict(encoder=enc,cohorts={})
    for coh,cs_all in (('korea',cases),('scotland',scot)):
        surv=C.primary(cs_all,cfg);R={}
        t=lambda cs:np.array([c['time'] for c in cs]);ev=lambda cs:np.array([c['event'] for c in cs]);g=lambda cs:np.array([ai[c['patient_id']] for c in cs])
        # A. ISUP grade 2
        i2=[c for c in surv if c['isup']==2];R['isup2']=dict(n=len(i2),events=int(ev(i2).sum()))
        if ev(i2).sum()>0:
            R['isup2'].update(c_ai=C.harrell_c(t(i2),ev(i2),g(i2)),cox_ai=cox1(g(i2),t(i2),ev(i2),f'{coh}:isup2:ai'),
                              cox_leibovich_ai=cox1(np.c_[[c['leibovich_fuhrman'] for c in i2],g(i2)],t(i2),ev(i2),f'{coh}:isup2:leibovich+ai'),
                              median_split=split_block(t(i2),ev(i2),g(i2),np.median(g(i2))),
                              split_2_5=split_block(t(i2),ev(i2),g(i2),CFG['ai_upgrade_threshold'],ge=True))
        # B. patch-level interpretation
        hg=lambda cs:np.array([float(feats[c['patient_id']]['hg_frac']) for c in cs])
        R['hg_frac_by_isup']={k:[float(np.median(hg([c for c in cs_all if c['isup']==k]))) if any(c['isup']==k for c in cs_all) else None,
                                  int(sum(c['isup']==k for c in cs_all))] for k in (1,2,3,4)}
        R['hg_frac_vs_ai_spearman']=C.spearman(hg(cs_all),g(cs_all))
        for name,sub in (('low_grade',[c for c in surv if c['isup']<=2]),('isup2',i2)):
            b=dict(n=len(sub),events=int(ev(sub).sum()))
            if ev(sub).sum()>0:
                z=(hg(sub)-hg(sub).mean())/hg(sub).std()
                b.update(c_hg_frac=C.harrell_c(t(sub),ev(sub),hg(sub)),cox_hg_frac_per_sd=cox1(z,t(sub),ev(sub),f'{coh}:{name}:hg_frac'),
                         median_split=split_block(t(sub),ev(sub),hg(sub),np.median(hg(sub))),
                         c_top_attn_egrade=C.harrell_c(t(sub),ev(sub),[float(feats[c['patient_id']]['top_attn_egrade']) for c in sub]))
            R[f'hg_frac_{name}']=b
        # C. nuclear morphology (immunofluorescence image analysis, independent of HE)
        sub=[c for c in cs_all if c['patient_id'] in IF];R['nuclear']=dict(n=len(sub))
        for col in ('avg_nucleus_area_m','avg_nucleus_perimeter_m','avg_nucleus_roundness','avg_cell_area_m'):
            v=np.array([float(IF[c['patient_id']][col]) for c in sub]);iv=np.array([c['isup'] for c in sub])
            wa,used=partial_spearman(g(sub),v,iv);wh,_=partial_spearman(hg(sub),v,iv)
            R['nuclear'][col]=dict(spearman_ai=C.spearman(g(sub),v),spearman_isup=C.spearman(iv,v),within_isup_spearman_ai=wa,
                                   spearman_hg_frac=C.spearman(hg(sub),v),within_isup_spearman_hg_frac=wh,within_isup_strata_used=used)
        # D. P21+/MCM2- biomarker (published cutoff 2 %: high = favourable)
        sub=[c for c in surv if c['patient_id'] in IF];p21=np.array([float(IF[c['patient_id']]['p21_mcm2_percentage']) for c in sub])
        D=dict(n=len(sub),events=int(ev(sub).sum()),spearman_ai_p21=C.spearman(g(sub),p21),c_ai=C.harrell_c(t(sub),ev(sub),g(sub)),
               c_p21_low_is_risk=C.harrell_c(t(sub),ev(sub),-p21))
        if ev(sub).sum()>=5:D['cox_ai_plus_logp21']=cox1(np.c_[g(sub),np.log(p21+CFG['if_offset'])],t(sub),ev(sub),f'{coh}:if:ai+logp21')
        D['ai_threshold']=float(np.median(g(surv)));D['ai_threshold_from']='median AI grade of the cohort survival set (PROTOCOL 2.4)'
        aihi=g(sub)>D['ai_threshold'];p21lo=p21<CFG['p21_cutoff_percent'];groups={}
        for name,m in (('ai_low_p21_high',~aihi&~p21lo),('ai_low_p21_low',~aihi&p21lo),('ai_high_p21_high',aihi&~p21lo),('ai_high_p21_low',aihi&p21lo)):
            groups[name]=dict(n=int(m.sum()),events=int(ev(sub)[m].sum()))
        D['groups']=groups;both=aihi&p21lo
        if both.any() and (~both).any():chi,p=C.logrank(t(sub),ev(sub),both);D['ai_high_and_p21_low_vs_rest']=dict(chi2=chi,p=p)
        R['if_biomarker']=D
        res['cohorts'][coh]=R
    res['cox_warnings']=list(C.COX_WARNINGS);res['finished']=C.now();res['gx1_code_signature']=C.signature()
    res=finite(res);C.save_json(HERE/'EXPLAIN_RESULTS.json',res);(HERE/'EXPLAIN_RESULTS.md').write_text(markdown(res))
    try:res['figures']=figures(res,cases,scot,cfg,ai,feats,IF)
    except Exception as ex:res['figure_warning']=repr(ex)[:300];print('WARNING figures',ex,flush=True)
    C.save_json(HERE/'EXPLAIN_RESULTS.json',finite(res));print('EXPLAIN DONE',flush=True);return 0

def fmt_hr(h,j=-1):
    if not h:return '拟合失败'
    x=h[j];return f"HR {x[0]:.2f}（{x[1]:.2f}–{x[2]:.2f}），p = {x[3]:.3g}"
def fmt_split(s):
    if not s:return '—'
    p=s.get('logrank_p');return f"高 {s['high_n']} 例 {s['high_events']} 次复发 / 低 {s['low_n']} 例 {s['low_events']} 次，log-rank p = {'—' if p is None else '%.3g'%p}"
def markdown(res):
    L=['# gx1 事后探索性分析：ISUP 2 级、图块级解释、细胞核形态、P21+/MCM2−','',f"生成时间 {res['finished']}；编码器 {res['encoder']}；gx1 代码签名 `{res['gx1_code_signature'][:16]}`。",
       '以下均为在看到 gx1 结果之后加做的探索性分析（计划在计算前锁定），不作判定。','']
    for coh,R in res['cohorts'].items():
        a=R['isup2'];name='韩国（折外）' if coh=='korea' else '苏格兰（外部）'
        L+=[f'## {name}','',f"**A. ISUP 2 级**（{a['n']} 例 {a['events']} 次复发）"]
        if 'c_ai' in a:
            L+=[f"- AI 分级分数 C 指数 {a['c_ai']:.3f}；Cox：每升 1 级 {fmt_hr(a['cox_ai'])}；调整 Leibovich 后 {fmt_hr(a['cox_leibovich_ai'])}",
                f"- 按本组中位数分：{fmt_split(a['median_split'])}",f"- 按 AI 分级分数 ≥ 2.5（「AI 升级」）分：{fmt_split(a['split_2_5'])}"]
        L+=['',f"**B. 图块级解释**：高级别样图块（P(3)+P(4) > 0.5）比例的中位数，ISUP 1/2/3/4 级 = "+' / '.join('—' if v[0] is None else '%.3f'%v[0] for v in R['hg_frac_by_isup'].values())+
            f"；与 AI 分级分数的 Spearman {R['hg_frac_vs_ai_spearman']:.2f}"]
        for nm,lab in (('low_grade','低级别（ISUP 1–2）'),('isup2','ISUP 2 级')):
            b=R[f'hg_frac_{nm}']
            if 'c_hg_frac' in b:L.append(f"- {lab} {b['n']} 例 {b['events']} 次复发：高级别样图块比例 C 指数 {b['c_hg_frac']:.3f}，每 SD {fmt_hr(b['cox_hg_frac_per_sd'])}；中位数分组 {fmt_split(b['median_split'])}；注意力最高 10% 图块的平均级别 C 指数 {b['c_top_attn_egrade']:.3f}")
        n=R['nuclear'];L+=['',f"**C. 细胞核形态**（免疫荧光图像分析，与 HE 独立；{n['n']} 例）"]
        for col,lab in (('avg_nucleus_area_m','细胞核面积'),('avg_nucleus_perimeter_m','细胞核周长'),('avg_nucleus_roundness','细胞核圆度'),('avg_cell_area_m','细胞面积')):
            v=n[col];f2=lambda x:'—' if x is None else '%+.2f'%x
            L.append(f"- {lab}：与 AI 分级分数 Spearman {f2(v['spearman_ai'])}（同一 ISUP 级内 {f2(v['within_isup_spearman_ai'])}，使用的级别及例数 {v['within_isup_strata_used']}）；与 ISUP {f2(v['spearman_isup'])}；与高级别样图块比例 {f2(v['spearman_hg_frac'])}（同级内 {f2(v['within_isup_spearman_hg_frac'])}）")
        d=R['if_biomarker'];L+=['',f"**D. P21+/MCM2−**（{d['n']} 例 {d['events']} 次复发；AI 切点 = 本队列复发分析人群中位数 {d['ai_threshold']:.2f}）：与 AI 分级分数 Spearman {d['spearman_ai_p21']:+.2f}；C 指数 AI {d['c_ai']:.3f}、P21（低=高风险）{d['c_p21_low_is_risk']:.3f}"]
        if d.get('cox_ai_plus_logp21'):L.append(f"- Cox(AI + log P21)：AI {fmt_hr(d['cox_ai_plus_logp21'],0)}；log P21 {fmt_hr(d['cox_ai_plus_logp21'],1)}")
        L.append('- 四组（AI 按中位数、P21 按已发表的 2% 切点）：'+'；'.join(f"{k} {v['n']} 例 {v['events']} 次" for k,v in d['groups'].items())+
                 (f"；AI 高且 P21 低 vs 其余 log-rank p = {d['ai_high_and_p21_low_vs_rest']['p']:.3g}" if 'ai_high_and_p21_low_vs_rest' in d else ''))
        L.append('')
    if res['cox_warnings']:L+=[f"注意：{len(res['cox_warnings'])} 个 Cox 拟合有问题（见 JSON）。",'']
    return '\n'.join(L)+'\n'

def figures(res,cases,scot,cfg,ai,feats,IF):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    out=HERE/'figures';out.mkdir(exist_ok=True);made=[]
    fig,ax=plt.subplots(2,3,figsize=(16,9))
    for r,(coh,cs_all) in enumerate((('korea',cases),('scotland',scot))):
        surv=C.primary(cs_all,cfg);name='Korea (out-of-fold)' if coh=='korea' else 'Scotland (external)'
        i2=[c for c in surv if c['isup']==2];g=np.array([ai[c['patient_id']] for c in i2]);thr=np.median(g)
        for m,col,lab in ((g>thr,'#c0392b','AI grade high'),(g<=thr,'#2471a3','AI grade low')):
            x,y=km([c['time'] for c,k in zip(i2,m) if k],[c['event'] for c,k in zip(i2,m) if k]);ax[r,0].step(x,y,where='post',color=col,label=f'{lab} (n={m.sum()})')
        p=res['cohorts'][coh]['isup2']['median_split'].get('logrank_p');ax[r,0].set(title=f'{name}: ISUP 2 only, log-rank p={p:.3g}',xlabel='months',ylabel='recurrence-free',ylim=(0,1.02));ax[r,0].legend()
        hg=[[float(feats[c['patient_id']]['hg_frac']) for c in cs_all if c['isup']==k] for k in (1,2,3,4)]
        ax[r,1].boxplot([v if v else [np.nan] for v in hg],tick_labels=['1','2','3','4']);ax[r,1].set(title=f'{name}: high-grade-like patch fraction',xlabel='ISUP grade',ylabel='fraction of patches P(3)+P(4)>0.5')
        sub=[c for c in cs_all if c['patient_id'] in IF]
        sc=ax[r,2].scatter([ai[c['patient_id']] for c in sub],[float(IF[c['patient_id']]['avg_nucleus_area_m']) for c in sub],c=[c['isup'] for c in sub],cmap='viridis',vmin=1,vmax=4,s=18)
        rho=res['cohorts'][coh]['nuclear']['avg_nucleus_area_m']['spearman_ai'];ax[r,2].set(title=f'{name}: nuclear area (IF) vs AI grade, rho={rho:+.2f}',xlabel='AI grade score',ylabel='mean nucleus area (um^2)')
        plt.colorbar(sc,ax=ax[r,2],label='ISUP')
    fig.tight_layout();p=out/'explain_summary.png';fig.savefig(p,dpi=130);plt.close(fig);made.append(p.name);return made

def main(argv):
    cmd=argv[0] if argv else 'run'
    if cmd=='run':return run()
    if cmd=='lock-verify':ok,bad=lock_status();print('LOCK','OK' if ok else 'BROKEN',bad);return 0 if ok else 1
    if cmd=='lock-write':
        if (HERE/'LOCK.json').exists():print('LOCK.json exists already');return 1
        if (HERE/'EXPLAIN_RESULTS.json').exists():print('REFUSED: results exist already');return 1
        lk=dict(locked_at=C.now(),files={n:sha(HERE/n) for n in LOCKED},note='exploratory analyses written after the gx1 results were seen; locked before computing them')
        C.save_json(HERE/'LOCK.json',lk);print('LOCKED',len(lk['files']),'files');return 0
    print(__doc__);return 2
if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
