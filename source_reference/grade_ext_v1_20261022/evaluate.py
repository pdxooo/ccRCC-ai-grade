"""Pre-registered analysis of the ISUP grading external validation and AI grade score study (PROTOCOL.md sections 4-6).
Run once, after all 225 runs. Refuses (exit 1) unless every run (9 encoders x 5 seeds x 5 folds) has a COMPLETE,
hash-verified receipt from this locked code signature and config on one platform (the technical check's), every Korean
case is predicted once per seed and every Scottish case by all 25 models of each encoder, and every run of an encoder read
the same feature bytes. Writes RESULTS.json and RESULTS.md and prints the decisions. Exit: 0 done; 1 refused."""
from common import *
import argparse

def expected_runs(cfg):return [(e,s,f) for e in cfg['encoders'] for s in cfg['seeds'] for f in cfg['folds']]

def collect(cfg,cases,assign,scot):
    sig=signature();problems=[];plats=set();kr={};sc={};feats={}
    for e,s,f in expected_runs(cfg):
        d=RUNS/e/f'seed{s}'/f'fold{f}';tag=f'{e}/seed{s}/fold{f}'
        rc=read_json(d/'complete.json')
        if not rc or rc.get('state')!='COMPLETE' or rc.get('check') is not False:problems.append(f'{tag}: no COMPLETE formal receipt');continue
        bad=[n for n,h in rc['artifacts'].items() if not (d/n).exists() or sha(d/n)!=h]
        if bad:problems.append(f'{tag}: artifacts changed {bad}');continue
        md=read_json(d/'metadata.json') or {};c=md.get('contract') or {}
        if hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()!=md.get('fingerprint') or md.get('fingerprint')!=rc['fingerprint']:
            problems.append(f'{tag}: fingerprint does not match its contract');continue
        want=dict(code=sig,encoder=e,seed_index=s,fold=f,check=False,epochs=cfg['epochs'],labels_sha256=cfg['labels_sha256'])
        diff={k:c.get(k) for k,v in want.items() if c.get(k)!=v}
        if diff or c.get('config')!=cfg:problems.append(f'{tag}: contract differs {diff or "config"}');continue
        test=sorted(x['patient_id'] for x in roles(cases,assign,f,cfg)['test']);P=read_json(d/'predictions.json') or []
        if sorted(c.get('test') or [])!=test or sorted(c.get('external') or [])!=sorted(x['patient_id'] for x in scot):
            problems.append(f'{tag}: contract test/external cases differ from the locked split');continue
        got_k=sorted(x['patient_id'] for x in P if x['cohort']=='korea');got_s=sorted(x['patient_id'] for x in P if x['cohort']=='scotland')
        if got_k!=test or got_s!=sorted(c['patient_id'] for c in scot):problems.append(f'{tag}: predictions are not the fold-{f} test cases + all Scottish cases');continue
        if any(len(x['probs'])!=4 or not np.isfinite(x['probs']).all() or abs(sum(x['probs'])-1)>1e-6 for x in P):problems.append(f'{tag}: invalid probabilities');continue
        plats.add(json.dumps(rc['platform'],sort_keys=True))
        for p,h in (md.get('feature_sha256') or {}).items():feats.setdefault((e,p),set()).add(h)
        for x in P:
            if x['cohort']=='korea':kr.setdefault((e,s),{})[x['patient_id']]=x['probs']
            else:sc.setdefault((e,s,f),{})[x['patient_id']]=x['probs']
    if len(plats)>1:problems.append(f'runs come from {len(plats)} platforms')
    if any(len(v)>1 for v in feats.values()):problems.append('some case was read with different feature bytes in different runs')
    for k,v in kr.items():
        if len(v)!=len(cases):problems.append(f'{k}: Korean OOF covers {len(v)} of {len(cases)} cases')
    return kr,sc,problems,sorted(plats)

# ----------------------------------------------------------------------------------------------------- statistics
def boot(n,fn,B,seed,strata=None,events=None):
    """Case bootstrap of fn(index array) with fixed models (stratified by strata when given: every grade keeps its size);
    percentile 95% interval. Resamples where fn is undefined (no event, no comparable pair, one class) are skipped."""
    rng=np.random.default_rng(seed);v=[]
    groups=[np.flatnonzero(np.asarray(strata)==g) for g in np.unique(strata)] if strata is not None else None
    for _ in range(B):
        ix=np.concatenate([rng.choice(g,len(g)) for g in groups]) if groups else rng.integers(0,n,n)
        if events is not None and np.asarray(events)[ix].sum()==0:continue
        try:v.append(fn(ix))
        except AssertionError:continue
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]

def grading(isup,P,cfg,seed):
    """Grading metrics of one cohort: ISUP 3-4 vs 1-2 AUC (primary), G1 vs G2 AUC, 4-grade argmax metrics."""
    A=cfg['analysis'];isup=np.asarray(isup);P=np.asarray(P);B=A['bootstrap_replicates']
    out=dict(n=int(len(isup)),n_per_grade={g:int((isup==g).sum()) for g in (1,2,3,4)},
             highlow_auc=highlow_auc(isup,P),highlow_auc_ci95=boot(len(isup),lambda ix:highlow_auc(isup[ix],P[ix]),B,seed,strata=isup),
             g12_auc=g12_auc(isup,P),g12_auc_ci95=boot(len(isup),lambda ix:g12_auc(isup[ix],P[ix]),B,seed,strata=isup))
    out.update(grade_metrics(isup,P));return out

def cox_safe(X,t,e,tag):
    """cox_fit that never stops the analysis: (beta, se, loglik, ok). ok is False when the fit failed (singular), is not
    finite, or common.cox_fit flagged it (no convergence or |beta| > 15); the warning is tagged with the fit's name."""
    n0=len(COX_WARNINGS)
    try:b,se,ll=cox_fit(X,t,e)
    except (np.linalg.LinAlgError,AssertionError,FloatingPointError) as ex:COX_WARNINGS.append(dict(fit=tag,error=repr(ex)[:200]));return None,None,None,False
    ok=bool(np.isfinite(b).all() and np.isfinite(se).all() and np.isfinite(ll)) and len(COX_WARNINGS)==n0
    for w in COX_WARNINGS[n0:]:w['fit']=tag
    if not ok and len(COX_WARNINGS)==n0:COX_WARNINGS.append(dict(fit=tag,error='non-finite estimate'))
    return b,se,ll,ok
def hr_of(b,se,j,ok):
    if not ok:return dict(ok=False,hr_ai_per_grade=None,hr_ai_ci95=None,p=None)
    return dict(ok=True,hr_ai_per_grade=float(np.exp(b[j])),hr_ai_ci95=[float(np.exp(b[j]-1.96*se[j])),float(np.exp(b[j]+1.96*se[j]))],p=float(2*norm_sf(abs(b[j]/se[j]))))

def prognosis(cs,g,cfg,seed,tag=''):
    """AI grade score vs the pathologist grades for recurrence in one cohort's survival set (no fitting across cohorts)."""
    B=cfg['analysis']['bootstrap_replicates'];g=np.asarray(g,float)
    t=np.array([c['time'] for c in cs]);e=np.array([c['event'] for c in cs]);I=np.array([c['isup'] for c in cs],float)
    Fu=np.array([c['fuhrman'] for c in cs],float);L=np.array([c['leibovich_fuhrman'] for c in cs],float)
    C=lambda s,ix=slice(None):harrell_c(t[ix],e[ix],s[ix])
    out=dict(n=int(len(cs)),events=int(e.sum()),c_isup=C(I),c_fuhrman=C(Fu),c_ai_grade=C(g),c_leibovich=C(L))
    out['delta_c_vs_isup']=out['c_ai_grade']-out['c_isup'];out['delta_c_vs_isup_ci95']=boot(len(cs),lambda ix:C(g,ix)-C(I,ix),B,seed,events=e)
    out['delta_c_vs_fuhrman']=out['c_ai_grade']-out['c_fuhrman']
    out['c_within_isup_grade']=harrell_c_within(t,e,g,I)
    out['c_within_isup_grade_ci95']=boot(len(cs),lambda ix:harrell_c_within(t[ix],e[ix],g[ix],I[ix]),B,seed,events=e)
    b,se,ll,ok=cox_safe(np.c_[I,g],t,e,f'{tag}:isup+ai');_,_,ll0,ok0=cox_safe(I[:,None],t,e,f'{tag}:isup')
    out['cox_isup_plus_ai']=dict(hr_of(b,se,1,ok),hr_isup=float(np.exp(b[0])) if ok else None,
                                 lr_test_p=float(math.erfc(math.sqrt(max(0.,2*(ll-ll0))/2))) if ok and ok0 else None)
    b,se,_,ok=cox_safe(np.c_[L,g],t,e,f'{tag}:leibovich+ai');out['cox_leibovich_plus_ai']=hr_of(b,se,1,ok)
    lo=I<=2;tl,el,gl=t[lo],e[lo],g[lo];out['low_grade']=dict(n=int(lo.sum()),events=int(el.sum()))
    if el.sum()>0:
        thr=float(np.median(gl));chi,p=logrank(tl,el,gl>thr)
        out['low_grade'].update(c_ai_grade=harrell_c(tl,el,gl),threshold=thr,high_n=int((gl>thr).sum()),high_events=int(el[gl>thr].sum()),
                                low_events=int(el[gl<=thr].sum()),logrank_chi2=chi,logrank_p=p)
    return out

def decide(r,cfg):
    D=cfg['decision'];s=r['scotland']['grading'];a=s['highlow_auc'];lo=s['highlow_auc_ci95'][0];G=D['grading']
    grading_v='SUCCESS' if a>=G['min_auc']-1e-12 and lo>G['min_ci_lower'] else 'FAIL' if a<G['fail_below'] else 'INCONCLUSIVE'
    p=r['scotland']['prognosis'];A=D['ai_grade'];hr=p['cox_isup_plus_ai']
    if not hr.get('ok',True):ai_v='NEGATIVE' if p['delta_c_vs_isup']<=0 else 'INCONCLUSIVE'  # no usable Cox fit: cannot be positive
    elif p['delta_c_vs_isup']>=A['min_delta_c']-1e-12 and (hr['hr_ai_ci95'][0]>1 or not A['require_hr_ci_above_1']):ai_v='POSITIVE'
    elif p['delta_c_vs_isup']<=0 or hr['hr_ai_per_grade']<=1:ai_v='NEGATIVE'
    else:ai_v='INCONCLUSIVE'
    return dict(grading=grading_v,ai_grade=ai_v,verdict=f'grading={grading_v}; ai_grade={ai_v}')

def analyse(cfg,cases,assign,scot,kr,sc):
    seeds=cfg['seeds'];folds=cfg['folds'];seed=cfg['analysis']['bootstrap_seed'];surv_k=primary(cases,cfg)
    res=dict(n=dict(korea=len(cases),korea_survival=len(surv_k),scotland=len(scot)),encoders={})
    for e in cfg['encoders']:
        Pk={p:np.mean([kr[(e,s)][p] for s in seeds],0) for p in kr[(e,seeds[0])]}
        Ps=np.array([np.mean([sc[(e,s,f)][c['patient_id']] for s in seeds for f in folds],0) for c in scot])
        PK=np.array([Pk[c['patient_id']] for c in cases]);gk={c['patient_id']:float(grade_score(Pk[c['patient_id']][None])[0]) for c in cases}
        r=dict(korea=dict(grading=grading([c['isup'] for c in cases],PK,cfg,seed),prognosis=prognosis(surv_k,[gk[c['patient_id']] for c in surv_k],cfg,seed,f'{e}:korea')),
               scotland=dict(grading=grading([c['isup'] for c in scot],Ps,cfg,seed),prognosis=prognosis(scot,grade_score(Ps),cfg,seed,f'{e}:scotland')))
        r['per_seed']={k:[] for k in ('korea_highlow_auc','scotland_highlow_auc','korea_g12_auc','scotland_g12_auc','korea_c_ai_grade','scotland_c_ai_grade')}
        ik=[c['isup'] for c in cases];isc=[c['isup'] for c in scot];ts=[c['time'] for c in scot];es=[c['event'] for c in scot]
        tk=[c['time'] for c in surv_k];ek=[c['event'] for c in surv_k]
        for s in seeds:
            a=np.array([kr[(e,s)][c['patient_id']] for c in cases]);b=np.array([np.mean([sc[(e,s,f)][c['patient_id']] for f in folds],0) for c in scot])
            gs={c['patient_id']:float(grade_score(np.asarray(kr[(e,s)][c['patient_id']])[None])[0]) for c in cases}
            for k,v in (('korea_highlow_auc',highlow_auc(ik,a)),('scotland_highlow_auc',highlow_auc(isc,b)),('korea_g12_auc',g12_auc(ik,a)),('scotland_g12_auc',g12_auc(isc,b)),
                        ('korea_c_ai_grade',harrell_c(tk,ek,[gs[c['patient_id']] for c in surv_k])),('scotland_c_ai_grade',harrell_c(ts,es,grade_score(b)))):r['per_seed'][k].append(v)
        r['family']=cfg['encoder_family'][e];res['encoders'][e]=r
    E=res['encoders'];prim=cfg['primary_encoder']
    sel=max(cfg['encoders'],key=lambda e:(E[e]['korea']['grading']['highlow_auc'],-list(cfg['encoders']).index(e)))
    fam={f:[E[e]['scotland']['grading']['highlow_auc'] for e in cfg['encoders'] if cfg['encoder_family'][e]==f] for f in ('pathology','imagenet')}
    res['benchmark']=dict(selected_by_korea=sel,selected_scotland_highlow_auc=E[sel]['scotland']['grading']['highlow_auc'],
                          selected_scotland_highlow_auc_ci95=E[sel]['scotland']['grading']['highlow_auc_ci95'],
                          family_mean_scotland_highlow_auc={f:float(np.mean(v)) for f,v in fam.items()})
    res['decision']=dict(decide(E[prim],cfg),encoder=prim,rule='PROTOCOL.md section 4')
    res['cox_warnings']=list(COX_WARNINGS)
    return res

def markdown(res,cfg):
    E=res['encoders'];prim=cfg['primary_encoder'];D=res['decision']
    f=lambda x,ci:f"{x:.3f}（{ci[0]:.3f}–{ci[1]:.3f}）"
    L=['# ISUP 分级外部验证与 AI 分级分数：结果（预先登记的分析）','',f"生成时间 {res['finished']}；代码签名 `{res['code_signature'][:16]}`；平台 {res['platform']}。",
       f"韩国开发 {res['n']['korea']} 例（复发分析 {res['n']['korea_survival']} 例）；苏格兰外部验证 {res['n']['scotland']} 例；84 个月截尾。",'',
       f"**判定 1（分级外推）：{D['grading']}**；**判定 2（AI 分级分数的预后价值）：{D['ai_grade']}**（主编码器 {prim}；规则见 PROTOCOL.md 第 4 节）",'']
    for e in [prim]:
        k,s=E[e]['korea'],E[e]['scotland']
        L+=[f'## {e}（主编码器）','','| | 韩国（折外） | 苏格兰（外部） |','|---|---|---|',
            f"| 高级别（3–4）对低级别（1–2）AUC | {f(k['grading']['highlow_auc'],k['grading']['highlow_auc_ci95'])} | {f(s['grading']['highlow_auc'],s['grading']['highlow_auc_ci95'])} |",
            f"| G1 对 G2 AUC | {f(k['grading']['g12_auc'],k['grading']['g12_auc_ci95'])} | {f(s['grading']['g12_auc'],s['grading']['g12_auc_ci95'])} |",
            f"| 四级：二次加权 kappa；准确率；宏 F1 | {k['grading']['qwk']:.3f}；{k['grading']['accuracy']:.3f}；{k['grading']['macro_f1']:.3f} | {s['grading']['qwk']:.3f}；{s['grading']['accuracy']:.3f}；{s['grading']['macro_f1']:.3f} |",
            f"| 复发 C 指数：ISUP / Fuhrman / AI 分级分数 | {k['prognosis']['c_isup']:.3f} / {k['prognosis']['c_fuhrman']:.3f} / {k['prognosis']['c_ai_grade']:.3f} | {s['prognosis']['c_isup']:.3f} / {s['prognosis']['c_fuhrman']:.3f} / {s['prognosis']['c_ai_grade']:.3f} |",
            f"| ΔC（AI − ISUP，95% 区间） | {k['prognosis']['delta_c_vs_isup']:+.3f}（{k['prognosis']['delta_c_vs_isup_ci95'][0]:+.3f} 至 {k['prognosis']['delta_c_vs_isup_ci95'][1]:+.3f}） | {s['prognosis']['delta_c_vs_isup']:+.3f}（{s['prognosis']['delta_c_vs_isup_ci95'][0]:+.3f} 至 {s['prognosis']['delta_c_vs_isup_ci95'][1]:+.3f}） |",
            f"| 同一 ISUP 级内的 C 指数（AI 分级分数） | {f(k['prognosis']['c_within_isup_grade'],k['prognosis']['c_within_isup_grade_ci95'])} | {f(s['prognosis']['c_within_isup_grade'],s['prognosis']['c_within_isup_grade_ci95'])} |"]
        for coh,b in (('韩国',k),('苏格兰',s)):
            x=b['prognosis']['cox_isup_plus_ai'];y=b['prognosis']['cox_leibovich_plus_ai'];lg=b['prognosis']['low_grade']
            hrs=lambda z:f"{z['hr_ai_per_grade']:.2f}（{z['hr_ai_ci95'][0]:.2f}–{z['hr_ai_ci95'][1]:.2f}），p = {z['p']:.3g}" if z['ok'] else '拟合失败（见 cox_warnings）'
            L+=['',f"- {coh} Cox(ISUP + AI 分级分数)：AI 每升高 1 级 HR {hrs(x)}；似然比检验 p = {'—' if x['lr_test_p'] is None else '%.3g'%x['lr_test_p']}；"
                f"Cox(Leibovich + AI) 中 AI 的 HR {hrs(y)}"]
            if 'c_ai_grade' in lg:
                L.append(f"- {coh}低级别（ISUP 1–2）{lg['n']} 例 {lg['events']} 次复发：AI 分级分数 C 指数 {lg['c_ai_grade']:.3f}；按本队列中位数分组，高分 {lg['high_n']} 例 {lg['high_events']} 次复发、低分 {lg['n']-lg['high_n']} 例 {lg['low_events']} 次，log-rank p = {lg['logrank_p']:.3g}")
        ps=E[e]['per_seed'];j=lambda k:', '.join('%.3f'%v for v in ps[k])
        L+=['',f"- 每个种子：高低级别 AUC 韩国 {j('korea_highlow_auc')}，苏格兰 {j('scotland_highlow_auc')}；G1/G2 AUC 韩国 {j('korea_g12_auc')}，苏格兰 {j('scotland_g12_auc')}；"
            f"AI 分级分数 C 指数 韩国 {j('korea_c_ai_grade')}，苏格兰 {j('scotland_c_ai_grade')}",'']
    B=res['benchmark']
    L+=['## 9 个编码器对比','','| 编码器 | 类型 | 韩国 高低 AUC | 苏格兰 高低 AUC | 苏格兰 G1/G2 AUC | 苏格兰 QWK | 苏格兰 C（AI / ISUP） | 苏格兰 ΔC |','|---|---|---|---|---|---|---|---|']
    for e,r in E.items():
        k,s=r['korea'],r['scotland']
        L.append(f"| {e}{'（主）' if e==prim else ''}{'（韩国选出）' if e==B['selected_by_korea'] else ''} | {'病理基础模型' if r['family']=='pathology' else 'ImageNet'} | {k['grading']['highlow_auc']:.3f} | {f(s['grading']['highlow_auc'],s['grading']['highlow_auc_ci95'])} | {s['grading']['g12_auc']:.3f} | {s['grading']['qwk']:.3f} | {s['prognosis']['c_ai_grade']:.3f} / {s['prognosis']['c_isup']:.3f} | {s['prognosis']['delta_c_vs_isup']:+.3f} |")
    L+=['',f"- 只用韩国折外结果选出的编码器：{B['selected_by_korea']}，苏格兰高低级别 AUC {f(B['selected_scotland_highlow_auc'],B['selected_scotland_highlow_auc_ci95'])}",
        f"- 苏格兰高低级别 AUC 平均：病理基础模型 {B['family_mean_scotland_highlow_auc']['pathology']:.3f}，ImageNet 模型 {B['family_mean_scotland_highlow_auc']['imagenet']:.3f}",'']
    if res['cox_warnings']:L+=[f"注意：{len(res['cox_warnings'])} 个 Cox 拟合未收敛或系数过大（见 RESULTS.json cox_warnings）。",'']
    L+=['苏格兰只有 120 例（ISUP 4 级 2 例、27 次复发），区间较宽。不得据此修改标签、病例、截尾或阈值。']
    return '\n'.join(L)+'\n'

def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    cfg=config();cases,assign=load_korea(cfg);scot=load_scotland(cfg);problems=[]
    ok,bad=lock_status()
    if not ok:problems.append(f'code is not the locked version: {bad[:5]}')
    chk=read_json(CHECK_PASS) or {}
    if chk.get('state')!='PASS' or chk.get('code_signature')!=signature():problems.append('CHECK_PASS.json is not a PASS for this code')
    kr,sc,more,plats=collect(cfg,cases,assign,scot);problems+=more
    if plats and json.dumps(chk.get('platform'),sort_keys=True) not in plats:problems.append('runs and the technical check ran on different platforms')
    if problems:
        print('REFUSED',len(problems),'problem(s):',flush=True)
        for p in problems[:40]:print(' ',p,flush=True)
        return 1
    res=analyse(cfg,cases,assign,scot,kr,sc);res.update(code_signature=signature(),platform=[json.loads(p) for p in plats],finished=now(),runs=len(expected_runs(cfg)))
    save_json(HERE/'RESULTS.json',res);(HERE/'RESULTS.md').write_text(markdown(res,cfg))
    s=res['encoders'][cfg['primary_encoder']]['scotland']
    print('DECISION grading',res['decision']['grading'],'ai_grade',res['decision']['ai_grade'],'scotland high/low AUC %.3f dC %+.3f'%(
          s['grading']['highlow_auc'],s['prognosis']['delta_c_vs_isup']),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
