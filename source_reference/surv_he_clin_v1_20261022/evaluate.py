"""Pre-registered analysis of HE + clinical recurrence prediction (PROTOCOL.md sections 3-6). Run once, after all 50 runs.
Refuses (exit 1) unless every run (2 encoders x 5 seeds x 5 folds) has a COMPLETE, hash-verified receipt from this locked
code signature and config on one platform (the technical check's), every Korean case is predicted once per seed and every
Scottish case by all 25 models of each encoder, and every run of an encoder read the same feature bytes.
Writes RESULTS.json and RESULTS.md and prints the decision. Exit: 0 done; 1 refused."""
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
        got_k=sorted(x['patient_id'] for x in P if x['cohort']=='korea');got_s=sorted(x['patient_id'] for x in P if x['cohort']=='scotland')
        if got_k!=test or got_s!=sorted(c['patient_id'] for c in scot):problems.append(f'{tag}: predictions are not the fold-{f} test cases + all Scottish cases');continue
        if any(not np.isfinite(x['risk']) for x in P):problems.append(f'{tag}: non-finite risk');continue
        plats.add(json.dumps(rc['platform'],sort_keys=True))
        for p,h in (md.get('feature_sha256') or {}).items():feats.setdefault((e,p),set()).add(h)
        for x in P:
            if x['cohort']=='korea':kr.setdefault((e,s),{})[x['patient_id']]=x['risk']
            else:sc.setdefault((e,s,f),{})[x['patient_id']]=x['risk']
    if len(plats)>1:problems.append(f'runs come from {len(plats)} platforms')
    if any(len(v)>1 for v in feats.values()):problems.append('some case was read with different feature bytes in different runs')
    for k,v in kr.items():
        if len(v)!=len(cases):problems.append(f'{k}: Korean OOF covers {len(v)} of {len(cases)} cases')
    return kr,sc,problems,sorted(plats)

def cindex(cs,score):return harrell_c([c['time'] for c in cs],[c['event'] for c in cs],score)
def boot_ci(cs,fn,B,seed):
    """Case bootstrap of a statistic fn(index array) with fixed models; percentile 95% interval (resamples without an
    event or without a comparable pair are skipped)."""
    rng=np.random.default_rng(seed);n=len(cs);v=[]
    for _ in range(B):
        ix=rng.integers(0,n,n)
        if sum(cs[i]['event'] for i in ix)==0:continue
        try:v.append(fn(ix))
        except AssertionError:continue
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]

def standardise(cfg,cases_all,assign,raw_k,raw_s):
    """HE_z (PROTOCOL.md section 3): Korean OOF risk standardised within its outer fold (mean and SD of the fold's primary
    test cases), Scottish risk standardised within the Scottish cohort. Only HE scores enter, never outcomes.
    raw_k: {patient_id: risk} for all Korean cases; raw_s: Scottish risks in load_scotland order."""
    fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'};zk={};diag=dict(korea_fold_mean=[],korea_fold_sd=[])
    for f in cfg['folds']:
        ref=np.array([raw_k[c['patient_id']] for c in primary(cases_all,cfg) if fold[c['patient_id']]==f]);mu,sd=float(ref.mean()),float(ref.std())
        diag['korea_fold_mean'].append(mu);diag['korea_fold_sd'].append(sd)
        zk.update({c['patient_id']:(raw_k[c['patient_id']]-mu)/sd for c in cases_all if fold[c['patient_id']]==f})
    rs=np.asarray(raw_s,float);diag.update(scotland_mean=float(rs.mean()),scotland_sd=float(rs.std()))
    return zk,(rs-rs.mean())/rs.std(),diag

def fuse(cfg,kr_cases,zk,sc_cases,zs,base,assign):
    """Leibovich alone vs Cox(Leibovich + HE_z): Korean cross-fitted C, Scottish frozen-model C, Scottish own Cox fit.
    zk, zs: HE_z of kr_cases and sc_cases (already standardised)."""
    A=cfg['analysis'];zk=np.asarray(zk,float);zs=np.asarray(zs,float)
    Lk=np.array([c[base] for c in kr_cases]);Ls=np.array([c[base] for c in sc_cases])
    tk=np.array([c['time'] for c in kr_cases]);ek=np.array([c['event'] for c in kr_cases]);ts=np.array([c['time'] for c in sc_cases]);es=np.array([c['event'] for c in sc_cases])
    fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'};fk=np.array([fold[c['patient_id']] for c in kr_cases])
    lp=np.zeros(len(kr_cases))
    for f in cfg['folds']:  # cross-fitting: coefficients from the other four folds' OOF data
        m=fk!=f;b,_,_=cox_fit(np.c_[Lk[m],zk[m]],tk[m],ek[m]);lp[~m]=np.c_[Lk[~m],zk[~m]]@b
    bK,seK,_=cox_fit(np.c_[Lk,zk],tk,ek);lps=np.c_[Ls,zs]@bK
    bS,seS,llS=cox_fit(np.c_[Ls,zs],ts,es);_,_,llS0=cox_fit(Ls[:,None],ts,es)
    hr=float(np.exp(bS[1]));ci=[float(np.exp(bS[1]-1.96*seS[1])),float(np.exp(bS[1]+1.96*seS[1]))];p=2*norm_sf(abs(bS[1]/seS[1]))
    cK0=cindex(kr_cases,Lk);cK1=cindex(kr_cases,lp);cS0=cindex(sc_cases,Ls);cS1=cindex(sc_cases,lps)
    dS=lambda ix:harrell_c(ts[ix],es[ix],lps[ix])-harrell_c(ts[ix],es[ix],Ls[ix])
    dK=lambda ix:harrell_c(tk[ix],ek[ix],lp[ix])-harrell_c(tk[ix],ek[ix],Lk[ix])
    med=float(np.median(zk));chi,lrp=logrank(ts,es,zs>med)
    return dict(korea=dict(c_baseline=cK0,c_fused=cK1,delta_c=cK1-cK0,delta_c_ci95=boot_ci(kr_cases,dK,A['bootstrap_replicates'],A['bootstrap_seed']),
                           c_he_alone=cindex(kr_cases,zk),cox_coef_korea=[float(x) for x in bK],cox_se_korea=[float(x) for x in seK]),
                scotland=dict(c_baseline=cS0,c_fused_frozen=cS1,delta_c=cS1-cS0,delta_c_ci95=boot_ci(sc_cases,dS,A['bootstrap_replicates'],A['bootstrap_seed']),
                              c_he_alone=cindex(sc_cases,zs),hr_he_per_sd=hr,hr_he_ci95=ci,hr_he_p=float(p),
                              lr_test_p=float(math.erfc(math.sqrt(max(0.,2*(llS-llS0))/2))),
                              logrank_split_at_korean_median=dict(chi2=chi,p=lrp,high_n=int((zs>med).sum()))))

def decide(r,cfg):
    D=cfg['decision'];s=r['scotland']
    if s['delta_c']>=D['min_delta_c']-1e-12 and (s['hr_he_ci95'][0]>1 or not D['require_hr_ci_above_1']):v='POSITIVE'
    elif s['delta_c']<=0 or s['hr_he_per_sd']<=1:v='NEGATIVE'
    else:v='INCONCLUSIVE'
    return v

def if_subset(cfg,cases,z,name):
    """Exploratory IF analysis on one cohort's IF subset (PROTOCOL.md section 6; same HE_z as the main analysis); no decision."""
    rows=labels(cfg,name);sub=[(c,h) for c,h in zip(cases,z) if c['patient_id'] in rows]
    cs=[c for c,_ in sub];h=np.array([x for _,x in sub]);pct=np.array([float(rows[c['patient_id']]['p21_mcm2_percentage']) for c in cs])
    out=dict(n=len(cs),events=int(sum(c['event'] for c in cs)),c_he=cindex(cs,h),c_p21_low_is_risk=cindex(cs,-pct),
             c_leibovich=cindex(cs,[c[cfg['analysis']['baseline']] for c in cs]),spearman_he_p21=spearman(h,pct))
    if out['events']>=5:
        lg=np.log(pct+cfg['analysis']['if_offset']);b,se,_=cox_fit(np.c_[h,lg],[c['time'] for c in cs],[c['event'] for c in cs])
        out['cox_he_hr']=[float(np.exp(b[0])),float(np.exp(b[0]-1.96*se[0])),float(np.exp(b[0]+1.96*se[0]))]
        out['cox_log_p21_hr']=[float(np.exp(b[1])),float(np.exp(b[1]-1.96*se[1])),float(np.exp(b[1]+1.96*se[1]))]
    return out

def analyse(cfg,cases_all,assign,scot,kr,sc):
    seeds=cfg['seeds'];folds=cfg['folds'];base=cfg['analysis']['baseline'];prim=primary(cases_all,cfg)
    res=dict(encoders={},n=dict(korea_primary=len(prim),korea_all=len(cases_all),scotland=len(scot)))
    Ls=np.array([c[base] for c in scot]);ts=[c['time'] for c in scot];es=[c['event'] for c in scot]
    for e in cfg['encoders']:
        raw_k={p:float(np.mean([kr[(e,s)][p] for s in seeds])) for p in kr[(e,seeds[0])]}
        raw_s=[float(np.mean([sc[(e,s,f)][c['patient_id']] for s in seeds for f in folds])) for c in scot]
        zk,zs,diag=standardise(cfg,cases_all,assign,raw_k,raw_s);kz=[zk[c['patient_id']] for c in prim]
        r=dict(main=fuse(cfg,prim,kz,scot,zs,base,assign),alt_baseline=fuse(cfg,prim,kz,scot,zs,cfg['analysis']['alt_baseline'],assign))
        sens=[dict(c,time=cfg['zero_dfs_sensitivity_months'],event=1) if c['zero_dfs'] else c for c in cases_all]  # DFS 0 -> 0.5 months
        r['sensitivity_with_zero_dfs']=fuse(cfg,sens,[zk[c['patient_id']] for c in sens],scot,zs,base,assign)
        bK=np.array(r['main']['korea']['cox_coef_korea']);c0=r['main']['scotland']['c_baseline'];pf=[]
        for f in folds:  # each fold's 5 models alone on Scotland, same frozen Korean coefficients
            v=np.array([float(np.mean([sc[(e,s,f)][c['patient_id']] for s in seeds])) for c in scot]);v=(v-v.mean())/v.std()
            pf.append(float(harrell_c(ts,es,np.c_[Ls,v]@bK)-c0))
        mu,sd=float(np.mean([raw_k[c['patient_id']] for c in prim])),float(np.std([raw_k[c['patient_id']] for c in prim]))
        pooled=fuse(cfg,prim,[(raw_k[c['patient_id']]-mu)/sd for c in prim],scot,(np.asarray(raw_s)-mu)/sd,base,assign)['scotland']
        r['scale']=dict(diag,per_fold_scotland_delta_c=pf,
                        pooled_korean_standardisation=dict(delta_c=pooled['delta_c'],hr_he_per_sd=pooled['hr_he_per_sd'],hr_he_ci95=pooled['hr_he_ci95']))
        r['per_seed']=dict(korea_c_he=[],scotland_c_he=[])
        for s in seeds:
            a,b_,_=standardise(cfg,cases_all,assign,{p:kr[(e,s)][p] for p in raw_k},[float(np.mean([sc[(e,s,f)][c['patient_id']] for f in folds])) for c in scot])
            r['per_seed']['korea_c_he'].append(cindex(prim,[a[c['patient_id']] for c in prim]));r['per_seed']['scotland_c_he'].append(cindex(scot,b_))
        r['if_exploratory']=dict(korea=if_subset(cfg,prim,kz,'korea_if.csv'),scotland=if_subset(cfg,scot,zs,'scotland_if.csv'))
        res['encoders'][e]=r
    res['decision']=dict(verdict=decide(res['encoders'][cfg['primary_encoder']]['main'],cfg),encoder=cfg['primary_encoder'],rule='PROTOCOL.md section 5')
    res['cox_warnings']=list(COX_WARNINGS)
    return res

def markdown(res,cfg):
    L=['# HE + 临床预测复发：结果（预先登记的分析）','',f"生成时间 {res['finished']}；代码签名 `{res['code_signature'][:16]}`；平台 {res['platform']}。",
       f"韩国开发 {res['n']['korea_primary']} 例（主分析）；苏格兰外部验证 {res['n']['scotland']} 例；84 个月截尾。",'',
       f"**判定：{res['decision']['verdict']}**（主编码器 {cfg['primary_encoder']}；规则见 PROTOCOL.md 第 5 节）",'']
    for e,r in res['encoders'].items():
        m=r['main'];k=m['korea'];s=m['scotland']
        L+=[f'## {e}'+('（主编码器）' if e==cfg['primary_encoder'] else '（重复验证）'),'','| | 韩国（交叉拟合） | 苏格兰（模型冻结） |','|---|---|---|',
            f"| Leibovich（Fuhrman 版）C 指数 | {k['c_baseline']:.3f} | {s['c_baseline']:.3f} |",
            f"| Leibovich + HE C 指数 | {k['c_fused']:.3f} | {s['c_fused_frozen']:.3f} |",
            f"| ΔC（95% 区间） | {k['delta_c']:+.3f}（{k['delta_c_ci95'][0]:+.3f} 至 {k['delta_c_ci95'][1]:+.3f}） | {s['delta_c']:+.3f}（{s['delta_c_ci95'][0]:+.3f} 至 {s['delta_c_ci95'][1]:+.3f}） |",
            f"| HE 单独 C 指数 | {k['c_he_alone']:.3f} | {s['c_he_alone']:.3f} |",'',
            f"- 苏格兰 Cox(Leibovich + HE)：HE 每升高 1 个标准差 HR {s['hr_he_per_sd']:.2f}（{s['hr_he_ci95'][0]:.2f}–{s['hr_he_ci95'][1]:.2f}），p = {s['hr_he_p']:.3g}；似然比检验 p = {s['lr_test_p']:.3g}",
            f"- 苏格兰按韩国中位数分高低风险：log-rank χ² = {s['logrank_split_at_korean_median']['chi2']:.2f}，p = {s['logrank_split_at_korean_median']['p']:.3g}（高风险 {s['logrank_split_at_korean_median']['high_n']} 例）",
            f"- 每个种子的 HE 单独 C 指数：韩国 {', '.join('%.3f'%x for x in r['per_seed']['korea_c_he'])}；苏格兰 {', '.join('%.3f'%x for x in r['per_seed']['scotland_c_he'])}",
            f"- ISUP 版 Leibovich 作基线：苏格兰 ΔC {r['alt_baseline']['scotland']['delta_c']:+.3f}；韩国 ΔC {r['alt_baseline']['korea']['delta_c']:+.3f}",
            f"- 敏感性分析（纳入韩国 DFS = 0 的 5 例，按 0.5 个月复发）：韩国 ΔC {r['sensitivity_with_zero_dfs']['korea']['delta_c']:+.3f}；苏格兰 ΔC {r['sensitivity_with_zero_dfs']['scotland']['delta_c']:+.3f}",
            f"- 每折 5 个模型单独用于苏格兰的 ΔC：{', '.join('%+.3f'%x for x in r['scale']['per_fold_scotland_delta_c'])}",
            f"- 尺度诊断：韩国各折原始风险标准差 {', '.join('%.3f'%x for x in r['scale']['korea_fold_sd'])}；苏格兰 25 模型平均风险标准差 {r['scale']['scotland_sd']:.3f}；"
            f"若改用韩国整体均值和标准差标准化，苏格兰 ΔC {r['scale']['pooled_korean_standardisation']['delta_c']:+.3f}、HR {r['scale']['pooled_korean_standardisation']['hr_he_per_sd']:.2f}",'']
        for coh,v in r['if_exploratory'].items():
            L.append(f"- 免疫荧光子集（探索性，{coh}，{v['n']} 例 {v['events']} 次复发）：C 指数 HE {v['c_he']:.3f}、P21+/MCM2− {v['c_p21_low_is_risk']:.3f}、Leibovich {v['c_leibovich']:.3f}；Spearman(HE, P21) {v['spearman_he_p21']:+.2f}"+
                     (f"；Cox HR HE {v['cox_he_hr'][0]:.2f}（{v['cox_he_hr'][1]:.2f}–{v['cox_he_hr'][2]:.2f}）、log P21 {v['cox_log_p21_hr'][0]:.2f}（{v['cox_log_p21_hr'][1]:.2f}–{v['cox_log_p21_hr'][2]:.2f}）" if 'cox_he_hr' in v else ''))
        L.append('')
    if res['cox_warnings']:L+=[f"注意：{len(res['cox_warnings'])} 个 Cox 拟合未收敛或系数过大（见 RESULTS.json cox_warnings）。",'']
    L+=['苏格兰只有 27 次复发，区间较宽；免疫荧光子集事件更少，只作假设。不得据此修改标签、病例、截尾或阈值。']
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
    m=res['encoders'][cfg['primary_encoder']]['main']['scotland']
    print('DECISION',res['decision']['verdict'],'scotland dC %+.3f HR %.2f'%(m['delta_c'],m['hr_he_per_sd']),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
