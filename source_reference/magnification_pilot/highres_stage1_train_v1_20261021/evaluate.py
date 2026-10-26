"""Pre-registered analysis of high-res stage 1 (PROTOCOL.md sections 5 and 6). Run once, after all 120 runs completed.
It refuses (exit 1) unless every run (2 encoders x 4 arms x 3 seeds x 5 folds) has a COMPLETE, hash-verified receipt from
this locked code signature, one platform for all runs, the locked case list, and each encoder x arm x seed OOF covers the
120 cases exactly once with the locked labels. Writes RESULTS.json and RESULTS.md; prints the decision.
Exit: 0 done; 1 refused (see the printed problems)."""
from common import *
import argparse

def expected_runs(cfg):
    return [(e,a,s,f) for e in cfg['encoders'] for a in cfg['decision_arms']+cfg['reference_arms'] for s in cfg['seeds'] for f in cfg['folds']]

def collect(cfg,cases):
    sig=signature();problems=[];plats=set();preds={}
    lab={c['patient_id']:c for c in cases}
    for e,a,s,f in expected_runs(cfg):
        d=RUNS/e/a/f'seed{s}'/f'fold{f}';tag=f'{e}/{a}/seed{s}/fold{f}'
        rc=read_json(d/'complete.json')
        if not rc or rc.get('state')!='COMPLETE' or rc.get('check') is not False:problems.append(f'{tag}: no COMPLETE formal receipt');continue
        bad=[n for n,h in rc['artifacts'].items() if not (d/n).exists() or sha(d/n)!=h]
        if bad:problems.append(f'{tag}: artifacts changed {bad}');continue
        md=read_json(d/'metadata.json') or {};c=md.get('contract') or {}
        if hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()!=md.get('fingerprint') or md.get('fingerprint')!=rc['fingerprint']:
            problems.append(f'{tag}: fingerprint does not match its contract');continue
        want=dict(code=sig,encoder=e,arm=a,seed_index=s,fold=f,check=False,epochs=cfg['epochs'],list_sha256=cfg['stage1_list_sha256'])
        diff={k:c.get(k) for k,v in want.items() if c.get(k)!=v}
        if diff:problems.append(f'{tag}: contract differs {diff}');continue
        if c.get('config')!=cfg:problems.append(f'{tag}: trained with another config');continue
        test=sorted(x['patient_id'] for x in roles(cases,f)['test'])
        P=read_json(d/'predictions.json') or []
        if sorted(x['patient_id'] for x in P)!=test or sorted(c['test'])!=test:problems.append(f'{tag}: predictions are not the fold-{f} test cases');continue
        if any(x['y']!=lab[x['patient_id']]['y'] or x['fold']!=f or not (0<=x['p_G2']<=1) for x in P):problems.append(f'{tag}: label/fold/probability mismatch');continue
        plats.add(json.dumps(rc['platform'],sort_keys=True))
        for x in P:preds.setdefault((e,a,s),{})[x['patient_id']]=x['p_G2']
    if len(plats)>1:problems.append(f'runs come from {len(plats)} platforms: {sorted(plats)}')
    # every run of an encoder x arm used the same feature bytes, and all runs the same extraction records
    feats={};recs=set()
    for e,a,s,f in expected_runs(cfg):
        md=read_json(RUNS/e/a/f'seed{s}'/f'fold{f}'/'metadata.json') or {}
        for p,h in (md.get('feature_sha256') or {}).items():feats.setdefault((e,a,p),set()).add(h)
        for p,r in ((md.get('contract') or {}).get('records') or {}).items():recs.add((p,a=='Lfull',json.dumps(r,sort_keys=True)))
    if any(len(v)>1 for v in feats.values()):problems.append('some case was read with different feature bytes in different runs')
    per={}
    for p,full,r in recs:per.setdefault((p,full),set()).add(r)
    if any(len(v)>1 for v in per.values()):problems.append('some case was trained/tested under different extraction records')
    for k,v in preds.items():
        if len(v)!=len(cases):problems.append(f'{k}: OOF covers {len(v)} of {len(cases)} cases')
    return preds,problems,sorted(plats)

def decide(enc_stats,cfg):
    """PROTOCOL.md section 6, verbatim logic."""
    D=cfg['decision'];per={}
    for e,st in enc_stats.items():
        d=st['delta_H_minus_L'];ds=st['seed_delta_H_minus_L'];h=st['delta_H_minus_Hd']
        all_pos=all(x>0 for x in ds);consistent=all_pos or all(x<=0 for x in ds)
        advance=d>=D['min_delta_H_minus_L']-1e-12 and (all_pos or not D['all_seed_deltas_positive']) and (h>0 or not D['H_must_beat_Hd'])
        negative=d<D['min_delta_H_minus_L']-1e-12 and (d<=0 or not consistent)  # user decision 2026-10-21: a mean gain >= 0.03 is never 'negative'
        per[e]=dict(advance=bool(advance),negative=bool(negative),delta=d,seed_deltas=ds,H_minus_Hd=h,all_seed_deltas_positive=all_pos,seed_directions_consistent=consistent)
    if any(v['advance'] for v in per.values()):verdict='ADVANCE_TO_STAGE2'
    elif all(v['negative'] for v in per.values()):verdict='NEGATIVE_STOP'
    else:verdict='INCONCLUSIVE_USER_DECIDES'
    return verdict,per

def analyse(cfg,cases,preds):
    A=cfg['analysis'];ids=[c['patient_id'] for c in cases];y=np.array([c['y'] for c in cases])
    arms=cfg['decision_arms']+cfg['reference_arms'];seeds=cfg['seeds']
    P={e:{a:np.array([[preds[(e,a,s)][p] for p in ids] for s in seeds]) for a in arms} for e in cfg['encoders']}
    avg={e:{a:P[e][a].mean(0) for a in arms} for e in P}
    out=dict(n_cases=len(ids),n_G1=int((y==0).sum()),n_G2=int((y==1).sum()),encoders={})
    i0=np.flatnonzero(y==0);i1=np.flatnonzero(y==1);rng=np.random.default_rng(A['bootstrap_seed']);B=A['bootstrap_replicates']
    boot={e:{a:np.empty(B) for a in arms} for e in P}
    for b in range(B):
        idx=np.concatenate([rng.choice(i0,len(i0),replace=True),rng.choice(i1,len(i1),replace=True)])
        for e in P:
            for a in arms:boot[e][a][b]=auc(y[idx],avg[e][a][idx])
    ci=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    pj=json.loads(PRIOR.read_text());prior=set(pj['isup2_all_wrong']);under=set(pj['isup2_all_called_G1'])
    prior_in=[p for k,p in enumerate(ids) if p in prior and y[k]==1]
    under_in=[p for p in prior_in if p in under]  # all 6 encoders called them G1: the G1/G2 errors proper
    prior_label_mismatch=[p for k,p in enumerate(ids) if p in prior and y[k]!=1]  # must be empty: the list holds ISUP-2 patients
    t=A['recall_threshold'];stats={}
    for e in P:
        r=dict(arms={})
        for a in arms:
            s_auc=[auc(y,P[e][a][k]) for k in range(len(seeds))]
            r['arms'][a]=dict(auc_seed_averaged=auc(y,avg[e][a]),auc_ci95=ci(boot[e][a]),auc_per_seed=s_auc,
                              recall_G2_at_0_5=float((avg[e][a][y==1]>=t).mean()),recall_G1_at_0_5=float((avg[e][a][y==0]<t).mean()),
                              prior_all_wrong_G2_called_G2=int(sum(avg[e][a][ids.index(p)]>=t for p in prior_in)),
                              prior_all_called_G1_G2_called_G2=int(sum(avg[e][a][ids.index(p)]>=t for p in under_in)))
        g=r['arms']
        r['delta_H_minus_L']=g['H']['auc_seed_averaged']-g['L']['auc_seed_averaged'];r['delta_H_minus_L_ci95']=ci(boot[e]['H']-boot[e]['L'])
        r['seed_delta_H_minus_L']=[g['H']['auc_per_seed'][k]-g['L']['auc_per_seed'][k] for k in range(len(seeds))]
        r['delta_H_minus_Hd']=g['H']['auc_seed_averaged']-g['Hd']['auc_seed_averaged'];r['delta_H_minus_Hd_ci95']=ci(boot[e]['H']-boot[e]['Hd'])
        r['seed_delta_H_minus_Hd']=[g['H']['auc_per_seed'][k]-g['Hd']['auc_per_seed'][k] for k in range(len(seeds))]
        r['delta_L_minus_Lfull']=g['L']['auc_seed_averaged']-g['Lfull']['auc_seed_averaged'];r['delta_L_minus_Lfull_ci95']=ci(boot[e]['L']-boot[e]['Lfull'])
        stats[e]=r
    out['encoders']=stats;out['prior_all_wrong_G2_in_sample']=len(prior_in);out['prior_all_wrong_G2_ids']=prior_in
    out['prior_list_label_mismatch']=prior_label_mismatch;out['prior_all_called_G1_in_sample']=len(under_in);out['prior_all_called_G1_ids']=under_in
    assert not prior_label_mismatch,f'prior-error list holds non-G2 cases: {prior_label_mismatch}'
    verdict,per=decide(stats,cfg);out['decision']=dict(verdict=verdict,per_encoder=per,rule='PROTOCOL.md section 6')
    return out

def markdown(res,cfg,plats):
    L=['# 高分辨率第 1 阶段结果（预先登记的分析）','',f"生成时间 {res['finished']}；代码签名 `{res['code_signature'][:16]}`；平台 {plats}。",
       f"病例 {res['n_cases']}（G1 {res['n_G1']}，G2 {res['n_G2']}）；每臂 3 个种子概率平均后在合并 OOF 上计算 AUC；区间为分层病例 bootstrap {cfg['analysis']['bootstrap_replicates']} 次。",
       '',f"**判定：{res['decision']['verdict']}**（规则见 PROTOCOL.md 第 6 节；第 1 阶段只看方向）",'']
    for e,r in res['encoders'].items():
        L+=[f'## {e}','','| 臂 | AUC（种子平均） | 95% 区间 | 各种子 AUC | G2 召回 | G1 召回 | 既往全错 G2 判为 G2 | 其中曾被全判为 G1 |','|---|---|---|---|---|---|---|---|']
        for a,v in r['arms'].items():
            L.append(f"| {a} | {v['auc_seed_averaged']:.3f} | {v['auc_ci95'][0]:.3f}–{v['auc_ci95'][1]:.3f} | {', '.join('%.3f'%x for x in v['auc_per_seed'])} | {v['recall_G2_at_0_5']:.3f} | {v['recall_G1_at_0_5']:.3f} | {v['prior_all_wrong_G2_called_G2']}/{res['prior_all_wrong_G2_in_sample']} | {v['prior_all_called_G1_G2_called_G2']}/{res['prior_all_called_G1_in_sample']} |")
        L+=['',f"- Δ(H−L) = {r['delta_H_minus_L']:+.3f}（{r['delta_H_minus_L_ci95'][0]:+.3f} 至 {r['delta_H_minus_L_ci95'][1]:+.3f}）；各种子 {', '.join('%+.3f'%x for x in r['seed_delta_H_minus_L'])}",
            f"- Δ(H−Hd) = {r['delta_H_minus_Hd']:+.3f}（{r['delta_H_minus_Hd_ci95'][0]:+.3f} 至 {r['delta_H_minus_Hd_ci95'][1]:+.3f}）；各种子 {', '.join('%+.3f'%x for x in r['seed_delta_H_minus_Hd'])}",
            f"- 抽样损失 Δ(L−Lfull) = {r['delta_L_minus_Lfull']:+.3f}（{r['delta_L_minus_Lfull_ci95'][0]:+.3f} 至 {r['delta_L_minus_Lfull_ci95'][1]:+.3f}）",
            f"- 判定要素：进入第 2 阶段 {res['decision']['per_encoder'][e]['advance']}，阴性 {res['decision']['per_encoder'][e]['negative']}",'']
    L+=['探索性结果：120 例时 ΔAUC 区间约 ±0.06，不能据此证明效果，也不能据此修改标签、病例、位置或阈值。']
    return '\n'.join(L)+'\n'

def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    cfg=config();cases,_=load_cases(cfg);problems=[]
    ok,bad=lock_status()
    if not ok:problems.append(f'code is not the locked version: {bad[:5]}')
    chk=read_json(CHECK_PASS) or {}
    if chk.get('state')!='PASS' or chk.get('code_signature')!=signature():problems.append('CHECK76_PASS.json is not a PASS for this code')
    preds,more,plats=collect(cfg,cases);problems+=more
    if plats and chk.get('platform') is not None and json.dumps(chk['platform'],sort_keys=True) not in plats:problems.append('runs and the 7.6 check ran on different platforms')
    if problems:
        print('REFUSED',len(problems),'problem(s):',flush=True)
        for p in problems[:40]:print(' ',p,flush=True)
        return 1
    res=analyse(cfg,cases,preds);res.update(code_signature=signature(),platform=[json.loads(p) for p in plats],finished=now(),
                                            runs=len(expected_runs(cfg)),prior_list_sha256=sha(PRIOR))
    save_json(HERE/'RESULTS.json',res);(HERE/'RESULTS.md').write_text(markdown(res,cfg,res['platform']))
    print('DECISION',res['decision']['verdict'],flush=True)
    for e,r in res['encoders'].items():
        print(e,'AUC',{a:round(v['auc_seed_averaged'],3) for a,v in r['arms'].items()},'dH-L %+.3f'%r['delta_H_minus_L'],
              'seeds',[round(x,3) for x in r['seed_delta_H_minus_L']],'dH-Hd %+.3f'%r['delta_H_minus_Hd'],flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
