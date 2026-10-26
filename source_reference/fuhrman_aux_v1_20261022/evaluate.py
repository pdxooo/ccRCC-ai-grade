"""Pre-registered analysis of direction 3 (PROTOCOL.md sections 5 and 6). Run once, after all 150 runs completed.
It refuses (exit 1) unless every run (2 encoders x 3 arms x 5 seeds x 5 folds) has a COMPLETE, hash-verified receipt from
this locked code signature and config, one platform for all runs (the technical check's), and each encoder x arm x seed
OOF covers the 300 cases exactly once with the database labels; and every run of an encoder read the same feature bytes.
Writes RESULTS.json and RESULTS.md; prints the decision. Exit: 0 done; 1 refused (see the printed problems)."""
from common import *
import argparse

def expected_runs(cfg):
    return [(e,a,s,f) for e in cfg['encoders'] for a in cfg['arms'] for s in cfg['seeds'] for f in cfg['folds']]

def collect(cfg,cases,assign):
    sig=signature();problems=[];plats=set();preds={};feats={};auxp={}
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
        want=dict(code=sig,encoder=e,arm=a,seed_index=s,fold=f,check=False,epochs=cfg['epochs'],labels_sha256=cfg['labels_csv_sha256'])
        diff={k:c.get(k) for k,v in want.items() if c.get(k)!=v}
        if diff:problems.append(f'{tag}: contract differs {diff}');continue
        if c.get('config')!=cfg:problems.append(f'{tag}: trained with another config');continue
        test=sorted(x['patient_id'] for x in roles(cases,assign,f,cfg)['test'])
        P=read_json(d/'predictions.json') or []
        if sorted(x['patient_id'] for x in P)!=test or sorted(c['test'])!=test:problems.append(f'{tag}: predictions are not the fold-{f} test cases');continue
        if any(x['y']!=lab[x['patient_id']]['y'] or x['fold']!=f or len(x['probabilities'])!=4 or abs(sum(x['probabilities'])-1)>1e-5 for x in P):
            problems.append(f'{tag}: label/fold/probability mismatch');continue
        plats.add(json.dumps(rc['platform'],sort_keys=True))
        for p,h in (md.get('feature_sha256') or {}).items():feats.setdefault((e,p),set()).add(h)
        for x in P:
            preds.setdefault((e,a,s),{})[x['patient_id']]=x['probabilities'];auxp.setdefault((e,a,s),{})[x['patient_id']]=x['p_aux_high']
    if len(plats)>1:problems.append(f'runs come from {len(plats)} platforms: {sorted(plats)}')
    if any(len(v)>1 for v in feats.values()):problems.append('some case was read with different feature bytes in different runs')
    for k,v in preds.items():
        if len(v)!=len(cases):problems.append(f'{k}: OOF covers {len(v)} of {len(cases)} cases')
    return (preds,auxp),problems,sorted(plats)

def decide(enc_stats,cfg):
    """PROTOCOL.md section 6 (primary encoder only; the other encoder is reported as a replication)."""
    D=cfg['decision'];r=enc_stats[cfg['primary_encoder']]
    d=r['delta_B_minus_A'];pos=sum(1 for x in r['seed_delta_B_minus_A'] if x>0)
    bc=r['delta_B_minus_C'];pos_bc=sum(1 for x in r['seed_delta_B_minus_C'] if x>0)
    if d>=D['min_delta_B_minus_A']-1e-12 and pos>=D['min_positive_seeds'] and bc>=D['min_delta_B_minus_C']-1e-12 and \
            pos_bc>=D['min_positive_seeds_B_minus_C']:verdict='POSITIVE'
    elif d<=0 or bc<=0:verdict='NEGATIVE'
    else:verdict='INCONCLUSIVE'
    return verdict,dict(encoder=cfg['primary_encoder'],delta_B_minus_A=d,positive_seeds=pos,delta_B_minus_C=bc,positive_seeds_B_minus_C=pos_bc)

def analyse(cfg,cases,preds,auxp):
    A=cfg['analysis'];ids=[c['patient_id'] for c in cases];isup=np.array([c['isup'] for c in cases])
    arms=list(cfg['arms']);seeds=cfg['seeds']
    P={e:{a:np.array([[preds[(e,a,s)][p] for p in ids] for s in seeds]) for a in arms} for e in cfg['encoders']}  # (seeds, cases, 4)
    avg={e:{a:P[e][a].mean(0) for a in arms} for e in P}
    rng=np.random.default_rng(A['bootstrap_seed']);B=A['bootstrap_replicates']
    groups=[np.flatnonzero(isup==g) for g in (1,2,3,4)]
    boot={e:{a:np.empty(B) for a in arms} for e in P}
    for b in range(B):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        for e in P:
            for a in arms:boot[e][a][b]=g12_auc(isup[idx],avg[e][a][idx])
    ci=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    stats={}
    for e in P:
        r=dict(arms={})
        for a in arms:
            per=[g12_auc(isup,P[e][a][k]) for k in range(len(seeds))]
            r['arms'][a]=dict(g12_auc=g12_auc(isup,avg[e][a]),g12_auc_ci95=ci(boot[e][a]),g12_auc_per_seed=per,
                              highlow_auc=highlow_auc(isup,avg[e][a]),**grade_metrics(isup,avg[e][a]))
        g=r['arms']
        for x,y in (('B','A'),('B','C'),('C','A')):
            r[f'delta_{x}_minus_{y}']=g[x]['g12_auc']-g[y]['g12_auc'];r[f'delta_{x}_minus_{y}_ci95']=ci(boot[e][x]-boot[e][y])
            r[f'seed_delta_{x}_minus_{y}']=[g[x]['g12_auc_per_seed'][k]-g[y]['g12_auc_per_seed'][k] for k in range(len(seeds))]
        # report only: does the auxiliary head itself predict the test cases' Fuhrman 1-2 vs 3-4 (read only here)?
        z=np.array([c['z'] for c in cases])
        r['aux_head_fuhrman_auc']={a:auc(z,np.mean([[auxp[(e,a,s)][p] for p in ids] for s in seeds],0)) for a in arms if cfg['arms'][a]['aux_weight']>0}
        stats[e]=r
    verdict,why=decide(stats,cfg)
    return dict(n_cases=len(ids),isup_counts={str(g):int((isup==g).sum()) for g in (1,2,3,4)},encoders=stats,
                decision=dict(verdict=verdict,basis=why,rule='PROTOCOL.md section 6'))

def markdown(res,cfg):
    L=['# 方向 3 结果：Fuhrman 辅助监督（预先登记的分析）','',f"生成时间 {res['finished']}；代码签名 `{res['code_signature'][:16]}`；平台 {res['platform']}。",
       f"病例 {res['n_cases']}（ISUP 1–4：{res['isup_counts']}）；每臂 {len(cfg['seeds'])} 个种子的 ISUP 概率平均后在合并 OOF 上计算；区间为按 ISUP 分层的病例 bootstrap {cfg['analysis']['bootstrap_replicates']} 次。",
       '',f"**判定：{res['decision']['verdict']}**（主编码器 {cfg['primary_encoder']}，主要指标 G1 对 G2 AUC；规则见 PROTOCOL.md 第 6 节）",'',
       'A = 只用 ISUP；B = ISUP + 真实 Fuhrman（1–2 对 3–4）辅助；C = ISUP + 在 ISUP 级内打乱的 Fuhrman 辅助（对照）。','']
    for e,r in res['encoders'].items():
        L+=[f'## {e}'+('（主编码器）' if e==cfg['primary_encoder'] else '（重复验证）'),'','| 臂 | G1/G2 AUC | 95% 区间 | 各种子 | 高/低级 AUC | QWK | Macro-F1 | 准确率 |','|---|---|---|---|---|---|---|---|']
        for a,v in r['arms'].items():
            L.append(f"| {a} | {v['g12_auc']:.3f} | {v['g12_auc_ci95'][0]:.3f}–{v['g12_auc_ci95'][1]:.3f} | {', '.join('%.3f'%x for x in v['g12_auc_per_seed'])} | {v['highlow_auc']:.3f} | {v['qwk']:.3f} | {v['macro_f1']:.3f} | {v['accuracy']:.3f} |")
        for x,y in (('B','A'),('B','C'),('C','A')):
            L.append(f"- Δ({x}−{y}) = {r[f'delta_{x}_minus_{y}']:+.3f}（{r[f'delta_{x}_minus_{y}_ci95'][0]:+.3f} 至 {r[f'delta_{x}_minus_{y}_ci95'][1]:+.3f}）；各种子 {', '.join('%+.3f'%v for v in r[f'seed_delta_{x}_minus_{y}'])}")
        L.append('- 辅助头对测试集 Fuhrman（1–2 对 3–4）的 AUC（只作报告）：'+'，'.join('%s %.3f'%(a,v) for a,v in r['aux_head_fuhrman_auc'].items()))
        L.append('')
    L+=['探索性结果：韩国 OOF 此前已用于多轮分析；结论需在苏格兰队列上用同样锁定的比较确认。不得据此修改标签、病例、权重或阈值。']
    return '\n'.join(L)+'\n'

def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    cfg=config();cases,assign=load_cases(cfg);problems=[]
    ok,bad=lock_status()
    if not ok:problems.append(f'code is not the locked version: {bad[:5]}')
    chk=read_json(CHECK_PASS) or {}
    if chk.get('state')!='PASS' or chk.get('code_signature')!=signature():problems.append('CHECK_PASS.json is not a PASS for this code')
    (preds,auxp),more,plats=collect(cfg,cases,assign);problems+=more
    if plats and json.dumps(chk.get('platform'),sort_keys=True) not in plats:problems.append('runs and the technical check ran on different platforms')
    if problems:
        print('REFUSED',len(problems),'problem(s):',flush=True)
        for p in problems[:40]:print(' ',p,flush=True)
        return 1
    res=analyse(cfg,cases,preds,auxp);res.update(code_signature=signature(),platform=[json.loads(p) for p in plats],finished=now(),runs=len(expected_runs(cfg)))
    save_json(HERE/'RESULTS.json',res);(HERE/'RESULTS.md').write_text(markdown(res,cfg))
    print('DECISION',res['decision']['verdict'],json.dumps(res['decision']['basis']),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
