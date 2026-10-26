"""Pre-registered analysis for fusion_gain_v2: pooled OOF over 300 patients, paired patient bootstrap.
Campaign check (it refuses to pool otherwise): each of the 100 runs (5 arms x 4 seeds x 5 folds) needs a COMPLETE, non-smoke
receipt whose artifacts (incl. metadata.json and test_predictions.json) hash-verify; its hashed metadata.json must carry
this code signature (contract.code == signature()), contract.smoke False, the run's own arm/seed/fold and the receipt
fingerprint; all 100 runs must share one platform, one split hash and one QC-pool hash; and SMOKE_PASS.json must still
record PASS, for this code signature and that same platform, for every smoke part covering an arm (a later FAIL revokes it).
Runs resumed on the other card (receipt cross_card_resume) are listed in RESULTS.json campaign, with a WARNING if no smoke
part proved cross-card bitwise resume."""
from common import *
def smoke_part(arm,cfg):return 'single' if cfg['arms'][arm]['fusion']=='none' else arm  # same rule as train.smoke_part (no torch import here)
def oof(arm,s,campaign):
    recs=[]
    for k in range(5):
        root=HERE/'runs'/arm.replace(':','_')/f'seed{s}'/f'fold{k}';done=json.loads((root/'complete.json').read_text())
        assert done.get('state')=='COMPLETE' and done.get('smoke') is False,f'{root}: receipt is not a COMPLETE formal run'
        assert {'metadata.json','test_predictions.json'}<=set(done['artifacts']),f'{root}: receipt does not hash metadata.json and test_predictions.json'
        for name,digest in done['artifacts'].items():assert sha(root/name)==digest,f'{root/name}: sha256 differs from the receipt'
        md=json.loads((root/'metadata.json').read_text());c=md['contract']
        assert md['fingerprint']==done['fingerprint'],f'{root}: metadata fingerprint differs from the receipt'
        assert c['code']==signature(),f'{root}: trained with code {c["code"][:12]}, this code is {signature()[:12]} (not one campaign)'
        assert c['smoke'] is False and (c['arm'],c['seed_index'],c['fold'])==(arm,s,k),f'{root}: contract is not the formal run {arm}/seed{s}/fold{k}'
        for key,value in (('platform',md['platform']),('split_sha256',c['split_sha256']),('pool_sha256',c['pool_sha256'])):campaign[key].add(json.dumps(value,sort_keys=True))
        campaign['physical_gpus'].update(done.get('physical_gpus') or [])
        if done.get('cross_card_resume'):campaign['cross_card_resumes'].append(f'{arm}|seed{s}|fold{k}')
        recs+=json.loads((root/'test_predictions.json').read_text())
    recs.sort(key=lambda r:r['patient_id']);assert len(recs)==300 and len({r['patient_id'] for r in recs})==300
    return [r['patient_id'] for r in recs],np.array([r['ISUP_grade']-1 for r in recs]),np.array([r['probabilities'] for r in recs])
def check_campaign(cfg,campaign):
    for key in ('platform','split_sha256','pool_sha256'):assert len(campaign[key])==1,f'runs disagree on {key}: {sorted(campaign[key])} (not one campaign)'
    platform=json.loads(next(iter(campaign['platform'])));sig=signature()
    sp=json.loads((HERE/'SMOKE_PASS.json').read_text());assert sp.get('code_signature')==sig,'SMOKE_PASS.json belongs to other code'
    smoke={}
    for part in sorted({smoke_part(a,cfg) for a in cfg['arms']}):
        rec=sp['parts'].get(part,{})
        assert rec.get('state')=='PASS' and rec.get('code_signature')==sig,f'smoke part {part} is {rec.get("state")} for this code (a FAIL after the runs revokes them)'
        assert rec.get('platform')==platform,f'smoke part {part} ran on {rec.get("platform")}, the runs on {platform}'
        smoke[part]=dict(attempt=rec.get('attempt'),cross_card_resume=rec.get('cross_card_resume'),finished=rec.get('finished'))
    return dict(runs=len(cfg['arms'])*len(cfg['seeds'])*5,code_signature=sig,platform=platform,split_sha256=json.loads(next(iter(campaign['split_sha256']))),
                pool_sha256=json.loads(next(iter(campaign['pool_sha256']))),smoke_parts=smoke,physical_gpus=sorted(campaign['physical_gpus']),cross_card_resumes=campaign['cross_card_resumes'],
                cross_card_resume_proven_by_smoke=all(v['cross_card_resume'] is True for v in smoke.values()))
def auc(score,pos):
    p=score[pos];n=score[~pos];return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def scores(y,P):
    m=metrics(y,P);return dict(auc_high_low=auc(P[:,2:].sum(1),y>=2),qwk=m['quadratic_kappa'],macro_f1=m['macro_f1'],accuracy=m['accuracy'],confusion=m['confusion_matrix'])
def main():
    cfg=config();ids=None;P={};Y=None;campaign=dict(platform=set(),split_sha256=set(),pool_sha256=set(),physical_gpus=set(),cross_card_resumes=[])
    for arm in cfg['arms']:
        for s in cfg['seeds']:
            i,y,p=oof(arm,s,campaign)
            if ids is None:ids,Y=i,y
            assert i==ids and (y==Y).all();P[(arm,s)]=p
    camp=check_campaign(cfg,campaign)
    ref='single:'+cfg['reference_single'];singles=['single:'+m for m in cfg['members']]
    S=cfg['seeds'];avg=lambda keys:np.mean([P[k] for k in keys],0)
    sets={'ref_seed0':avg([(ref,0)]),'ref_4seeds':avg([(ref,s) for s in S]),
          'late4_seed0':avg([(a,0) for a in singles]),'late4x4seeds':avg([(a,s) for a in singles for s in S]),
          'mdrop4_4seeds':avg([('mdrop4',s) for s in S])}
    for a in singles:sets[a+'_4seeds']=avg([(a,s) for s in S])
    out=dict(per_set={k:scores(Y,v) for k,v in sets.items()},per_run={f'{a}|seed{s}':scores(Y,p) for (a,s),p in P.items()})
    rng=np.random.default_rng(20261021);B=[rng.integers(0,300,300) for _ in range(2000)]
    def boot(a,b):
        res={}
        for metric in ('auc_high_low','qwk','macro_f1'):
            def f(P_,ix):
                y=Y[ix];p=P_[ix]
                if metric=='auc_high_low':return auc(p[:,2:].sum(1),y>=2)
                m=metrics(y,p);return m['quadratic_kappa'] if metric=='qwk' else m['macro_f1']
            d=np.array([f(sets[a],ix)-f(sets[b],ix) for ix in B]);point=out['per_set'][a][metric]-out['per_set'][b][metric]
            res[metric]=dict(diff=point,ci95=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],p_le_0=float((d<=0).mean()))
        return res
    out['preregistered']={'P1_encoder_vs_seed_diversity':boot('late4_seed0','ref_4seeds'),
                          'P2_full_late_fusion':boot('late4x4seeds','ref_4seeds'),
                          'P3_learned_fusion':boot('mdrop4_4seeds','ref_4seeds')}
    out['campaign']=camp
    out['note']='Exploratory: Korean OOF folds were inspected in earlier rounds. Confirm on Scotland with the same locked comparisons.'
    save_json(HERE/'RESULTS.json',out);print(json.dumps(out['preregistered'],indent=1));print(json.dumps({k:{m:round(v[m],4) for m in ('auc_high_low','qwk','macro_f1','accuracy')} for k,v in out['per_set'].items()},indent=1))
    print('CAMPAIGN',json.dumps({k:camp[k] for k in ('runs','code_signature','physical_gpus','cross_card_resumes','cross_card_resume_proven_by_smoke')}),flush=True)
    if camp['cross_card_resumes'] and not camp['cross_card_resume_proven_by_smoke']:
        print('WARNING',len(camp['cross_card_resumes']),'run(s) resumed on the other card, but no smoke part proved cross-card bitwise resume (run_smoke.sh 0,1):',camp['cross_card_resumes'],flush=True)
if __name__=='__main__':main()
