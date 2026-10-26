"""Per-case exports behind the manuscript (PLAN.md section D), read through each locked study's own gates
(evaluate.collect: every run COMPLETE, hash-verified, from the locked code and config); nothing is trained or refitted.
Usage: python export_predictions.py gx1   -> outputs/grading_*.csv, outputs/korea_split.csv, outputs/grading_runs.csv
       python export_predictions.py surv1 -> outputs/recurrence_he_risk.csv, outputs/recurrence_runs.csv
The two parts run as separate processes because both studies name their modules common.py and evaluate.py."""
import sys,csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
STUDY={'gx1':HERE.parent/'grade_ext_v1_20261022','surv1':HERE.parent/'surv_he_clin_v1_20261022'}

def write(name,header,rows):
    with (OUT/name).open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
    print('WROTE',name,len(rows),'rows',flush=True)
def f6(x):return f'{float(x):.6f}'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(C,cfg,name):
    rows=[]
    for e in cfg['encoders']:
        for s in cfg['seeds']:
            for f in cfg['folds']:
                d=C.RUNS/e/f'seed{s}'/f'fold{f}';rc=json.loads((d/'complete.json').read_text())
                plat=rc.get('platform') or {};gpu=plat.get('gpu') if isinstance(plat,dict) else None
                rows.append([e,s,f,rc.get('state'),rc.get('fingerprint'),sha(d/'complete.json'),sha(d/'predictions.json'),gpu,rc.get('finished') or rc.get('completed') or ''])
    write(name,['encoder','seed','fold','state','contract_fingerprint','complete_json_sha256','predictions_json_sha256','gpu','finished'],rows)

def gx1():
    sys.path.insert(0,str(STUDY['gx1']));import common as C,evaluate as E;np=C.np
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,plats=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'};surv={c['patient_id'] for c in C.primary(cases,cfg)}
    write('korea_split.csv',['patient_id','fold','role'],[[r['patient_id'],r['fold'],r['role']] for r in sorted(assign,key=lambda r:(r['patient_id'],int(r['fold'])))])
    per=[];ens=[]
    for e in cfg['encoders']:
        for c in cases:
            pid=c['patient_id'];ps=[kr[(e,s)][pid] for s in cfg['seeds']]
            for s,p in zip(cfg['seeds'],ps):per.append([e,'korea',pid,s,fold[pid]]+[f6(x) for x in p])
            m=np.mean(ps,0);ens.append([e,'korea',pid,fold[pid],c['isup'],c['fuhrman'],c['leibovich_fuhrman'],int(pid in surv),c['time'],c['event']]+[f6(x) for x in m]+[f6(C.grade_score(m[None])[0])])
        for c in scot:
            pid=c['patient_id'];ps=[]
            for s in cfg['seeds']:
                for f in cfg['folds']:p=sc[(e,s,f)][pid];ps.append(p);per.append([e,'scotland',pid,s,f]+[f6(x) for x in p])
            m=np.mean(ps,0);ens.append([e,'scotland',pid,'',c['isup'],c['fuhrman'],c['leibovich_fuhrman'],1,c['time'],c['event']]+[f6(x) for x in m]+[f6(C.grade_score(m[None])[0])])
    write('grading_predictions_per_run.csv',['encoder','cohort','patient_id','seed','fold','p1','p2','p3','p4'],per)
    write('grading_predictions.csv',['encoder','cohort','patient_id','korean_test_fold','isup','fuhrman','leibovich_fuhrman','in_recurrence_set',
                                     'time_months_censored84','recurrence','p1','p2','p3','p4','ai_grade'],ens)
    manifest(C,cfg,'grading_runs.csv');print('PLATFORMS',plats)

def surv1():
    sys.path.insert(0,str(STUDY['surv1']));import common as C,evaluate as E;np=C.np
    cfg=C.config();cases,assign=C.load_korea(cfg);scot=C.load_scotland(cfg)
    kr,sc,problems,plats=E.collect(cfg,cases,assign,scot);assert not problems,problems[:3]
    fold={r['patient_id']:int(r['fold']) for r in assign if r['role']=='test'};prim={c['patient_id'] for c in C.primary(cases,cfg)};rows=[]
    for e in cfg['encoders']:
        raw_k={p:float(np.mean([kr[(e,s)][p] for s in cfg['seeds']])) for p in kr[(e,cfg['seeds'][0])]}
        raw_s=[float(np.mean([sc[(e,s,f)][c['patient_id']] for s in cfg['seeds'] for f in cfg['folds']])) for c in scot]
        zk,zs,_=E.standardise(cfg,cases,assign,raw_k,raw_s)
        for c in cases:rows.append([e,'korea',c['patient_id'],fold[c['patient_id']],int(c['patient_id'] in prim),c['isup'],c['leibovich_fuhrman'],c['time'],c['event'],f6(raw_k[c['patient_id']]),f6(zk[c['patient_id']])])
        for c,r,z in zip(scot,raw_s,zs):rows.append([e,'scotland',c['patient_id'],'',1,c['isup'],c['leibovich_fuhrman'],c['time'],c['event'],f6(r),f6(z)])
    write('recurrence_he_risk.csv',['encoder','cohort','patient_id','korean_test_fold','in_recurrence_set','isup','leibovich_fuhrman',
                                    'time_months_censored84','recurrence','he_risk_raw','he_risk_standardised'],rows)
    manifest(C,cfg,'recurrence_runs.csv');print('PLATFORMS',plats)

if __name__=='__main__':
    {'gx1':gx1,'surv1':surv1}[sys.argv[1]]()
