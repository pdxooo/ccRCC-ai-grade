"""Freeze a conservative technical-eligibility pool; does not certify pathology quality."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv, gzip, json, hashlib, sys
from pathlib import Path
ROOT=Path('/data/cluster/projects/korean_ccrcc_SBIAD2452')
sys.path.insert(0,str(ROOT/'qc_deps'))
import numpy as np
SOURCE=ROOT/'qc/native_qc_v2_all300_20261009'
AUDIT=ROOT/'qc/review_all300_20261010'
OUT=ROOT/'qc/primary_technical_pool_v1_20261010'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=json.loads((AUDIT/'audit_summary.json').read_text())
    assert audit['audited']==300 and audit['record_pass']==300 and not audit['errors']
    review=json.loads((AUDIT/'visual_review_gate.json').read_text())
    assert review['allow_conservative_technical_export'] is True
    assert not (OUT/'LOCK.json').exists(),'Already locked: never overwrite frozen export'
    OUT.mkdir(exist_ok=True)
    for folder in ('primary','review_reserve','excluded'): (OUT/folder).mkdir(exist_ok=True)
    manifest=ROOT/'bib_phase1/korea_model_manifest.csv'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    results=[];groups={};reasons={}
    for row in rows:
        stem=Path(row['slide_filename']).stem
        folder=SOURCE/stem;r=json.loads((folder/'report.json').read_text())
        assert r['patient_id']==row['patient_id'] and r['source_sha256']==row['source_sha256']
        buckets={'primary':[],'review_reserve':[],'excluded':[]};flags=[]
        combos={}
        with gzip.open(folder/'patch_qc.csv.gz','rt') as f:
            for item in csv.DictReader(f):
                coord=tuple(int(item[k]) for k in ('scene','x','y','w','h'))
                decision=item['decision'];reason=item['reason']
                bucket='primary' if decision=='PASS_AUTO' else 'review_reserve' if decision=='REVIEW' else 'excluded'
                buckets[bucket].append(coord)
                if bucket=='review_reserve':
                    flags.append(sum(bit for name,bit in [('low_focus_score',1),('sparse_or_pale',2),('black_edge',4)] if name in reason.split(';')))
                    combos[reason]=combos.get(reason,0)+1
                    reasons[reason]=reasons.get(reason,0)+1
        assert sum(map(len,buckets.values()))==r['processed']
        assert len(buckets['primary'])==r['counts'].get('PASS_AUTO',0)
        item=dict(patient_id=row['patient_id'],clinical_id=row['clinical_id'],slide_filename=row['slide_filename'],
                  ISUP_grade=row['ISUP_grade'],source_sha256=row['source_sha256'],candidate_count=r['processed'],
                  primary_count=len(buckets['primary']),review_count=len(buckets['review_reserve']),excluded_count=len(buckets['excluded']),
                  mpp_x=r['mpp']['X'],mpp_y=r['mpp']['Y'],field_um=128,
                  label_scope='patient_level_weak_label',eligibility='TECHNICAL_PRIMARY_POOL_NOT_EXPERT_CERTIFIED')
        assert item['primary_count']>0,'Patient has no primary patches; requires an explicit case decision'
        for bucket,coords in buckets.items():
            a=np.asarray(coords,dtype=np.int64).reshape((-1,5))
            assert np.all(a[:,3:]>0)
            p=OUT/bucket/(stem+'.npz');tmp=p.with_suffix('.tmp')
            payload={'coords':a,'columns':np.array(['scene','x','y','w','h'])}
            if bucket=='review_reserve':payload['reason_bits']=np.asarray(flags,dtype=np.uint8)
            with tmp.open('wb') as f:np.savez_compressed(f,**payload)
            tmp.replace(p)
            with np.load(p,allow_pickle=False) as check:
                assert np.array_equal(check['coords'],a)
            item[bucket+'_path']=str(p);item[bucket+'_sha256']=sha(p)
        g=groups.setdefault(row['ISUP_grade'],dict(patients=0,candidates=0,primary=0,review=0))
        g['patients']+=1;g['candidates']+=r['processed'];g['primary']+=item['primary_count'];g['review']+=item['review_count']
        results.append(item)
        print(json.dumps({'slide':stem,'primary':item['primary_count'],'review':item['review_count']}),flush=True)
    p=OUT/'patient_manifest.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    summary=dict(patients=len(results),primary_patches=sum(r['primary_count'] for r in results),
                 review_reserved=sum(r['review_count'] for r in results),excluded=sum(r['excluded_count'] for r in results),
                 min_primary=min(r['primary_count'] for r in results),max_primary=max(r['primary_count'] for r in results),
                 grade_retention_audit=groups,review_reason_combinations=reasons,
                 scope='Fixed conservative technical pool. REVIEW excluded from primary analysis, retained for sensitivity study. No per-patch grade or tumor label. Not a final clinical/pathologist certificate.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lock=dict(version='primary_technical_pool_v1_20261010',source_manifest_sha256=sha(manifest),patient_manifest_sha256=sha(p),
              source_qc_code_sha256=audit['code_hashes'],export_code_sha256=sha(Path(__file__)),visual_review_gate_sha256=sha(AUDIT/'visual_review_gate.json'),
              selection='PASS_AUTO only; original thresholds unchanged; no label-dependent filtering',
              coordinate_order='original QC spatial order, scene,x,y,w,h in native pixel coordinates',
              model_input_rule='All encoders use identical primary coords and 128 um FOV; encoder-specific resize/normalization occurs later',
              split_rule='Patient-level only. Multiple scenes remain in one patient. No train/test assignment made by this export.')
    (OUT/'LOCK.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
