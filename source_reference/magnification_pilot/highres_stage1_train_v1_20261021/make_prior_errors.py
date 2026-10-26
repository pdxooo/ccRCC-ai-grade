"""Builds inputs/prior_all_wrong_isup2_v1.json (secondary metric, PROTOCOL.md section 5) from the earlier first-round OOF
test predictions (mil_first_round_v2_20261015, copied read-only from lxh_hd4 into inputs/; sha256 recorded and checked).
Rule (as in takeover_20261021/训练方法与准确率诊断_20261021.md): a patient is 'all wrong' when the 4-grade argmax of every
one of the 6 pathology encoders (uni, uni2-h, phikon-v2, conch, virchow2, gigapath) differs from the true ISUP grade.
Only ISUP-2 patients enter the list. Expected: 71 all wrong, 46 of them ISUP 2. Uses only results that existed before
the high-res pilot; nothing from stage 1."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;SRC=HERE/'inputs/mil_first_round_v2_test_predictions'
ENC=['uni','uni2-h','phikon-v2','conch','virchow2','gigapath']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    hpc={l.split()[1]:l.split()[0] for l in (HERE/'inputs/hpc_sha256.txt').read_text().splitlines() if 'test_predictions.json' in l}
    files={f'{e}/fold{k}/test_predictions.json' for e in ENC for k in range(5)}
    assert set(hpc)==files and all(sha(SRC/f)==hpc[f] for f in files),'local copies differ from the HPC sha256'
    wrong={};grade={};called={}
    for e in ENC:
        recs=[r for k in range(5) for r in json.loads((SRC/e/f'fold{k}'/'test_predictions.json').read_text())]
        assert len(recs)==300 and len({r['patient_id'] for r in recs})==300
        for r in recs:
            g=int(r['ISUP_grade']);assert grade.setdefault(r['patient_id'],g)==g
            p=r['probabilities'];a=max(range(4),key=lambda i:p[i])
            wrong.setdefault(r['patient_id'],[]).append(a!=g-1);called.setdefault(r['patient_id'],[]).append(a)
    allw=sorted(p for p,v in wrong.items() if len(v)==6 and all(v));g2=[p for p in allw if grade[p]==2]
    assert (len(allw),len(g2))==(71,46),(len(allw),len(g2))
    out=dict(rule=__doc__.strip(),encoders=ENC,source='lxh_hd4:K/bib_phase1/mil_first_round_v2_20261015/runs/<encoder>/fold<k>/test_predictions.json',
             source_sha256={f:hpc[f] for f in sorted(files)},all_wrong=len(allw),isup2_all_wrong=g2,
             isup2_all_called_G1=[p for p in g2 if all(a==0 for a in called[p])])
    (HERE/'inputs/prior_all_wrong_isup2_v1.json').write_text(json.dumps(out,indent=1,sort_keys=True)+'\n')
    print('all wrong',len(allw),'ISUP2',len(g2))
if __name__=='__main__':main()
