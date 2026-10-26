"""Writes CAMPAIGN.json: what the lab-a6000 scheduler runs for this study (its generic 'campaign' jobs).
  check     check.py on both cards; waits (exit 3) until the Scottish features are verified; must PASS before any run
  runs      50 x train.py (primary encoder first, then seed, fold); each predicts its Korean test fold and all Scottish cases
  evaluate  evaluate.py once every run has its complete.json
The scheduler only gates and launches; train.py / check.py / evaluate.py re-check every gate themselves."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    cfg=json.loads((HERE/'config.json').read_text())
    encs=[cfg['primary_encoder']]+[e for e in cfg['encoders'] if e!=cfg['primary_encoder']]
    runs=[dict(id=f'{e}:s{s}:f{f}',argv=['train.py','--encoder',e,'--seed-index',str(s),'--fold',str(f),'--device','cuda','--threads','4',
                                        '--ram-margin-gib','64'],complete=f'runs/{e}/seed{s}/fold{f}/complete.json')
          for e in encs for s in cfg['seeds'] for f in cfg['folds']]
    camp=dict(name='surv1',title='HE + clinical recurrence prediction (Korea development, Scotland external)',version=cfg['version'],
              signature_files=['common.py','train.py','check.py','evaluate.py','config.json','inputs/korea_surv.csv','inputs/scotland_surv.csv',
                               'inputs/korea_if.csv','inputs/scotland_if.csv'],
              lock_file='LOCK.json',check=dict(argv=['check.py','--device','cuda','--threads','4'],pass_file='CHECK_PASS.json'),
              runs=runs,evaluate=dict(argv=['evaluate.py'],results='RESULTS.json',report='RESULTS.md'))
    (HERE/'CAMPAIGN.json').write_text(json.dumps(camp,indent=1)+'\n');print('runs',len(runs))
if __name__=='__main__':main()
