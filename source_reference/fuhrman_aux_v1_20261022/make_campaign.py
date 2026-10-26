"""Writes CAMPAIGN.json: what the lab-a6000 scheduler runs for this study (its generic 'campaign' jobs).
  check     check.py on both cards (resume legs on the other card); must PASS (CHECK_PASS.json) before any formal run
  runs      150 x train.py, primary encoder first, then seed, arm, fold
  evaluate  evaluate.py once every run has its complete.json
The scheduler only gates and launches; train.py / check.py / evaluate.py re-check every gate themselves."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    cfg=json.loads((HERE/'config.json').read_text())
    encs=[cfg['primary_encoder']]+[e for e in cfg['encoders'] if e!=cfg['primary_encoder']]
    runs=[dict(id=f'{e}:{a}:s{s}:f{f}',argv=['train.py','--encoder',e,'--arm',a,'--seed-index',str(s),'--fold',str(f),'--device','cuda',
                                            '--threads','4','--ram-margin-gib','64'],complete=f'runs/{e}/{a}/seed{s}/fold{f}/complete.json')
          for e in encs for s in cfg['seeds'] for a in cfg['arms'] for f in cfg['folds']]
    camp=dict(name='fx1',title='direction 3: Fuhrman auxiliary supervision (A/B/C)',version=cfg['version'],
              signature_files=['common.py','train.py','check.py','evaluate.py','config.json','inputs/korea_labels.csv'],
              lock_file='LOCK.json',
              check=dict(argv=['check.py','--device','cuda','--threads','4'],pass_file='CHECK_PASS.json',both_gpus=True),
              runs=runs,evaluate=dict(argv=['evaluate.py'],results='RESULTS.json',report='RESULTS.md'),
              resources=dict(cost=4,gpu_mb=6000,ram_gb=24,ram_margin_gb=64,eval_cost=2,eval_ram_gb=8,check_ram_gb=8))
    (HERE/'CAMPAIGN.json').write_text(json.dumps(camp,indent=1)+'\n');print('runs',len(runs))
if __name__=='__main__':main()
