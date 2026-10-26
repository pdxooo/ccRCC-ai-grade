"""Run one shard with a pre-deadline checkpoint and bounded Slurm continuation."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
MAX_CONTINUATIONS = 2
RESERVE_SECONDS = 1800

def slurm_seconds(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    fields = [int(x) for x in clock.split(':')]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, *fields
    elif len(fields) == 1:
        hours, minutes, seconds = 0, fields[0], 0
    else:
        raise ValueError(value)
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds

def continuation_command(shard, job_id, attempt):
    assert 0 <= shard < 4 and job_id.isdigit()
    assert 0 <= attempt < MAX_CONTINUATIONS
    return ['sbatch', '--parsable', '--array='+str(shard),
            '--dependency=afterany:'+job_id,
            '--export=ALL,NINE_CONTINUATION='+str(attempt+1),
            str(BASE/'run_extract_nine.slurm'), 'full']

def finish_run(code, shard, job_id, attempt, report_path, runner=subprocess.run):
    if code != 75:
        return code
    report = json.loads(report_path.read_text())
    assert report['state'] == 'paused_for_walltime'
    assert str(report['job_id']) == job_id
    if attempt >= MAX_CONTINUATIONS:
        print('CONTINUATION_LIMIT_REACHED; checkpoint preserved; inspect throughput',flush=True)
        return 75
    command = continuation_command(shard, job_id, attempt)
    result = runner(command, capture_output=True, text=True, check=True)
    new_id = result.stdout.strip().split(';')[0]
    assert re.fullmatch(r'\d+', new_id), result.stdout
    record = dict(previous_job=job_id, next_job=new_id, shard=shard,
                  continuation=attempt+1, command=command, time_unix=time.time())
    path = report_path.parent/('continuation_'+job_id+'.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)); tmp.replace(path)
    print('CONTINUATION_SUBMITTED',json.dumps(record),flush=True)
    return 0

def main():
    shard = int(os.environ['SLURM_ARRAY_TASK_ID'])
    job_id = os.environ['SLURM_JOB_ID']
    attempt = int(os.environ.get('NINE_CONTINUATION','0'))
    assert 0 <= shard < 4 and 0 <= attempt <= MAX_CONTINUATIONS
    # Use the scheduler's actual time limit, including any operator updates.
    result = subprocess.run(['scontrol','show','job','-o',job_id],
                            capture_output=True,text=True,check=True)
    fields = dict(token.split('=',1) for token in result.stdout.split() if '=' in token)
    remaining = slurm_seconds(fields['TimeLimit'])-slurm_seconds(fields['RunTime'])
    assert remaining > RESERVE_SECONDS, 'Insufficient allocated time for extraction'
    env = os.environ.copy()
    env['NINE_STOP_AT'] = str(time.time()+remaining-RESERVE_SECONDS)
    print('RUNTIME_GUARD',json.dumps(dict(job_id=job_id,shard=shard,
          time_limit=fields['TimeLimit'],seconds_remaining=remaining,
          checkpoint_margin_seconds=RESERVE_SECONDS,continuation=attempt)),flush=True)
    result = subprocess.run([sys.executable,'-u',str(BASE/'extract_nine_korea.py'),
                             '--shard',str(shard),'--shards','4'],env=env)
    report = BASE.parent/'features/nine_encoder_full_v1_20261010'/('summary_shard'+str(shard)+'.json')
    return finish_run(result.returncode,shard,job_id,attempt,report)

if __name__ == '__main__':
    sys.exit(main())
