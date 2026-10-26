"""Tests of run_a6000.sh (the fallback runner) with a fake python: exit-code policy, scheduler-lock refusal, own pid file,
TERM forwarding to the job and its children, nice + CUDA_DEVICE_ORDER. Usage: test_runner.py <scratch_dir>
Prints PASS/FAIL lines; exit 1 on any failure."""
import os,sys,time,fcntl,shutil,signal,subprocess
from pathlib import Path
TESTS=Path(__file__).resolve().parent;COMP=TESTS.parent
FAKE_PY='''#!/bin/bash
echo "pid=$$ nice=$(ps -o nice= -p $$ | tr -d ' ') order=${CUDA_DEVICE_ORDER:-} args=$*" >> "$FAKE_PY_LOG"
case "$*" in *verify_no_work_paths.py*) exit 0;; esac
code=$(head -n 1 "$FAKE_PY_CODES"); tail -n +2 "$FAKE_PY_CODES" > "$FAKE_PY_CODES.tmp"; mv "$FAKE_PY_CODES.tmp" "$FAKE_PY_CODES"
if [ "$code" = sleep ]; then
  trap 'echo "pid=$$ got TERM" >> "$FAKE_PY_LOG"; exit 143' TERM
  sleep 60 & echo "worker=$!" >> "$FAKE_PY_LOG"
  sleep 60 & wait $!; exit 0
fi
exit "${code:-0}"
'''
FAKE_FLOCK='''#!{py}
import sys,fcntl,subprocess
args=sys.argv[1:];mode=fcntl.LOCK_EX;nb=0
while args and args[0].startswith('-'):
    a=args.pop(0);nb=fcntl.LOCK_NB if a=='-n' else nb;mode=fcntl.LOCK_SH if a=='-s' else mode
f=open(args[0],'a')
try:fcntl.flock(f,mode|nb)
except BlockingIOError:sys.exit(1)
sys.exit(subprocess.call(args[1:]))
'''
results=[]
def check(name,ok,detail=''):
    results.append((name,bool(ok)));print(('PASS' if ok else 'FAIL'),name,('| '+str(detail)) if detail else '',flush=True)

def setup(scratch):
    scratch=Path(scratch);P=scratch/'P';(P/'logs').mkdir(parents=True);(P/'pids').mkdir()
    py=scratch/'fakepy';py.write_text(FAKE_PY);py.chmod(0o755)
    path=os.environ['PATH']
    if not shutil.which('flock'):
        b=scratch/'bin';b.mkdir();fl=b/'flock';fl.write_text(FAKE_FLOCK.format(py=sys.executable));fl.chmod(0o755);path=f'{b}:{path}'
    return P,py,path

def run(P,py,path,args,codes,background=False,**env):
    log=P/'fakepy.log';log.unlink(missing_ok=True);cf=P/'codes.txt';cf.write_text(''.join(f'{c}\n' for c in codes))
    e=dict(os.environ,PATH=path,HE_PROJECT_DIR=str(P),HE_PATH_PREFIX='/nonexistent',PY=str(py),FAKE_PY_LOG=str(log),FAKE_PY_CODES=str(cf),
           RETRY_SECONDS='0',ERROR_BACKOFF_SECONDS='0',MAX_ERROR_RETRIES='2');e.update(env)
    cmd=['bash',str(COMP/'run_a6000.sh'),*args]
    if background:return subprocess.Popen(cmd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True),log
    p=subprocess.run(cmd,env=e,capture_output=True,text=True,timeout=120)
    lines=log.read_text().splitlines() if log.exists() else []
    return p.returncode,[l for l in lines if 'verify_no_work_paths' not in l and l.startswith('pid=') and 'args=' in l],p.stdout+p.stderr

def main(scratch):
    P,py,path=setup(scratch)
    rc,jobs,out=run(P,py,path,['stage1','0','2','cuda:0'],[3,3,0])
    j=jobs[0] if jobs else ''
    check('runner: stage1 exit 3,3,0 -> retried, stops with 0, pid file removed',rc==0 and len(jobs)==3 and not (P/'pids/highres_stage1_shard0of2.pid').exists(),f'rc={rc} runs={len(jobs)}')
    check('runner: job runs with nice 10, CUDA_DEVICE_ORDER=PCI_BUS_ID, --shard/--nshards/--device/--wait',
          ' nice=10 ' in j and 'order=PCI_BUS_ID' in j and '--shard 0 --nshards 2 --device cuda:0' in j and '--wait 1800' in j,j[:200])
    rc,jobs,_=run(P,py,path,['stage0','cuda:1'],[2,2,2,2])
    check('runner: stage0 crash (exit 2) retried MAX_ERROR_RETRIES=2 times, then stops with 2',rc==2 and len(jobs)==3,f'rc={rc} runs={len(jobs)}')
    rc,jobs,_=run(P,py,path,['stage0','cuda:1'],[3,1,0])
    check('runner: stage0 FAIL (exit 1) stops at once',rc==1 and len(jobs)==2,f'rc={rc} runs={len(jobs)}')
    rc,jobs,_=run(P,py,path,['stage1','1','2','cuda:1'],[1,1,3,1,1,0])
    check('runner: stage1 error (exit 1) retried; a waiting exit 3 resets the consecutive-error count',rc==0 and len(jobs)==6,f'rc={rc} runs={len(jobs)}')
    rc,jobs,_=run(P,py,path,['stage1','1','2','cuda:1'],[5,0])
    check('runner: lock held by another launcher (exit 5) stops at once',rc==5 and len(jobs)==1,f'rc={rc} runs={len(jobs)}')
    holder=open(P/'logs/scheduler.lock','a');fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rc,jobs,out=run(P,py,path,['stage0','cuda:1'],[0])
    holder.close()
    check('runner: refuses (exit 2, nothing run) while the scheduler holds P/logs/scheduler.lock',rc==2 and not jobs and 'REFUSED' in out,out.strip()[-120:])
    rc,jobs,_=run(P,py,path,['stage0','cuda:1'],[0])
    check('runner: scheduler.lock present but not held -> runs',rc==0 and len(jobs)==1)
    p,log=run(P,py,path,['stage0','cuda:1'],['sleep'],background=True);pidfile=P/'pids/highres_stage0.pid'
    t=time.monotonic()
    while time.monotonic()-t<20 and not (log.exists() and 'worker=' in log.read_text()):time.sleep(.1)
    pid=int(pidfile.read_text()) if pidfile.exists() else None
    worker=int([l for l in log.read_text().splitlines() if l.startswith('worker=')][0].split('=')[1])
    os.kill(pid,signal.SIGTERM);t=time.monotonic()
    try:rc=p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.kill();rc=None
    time.sleep(.3)
    try:os.kill(worker,0);worker_alive=True
    except ProcessLookupError:worker_alive=False
    check('runner: pid file holds the runner\'s own pid; TERM stops the job and its children, exit 143, pid file removed',
          pid==p.pid and rc==143 and not worker_alive and 'got TERM' in log.read_text() and not pidfile.exists(),
          f'pidfile={pid} runner={p.pid} rc={rc} worker_alive={worker_alive} {time.monotonic()-t:.1f}s')
    bad=[n for n,ok in results if not ok]
    print(f'RUNNER {len(results)-len(bad)}/{len(results)} passed');return 1 if bad else 0

if __name__=='__main__':
    s=Path(sys.argv[1]).resolve()
    if s.exists():shutil.rmtree(s)
    s.mkdir(parents=True);sys.exit(main(s))
