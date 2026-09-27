"""Sequential CPU research queue; never edits or launches GPU training jobs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
OUT=ROOT/'research/session_20260927'
PYTHON=ROOT/'venv/bin/python'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(value):
    value['updated_utc']=utc();tmp=OUT/'queue_status.json.tmp'
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'queue_status.json').exists():raise RuntimeError('Queue state already exists; inspect before starting another queue')
    os.nice(10)
    commands=[['scripts/analyze_reference_validation_20260927.py'],
              ['scripts/analyze_depth_source_features_20260927.py'],
              ['experiments/dcvcuf_patch_control_20260927/evaluate_patching.py']]
    hashes={args[0]:hashlib.sha256((REPO/args[0]).read_bytes()).hexdigest() for args in commands}
    preflight=json.loads((ROOT/'research/patch_control_preflight/verification.json').read_text())
    if preflight['patch_script_sha256']!=hashes[commands[-1][0]] or not preflight['metric_geometry_check']:
        raise ValueError('Patch-control preflight does not match queued code')
    state={'state':'waiting_reference','pid':os.getpid(),'started_utc':utc(),'commands':commands,'script_sha256':hashes,'completed_commands':[]}
    write(state);child=None
    try:
        while True:
            progress=json.loads((ROOT/'research/div2k100_reference_epoch020/progress.json').read_text())
            state['reference_progress']={k:progress.get(k) for k in ('state','completed','total','depth','image','qp')};write(state)
            if progress['state']=='failed':raise RuntimeError('Reference evaluation failed: '+progress.get('error','unknown'))
            if progress['state']=='complete':break
            time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        for index,args in enumerate(commands):
            if hashlib.sha256((REPO/args[0]).read_bytes()).hexdigest()!=hashes[args[0]]:
                raise RuntimeError('Queued code changed: '+args[0])
            with (OUT/f'command_{index}.log').open('a') as log:
                child=subprocess.Popen([str(PYTHON),*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(state='running_command',command_index=index,command=args,child_pid=child.pid);write(state)
                while child.poll() is None:time.sleep(10);write(state)
                if child.returncode:raise RuntimeError(f'Command {index} failed with exit {child.returncode}; inspect command log')
            state['completed_commands'].append({'command':args,'finished_utc':utc()});child=None;write(state)
        state.update(state='complete',finished_utc=utc(),child_pid=None);write(state)
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));write(state);raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=30)


if __name__=='__main__':main()
