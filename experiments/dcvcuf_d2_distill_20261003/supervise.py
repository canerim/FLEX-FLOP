"""Wait for D6 to finish, then run the isolated D2 distillation on GPU 6."""
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
D6 = ROOT / 'runs/d6'
RUN = ROOT / 'runs/d2_distilled_released'
HERE = Path(__file__).resolve().parent
GPU_ORDER = ('6', '7', '1', '3', '0', '2', '4', '5')
STOP = False


def on_stop(_signum, _frame):
    global STOP
    STOP = True


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(kind, **fields):
    RUN.mkdir(parents=True, exist_ok=True)
    row = {'timestamp': stamp(), 'event': kind, **fields}
    with (RUN / 'supervisor.jsonl').open('a', buffering=1) as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def gpu_free(gpu):
    info = subprocess.check_output(
        ['nvidia-smi', '--id=' + gpu, '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).strip()
    memory, utilization = map(int, info.split(','))
    return memory < 512 and utilization < 10, memory, utilization


def first_free_gpu():
    for gpu in GPU_ORDER:
        if gpu_free(gpu)[0]:
            return gpu
    return None


def d6_finished():
    status = D6 / 'status.json'
    if not status.exists() or not (D6 / 'ckpt.pth.tar').exists():
        return False
    state = json.loads(status.read_text())
    return state.get('state') == 'complete' and state.get('epoch') == 105


def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_stop)
    RUN.mkdir(parents=True, exist_ok=True)
    event('armed', gpu_priority=GPU_ORDER,
          prerequisite='D6 final checkpoint + complete status + idle GPU')
    while not STOP:
        ready = d6_finished()
        gpu = first_free_gpu() if ready else None
        if ready and gpu is not None:
            break
        if int(time.monotonic()) % 600 < 60:
            _, memory, utilization = gpu_free('6')
            event('waiting', d6_complete=ready, preferred_gpu_memory_mib=memory,
                  preferred_gpu_utilization_pct=utilization)
        time.sleep(60)
    if STOP:
        event('stopped_before_launch')
        return
    event('gpu_selected', gpu=gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               TORCHINDUCTOR_COMPILE_THREADS='4',
               TORCHINDUCTOR_CACHE_DIR=str(ROOT/'inductor_d2_distilled'))
    exe = str(ROOT/'venv/bin/python')
    for patch in (256, 512):
        smoke_dir = RUN / f'preflight_{patch}'
        if (smoke_dir/'status.json').exists():
            if json.loads((smoke_dir/'status.json').read_text()).get('state') == 'smoke_complete':
                continue
        smoke_cmd = [exe, '-u', str(HERE/'train.py'), '--save-dir', str(smoke_dir),
                     '--smoke', '--smoke-steps', '1', '--smoke-patch', str(patch), '--compile']
        while not STOP and not gpu_free(gpu)[0]:
            time.sleep(60)
        if STOP:
            return
        event('preflight_start', patch=patch, gpu=gpu)
        smoke_dir.mkdir(parents=True, exist_ok=True)
        with (smoke_dir/'train.log').open('a', buffering=1) as log:
            subprocess.run(smoke_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        event('preflight_pass', patch=patch)
    command = [exe, '-u', str(HERE/'train.py'), '--save-dir', str(RUN), '--compile']
    for attempt in range(3):
        if STOP:
            return
        while not STOP and not gpu_free(gpu)[0]:
            time.sleep(60)
        if STOP:
            return
        event('launch', attempt=attempt + 1, command=command)
        with (RUN/'train.log').open('a', buffering=1) as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            (RUN/'pid').write_text(str(child.pid))
            while child.poll() is None:
                if STOP:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                    event('stopped', child_exit=child.returncode)
                    return
                time.sleep(10)
        event('train_exit', attempt=attempt + 1, exit_code=child.returncode)
        if child.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError('D2 distillation failed after three attempts; inspect train.log')
        time.sleep(60)
    if json.loads((RUN/'status.json').read_text()).get('state') != 'complete':
        raise RuntimeError('Training exited without complete status')
    with (RUN/'evaluation.log').open('a', buffering=1) as log:
        subprocess.run([exe, str(ROOT/'code_snapshot_v2/evaluate.py'), '--depth', '2',
                        '--checkpoint', str(RUN/'ckpt.pth.tar'),
                        '--out', str(RUN/'kodak_final.json')],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    event('complete', kodak=str(RUN/'kodak_final.json'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event('failed', error=repr(exc))
        raise
