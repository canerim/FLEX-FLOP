"""Wait for a quiet CPU host, then collect paired PyTorch CPU synthesis timings.

Polling performs no inference. Each child is nice(19), node-0 CPU-affined,
and matched_routed_cpu.py aborts if host load crosses the threshold.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT/'proof/early_exit_vs_released'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=PROOF/'.local/DCVC')
    parser.add_argument('--extension', type=Path, default=PROOF/'.local/entropy')
    parser.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--router', type=Path, default=PROOF/'artifacts/router_stem_qp.pth')
    parser.add_argument('--manifest', type=Path, default=PROOF/'results/bitstream_kodak3x3/manifest.json')
    parser.add_argument('--out-dir', type=Path, default=PROOF/'results/cpu_matched_idle')
    parser.add_argument('--max-load', type=float, default=12)
    parser.add_argument('--stable-polls', type=int, default=5)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--timeout-hours', type=float, default=72)
    parser.add_argument('--blocks', type=int, default=30)
    args = parser.parse_args()
    if args.max_load <= 0 or args.stable_polls < 1 or args.poll_seconds < 5:
        parser.error('Invalid idle guard')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if manifest['cases'] != 9:
        raise RuntimeError('Expected predeclared Kodak3x3 cohort')
    deadline = time.monotonic() + 3600*args.timeout_hours
    stable = 0
    log = args.out_dir/'watcher.jsonl'

    def write(event, **extra):
        row = {'time_unix': time.time(), 'event': event,
               'load_1m': os.getloadavg()[0], **extra}
        with log.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    def wait_idle():
        nonlocal stable
        while time.monotonic() < deadline:
            load = os.getloadavg()[0]
            stable = stable + 1 if load <= args.max_load else 0
            if stable >= args.stable_polls:
                return True
            if stable == 0 or stable == 1:
                write('waiting', stable_polls=stable)
            time.sleep(args.poll_seconds)
        return False

    for thread_count in (1, 2, 4, 8):
        affinity = ','.join(str(cpu) for cpu in range(thread_count))
        for item in manifest['rows']:
            stem = Path(item['stream']).stem
            output = args.out_dir/f'{stem}_t{thread_count}.json'
            if output.exists():
                old = json.loads(output.read_text())
                if (old.get('claim_eligible') and old.get('stream_sha256') ==
                        item['stream_sha256'] and old.get('threads') == thread_count):
                    write('skip_verified', output=str(output))
                    continue
            while True:
                if not wait_idle():
                    write('timeout', output=str(output))
                    return
                command = ['taskset', '-c', affinity, sys.executable,
                           str(ROOT/'proof/cpu_early_exit/matched_routed_cpu.py'),
                           '--upstream', str(args.upstream),
                           '--extension', str(args.extension),
                           '--release', str(args.release), '--e15', str(args.e15),
                           '--router', str(args.router), '--stream', str(ROOT/item['stream']),
                           '--source', str(ROOT/'data/kodak'/item['image']),
                           '--threads', str(thread_count), '--blocks', str(args.blocks),
                           '--max-load', str(args.max_load), '--out', str(output)]
                env = {**os.environ, 'OMP_NUM_THREADS': str(thread_count),
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': str(thread_count)}
                write('start', case=stem, threads=thread_count)
                with (args.out_dir/f'{stem}_t{thread_count}.log').open('w') as handle:
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode == 0 and output.exists():
                    record = json.loads(output.read_text())
                    if record['claim_eligible']:
                        write('complete', case=stem, threads=thread_count,
                              medians=record['timing_ms_median'])
                        break
                if os.getloadavg()[0] <= args.max_load and result.returncode != 0:
                    write('implementation_failure', case=stem,
                          threads=thread_count, returncode=result.returncode)
                    raise RuntimeError(f'CPU run failed while host idle; inspect {stem} log')
                write('retry_after_load_or_failure', case=stem,
                      threads=thread_count, returncode=result.returncode)
                time.sleep(args.poll_seconds)
    write('cohort_complete')


if __name__ == '__main__':
    main()
