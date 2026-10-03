"""Run the predeclared nine-case GPU cohort and audit its raw outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--matched-kernels', action='store_true')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse already verified cases in a fixed output folder')
    args = parser.parse_args()
    folder = args.out or HERE/'results'/('cohort_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(parents=True, exist_ok=args.resume)
    benchmark_sha = hashlib.sha256((HERE/'benchmark.py').read_bytes()).hexdigest()
    if args.resume:
        from benchmark import RELEASE, E15
        checkpoints = {'released_d12': file_sha(RELEASE), 'e15': file_sha(E15)}
    results = []
    for sequence in ('videoSRC05', 'FourPeople', 'BQMall'):
        for qp in (16, 32, 48):
            output = folder/f'{sequence}_qp{qp}.json'
            if args.resume and output.exists():
                old = json.loads(output.read_text())
                if (old.get('claim_eligible') and old.get('schema') == 3 and
                        old.get('sequence', '').startswith(sequence) and
                        old.get('qp') == qp and old.get('gpu_index') == args.gpu and
                        old.get('benchmark_sha256') == benchmark_sha and
                        old.get('checkpoint_sha256') == checkpoints and
                        len(old.get('samples', {}).get('released_d12_triton', [])) == args.blocks):
                    results.append(output)
                    print(f'reused {output}', flush=True)
                    continue
            command = [sys.executable, str(HERE/'benchmark.py'), '--gpu', str(args.gpu),
                       '--sequence', sequence, '--qp', str(qp), '--blocks', str(args.blocks),
                       '--out', str(output)]
            if args.matched_kernels:
                command.append('--matched-kernels')
            with (folder/f'{sequence}_qp{qp}.jsonl').open('w') as log:
                status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f'{sequence}/QP{qp} failed (exit {status.returncode}); see JSONL')
            results.append(output)
    subprocess.run([sys.executable, str(HERE/'summarize.py'), *map(str, results),
                    '--out', str(folder/'cohort_summary.json')], cwd=ROOT, check=True)
    print(folder/'cohort_summary.json')


if __name__ == '__main__':
    main()
