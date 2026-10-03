"""Run the predeclared nine-case GPU cohort and audit its raw outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    folder = args.out or HERE/'results'/('cohort_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    for sequence in ('videoSRC05', 'FourPeople', 'BQMall'):
        for qp in (16, 32, 48):
            output = folder/f'{sequence}_qp{qp}.json'
            command = [sys.executable, str(HERE/'benchmark.py'), '--gpu', str(args.gpu),
                       '--sequence', sequence, '--qp', str(qp), '--blocks', str(args.blocks),
                       '--out', str(output)]
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
