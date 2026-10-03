"""Run strict BD-rate analysis when the resumable 480-case encoder completes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FOLDER = HERE/'results/kodak_final_verified'


def main():
    while True:
        path = FOLDER/'progress.json'
        if path.exists():
            progress = json.loads(path.read_text())
            if progress['state'] == 'complete':
                if progress['completed'] != 480:
                    raise RuntimeError('Incomplete evaluation marked complete')
                subprocess.run([sys.executable, str(HERE/'analyze.py'), '--folder', str(FOLDER)], check=True)
                return
            if progress['state'] not in ('running', 'partial_smoke'):
                raise RuntimeError(f'Evaluation stopped: {progress}')
        time.sleep(20)


if __name__ == '__main__':
    main()
