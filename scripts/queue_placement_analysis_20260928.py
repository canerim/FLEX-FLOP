"""Run the pinned placement analysis once the complete CPU replay is verified."""
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/placement_replay_qp32')
SCRIPT = REPO / 'scripts/analyze_placement_replay_20260928.py'
GROUPS = REPO / 'cvpr2027/data/research20260927/shared_crossfit_qp32/proposed_content_groups.json'
STATUS = SOURCE / 'analysis_queue.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(state, **extra):
    record = {'state': state, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(STATUS)


def main():
    pinned = {str(path): sha(path) for path in (SCRIPT, GROUPS, Path(__file__))}
    write('waiting_for_complete_cohort', pinned=pinned)
    deadline = time.monotonic() + 48 * 3600
    try:
        while time.monotonic() < deadline:
            for path, digest in pinned.items():
                if sha(Path(path)) != digest:
                    raise ValueError(f'Queued analysis input changed: {path}')
            progress = json.loads((SOURCE / 'progress.json').read_text())
            if progress['state'] == 'failed':
                raise RuntimeError('Replay failed: ' + progress.get('error', 'unknown'))
            if progress['state'] == 'complete':
                write('analysing_complete_cohort', pinned=pinned)
                with (SOURCE / 'analysis.log').open('w') as log:
                    subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                write('complete', pinned=pinned,
                      output=str(REPO / 'docs/research/2026-09-28-paper-editorial/placement_replay'))
                return
            time.sleep(30)
        raise TimeoutError('No complete cohort within 48 hours; no partial analysis produced')
    except BaseException as error:
        write('failed', pinned=pinned, error=repr(error))
        raise


if __name__ == '__main__':
    main()
