#!/usr/bin/env bash
# Deterministic CPU-only paired analysis after all Kodak context arms finish.
set -euo pipefail

repo=/home/can_karsal/FLEX-PLUS
python_bin="$repo/.venv/bin/python"
results="$repo/proof/cpu_early_exit/results"
cd "$repo"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''

while tmux has-session -t reglic_active_canvas_sequence_20261005 2>/dev/null; do
    sleep 20
done
"$python_bin" - "$results" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('kodak_canvas_coupling_20261005','kodak_active_canvas_20261005',
            'kodak_active_canvas_replicate_20261005'):
    data=json.loads((root/f'{arm}.json').read_text())
    assert data['complete'] and len(data['rows'])==120, f'Incomplete arm: {arm}'
PY

"$python_bin" proof/cpu_early_exit/compare_canvas_arms.py
"$python_bin" proof/cpu_early_exit/analyze_canvas_bdrate.py
"$python_bin" proof/cpu_early_exit/plot_canvas_context.py
