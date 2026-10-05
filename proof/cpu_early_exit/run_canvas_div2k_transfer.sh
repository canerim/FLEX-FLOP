#!/usr/bin/env bash
# Sequential low-priority transfer replay after the Kodak arm comparisons.
set -euo pipefail

repo=/home/can_karsal/FLEX-PLUS
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
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
    d=json.loads((root/f'{arm}.json').read_text())
    assert d['complete'] and len(d['rows'])==120, f'Kodak prerequisite incomplete: {arm}'
PY

for fallback in zero replicate; do
    nice -n 19 "$python_bin" proof/cpu_early_exit/audit_active_canvas_div2k.py \
        --fallback "$fallback"
    source="$results/div2k_beta/quality_floor/active_canvas_validation24_$fallback.json"
    "$python_bin" proof/cpu_early_exit/summarize_canvas_coupling_kodak.py \
        --source "$source"
done
