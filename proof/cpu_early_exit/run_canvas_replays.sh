#!/usr/bin/env bash
# One low-priority CPU worker, sequential Kodak bitstream replays; no GPU use.
set -euo pipefail

repo=/home/can_karsal/FLEX-PLUS
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
results="$repo/proof/cpu_early_exit/results"
cd "$repo"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''

# The stale-canvas control was already running when this sequence was frozen.
while tmux has-session -t reglic_kodak_canvas_20261005 2>/dev/null; do
    sleep 20
done
"$python_bin" - "$results/kodak_canvas_coupling_20261005.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['complete'] and len(d['rows'])==120, 'Stale-canvas control incomplete'
PY

nice -n 19 "$python_bin" proof/cpu_early_exit/audit_active_canvas_kodak.py
"$python_bin" - "$results/kodak_active_canvas_20261005.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['complete'] and len(d['rows'])==120, 'Active-zero replay incomplete'
PY

nice -n 19 "$python_bin" proof/cpu_early_exit/audit_active_canvas_replicate_kodak.py
"$python_bin" - "$results/kodak_active_canvas_replicate_20261005.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['complete'] and len(d['rows'])==120, 'Active-replicate replay incomplete'
PY

for arm in kodak_canvas_coupling_20261005 kodak_active_canvas_20261005 kodak_active_canvas_replicate_20261005; do
    "$python_bin" proof/cpu_early_exit/summarize_canvas_coupling_kodak.py \
        --source "$results/$arm.json"
done
