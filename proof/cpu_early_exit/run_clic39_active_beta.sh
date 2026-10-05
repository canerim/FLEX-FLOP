#!/usr/bin/env bash
# Frozen CLIC39 transfer, sequenced after the primary DIV2K CPU analyses.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_tail_validation_20261005 2>/dev/null; do
    sleep 20
done
python3 - <<'PY'
import json
from pathlib import Path
base=Path('proof/cpu_early_exit/results/div2k_beta/quality_floor')
raw=json.loads((base/'active_replicate_beta_validation24.json').read_text())
assert raw['complete'] and len(raw['rows'])==120, 'Primary beta validation did not finish'
assert (base/'active_replicate_beta_validation24_analysis.json').exists(), 'Primary beta analysis did not finish'
PY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_clic39_active_beta.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_clic39_active_beta.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/analyze_clic39_active_beta.py
