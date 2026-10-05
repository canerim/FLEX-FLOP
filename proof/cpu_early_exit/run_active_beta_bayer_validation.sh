#!/usr/bin/env bash
# Wait for the frozen third-cohort transfer, then run the matched-cost control.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_clic39_active_beta_20261005 2>/dev/null; do
    sleep 20
done
python3 - <<'PY'
import json
from pathlib import Path
base=Path('proof/cpu_early_exit/results')
clic=json.loads((base/'clic39_active_beta/raw.json').read_text())
primary=json.loads((base/'div2k_beta/quality_floor/active_replicate_beta_validation24.json').read_text())
assert clic['complete'] and len(clic['rows'])==195, 'Preceding CLIC job incomplete'
assert (base/'clic39_active_beta/summary.json').exists(), 'Preceding CLIC analysis incomplete'
assert primary['complete'] and len(primary['rows'])==120, 'Primary DIV2K beta validation incomplete'
PY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_bayer_validation.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_bayer_validation.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/analyze_active_beta_bayer_validation.py
