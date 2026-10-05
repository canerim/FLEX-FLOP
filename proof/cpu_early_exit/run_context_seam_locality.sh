#!/usr/bin/env bash
# Run only after the calibration, validation and analysis CPU chain ends.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_exact_selector_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_context_seam_locality.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_context_seam_locality.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/analyze_context_seam_locality.py
