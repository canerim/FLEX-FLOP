#!/usr/bin/env bash
# Full calibration frontier only after the isolated repair-off control ends.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_isolated_no_repair_div2k_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
# One-case smoke first; the second invocation resumes the identical source
# and processes the remaining 119 cases if and only if that check succeeds.
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_replicate_beta_calibration.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_replicate_beta_calibration.py
