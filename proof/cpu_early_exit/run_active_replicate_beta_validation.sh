#!/usr/bin/env bash
# Calibrate on DIV2K calibration24, freeze beta, then replay validation24.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_calibration_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
# The selector refuses incomplete calibration; no validation datum enters beta.
nice -n 19 "$python_bin" -u proof/cpu_early_exit/select_active_replicate_beta.py
# A one-case replay checks the new policy before the resumable 120-case sweep.
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_replicate_beta_validation.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_replicate_beta_validation.py
