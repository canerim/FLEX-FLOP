#!/usr/bin/env bash
# Sequential CPU-only budget sweep after the primary and seam-locality jobs.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_context_seam_locality_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
nice -n 19 "$python_bin" -u proof/cpu_early_exit/select_active_beta_budget_sweep.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_budget_transfer.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_budget_transfer.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/analyze_active_beta_budget_transfer.py
