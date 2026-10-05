#!/usr/bin/env bash
# Render only after the complete frozen budget-transfer analysis exists.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_budget_transfer_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
nice -n 19 /home/can_karsal/FLEX-PLUS/.venv/bin/python -u \
    proof/cpu_early_exit/plot_active_beta_budget_transfer.py
