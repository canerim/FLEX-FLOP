#!/usr/bin/env bash
# Tail-risk ablation runs after the mean-policy and budget-frontier CPU chain.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_budget_figure_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
nice -n 19 "$python_bin" -u proof/cpu_early_exit/select_active_beta_tail_policy.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_tail_validation.py \
    --max-new-cases 1
nice -n 19 "$python_bin" -u proof/cpu_early_exit/audit_active_beta_tail_validation.py
nice -n 19 "$python_bin" -u proof/cpu_early_exit/analyze_active_beta_tail_validation.py
