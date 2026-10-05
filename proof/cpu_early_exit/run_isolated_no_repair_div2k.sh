#!/usr/bin/env bash
# Follow the repair-scale pilot; never compete with the frozen transfer job.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_repair_alpha_pilot_20261005 2>/dev/null; do
    sleep 20
done
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
nice -n 19 /data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python \
    -u proof/cpu_early_exit/audit_isolated_no_repair_div2k.py
