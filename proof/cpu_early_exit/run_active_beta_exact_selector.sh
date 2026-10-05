#!/usr/bin/env bash
# Read-only post-lock arithmetic audit; do not modify the frozen beta policy.
set -euo pipefail
cd /home/can_karsal/FLEX-PLUS
while tmux has-session -t reglic_active_beta_analysis_20261005 2>/dev/null; do
    sleep 20
done
test -s proof/cpu_early_exit/results/div2k_beta/quality_floor/active_replicate_beta_validation24_analysis.json
python3 proof/cpu_early_exit/audit_active_beta_exact_selector.py
