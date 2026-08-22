#!/bin/bash
# Evaluate each joint pair as it finishes.
#
# Separate from narrow_eval because joint training moves the trunk as well as
# the stem, and loading the pinned decoder here would measure a narrow stem
# against a trunk that never saw it.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
for W in 0.5 0.25; do
  for _ in $(seq 1 900); do
    [ -f "flexplus/results/joint_w${W}.done" ] && break
    sleep 30
  done
  [ -f "flexplus/results/narrow_stem_w${W}_joint.pth" ] || continue
  echo "=== joint eval, width $W, $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/joint_eval.py --tag "w${W}_joint" \
      --device cuda:0 --max_seqs 12
done
