#!/bin/bash
# Evaluate each jointly trained run against its own decoder, not the pinned
# one: --unfreeze_trunk moves the trunk and the adapters too.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
for W in 0.5 0.25; do
  for _ in $(seq 1 900); do
    [ -f "flexplus/results/joint_w${W}.done" ] && break
    sleep 60
  done
  ST="flexplus/results/narrow_stem_w${W}_joint.pth"
  DE="flexplus/results/joint_dec_w${W}_joint.pth"
  [ -f "$ST" ] && [ -f "$DE" ] || { echo "width $W: files missing"; continue; }
  echo "=== jointly trained width $W  $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py --stems "$ST" \
      --dec "$DE" --device cuda:0 --max_seqs 12 \
      --out "flexplus/results/narrow_eval_w${W}_joint.json"
done
