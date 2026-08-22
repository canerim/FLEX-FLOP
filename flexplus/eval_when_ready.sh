#!/bin/bash
# Evaluate each width as soon as its training finishes, rather than waiting
# for the whole sweep: the first width decides whether the axis is alive, and
# a negative answer should arrive before three more hours of training on it.
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
for W in 0.5 0.707 0.35 0.25; do
  for _ in $(seq 1 480); do
    [ -f "flexplus/results/narrow_stem_w${W}.done" ] && break
    sleep 30
  done
  [ -f "flexplus/results/narrow_stem_w${W}.pth" ] || continue
  echo "=== evaluating width $W  $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py \
      --stems "flexplus/results/narrow_stem_w${W}.pth" \
      --device cuda:0 --max_seqs 12 \
      --out "flexplus/results/narrow_eval_w${W}.json"
done
