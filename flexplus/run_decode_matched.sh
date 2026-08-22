#!/bin/bash
# The decode objective at the feature objective's training budget.
#
# The first decode run used 12000 steps at batch 4 against the feature run's
# 20000 at batch 8, and it also splits its gradient across four exits, one
# sampled per step. So it saw about a sixth of the updates per exit and came
# out worse. That is not a result about the objective; it is a result about
# the budget, and comparing them as they stand would be comparing the wrong
# thing.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/widths_decode_matched.log
mkdir -p flexplus/logs flexplus/results
# wait for the short decode sweep to release the card
for _ in $(seq 1 900); do
  [ -f flexplus/results/narrow_stem_w0.5_decode.done ] && break
  sleep 60
done
W=0.707
echo "=== decode at a matched budget, width $W, $(date '+%F %T') ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
    --objective decode --steps 20000 --batch 8 --crop 256 --device cuda:0 \
    --tag "w${W}_decode_matched" >>"$LOG" 2>&1
CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py \
    --stems "flexplus/results/narrow_stem_w${W}_decode_matched.pth" \
    --device cuda:0 --max_seqs 12 \
    --out "flexplus/results/narrow_eval_w${W}_decode_matched.json" \
    >>"$LOG" 2>&1
echo "=== done $(date '+%F %T') ===" | tee -a "$LOG"
