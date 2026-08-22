#!/bin/bash
# The decode objective with every exit supervised each step.
#
# Sampling one exit per step gave each exit a quarter of the updates the
# feature objective gives all of them at once, so matching the two on steps
# still left them unmatched on updates per exit -- and decode lost by about
# that margin. This is the clean comparison.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/widths_decode_allexits.log
mkdir -p flexplus/logs flexplus/results
for _ in $(seq 1 900); do
  [ -f flexplus/results/narrow_eval_w0.707_decode_matched.json ] && break
  sleep 30
done
W=0.707
echo "=== decode, every exit, width $W, $(date '+%F %T') ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
    --objective decode --steps 20000 --batch 8 --crop 256 --device cuda:0 \
    --tag "w${W}_decode_allexits" >>"$LOG" 2>&1
CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py \
    --stems "flexplus/results/narrow_stem_w${W}_decode_allexits.pth" \
    --device cuda:0 --max_seqs 12 \
    --out "flexplus/results/narrow_eval_w${W}_decode_allexits.json" >>"$LOG" 2>&1
echo "=== done $(date '+%F %T') ===" | tee -a "$LOG"
