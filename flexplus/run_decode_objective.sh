#!/bin/bash
# The same widths, trained on the decode instead of on the stem's feature.
#
# The feature objective was measured to be a poor proxy: 5% relative error in
# the feature came out as 5 to 10 times the dB at the exit. This trains on the
# reconstruction, which is the quantity the budget is set in, sampling an exit
# per step so the stem stays usable for the whole ladder.
#
# Queued behind the feature sweep: one card for this branch.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/widths_decode.log
while pgrep -f "flexplus/narrow_stem.py" >/dev/null; do sleep 60; done
echo "=== decode-objective sweep $(date '+%F %T') ===" | tee -a "$LOG"
for W in 0.707 0.5; do
  M="flexplus/results/narrow_stem_w${W}_decode.done"
  [ -f "$M" ] && { echo "  width $W done" | tee -a "$LOG"; continue; }
  echo "  $(date '+%H:%M:%S')  width $W" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
      --objective decode --steps 12000 --batch 4 --crop 256 --device cuda:0 \
      >>"$LOG" 2>&1 && : > "$M"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py \
      --stems "flexplus/results/narrow_stem_w${W}_decode.pth" \
      --device cuda:0 --max_seqs 12 \
      --out "flexplus/results/narrow_eval_w${W}_decode.json" >>"$LOG" 2>&1
done
echo "=== decode sweep done $(date '+%F %T') ===" | tee -a "$LOG"
