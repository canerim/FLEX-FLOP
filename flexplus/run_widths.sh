#!/bin/bash
# The width sweep: one narrow stem per width, each trained to match the full
# stem's output, then evaluated for what it costs and what ceiling it buys.
#
# Card 4. The consistency chain owns card 7 and the main experiment must not
# be slowed by this.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/widths.log
mkdir -p flexplus/logs flexplus/results
echo "=== width sweep $(date '+%F %T') ===" | tee -a "$LOG"
for W in 0.5 0.707 0.35 0.25; do
  if [ -s "flexplus/results/narrow_stem_w${W}.done" ]; then
    echo "  width $W already done" | tee -a "$LOG"; continue
  fi
  echo "  $(date '+%H:%M:%S')  training width $W" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
      --steps 20000 --batch 8 --crop 256 --device cuda:0 \
      >>"$LOG" 2>&1 && : > "flexplus/results/narrow_stem_w${W}.done"
done
echo "=== width sweep done $(date '+%F %T') ===" | tee -a "$LOG"
