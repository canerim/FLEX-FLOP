#!/bin/bash
# The one idea the branch has not tested: let the rest of the decoder move.
#
# Every probe so far asked a cheaper stem to reproduce a representation the
# trunk and the exits were fitted to. This trains the per-tile trunk and the
# adapters alongside it, so the exits can learn to work from the cheaper stem
# instead. If the stem's work is redundant given a trunk allowed to adapt,
# this is where it shows; if it is not, the branch is finished.
#
# Two widths: 0.5 for the ceiling the target needs, 0.25 for the one it would
# be worth paying for.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/joint.log
mkdir -p flexplus/logs flexplus/results
echo "=== joint training $(date '+%F %T') ===" | tee -a "$LOG"
for W in 0.5 0.25; do
  M="flexplus/results/joint_w${W}.done"
  [ -f "$M" ] && { echo "  width $W done" | tee -a "$LOG"; continue; }
  echo "  $(date '+%H:%M:%S')  width $W, trunk and adapters unfrozen" \
    | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
      --objective decode --unfreeze_trunk --steps 20000 --batch 8 \
      --crop 256 --lr 1e-4 --device cuda:0 --tag "w${W}_joint" \
      >>"$LOG" 2>&1 && : > "$M"
done
echo "=== joint training done $(date '+%F %T') ===" | tee -a "$LOG"
