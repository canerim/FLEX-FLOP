#!/bin/bash
# The same widths, trained on the decode instead of on the stem's feature.
#
# Replaces run_decode_objective.sh, which waited on
# `pgrep -f flexplus/narrow_stem.py` and would never have fired: the shells
# that wrote these scripts carry their whole text, and that text contains the
# path. Markers on disk instead. See ../FLEX-UF/scripts/WAITING.md.
set -uo pipefail
cd "$HOME/FLEX-PLUS"
PY=./.venv/bin/python
LOG=flexplus/logs/widths_decode.log
mkdir -p flexplus/logs flexplus/results
# The feature sweep trains four widths; wait for its four markers.
for _ in $(seq 1 900); do
  n=$(ls flexplus/results/narrow_stem_w*.done 2>/dev/null \
      | grep -vc decode || true)
  [ "${n:-0}" -ge 4 ] && break
  sleep 60
done
echo "=== decode-objective sweep $(date '+%F %T') ===" | tee -a "$LOG"
for W in 0.707 0.5; do
  M="flexplus/results/narrow_stem_w${W}_decode.done"
  [ -f "$M" ] && { echo "  width $W done" | tee -a "$LOG"; continue; }
  echo "  $(date '+%H:%M:%S')  training width $W on the decode" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_stem.py --width "$W" \
      --objective decode --steps 12000 --batch 4 --crop 256 --device cuda:0 \
      >>"$LOG" 2>&1 && : > "$M"
  echo "  $(date '+%H:%M:%S')  evaluating width $W" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=4 $PY -u flexplus/narrow_eval.py \
      --stems "flexplus/results/narrow_stem_w${W}_decode.pth" \
      --device cuda:0 --max_seqs 12 \
      --out "flexplus/results/narrow_eval_w${W}_decode.json" >>"$LOG" 2>&1
done
echo "=== decode sweep done $(date '+%F %T') ===" | tee -a "$LOG"
