#!/bin/bash
# The budget grid, on budgets that lie inside epoch 4's bands.
#
# The grid samples 0.05 to 0.5 dB, which was right when the ladder saturated
# around 0.3 to 0.5. On epoch 4 it saturates much earlier -- q0's usable band
# is [0.026, 0.139] -- so only three of the nine budgets fall inside it, and
# the band collapse is being fitted and scored on three points at the lowest
# rate. That is why the rescaled spread reads 3.2 points against the 1.7 the
# paper reports: the phenomenon did not weaken, the sampling went stale with
# the checkpoint.
#
# New grid: ten budgets from 0.03 to 0.30, which puts six or more inside every
# rate's band.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
GPU=0
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/repin_stage5.log
mkdir -p "$TMP"
echo "=== stage 5 on card $GPU, $(date '+%F %T') ===" | tee -a "$LOG"
if [ -f "$TMP/signalled_RECIPE512_grid.json.done" ]; then
  echo "  already done" | tee -a "$LOG"; exit 0
fi
( flock -w 43200 9 || exit 1
  CUDA_VISIBLE_DEVICES="$GPU" $PY -u scripts/signalled_curve.py --ckpt "$PIN" \
      --budgets 0.03 0.04 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 \
      --frames 1 --device cuda:0 \
      --out "$TMP/signalled_RECIPE512_grid.json"
) 9>/tmp/flexuf_eval_gpu${GPU}.lock >>"$LOG" 2>&1
if [ -s "$TMP/signalled_RECIPE512_grid.json" ]; then
  cp results/signalled_RECIPE512_grid.json "$TMP/grid_9budget.bak" 2>/dev/null
  mv "$TMP/signalled_RECIPE512_grid.json" results/signalled_RECIPE512_grid.json
  $PY scripts/stamp_ckpt.py results/signalled_RECIPE512_grid.json "$PIN" \
      >>"$LOG" 2>&1
  : > "$TMP/signalled_RECIPE512_grid.json.done"
  echo "  -> results/signalled_RECIPE512_grid.json" | tee -a "$LOG"
  $PY scripts/tradeoff_figure.py 2>&1 | tail -2 | tee -a "$LOG"
else
  echo "  FAILED" | tee -a "$LOG"
fi
echo "=== stage 5 done $(date '+%F %T') ===" | tee -a "$LOG"
