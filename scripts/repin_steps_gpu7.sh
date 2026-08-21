#!/bin/bash
# The measurement half of repin.sh, run directly on this account's card 7.
#
# Why this exists. repin.sh serialises every step on one global lock,
# /tmp/flexuf_eval.lock, so that two evaluations never share a card. On this
# machine the per-run watchers hold that lock for their own sequences on card
# 4, flock is not fair, and a repin queued behind them made no progress in
# half an hour while card 7 sat free. The lock guards against a collision that
# is not happening: the watchers are on another card.
#
# Each step writes into a scratch directory and is moved into results/ only
# after it exits 0, so a step that dies leaves the previous measurement in
# place rather than a truncated file. The measurements are deterministic on a
# pinned checkpoint, so if the queued repin.sh later re-runs a step it writes
# the same bytes.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
TMP=${TMP_DIR:?set TMP_DIR to a scratch directory}
LOG=results/repin_gpu7.log
mkdir -p "$TMP"

step () {           # step <final path in results/> <command...>
  local out="$1"; shift
  local base; base=$(basename "$out")
  if [ -s "$TMP/$base.done" ]; then
    echo "  $(date '+%H:%M:%S')  $base already done, skipping" | tee -a "$LOG"
    return 0
  fi
  echo "  $(date '+%H:%M:%S')  $base" | tee -a "$LOG"
  ( flock -w 43200 9 || { echo "    lock timeout" | tee -a "$LOG"; exit 1; }
    CUDA_VISIBLE_DEVICES=7 "$@" ) 9>/tmp/flexuf_eval_gpu7.lock >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$TMP/$base" ]; then
    mv "$TMP/$base" "results/$base"
    : > "$TMP/$base.done"
    echo "    -> results/$base" | tee -a "$LOG"
  else
    echo "    FAILED rc=$rc, results/$base left as it was" | tee -a "$LOG"
  fi
}

echo "=== repin steps on card 7, $(date '+%F %T') ===" | tee -a "$LOG"

step supp_anchor_PAPER.json \
  $PY scripts/anchor_drift.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/supp_anchor_PAPER.json"

step saturation_RECIPE512_ctc53.json \
  $PY scripts/saturation.py --ckpt "$PIN" --qps 0 8 16 24 32 40 48 56 63 \
      --device cuda:0 --out "$TMP/saturation_RECIPE512_ctc53.json"

for B in 01 03 05; do
  D="0.$(echo $B | sed 's/^0//')"
  step router_RECIPE512_b${B}_PAPER.json \
    $PY -u scripts/router_curve.py --ckpt "$PIN" \
        --router2 runs/RECIPE512/routers2/v2_lam1.3e-5.pth --device cuda:0 \
        --frames 1 --budget "$D" \
        --out "$TMP/router_RECIPE512_b${B}_PAPER.json"
done

step raterank_RECIPE512_b01.json \
  $PY -u scripts/raterank_curve.py --ckpt "$PIN" --device cuda:0 \
      --budget 0.1 --out "$TMP/raterank_RECIPE512_b01.json"

step supp_per_class_budgets.json \
  $PY -u scripts/per_class.py --ckpt "$PIN" --qps 0 16 32 48 63 \
      --budgets 0.1 0.3 0.5 --frames 1 --device cuda:0 \
      --out "$TMP/supp_per_class_budgets.json"

step ceiling_measured.json \
  $PY scripts/ceiling_measured.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/ceiling_measured.json"

step why_qp_PAPER.json \
  $PY -u scripts/why_qp.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/why_qp_PAPER.json"

echo "=== measurement steps done $(date '+%F %T') ===" | tee -a "$LOG"
