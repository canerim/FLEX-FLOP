#!/bin/bash
# The files the documents actually read that stages 1-3 leave behind.
#
# Twenty-eight files were stale or unlabelled after stage 2. Most are never
# read: make_paper_tables reaches them through pick(), which prefers a file
# measured on the pinned checkpoint, so an epoch-4 sibling shadows them. The
# exceptions are the chains in paper_metrics, which use J() -- first name that
# exists, not freshest -- so a stale first entry is read whatever else is
# there. Those are what this measures.
#
#   curve_RECIPE512_ctc53   read for the per-dataset table, epoch 0
#   router_*_fixed          read for configuration B in paper_metrics
#   hybrid_lorenz_b01       chosen because no _fixed sibling exists
#
# Card 0, like stage 3: card 7 belongs to the from-scratch run.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
R2=runs/RECIPE512/routers2/v2_lam1.3e-5.pth
GPU=0
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/repin_stage4.log
mkdir -p "$TMP"

step () {
  local base="$1"; shift
  [ -f "$TMP/$base.done" ] && { echo "  $base done" | tee -a "$LOG"; return 0; }
  echo "  $(date '+%H:%M:%S')  $base" | tee -a "$LOG"
  ( flock -w 43200 9 || exit 1
    CUDA_VISIBLE_DEVICES="$GPU" "$@" ) 9>/tmp/flexuf_eval_gpu${GPU}.lock \
    >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -s "$TMP/$base" ]; then
    mv "$TMP/$base" "results/$base"
    $PY scripts/stamp_ckpt.py "results/$base" "$PIN" >>"$LOG" 2>&1
    : > "$TMP/$base.done"
    echo "    -> results/$base" | tee -a "$LOG"
  else
    echo "    FAILED rc=$rc" | tee -a "$LOG"
  fi
}

echo "=== stage 4 on card $GPU, $(date '+%F %T') ===" | tee -a "$LOG"

step curve_RECIPE512_ctc53.json \
  $PY -u scripts/paper_curve.py --ckpt "$PIN" --qps 0 16 32 48 63 \
      --frames 1 --device cuda:0 --out "$TMP/curve_RECIPE512_ctc53.json"

for b in 01:0.1 03:0.3 05:0.5; do
  step "router_RECIPE512_b${b%%:*}_fixed.json" \
    $PY -u scripts/router_curve.py --ckpt "$PIN" --router2 "$R2" \
        --budget "${b#*:}" --frames 1 --device cuda:0 \
        --out "$TMP/router_RECIPE512_b${b%%:*}_fixed.json"
done

step hybrid_lorenz_b01.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.1 \
      --device cuda:0 --out "$TMP/hybrid_lorenz_b01.json"

step why_qp.json \
  $PY -u scripts/why_qp.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/why_qp.json"

step anchor_RECIPE512_ctc53.json \
  $PY scripts/anchor_drift.py --ckpt "$PIN" --qps 0 16 32 48 63 \
      --device cuda:0 --out "$TMP/anchor_RECIPE512_ctc53.json"

echo "=== stage 4 done $(date '+%F %T') ===" | tee -a "$LOG"
