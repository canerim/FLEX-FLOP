#!/bin/bash
# Everything else the documents read, on the pinned checkpoint.
#
# repin.sh covers nine files. check_epoch.py counts fifty-seven that are
# neither structural nor deliberately historical, so repin.sh was never a
# complete migration -- it moved the headline and left the supplement behind.
# This is the rest, in one pass, on one card, with every output stamped with
# the checkpoint it came from so that check_epoch can see it.
#
# Ordered cheapest first. A pass that dies four hours in should have banked
# the small things rather than none of them.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
R2=runs/RECIPE512/routers2/v2_lam1.3e-5.pth
GPU=${GPU:-7}
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/repin_stage2.log
mkdir -p "$TMP"

step () {            # step <basename> <command...>
  local base="$1"; shift
  if [ -s "$TMP/$base.done" ]; then
    echo "  $(date '+%H:%M:%S')  $base already done" | tee -a "$LOG"; return 0
  fi
  echo "  $(date '+%H:%M:%S')  $base" | tee -a "$LOG"
  ( flock -w 43200 9 || exit 1
    CUDA_VISIBLE_DEVICES="$GPU" "$@" ) 9>/tmp/flexuf_eval_gpu7.lock \
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

echo "=== stage 2 on card $GPU, $(date '+%F %T') ===" | tee -a "$LOG"

# --- the non-PAPER router names the supplement still reads
for b in 01:0.1 03:0.3 05:0.5; do
  step "router_RECIPE512_b${b%%:*}.json" \
    $PY -u scripts/router_curve.py --ckpt "$PIN" --router2 "$R2" \
        --budget "${b#*:}" --device cuda:0 --frames 1 \
        --out "$TMP/router_RECIPE512_b${b%%:*}.json"
done

# --- the parameter-free rule at the second budget
step raterank_RECIPE512_b03.json \
  $PY -u scripts/raterank_curve.py --ckpt "$PIN" --budget 0.3 \
      --device cuda:0 --out "$TMP/raterank_RECIPE512_b03.json"
step raterank_RECIPE512_b05.json \
  $PY -u scripts/raterank_curve.py --ckpt "$PIN" --budget 0.5 \
      --device cuda:0 --out "$TMP/raterank_RECIPE512_b05.json"

# --- configuration C, both predictors
step hybrid_RECIPE512_b01_fixed.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.1 \
      --device cuda:0 --out "$TMP/hybrid_RECIPE512_b01_fixed.json"
step hybrid_raterank_b01.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.1 \
      --predictor raterank --device cuda:0 \
      --out "$TMP/hybrid_raterank_b01.json"
step hybrid_RECIPE512_b03_fixed.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.3 \
      --device cuda:0 --out "$TMP/hybrid_RECIPE512_b03_fixed.json"

# --- the static controls
step static_RECIPE512_b01.json \
  $PY -u scripts/static_baseline.py --ckpt "$PIN" --budget 0.1 \
      --device cuda:0 --out "$TMP/static_RECIPE512_b01.json"

# --- per class, with the exit histogram, both file names the documents use
step per_class_RECIPE512.json \
  $PY -u scripts/per_class.py --ckpt "$PIN" --budgets 0.1 0.3 0.5 \
      --device cuda:0 --out "$TMP/per_class_RECIPE512.json"

# --- the two decoder-side signals blended
step combined_RECIPE512_b01.json \
  $PY -u scripts/combined_curve.py --ckpt "$PIN" --router2 "$R2" \
      --device cuda:0 --out "$TMP/combined_RECIPE512_b01.json"

# --- the dense budget grid the band collapse is fitted on
step signalled_RECIPE512_grid.json \
  $PY -u scripts/signalled_curve.py --ckpt "$PIN" \
      --budgets 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \
      --device cuda:0 --frames 1 --out "$TMP/signalled_RECIPE512_grid.json"

# --- the ablations. Comparisons within a checkpoint, so they were valid on
#     the old one; they are re-measured here because the paper quotes their
#     numbers next to numbers from this one.
step adapter_ablation.json \
  $PY -u scripts/adapter_ablation.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/adapter_ablation.json"
step coupling_ablation.json \
  $PY -u scripts/coupling_ablation.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/coupling_ablation.json"
step map_transfer.json \
  $PY -u scripts/map_transfer.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/map_transfer.json"

# --- the per-component split, which reads the sweep from stage 1
step rd_yuv_PAPER.json \
  $PY -u scripts/rd_yuv.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/rd_yuv_PAPER.json"

echo "=== stage 2 done $(date '+%F %T') ===" | tee -a "$LOG"
