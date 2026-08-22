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
# The head fitted to the pinned weights, not the one fitted on 19 August to
# whatever ckpt_eval held then. Configuration C is built on configuration B's
# head, so C has to be measured on the head B reports.
R2=runs/RECIPE512/routers2/v2_lam1.3e-5_e4.pth
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

# The three configuration-B curves that stood here are already measured on the
# pin with the head fitted to it, in results/router_RECIPE512_b0*_e4head.json,
# which is what every reader picks first. Re-measuring them with the old head
# would spend an hour producing files nothing reads.

step hybrid_RECIPE512_b01_e4head.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.1 \
      --device cuda:0 --out "$TMP/hybrid_RECIPE512_b01_e4head.json"

# The other side of the Gini comparison: the same configuration C on the head
# retrained after the exit-mask fix, on the pinned weights. Its partner,
# results/hybrid_RECIPE512_b01_fixed.json, is the pre-fix head on the same
# weights, so the pair differs only in the head, which is what the claim is
# about.
step hybrid_v3_b01_pin.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" \
      --router2 runs/RECIPE512/routers2/v3_masked_lam1.3e-5.pth --budget 0.1 \
      --device cuda:0 --out "$TMP/hybrid_v3_b01_pin.json"

step hybrid_lorenz_b01.json \
  $PY -u scripts/hybrid_curve.py --ckpt "$PIN" --router2 "$R2" --budget 0.1 \
      --device cuda:0 --out "$TMP/hybrid_lorenz_b01.json"

step why_qp.json \
  $PY -u scripts/why_qp.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/why_qp.json"

# The anchor step that stood here measured $PIN at five rates over the 53 CTC
# frames -- which is exactly what results/supp_anchor_PAPER.json already is on
# this pin. Its only reader takes it as a fallback behind that file. Twenty
# minutes of card time for a second copy of a number we hold.

echo "=== stage 4 done $(date '+%F %T') ===" | tee -a "$LOG"
