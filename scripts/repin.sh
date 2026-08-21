#!/bin/bash
# Move the paper onto a later checkpoint, in one command.
#
# Every table, macro and figure in the paper comes from one set of weights, so
# changing which set is not a matter of editing a number: the whole chain has to
# be re-measured and regenerated. This is that chain, in dependency order.
#
# It refuses to run against a moving file. runs/*/ckpt_eval.pth.tar is
# overwritten by the training loop and was overwritten mid-measurement once
# already, which is the reason the pinned name exists. Pass a snapshot, or let
# this script take one for you.
#
#   scripts/repin.sh runs/RECIPE512/ckpt_eval.pth.tar     # snapshots it first
#   scripts/repin.sh runs/RECIPE512/ckpt_PIN_e3.pth.tar   # uses it as given
set -euo pipefail
cd "$HOME/FLEX-UF"
SRC=${1:?usage: repin.sh <checkpoint>}
# GPU 2 belongs to another account on this machine, and it was the default
# here for a week. Nothing in this script checks, so the default is now a card
# this account owns; scripts/gpu.py picks the freest of those when it can.
GPU=${GPU:-$(./.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
import gpu
print(gpu.pick('cuda:7').replace('cuda:', '') or 7)" 2>/dev/null || echo 7)}
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
LOG=results/repin.log

EP=$($PY -c "
import torch,sys
c=torch.load('$SRC',map_location='cpu',weights_only=False)
print(c.get('epoch','?'))")
echo "=== repin from $SRC (epoch $EP) at $(date '+%F %T') ===" | tee -a "$LOG"

if [[ "$SRC" == *ckpt_eval.pth.tar || "$SRC" == *ckpt_step.pth.tar ]]; then
  SNAP="runs/RECIPE512/ckpt_PIN_e${EP}.pth.tar"
  echo "  $SRC is a moving file; snapshotting to $SNAP" | tee -a "$LOG"
  cp "$SRC" "$SNAP"
  SRC="$SNAP"
fi

cp "$PIN" "${PIN%.pth.tar}_prev.pth.tar"
cp "$SRC" "$PIN"
echo "  pinned $SRC -> $PIN (previous kept as ${PIN%.pth.tar}_prev.pth.tar)" | tee -a "$LOG"

run () {
  echo "  $(date '+%H:%M:%S')  $*" | tee -a "$LOG"
  ( flock -w 21600 9 || { echo "    lock timeout" | tee -a "$LOG"; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" "$@" >>"$LOG" 2>&1 ) 9>/tmp/flexuf_eval.lock
}

# 1. the anchor first: if the deepest exit has drifted from the release, every
#    saving below is measured against the wrong reference.
run $PY scripts/anchor_drift.py --ckpt "$PIN" --device cuda:0 \
    --out results/supp_anchor_PAPER.json
# 2. the headline, four budgets
run $PY -u scripts/signalled_curve.py --ckpt "$PIN" --qps 0 16 32 48 63 \
    --frames 1 --device cuda:0 --budgets 0.1 0.2 0.3 0.5 \
    --out results/signalled_RECIPE512_ctc53.json
# 3. floor and saturation
run $PY scripts/saturation.py --ckpt "$PIN" --qps 0 8 16 24 32 40 48 56 63 \
    --device cuda:0 --out results/saturation_RECIPE512_ctc53.json
# 4. configuration B at three budgets, with the head the paper reports
for B in 01 03 05; do
  D="0.$(echo $B | sed 's/^0//')"
  run $PY -u scripts/router_curve.py --ckpt "$PIN" \
      --router2 runs/RECIPE512/routers2/v2_lam1.3e-5.pth --device cuda:0 \
      --frames 1 --budget "$D" --out results/router_RECIPE512_b${B}_PAPER.json
done
# 5. the parameter-free rule
run $PY -u scripts/raterank_curve.py --ckpt "$PIN" --device cuda:0 \
    --budget 0.1 --out results/raterank_RECIPE512_b01.json
# 6. per class, three budgets
run $PY -u scripts/per_class.py --ckpt "$PIN" --qps 0 16 32 48 63 \
    --budgets 0.1 0.3 0.5 --frames 1 --device cuda:0 \
    --out results/supp_per_class_budgets.json
# 7. the per-exit cost, hooks against the model
run $PY scripts/ceiling_measured.py --ckpt "$PIN" --device cuda:0 \
    --out results/ceiling_measured.json
# 8. the bitrate BD-Rate is built from
run $PY -u scripts/why_qp.py --ckpt "$PIN" --device cuda:0 \
    --out results/why_qp_PAPER.json
# 8b. per-component rate-distortion. Reads the sweep from step 2 for its
#     operating points, so it has to come after it.
run $PY -u scripts/rd_yuv.py --ckpt "$PIN" --device cuda:0 \
    --out results/rd_yuv_PAPER.json

# 9. figures that read the checkpoint rather than a results file
for F in patchify_figure pipeline_stage_figs seam_repair_grid_figure; do
  run $PY scripts/$F.py --device cuda:0
done

# 10. everything derived, then the documents
$PY scripts/paper_metrics.py    >>"$LOG" 2>&1
$PY scripts/make_paper_tables.py >>"$LOG" 2>&1
$PY scripts/nature_plots.py      >>"$LOG" 2>&1
# The curves against the released decoder, and the BD trade-off, both read
# results files that step 10 has just rewritten.
$PY scripts/rd_vs_uf_figure.py   >>"$LOG" 2>&1
$PY scripts/bd_figure.py --layout wide   --out docs/figures/bdrate_wide.png >>"$LOG" 2>&1
$PY scripts/bd_figure.py --layout column --out docs/figures/bdrate.png      >>"$LOG" 2>&1
$PY scripts/spread_figs.py       >>"$LOG" 2>&1
$PY scripts/paper_figures.py     >>"$LOG" 2>&1
$PY scripts/build_pdf.py         2>&1 | tee -a "$LOG"
$PY scripts/build_supp_pdf.py    >>"$LOG" 2>&1
$PY scripts/check_paper.py       2>&1 | tail -3 | tee -a "$LOG"
echo "=== repin done $(date '+%F %T') ===" | tee -a "$LOG"
