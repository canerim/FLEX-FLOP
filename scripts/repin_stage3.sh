#!/bin/bash
# The supplement's own measurements, on the pinned checkpoint.
#
# Stages 1 and 2 move the paper's chain. These are the files only the
# supplement reads -- the frontier it breaks down per sequence, the worst
# operating points, the exit maps and the seam picture. supp_experiments2.sh
# produced them once and skips anything whose output already exists, so it
# cannot be used to move them; this repeats the same commands with the same
# arguments and lets the driver's own guard decide what to redo.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
CURVE=results/supp_paper_curve_PAPER.json
# Card 0, not 7. Card 7 is the from-scratch run's, and sharing it cost that
# run about sixty per cent of its step rate. Card 0 carries RECIPE512 and has
# the most free memory of the cards this account holds, so the measurements
# go there and the long run gets its own card back. Set deliberately rather
# than taken from the environment: the caller passes 7.
GPU=0
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/repin_stage3.log
mkdir -p "$TMP"

step () {
  local base="$1"; shift
  if [ -f "$TMP/$base.done" ]; then
    echo "  $(date '+%H:%M:%S')  $base already done" | tee -a "$LOG"; return 0
  fi
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

echo "=== stage 3 on card $GPU, $(date '+%F %T') ===" | tee -a "$LOG"

# The frontier the per-sequence breakdown and the BD numbers are built on.
step supp_paper_curve_PAPER.json \
  $PY -u scripts/paper_curve.py --ckpt "$PIN" --qps 0 16 32 48 63 \
      --frames 1 --device cuda:0 --out "$TMP/supp_paper_curve_PAPER.json"
if [ -s "$CURVE" ]; then
  for CONV in per_frame pooled; do
    step "supp_bd_PAPER_${CONV}.json" \
      $PY -u scripts/bd_saving.py --curve "$CURVE" --convention "$CONV" \
          --out "$TMP/supp_bd_PAPER_${CONV}.json"
  done
fi

step supp_opquality_PAPER.json \
  $PY -u scripts/operating_point_quality.py --ckpt "$PIN" \
      --signalled results/signalled_RECIPE512_ctc53.json --budget 0.1 \
      --qps 0 16 32 48 63 --worst 40 --device cuda:0 \
      --out "$TMP/supp_opquality_PAPER.json"

step supp_quant_PAPER.json \
  $PY -u scripts/quant_sweep.py --ckpt "$PIN" --bits 8 6 4 \
      --device cuda:0 --out "$TMP/supp_quant_PAPER.json"

step beta_calibration.json \
  $PY -u scripts/beta_calibration.py --ckpt "$PIN" --device cuda:0 \
      --out "$TMP/beta_calibration.json"

# The five exit maps and the seam picture write a figure and a sidecar; the
# sidecar is the file the supplement quotes, so it is what the guard watches.
for S in Bosphorus BasketballDrive FourPeople PartyScene RaceHorses_416x240; do
  L=$(echo "$S" | tr 'A-Z' 'a-z')
  step "supp_exitmap_${L}_q32_b01.json" \
    $PY -u scripts/exit_map_figure.py --ckpt "$PIN" --seq "$S" --qp 32 \
        --budget 0.1 --device cuda:0 \
        --out "paper/figures/exitmap_PAPER_${L}_q32_b01.png" \
        --sidecar "$TMP/supp_exitmap_${L}_q32_b01.json"
done

step supp_seam_problem_bosphorus_q63.json \
  $PY -u scripts/seam_problem.py --ckpt "$PIN" --seq Bosphorus --qp 63 \
      --device cuda:0 --out paper/figures/seam_PAPER_bosphorus_q63.png \
      --sidecar "$TMP/supp_seam_problem_bosphorus_q63.json"

echo "=== stage 3 done $(date '+%F %T') ===" | tee -a "$LOG"
