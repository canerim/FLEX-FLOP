#!/bin/bash
# The jointly trained head, on the checkpoint the paper reports.
#
# results/router_RECIPE512_b0*_jointhead.json is the configuration where the
# decoder uses the head that came out of joint training rather than a head
# fitted afterwards against the frozen decoder. It was measured before the
# repin, so the section that sets the two head-training regimes against each
# other had one of them on epoch 0 and the other on epoch 4.
#
# router_curve.py without --router2 uses the checkpoint's own head, which is
# what "jointly trained" means here.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
PIN=runs/RECIPE512/ckpt_PAPER.pth.tar
GPU=0
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/repin_stage6.log
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

echo "=== stage 6 on card $GPU, $(date '+%F %T') ===" | tee -a "$LOG"
for b in 01:0.1 03:0.3 05:0.5; do
  step "router_RECIPE512_b${b%%:*}_jointhead.json" \
    $PY -u scripts/router_curve.py --ckpt "$PIN" --budget "${b#*:}" \
        --frames 1 --device cuda:0 \
        --out "$TMP/router_RECIPE512_b${b%%:*}_jointhead.json"
done
echo "=== stage 6 done $(date '+%F %T') ===" | tee -a "$LOG"
