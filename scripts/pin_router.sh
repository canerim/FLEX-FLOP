#!/bin/bash
# Configuration B on the pinned checkpoint, with the head the paper reports.
#
# The first attempt at this measured router2=None, which is the checkpoint's own
# jointly-trained head. That head has collapsed (DECISIONS 60) and is not what
# configuration B is: B is a head retrained against the FROZEN decoder, and it
# lives in runs/RECIPE512/routers2/. Measuring the wrong head produced 26.8% at
# q0 against the right head's 24.4%, and would have gone into the abstract.
set -u
cd "$HOME/FLEX-UF"
CK=runs/RECIPE512/ckpt_PAPER.pth.tar
R2=runs/RECIPE512/routers2/v2_lam1.3e-5.pth
LOG=results/pin_router.log
for B in 01 03 05; do
  D="0.$(echo $B | sed 's/^0//')"
  OUT=results/router_RECIPE512_b${B}_PAPER.json
  [ -s "$OUT" ] && { echo "have $OUT" >>"$LOG"; continue; }
  echo "$(date '+%F %T')  -> $OUT (budget $D, head $R2)" >>"$LOG"
  ( flock -w 21600 9 || exit 1
# GPU chosen by scripts/pick_gpu.sh, not hardcoded: the card this script
# used to pin to belongs to another user now. See watch_ckpts.sh.
GPU=$("$(dirname "$0")/pick_gpu.sh")
[ -n "${GPU:-}" ] || { echo "no GPU free of other users" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/router_curve.py \
      --ckpt "$CK" --router2 "$R2" --device cuda:0 --frames 1 --budget "$D" \
      --out "$OUT" >>"$LOG" 2>&1 ) 9>/tmp/flexuf_eval.lock
  echo "$(date '+%F %T')  done rc=$?" >>"$LOG"
done
echo "=== done $(date '+%F %T') ===" >>"$LOG"
