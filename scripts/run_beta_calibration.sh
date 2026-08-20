#!/bin/bash
# The held-out beta table, and what it delivers on the test set.
#
# Two invocations, not one. Calibration decodes 512 validation images at five
# rates and evaluation decodes 53 CTC frames at five rates; together that is
# most of an hour, and every evaluation on this machine runs behind one lock so
# the checkpoint watchers can only measure while nothing else holds it. Two
# stages release it in between. It is also the shape the deployment has: the
# table is produced once, offline, and the decoder only reads it.
#
# Same checkpoint and same head as scripts/pin_router.sh, so the test-set
# bisection recomputed inside stage 2 is comparable line by line with
# results/router_RECIPE512_b01_PAPER.json, and the file records both.
set -u
cd "$HOME/FLEX-UF"
CK=runs/RECIPE512/ckpt_PAPER.pth.tar
R2=runs/RECIPE512/routers2/v2_lam1.3e-5.pth
OUT=results/beta_calibration.json
LOG=results/beta_calibration.log
for STAGE in calibrate evaluate; do
  echo "$(date '+%F %T')  -> $OUT ($STAGE, head $R2)" >>"$LOG"
  ( flock -w 21600 9 || exit 1
    CUDA_VISIBLE_DEVICES=2 ./.venv/bin/python -u scripts/beta_calibration.py \
      --ckpt "$CK" --router2 "$R2" --device cuda:0 --frames 1 --budget 0.1 \
      --stage "$STAGE" --out "$OUT" >>"$LOG" 2>&1 ) 9>/tmp/flexuf_eval.lock
  echo "$(date '+%F %T')  done rc=$?" >>"$LOG"
done
echo "=== done $(date '+%F %T') ===" >>"$LOG"
