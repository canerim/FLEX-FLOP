#!/bin/bash
# Everything BD-Rate is built from, measured on the pinned checkpoint.
#
# The reviewer's objection is provenance, not arithmetic. results/bdrate.json
# recorded its own sources and they were mixed: configuration A came from the
# pinned checkpoint, configuration B from ckpt_eval.pth.tar, which the watchers
# overwrite, and the bitrate came from runs/BEST entirely. The bitrate is a
# property of the frozen encoder and should not depend on which ladder was
# trained, but that is an argument, and BD-Rate is one of the two numbers the
# abstract turns on. It gets a measurement instead.
set -u
cd "$HOME/FLEX-UF"
CK=runs/RECIPE512/ckpt_PAPER.pth.tar
GPU=${GPU:-2}
LOG=results/pin_bdrate.log
PY=./.venv/bin/python

run () {
  local out="$1"; shift
  if [ -s "$out" ]; then echo "$(date '+%F %T')  have $out, skipping" >>"$LOG"; return; fi
  echo "$(date '+%F %T')  -> $out" >>"$LOG"
  ( flock -w 21600 9 || { echo "  lock timeout" >>"$LOG"; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" "$@" >>"$LOG" 2>&1 ) 9>/tmp/flexuf_eval.lock
  echo "$(date '+%F %T')  done rc=$? $(ls -la "$out" 2>/dev/null | awk '{print $5}') bytes" >>"$LOG"
}

echo "=== pinning BD-Rate inputs $(date '+%F %T') ===" >>"$LOG"

# The per-rate bitrate, on the pinned checkpoint rather than on runs/BEST.
run results/why_qp_PAPER.json $PY -u scripts/why_qp.py --ckpt "$CK" \
    --device cuda:0 --out results/why_qp_PAPER.json

# Configuration B at the three budgets, on the pinned checkpoint rather than on
# the file the watchers overwrite.
for B in 01 03 05; do
  D="0.${B#0}"; D="0.$(echo $B | sed 's/^0//')"
  run results/router_RECIPE512_b${B}_PAPER.json $PY -u scripts/router_curve.py \
      --ckpt "$CK" --device cuda:0 --frames 1 --budget "$D" \
      --out results/router_RECIPE512_b${B}_PAPER.json
done

echo "=== done $(date '+%F %T') ===" >>"$LOG"
