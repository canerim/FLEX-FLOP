#!/bin/bash
# Measure SCRATCH105 every quarter hour, on the card it trains on.
#
# On its own card because the alternative is taking one of the other four runs'
# cards, and this measurement is 48 images: seconds, not the twenty minutes the
# CTC chain costs. It slows the run it watches by a percent or two and nothing
# else on the machine at all.
set -u
cd "$HOME/FLEX-UF"
TAG=${1:-SCRATCH105}
GPU=${2:-7}
EVERY=${3:-900}
while true; do
  ./.venv/bin/python scripts/val_scratch.py --run "$TAG" --device "cuda:$GPU" \
    >> "runs/$TAG/val.log" 2>&1
  sleep "$EVERY"
done
