#!/bin/bash
# setsid + python -u: a killed supervising shell took this job with it once, and
# buffered stdout hid the fact for twenty minutes.
set -u
cd "$HOME/FLEX-UF"
CUDA_VISIBLE_DEVICES=4 setsid nohup ./.venv/bin/python -u scripts/signalled_curve.py \
  --ckpt runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar --device cuda:0 \
  --out results/signalled_grid128.json > results/signalled.log 2>&1 < /dev/null &
disown
