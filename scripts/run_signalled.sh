#!/bin/bash
# setsid + python -u: a killed supervising shell took this job with it once, and
# buffered stdout hid the fact for twenty minutes.
set -u
cd "$HOME/FLEX-UF"
# GPU chosen by scripts/pick_gpu.sh, not hardcoded: the card this script
# used to pin to belongs to another user now. See watch_ckpts.sh.
GPU=$("$(dirname "$0")/pick_gpu.sh")
[ -n "${GPU:-}" ] || { echo "no GPU free of other users" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup ./.venv/bin/python -u scripts/signalled_curve.py \
  --ckpt runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar --device cuda:0 \
  --out results/signalled_grid128.json > results/signalled.log 2>&1 < /dev/null &
disown
