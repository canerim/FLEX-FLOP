#!/bin/bash
# The signalled system at 256px tiles.
#
# At 128px it gave 24.2/21.2/15.7/11.0/8.6% saving at qp0/16/32/48/63, all under
# 0.1 dB of the released decoder with the exit map's own cost inside the bitrate.
# 256px halves the pure seam (0.0879 dB against 0.1785 at qp63) at the same
# 43.4% ceiling, and the map gets cheaper too -- 40 tiles per 1080p frame instead
# of 135 -- so both terms should move the right way.
#
# CONTROL is the checkpoint to use: 256px tiles with grid seam repair and nothing
# else, so this isolates the tile size rather than measuring a pile of additions.
set -u
cd "$HOME/FLEX-UF"
CUDA_VISIBLE_DEVICES=1 setsid nohup ./.venv/bin/python -u scripts/signalled_curve.py \
  --ckpt runs/CONTROL/ckpt_epo0.pth.tar --device cuda:0 \
  --out results/signalled_control256.json > results/signalled_256.log 2>&1 < /dev/null &
disown
