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
# GPU chosen by scripts/pick_gpu.sh, not hardcoded: the card this script
# used to pin to belongs to another user now. See watch_ckpts.sh.
GPU=$("$(dirname "$0")/pick_gpu.sh")
[ -n "${GPU:-}" ] || { echo "no GPU free of other users" >&2; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup ./.venv/bin/python -u scripts/signalled_curve.py \
  --ckpt runs/CONTROL/ckpt_epo0.pth.tar --device cuda:0 \
  --out results/signalled_control256.json > results/signalled_256.log 2>&1 < /dev/null &
disown
