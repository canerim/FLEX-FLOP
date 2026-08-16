#!/bin/bash
# Does a stronger anchor buy back the budget, or just cost the ceiling?
#
# Measured on wdec_j2_p128_grid: the deepest exit drifts 0.074 dB from released
# DCVC-UF at qp63 -- 74% of the project's entire 0.1 dB budget, spent before any
# routing happens. --anchor_weight targets exactly that.
#
# But raising it constrains the shared trunk, which is also what the shallow
# exits need in order to improve, so it may buy back drift by lowering the
# oracle ceiling. That is a trade, and today has twice punished assuming a
# change is an improvement (arls, the richer router representation). So it is
# measured: identical to runs/wdec_j2_p128_grid in every respect except the
# anchor weight, 1.0 -> 10.0.
set -u
cd "$HOME/FLEX-UF"
D=runs/wdec_j2_p128_anchor10; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=2 setsid nohup ./.venv/bin/python train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched \
  --epoch_offset 75 --new_lr_scale 20 --anchor_weight 10.0 \
  --lambdas 10 2048 --batch_size 16 -n 6 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 8 --latent_halo 2 \
  --adapter_kind conv1x1 --aux_weight 1.0 --aux_schedule constant \
  --seam_repair grid --tile_pad replicate --device 0 --tag wdec_j2_p128_anchor10 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
