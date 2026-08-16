#!/bin/bash
# The router problem, attacked at its root.
#
# The router was measured blind: none of six hand-made stem statistics
# correlates past |r| = 0.12 with the oracle's choice, and widening the
# description to 768 pooled channels measured WORSE at the operating point that
# matters. Both attempts asked the same question -- which fixed function of a
# fixed stem predicts the right exit -- and the stem was never built to answer
# it. Here the stem and the router are trained together, so the representation
# can become routable instead of being hoped to already be.
#
# MAC budget, the user's one constraint: a 1x1 of 384->16 on the stem, which
# sits at 1/64 of the pixel count, is 96 MAC per RGB pixel against the decode's
# 217,113 -- 0.044%. GridSeamRepair is 0.951% for scale. The pooling and the
# 8,726-parameter MLP act on one vector per tile and do not register.
set -u
cd "$HOME/FLEX-UF"
D=runs/joint_j2_p256; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=4 setsid nohup ./.venv/bin/python train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched --min_crop 512 --anchor_weight 1.0 \
  --epoch_offset 75 --new_lr_scale 20 \
  --joint_router --router_beta 1.0 --gumbel_tau 1.0 \
  --distill_weight 1.0 --distill_teacher adjacent \
  --adapter_kind scaled --seam_repair grid \
  --lambdas 10 2048 --batch_size 6 -n 5 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag joint_j2_p256 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
