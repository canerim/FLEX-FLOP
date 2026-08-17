#!/bin/bash
# RECIPE512 -- the run with no deviation from Microsoft's schedule left in it.
#
# Every other warm-start run here uses --epoch_offset 75, which is lr 1e-5 at
# 256x256, and then overrides the patch size with --min_crop 512. That override
# is mine, not theirs, and it exists for a structural reason: a 256px tile inside
# a 256px crop is ONE tile, so no tile border is ever convolved against a missing
# neighbour and patched training would silently train a configuration that cannot
# occur at inference.
#
# Offset 99 removes the override instead of justifying it. Microsoft's own
# schedule spends its last fifteen epochs at 512x512, and offset 99 lands exactly
# in that phase at lr 1e-5. So the patch size the experiment needs is the patch
# size the recipe asks for, and --min_crop is not passed at all. Verified against
# ~/DCVC/train_image.py by scripts/verify_recipe.py.
#
# No --joint_router. The router problem was resolved by signalling the exit map
# in the bitstream at 1.3e-4 bpp rather than predicting it at the decoder, which
# makes a learned router moot; leaving it in would spend capacity and 0.044% of
# the decode on a decision that is now transmitted.
#
# Everything else is the set that earned its place by measurement: 256px tiles
# (seam 0.0879 dB against 128px's 0.1785 at the same ceiling), grid seam repair,
# depth-scaled adapters, ladder distillation, anchor weight 10.
set -u
cd "$HOME/FLEX-UF"
D=runs/RECIPE512; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=0 setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched \
  --epoch_offset 99 --new_lr_scale 20 --anchor_weight 10.0 \
  --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent \
  --lambdas 10 2048 --batch_size 8 -n 8 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag RECIPE512 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
