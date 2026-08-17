#!/bin/bash
# BEST128 -- BEST with one variable changed: 128px tiles instead of 256px.
#
# Why now. The 2x2 isolation showed the 256px-trained checkpoint gives about half
# the saving of the 128px-trained one, and that this is NOT a tile-size-at-
# inference effect (0.4-1.5 points) but a property of the weights (12.4 points).
# The per-exit diagnostic located it: the 256px run holds the anchor better and
# is markedly worse at the shallow exits, which is where all the saving comes
# from.
#
# All four running experiments use 256px tiles, so if that is the cause, all four
# are on the wrong side of it.
#
# The comparison behind this is still confounded: the two checkpoints differ in
# tile size AND crop size (256 vs 512). This run holds the crop at 512 like BEST
# and changes only the tile, so BEST vs BEST128 attributes the difference to the
# tile alone -- which the earlier pair could not.
set -u
cd "$HOME/FLEX-UF"
D=runs/BEST128; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=1 setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched --min_crop 512 \
  --epoch_offset 75 --new_lr_scale 20 --anchor_weight 10.0 \
  --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent \
  --joint_router --router_beta 1.0 \
  --lambdas 10 2048 --batch_size 8 -n 8 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 8 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag BEST128 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
