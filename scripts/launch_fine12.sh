#!/bin/bash
# FINE12 -- twelve exits instead of six, aimed at the constraint the diagnostic found.
#
# The per-exit diagnostic says the binding limit is shallow-exit quality: exit 2
# sits 0.34 dB below the released decoder at qp0 while exit 4 sits at 0.06, and
# all the saving comes from the shallow end. With K=6 over 12 blocks, one adapter
# must stand in for TWO skipped DepthConvBlocks. With K=12 it stands in for one.
#
# split_depth 4 keeps the shared stem at four blocks, exactly as j=2 does at K=6,
# so the stem is unchanged and only the ladder's granularity differs. The cost
# ladder also becomes finer -- twelve rungs instead of six -- which by the hull
# argument can only enlarge the achievable set, never shrink it.
#
# The cost is more adapters: 11 instead of 5. At conv1x1 that is C^2 each, an
# eighth of a block, so the exit-2 tile pays the same as before; only the
# parameter count grows.
set -u
cd "$HOME/FLEX-UF"
D=runs/FINE12; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=4 setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart_K12.pth.tar" \
  --freeze_encoder --train_patched --min_crop 512 \
  --epoch_offset 75 --new_lr_scale 20 --anchor_weight 10.0 \
  --seam_repair grid --adapter_kind conv1x1 \
  --distill_weight 1.0 --distill_teacher adjacent \
  --lambdas 10 2048 --batch_size 8 -n 8 -e 16 \
  --num_exits 12 --split_depth 4 --latent_patch 8 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag FINE12 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
