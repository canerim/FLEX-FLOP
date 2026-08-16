#!/bin/bash
# The independent architecture experiment: ladder distillation + depth-scaled
# adapters, on the most promising configuration (j=2, 256px tiles, grid repair).
#
# Its control is runs/wdec_j2_p256_grid, which is identical except for these two
# additions -- so whatever the difference turns out to be is attributable.
set -u
cd "$HOME/FLEX-UF"
D=runs/wdec_j2_p256_distill; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=6 setsid nohup ./.venv/bin/python train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched --min_crop 512 \
  --epoch_offset 75 --new_lr_scale 20 --anchor_weight 1.0 \
  --distill_weight 1.0 --distill_teacher adjacent --adapter_kind scaled \
  --lambdas 10 2048 --batch_size 8 -n 6 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --seam_repair grid --tile_pad replicate --device 0 --tag wdec_j2_p256_distill \
  > "$D/train.log" 2>&1 < /dev/null &
disown
