#!/bin/bash
# Launch the warm-start runs, detached.
#
# setsid, not bare nohup: nohup only blocks SIGHUP, and a supervising shell that
# is killed takes its whole process group with it. Four runs were lost that way.
#
# --epoch_offset 75: Microsoft's schedule read at the right place. The released
# checkpoint is the OUTPUT of epoch 105, so a warm start continues from the
# fine-tune regime (1e-5 @ 256px), not from epoch 0's from-scratch 2e-4. Applying
# 2e-4 cost the deepest exit 0.15/0.20/0.26 dB against released DCVC-UF at
# qp0/32/63 in ONE epoch -- measured by scripts/anchor_drift.py.
#
# --new_lr_scale 20: the zero-init adapters see 2e-4 while the inherited trunk
# sees 1e-5. One lr cannot serve a converged backbone and a module starting from
# nothing.
#
# --anchor_weight 1.0: the deepest exit is pinned to the released decoder by MSE.
# The mechanism existed and was left at its default of 0, which is why the drift
# happened at all.
set -u
cd "$HOME/FLEX-UF"
launch () {
  tag=$1; gpu=$2; lp=$3; bs=$4; sr=$5; sd=$6; extra=${7:-}
  D="runs/$tag"; mkdir -p "$D"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python train_flexuf_image.py \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$HOME/FLEX-UF/$D" \
    --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
    --freeze_encoder --train_patched $extra \
    --epoch_offset 75 --new_lr_scale 20 --anchor_weight 1.0 \
    --lambdas 10 2048 --batch_size "$bs" -n 6 -e 16 \
    --num_exits 6 --split_depth "$sd" --latent_patch "$lp" --latent_halo 2 \
    --adapter_kind conv1x1 --aux_weight 1.0 --aux_schedule constant \
    --seam_repair "$sr" --tile_pad replicate --device 0 --tag "$tag" \
    > "$D/train.log" 2>&1 < /dev/null &
  disown
  echo "  $tag -> GPU$gpu"
}
launch wdec_j2_p128      7 8  16 full 2
launch wdec_j2_p128_grid 4 8  16 grid 2
launch wdec_j2_p256_grid 7 16 8  grid 2 "--min_crop 512"
launch wdec_j4_p256      2 16 8  full 4 "--min_crop 512"
