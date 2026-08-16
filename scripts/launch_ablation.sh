#!/bin/bash
# One-factor-at-a-time ablation around the best configuration (j=2, 256px tiles).
#
# A single training run cannot be split across GPUs -- this trainer has no DDP --
# so freed cards advance the main experiment the only way they can: by testing,
# in parallel, which of its additions actually pays. Each arm differs from
# runs/wdec_j2_p256_distill in EXACTLY ONE thing, so every difference is
# attributable.
#
#   distill_deepest   teacher = deepest, not adjacent.  Tests directly, on our
#                     data, the FITEE 2024 claim that too large a student-teacher
#                     gap degrades the shallowest exits. If it is false here, the
#                     simpler choice wins and we should say so.
#   distill_plain     adapters back to a uniform 1x1.  Separates "distillation
#                     helped" from "more adapter capacity helped" -- the main run
#                     changes both at once and could not tell them apart.
#   distill_fullrep   seam repair back to the plain translation-invariant module.
#                     Isolates the grid gate at 256px tiles; it has only ever been
#                     measured at 128px.
#
# batch 6, not 8: the distillation term costs one extra full-depth trunk pass, and
# two arms have to share a card.
set -u
cd "$HOME/FLEX-UF"
arm () {
  tag=$1; gpu=$2; shift 2
  D="runs/$tag"; mkdir -p "$D"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python train_flexuf_image.py \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$HOME/FLEX-UF/$D" \
    --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
    --freeze_encoder --train_patched --min_crop 512 \
    --epoch_offset 75 --new_lr_scale 20 --anchor_weight 1.0 \
    --lambdas 10 2048 --batch_size 6 -n 5 -e 16 \
    --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
    --aux_weight 1.0 --aux_schedule constant \
    --tile_pad replicate --device 0 --tag "$tag" "$@" \
    > "$D/train.log" 2>&1 < /dev/null &
  disown
  echo "  $tag -> GPU$gpu"
}
arm wdec_j2_p256_dist_deepest 2 --distill_weight 1.0 --distill_teacher deepest \
                                --adapter_kind scaled --seam_repair grid
arm wdec_j2_p256_dist_plain   2 --distill_weight 1.0 --distill_teacher adjacent \
                                --adapter_kind conv1x1 --seam_repair grid
arm wdec_j2_p256_dist_fullrep 4 --distill_weight 1.0 --distill_teacher adjacent \
                                --adapter_kind scaled --seam_repair full
