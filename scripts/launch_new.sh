#!/bin/bash
# Two experiments the user asked for, both on the most promising configuration.
#
# A · runs/heads_only_j2_p256
#     Backbone completely frozen; only the exit heads train. The deepest exit is
#     then bit-exact released DCVC-UF BY CONSTRUCTION -- not pinned by a loss
#     term, not measured after the fact, structurally incapable of drifting,
#     because no tensor it uses is in the optimiser. That is the strongest form
#     of the guarantee we have been chasing all day. With nothing else competing
#     for capacity the heads can afford to be larger: FFNAdapter at every exit,
#     the same expand/activate/contract the skipped blocks actually contain.
#
# B · runs/coupled_j2_p256
#     Canvas-coupled tiling + ladder distillation. Coupling removes the seam
#     exactly where neighbours share a depth (measured 0.0000 dB) and costs
#     +0.032% of the decode; where they differ it currently loses 0.12 dB because
#     a deep block reads a shallow neighbour's feature. Distillation trains those
#     features to match. The experiment is whether the second fixes the first.
set -u
cd "$HOME/FLEX-UF"
run () {
  tag=$1; gpu=$2; shift 2
  D="runs/$tag"; mkdir -p "$D"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python train_flexuf_image.py \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$HOME/FLEX-UF/$D" \
    --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
    --train_patched --min_crop 512 \
    --epoch_offset 75 --new_lr_scale 20 \
    --lambdas 10 2048 --batch_size 8 -n 6 -e 16 \
    --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
    --aux_weight 1.0 --aux_schedule constant \
    --tile_pad replicate --device 0 --tag "$tag" "$@" \
    > "$D/train.log" 2>&1 < /dev/null &
  disown
  echo "  $tag -> GPU$gpu"
}
# A: frozen backbone, big heads. No --anchor_weight: freezing makes it vacuous.
run heads_only_j2_p256 2 --freeze_backbone --adapter_kind ffn --seam_repair grid
# B: coupling + distillation, encoder frozen, anchor pinned.
run coupled_j2_p256    2 --freeze_encoder --anchor_weight 1.0 \
                         --tile_coupling --distill_weight 1.0 \
                         --distill_teacher adjacent --adapter_kind scaled \
                         --seam_repair none
