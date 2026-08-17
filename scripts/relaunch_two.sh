#!/bin/bash
# BEST128 and FINE12, restarted only to pick up --ckpt_every.
#
# An epoch is 47451 steps -- ~10 h for these two -- and the trainer saved only at
# epoch end, so neither run could be MEASURED before tomorrow. FINE12 in
# particular is a new K=12 ladder that might simply be wrong. Both were under
# 700 steps in, so the restart costs minutes and buys a read every ~1 h.
set -u
cd "$HOME/FLEX-UF"
COMMON="--train_dataset /data10/shareddata/openimages/dcvc_train
 --freeze_encoder --train_patched --min_crop 512
 --epoch_offset 75 --new_lr_scale 20 --anchor_weight 10.0
 --seam_repair grid --distill_weight 1.0 --distill_teacher adjacent
 --lambdas 10 2048 --batch_size 8 -n 8 -e 16
 --latent_patch 8 --latent_halo 2
 --aux_weight 1.0 --aux_schedule constant --tile_pad replicate
 --device 0 --ckpt_every 4000"

go () {
  tag=$1; gpu=$2; shift 2
  D="runs/$tag"; mkdir -p "$D"
  echo "./.venv/bin/python -u train_flexuf_image.py $COMMON --save_dir $HOME/FLEX-UF/$D --tag $tag $*" \
    | tr -s ' \n' '  ' > "$D/cmd.txt"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
    $COMMON --save_dir "$HOME/FLEX-UF/$D" --tag "$tag" "$@" \
    > "$D/train.log" 2>&1 < /dev/null &
  disown; echo "  $tag -> GPU$gpu"
}

# BEST128: BEST with the 128px tile. Crop stays 512, so BEST vs BEST128 differs
# in the tile alone -- the variable the 2x2 isolation implicated.
go BEST128 1 --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
   --num_exits 6 --split_depth 2 --adapter_kind scaled --joint_router --router_beta 1.0

# FINE12: one exit per block, so each adapter bridges one skipped block instead
# of two. Aimed at the binding constraint, which is shallow-exit quality.
go FINE12 4 --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart_K12.pth.tar" \
   --num_exits 12 --split_depth 4 --adapter_kind conv1x1
