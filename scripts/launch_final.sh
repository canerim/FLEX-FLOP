#!/bin/bash
# Two runs, not seven. Every piece in BEST earned its place by measurement:
#
#   j=2 / 256px tiles   pure seam 0.0879 dB at qp63 against 128px's 0.1785, at the
#                       same 43.4% ceiling -- the tile size dominates both others
#   GridSeamRepair      told where the seams are instead of inferring them; 256
#                       scalars, same 0.951% as the plain module
#   ladder distillation adapters supervised in feature space: spread +1.60 against
#                       the control's +4.34 at identical anchor fidelity, so the
#                       shallow exits are genuinely better rather than the deep one
#                       being dragged down
#   scaled adapters     capacity matched to the gap each exit bridges, billed per
#                       exit in the cost model
#   anchor_weight 10    58.6 dB fidelity to released DCVC-UF against 54.6 at
#                       weight 1. Drift was eating 74% of the 0.1 dB budget at
#                       qp63 before any routing happened.
#   joint learned router  the only router that did not collapse to a constant;
#                       0.044% of the decode
#
# One honest caveat: anchor_weight 10's effect on the CEILING was never measured
# -- its run was shelved before its checkpoint. Tightening the anchor constrains
# the shared trunk, which is also what the shallow exits need, so it could buy
# fidelity by costing headroom. BEST vs CONTROL shows the combined effect; if the
# result disappoints, this is the first knob to suspect.
#
# CONTROL is the same configuration with none of the additions, so the pair
# answers "does this work at all" before "which part did it".
#
# One run per card, so each gets a whole GPU instead of half -- roughly 2x the
# throughput of the seven-run layout, and two cards handed back to the server.
set -u
cd "$HOME/FLEX-UF"
run () {
  tag=$1; gpu=$2; shift 2
  D="runs/$tag"; mkdir -p "$D"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup ./.venv/bin/python train_flexuf_image.py \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$HOME/FLEX-UF/$D" \
    --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
    --freeze_encoder --train_patched --min_crop 512 \
    --epoch_offset 75 --new_lr_scale 20 \
    --lambdas 10 2048 --batch_size 8 -n 8 -e 16 \
    --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
    --aux_weight 1.0 --aux_schedule constant \
    --tile_pad replicate --device 0 --tag "$tag" "$@" \
    > "$D/train.log" 2>&1 < /dev/null &
  disown
  echo "  $tag -> GPU$gpu"
}
run BEST    6 --anchor_weight 10.0 --seam_repair grid --adapter_kind scaled \
              --distill_weight 1.0 --distill_teacher adjacent \
              --joint_router --router_beta 1.0
run CONTROL 7 --anchor_weight 1.0 --seam_repair grid --adapter_kind conv1x1
