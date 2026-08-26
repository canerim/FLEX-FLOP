#!/bin/bash
# Tier B, corrected. Three changes, each against a measured fault of the first
# attempt (which lost 8.21 points on its control arm):
#
#  1. The anchor is pinned to the RELEASE, as the main run does. Pinning it to
#     the starting checkpoint made anchor_mse identically zero -- it is computed
#     on the full-frame path, where a frozen backbone's deepest exit cannot
#     move -- so nothing constrained the tiled deepest path and the floor drifted
#     upward at high rate.
#  2. --epoch_offset 105 puts the schedule at lr 1e-6, which with
#     --new_lr_scale 20 gives the adapters 2e-5: the regime the main run is
#     converged in. The first attempt used offset 99, a hundred times that.
#  3. 6000 steps rather than 3000.
#
# Both arms again: the prior arm and a control with uniform weights, because
# the first attempt showed the control is what separates the idea from the
# harness.
ARM=${1:-prior}
GPU=${2:-4}
EXTRA=""
[ "$ARM" = "prior" ] && EXTRA="--exit_prior 0,0,0.710,0.211,0.043,0.036"
cd ~/FLEX-UF
CUDA_VISIBLE_DEVICES=$GPU nohup ~/FLEX-UF/.venv/bin/python -u \
  ~/FLEX-PLUS/flexplus/train_calex.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir /home/can_karsal/FLEX-PLUS/experiments/CALEX_B2_$ARM \
  --pretrain /home/can_karsal/FLEX-UF/runs/RECIPE512/ckpt_PIN_e9.pth.tar \
  --anchor_ckpt /home/can_karsal/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar \
  --freeze_backbone --train_patched --epoch_offset 105 --new_lr_scale 20 \
  --anchor_weight 10.0 --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent --lambdas 10 2048 \
  --batch_size 8 -n 8 -e 1 --num_exits 6 --split_depth 2 \
  --latent_patch 16 --latent_halo 2 --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --ckpt_every 1500 --steps_per_epoch 6000 \
  --tag CALEX_B2_$ARM $EXTRA \
  > /home/can_karsal/FLEX-PLUS/experiments/CALEX_B2_$ARM.log 2>&1 &
echo "CALEX_B2_$ARM basladi kart $GPU, pid $!"
