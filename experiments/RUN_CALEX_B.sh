#!/bin/bash
# Tier B, both arms. A copy of the main trainer lives in flexplus/train_calex.py;
# the ONLY difference from the main recipe is --exit_prior, and the control arm
# omits it, so a gain cannot be credited to "more training at a larger lr".
#
# Frozen backbone: 3,113,366 / 45,292,694 params (6.87%) -- adapters and seam
# repair only. --freeze_encoder is deliberately NOT passed: the trainer's
# if/elif makes it shadow --freeze_backbone, which is how UNLOCK v1 ended up
# training 17.3M params by accident.
ARM=${1:-prior}
EXTRA=""
[ "$ARM" = "prior" ] && EXTRA="--exit_prior 0,0,0.710,0.211,0.043,0.036"
cd ~/FLEX-UF
CUDA_VISIBLE_DEVICES=4 nohup ~/FLEX-UF/.venv/bin/python -u \
  ~/FLEX-PLUS/flexplus/train_calex.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir /home/can_karsal/FLEX-PLUS/experiments/CALEX_B_$ARM \
  --pretrain /home/can_karsal/FLEX-UF/runs/RECIPE512/ckpt_PIN_e9.pth.tar \
  --freeze_backbone --train_patched --epoch_offset 99 --new_lr_scale 20 \
  --anchor_weight 10.0 --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent --lambdas 10 2048 \
  --batch_size 8 -n 8 -e 1 --num_exits 6 --split_depth 2 \
  --latent_patch 16 --latent_halo 2 --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --ckpt_every 1000 --steps_per_epoch 3000 --tag CALEX_B_$ARM \
  $EXTRA \
  > /home/can_karsal/FLEX-PLUS/experiments/CALEX_B_$ARM.log 2>&1 &
echo "CALEX_B_$ARM basladi, pid $!"
