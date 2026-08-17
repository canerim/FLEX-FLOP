#!/bin/bash
# VERBATIM -- Microsoft's recipe applied to early exit, with one stated exception.
#
# The exception is the encoder: --freeze_encoder holds the analysis transform,
# hyperprior and entropy model at the released values, so the latent and the
# bitstream are byte-identical to stock DCVC-UF and any measured difference is
# the decoder alone. That is the user's stated carve-out and the only one.
#
# Everything else is theirs:
#
#   --epoch_offset 90 -e 15   their schedule's final phase, run in full: five
#                             epochs at 2e-4, four at 5e-5, four at 1e-5, two at
#                             1e-6, all at 512x512. A complete self-contained
#                             phase with its own decay, which is what continuing
#                             a converged model means -- not epoch 0's 2e-4,
#                             which cost the deepest exit 0.20 dB in one epoch.
#   --batch_size 8 --grad_accum 2   effective batch 16, the recipe's value.
#                             Microsoft reach 16 across GPUs; at 512x512 it does
#                             not fit on one card, so it is accumulated rather
#                             than quietly halved.
#   AdamW 1e-4, clip 0.1, non-finite skip, ImageFolder, get_training_lambdas,
#   64 QP levels           all unchanged, asserted by scripts/verify_recipe.py.
#
# And NONE of my additions: no anchor weight, no distillation, no grid seam
# repair, no scaled adapters, no separate lr for new modules. The only thing
# early exit forces is the zero-initialised 1x1 adapters and the per-exit
# auxiliary loss, without which there is no ladder to train at all.
#
# So this run answers a question none of the others can: what does Microsoft's
# own recipe, unaltered, make of a multi-exit decoder?
set -u
cd "$HOME/FLEX-UF"
D=runs/VERBATIM; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=2 setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched \
  --epoch_offset 90 -e 15 \
  --batch_size 8 --grad_accum 2 -n 8 \
  --lambdas 10 2048 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --adapter_kind conv1x1 --seam_repair none \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag VERBATIM \
  > "$D/train.log" 2>&1 < /dev/null &
disown
