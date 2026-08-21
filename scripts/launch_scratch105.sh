#!/bin/bash
# SCRATCH105 -- FLEX-UF trained from random initialisation on Microsoft's own
# image recipe, end to end, with nothing warm-started and nothing frozen.
#
# Every run before this one fine-tunes the released DCVC-UF intra decoder with
# the encoder frozen, which is the right experiment for the paper's claim
# ("x% cheaper at y dB *versus the release*") and the wrong one for the
# question a reviewer asks next: is the ladder an artefact of starting from a
# decoder that was trained without it? This run answers that, and it can only
# answer it by paying for the whole schedule.
#
# What is Microsoft's, verbatim, from ~/DCVC/train_image.py and training.md:
#   --lambdas 10 2048        their image lambdas
#   --batch_size 16          their batch. start_train() runs single-process on
#                            one visible GPU, so 16 is what their script does
#                            here; it is not a per-rank number we have halved.
#   -e 105                   their epoch count
#   --epoch_offset 0         the schedule read from its beginning: 45 epochs at
#                            2e-4/256px, 25 at 5e-5, 20 at 1e-5, then 5 at
#                            2e-4/512px, 4 at 5e-5, 4 at 1e-5, 2 at 1e-6
#   AdamW(1e-4) overridden per epoch, clip_grad_norm_ 0.1, non-finite batches
#   dropped -- all already in train_flexuf_image.py and checked by
#   scripts/verify_recipe.py on every launch.
#
# What is ours, and why each one is here:
#   --num_exits 6 --split_depth 2      the ladder the paper measures
#   --latent_patch 16 --latent_halo 2  256px tiles, the size that measured best
#   --adapter_kind scaled              depth-scaled adapters
#   --seam_repair grid                 the repair that survived Section 4
#   --distill_weight 1.0 --distill_teacher adjacent
#                                      ladder distillation between neighbours
#   --aux_weight 1.0 --aux_schedule constant
#   --aux_schedule warmup              alpha ramped from 0 over 10 epochs
#
# Why NOT --train_patched, which every warm-started run uses
# ----------------------------------------------------------
# It was used here first, and the ladder inverted. Over three thousand steps
# the usable exits went from a 0.96 dB span to 0.28 and stopped being monotone:
# [25.23, 25.18, 24.87, 25.01] for k = 2..5, deeper decoding worse than
# shallower. That is not a surprise in hindsight. Random-depth training asks
# only that the mixed-depth frame be good; nothing in it asks exit k+1 to beat
# exit k, and the equilibrium is every exit equally good, which is a ladder
# with no rungs.
#
# A warm start does not have this problem because the ordering is inherited --
# the released trunk is already ordered by depth and the objective only has to
# avoid destroying it. From random initialisation the ordering has to be
# created, and what creates it is the paper's own objective: Eq. (6)-(7), a
# reconstruction loss at every exit, so exit k is optimised to be the best
# reconstruction available at depth k and exit k+1 -- which has strictly more
# computation -- cannot do worse at its own optimum.
#
# The cost is real: one trunk and K heads instead of one decode, 0.600 s/step
# against 0.330. flexuf/losses.py already recorded the mitigation for exactly
# this failure ("FLEX-FLOP's from-scratch attempt collapsed with every exit
# stuck near 26 dB"), which is the warmup schedule, and this run is the
# experiment that note said was needed.
#
# What this gives up is the mixed-depth condition, and for the first 90 epochs
# it gives up nothing at all: Microsoft's schedule trains at 256x256 and a
# 256px tile in a 256px crop is one tile, so patched and full-frame training
# are the same computation. It becomes a real difference at epoch 90, where
# the recipe moves to 512x512 and a crop holds four tiles. That is eight days
# away and is recorded in PENDING_DECISIONS rather than decided now.
#   --tile_pad replicate
#
# What is NOT here, and why:
#   --pretrain          nothing to warm start from; that is the point
#   --freeze_encoder    the encoder is trained too. Freezing it is a device for
#                       keeping a warm start bit-exact with the release, and
#                       there is no release in this run to stay exact with.
#   --anchor_weight     the anchor holds the deepest exit on the released
#                       decoder's output. From random init that term would be
#                       asking the model to reproduce a checkpoint it has no
#                       relation to, from epoch 0, against the RD objective.
#                       It is a warm-start device and it is off.
#   --min_crop          Microsoft's schedule is 256px for 90 epochs and 512 for
#                       the last 15. A 256px tile in a 256px crop is one tile,
#                       so tile borders are only trained in those last 15 epochs.
#                       That is what the recipe says and the recipe is what this
#                       run exists to follow.
#
# The one departure: --bf16. Their trainer is fp32. bfloat16 keeps fp32's
# exponent range, so the 0.1 gradient clip and the non-finite guard behave as
# they do in fp32, which fp16 and a loss scaler would not; and the losses are
# computed in fp32 outside the autocast region, so the RD trade-off itself is
# never evaluated at reduced precision.
set -u
cd "$HOME/FLEX-UF"
./.venv/bin/python scripts/verify_recipe.py >/dev/null || { echo "recipe check failed"; exit 1; }
D=runs/SCRATCH105; mkdir -p "$D"
CUDA_VISIBLE_DEVICES=${GPU:-7} setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --bf16 \
  --epoch_offset 0 --anchor_weight 0.0 \
  --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent \
  --lambdas 10 2048 --batch_size 16 -n 8 -e 105 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule warmup --aux_warmup_epochs 10 \
  --ckpt_every 2500 --log_every 200 \
  --tile_pad replicate --device 0 --tag SCRATCH105 \
  >> "$D/train.log" 2>&1 < /dev/null &
disown
echo "SCRATCH105 launched on GPU${GPU:-7}"
