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
#   --train_patched                    mixed-depth training, the condition that
#                                      actually occurs at inference
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
  --bf16 --train_patched \
  --epoch_offset 0 --anchor_weight 0.0 \
  --seam_repair grid --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent \
  --lambdas 10 2048 --batch_size 16 -n 8 -e 105 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --ckpt_every 2500 --log_every 200 \
  --tile_pad replicate --device 0 --tag SCRATCH105 \
  >> "$D/train.log" 2>&1 < /dev/null &
disown
echo "SCRATCH105 launched on GPU${GPU:-7}"
