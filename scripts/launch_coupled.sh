#!/bin/bash
# COUPLED512 -- the decoder trained WITH canvas coupling, which is the control
# the coupling result currently lacks.
#
# What the paper can say today comes from results/coupling_ablation.json, and it
# is one arm of a two-arm comparison:
#
#   replicate-trained / coupled inference   measured, results/coupling_ablation.json
#   coupling-trained  / coupled inference   this run
#
# The measured arm switches coupling on at inference over a decoder that was
# trained with replicate padding, so it is a distribution shift and its own
# docstring calls the numbers a BOUND rather than a deployment figure. It found
# coupling bit-exact at uniform depth and 47 to 91% off the seam floor, but the
# routed saving collapsed. Two different explanations fit that single arm equally
# well, and they lead to opposite conclusions:
#
#   training mismatch     the decoder never saw a coupled neighbour, so the
#                         collapse is an artefact of evaluating off-distribution
#                         and a coupled-trained decoder would recover the saving.
#   structural dependency the saving comes from tiles being genuinely
#                         independent, coupling reintroduces the dependency, and
#                         no amount of training recovers it.
#
# This run separates them, and it is worth running precisely because the negative
# result is the more useful one. If a decoder trained from the start with
# coupling still loses the routed saving, "this might be a training mismatch"
# becomes "this is a structural dependency", which is a far stronger claim and
# the one a reviewer will ask us to support.
#
# Matched to RECIPE512 (scripts/launch_recipe512.sh) flag for flag, from the same
# warm start, with exactly two deliberate differences:
#
#   --tile_coupling         the variable under test.
#   --seam_repair none      not a free choice. Coupling and GridSeamRepair repair
#                           the SAME artefact by different means, so leaving grid
#                           on would confound the two and cost 0.951% of the
#                           decode to repair a seam that coupling has already
#                           removed. The comparison arm in coupling_ablation.py
#                           is read the same way, so this keeps the two arms
#                           measuring one variable rather than two.
#
# --tile_pad replicate is carried over verbatim even though coupling makes it
# INERT: flexuf/backbone/decoder.py takes the `if cfg.tile_coupling` branch and
# never evaluates the `elif cfg.tile_pad_mode` beside it, because with a real
# neighbour on hand there is nothing left to invent. It stays so that a diff
# against launch_recipe512.sh shows only the two lines that are meant to differ.
#
# No --min_crop, for the same reason RECIPE512 does not pass it: offset 99 lands
# in Microsoft's own 512x512 phase at lr 1e-5, so the recipe already asks for the
# patch size the experiment needs and no override has to be justified.
#
# Compatibility with the TRAINING path was checked rather than assumed. The flag
# already exists in the trainer (--tile_coupling, train_flexuf_image.py) and
# reaches the decode through forward_random_depth, which is the path
# --train_patched takes. The one incompatibility the code asserts is with
# sorted_tiles, and build_cfg never sets it, so it stays at its default of False.
# The anchor and distillation passes go through forward_full and exit_features,
# which decode full-frame and never install the coupler, so they cannot collide
# with it. Confirmed empirically as well: runs/coupled_j2_p256 trained 400 steps
# under this flag on 2026-08-16 with finite gradients and a falling loss.
#
# Matched comparison point: RECIPE512's published arm was measured from
# ckpt_PAPER.pth.tar, which md5s identical to its ckpt_epo0.pth.tar, so the
# comparison is at EPOCH 0. This run writes its own ckpt_epo0.pth.tar with the
# epoch recorded inside it, and that pinned file, not ckpt_eval.pth.tar, is the
# one to measure. Do not point a checkpoint watcher at this directory; the whole
# reason ckpt_eval.pth.tar cannot be cited is that a watcher rewrites it every
# epoch and the epoch behind a published number is then unrecoverable.
#
# When epoch 0 lands, the arm that completes the table is:
#   ./.venv/bin/python scripts/coupling_ablation.py \
#     --ckpt runs/COUPLED512/ckpt_epo0.pth.tar --device cuda:7 \
#     --out results/coupling_trained.json
set -u
cd "$HOME/FLEX-UF"
D=runs/COUPLED512; mkdir -p "$D"
# CUDA_VISIBLE_DEVICES picks the physical card and --device 0 indexes within it.
# Both are needed and they are not redundant: the trainer calls setdefault on
# CUDA_VISIBLE_DEVICES, so an exported value wins and --device alone would be
# silently ignored on a machine where the variable is already set.
CUDA_VISIBLE_DEVICES=7 setsid nohup ./.venv/bin/python -u train_flexuf_image.py \
  --train_dataset /data10/shareddata/openimages/dcvc_train \
  --save_dir "$HOME/FLEX-UF/$D" \
  --pretrain "$HOME/FLEX-UF/runs/warmstart/ckpt_warmstart.pth.tar" \
  --freeze_encoder --train_patched \
  --epoch_offset 99 --new_lr_scale 20 --anchor_weight 10.0 \
  --seam_repair none --tile_coupling --adapter_kind scaled \
  --distill_weight 1.0 --distill_teacher adjacent \
  --lambdas 10 2048 --batch_size 8 -n 8 -e 16 \
  --num_exits 6 --split_depth 2 --latent_patch 16 --latent_halo 2 \
  --aux_weight 1.0 --aux_schedule constant \
  --tile_pad replicate --device 0 --tag COUPLED512 \
  > "$D/train.log" 2>&1 < /dev/null &
disown
echo "COUPLED512 -> GPU7, pid $!, log $D/train.log"
