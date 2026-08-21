#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Train the DMCI image model (DCVC-UF-Intra) as a variable-rate model, which is
# the official recipe: all 64 QPs at once (no --train_qp, so the dataset draws a
# random QP per sample and interpolates its lambda between the LAMBDAS
# endpoints), batch 16, 105 epochs. train_image.py's training_strategy switches
# the patch size from 256x256 to 512x512 at epoch 90 on its own.
#
# Variables marked "overridable" read from the environment, so a one-off run
# needs no edit here:
#   EPOCHS=2 COMPILE_MODE=default SAVE_DIR=checkpoints/_probe bash train_image_allqp_OpenImage.sh

set -e

export PYTHONUNBUFFERED=1   # unbuffered stdout, so `tail -f` on the log is live

# ========== User-configurable variables ==========
# Open Images: 384,795 jpg in three train_0/1/2 subfolders. ImageFolder reads
# <root>/description.json and os.walk collects the subfolders recursively, so
# nothing has to be flattened. docker/prepare_datasets.sh generates that file on
# container start.
TRAIN_DATASET=../datasets/Open_Image/train
# train_image.py resumes from whatever is in SAVE_DIR (load_existing_weights),
# so a different dataset needs a different SAVE_DIR -- otherwise it keeps
# training the previous dataset's weights.
SAVE_DIR=${SAVE_DIR:-checkpoints/image_model_allqp_openimage}   # overridable
EPOCHS=${EPOCHS:-105}                                           # overridable
LAMBDAS="10 2048"           # lambda endpoints, log-interpolated over the 64 QPs
                            # (QP0 -> 10, QP63 -> 2048)

# ---- training distortion (loss) ----
MSE_YUV_MEAN=geometric
MSE_YUV_WEIGHTS="10 1 1"
MSE_RGB_WEIGHT=0.2

NUM_WORKERS=16

# Global batch, split across ranks under DDP (get_dataloader asserts it divides
# by the world size, so 3 GPUs needs 15 or 18 rather than 16).
BATCH_SIZE=${BATCH_SIZE:-16}   # overridable

SEED=42                     # model init, sampling order, cropping, QP draw, noise.
                            # Add --deterministic for bit-exact reruns, ~15% slower.

# --train_qp is deliberately NOT set: passing it collapses this into
# single-rate training.

# ---- checkpoint retention ----
KEEP_STATUS_NUM=2           # rolling resume points kept (weights + optimizer state)
ARCHIVE_INTERVAL=10         # also archive a weights-only snapshot every N epochs
                            # to <SAVE_DIR>/archives/; 0 disables it

# ---- validation ----
VAL_DATASET=../datasets/jpegai_validation_set
# JSON like the training set (ImageValFolder accepts both formats).
# docker/prepare_datasets.sh converts the same-named .txt, order preserved.
VAL_LIST=${VAL_DATASET}/jpegai_validation_set_10.json
VAL_QPS="15,32,46,60,63"    # five rate points, aligned with the reference curve
VAL_INTERVAL=5
VAL_MAX_SIDE=1280           # center-crop val images to this max side; 0 disables

# ---- reference RD curve: the official pretrained model, drawn on the val plots ----
REF_CKPT=checkpoints/cvpr2026_image.pth.tar
REF_QPS="15,32,46,60,63"    # same as VAL_QPS, so the two curves are point-comparable
# Shared cache: the official model is evaluated once and every run reuses it.
# The cache key holds the *absolute* paths (plus size and mtime) of REF_CKPT,
# VAL_DATASET and VAL_LIST -- see _file_identity in train_image.py -- so one
# cache file is only valid for one execution environment. This one is keyed to
# the container paths (/workspace/DCVC-UF/...) and verified to hit. Running this
# script on the host instead misses, re-evaluates once, and rewrites the cache
# with host paths.
REF_CACHE=checkpoints/ref_rd_cache/ref_rd_points_jpegai10.json

# ---- speed ----
# See the measured ms/step table in README.md ("Training speed"). full +
# max-autotune is the fastest and what production runs use; its first steps pay
# several minutes of compilation, so short probe runs are better off with
# COMPILE_MODE=default.
COMPILE=${COMPILE:-full}                   # overridable: off / parts / full
COMPILE_MODE=${COMPILE_MODE:-max-autotune} # overridable: default / reduce-overhead / max-autotune

# ---- wandb monitoring (WANDB_MODE=disabled, or drop the flags, to turn it off) ----
WANDB_PROJECT=DCVC-UF-Intra
WANDB_STEP_INTERVAL=500     # log a train-step/* snapshot every N optimizer steps.
                            # Keeping it sparse avoids the wandb filestream being
                            # throttled (409) and dropping the run.
EXP_NAME=${EXP_NAME:-image_openimage_allqp}   # overridable: wandb run name
# ==================================================

python train_image.py \
    --train_dataset "${TRAIN_DATASET}" \
    --save_dir "${SAVE_DIR}" \
    --lambdas ${LAMBDAS} \
    --mse_yuv_mean ${MSE_YUV_MEAN} \
    --mse_yuv_weights ${MSE_YUV_WEIGHTS} \
    --mse_rgb_weight ${MSE_RGB_WEIGHT} \
    --batch_size ${BATCH_SIZE} \
    -n ${NUM_WORKERS} \
    -e ${EPOCHS} \
    --val_dataset "${VAL_DATASET}" \
    --val_list "${VAL_LIST}" \
    --val_qps "${VAL_QPS}" \
    --val_interval ${VAL_INTERVAL} \
    --val_max_side ${VAL_MAX_SIDE} \
    --ref_ckpt "${REF_CKPT}" \
    --ref_qps "${REF_QPS}" \
    --ref_cache "${REF_CACHE}" \
    --keep_status_num ${KEEP_STATUS_NUM} \
    --archive_interval ${ARCHIVE_INTERVAL} \
    --amp \
    --compile ${COMPILE} \
    --compile_mode ${COMPILE_MODE} \
    --fused_adam \
    --seed ${SEED} \
    --wandb_step_interval ${WANDB_STEP_INTERVAL} \
    --wandb_project "${WANDB_PROJECT}" \
    --exp_name "${EXP_NAME}"
