#!/bin/bash
# Launch the three FLEX-UF experiments, one per idle GPU.
#
# GPU discipline
# --------------
# Only 4, 6 and 7 are used. GPUs 0/1/2/5 are running other people's jobs, and
# GPU 3 holds 44 GB belonging to another user (an_li) even though its
# utilisation reads 0 — an idle-looking GPU is not a free GPU. Each run is
# pinned with CUDA_VISIBLE_DEVICES so it physically cannot touch another card.
#
# Why three independent single-GPU runs and not one DDP run across three
# ---------------------------------------------------------------------
# One three-times-faster run of a single configuration answers one question.
# Three runs answer three, and the axes below are exactly the ones nobody has
# measured on UF. Each experiment differs from E1 in exactly ONE variable, so
# every comparison is a clean ablation rather than a confound.
#
#   E1  GPU 4   j=2  tile=128px  halo=2lat  adapter=1x1   the headline config
#   E2  GPU 6   j=4  tile=128px  halo=2lat  adapter=1x1   isolates split depth
#   E3  GPU 7   j=2  tile= 64px  halo=2lat  adapter=1x1   isolates tile size
#
# What each one tests
# -------------------
# E1: the aggressive split. Groups 0-1 shared, 8 of 12 blocks routable, which the
#     measured cost table says is where the saving lives (58.7% at the shallowest
#     exit vs 28.9% at j=4). Hypothesis: the 1x1 adapters can absorb the seam of
#     8 per-tile blocks well enough to keep the PSNR loss small.
# E2: the conservative split. Half the routable range, so less possible saving,
#     but only 4 blocks ever run per-tile and the boundary damage should be much
#     lower. Hypothesis: better dB, worse saving — this pins down the shape of
#     the frontier in j.
# E3: the tile-size axis. The Boundary Law says seam penalty scales as 1/side, so
#     64px tiles should cost roughly twice the dB of 128px ones, in exchange for
#     4x the routing granularity. Hypothesis: 128 wins on the frontier; this is
#     the experiment that proves it rather than assuming it.
#
# All three share the Microsoft recipe verbatim: 105 epochs, AdamW, lr schedule
# 2e-4 -> 1e-6, 256x256 crops until epoch 90 then 512x512, batch 16, grad clip
# 0.1, lambdas log-spaced 10 -> 2048 across the 64 QPs.

set -u

ROOT="$HOME/FLEX-UF"
PY="$ROOT/.venv/bin/python"
DATA="${FLEXUF_DATA:-/data10/shareddata/openimages/dcvc_train}"
RUNS="$ROOT/runs"
mkdir -p "$RUNS"

if [ ! -f "$DATA/description.json" ]; then
    echo "ERROR: $DATA/description.json missing."
    echo "Run scripts/prepare_openimages.py first, or set FLEXUF_DATA."
    exit 1
fi

launch () {
    local tag=$1 gpu=$2 j=$3 patch=$4 halo=$5 adapter=$6
    local dir="$RUNS/$tag"
    mkdir -p "$dir"
    echo "launching $tag on GPU $gpu  (j=$j patch=${patch}lat halo=${halo}lat adapter=$adapter)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$ROOT/train_flexuf_image.py" \
        --train_dataset "$DATA" \
        --save_dir "$dir" \
        --lambdas 10 2048 \
        --batch_size 16 \
        -n 8 \
        -e 105 \
        --num_exits 6 \
        --split_depth "$j" \
        --latent_patch "$patch" \
        --latent_halo "$halo" \
        --adapter_kind "$adapter" \
        --aux_weight 1.0 \
        --aux_schedule warmup \
        --device 0 \
        --tag "$tag" \
        > "$dir/stdout.log" 2>&1 &
    echo "  pid $! -> $dir/stdout.log"
    sleep 3
}

#      tag              gpu  j  patch halo adapter
launch e1_j2_p128        4    2    8    2   conv1x1
launch e2_j4_p128        6    4    8    2   conv1x1
launch e3_j2_p64         7    2    4    2   conv1x1

echo
echo "all three launched. monitor with:  bash $ROOT/scripts/status.sh"
