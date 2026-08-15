#!/bin/bash
# Start the encoder-frozen / decoder-trained run on GPU 7 as soon as Stage A ends.
#
# GPU 7 will not become "free" by the opportunist's strict test for weeks — e3
# still lives there — but Stage A vacates most of its memory and compute in a few
# hours, and that is the capacity this run needs. So it waits on Stage A
# specifically rather than on the card being empty.
#
# What this run is: encoder, hyperprior and entropy model frozen at Microsoft's
# released values, the whole decoder trained (14,971,008 params, 34.9%). The
# latent, the bitstream and the bpp are therefore identical to real DCVC-UF, so
# the rate axis is pinned and every difference measured belongs to the decoder.
# Trained through the deployed patched path, so the adapters see the tile seams
# they will actually face.
set -u
ROOT="$HOME/FLEX-UF"
LOG="$ROOT/autopilot.log"
log () { echo "[$(date '+%F %T')] wdec-gpu7: $*" >> "$LOG"; }

# Wait for Stage A AND for the chain that follows it.
#
# Both want GPU 7 the moment Stage A ends: finish_warmstart_chain runs the router
# sweep and the RD curve there, and this run would train there too — on a card
# that already carries e3. Three jobs on one GPU makes all three slow and the
# deliverable late. So this waits for the chain to produce its result first, then
# takes the card.
log "waiting for Stage A (warmstart_adapters) to finish"
while pgrep -f "train_flexuf_image.py.*--tag warmstart_adapters" > /dev/null; do
    sleep 120
done
log "Stage A finished; waiting for the router sweep + RD curve to clear GPU 7"
sleep 60
while pgrep -f "train_router.py|rd_curve.py" > /dev/null; do
    sleep 120
done
log "chain finished; starting encoder-frozen decoder training on GPU 7"

mkdir -p "$ROOT/runs/wdec_j2_p128"
CUDA_VISIBLE_DEVICES=7 setsid nohup "$ROOT/.venv/bin/python" \
    "$ROOT/train_flexuf_image.py" \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$ROOT/runs/wdec_j2_p128" \
    --pretrain "$ROOT/runs/warmstart/ckpt_warmstart.pth.tar" \
    --freeze_encoder --train_patched \
    --lambdas 10 2048 --batch_size 16 -n 6 -e 6 \
    --num_exits 6 --split_depth 2 --latent_patch 8 --latent_halo 2 \
    --adapter_kind conv1x1 --aux_weight 1.0 --aux_schedule constant \
    --seam_repair full \
    --device 0 --tag wdec_j2_p128 \
    >> "$ROOT/runs/wdec_j2_p128/stdout.log" 2>&1 < /dev/null &
sleep 20
log "launched wdec_j2_p128 (encoder frozen, whole decoder trained, patched path)"
