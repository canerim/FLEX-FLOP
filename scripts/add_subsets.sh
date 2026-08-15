#!/bin/bash
# Bring the dataset up to the recipe's full spec: Open Images subsets 0, 1, 2.
#
# We already train on subset 0 (154,723 usable images after the min-side-512
# filter). training.md names subsets 0, 1 AND 2, so this adds the other two:
#   train_1.tar.gz  34.4 GiB
#   train_2.tar.gz  33.0 GiB
#
# After extracting, description.json is rebuilt over the whole tree and the three
# runs are restarted. Restarting is safe and cheap here: train_flexuf_image.py
# resumes from status_latest.pth.tar, so it costs at most the epoch in flight,
# and doing it now — a couple of epochs in — is far cheaper than discovering at
# epoch 90 that we trained on a third of the data the recipe calls for.
#
# The tarballs are removed after a verified extraction. They are pure duplication
# once unpacked, and this is a shared disk.
set -u

OI=/data10/shareddata/openimages
DEST=$OI/dcvc_train
ROOT=$HOME/FLEX-UF
BASE=https://open-images-dataset.s3.amazonaws.com/tar
LOG=$OI/add_subsets.log

log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for i in 1 2; do
    f="train_${i}.tar.gz"
    log "fetching $f"
    ( cd "$OI" && wget --continue --tries=20 --waitretry=15 --read-timeout=60 \
        -q "$BASE/$f" )
    log "verifying $f"
    if ! gzip -t "$OI/$f" 2>/dev/null; then
        log "FATAL: $f failed integrity check — leaving it and stopping"
        exit 1
    fi
    log "extracting $f"
    tar -xzf "$OI/$f" -C "$DEST" && rm -f "$OI/$f" && log "$f extracted and tarball removed"
done

log "rebuilding description.json over the full tree"
"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_openimages.py" --dest "$DEST" --min-side 512 \
    2>&1 | tail -12 | tee -a "$LOG"

log "restarting so every run picks up the enlarged dataset"
# NOTE: this pkill hits the K=1 baseline too, and launch_experiments.sh only
# knows about e1/e2/e3 — so the baseline has to be relaunched explicitly or it
# dies here and the anchor comparison is silently lost.
pkill -f "train_flexuf_image.py" ; sleep 8
FLEXUF_DATA="$DEST" bash "$ROOT/scripts/launch_experiments.sh" 2>&1 | tee -a "$LOG"

log "relaunching the K=1 baseline (shares GPU 6 with e2, by design)"
CUDA_VISIBLE_DEVICES=6 setsid nohup "$ROOT/.venv/bin/python" \
    "$ROOT/train_flexuf_image.py" \
    --train_dataset "$DEST" --save_dir "$ROOT/runs/baseline_singleexit" \
    --lambdas 10 2048 --batch_size 16 -n 4 -e 6 \
    --num_exits 1 --split_depth 1 --latent_patch 8 --latent_halo 2 \
    --device 0 --tag baseline_singleexit \
    >> "$ROOT/runs/baseline_singleexit/stdout.log" 2>&1 < /dev/null &
sleep 5
log "done — all four runs resumed from their last epoch on subsets 0+1+2"
