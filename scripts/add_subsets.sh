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

log "restarting the three runs so they pick up the enlarged dataset"
pkill -f "train_flexuf_image.py" ; sleep 8
FLEXUF_DATA="$DEST" bash "$ROOT/scripts/launch_experiments.sh" 2>&1 | tee -a "$LOG"
log "done — runs resumed from their last epoch on the full subset 0+1+2"
