#!/bin/bash
# Autonomous chain: wait for the tarball -> extract -> build description.json
# -> launch the three experiments. Written so no human step sits between the
# download finishing and training starting.
#
# Each stage is idempotent and leaves a marker, so re-running after any
# interruption picks up where it stopped rather than redoing 46 GB of work.

set -u

ROOT="$HOME/FLEX-UF"
PY="$ROOT/.venv/bin/python"
OI=/data10/shareddata/openimages
TAR="$OI/train_0.tar.gz"
DEST="$OI/dcvc_train"
EXPECTED=49310925536

log () { echo "[$(date '+%F %T')] $*"; }

# ---- 1. wait for the download ------------------------------------------
log "stage 1: waiting for $TAR to reach $EXPECTED bytes"
while true; do
    have=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
    if [ "$have" -ge "$EXPECTED" ]; then
        log "download complete: $have bytes"
        break
    fi
    if ! pgrep -f "train_0.tar.gz" > /dev/null; then
        log "wget is not running and file is short ($have/$EXPECTED) — restarting it"
        (cd "$OI" && nohup wget --continue --tries=20 --waitretry=15 --read-timeout=60 \
            https://open-images-dataset.s3.amazonaws.com/tar/train_0.tar.gz \
            >> resume.log 2>&1 &)
    fi
    sleep 60
done

# ---- 2. verify the archive ---------------------------------------------
# A truncated 46 GB tarball that only fails at image 400,000 would waste hours,
# so the integrity of the gzip stream is checked once, up front.
log "stage 2: verifying gzip integrity"
if ! gzip -t "$TAR" 2>/dev/null; then
    log "FATAL: $TAR failed gzip integrity check"
    exit 1
fi
log "archive OK"

# ---- 3. extract ---------------------------------------------------------
log "stage 3: extracting to $DEST"
mkdir -p "$DEST"
if [ ! -f "$DEST/.extracted" ]; then
    tar -xzf "$TAR" -C "$DEST" && echo "$(date)" > "$DEST/.extracted"
    log "extraction done"
else
    log "already extracted, skipping"
fi
df -h /data10 | tail -1

# ---- 4. build description.json -----------------------------------------
log "stage 4: building description.json (min short side 512, matching the recipe's final crop)"
if [ ! -f "$DEST/description.json" ]; then
    "$PY" "$ROOT/scripts/prepare_openimages.py" --dest "$DEST" --min-side 512
else
    log "description.json exists, skipping"
fi
"$PY" -c "
import json; d=json.load(open('$DEST/description.json')); print(f'  usable images: {len(d):,}')"

# ---- 5. launch ----------------------------------------------------------
log "stage 5: launching the three experiments on GPUs 4, 6, 7"
FLEXUF_DATA="$DEST" bash "$ROOT/scripts/launch_experiments.sh"

log "pipeline complete — training is running"
