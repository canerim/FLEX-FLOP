#!/bin/bash
# Wait for the browser's download of cvpr2026_image.pth.tar to finish, verify it,
# then warm-start the ladder from Microsoft's released weights.
#
# "Finished" is not "the file exists". A partial torch archive still starts with
# the PK zip magic and only fails at the central directory, which is written
# last — so size stability plus a real zipfile open is the test, and the file
# grew from 35 MB to 161 MB while it was being checked.
#
# Warm-start matters more than convenience here. Training from scratch produced
# a ladder that is not monotonic through the deployed path (exit 3 worse than
# exit 2 while costing 15 points more compute), which is why routing currently
# loses to a uniform depth. FLEX hit the same wall and the released weights are
# what got it past: they make the deepest exit BE the reference codec from step
# zero instead of something that has to be learned.
set -u
ROOT="$HOME/FLEX-UF"
CK="$HOME/DCVC/checkpoints/cvpr2026_image.pth.tar"
LOG="$ROOT/autopilot.log"
log () { echo "[$(date '+%F %T')] checkpoint-watch: $*" >> "$LOG"; }

log "waiting for $CK to stop growing and open as a torch archive"
prev=-1
while true; do
    [ -f "$CK" ] || { sleep 30; continue; }
    cur=$(stat -c%s "$CK")
    if [ "$cur" -eq "$prev" ] && [ "$cur" -gt 1000000 ]; then
        if "$ROOT/.venv/bin/python" -c "
import zipfile,sys
sys.exit(0 if zipfile.is_zipfile('$CK') else 1)" 2>/dev/null; then
            log "download complete and valid: $cur bytes"
            break
        fi
        log "size stable at $cur but archive still invalid — waiting"
    fi
    prev=$cur
    sleep 30
done

log "verifying against stock DMCI and warm-starting the ladder"
if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/warmstart_from_release.py" \
        >> "$ROOT/warmstart.log" 2>&1; then
    log "WARM-START FAILED — see warmstart.log"
    exit 1
fi
log "warm-start prepared -> runs/warmstart/ckpt_warmstart.pth.tar"

# Stage A: adapters only, backbone frozen.
#
# ~739k trainable parameters out of 42.9M. This is FLEX's recipe and it is why
# the warm-start path is worth waiting for: hours instead of the ~37 days a
# full-recipe from-scratch run of this dataset size needs, and it starts from a
# ladder that is ordered by construction rather than one that has to learn to be.
#
# GPU 7 carries e3, the fastest of the three runs, so it has the most headroom
# to share.
log "stage A: training adapters on the frozen release backbone (GPU 7)"
CUDA_VISIBLE_DEVICES=7 setsid nohup "$ROOT/.venv/bin/python" \
    "$ROOT/train_flexuf_image.py" \
    --train_dataset /data10/shareddata/openimages/dcvc_train \
    --save_dir "$ROOT/runs/warmstart" \
    --pretrain "$ROOT/runs/warmstart/ckpt_warmstart.pth.tar" \
    --freeze_backbone \
    --lambdas 10 2048 --batch_size 16 -n 6 -e 3 \
    --num_exits 6 --split_depth 2 --latent_patch 8 --latent_halo 2 \
    --adapter_kind conv1x1 --aux_weight 1.0 --aux_schedule constant \
    --device 0 --tag warmstart_adapters \
    >> "$ROOT/runs/warmstart/stdout.log" 2>&1 < /dev/null &
log "stage A launched; frontier and RD curve follow once it checkpoints"
