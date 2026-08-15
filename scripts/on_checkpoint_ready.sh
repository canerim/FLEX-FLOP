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
"$ROOT/.venv/bin/python" "$ROOT/scripts/warmstart_from_release.py" \
    >> "$ROOT/warmstart.log" 2>&1 \
  && log "warm-start prepared -> runs/warmstart/" \
  || log "WARM-START FAILED — see warmstart.log"
