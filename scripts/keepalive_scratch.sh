#!/bin/bash
# Restart SCRATCH105 if it dies, because a 23-day run left dead overnight is
# the whole run.
#
# The trainer resumes from status_latest.pth.tar at the last epoch BOUNDARY --
# mid-epoch progress is not recoverable, because resuming there would need the
# sampler position too. So a restart costs up to one epoch, which is about two
# hours; leaving it dead until somebody looks costs however long that is.
#
# Three guards, because a watchdog that restarts a broken run forever is worse
# than no watchdog:
#   * it stops once the run has written its final checkpoint;
#   * it refuses after MAXRESTART restarts, so a crash loop surfaces as a dead
#     run and a log rather than as a card burning for a week;
#   * it waits for the card to be free of our own process before relaunching,
#     so two trainers never share the GPU.
#
# Every restart is appended to runs/SCRATCH105/keepalive.log with the reason,
# so the training log's gaps can be accounted for afterwards.
set -u
cd "$HOME/FLEX-UF"
TAG=${1:-SCRATCH105}
GPU=${2:-7}
EVERY=${3:-300}
MAXRESTART=${MAXRESTART:-20}
D="runs/$TAG"
LOG="$D/keepalive.log"
N=0
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
say "keepalive started for $TAG on GPU$GPU, every ${EVERY}s, max $MAXRESTART restarts"
while true; do
  sleep "$EVERY"
  # Alive? The trainer's own process, not this script and not the watchers.
  if pgrep -f "train_flexuf_image.py.*save_dir[= ]*$HOME/FLEX-UF/$D" >/dev/null 2>&1; then
    continue
  fi
  if grep -q "training complete" "$D/train.log" 2>/dev/null; then
    say "$TAG finished; keepalive exiting"; exit 0
  fi
  if [ "$N" -ge "$MAXRESTART" ]; then
    say "$TAG down and $N restarts already spent; refusing to restart again"
    exit 1
  fi
  WHY=$(tail -40 "$D/train.log" 2>/dev/null \
        | grep -oE "CUDA out of memory|RuntimeError[^\"]{0,60}|Killed|Traceback" \
        | tail -1)
  # Wait for the card, in case the dying process still holds it.
  for _ in $(seq 1 12); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null \
         | while read -r p; do ps -o user= -p "${p// }" 2>/dev/null; done \
         | grep -q "$(id -un)"; then break; fi
    sleep 10
  done
  N=$((N + 1))
  say "$TAG is not running (${WHY:-no error in the log}); restart $N of $MAXRESTART"
  GPU="$GPU" bash scripts/launch_scratch105.sh >> "$LOG" 2>&1
done
