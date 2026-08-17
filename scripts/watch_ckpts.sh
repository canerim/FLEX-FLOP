#!/bin/bash
# Measure every run at every epoch, for as long as it trains.
#
# Replaces on_first_ckpt.sh, which fired once and then stopped. That was fine
# when a checkpoint was expected within the hour; it is not fine now that the
# measured epoch length is 13-21 h and the numbered checkpoint is written only
# every 5th epoch. Over a 2-3 day budget no run reaches epoch 5, so
# on_first_ckpt.sh would have produced exactly one measurement per run and
# thrown away every later epoch.
#
# scripts/promote_ckpt.py rebuilds a proper evaluation checkpoint from the
# per-epoch resume file plus meta.json -- verified bit-identical to the numbered
# checkpoint (419 tensors, max|diff| = 0.0) -- so every epoch becomes readable.
#
# One evaluation at a time, machine-wide
# --------------------------------------
# Five watchers sharing one card would collide, and an OOM here would look like
# a failed experiment rather than a scheduling mistake. flock serialises them:
# whoever holds the lock evaluates, the rest wait their turn.
set -u
cd "$HOME/FLEX-UF"
TAG=${1:?usage: watch_ckpts.sh <run_tag> [gpu] [poll_seconds]}
GPU=${2:-2}
POLL=${3:-900}
D="runs/$TAG"
LOCK=/tmp/flexuf_eval.lock

while true; do
  # A step snapshot, when the run writes them, is newer than any epoch file and
  # needs no promotion -- it already carries its own config.
  NEW=""; SCOPE=""
  if [ -f "$D/ckpt_step.pth.tar" ] && \
     [ ! -f "$D/.evaluated_step" -o "$D/ckpt_step.pth.tar" -nt "$D/.evaluated_step" ]; then
    NEW="$D/ckpt_step.pth.tar"; MARK="$D/.evaluated_step"; SCOPE="--max_seqs 10"
  elif ./.venv/bin/python scripts/promote_ckpt.py "$D" >/dev/null 2>&1; then
    NEW="$D/ckpt_eval.pth.tar"; MARK="$D/.evaluated_epoch"; SCOPE=""
  fi

  if [ -n "$NEW" ]; then
    (
      flock -w 7200 9 || exit 1
      EP=$(./.venv/bin/python -c "
import torch,sys
c=torch.load('$NEW',map_location='cpu',weights_only=False)
print(f\"epoch {c.get('epoch')} step {c.get('step','-')}\")" 2>/dev/null)
      echo "=== $TAG  $(basename "$NEW")  $EP  @ $(date '+%F %T') ==="

      # anchor FIRST: if the deepest exit has drifted from released DCVC-UF,
      # every saving figure below is measured against the wrong reference.
      echo; echo "--- 1. anchor: still bit-comparable to released DCVC-UF? ---"
      ./.venv/bin/python scripts/anchor_drift.py --ckpt "$NEW" \
        --device "cuda:$GPU" 2>&1 | tail -8

      echo; echo "--- 2. oracle ceiling (assumes a perfect router) ---"
      for q in 0 32 63; do
        ./.venv/bin/python scripts/oracle_diagnostic.py --ckpt "$NEW" --qp $q \
          --crop 512 --batches 8 --batch_size 4 --device "$GPU" 2>&1 \
          | sed -n '/HEADROOM/,/gain > 0/p' | sed "s/^/  qp$q  /" | head -10
      done

      # The deliverable: the system as it would ship, with the exit map's own
      # cost inside the bitrate.
      echo; echo "--- 3. SIGNALLED system, referenced to released DCVC-UF ---"
      CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/signalled_curve.py \
        --ckpt "$NEW" --device cuda:0 $SCOPE \
        --out "results/signalled_${TAG}_$(date +%m%d_%H%M).json" 2>&1 | tail -9
      touch "$MARK"
    ) 9>"$LOCK"
  fi
  sleep "$POLL"
done
