#!/bin/bash
# Fire the whole deliverable the moment a run writes its first epoch checkpoint.
#
# Written because today's pattern was: measure, find a problem, restart, repeat --
# and nothing ever reached the point of producing a number. This removes me from
# the critical path: the checks that matter run themselves, in the order that
# makes them diagnostic, without waiting for anyone to notice a file appeared.
#
# Order is deliberate. anchor_drift FIRST: if the deepest exit has drifted from
# released DCVC-UF again, every saving figure below it is measured against the
# wrong reference and the frontier is not worth computing.
set -u
cd "$HOME/FLEX-UF"
TAG=${1:?usage: on_first_ckpt.sh <run_tag> [gpu]}
GPU=${2:-4}
D="runs/$TAG"
# Either kind of checkpoint will do. The epoch checkpoint is the real one, but an
# epoch is 47451 steps and the mid-epoch snapshot (--ckpt_every) carries the same
# state_dict and config -- enough for every measurement below, none of which
# touches the optimiser. Waiting for the epoch boundary would mean a new
# configuration goes ~10 h unmeasured.
until ls "$D"/ckpt_epo*.pth.tar "$D"/ckpt_step.pth.tar >/dev/null 2>&1; do sleep 120; done
CK=$(ls -t "$D"/ckpt_epo*.pth.tar "$D"/ckpt_step.pth.tar 2>/dev/null | head -1)
echo "=== $TAG -> $(basename "$CK") @ $(date '+%F %T') ==="

echo; echo "--- 1. anchor: hala gercek DCVC-UF mi? ---"
./.venv/bin/python scripts/anchor_drift.py --ckpt "$CK" --device "cuda:$GPU" 2>&1 | tail -8

echo; echo "--- 2. oracle tavani (kusursuz router varsayimi) ---"
for q in 0 32 63; do
  ./.venv/bin/python scripts/oracle_diagnostic.py --ckpt "$CK" --qp $q \
    --crop 512 --batches 8 --batch_size 4 --device "$GPU" 2>&1 \
    | sed -n '/HEADROOM/,/gain > 0/p' | sed "s/^/  qp$q  /" | head -10
done

# 3. The deliverable itself. The oracle above is an upper bound; this is the
# system as it would ship -- the encoder computes the exit map and signals it,
# with the map's own cost inside the bitrate. Run here so the number exists the
# moment the checkpoint does, rather than waiting for someone to notice.
echo; echo "--- 3. SINYALLI sistem (calisan), gercek DCVC-UF referansli ---"
CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/signalled_curve.py \
  --ckpt "$CK" --device cuda:0 \
  --out "$D/signalled_$(basename "$CK" .pth.tar).json" 2>&1 | tail -9
