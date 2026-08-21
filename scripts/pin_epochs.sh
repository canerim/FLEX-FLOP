#!/bin/bash
# Keep every epoch boundary a run produces, instead of the last one only.
#
# ckpt_eval.pth.tar is rewritten at each boundary, so the weights behind an
# epoch's measurement live until the next epoch closes and then do not. That is
# how the RECIPE512 epoch-2 measurement became unreproducible, and epoch 4 --
# the best this run has produced, 27.6% against the pinned checkpoint's 21.5 --
# came within fifteen hours of going the same way.
#
# A checkpoint is 182 MB and there is over a terabyte free. Keeping them is the
# cheap side of this trade by three orders of magnitude.
#
# Copies only when the epoch inside the file is one that has no pin yet, so it
# is safe to run on a timer and does no work in between.
set -u
cd "$HOME/FLEX-UF"
for D in runs/*/; do
  # ckpt_eval is what the watcher promotes; a run with no watcher -- SCRATCH105
  # has none, because the CTC chain is meaningless for it -- has only
  # status_latest.pth.tar at its epoch boundaries, and that is overwritten just
  # as readily. Pin from whichever exists.
  CK=""
  for CAND in "$D/ckpt_eval.pth.tar" "$D/status_latest.pth.tar"; do
    if [ -f "$CAND" ]; then CK="$CAND"; break; fi
  done
  [ -n "$CK" ] || continue
  EP=$(./.venv/bin/python - "$CK" <<'PYEOF' 2>/dev/null
import sys, torch
try:
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    e = d.get("epoch")
    print(int(e) if e is not None else "", end="")
except Exception:
    pass
PYEOF
)
  [ -n "$EP" ] || continue
  PIN="$D/ckpt_PIN_e$EP.pth.tar"
  if [ ! -f "$PIN" ]; then
    if ./.venv/bin/python scripts/pin_one.py "$CK" "$PIN.tmp"; then
      mv "$PIN.tmp" "$PIN"
      echo "$(date '+%F %T') pinned $(basename "$D") epoch $EP -> $(basename "$PIN")"
    else
      rm -f "$PIN.tmp"
    fi
  fi
done
