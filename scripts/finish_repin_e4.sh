#!/bin/bash
# Wait for the epoch-4 measurements, put the headline file in place, then
# regenerate every derived thing and check it.
#
# The headline sweep runs outside the step driver because it was already
# running when the driver was written, so this waits on both and only then
# rebuilds. It refuses to rebuild from a half-written sweep: the sweep's
# output file appears only when signalled_curve finishes, so the test is the
# file's existence, not the process's.
set -uo pipefail
cd "$HOME/FLEX-UF"
TMP=${TMP_DIR:?set TMP_DIR}
LOG=results/finish_repin_e4.log
SWEEP="$TMP/signalled_RECIPE512_ctc53.json"

echo "=== waiting for the epoch-4 measurements, $(date '+%F %T') ===" | tee -a "$LOG"
for _ in $(seq 1 720); do          # up to 12 h
  driver_alive=$(pgrep -f "repin_steps_gpu7.sh" >/dev/null && echo 1 || echo 0)
  sweep_done=$([ -s "$SWEEP" ] && echo 1 || echo 0)
  if [ "$driver_alive" = "0" ] && [ "$sweep_done" = "1" ]; then
    break
  fi
  sleep 60
done

if [ -s "$SWEEP" ]; then
  cp results/signalled_RECIPE512_ctc53.json \
     "$TMP/signalled_RECIPE512_ctc53.epoch0.bak" 2>/dev/null
  mv "$SWEEP" results/signalled_RECIPE512_ctc53.json
  echo "  headline sweep in place" | tee -a "$LOG"
else
  echo "  headline sweep never appeared; rebuilding on what is there" \
    | tee -a "$LOG"
fi

echo "  regenerating everything" | tee -a "$LOG"
bash scripts/regen_all.sh 2>&1 | tee -a "$LOG"
echo "=== done $(date '+%F %T') ===" | tee -a "$LOG"
