#!/bin/bash
# Hold the autopilot until add_subsets.sh has finished its own restart.
#
# Both relaunch training runs, and while add_subsets is mid-restart the autopilot
# sees dead runs and starts its own copies — which is exactly how the baseline
# and e3 ended up with two main processes each, both writing the same
# status_latest.pth.tar. The flock in autopilot.sh closes that window from now
# on, but add_subsets.sh cannot be edited while it runs (editing a live bash
# script is what caused this in the first place), so for this one cycle the
# autopilot simply waits it out.
#
# The pattern is ANCHORED on purpose. `pgrep -f "bash .*add_subsets.sh"` also
# matches any interactive shell whose command line happens to contain that
# string — including the ones used to inspect this very situation — so an
# unanchored check would wait forever on a process that is really just a `ps`.
set -u

ROOT="$HOME/FLEX-UF"
PATTERN="^bash ${ROOT}/scripts/add_subsets.sh"

while pgrep -f "$PATTERN" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date '+%F %T')] add_subsets finished; starting autopilot" >> "$ROOT/autopilot.log"
sleep 30   # let its final relaunch settle before the watchdog forms an opinion

# Only start if one is not already running.
if ! pgrep -f "^bash ${ROOT}/scripts/autopilot.sh" > /dev/null 2>&1; then
    setsid nohup bash "$ROOT/scripts/autopilot.sh" > /dev/null 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] autopilot started by resume_autopilot" >> "$ROOT/autopilot.log"
fi
