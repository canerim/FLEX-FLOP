#!/bin/bash
# Hold the autopilot until add_subsets.sh has finished its own restart.
#
# The two both relaunch training runs, and while add_subsets is mid-restart the
# autopilot sees dead runs and starts its own copies. The flock in autopilot.sh
# closes that window going forward, but add_subsets.sh cannot be edited while it
# is running — editing a live bash script is what caused this mess in the first
# place — so for this one cycle the autopilot simply waits it out.
set -u
ROOT="$HOME/FLEX-UF"
while pgrep -f "bash .*add_subsets.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] add_subsets finished; starting autopilot" >> "$ROOT/autopilot.log"
sleep 30   # let its final relaunch settle
setsid nohup bash "$ROOT/scripts/autopilot.sh" > /dev/null 2>&1 < /dev/null &
