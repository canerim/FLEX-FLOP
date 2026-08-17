#!/bin/bash
# Start / stop / list the per-run checkpoint watchers.
#
# Exists because three separate attempts to manage them with pgrep and
# ps|grep went wrong the same way: the pattern matched the shell that was
# doing the matching, whose command line contains the script name. Twice that
# killed the shell mid-command; once it looked like two evaluations racing for
# one output file.
#
# A watcher is /bin/bash ./scripts/watch_ckpts.sh TAG GPU POLL, so argv[1] is
# the script path. The shell running this has argv[1] == "-c". Comparing the
# whole prefix separates them with no ambiguity.
set -u
cd "$HOME/FLEX-UF"
RUNS="BEST BEST128 CONTROL FINE12 RECIPE512 VERBATIM"
GPU=${GPU:-2}
POLL=${POLL:-900}
PREFIX="/bin/bash ./scripts/watch_ckpts.sh "

# A tag can legitimately appear twice: watch_ckpts.sh runs its evaluation
# inside `( flock 9 ... ) 9>LOCK`, and that subshell inherits the parent's
# command line. Two lines for one tag means that watcher is mid-evaluation, not
# that it was started twice. `stop` lists both, which is what we want -- killing
# only the parent would orphan a running evaluation holding the lock.
pids () {
  for pid in /proc/[0-9]*; do
    [ -r "$pid/cmdline" ] || continue
    cl=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null) || continue
    case "$cl" in "$PREFIX"*) echo "${pid#/proc/} $cl";; esac
  done
}

case "${1:-list}" in
  list) pids ;;
  stop) pids | while read -r p _; do kill "$p" 2>/dev/null && echo "stopped $p"; done ;;
  start)
    mkdir -p runs/_waiters
    for t in $RUNS; do
      setsid nohup /bin/bash ./scripts/watch_ckpts.sh "$t" "$GPU" "$POLL" \
        >> "runs/_waiters/$t.log" 2>&1 < /dev/null &
      disown
    done
    sleep 3; pids ;;
  restart)
    # A watcher killed mid-evaluation leaves its `( flock 9 ... )` subshell
    # running, reparented to init. It finishes its work correctly, but it still
    # holds the lock, so anything queued behind it waits -- and the restarted
    # watcher may queue a second evaluation of the same checkpoint behind it.
    # Warn rather than kill: interrupting a measurement halfway is worse than
    # letting it finish.
    if fuser /tmp/flexuf_eval.lock >/dev/null 2>&1; then
      echo "note: an evaluation holds the lock and will be orphaned by this"
      echo "      restart. It completes on its own; check with"
      echo "      'fuser -v /tmp/flexuf_eval.lock' before queueing more work."
    fi
    "$0" stop; sleep 2; "$0" start ;;
  *) echo "usage: watchers.sh [list|start|stop|restart]"; exit 1 ;;
esac
