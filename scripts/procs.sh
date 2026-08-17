#!/bin/bash
# Count/list real processes running a script, without matching the shell that
# launched them.
#
# `pgrep -f foo.py` and `ps | grep foo.py` both match any shell whose command
# line merely CONTAINS the string -- including the heredoc that wrote the
# launcher. That produced three false readings today, one of which looked like
# two evaluations racing for the same output file. Matching on the executable
# being a python interpreter removes the whole class.
set -u
pat=${1:?usage: procs.sh <script-name> [-c]}
out=$(ps -eo pid,etime,args --no-headers \
      | awk -v p="$pat" '$3 ~ /python/ && index($0, p) > 0 {print}')
if [ "${2:-}" = "-c" ]; then
  [ -z "$out" ] && echo 0 || printf '%s\n' "$out" | wc -l
else
  printf '%s\n' "$out"
fi
