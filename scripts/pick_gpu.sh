#!/usr/bin/env bash
# Print the index of a GPU this account has to itself, most free memory first.
#
# Why this exists: watch_ckpts.sh defaulted to GPU2 because GPU2 used to hold
# one of our own runs. It stopped being ours and the default did not notice, so
# a nightly eval landed on a card another user had been on for eight hours.
# This is a shared server and that is not ours to do. Ask the driver who is on
# each card rather than trusting a number written weeks ago.
#
# Falls back to our own busiest-but-still-ours card. Never prints a card
# somebody else is using, and prints nothing if there is no such card, so a
# caller that forgets to check gets an empty CUDA_VISIBLE_DEVICES and fails
# loudly instead of quietly borrowing a stranger's GPU.
set -u
ME=$(id -un)
best=""; bestfree=-1
while IFS=, read -r idx uuid used total; do
  idx=${idx// /}; uuid=${uuid// /}
  used=$(echo "$used" | tr -dc '0-9'); total=$(echo "$total" | tr -dc '0-9')
  mine_only=1
  for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
               | grep -F "$uuid" | cut -d, -f1); do
    u=$(ps -o user= -p "${pid// /}" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && mine_only=0
  done
  [ "$mine_only" -eq 1 ] || continue
  free=$(( total - used ))
  if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader)
[ -n "$best" ] && echo "$best"
