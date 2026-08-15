#!/bin/bash
# Stage A -> routers -> the deliverable RD curve, without a human in between.
#
# Waits for the frozen-backbone adapter run to write a checkpoint, then trains
# one router per beta against it and produces the dense-vs-routed RD curve that
# is the project's output.
#
# Why the routers are trained AFTER the adapters and against a frozen decoder:
# FLEX found that a router trained against a decoder still in motion goes stale.
# It learns some exit is bad, stops sending tiles there, that exit then receives
# no gradient and stays bad, and the ladder collapses to two live exits.
# Freezing first makes the router's problem stationary.
set -u
ROOT="$HOME/FLEX-UF"
PY="$ROOT/.venv/bin/python"
DATA=/data10/shareddata/openimages/dcvc_train
WS="$ROOT/runs/warmstart"
LOG="$ROOT/autopilot.log"
log () { echo "[$(date '+%F %T')] warmstart-chain: $*" >> "$LOG"; }

log "waiting for Stage A to checkpoint"
while [ ! -f "$WS/status_latest.pth.tar" ]; do sleep 120; done
sleep 60   # let the write settle
log "Stage A checkpoint present"

CK="$WS/adapters_snapshot.pth.tar"
cp "$WS/status_latest.pth.tar" "$CK"

# Is the ladder worth routing on at all? Same gate the autopilot uses.
if ! "$PY" "$ROOT/scripts/ladder_health.py" --ckpt "$CK" --device 7 --frames 48 \
        > "$WS/ladder_health.log" 2>&1; then
    log "ladder still not monotonic — recording and continuing anyway for the record"
fi

log "training routers (beta sweep) against the frozen adapters"
for beta in 0 10 30 100 300; do
    d="$WS/router_beta_$beta"; mkdir -p "$d"
    [ -s "$d/router.pth.tar" ] && { log "beta=$beta already done"; continue; }
    CUDA_VISIBLE_DEVICES=7 "$PY" "$ROOT/train_router.py" \
        --ckpt "$CK" --train_dataset "$DATA" --save_dir "$d" \
        --steps 1500 --batch_size 8 --crop 512 -n 4 --beta "$beta" --device 0 \
        > "$d/train.log" 2>&1 || { log "beta=$beta FAILED"; continue; }
    log "router beta=$beta trained"
done

# The deliverable: dense vs routed on one set of axes, swept over QP.
best="$WS/router_beta_30/router.pth.tar"
[ -s "$best" ] || best=$(ls "$WS"/router_beta_*/router.pth.tar 2>/dev/null | head -1)
if [ -n "${best:-}" ] && [ -s "$best" ]; then
    log "producing the RD curve with $best"
    CUDA_VISIBLE_DEVICES=7 "$PY" "$ROOT/rd_curve.py" \
        --ckpt "$CK" --router "$best" \
        --qps 0 8 16 24 32 40 48 56 63 --frames 48 --crop 512 --batch_size 2 \
        --device 0 --out "$ROOT/results/rd_warmstart.json" \
        > "$ROOT/results/rd_warmstart.log" 2>&1 \
      && log "RD CURVE READY -> results/rd_warmstart.json" \
      || log "RD curve FAILED — see results/rd_warmstart.log"
else
    log "no router available; RD curve skipped"
fi
