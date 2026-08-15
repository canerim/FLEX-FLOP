#!/bin/bash
# Autopilot: keep the project moving without a human at the keyboard.
#
# Three jobs, on a loop:
#   1. WATCHDOG  — if a training run has died, restart it. `train_flexuf_image.py`
#                  resumes from status_latest.pth.tar, so a restart costs at most
#                  the current epoch, not the whole run.
#   2. HEALTH    — append a status line to autopilot.log every cycle, so there is
#                  a record of what happened while nobody was watching.
#   3. FRONTIER  — when a new ckpt_epo*.pth.tar appears, measure the frontier on
#                  it once. This is the number the project exists to produce, and
#                  waiting 7 days for the final checkpoint to find out whether the
#                  idea works would be a waste of 7 days.
#
# The frontier sweep shares a GPU with the training run that produced the
# checkpoint. It costs ~15 minutes against a ~5.7 hour checkpoint interval, so
# the contention is real but small, and an early answer is worth far more.
set -u

ROOT="$HOME/FLEX-UF"
RUNS="$ROOT/runs"
LOG="$ROOT/autopilot.log"
DATA=/data10/shareddata/openimages/dcvc_train
declare -A GPU=( [e1_j2_p128]=4 [e2_j4_p128]=6 [e3_j2_p64]=7 )
declare -A ARGS=(
  [e1_j2_p128]="--split_depth 2 --latent_patch 8"
  [e2_j4_p128]="--split_depth 4 --latent_patch 8"
  [e3_j2_p64]="--split_depth 2 --latent_patch 4"
)

log () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "autopilot started"

while true; do
    for tag in "${!GPU[@]}"; do
        dir="$RUNS/$tag"

        # ---- 1. watchdog ------------------------------------------------
        if ! pgrep -f "train_flexuf_image.py.*--tag $tag" > /dev/null; then
            if [ -f "$dir/ckpt.pth.tar" ]; then
                log "$tag: finished (final ckpt present)"
            else
                log "$tag: DIED — restarting on GPU ${GPU[$tag]} (resumes from last epoch)"
                CUDA_VISIBLE_DEVICES="${GPU[$tag]}" nohup "$ROOT/.venv/bin/python" \
                    "$ROOT/train_flexuf_image.py" \
                    --train_dataset "$DATA" --save_dir "$dir" --lambdas 10 2048 \
                    --batch_size 16 -n 8 -e 105 --num_exits 6 ${ARGS[$tag]} \
                    --latent_halo 2 --adapter_kind conv1x1 \
                    --aux_weight 1.0 --aux_schedule warmup --device 0 --tag "$tag" \
                    >> "$dir/stdout.log" 2>&1 &
                sleep 10
            fi
        fi

        # ---- 2. health --------------------------------------------------
        if [ -f "$dir/train_log.jsonl" ]; then
            python3 - "$dir/train_log.jsonl" "$tag" >> "$LOG" 2>/dev/null <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if rows:
    r=rows[-1]
    sp=r['spread_dB']; deep=r['psnr_per_exit'][-1]
    # Same two-axis judgement as status.sh: a small spread is only bad if the
    # deepest exit is also bad. Shallow exits catching a GOOD deepest exit is
    # the result we are after, not a failure.
    if deep < 20:   flag = "ANCHOR-WEAK"
    elif sp < 0.05: flag = "EXITS-INDISTINGUISHABLE"
    elif sp < 2.0:  flag = "GOOD"
    else:           flag = "OK"
    print(f"  {sys.argv[2]:<12} ep{r['epoch']:<3} step{r['step']:<6} "
          f"loss {r['loss']:8.4f} bpp {r['bpp']:.4f} "
          f"deepest {deep:6.2f}dB spread {sp:+.2f} {flag}")
PY
        fi

        # ---- 3. frontier on any unmeasured checkpoint --------------------
        # The right beta depends on how FLAT the ladder is, and that changes as
        # training proceeds -- so a short fixed list measures nothing useful.
        # Measured: at ckpt_epo0 the ladder spans 10.4 dB, so moving a tile one
        # exit shallower costs 50*1.56 = 78 in the image term against beta*0.14
        # in the complexity term; nothing moves below beta ~ 550. By epoch 1 the
        # spread is 0.6 dB and the balance point is beta ~ 10. Log-spacing
        # 0..3000 covers both regimes with points in the interesting middle.
        for ck in "$dir"/ckpt_epo*.pth.tar; do
            [ -e "$ck" ] || continue
            marker="$dir/.frontier_done_$(basename "$ck" .pth.tar)"
            [ -e "$marker" ] && continue
            log "$tag: measuring frontier on $(basename "$ck")"
            bash "$ROOT/scripts/sweep_frontier.sh" "$ck" "${GPU[$tag]}" 800 0 10 30 100 300 1000 3000 \
                >> "$dir/frontier.log" 2>&1
            touch "$marker"
            log "$tag: frontier done -> $dir/frontier_$(basename "$ck" .pth.tar)/frontier.tsv"
            tail -5 "$dir/frontier_$(basename "$ck" .pth.tar)/frontier.tsv" >> "$LOG" 2>/dev/null
        done
    done
    sleep 300   # 5 minutes
done
