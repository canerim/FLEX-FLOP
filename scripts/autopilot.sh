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
# The baseline is watchdogged alongside the experiments. It is K=1 — structurally
# stock DMCI — and exists to answer whether the multi-exit objective costs the
# deepest exit any quality. Without it the frontier is unfalsifiable: "dB lost
# vs our own deepest exit" looks excellent even if that exit has been degraded.
# So if it dies unnoticed, the three experiments lose their interpretation.
declare -A GPU=( [e1_j2_p128]=4 [e2_j4_p128]=6 [e3_j2_p64]=7 [baseline_singleexit]=6 )
declare -A ARGS=(
  [e1_j2_p128]="--num_exits 6 --split_depth 2 --latent_patch 8"
  [e2_j4_p128]="--num_exits 6 --split_depth 4 --latent_patch 8"
  [e3_j2_p64]="--num_exits 6 --split_depth 2 --latent_patch 4"
  [baseline_singleexit]="--num_exits 1 --split_depth 1 --latent_patch 8"
)
declare -A EPOCHS=( [e1_j2_p128]=105 [e2_j4_p128]=105 [e3_j2_p64]=105 [baseline_singleexit]=6 )

log () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }


# --- why the body lives in a function -------------------------------------
# bash reads a script INCREMENTALLY, by byte offset, while it runs. Editing a
# long-running script in place makes the interpreter resume at a stale offset in
# the new file and execute whatever now sits there. Not theoretical: editing
# add_subsets.sh mid-flight made it jump into its restart section early,
# rebuilding description.json from a partial dataset and restarting all four
# training runs while a tar extraction was still going. Wrapping everything in
# main() and calling it on the last line forces bash to parse the whole file
# before running a single command, so an edit can never take effect halfway.
# (The body is deliberately NOT re-indented: heredoc terminators must sit at
# column 0.)

main () {
log "autopilot started"

while true; do
    for tag in "${!GPU[@]}"; do
        dir="$RUNS/$tag"

        # ---- 1. watchdog ------------------------------------------------
        if ! pgrep -f "train_flexuf_image.py.*--tag $tag" > /dev/null; then
            if [ -f "$dir/ckpt.pth.tar" ]; then
                log "$tag: finished (final ckpt present)"
            else
                # Take an exclusive lock around the relaunch.
                #
                # Without it there is a window: add_subsets.sh kills every run,
                # and if the watchdog's liveness check lands in the gap before
                # add_subsets relaunches them, BOTH launch — two processes with
                # the same --save_dir, interleaving writes to the same
                # status_latest.pth.tar. That happened, and it produced two main
                # processes each for the baseline and e3. A corrupted checkpoint
                # would not have announced itself.
                exec 9>"$ROOT/.relaunch.lock"
                if ! flock -n 9; then
                    log "$tag: looks dead but another process holds the relaunch lock — skipping"
                    exec 9>&-
                    continue
                fi
                if pgrep -f "train_flexuf_image.py.*--tag $tag" > /dev/null; then
                    log "$tag: came back on its own while waiting for the lock"
                    flock -u 9; exec 9>&-
                    continue
                fi
                log "$tag: DIED — restarting on GPU ${GPU[$tag]} (resumes from last epoch)"
                CUDA_VISIBLE_DEVICES="${GPU[$tag]}" nohup "$ROOT/.venv/bin/python" \
                    "$ROOT/train_flexuf_image.py" \
                    --train_dataset "$DATA" --save_dir "$dir" --lambdas 10 2048 \
                    --batch_size 16 -n 8 -e "${EPOCHS[$tag]}" ${ARGS[$tag]} \
                    --latent_halo 2 --adapter_kind conv1x1 \
                    --aux_weight 1.0 --aux_schedule warmup --device 0 --tag "$tag" \
                    >> "$dir/stdout.log" 2>&1 &
                sleep 10
                flock -u 9; exec 9>&-
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
    if len(r['psnr_per_exit']) == 1:
        # The single-exit baseline has no ladder to differentiate; spread is 0
        # by construction. It exists to answer a different question: does adding
        # exits cost the anchor anything versus plain UF trained identically?
        flag = "BASELINE"
    elif deep < 20: flag = "ANCHOR-WEAK"
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
            # Cheap gate before an expensive sweep. A frontier sweep is ~1 h of
            # GPU time on a card that is also training; measuring one on a
            # checkpoint whose oracle is constant yields a table of zeros. That
            # is exactly what ckpt_epo0 produced: the shallowest alternative was
            # 5.2 dB worse, so no dB budget could ever pick it. The diagnostic
            # costs ~1 min and exits non-zero when there is nothing to route.
            if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/oracle_diagnostic.py" \
                    --ckpt "$ck" --device "${GPU[$tag]}" --batches 6 \
                    > "$dir/oracle_$(basename "$ck" .pth.tar).log" 2>&1; then
                log "$tag: $(basename "$ck") has no routing headroom yet — sweep skipped"
                touch "$marker"
                continue
            fi
            # Gate the expensive sweep on a cheap diagnostic.
            #
            # The sweep trains seven routers (7 x 800 steps); the oracle
            # diagnostic is a single pass. And the sweep is pointless whenever
            # the oracle is constant: if no per-tile assignment beats a uniform
            # depth at equal quality, every router it trains collapses onto one
            # exit and it reports the same trivial point seven times. Measured on
            # e3's epoch-0 checkpoint: exit 0 cost +12.09 dB per tile, the oracle
            # put 100% of tiles on the deepest exit, headroom 0.0pp at every tau.
            # So diagnose first, and sweep only when there is something to find.
            log "$tag: oracle diagnostic on $(basename "$ck")"
            diag="$dir/oracle_$(basename "$ck" .pth.tar).txt"
            "$ROOT/.venv/bin/python" "$ROOT/scripts/oracle_diagnostic.py" \
                --ckpt "$ck" --device "${GPU[$tag]}" --batches 8 > "$diag" 2>&1
            grep -E "VERDICT" "$diag" >> "$LOG" 2>/dev/null

            if grep -q "VERDICT: headroom exists" "$diag" 2>/dev/null; then
                log "$tag: headroom found -> measuring frontier"
                bash "$ROOT/scripts/sweep_frontier.sh" "$ck" "${GPU[$tag]}" 800 0 10 30 100 300 1000 3000 \
                    >> "$dir/frontier.log" 2>&1
            else
                log "$tag: no headroom at this checkpoint, skipping the sweep"
            fi
            touch "$marker"
            log "$tag: frontier done -> $dir/frontier_$(basename "$ck" .pth.tar)/frontier.tsv"
            tail -5 "$dir/frontier_$(basename "$ck" .pth.tar)/frontier.tsv" >> "$LOG" 2>/dev/null
        done
    done
    sleep 300   # 5 minutes
done
}

main "$@"
