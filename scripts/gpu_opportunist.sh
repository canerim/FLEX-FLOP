#!/bin/bash
# Claim a GPU the moment one genuinely frees up, and start the next experiment.
#
# What counts as free, and why the test is strict
# -----------------------------------------------
# GPU 3 reads 0% utilisation and has been doing so all day — while holding 44 GB
# of another user's process. Utilisation alone would have called it free and any
# job started there would have OOM'd someone else's work. So a card is only taken
# when BOTH hold:
#   - no compute process is registered on it at all, and
#   - its memory is essentially empty (< 500 MiB)
# and only after the condition has held for two consecutive checks, so a card is
# not grabbed in the gap between someone's job finishing and their next one
# starting.
#
# The queue, in priority order. Each entry differs from e1 in exactly one
# variable, so every comparison stays a clean ablation.
#
#   e4  adapter=ffn      Does the 1x1 adapter have enough capacity? The 1x1 was
#                        chosen because a DepthConvBlock is 74.8% FFN and 0.3%
#                        spatial, so an early exit loses mostly pointwise
#                        capacity. The FFN-shaped adapter is the same idea with
#                        more of it — still entirely pointwise, so still zero
#                        boundary penalty, at 2C^2 instead of C^2 per pixel.
#                        This is the one design choice we have not tested.
#   e5  j=3              Fills the gap between e1 (j=2) and e2 (j=4), turning the
#                        split-depth axis from two points into three.
#   e6  K=4              The exit-count axis. FLEX found K is not a free knob —
#                        K=3 and K=12 both collapsed — but that was on DCVC-RT.
#   e7  K=12             The other end of it.
set -u

ROOT="$HOME/FLEX-UF"
DATA=/data10/shareddata/openimages/dcvc_train
LOG="$ROOT/autopilot.log"

# tag|extra args
QUEUE=(
  "e4_ffn_adapter|--num_exits 6 --split_depth 2 --latent_patch 8 --adapter_kind ffn"
  "e5_j3_p128|--num_exits 6 --split_depth 3 --latent_patch 8 --adapter_kind conv1x1"
  "e6_k4_p128|--num_exits 4 --split_depth 2 --latent_patch 8 --adapter_kind conv1x1"
  "e7_k12_p128|--num_exits 12 --split_depth 4 --latent_patch 8 --adapter_kind conv1x1"
)

log () { echo "[$(date '+%F %T')] gpu-opportunist: $*" >> "$LOG"; }
log "started; queue = ${#QUEUE[@]} experiments"

declare -A streak

free_gpus () {
    # A card is free only if no compute process is registered AND memory is tiny.
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader \
      | while IFS=, read -r idx uuid mem; do
            idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' ')
            mem=$(echo "$mem" | tr -dc '0-9')
            if ! grep -q "$uuid" <<< "$busy" && [ "${mem:-99999}" -lt 500 ]; then
                echo "$idx"
            fi
        done
}

while [ ${#QUEUE[@]} -gt 0 ]; do
    for g in $(free_gpus); do
        streak[$g]=$(( ${streak[$g]:-0} + 1 ))
        # require two consecutive sightings before claiming
        if [ "${streak[$g]}" -lt 2 ]; then
            log "GPU $g looks free (1/2) — waiting one more cycle"
            continue
        fi
        entry="${QUEUE[0]}"
        tag="${entry%%|*}"
        args="${entry#*|}"
        log "GPU $g free — launching $tag ($args)"
        mkdir -p "$ROOT/runs/$tag"
        CUDA_VISIBLE_DEVICES="$g" setsid nohup "$ROOT/.venv/bin/python" \
            "$ROOT/train_flexuf_image.py" \
            --train_dataset "$DATA" --save_dir "$ROOT/runs/$tag" \
            --lambdas 10 2048 --batch_size 16 -n 8 -e 105 \
            $args --latent_halo 2 --aux_weight 1.0 --aux_schedule warmup \
            --device 0 --tag "$tag" \
            >> "$ROOT/runs/$tag/stdout.log" 2>&1 < /dev/null &
        sleep 20
        QUEUE=("${QUEUE[@]:1}")
        unset 'streak[$g]'
        break
    done
    # decay streaks for cards that stopped looking free
    for g in "${!streak[@]}"; do
        grep -qx "$g" <<< "$(free_gpus)" || unset 'streak[$g]'
    done
    sleep 120
done

log "queue exhausted"
