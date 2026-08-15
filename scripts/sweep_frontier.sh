#!/bin/bash
# Trace the compute/quality frontier for one trained decoder.
#
# One router is trained per beta, and each becomes one point on the curve. beta
# is the only thing that differs between points -- same decoder, same data, same
# everything else -- so the curve is a property of the decoder, not of tuning.
#
#   beta = 0      router only cares about quality, parks every tile on the
#                 deepest exit: ~0% saving. The trivial end -- and useful, since
#                 the dB it still loses there is the pure patch-boundary cost.
#   beta large    compute dominates, everything goes shallow.
#   in between    the interesting region.
#
# The useful beta range depends on how FLAT the ladder is at this checkpoint,
# and that changes by two orders of magnitude over training: at ckpt_epo0 the
# ladder spans 10.4 dB and nothing moves below beta ~ 550, while by epoch 1 the
# spread is 0.6 dB and the balance point is beta ~ 10. Hence a wide log sweep.
#
# Resumable and idempotent, deliberately
# --------------------------------------
# A sweep is a sequence of independent router trainings that together take the
# better part of an hour, on a GPU shared with a run that must not be disturbed.
# Anything can interrupt it: a restart of the supervising process, a reboot, a
# kill. The first version truncated frontier.tsv on every start, so an
# interruption threw completed betas away and left an EMPTY table sitting next
# to four finished eval.json files. Work that is done must survive. So a beta
# whose eval.json already exists is skipped, and the table is rebuilt at the end
# from whatever eval.json files are present rather than appended to as we go.
#
# Usage:  bash scripts/sweep_frontier.sh <decoder-ckpt> [gpu] [steps] [betas...]
set -u

CKPT=${1:?usage: sweep_frontier.sh <decoder-ckpt> [gpu] [steps] [betas...]}
GPU=${2:-4}
STEPS=${3:-1000}
shift 3 2>/dev/null || shift $#
BETAS=("$@")
[ ${#BETAS[@]} -eq 0 ] && BETAS=(0 10 30 100 300 1000 3000)

ROOT="$HOME/FLEX-UF"
PY="$ROOT/.venv/bin/python"
DATA="${FLEXUF_DATA:-/data10/shareddata/openimages/dcvc_train}"
OUT="$(dirname "$CKPT")/frontier_$(basename "$CKPT" .pth.tar)"
mkdir -p "$OUT"
SUM="$OUT/frontier.tsv"

echo "frontier sweep: $CKPT on GPU $GPU, betas ${BETAS[*]}, $STEPS steps each"

for beta in "${BETAS[@]}"; do
    dir="$OUT/beta_${beta}"
    mkdir -p "$dir"
    if [ -s "$dir/eval.json" ]; then
        echo "  beta=$beta already measured, skipping"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/train_router.py" \
        --ckpt "$CKPT" --train_dataset "$DATA" --save_dir "$dir" \
        --steps "$STEPS" --batch_size 8 --crop 512 -n 4 \
        --beta "$beta" --device 0 > "$dir/train.log" 2>&1 \
        || { echo "  beta=$beta router FAILED"; continue; }

    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/evaluate.py" \
        --ckpt "$CKPT" --dataset "$DATA" --router "$dir/router.pth.tar" \
        --n_frames 96 --crop 512 --batch_size 2 --qps 63 --device 0 \
        --out "$dir/eval.json" > "$dir/eval.log" 2>&1 \
        || { echo "  beta=$beta eval FAILED"; continue; }
    echo "  beta=$beta measured"
done

# Rebuild the table from every eval.json present, sorted by beta.
"$PY" "$ROOT/scripts/collect_frontier.py" "$OUT" "$SUM"
echo "frontier -> $SUM"
