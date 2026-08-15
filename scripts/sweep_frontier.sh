#!/bin/bash
# Trace the compute/quality frontier for one trained decoder.
#
# One router is trained per beta, and each becomes one point on the curve. beta
# is the only thing that changes between points -- same decoder, same data,
# same everything else -- so the curve is a property of the decoder rather than
# of the tuning.
#
#   beta = 0      router only cares about quality, parks every tile on the
#                 deepest exit: ~0% saving, ~0 dB loss. Trivial end.
#   beta large    compute dominates, everything goes shallow: large saving,
#                 large dB loss. Other trivial end.
#   in between    the interesting region, and the point of the whole project.
#
# Usage:  bash scripts/sweep_frontier.sh <decoder-ckpt> [gpu] [steps] [betas...]
set -u

CKPT=${1:?usage: sweep_frontier.sh <decoder-ckpt> [gpu] [steps] [betas...]}
GPU=${2:-4}
STEPS=${3:-1500}
shift 3 2>/dev/null || shift $#
BETAS=("$@")
[ ${#BETAS[@]} -eq 0 ] && BETAS=(0 10 25 50 100)

ROOT="$HOME/FLEX-UF"
PY="$ROOT/.venv/bin/python"
DATA="${FLEXUF_DATA:-/data10/shareddata/openimages/dcvc_train}"
OUT="$(dirname "$CKPT")/frontier_$(basename "$CKPT" .pth.tar)"
mkdir -p "$OUT"
SUM="$OUT/frontier.tsv"
printf 'beta\tsaving_pct\tpsnr_loss_dB\texit_share\n' > "$SUM"

echo "frontier sweep: $CKPT on GPU $GPU, betas ${BETAS[*]}, $STEPS steps each"

for beta in "${BETAS[@]}"; do
    dir="$OUT/beta_${beta}"; mkdir -p "$dir"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/train_router.py" \
        --ckpt "$CKPT" --train_dataset "$DATA" --save_dir "$dir" \
        --steps "$STEPS" --batch_size 8 --crop 512 -n 4 \
        --beta "$beta" --device 0 > "$dir/train.log" 2>&1 || { echo "  beta=$beta router FAILED"; continue; }

    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/evaluate.py" \
        --ckpt "$CKPT" --dataset "$DATA" --router "$dir/router.pth.tar" \
        --n_frames 96 --crop 512 --batch_size 2 --qps 63 --device 0 \
        --out "$dir/eval.json" > "$dir/eval.log" 2>&1 || { echo "  beta=$beta eval FAILED"; continue; }

    "$PY" - "$dir/eval.json" "$beta" "$SUM" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); beta = sys.argv[2]
r = d["results"]["63"]["routed"]
line = f"{beta}\t{r['saving_pct']}\t{r['psnr_loss_dB']}\t{r['exit_share']}"
open(sys.argv[3], "a").write(line + "\n")
print(f"  beta={beta:>4}  saving {r['saving_pct']:6.2f}%   "
      f"PSNR loss {r['psnr_loss_dB']:+.4f} dB   share {r['exit_share']}")
PY
done
echo "frontier -> $SUM"
cat "$SUM"
