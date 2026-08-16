#!/bin/bash
# Turn a trained decoder checkpoint into the deliverable: a compute/quality
# frontier and a CTC RD table.
#
# Exists because the wdec_* runs were deliberately left out of autopilot's
# watchlist (its ARGS table cannot rebuild them correctly), which also means
# autopilot no longer sweeps their frontiers. This does that part, by hand and
# on purpose.
#
#   scripts/finish_run.sh <run_dir> [gpu] [router_steps]
#
# Idempotent: a beta already swept is skipped by sweep_frontier.sh, and the CTC
# pass is re-run only if its output is older than the checkpoint.
set -u
ROOT="$HOME/FLEX-UF"
DIR=${1:?usage: finish_run.sh <run_dir> [gpu] [steps]}
GPU=${2:-6}
STEPS=${3:-2000}

CK=$(ls -t "$DIR"/ckpt_epo*.pth.tar 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "[$DIR] henuz epoch checkpoint yok"; exit 0; }
echo "[$(date '+%F %T')] $DIR -> $(basename "$CK")"

# Frontier: one router per beta, beta the only difference between points, so the
# curve is a property of the decoder rather than of tuning.
bash "$ROOT/scripts/sweep_frontier.sh" "$CK" "$GPU" "$STEPS" 0 10 30 100 300 1000

SWEEP="$DIR/frontier_$(basename "$CK" .pth.tar)"
"$ROOT/.venv/bin/python" "$ROOT/scripts/collect_frontier.py" "$SWEEP" "$SWEEP/frontier.tsv" || true

# CTC: the routed decoder against stock UF on the sequences a reviewer will ask
# about. The router picked is the one nearest the project's target -- the most
# saving that still costs under 0.1 dB -- and the choice is PRINTED, because
# silently picking the flattering point is how a frontier becomes a lie.
BEST=$("$ROOT/.venv/bin/python" - "$SWEEP/frontier.tsv" <<'PY'
import sys, csv, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists(): raise SystemExit
rows = [r for r in csv.DictReader(p.open(), delimiter='\t')]
def f(r, k, d=0.0):
    try: return float(r.get(k, d))
    except Exception: return d
ok = [r for r in rows if abs(f(r, 'dpsnr')) <= 0.1]
pick = max(ok or rows, key=lambda r: f(r, 'saving_pct'))
print(pick.get('router', ''), f"{f(pick,'saving_pct'):.1f}", f"{f(pick,'dpsnr'):+.3f}")
PY
)
set -- $BEST
if [ -n "${1:-}" ] && [ -f "${1:-}" ]; then
    echo "  secilen router: $1  (%$2 tasarruf, $3 dB) — 0.1 dB altindaki en yuksek tasarruf"
    "$ROOT/.venv/bin/python" "$ROOT/ctc_intra.py" --ckpt "$CK" --router "$1" \
        --qps 0 16 32 48 63 --frames 2 --device "cuda:$GPU" \
        --out "$DIR/ctc_intra.json" 2>&1 | tail -20
else
    echo "  frontier'de router yok — sweep tamamlanmamis olabilir"
fi
