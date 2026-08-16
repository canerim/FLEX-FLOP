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
BEST=$("$ROOT/.venv/bin/python" - "$SWEEP" <<'PY'
# Pick the router for the CTC pass from the frontier table.
#
# The first version of this read columns named 'dpsnr' and 'router'. Neither
# exists -- collect_frontier.py writes beta/qp/saving_pct/psnr_loss_dB/
# exit_share/control_max_diff -- so the dB filter passed EVERY row (missing key
# -> 0.0 -> "within budget") and the pick degenerated to "most saving, any
# cost". Exactly the failure the comment below warns about, written into the
# code that was supposed to prevent it. Column names are now asserted.
import sys, csv, pathlib
sweep = pathlib.Path(sys.argv[1])
tsv = sweep / "frontier.tsv"
if not tsv.exists(): raise SystemExit
rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
if not rows: raise SystemExit
need = {"beta", "qp", "saving_pct", "psnr_loss_dB"}
missing = need - set(rows[0])
if missing:
    print(f"FRONTIER-SCHEMA-DEGISTI:{sorted(missing)}", file=sys.stderr)
    raise SystemExit

# One beta is one router; it must satisfy the budget across the WHOLE QP range,
# not at its most favourable QP. So aggregate per beta by its WORST qp.
per = {}
for r in rows:
    try:
        b, sv, db = r["beta"], float(r["saving_pct"]), float(r["psnr_loss_dB"])
    except (ValueError, KeyError):
        continue
    cur = per.get(b)
    per[b] = (min(cur[0], sv), max(cur[1], db)) if cur else (sv, db)

ok = {b: v for b, v in per.items() if v[1] <= 0.1}
if not ok:
    b, (sv, db) = min(per.items(), key=lambda kv: kv[1][1])
    print(f"NONE-IN-BUDGET beta={b} en_iyi_kayip={db:.3f}dB tasarruf={sv:.1f}%", file=sys.stderr)
else:
    b, (sv, db) = max(ok.items(), key=lambda kv: kv[1][0])
ck = sweep / f"beta_{b}" / "router.pth.tar"
print(ck if ck.exists() else "", f"{sv:.1f}", f"{db:.3f}")
PY
)
set -- $BEST
if [ -n "${1:-}" ] && [ -f "${1:-}" ]; then
    echo "  secilen router: $1  (%$2 tasarruf, en kotu qp kaybi $3 dB) — butce icindeki en yuksek tasarruf"
    "$ROOT/.venv/bin/python" "$ROOT/ctc_intra.py" --ckpt "$CK" --router "$1" \
        --qps 0 16 32 48 63 --frames 2 --device "cuda:$GPU" \
        --out "$DIR/ctc_intra.json" 2>&1 | tail -20
else
    echo "  frontier'de router yok — sweep tamamlanmamis olabilir"
fi
