#!/usr/bin/env bash
# The router input ablation, on the checkpoint the paper reports.
#
# scripts/router_ablation.sh ran this sweep on the weights that held on
# 20 August. The paper now reports epoch 4, and a head trained against other
# weights answers a question about those weights, so the sweep has to be
# repeated rather than re-labelled.
#
# It writes under a _e4 suffix instead of overwriting, for one practical
# reason: the sweep takes about six hours and the supplement reads
# router_ablation.json throughout. Deleting the old files at the start would
# leave the document unbuildable for the length of the sweep. The readers
# prefer the _e4 file and fall back to the older one, so the document is
# correct before and after and never broken in between.
#
#   nohup scripts/router_ablation_e4.sh runs/RECIPE512/ckpt_PAPER.pth.tar 4 \
#       > logs/router_ablation_e4.log 2>&1 &
set -u
cd "$HOME/FLEX-UF"
CK=${1:-runs/RECIPE512/ckpt_PAPER.pth.tar}
GPU=${2:-4}
STEPS=${3:-3000}
LAM=${4:-1.3e-5}
EVAL=${5:-200}
PY=./.venv/bin/python
OUT=runs/RECIPE512/routers2/ablation
mkdir -p "$OUT" results logs

VARIANTS=${VARIANTS:-"stem:stem,qp latent:latent,qp scales:scales,qp bits:bits,qp all:all qp:qp"}

echo "=== router input ablation, pinned checkpoint"
echo "    ckpt   $CK"
echo "    gpu    $GPU (as cuda:0 inside each run)"
echo "    steps  $STEPS   lam $LAM   eval batches $EVAL"
echo "    order  $VARIANTS"

for v in $VARIANTS; do
  name=${v%%:*}; groups=${v#*:}
  f="$OUT/inputs_${name}_e4.pth"
  r="results/router_ablation_${name}_e4.json"
  if [ -s "$r" ]; then
    echo; echo "=== $name already measured on this pin, skipping ($r)"
    continue
  fi
  echo; echo "=== $(date -Is)  $name   live: $groups"
  CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/train_router2.py \
      --ckpt "$CK" --lam "$LAM" --steps "$STEPS" --eval_steps "$EVAL" \
      --device cuda:0 --inputs "$groups" --label "$name" \
      --out "$f" --results "$r" 2>&1 | tee "logs/router_ablation_${name}_e4.log"
  st=${PIPESTATUS[0]}
  echo "=== $(date -Is)  $name exit $st"
done

echo; echo "=== $(date -Is)  collecting"
$PY - <<'PYEOF'
"""Merge the per-variant records, this time only the ones on the pin."""
import json
from pathlib import Path

rows = []
for p in sorted(Path("results").glob("router_ablation_*_e4.json")):
    d = json.loads(p.read_text())
    e = d.get("eval") or {}
    rows.append({"label": d["label"], "inputs": d["inputs"],
                 "router_params": d["router_params"],
                 "agree": e.get("agree"), "stderr": e.get("stderr_across_frames"),
                 "n_frames": e.get("n_frames"), "n_tiles": e.get("n_tiles"),
                 "constant_best_agree": e.get("constant_best_agree"),
                 "heldout_agree_last250": d.get("heldout_agree_last250"),
                 "pred_hist": e.get("pred_hist"), "oracle_hist": e.get("oracle_hist"),
                 "router_ckpt": d["router_ckpt"], "ckpt": d["ckpt"],
                 "ckpt_epoch": d["ckpt_epoch"], "lam": d["lam"],
                 "steps": d["steps"], "seed": d["seed"], "wall_s": d["wall_s"]})
if rows:
    rows.sort(key=lambda r: (r["agree"] is None, -(r["agree"] or 0)))
    ck = {(r["ckpt"], r["ckpt_epoch"]) for r in rows}
    out = {"script": "scripts/router_ablation_e4.sh",
           "measured": "agreement with the oracle's exit choice, on fresh "
                       "images the head never trained on, +- one standard "
                       "error taken across frames",
           "ckpt": rows[0]["ckpt"], "ckpt_epoch": rows[0]["ckpt_epoch"],
           "one_checkpoint": len(ck) == 1, "lam": rows[0]["lam"],
           "qp_note": "qp is live in every single-group variant because it is a "
                      "per-frame constant and cannot separate tiles; the qp-only "
                      "row is the no-per-tile-information floor",
           "variants": rows}
    Path("results/router_ablation_e4.json").write_text(json.dumps(out, indent=2))
    w = max(len(r["label"]) for r in rows)
    print(f"  {'variant':<{w}}  agreement          floor   params")
    for r in rows:
        a, s, c = r["agree"], r["stderr"], r["constant_best_agree"]
        cell = "-" if a is None else f"{a:.4f} +- {s:.4f}"
        floor = "-" if c is None else f"{c:.4f}"
        print(f"  {r['label']:<{w}}  {cell:>17}  {floor:>6}  "
              f"{r['router_params']:>8,}")
    print("  -> results/router_ablation_e4.json")
else:
    print("  no per-variant results found")
PYEOF
echo "=== $(date -Is)  done"
