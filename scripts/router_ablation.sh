#!/usr/bin/env bash
# Which of the router's inputs actually carry the signal?
#
# The paper's 144K head loses to a zero-parameter rule that reads the entropy
# coder's per-tile bit count, so a reviewer is entitled to ask what the learned
# router is for. This closes the mechanism the only way that means anything:
# train the SAME head once per input group, each time with every other group
# zeroed AFTER its projection, and read off which group the agreement survives.
#
# Zeroing after the projection rather than deleting the projection is the whole
# trick. Architecture, parameter count, optimiser, seed, image order and qp
# draws are identical across the variants; the only thing that differs is how
# much the head is allowed to know.
#
# qp stays live in the single-group variants deliberately. It is one number per
# FRAME, the same for every tile in it, so it can say which operating point we
# are at and cannot, even in principle, tell two tiles of one frame apart.
# Holding it live keeps the variants comparable on PER-TILE information alone.
# The last run, qp on its own, is the floor that choice implies: the agreement
# reachable with no per-tile information whatsoever. Without it a number like
# 0.62 for stem-only cannot be read, because nobody knows what 0.62 is worth.
#
# One GPU, one variant at a time, resumable: a variant whose results JSON
# already exists is skipped, so the script can be re-run after an interruption
# without repeating work. CUDA_VISIBLE_DEVICES pins the GPU and --device cuda:0
# addresses it from inside, which means a mistake in this script cannot reach a
# neighbour's GPU even if it tries.
#
#   nohup scripts/router_ablation.sh runs/RECIPE512/ckpt_PAPER.pth.tar 7 \
#       > logs/router_ablation.log 2>&1 &
set -u
cd "$HOME/FLEX-UF"
CK=${1:-runs/RECIPE512/ckpt_PAPER.pth.tar}
GPU=${2:-7}
STEPS=${3:-3000}
LAM=${4:-1.3e-5}
EVAL=${5:-200}
PY=./.venv/bin/python
OUT=runs/RECIPE512/routers2/ablation
mkdir -p "$OUT" results logs

# label:live-groups, in the order they run. The label is what the results file
# is named after; the groups are what the head is actually given.
VARIANTS=${VARIANTS:-"stem:stem,qp latent:latent,qp scales:scales,qp bits:bits,qp all:all qp:qp"}

echo "=== router input ablation"
echo "    ckpt   $CK"
echo "    gpu    $GPU (as cuda:0 inside each run)"
echo "    steps  $STEPS   lam $LAM   eval batches $EVAL"
echo "    order  $VARIANTS"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | sed 's/^/    gpu /'

for v in $VARIANTS; do
  name=${v%%:*}; groups=${v#*:}
  f="$OUT/inputs_${name}.pth"
  r="results/router_ablation_${name}.json"
  if [ -s "$r" ]; then
    echo; echo "=== $name zaten olculmus, atlaniyor ($r)"
    continue
  fi
  echo; echo "=== $(date -Is)  $name   live: $groups"
  CUDA_VISIBLE_DEVICES="$GPU" $PY scripts/train_router2.py \
      --ckpt "$CK" --lam "$LAM" --steps "$STEPS" --eval_steps "$EVAL" \
      --device cuda:0 --inputs "$groups" --label "$name" \
      --out "$f" --results "$r" 2>&1 | tee "logs/router_ablation_${name}.log"
  # tee is the last command in the pipe, so its status is not the one that
  # matters. A variant that dies must not take the rest of the sweep with it.
  st=${PIPESTATUS[0]}
  echo "=== $(date -Is)  $name exit $st"
done

echo; echo "=== $(date -Is)  collecting"
$PY - <<'PYEOF'
"""Merge the per-variant results into one file, so the comparison exists as a
measurement and not only as six separate ones a reader has to line up."""
import json
from pathlib import Path

rows = []
for p in sorted(Path("results").glob("router_ablation_*.json")):
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
    out = {"script": "scripts/router_ablation.sh",
           "measured": "agreement with the oracle's exit choice, on fresh "
                       "images the head never trained on, +- one standard "
                       "error taken across frames",
           "ckpt": rows[0]["ckpt"], "ckpt_epoch": rows[0]["ckpt_epoch"],
           "one_checkpoint": len(ck) == 1, "lam": rows[0]["lam"],
           "qp_note": "qp is live in every single-group variant because it is a "
                      "per-frame constant and cannot separate tiles; the qp-only "
                      "row is the no-per-tile-information floor",
           "variants": rows}
    Path("results/router_ablation.json").write_text(json.dumps(out, indent=2))
    w = max(len(r["label"]) for r in rows)
    print(f"  {'variant':<{w}}  agreement          floor   params")
    for r in rows:
        a, s, c = r["agree"], r["stderr"], r["constant_best_agree"]
        cell = "-" if a is None else f"{a:.4f} +- {s:.4f}"
        floor = "-" if c is None else f"{c:.4f}"
        print(f"  {r['label']:<{w}}  {cell:>17}  {floor:>6}  "
              f"{r['router_params']:>8,}")
    print("  -> results/router_ablation.json")
else:
    print("  no per-variant results found")
PYEOF
echo "=== $(date -Is)  done"
