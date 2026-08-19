#!/usr/bin/env bash
# Re-measure the whole dependent chain on ONE pinned checkpoint.
#
# Written after two things went wrong at once: the cost model under-billed the
# FFN adapter (DECISIONS 90), and runs/RECIPE512/ckpt_eval.pth.tar turned out to
# be a moving target -- the watcher overwrites it every epoch, so files written
# hours apart described different decoders. check_paper.py then flagged the mix
# rather than the individual numbers, which is the only reason it was visible.
#
# So: one checkpoint, passed explicitly, and everything downstream of it in
# dependency order. Anything that reads results/ (tables, figures, the paper)
# comes after, and is not run here.
#
#   scripts/remeasure_all.sh runs/RECIPE512/ckpt_PAPER.pth.tar 2
set -u
CK="${1:?usage: remeasure_all.sh <ckpt> <gpu>}"
GPU="${2:?}"
PY=./.venv/bin/python
D="cuda:${GPU}"
R2=runs/RECIPE512/routers2/v2_lam1.3e-5.pth
log() { echo; echo "=== $* ==="; }

log "1/8 signalled, three budgets (the headline)"
$PY scripts/signalled_curve.py --ckpt "$CK" --budgets 0.1 0.3 0.5 \
    --device "$D" --out results/signalled_RECIPE512_ctc53.json

log "2/8 saturation (floor and ceiling)"
$PY scripts/saturation.py --ckpt "$CK" --qps 0 8 16 24 32 40 48 56 63 \
    --device "$D" --out results/saturation_RECIPE512_ctc53.json

log "3/8 configuration B, three budgets"
for b in 01:0.1 03:0.3 05:0.5; do
  $PY scripts/router_curve.py --ckpt "$CK" --router2 "$R2" \
      --budget "${b#*:}" --device "$D" \
      --out "results/router_RECIPE512_b${b%%:*}.json"
done

log "4/8 the parameter-free rule"
for b in 01:0.1 03:0.3; do
  $PY scripts/raterank_curve.py --ckpt "$CK" --budget "${b#*:}" \
      --device "$D" --out "results/raterank_RECIPE512_b${b%%:*}.json"
done

log "5/8 configuration C, both predictors"
$PY scripts/hybrid_curve.py --ckpt "$CK" --router2 "$R2" --budget 0.1 \
    --device "$D" --out results/hybrid_RECIPE512_b01_fixed.json
$PY scripts/hybrid_curve.py --ckpt "$CK" --router2 "$R2" --budget 0.1 \
    --predictor raterank --device "$D" --out results/hybrid_raterank_b01.json

log "6/8 the static controls"
$PY scripts/static_baseline.py --ckpt "$CK" --budget 0.1 --device "$D" \
    --out results/static_RECIPE512_b01.json

log "7/8 per class, with the exit histogram"
$PY scripts/per_class.py --ckpt "$CK" --budgets 0.1 0.3 0.5 --device "$D" \
    --out results/per_class_RECIPE512.json

log "8/8 the dense budget grid (the band collapse)"
$PY scripts/signalled_curve.py --ckpt "$CK" \
    --budgets 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \
    --device "$D" --out results/signalled_RECIPE512_grid.json

echo; echo "=== done. now: make tables paper report, then check_paper ==="
