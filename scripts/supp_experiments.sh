#!/bin/bash
# Everything the supplementary reports that was not already measured.
#
# One job at a time, on the evaluation card, behind the same lock the
# per-checkpoint watchers use. Five watchers and this queue sharing one GPU
# without the lock would collide, and an OOM here would read as a failed
# experiment rather than a scheduling mistake.
#
# Every job runs against the PINNED checkpoint. ckpt_eval.pth.tar is a moving
# target -- it was overwritten mid-measurement once already -- and a
# supplementary whose tables come from three different epochs is worse than no
# supplementary.
set -u
cd "$HOME/FLEX-UF"
CKPT=runs/RECIPE512/ckpt_PAPER.pth.tar
GPU=${GPU:-2}
LOCK=/tmp/flexuf_eval.lock
LOG=results/supp_queue.log
PY=./.venv/bin/python

run () {  # run <name> <command...>
  local name="$1"; shift
  local out="results/supp_${name}.json"
  if [ -s "$out" ]; then echo "$(date '+%F %T')  $name: already have $out, skipping" >>"$LOG"; return; fi
  echo "$(date '+%F %T')  $name: start" >>"$LOG"
  (
    flock -w 21600 9 || { echo "  $name: could not take the lock in 6 h" >>"$LOG"; exit 1; }
    CUDA_VISIBLE_DEVICES="$GPU" "$@" >>"$LOG" 2>&1
  ) 9>"$LOCK"
  echo "$(date '+%F %T')  $name: done rc=$? -> $(ls -la $out 2>/dev/null | awk '{print $5}') bytes" >>"$LOG"
}

echo "=== supplementary queue started $(date '+%F %T') on GPU $GPU ===" >>"$LOG"

# 1. Saving against budget, the whole grid rather than the single 0.1 dB column
#    the main paper quotes. The band argument of Section 5 lives on this grid.
run budget_grid $PY -u scripts/budget_table.py --ckpt "$CKPT" \
    --budgets 0.05 0.1 0.15 0.2 0.3 0.5 0.75 1.0 --qps 0 16 32 48 63 \
    --frames 1 --device cuda:0 --out results/supp_budget_grid.json

# 2. Per test class at three budgets, not one. The main paper's per-class table
#    is a single row of this.
run per_class_budgets $PY -u scripts/per_class.py --ckpt "$CKPT" \
    --qps 0 16 32 48 63 --budgets 0.1 0.3 0.5 --frames 1 --device cuda:0 \
    --out results/supp_per_class_budgets.json

# 3. Split depth. Sweeping j moves the seam and the ceiling together, and the
#    supplementary is where the whole sweep belongs.
run seam_vs_split $PY -u scripts/seam_vs_split.py --ckpt "$CKPT" \
    --qps 0 16 32 48 63 --max_seqs 20 --device cuda:0 \
    --out results/supp_seam_vs_split.json

# 4. Wall clock, at three resolutions. MACs are what we optimise; seconds are
#    what a reader wants to see, and the gap between them is a result.
for WH in "1920 1080" "1280 720" "3840 2160"; do
  set -- $WH
  run latency_${1}x${2} $PY -u scripts/latency_profile.py --ckpt "$CKPT" \
      --qp 32 --budget_db 0.1 --width "$1" --height "$2" \
      --warmup 20 --iters 100 --device cuda:0 \
      --out results/supp_latency_${1}x${2}.json
done

echo "=== supplementary queue finished $(date '+%F %T') ===" >>"$LOG"
