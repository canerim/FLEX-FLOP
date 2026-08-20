#!/bin/bash
# The second supplementary queue: the measurements the gap analysis asks for
# that can be taken on one evaluation card in an afternoon.
#
# Modelled on scripts/supp_experiments.sh, and for the same reasons. One job at
# a time on the evaluation card, behind the lock the per-checkpoint watchers
# use, because five watchers and this queue sharing a card without the lock
# would collide and the resulting OOM would read as a failed experiment rather
# than a scheduling mistake. Every job against the PINNED checkpoint, because a
# supplementary whose tables come from three different epochs is worse than no
# supplementary. Every job skipped if its output is already there, so the queue
# can be restarted after an interruption without repeating work. A job that
# fails is logged and the queue carries on: one broken measurement should not
# cost the other nine their card time.
#
# What is NOT here, and why:
#   * a repeated training run for a training-side interval. Forbidden on this
#     machine while four runs are live. The substitute is the per-sequence
#     spread below, which is an evaluation-side interval and is labelled as one.
#   * anything above 1080p. GPU 2 has about 5 GB free and a 4K decode does not
#     fit; the queued 4K job in the first queue returned without writing a file.
#   * a second GPU class. All eight cards in this machine are RTX A6000s, so
#     the only second device class reachable is the CPU, which is measured here.
#   * the RD and frontier figure block. Its data is already measured; it is a
#     plotting job and needs no card time.
set -u
cd "$HOME/FLEX-UF"
CKPT=runs/RECIPE512/ckpt_PAPER.pth.tar
SIGNALLED=results/signalled_RECIPE512_ctc53.json
CURVE=results/supp_paper_curve_PAPER.json
GPU=${GPU:-2}
LOCK=/tmp/flexuf_eval.lock
LOG=results/supp_queue2.log
PY=./.venv/bin/python

log () { echo "$(date '+%F %T')  $*" >>"$LOG"; }

size_of () { [ -f "$1" ] && stat -c%s "$1" || echo 0; }

# run <name> <output file> <command...>   -- on the card, behind the lock
run () {
  local name="$1" out="$2"; shift 2
  if [ -s "$out" ]; then log "$name: already have $out, skipping"; return 0; fi
  log "$name: start -> $out"
  (
    flock -w 21600 9 || { echo "  $name: could not take the lock in 6 h"; exit 99; }
    CUDA_VISIBLE_DEVICES="$GPU" "$@"
  ) 9>"$LOCK" >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
    log "$name: FAILED rc=$rc, $(size_of "$out") bytes in $out"
  else
    log "$name: done rc=$rc -> $(size_of "$out") bytes"
  fi
  return 0
}

# run_off <name> <output file> <command...> -- no card, no lock: post-processing
# of files already measured, and the CPU timing, which must not hold the card
# for the twenty minutes it spends not using it.
run_off () {
  local name="$1" out="$2"; shift 2
  if [ -s "$out" ]; then log "$name: already have $out, skipping"; return 0; fi
  log "$name: start (no GPU) -> $out"
  CUDA_VISIBLE_DEVICES="" "$@" >>"$LOG" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
    log "$name: FAILED rc=$rc, $(size_of "$out") bytes in $out"
  else
    log "$name: done rc=$rc -> $(size_of "$out") bytes"
  fi
  return 0
}

mkdir -p paper/figures results
echo "=== supplementary queue 2 started $(date '+%F %T') on GPU $GPU, $CKPT ===" >>"$LOG"

# ---------------------------------------------------------------------------
# 1. Visual comparison at the operating point, with provenance.
#
# The most conspicuous absence in the document: no side-by-side decode anywhere,
# in a paper whose claim is that tiles can stop early without visible loss.
# Figures for it exist, but each script defaulted to a per-epoch checkpoint the
# watchers have since overwritten and none wrote a sidecar, so no qualitative
# figure in this work can state what decoded it. Re-measured here against the
# pinned checkpoint, each writing a JSON sidecar naming the checkpoint, the
# sequence, the rate, the budget and the numbers printed on the panels.
# ---------------------------------------------------------------------------
for B in 01 03 05; do
  case $B in 01) BD=0.1;; 03) BD=0.3;; 05) BD=0.5;; esac
  run qualitative_b$B results/supp_qual_bosphorus_q32_b$B.json \
      $PY -u scripts/qualitative.py --ckpt "$CKPT" --seq Bosphorus --qp 32 \
      --budget $BD --device cuda:0 \
      --out paper/figures/qual_PAPER_bosphorus_q32_b$B.png \
      --sidecar results/supp_qual_bosphorus_q32_b$B.json
done

# The same panel on the rate and the content where the method is worst, so the
# grid is not three views of the easy case.
run qualitative_hard results/supp_qual_basketballdrive_q63_b01.json \
    $PY -u scripts/qualitative.py --ckpt "$CKPT" --seq BasketballDrive --qp 63 \
    --budget 0.1 --device cuda:0 \
    --out paper/figures/qual_PAPER_basketballdrive_q63_b01.png \
    --sidecar results/supp_qual_basketballdrive_q63_b01.json

# Where the decoder spends, by content: one 1080p sequence of still water, one
# of moving detail, one at 832x480 where a frame is 8 tiles rather than 40.
# Five sequences across four resolutions: still water and moving detail at
# 1080p, a conference scene at 720p where a frame is 18 tiles, 832x480 where it
# is 8, and 416x240 where it is 2 and there is barely an allocation to make.
for S in Bosphorus BasketballDrive FourPeople PartyScene RaceHorses_416x240; do
  L=$(echo "$S" | tr 'A-Z' 'a-z')
  run exitmap_$L results/supp_exitmap_${L}_q32_b01.json \
      $PY -u scripts/exit_map_figure.py --ckpt "$CKPT" --seq "$S" --qp 32 \
      --budget 0.1 --device cuda:0 \
      --out paper/figures/exitmap_PAPER_${L}_q32_b01.png \
      --sidecar results/supp_exitmap_${L}_q32_b01.json
done

# The failure case the measurement chose, rather than one chosen by hand:
# results/supp_opquality_PAPER.json ranks videoSRC24 at qp 0 as holding the
# worst tile in the whole test set at the 0.1 dB budget.
run qualitative_worst results/supp_qual_videoSRC24_q0_b01.json \
    $PY -u scripts/qualitative.py --ckpt "$CKPT" --seq videoSRC24 --qp 0 \
    --budget 0.1 --device cuda:0 \
    --out paper/figures/qual_PAPER_videosrc24_q0_b01.png \
    --sidecar results/supp_qual_videoSRC24_q0_b01.json

# The seam at the rate where it is largest, with the difference amplified.
run seam_q63 results/supp_seam_problem_bosphorus_q63.json \
    $PY -u scripts/seam_problem.py --ckpt "$CKPT" --seq Bosphorus --qp 63 \
    --device cuda:0 --out paper/figures/seam_PAPER_bosphorus_q63.png \
    --sidecar results/supp_seam_problem_bosphorus_q63.json

# ---------------------------------------------------------------------------
# 2. The frontier on the pinned checkpoint, with its per-sequence breakdown.
#
# results/per_sequence.json is on runs/BEST/ckpt_eval.pth.tar and cannot be
# regenerated from the curve it names. The spread across sequences is also the
# only interval this project can honestly report: no configuration was trained
# twice and no seed is set in the decoder trainer, so a training-side interval
# does not exist and is stated as absent rather than substituted for.
# ---------------------------------------------------------------------------
run paper_curve "$CURVE" \
    $PY -u scripts/paper_curve.py --ckpt "$CKPT" --qps 0 16 32 48 63 \
    --frames 1 --device cuda:0 --out "$CURVE"

# 0.10 through 0.30 and no further. Above the saturation point every tile is
# already at the shallowest exit the split depth allows, the frontier stops at
# the ceiling (39.12% of a released decode, results/saturation_RECIPE512_ctc53.
# json), and paper_curve.py marks such an operating point saturated and attaches
# no per-sequence breakdown to it because there is no allocation left to break
# down. At 0.50 dB every rate is saturated, so that budget has no spread to
# report; at 0.30 dB only qp48 and qp63 are still on the frontier.
for T in 0.10 0.15 0.20 0.25 0.30; do
  TAG=$(echo "$T" | tr -d '.')
  run_off per_sequence_$TAG results/supp_per_sequence_PAPER_b$TAG.json \
      $PY -u scripts/per_sequence_spread.py --curve "$CURVE" --target "$T" \
      --out results/supp_per_sequence_PAPER_b$TAG.json
done

# ---------------------------------------------------------------------------
# 3. BD-Rate and BD-saving on the pinned checkpoint.
#
# results/bdrate.json records no checkpoint at all and predates the current
# compute-cost model; results/bd_RECIPE512_ctc53.json is ckpt_eval. The anchor
# drift the BD-Rate rows subtract is also on ckpt_eval, so it is re-measured
# first. BD-saving is reported in both decibel conventions because the two
# differ by about a third of a 0.1 dB budget and a single number would silently
# pick a side.
# ---------------------------------------------------------------------------
run anchor results/supp_anchor_PAPER.json \
    $PY -u scripts/anchor_drift.py --ckpt "$CKPT" --qps 0 16 32 48 63 \
    --frames 1 --device cuda:0 --out results/supp_anchor_PAPER.json

for CONV in per_frame pooled; do
  run_off bd_$CONV results/supp_bd_PAPER_$CONV.json \
      $PY -u scripts/bd_saving.py --curve "$CURVE" --convention $CONV \
      --out results/supp_bd_PAPER_$CONV.json
done

# paper_metrics.py rewrites results/bdrate.json in place. The old file has no
# provenance of any kind, so it is kept under a name that says what it is
# rather than overwritten silently.
if [ -s results/bdrate.json ] && ! grep -q '"sources"' results/bdrate.json; then
  mv results/bdrate.json results/bdrate_before_pinning.json
  log "bdrate: moved the unprovenanced file to results/bdrate_before_pinning.json"
fi
run_off bdrate results/bdrate.json $PY -u scripts/paper_metrics.py

# ---------------------------------------------------------------------------
# 4. What the routed decode delivers beyond the mean decibel: a perceptual
#    metric beside PSNR, and the per-tile tail the mean hides.
#
# The budget is denominated in decibels and the two conventions in use differ by
# roughly a third of it, so an independent metric is the cheapest answer to the
# reviewer who observes that PSNR superiority does not always accord with what
# is visible. The same pass ranks the worst tiles and names where they came
# from, which is what a failure-case section needs.
# ---------------------------------------------------------------------------
run opquality results/supp_opquality_PAPER.json \
    $PY -u scripts/operating_point_quality.py --ckpt "$CKPT" \
    --signalled "$SIGNALLED" --budget 0.1 --qps 0 16 32 48 63 \
    --worst 40 --device cuda:0 --out results/supp_opquality_PAPER.json

# ---------------------------------------------------------------------------
# 5. Composition with a second compression family.
#
# results/quant_BEST.json is the same experiment on runs/BEST/ckpt_eval.pth.tar.
# Re-measured here so it can be quoted beside everything else, and so the
# question of whether the two levers compose or compete is answered on the
# checkpoint the paper reports.
# ---------------------------------------------------------------------------
run quant results/supp_quant_PAPER.json \
    $PY -u scripts/quant_sweep.py --ckpt "$CKPT" --bits 8 6 4 \
    --qps 0 32 63 --images 32 --crop 512 --device cuda:0 \
    --out results/supp_quant_PAPER.json

# ---------------------------------------------------------------------------
# 6. The compute axis: batch size, and a second device class.
#
# Every latency number in this project is at batch 1 and nothing says so. The
# batch sweep makes the claim empirical rather than asserted. The CPU is the
# only second device class this machine can offer, and it is the end that
# flatters the method least, which is the reason to report it.
# ---------------------------------------------------------------------------
run latency_batch results/supp_latency_batch_1920x1080.json \
    $PY -u scripts/latency_device.py --ckpt "$CKPT" --curve "$CURVE" \
    --device cuda:0 --qps 0 32 63 --batches 1 2 4 \
    --width 1920 --height 1080 --warmup 20 --iters 40 --mem_fraction 0.16 \
    --note "shared card: four training runs live on other GPUs" \
    --out results/supp_latency_batch_1920x1080.json

run_off latency_cpu results/supp_latency_cpu_1920x1080.json \
    $PY -u scripts/latency_device.py --ckpt "$CKPT" --curve "$CURVE" \
    --device cpu --qps 0 32 63 --batches 1 --width 1920 --height 1080 \
    --warmup 1 --iters 5 --threads 8 \
    --note "8 threads on a shared machine; not an idle-box measurement" \
    --out results/supp_latency_cpu_1920x1080.json

# ---------------------------------------------------------------------------
# 7. What the encoder-side search costs, on the pinned checkpoint.
#
# The cost-of-tuning statement the supplementary is missing needs a priced
# search, not a claim that nothing was tuned. results/encoder_cost.json carries
# no checkpoint field; this one does.
# ---------------------------------------------------------------------------
run encoder_cost results/supp_encoder_cost_PAPER.json \
    $PY -u scripts/encoder_cost.py --ckpt "$CKPT" --qp 63 --iters 15 \
    --seq Bosphorus --device cuda:0 \
    --out results/supp_encoder_cost_PAPER.json

echo "=== supplementary queue 2 finished $(date '+%F %T') ===" >>"$LOG"
