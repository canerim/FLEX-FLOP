#!/bin/bash
# Measure every run at every epoch, for as long as it trains.
#
# Replaces on_first_ckpt.sh, which fired once and then stopped. That was fine
# when a checkpoint was expected within the hour; it is not fine now that the
# measured epoch length is 13-21 h and the numbered checkpoint is written only
# every 5th epoch. Over a 2-3 day budget no run reaches epoch 5, so
# on_first_ckpt.sh would have produced exactly one measurement per run and
# thrown away every later epoch.
#
# scripts/promote_ckpt.py rebuilds a proper evaluation checkpoint from the
# per-epoch resume file plus meta.json -- verified bit-identical to the numbered
# checkpoint (419 tensors, max|diff| = 0.0) -- so every epoch becomes readable.
#
# One test set, not two
# ---------------------
# Mid-epoch snapshots used a 10-sequence subset while epoch checkpoints got all
# 40, because a full evaluation took over half an hour and would have kept the
# card busy permanently. signalled_curve was then found doing 25x the work it
# needed to -- its lambda sweep re-decoded every frame for every lambda -- and a
# full run now takes minutes. The subset is dropped: two tiers meant two kinds
# of number in the same log, and the cheaper one existed only to work around a
# bug.
#
# Which card, and what it costs
# -----------------------------
# Evaluations share GPU2 with VERBATIM, and that is not free: VERBATIM's step
# time went from 0.377 s to 0.601 s once the chain started running regularly, a
# 60% slowdown taking its epoch from 5.0 h to 7.9 h.
#
# It is still the deliberate choice. Every card on this machine is occupied --
# six of ours and two other people's -- so the evaluation has to sit on top of
# some run, and VERBATIM has the most slack by a wide margin: even slowed it is
# the fastest of the six, against BEST's 21 h epoch. Putting the chain on GPU7
# would tax CONTROL at 1.0 s/step instead, for no gain.
#
# If this needs revisiting, the lever is --ckpt_every on the two runs that write
# step snapshots, not the size of the test set: a smaller test set was tried and
# removed, because two tiers of measurement meant two kinds of number.
#
# One evaluation at a time, machine-wide
# --------------------------------------
# Five watchers sharing one card would collide, and an OOM here would look like
# a failed experiment rather than a scheduling mistake. flock serialises them:
# whoever holds the lock evaluates, the rest wait their turn.
set -u
cd "$HOME/FLEX-UF"
TAG=${1:?usage: watch_ckpts.sh <run_tag> [gpu] [poll_seconds]}
GPU=${2:-2}
POLL=${3:-900}
D="runs/$TAG"
LOCK=/tmp/flexuf_eval.lock

# Check at STARTUP that this run's reference exists.
#
# FINE12's first evaluation failed two of three checks because the reference was
# hard-coded to the K=6 warm start and it trains a K=12 ladder. The failure was
# correct -- load_flexuf_state refuses a mismatched reference rather than
# measuring against the wrong decoder -- but it surfaced hours later, at the
# first checkpoint, after the run had already spent that time. Resolving it now
# turns a delayed surprise into an immediate one.
if [ -f "$D/meta.json" ]; then
  ./.venv/bin/python - "$D/meta.json" <<'PYEOF' || echo "  $TAG: reference check failed; its evaluations will not run until this is fixed"
import json, sys
sys.path.insert(0, ".")
from flexuf.config import FlexUFConfig
from flexuf.reference import reference_for
cfg = FlexUFConfig(**json.load(open(sys.argv[1]))["config"])
print(f"  reference: {reference_for(cfg)}")
PYEOF
fi

while true; do
  # A step snapshot, when the run writes them, is newer than any epoch file and
  # needs no promotion -- it already carries its own config.
  NEW=""
  # Rate limit on MID-EPOCH snapshots only.
  #
  # BEST128 and FINE12 each write one every 4000 steps, about every 1.2 h, and
  # the chain is now six stages taking 20-25 min -- so between them they were
  # asking for roughly two thirds of the single evaluation card, which VERBATIM
  # shares and has already slowed 60% for. Epoch checkpoints are never skipped:
  # they are the real measurement points and arrive every 13-21 h anyway.
  #
  # Done here rather than by raising --ckpt_every, which would mean restarting
  # two healthy runs and losing their progress.
  MIN_GAP=${MIN_GAP:-10800}
  SKIP=0
  if [ -f "$D/.evaluated_step" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$D/.evaluated_step") ))
    [ "$AGE" -lt "$MIN_GAP" ] && SKIP=1
  fi
  if [ -f "$D/ckpt_step.pth.tar" ] && [ "$SKIP" = "1" ] && \
     [ "$D/ckpt_step.pth.tar" -nt "$D/.evaluated_step" ]; then
    echo "$(date '+%F %T') $TAG: snapshot ready but last evaluation was ${AGE}s "\
         "ago (<${MIN_GAP}s); skipping to leave the card free"
  fi
  if [ -f "$D/ckpt_step.pth.tar" ] && [ "$SKIP" = "0" ] && \
     [ ! -f "$D/.evaluated_step" -o "$D/ckpt_step.pth.tar" -nt "$D/.evaluated_step" ]; then
    NEW="$D/ckpt_step.pth.tar"; MARK="$D/.evaluated_step"
  elif ./.venv/bin/python scripts/promote_ckpt.py "$D" >/dev/null 2>&1; then
    NEW="$D/ckpt_eval.pth.tar"; MARK="$D/.evaluated_epoch"
  fi

  if [ -n "$NEW" ]; then
    (
      flock -w 7200 9 || exit 1
      EP=$(./.venv/bin/python -c "
import torch,sys
c=torch.load('$NEW',map_location='cpu',weights_only=False)
print(f\"epoch {c.get('epoch')} step {c.get('step','-')}\")" 2>/dev/null)
      echo "=== $TAG  $(basename "$NEW")  $EP  @ $(date '+%F %T') ==="

      # anchor FIRST: if the deepest exit has drifted from released DCVC-UF,
      # every saving figure below is measured against the wrong reference.
      echo; echo "--- 1. anchor: still bit-comparable to released DCVC-UF? ---"
      ./.venv/bin/python scripts/anchor_drift.py --ckpt "$NEW" \
        --device "cuda:$GPU" --out "results/anchor_${TAG}.json" 2>&1 | tail -9

      echo; echo "--- 2. oracle ceiling (assumes a perfect router) ---"
      for q in 0 32 63; do
        ./.venv/bin/python scripts/oracle_diagnostic.py --ckpt "$NEW" --qp $q \
          --crop 512 --batches 8 --batch_size 4 --device "$GPU" 2>&1 \
          | sed -n '/HEADROOM/,/gain > 0/p' | sed "s/^/  qp$q  /" | head -10
      done

      # The deliverable: the system as it would ship, with the exit map's own
      # cost inside the bitrate.
      echo; echo "--- 3. SIGNALLED system, referenced to released DCVC-UF ---"
      SIG="results/signalled_${TAG}_$(date +%m%d_%H%M).json"
      # --frames 1 to match paper_curve below and the pinned reference
      # measurement. Their defaults differ (2 against 1), which would have put
      # a different frame count in each file -- exactly the mismatch that took
      # an afternoon to find when the two paths disagreed by 5.4 points, and
      # crosscheck_paths.py would rightly refuse to compare them.
      CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/signalled_curve.py \
        --ckpt "$NEW" --device cuda:0 --frames 1 \
        --out "$SIG" 2>&1 | tail -9

      # 4. The trade-off integrated over the curve rather than read at one
      # budget. A saving quoted at 0.1 dB is one sample of a frontier, and this
      # project has already been bitten by that: the grid-readout rule made a
      # test-set comparison look twice as large as it was. BD-saving and
      # BD-quality summarise the whole curve over stated intervals.
      #
      # Affordable only because signalled_curve's 25x-redundant sweep was
      # fixed; before that a single checkpoint took over half an hour.
      echo; echo "--- 4. frontier and the integrated trade-off ---"
      CURVE="results/curve_${TAG}.json"
      CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/paper_curve.py \
        --ckpt "$NEW" --device cuda:0 --out "$CURVE" 2>&1 | tail -8
      # The interval is PINNED, not derived.
      #
      # bd_saving's default lower limit is the largest per-rate floor of the
      # curve it is given. That makes each checkpoint's number internally
      # sound and mutually incomparable: BEST128's tighter anchor put its floor
      # at 0.033 where the reference sits at 0.064, so the two were integrated
      # over different intervals and BEST128 looked 1.1 points WORSE. Over the
      # same interval it is 0.6 points better. The mistake is the one this
      # script's own docstring warns about between rates, made between
      # checkpoints instead.
      #
      # And it is computed from every curve present, not fixed by hand.
      #
      # Fixing it at the baseline's [0.064, 0.30] broke the opposite way from
      # deriving it per curve. BEST's exits are good enough that its whole qp0
      # frontier spans only 0.197 dB -- its shallowest exit costs that much,
      # against the baseline's 0.309 -- so 0.30 ran off the end of the curve,
      # bd_saving returned n/a for qp0 and qp16, and the mean silently became a
      # mean over THREE rates compared against the baseline's five. The same
      # error as before, one level up.
      #
      # common_interval.py takes the largest floor and the smallest maximum over
      # every (curve, rate) present, so each curve spans it and no rate drops
      # out. A better decoder shortening the interval is not a defect: a
      # frontier reaching 42.5% saving for 0.197 dB is a shorter curve, and the
      # part they share is the only comparison available.
      # Recompute EVERY curve's BD, not just this run's.
      #
      # The interval is derived from all curves present, so it moves as runs are
      # added -- CONTROL got [0.064, 0.196] where the earlier four had
      # [0.066, 0.196], leaving five files that were each internally sound and
      # not quite comparable. bd_saving is arithmetic on JSON with no GPU in it,
      # so redoing all of them costs a second and keeps the set in step.
      read -r LO HI < <(./.venv/bin/python scripts/common_interval.py 2>/dev/null \
        | sed -n 's/.*--db_lo \([0-9.]*\) --db_hi \([0-9.]*\).*/\1 \2/p')
      if [ -z "${LO:-}" ]; then
        echo "  the measured curves share no dB interval — BD comparison skipped."
        echo "  (that is a result, not a failure: a run whose floor sits above"
        echo "   another's ceiling cannot be compared by integral. Quote per-rate"
        echo "   points for it instead.)"
      else
        echo "  BD interval common to every measured curve: [$LO, $HI]"
        for C in results/curve_*.json results/paper_curve_grid128.json; do
          [ -f "$C" ] || continue
          B="results/bd_$(basename "$C" .json | sed 's/^curve_//;s/^paper_curve_grid128$/saving/').json"
          ./.venv/bin/python scripts/bd_saving.py --curve "$C" \
            --db_lo "$LO" --db_hi "$HI" --out "$B" >/dev/null 2>&1
        done
        ./.venv/bin/python scripts/bd_saving.py --curve "$CURVE" \
          --db_lo "$LO" --db_hi "$HI" --out "results/bd_${TAG}.json" 2>&1 | tail -10
      fi

      # 5. Do the two paths still agree? They reach the same quantity through
      # different code, so a disagreement means one of them has a bug -- which
      # is the only check available that neither carries one the other lacks.
      echo; echo "--- 5. cross-check: two paths, one number ---"
      ./.venv/bin/python scripts/crosscheck_paths.py --curve "$CURVE" \
        --signalled "$SIG" 2>&1 | tail -10

      # 7 is placed before 6 in cost, not importance: it is a minute of work
      # and it is the measurement that settles a question the others cannot.
      #
      # CONTROL's shallow exits degraded over its second epoch on CTC AND on
      # held-out OpenImages, so it is not a distribution artefact. Whether the
      # additions PREVENT that is the missing control, and answering it needs
      # per-exit quality on held-out data tracked through training -- which is
      # this, collected per checkpoint from now on rather than reconstructed
      # afterwards from whatever checkpoints happen to survive.
      echo; echo "--- 7. per-exit quality on HELD-OUT OpenImages ---"
      CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/why_qp_val.py \
        --ckpt "$NEW" --device cuda:0 \
        --out "results/val_${TAG}_$(date +%m%d_%H%M).json" 2>&1 | tail -7

      echo; echo "--- 6. distance to the 30% target ---"
      ./.venv/bin/python scripts/target_gap.py --curve "$CURVE" \
        --out "results/target_gap_${TAG}.json" 2>&1 | tail -12
      touch "$MARK"
    ) 9>"$LOCK"
  fi
  sleep "$POLL"
done
