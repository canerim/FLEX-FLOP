#!/bin/bash
# Re-measure the paper's four tables on the full CTC set, on the SAME checkpoint.
#
# Until MCL-JCV landed, the test set was 10 sequences -- 7 UVG and 3 HEVC class
# E -- and the paper's central empirical claim (the adaptivity gain Delta/J
# rising from 0.68% to 4.52% with rate) rested on those. MCL-JCV adds 30, so
# the same claim can be made on 40.
#
# The checkpoint is pinned deliberately. If the model changed at the same time
# as the test set, a change in the numbers could not be attributed to either.
# So: same weights, same QPs, same frames per sequence, only the sequence list
# differs.
#
# Order matters -- logconvex_check.py reads paper_curve_grid128.json, and
# theory_check.py reads it too, so the curve is regenerated first.
set -eu
cd "$HOME/FLEX-UF"
GPU=${1:-2}
CK=runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar
[ -f "$CK" ] || { echo "pinned checkpoint $CK is gone; refusing to substitute another"; exit 1; }

N=$(ls -1 /data10/shareddata/test_sequences/YUV/MCL-JCV/*.yuv 2>/dev/null | wc -l)
echo "=== re-measure on the full CTC set (MCL-JCV: $N sequences present) ==="
echo "    checkpoint pinned to $CK"

# Serialise against the training-run watchers, which share this card.
exec 9>/tmp/flexuf_eval.lock
flock -w 14400 9 || { echo "could not take the evaluation lock"; exit 1; }

echo; echo "--- 1/4  paper_curve (the frontier; Table 3 and everything downstream)"
CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/paper_curve.py \
  --ckpt "$CK" --device cuda:0 --out results/paper_curve_grid128.json 2>&1 | tail -12

echo; echo "--- 2/4  why_qp (Table 1: per-exit dB and bpp)"
CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/why_qp.py \
  --ckpt "$CK" --device cuda:0 --out results/why_qp.json 2>&1 | tail -10

echo; echo "--- 3/4  theory_check (Table 2: the adaptivity gain)"
CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python -u scripts/theory_check.py 2>&1 | tail -12

echo; echo "--- 4/4  logconvex_check (Table 4: convexity of dB(S))"
./.venv/bin/python -u scripts/logconvex_check.py 2>&1 | tail -10

echo; echo "=== rewriting the tables, and testing the claims made about them ==="
# Non-zero here means a sentence in the paper no longer follows from the data.
# Reported, not patched: the tables are rewritten either way, and the prose is
# left for a human, because a caption is an argument rather than a cell.
./.venv/bin/python scripts/refresh_theory_tables.py || true
./.venv/bin/python scripts/verify_theory_tables.py

echo; echo "=== figures and deck, rebuilt from the new results ==="
./.venv/bin/python scripts/make_figs_nature.py 2>&1 | tail -2
./.venv/bin/python scripts/make_deck.py 2>&1 | tail -1
echo "done -- review refresh_theory_tables' warnings before committing the paper"
