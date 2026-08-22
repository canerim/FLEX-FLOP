# Moving the pin, and what has to move with it

`runs/RECIPE512/ckpt_PAPER.pth.tar` is the checkpoint every number in the
paper is measured on. Moving it is not one command; it invalidates about
fifty result files, four figures whose captions quote them, a dozen derived
tables and two documents. This is the order that worked for the epoch 0 to
epoch 4 move, and the traps that cost the most time.

## Before anything

    cp runs/RECIPE512/ckpt_<new>.pth.tar runs/RECIPE512/ckpt_PAPER.pth.tar
    ./.venv/bin/python scripts/check_epoch.py

`check_epoch` classifies every file the documents read. Everything it calls
stale has to be re-measured; everything it calls unlabelled has to be
stamped or re-measured. Two more classes it checks are easy to miss:

  * **a head off the pin** -- a file measured on the pinned decoder but with
    a router head fitted to other weights. Re-fit the head first
    (`scripts/train_router2.py --ckpt <pin> --lam 1.3e-5 --steps 3000`),
    then re-measure everything that names it. Configuration B, the hybrid
    curves and the held-out beta table all do.
  * **derived** -- `check_derived_fresh.py`, run from the gate, compares a
    derived file's mtime against the files it is computed from. The Pareto
    enumeration, the log-convexity check and the tile-size splice are all
    arithmetic over `results/tile_table.json`, and none of them notices when
    the table is re-measured.

## The stages

Each is idempotent, marker-guarded, and takes the GPU lock
(`/tmp/flexuf_eval_gpu0.lock`) so two of them can be queued without
treading on each other. `TMP_DIR` must be set; a step writes there and is
moved into `results/` only when it exits zero and non-empty.

    TMP_DIR=... bash scripts/repin_stage2.sh   # the sweeps the tables read
    TMP_DIR=... bash scripts/repin_stage3.sh   # the supplement's own files
    TMP_DIR=... bash scripts/repin_stage4.sh   # curve, hybrid C, why_qp
    TMP_DIR=... bash scripts/repin_stage5.sh   # the ten-budget band grid
    TMP_DIR=... bash scripts/repin_stage6.sh   # the jointly trained head

Then the ones that are not in a stage because they were only found by
`check_epoch` on the day:

    scripts/reference_gap.py      scripts/seam_ring.py
    scripts/halo_exactness.py     scripts/qualitative.py --qp 32 and 63
    scripts/dump_tile_table.py    scripts/rd_absolute.py
    scripts/beta_calibration.py --router2 <the re-fitted head>
    scripts/router_ablation_e4.sh   (six variants, about six hours)

## Then

    scripts/make_paper_tables.py && scripts/paper_figures.py
    scripts/build_pdf.py && scripts/build_supp_pdf.py
    scripts/check_paper.py

The gate is green when every sub-check passes AND the claim count reads
n/n. `check_paper.py`'s expectations are typed on purpose: a claim that
goes stale is the point, and each one has to be looked at rather than
bulk-updated.

## What cost the most time, in order

1. **A driver from the previous chain was still alive.** It rewrote three
   configuration-B curves with the old head at 13:28 and, because both
   readers ranked candidates by "on the pin, then newest", the stale file
   won on recency and took the headline. Hand order is the last key now,
   but check `ps -ef | grep repin_` before starting.
2. **Numbers typed into prose.** Nine were wrong after the move, including
   a figure caption off by a factor of two and a proposition's headroom off
   by three. `check_numbers.py` and the `lint:` line in `check_paper.py`
   list every numeric literal in `main.tex`; work through them.
3. **Files exempted as structural that are not.** `tile_table.json` was
   "tile geometry" and its D matrix is a measurement; `rd_absolute_PAPER`
   was "release PSNR" and two of its four columns are ours. Read the
   exemption list against what the files actually contain.
4. **Paths the sweep could not see.** `R / "results/x.json"` has a slash in
   it and the name pattern did not. Two files feeding the qualitative
   figures were invisible for that reason.
