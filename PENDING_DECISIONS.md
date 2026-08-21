# Two decisions waiting, both the author's

Neither is a defect and neither is urgent. Both would improve the paper, both
change numbers that appear in the abstract, and both are cheap to execute once
decided. They are here so they are not lost between sessions.

---

## 1. Move the paper onto a later checkpoint

**What is true now.** Every number comes from `ckpt_PAPER.pth.tar`, which is
RECIPE512 at the end of its first pass. On the hook count, the same 53 frames
and the same 0.1 dB budget, the run reads:

| epoch | mean saving |
|---|---|
| 0, the pinned checkpoint | 21.5% |
| 1 | 22.8% |
| 2 | 24.0% |
| 3 | 25.8% |

Monotone, 4.3 points. Epoch 4 is still in progress: at 2026-08-21 10:50 it was
at step 32,800 of 47,451, so the next point on this table is a few hours away
and the one after it is a day. `ckpt_eval.pth.tar` for this run was last
written at the epoch-3 boundary on 08-20 at 21:23; the 20-minute checkpoint
ages the heartbeat reports belong to the other three runs.

**Why it is a decision and not a fix.** Reporting a later checkpoint is not
cherry-picking, because there is no peak to pick; it is reporting the most
recent measurement of a run that is still improving. But every table, macro and
figure comes from one set of weights, so the whole chain has to be re-measured.

**Cost.** One command, a few hours of the evaluation card:
`scripts/repin.sh runs/RECIPE512/ckpt_PIN_e3.pth.tar`. Epoch 3 is already
snapshotted, so the option does not expire when the training loop overwrites
`ckpt_eval.pth.tar`.

**My correction on the record.** I argued against moving, on the grounds that
epoch 2 read below epoch 1 and picking epoch 1 would be picking a peak. That
reading came from files with no hook count, where the arithmetic model
over-reports by up to 0.8 points and not uniformly. DECISIONS 94 has it.

---

## 2. Make the bit rule configuration B

**What is true now.** Configuration B, the bitstream-identical mode, is the
trained 144,030-parameter head. The calibrated bit rule is presented as a
control that beats it. At the 0.1 dB budget:

| q | A, signalled | bit rule | trained head |
|---|---|---|---|
| 0 | 29.2 | **27.1** | 23.6 |
| 16 | 25.0 | **22.0** | 20.3 |
| 32 | 20.4 | **17.7** | 16.8 |
| 48 | 17.9 | **14.2** | 13.6 |
| 63 | 15.2 | **11.6** | 10.7 |
| mean | 21.5 | **18.5** | 17.0 |

**The oddity this leaves.** The paper reports, as its second deployment mode,
the second-best method available in that mode. A reviewer who reads Section 5.6
will ask why, and the honest answer is that the sections were written in the
order the work was done.

**What changes if it moves.** The bitstream-identical headline goes from 17.0%
to 18.5% mean, and from 23.6-10.7% to 27.1-11.6% across the rate range. The
trained head becomes what it measures as: a control the shipped rule beats at
five rates out of five, with 144,030 parameters against none learned.

**What does not change.** Both need a scalar calibrated offline, the rule a
multiplier and the head a tilt, and neither reads the source at decode time.
That symmetry is already stated in Section 3.5 and survives the swap.

**Cost.** No new measurement. It is a renaming and a reordering of Sections 5.5
to 5.7, and the numbers already exist in
`results/raterank_RECIPE512_b01.json`.

---

## 3. Hook-count the configuration C sweep

**What is true now.** Every saving in the paper is the hook count except one
table and one figure: the configuration C sweep, which was never re-measured
with hooks. Its ends therefore sit 0.43 to 0.78 points above the hook-counted A
and B they should reproduce, which is exactly the offset the reporting
conventions describe for the arithmetic model.

**Why it matters.** The paper claimed the ends "reproduce A and B to the second
decimal". That was true when all three were modelled and stopped being true when
A and B moved to the hook count. It now states the offset and its cause, which
is honest, but it is the only place in the paper where two conventions meet
inside one sentence.

**Cost.** One sweep: five rates by seven values of rho on the pinned checkpoint,
`scripts/hybrid_curve.py` with hooks on. A few hours of the evaluation card,
which is currently idle between checkpoint boundaries.

**Not done unattended because** it changes a published table and two macros, and
the author asked that the main experiment not be touched without a decision.

## 4. `ctca` is printed without authors

`@article{ctca}` in `paper/refs.bib` -- "Adaptive Test-Time Compute Allocation
for Reasoning LLMs via Constrained Policy Optimization", arXiv:2604.14853 --
has no `author` field, so it prints as reference [28] with the title first
while every other entry starts with a name. It is the only such entry.
`check_cites.py` reports it and does not fail on it, because the authors are
not something a build can look up and not something I will invent.

It is cited once, in Section 2, for the observation that the Lagrangian
relaxation has resurfaced for test-time compute in language models. Either
fill the author list in or drop the sentence; both are one line.

## 5. What SCRATCH105 does at epoch 90, when the recipe reaches 512x512

SCRATCH105 trains on the paper's own Eq. (6)-(7) objective -- a reconstruction
loss at every exit -- rather than on the random-depth objective every
warm-started run uses. It has to: random depth asks only that the mixed-depth
frame be good, nothing in it asks exit k+1 to beat exit k, and from random
initialisation the ladder inverted within three thousand steps. The per-exit
objective produced a monotone ladder within two hundred.

For the first 90 epochs this gives up nothing. Microsoft's schedule trains at
256x256, a 256 px tile in a 256 px crop is one tile, and patched training is
the same computation as full-frame. At epoch 90 the recipe moves to 512x512
and a crop becomes four tiles, so from that point the two objectives really
differ and the mixed-depth condition -- a tile at exit 2 beside one at exit 5,
which is what deployment produces -- stops being trained.

Three options, none of them free:

  * stay on Eq. (6)-(7) for all 105 epochs, and accept that the seam between
    tiles of different depth is handled by the architecture (grid seam repair,
    replicate padding) rather than by the objective;
  * switch to --train_patched at epoch 90, which trains the deployed condition
    for the last fifteen epochs on a ladder that is by then established, and
    is the closest thing to what the warm-started runs do;
  * add both terms, which is a third objective neither the paper nor Microsoft
    describes.

The second is what I would choose, and it is eight days away, so it is written
down rather than decided. Whichever is picked, the run's meta.json and the
supplement's recipe table have to say which objective trained which epochs.

## 6. Two figures still carry the old type, and regenerating them moves a checkpoint

The type in every figure went up a point, but a figure only gets the new style
when it is regenerated, and two of them cannot be regenerated safely.

`exit_map.png` and `seam_PAPER_bosphorus_q63.png` are drawn from
`runs/BEST/ckpt_eval.pth.tar`. BEST has no pinned checkpoint -- `ckpt_eval` is
a name the watcher rewrites, and BEST has trained five more epochs since those
figures were made. Regenerating them would silently replace the checkpoint
behind two published pictures, which is the class of defect DECISIONS 106 is
about, and it would do it to make the labels one point bigger.

The other five unaudited figures need nothing: `flexuf_overview` and
`dcvcuf_framework` are drawings rather than plots, `exitmap_PAPER_*` sets its
own type at 8 to 11 pt, and `budget_band_BEST` is a supplement figure on the
same BEST problem.

Two ways out, both the author's call: pin a BEST checkpoint and record which
epoch it is, so those two figures have a reference that stays still; or accept
that they are a point smaller than the rest and say so nowhere, because nobody
will notice a single point on an exit map.
