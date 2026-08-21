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
