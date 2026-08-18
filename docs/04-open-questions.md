# Open questions

## 1. CONTROL degrades over its second epoch. Nobody knows why.

Measured on the same fixed CTC frames, compute saved at 0.1 dB, qp 63:

| | epoch 0 | epoch 1 |
|---|---|---|
| CONTROL | 4.25% | 2.47% |

Per exit, the change from epoch 0 to epoch 1 (positive = worse, dB above the
released decoder):

| qp | run | exit 2 | exit 3 | exit 4 | exit 5 |
|---|---|---|---|---|---|
| 0 | VERBATIM | −0.025 | −0.011 | −0.023 | −0.026 |
| | CONTROL | **+0.290** | +0.258 | +0.123 | +0.002 |
| 32 | VERBATIM | −0.043 | −0.033 | −0.034 | −0.033 |
| | CONTROL | **+0.478** | +0.393 | +0.163 | −0.004 |
| 63 | VERBATIM | −0.055 | −0.056 | −0.049 | −0.044 |
| | CONTROL | **+0.664** | +0.572 | +0.201 | −0.013 |

The pattern is sharp: the damage is concentrated in the **shallow** exits, scales
with rate, and leaves the deepest exit essentially untouched. `VERBATIM` improves
slightly over the same interval.

### What has been ruled out

**It is not the ladder collapsing toward the deepest exit.** The training-batch
spread between exits did not move.

**It is not overfitting to the training distribution.** The same measurement on
512 held-out OpenImages — same distribution, images the model has never seen —
degrades too:

| qp | held-out OpenImages | CTC video |
|---|---|---|
| 0 | +0.146 | +0.290 |
| 32 | +0.284 | +0.478 |
| 63 | +0.513 | +0.664 |

So it is not a failure to transfer from photographs to video frames. Whatever is
being lost is lost on the training distribution itself.

**A predicted mechanism was falsified.** A pre-registered prediction that
`VERBATIM` would collapse harder than `CONTROL`, for the same reason, was
recorded before the measurement. `VERBATIM` improved instead. The explanation was
withdrawn from `DECISIONS.md` and from the presentation rather than adjusted to
fit.

### Candidate causes still standing

`CONTROL` and `VERBATIM` differ in six flags, so the comparison isolates nothing
on its own:

| | CONTROL | VERBATIM |
|---|---|---|
| `--new_lr_scale` | 20 | — (base lr) |
| `--anchor_weight` | 1.0 | — (no anchor term) |
| `--seam_repair` | grid | none |
| `--min_crop` | 512 | — |
| `--epoch_offset` | 75 | 90 |
| effective batch | 8 | 16 (`--grad_accum 2`) |

`seam_repair` also makes the two runs' savings differently normalised: VERBATIM's
deepest exit costs 1.0 stock decodes and CONTROL's 1.0095, so any saving
comparison between them must use the release denominator or it favours CONTROL
by ~0.9 points. The per-exit dB table above is unaffected — those are decibels,
not ratios of compute.

The leading candidate is `--new_lr_scale 20`: newly-initialised adapters and the
seam-repair path train at twenty times the base learning rate, and the shallow
exits are the ones that depend on those adapters most — which matches where the
damage lands. But this is a hypothesis chosen because it fits the pattern, and
patterns have fit wrong hypotheses three times already in this project.

### The measurement that discriminates

`BEST` also runs with `--new_lr_scale 20`, but with `--anchor_weight 10` instead
of 1.0. Its epoch-1 checkpoint is the missing control, and it is due within a day:

- **If BEST degrades the same way**, a 20× learning rate on the adapters is
  implicated and a strong anchor does not rescue it.
- **If BEST holds or improves**, the 20× rate is not sufficient on its own, and
  the weak anchor (1.0) is implicated instead.

Either outcome is informative, which is why it is worth waiting for rather than
launching a dedicated ablation. Note that it cannot cleanly isolate the learning
rate by itself — `BEST` differs from `CONTROL` in four flags, not one.

## 2. Can the decoder route without being told?

`BEST`'s jointly-trained router picks one exit per qp and ignores the content
entirely: agreement with the oracle is 0.000 at qp 63, where it sends 239 of 240
tiles to exit 4 and the oracle sends all 240 to exit 2. So the zero-added-bits
configuration — bitstream byte-identical to a stock stream, decoder decides for
itself — currently does not work at all.

This is a training failure rather than a limit on the information available, and
it is specific to **that** router rather than to routing. Three routers, three
outcomes, all measured:

| router | outcome |
|---|---|
| BEST, trained jointly | collapsed to a qp-dependent constant, agreement 0.000 at qp 63 |
| FINE12, trained jointly | **did not collapse** — budget met at every rate, β ≈ −0.01 |
| retrained against a FROZEN decoder | 0.848–0.923 held-out agreement, budget met at every rate |

So joint training is not automatically fatal: FINE12 does it and survives. Why
BEST's collapsed and FINE12's did not is open — they differ in K, j, tile size
and adapter kind, so the comparison isolates nothing.

**Answered.** A `StemRouterHeadV2` retrained against BEST's *frozen* decoder
reaches 0.848 held-out agreement and hits the budget at every rate: 30.07% saved
at qp 0 falling to 14.31% at qp 63, against the signalled system's 34.04%/20.80%.
So an unchanged bitstream costs 3.3–6.5 points, and the 30% target is met at the
lowest rate with a byte-identical file. Details in
[03 — Results](03-results.md#two-ways-to-decide-and-what-the-difference-costs).

**Still open: why the gap widens with rate.** Measured at matched quality, the
router's prediction loss *falls* almost tenfold from qp 0 to qp 63 (7.23 → 0.80
points, agreement 0.554 → 0.861) while the budgeted gap *rises* (3.96 → 6.49).
The gap instead tracks |β|, the tilt needed to drag the router from its single
training λ to the λ that lands on 0.1 dB. The hypothesis is that a large tilt
lets the cost term dominate the logits and discards the content ranking.

A prediction was registered before the test was run (`DECISIONS.md` 63a): a
router trained at λ = 4.1 × 10⁻⁶, the rate-appropriate value for qp 63, should
bring the 6.49-point gap **below 3 points**. If it stays above 5, the mechanism
explanation is wrong and will be withdrawn. Four mechanism hypotheses in this
project have already been wrong, which is why the prediction is written down
first.

## 3. Does the gain continue past four epochs?

The measured saving is still climbing at every checkpoint with more than one
measurement (BEST128: 11.23% → 12.56% over 8k–20k steps; FINE12: 14.96% → 16.76%
over 12k–24k). Four epochs is a selection point chosen for scheduling reasons,
not a point where anything is known to converge. The training loss is no help
here — it is flat from step 5,000 and correlates 0.97 with the batch's bitrate.

## 4. Why does saving fall so steeply with rate?

From 34.0% at qp 0 to 20.8% at qp 63. The plausible reading is that
high-rate reconstructions carry detail the shallow exits cannot reproduce, so
fewer tiles can leave early. That is consistent with the floor rising with rate
(−0.002 dB at qp 0 to +0.030 dB at qp 63) and with the convexity margin growing
tenfold. It has not been tested against alternatives — for instance, that the
shallow exits are simply under-trained at high qp because the training λ range
weights low rate more heavily.

## 5. Is `K=6, j=2` the right ladder, given FINE12's higher ceiling?

FINE12 (`K=12, j=4`) and BEST (`K=6, j=2`) share a 4-block full-frame stem, but
FINE12's architectural ceiling is 50.3% against BEST's 41.9%: with K = 12 there
is one block per exit, so its shallowest usable exit runs 5 of 12 blocks where
BEST's runs 6.

Both ceilings ARE reachable, contrary to what this file said before: BEST hits
41.9% at 0.167 dB at qp 0, rising to 0.586 dB at qp 63 (`curve_BEST.json`). That
is what makes the finer ladder interesting rather than academic — once a budget
is loose enough to saturate K = 6, the only way to spend more is a rung that does
not exist there.

The promise is untaken so far: the measured means are 27.18% (BEST) and 26.93%
(FINE12), i.e. the 8.4-point ceiling advantage has produced no measured
advantage at all. FINE12 is behind on measured saving today (16.76% vs 20.80% at qp 63) and has
had less than half the training.

**A prediction here is currently unsupported.** The finer ladder was expected to
help *most* at high rate, because K=6's rungs straddle the budget there (41.9%
saving is five times over budget at qp 63, 27.0% is affordable, and K=12 offers
34.5% in between). If that were the dominant effect, FINE12's saving profile
should be flatter across rate than BEST's. It is steeper: qp63/qp0 = 0.521 for
FINE12 against 0.611 for BEST. The comparison is confounded by training time —
0.58 epochs against 1.38, and undertrained shallow exits hurt most at high rate —
so this falsifies nothing yet. It is recorded so the four-epoch comparison is
read as a test of a stated expectation rather than as a fresh look.

## 6. Is 256px the right tile size?

`BEST128` tests 128px tiles: four times as many tiles, finer routing, but a
larger seam-repair burden and a worse halo ratio. The two cannot yet be
compared — BEST has one measured checkpoint (end of epoch 0, 47k steps) and
BEST128 has three, none past 20k steps, so there is no matched-step pair. Four
epochs each will settle it.

---

## How to read this file

Every claim above is either a measurement with its source, or explicitly labelled
as a hypothesis. That distinction is deliberate. The mechanism behind item 1 has
been asserted three times in this project and corrected three times; the current
state of knowledge is that the *phenomenon* is solid and reproducible and the
*explanation* is unknown. `DECISIONS.md` keeps the full history, including the
withdrawn claims, rather than being edited to look consistent afterwards.
