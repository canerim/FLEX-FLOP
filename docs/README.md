# FLEX-UF — documentation

Content-adaptive early-exit decoding for the DCVC-UF intra decoder. The decoder
is split into a ladder of exits; each 256×256 tile of a frame leaves at whatever
depth its content needs, so an easy tile pays for a fraction of the network and
a hard tile pays for all of it. The encoder and the entropy model are frozen, so the coded payload is
bit-identical to Microsoft's release and a stock decoder can read a FLEX-UF
stream. The shipped configuration adds one small thing to the file: a 79–95 bit
per-frame exit map, 0.008–0.020% of the bitrate.

**Where it stands today (2026-08-19).** Against the released DCVC-UF decoder, at
a quality budget of 0.1 dB, on the full 53-sequence CTC set:

| qp | 0 | 16 | 32 | 48 | 63 |
|---|---|---|---|---|---|
| **A** signalled — decode MACs saved | 32.3% | 27.6% | 22.5% | 19.8% | 16.8% |
| **B** predicted, zero added bits | 30.6% | 25.9% | 18.1% | 9.1% | 3.5% |
| **bits**, no parameters and no bits | 29.9% | 24.6% | 19.9% | 15.4% | 12.1% |

> **These are MAC counts, not wall-clock.** Measured end to end the same
> allocation runs 29.1% / 22.4% / 15.6% faster at qp 0 / 32 / 63 against a
> 35.3% / 28.0% / 20.1% arithmetic prediction: the tiling machinery costs a fixed
> 3.6% and per-group bookkeeping the rest. Operations are an *optimistic* bound
> and the optimism grows with the saving. Numbers and method in
> [06 — CVPR plan](06-cvpr-plan.md); do not quote a MAC saving as a speedup.
>
> Everything timed before 2026-08-19 was measured with `torch.cuda.Event` on the
> wrong device and is wrong — see [DECISIONS 88](../DECISIONS.md).

The third row is a control, not a proposal: routing on the bits the entropy
model already spent per tile needs no parameters, no training and no added bits,
and it beats the 144 K learned head above qp 32.

Saving is divided by the cost of the released decoder. Results produced before
2026-08-17 divided by our own ladder at full depth instead and read 0.6–0.75
points higher; see [03 — Results](03-results.md#the-denominator-and-why-it-moved-every-number).

The A row costs 79–95 bits per frame, 0.008–0.020% of the bitrate; the other two
cost nothing and leave the file byte-identical. The three are not a ranking of
designs — A and B are the ends of one scale, and overriding the worst fifth of
tiles recovers 42–83% of the gap between them for 44 bits per frame.

Two configurations, one measurement each, are in
[09 — report §5](09-report.md); the mechanism is in
[08 — the system end to end](08-system.md).

## The documents

**The paper.** `paper/FLEX-UF.pdf` is the CVPR submission, built from
`paper/main.tex` and from tables that regenerate out of `results/`.
`paper/FLEX-UF-router.pdf` is a standalone explainer for how the exit map is
chosen on each side. Both rebuild with:

```bash
python scripts/make_paper_tables.py   # tables and macros from results/
python scripts/paper_figures.py       # figures into paper/figures/
python scripts/build_pdf.py           # the paper
python scripts/build_router_pdf.py    # the router explainer
python scripts/check_paper.py         # the numbers typed into the prose
python scripts/verify_theory.py       # the six structural propositions
```

**Start here:** [09 — Results and status](09-report.md) is the compiled report —
what was built, what it delivers, what it cost, and which numbers were wrong and
why. It regenerates from `results/` (`scripts/make_report.py`), so it cannot
drift from the measurements.

[08 — FLEX-UF end to end](08-system.md) is the single document
that covers the whole system — the data path, the exit adapters' internals, the
seam and how it is handled, both routing configurations, and how training works.
The others go deeper on one topic each.

| | |
|---|---|
| [09 — Results and status](09-report.md) | The compiled report: headline numbers, the operating-point structure, the seam, A vs B, and every correction that moved a number |
| [08 — The system, end to end](08-system.md) | Everything in one place, with diagrams: pipeline, adapter internals, seam repair, A vs B, the training objective |
| [01 — Experiment plan](01-experiment-plan.md) | The question, the protocol, the six runs, how the winner gets picked, what happens next |
| [02 — Architecture and accounting](02-architecture.md) | The exit ladder, what the router costs, and exactly how "compute saved" is computed |
| [03 — Results](03-results.md) | Rate–quality curves, the trade-off between dB given up and compute saved, training and convergence |
| [04 — Open questions](04-open-questions.md) | What is not yet explained, and the measurement that would settle it — including a router that has collapsed to a constant |
| [07 — The seam artefact](07-seam.md) | Where tiling damage comes from, the four fixes tried, the two kept, and what the repair module is actually worth |
| [06 — What this needs for CVPR](06-cvpr-plan.md) | An honest gap analysis: the two measurements that can still change what the paper claims, and the order to make them in |
| [05 — A versus B](05-decision-ab.md) | Encoder search against decoder prediction, compared on every axis: bits, compute on each side, optimality, deployability, and where the gap really comes from |

## Figures

Generated by `scripts/make_docs_figs.py` and `scripts/system_figs.py` from files
in `results/`, `docs/runs.json` and the checkpoints themselves; none is drawn by
hand — the seam-repair gate in `seam_module.png`, for instance, is read straight
out of `runs/BEST/ckpt_eval.pth.tar`.

| file | shows |
|---|---|
| `figures/topology.png` | the decoder, the split, the exits, what is trained |
| `figures/run_tree.png` | every training run, its lineage and its flags |
| `figures/rate_quality.png` | absolute PSNR against bitrate at three compute budgets |
| `figures/tradeoff.png` | dB against saving, read in both directions, with the ceiling |
| `figures/ab_decision.png` | encoder search vs decoder prediction, and what the unchanged bitstream costs |
| `figures/ab_budgets.png` | how the cost of not signalling shrinks as the budget loosens |
| `figures/latency.png` | MACs against measured wall-clock, and the price of speed |
| `figures/paper_rd.png` | rate-quality in the DCVC-UF paper's layout |
| `figures/paper_rdc.png` | BD-Rate against saving, and the complexity table drawn |
| `figures/training_BEST.png` | training loss, per-exit quality, measured convergence |

## Reproducing the figures

```bash
python scripts/snapshot_runs.py     # freeze the live training flags
python scripts/make_docs_figs.py    # redraw every figure
python scripts/paper_metrics.py     # BD-Rate and the DCVC-UF paper's tables
python scripts/check_docs.py        # do the numbers in these files still match results/?
```

`check_docs.py` exists because every table here was typed by hand from a
measurement, measurements are recomputed whenever a checkpoint lands, and
markdown does not notice. It recomputes the headline quantities and checks each
one still appears in the text.

`snapshot_runs.py` has to run while the jobs are alive: the training flags exist
only in the process argv, and `runs/<tag>/meta.json` records the model config but
not the recipe.

## A note on what these documents are

The numbers here are measurements, not targets. Where a result is a single
checkpoint, it says so; where a mechanism is not understood, document 04 says
that instead of offering a plausible story. Several explanations advanced during
this project turned out to be wrong and were withdrawn — the history is in
`DECISIONS.md`, which is kept as a record of what was believed and when, not
edited to look consistent in hindsight.
