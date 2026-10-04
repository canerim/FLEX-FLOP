# Early-exit BD proxy across six nominal budgets

[Vector plot](../../figs/bd_rate_budget_20261005/bd_rate_budget_sweep.pdf) ·
[Per-budget results](analysis.json) ·
[Reproduction script](../../scripts/bd_rate_budget_sweep_20261005.py)

This reuses the archived CTC first-frame reconstructions at QP 0/16/32/48/63
for nominal centred YCbCr444-loss targets 0.05, 0.10, 0.15, 0.20, 0.30 and
0.50 dB. The plotted cohort is the **same 25 sequences** with all five QPs
and all policies present at every target. The JSON also reports every complete
cohort at each target: 25/51/53/53/53/53 sequences. Thus the plotted changes
with budget cannot be explained by changing sequence membership.

At each sequence and budget, monotone PCHIP curves integrate log latent-payload
bpp over the common quality support of dense, uniform, dither, router and
oracle. Results average sequence-level percentages; uncertainty is 5,000
sequence-cluster bootstrap draws. The router-minus-dither difference is
paired by sequence. All policies reuse the same five actual round-tripped
latent payloads. Mode-map/container bytes are absent, and policy parameters
were calibrated with source-side quality feedback. The number is a **BD-rate
equivalent proxy**, not full-codec BD-rate or deployable inference performance.
The MAC model excludes the router, map signalling and execution overhead.

| Nominal target | Complete sequences | Router BD proxy | Router MAC saved | Router − dither BD proxy | Router − dither MAC saved |
|--:|--:|--:|--:|--:|--:|
| 0.05 dB | 25 | +1.30% | 19.21% | −0.050 points [−0.116, +0.009] | +2.12 points [1.24, 3.05] |
| 0.10 dB | 25 | +2.47% | 33.45% | −0.085 [−0.153, −0.028] | +2.39 [1.76, 3.05] |
| 0.15 dB | 25 | +3.30% | 37.56% | −0.036 [−0.066, −0.010] | +1.28 [0.88, 1.72] |
| 0.20 dB | 25 | +3.72% | 38.81% | −0.015 [−0.028, −0.005] | +0.32 [0.11, 0.56] |
| 0.30 dB | 25 | +3.84% | 39.12% | 0.000 | 0.00 |
| 0.50 dB | 25 | +3.84% | 39.12% | 0.000 | 0.00 |

On the *per-budget complete* cohort, the router–dither MAC margin is 2.12,
2.60, 2.02, 1.31, 0.38 and 0.02 percentage points respectively. The
0.10-dB cohort reproduces the earlier 51-sequence BD analysis to rounding:
router +2.26% proxy and 28.22% MAC saved; dither +2.38% and 25.63%.
The fixed-cohort gap vanishes at the loose targets because the archived
maps saturate at the shallowest available route on those 25 sequences. It
is not evidence that spatial routing never helps on other content or budgets.

The separate final D2/D4/D6 Kodak curves are **independent codec-bank models**,
not these early exits. They have estimated entropy rates rather than emitted
bitstream bytes; D8/D10/scratch D12 are still in training. Consequently the
depth-bank six-model BD plot and the shared early-exit full-codec BD plot are
not complete. Do not combine their percentages into one method ranking.
