# How much compute remains if exact tile context is shared perfectly?

The static halo experiment repeats overlapping suffix work per tile. A
blockwise taper discards dependency rings as soon as they are no longer
needed and matches static-halo output exactly, yet costs 1.151× full synthesis
MAC on `0828/QP63`'s fixed map. We therefore computed an **optimistic
feature-cell sharing estimate**: at suffix block `i`, take the union of the
spatial cells required by all tile cores at their chosen exit depths. Charge
each distinct cell for that block only once. Add the actual shared stem,
full-frame head, and core-only exit adapter costs. This assumes a hypothetical
masked scheduler executes exactly those cells with zero packing, control,
memory, or launch overhead. It is **not measured latency**, nor a general
lower bound over all possible decoder designs.

| Real bitstream/crop | Fixed map | Deployed Δ444 | Exact-context Δ444 | Optimistic shared-context MAC / full e15 |
|:--|:--|--:|--:|--:|
| 0828/QP0 | 3/3/3/2/3/3 | 0.7918 dB | 0.7202 dB | 0.7257 |
| 0828/QP63 | 5/5/5/3/5/5 | 0.1565 dB | **0.0159 dB** | **0.9655** |
| 0844/QP48 | 2/2/2/2/3/3 | 0.4961 dB | 0.4647 dB | 0.6525 |
| 0898/QP48 | 3/5/5/5/5/3 | 0.1080 dB | **0.0613 dB** | **0.9309** |

All-deep maps require the full frame at all eight suffix blocks, giving
relative MAC **1.0000** and zero exact-context loss on each case. The result
reveals the central quality–compute conflict in these examples: the shallow
maps retain 27–35% ideal synthesis savings but have substantial quality loss
even with perfect context; the quality-preserving, mostly-D12 maps leave only
3.5–6.9% ideal synthesis saving. The real implementation can be slower than
this optimistic estimate, especially for narrow sparse regions on GPU/CPU.
The current zero-halo decoder's deployed cost is a different implementation
with seam-repair overhead and cannot be compared as measured latency.

### Frozen policy across 48 disjoint DIV2K images

We applied the same cell-union accounting to **all 240 fixed-map cases** in
the 24-image calibration and 24-image validation cohorts. There was no new
decoder execution. Mean ideal synthesis MAC saving is **21.30%** on calibration
and **20.55%** on validation; 90/120 and 91/120 maps respectively leave at
least 10% ideal arithmetic saving. Thus the `0828/QP63` map's 3.5% is an
outlier, not a representative estimate of the whole policy.

| QP | Calibration mean ideal saving | Validation mean ideal saving |
|--:|--:|--:|
| 0 | 32.13% | 33.45% |
| 16 | 21.75% | 22.82% |
| 32 | 21.84% | 21.18% |
| 48 | 17.69% | 16.30% |
| 63 | 13.06% | 9.00% |

The decline with QP follows the frozen router's deeper decisions. On
validation, maps whose **deployed zero-halo** reconstruction already meets
the 0.1 dB target have 18.61% mean ideal saving (85 cases); the 35 maps that
violate the target have 25.28%. This grouping is deliberately labelled by
*deployed* quality. We did **not** measure exact-context quality across these
240 cases, and perfect context can change distortion in either direction.
The observed association says that the most aggressive maps tend to be the
risky ones; it does not prove an exact-context decoder would retain the same
quality classification.

The vector figure `ideal_context_cohort.pdf` shows the QP trajectory with
24-image cluster-bootstrap intervals and the validation quality groups.
`analyze_ideal_context_cohort.py` verifies case/manifest/policy hashes and
archives every per-block union count in `ideal_shared_context_cohort.json`.

For reproducibility, `proof/cpu_early_exit/analyze_ideal_shared_context.py`
reads the archived case-level exact-context JSONs and e15 config, constructs
one 64×96 Boolean feature mask for each suffix block, and archives the eight
union-cell counts and costs in
`proof/cpu_early_exit/results/div2k_beta/quality_floor/ideal_shared_context_bound.json`.
No GPU model execution or timing enters this analysis.
