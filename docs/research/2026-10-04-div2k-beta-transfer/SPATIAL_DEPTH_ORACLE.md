# Where does a depth fallback need to act?

The frozen beta router, real FUFREF2 streams, epoch-15 early-exit checkpoint and
disjoint 24-image DIV2K validation cohort are unchanged. Exact tile context
without seam repair leaves **24/120 image–QP cases** above the predeclared
0.1 dB centred YCbCr444 loss cap against full-frame epoch-15 reconstruction.
This experiment asks how many of each failing image's six tiles must advance
**one exit group** to cross the cap. It is a source-informed counterfactual:
the original image determines the chosen tiles and is unavailable to a real
decoder. It is not a trained policy, a prospective test or a speed benchmark.

All 1-step subsets are reconstructed with the same released analysis/entropy
stream and clipped sufficient-context decoder. We first find the *smallest
number of upgraded tiles* that meets the cap and then select the greatest
ideal shared-context MAC saving among subsets of that cardinality. This is
not an unconstrained global optimum over arbitrary multi-step tile changes.
Inputs and checkpoints are SHA256-checked; the unchanged-map reconstruction
is matched to the archived exact-context result for each case.

| Minimum tiles advanced by one exit | Failing cases |
|---:|---:|
| 1 | 8 |
| 2 | 2 |
| 3 | 4 |
| 4 | 5 |
| 5 | 2 |
| 6 | 1 |
| Not feasible with +1 per tile | 2 |

Thus **22/24** failures can be rescued by a subset of +1 tile upgrades. The
two exceptions are 0844/QP32 and 0844/QP48; the separate uniform-depth
counterfactual verifies that both need a +2 exit increment. Among the 22
subset-feasible failures, selected maps retain **21.53% mean ideal synthesis
MAC saving**. Across all 120 validation cases, a hypothetical source-informed
fallback that keeps the original 96 safe maps, uses the selected +1 subsets
for 22 failures and uniform +2 for the two exceptions retains **19.05% mean
ideal synthesis MAC saving** with zero cap violations. The earlier
source-informed *uniform* +1/+2 fallback retains **17.89%**, so targeted
subsets add **1.17 percentage points** of arithmetic headroom. Unchanged maps
retain 20.55% but leave 24 violations. All-deep fallback on failures retains
14.96%.

These values charge each feature cell needed by multiple tile windows once
and assume a zero-overhead sparse scheduler. The current per-tile exact-window
implementation recomputes overlaps and does **not** realize these MAC savings
or demonstrate a latency gain. A deployed method still needs a reliable
decoder-visible predictor for *which* image and tiles require extra depth,
followed by fresh held-out quality and end-to-end latency tests. The negative
held-out latent-risk ablation means we cannot currently claim that predictor.

Reproduce the source-informed reconstructions with
`proof/cpu_early_exit/audit_single_tile_depth_oracle.py` against the archived
120-case exact-context JSON, then run
`python3 proof/cpu_early_exit/summarize_spatial_depth_oracle.py`. The latter
reads only JSON and writes `spatial_depth_oracle_summary.json` beside the
per-case file. Neither step modifies the D8/D10/D12 training jobs.
