# Active-neighbour fallback transfer: complete DIV2K validation24

Both active-neighbour fallback arms completed 24 DIV2K validation crops at
five QPs (120 cases each). The paired audit requires each arm to match the
archived source case, FUFREF2 stream hash, checkpoint, exit map, deployed
reference and exact-context reference. The DIV2K calibration beta and all
routes are frozen. Each arm uses the same stage-synchronous real feature
from a neighbour while that neighbour remains active. When it exits,
active-zero inserts zero and active-replicate uses the current tile's own
edge value, matching deployed internal-tile replicate padding.

The **predefined primary comparison** holds the trained seam repair fixed.
Active-replicate gains **+0.002618 dB** mean YCbCr 4:4:4 PSNR over active-zero
(24-image clustered 95% interval **[+0.000824,+0.005275] dB**). It improves
65 cases, worsens 14 and ties all 41 uniform-depth maps. The Kodak24
five-QP cohort had shown a smaller but same-direction +0.001727 dB
([+0.000890,+0.002724] dB) with 81 improvements, 16 regressions and 23
uniform-map ties. The fallback effect transfers across these two cohorts,
but its absolute quality increment is small.

| DIV2K decoder path | Mean 4:4:4 PSNR gain over deployed | 95% image-cluster interval | Cases >0.1 dB below full-frame e15 |
|---|---:|---:|---:|
| Deployed isolated tiles, trained repair | anchor | — | 35/120 |
| Active-zero, same repair | +0.00646 dB | [−0.00002,+0.01255] dB | 31/120 |
| Active-replicate, same repair | **+0.00908 dB** | **[+0.00456,+0.01388] dB** | **30/120** |
| Active-zero, repair removed | +0.01695 dB | [+0.01156,+0.02282] dB | 24/120 |
| Active-replicate, repair removed | **+0.01969 dB** | **[+0.01464,+0.02519] dB** | **24/120** |
| Exact-context diagnostic, repair removed | see original exact audit | — | 24/120 |

The repair-off comparisons are exploratory because the repair interaction
was highlighted after the active-zero validation arm completed. Within
active-replicate, removing repair raises mean PSNR by +0.010618 dB
([+0.005095,+0.017316] dB), improving 90 cases and worsening 30. The
repair-off replicate arm rescues 11 of the deployed path's over-threshold
cases and creates none. Directly against repair-off active-zero, replicate
gains +0.002744 dB ([+0.000868,+0.005653] dB). The repair-off arm changes
both context and repair relative to deployed, so it **cannot** isolate the
context contribution until the isolated-tile repair-off control finishes.

The per-QP transfer figure shows that both the fallback increment and the
old repair penalty tend to grow toward QP63. This describes these fixed
images and routes; it does not prove a general rate law. DIV2K validation
also informed earlier exploratory project work, so it is disjoint from the
current calibration24 cohort but not an untouched benchmark. No native
DCVC-UF bitstream, YUV BD-rate, GPU/CPU latency or new training result is
inferred from these scalar Delta444 records. The two active fallbacks have
the same analytical depthwise-halo convolution count; memory and packing
cost could differ. Active-replicate is the stronger *quality candidate* for
the next matched timing and repair-off control, not a deployment verdict.

Sources: [replicate raw replay](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_replicate.json),
[hash-linked summary](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_replicate_summary.json),
[paired comparison](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_transfer_comparison.json),
and [vector figure](../../figures/active-canvas-transfer-20261005/canvas_transfer.pdf).
