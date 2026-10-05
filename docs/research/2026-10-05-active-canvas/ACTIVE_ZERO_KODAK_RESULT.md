# Stage-synchronous active-zero context: complete Kodak replay

The active-only zero-fallback arm completed all 120 frozen Kodak24 × QP5
FUFREF2 replays. The same release and e15 checkpoints, latent streams,
DIV2K-calibrated exit maps, source images, and full-frame reference are used
in every paired comparison. This is a CPU FP32 reconstruction experiment,
not a native CUDA or wall-clock measurement. The rule was designed after a
five-case pilot, so Kodak is exploratory; the disjoint DIV2K validation and
active-replicate comparator remain pending.

| Comparison with deployed isolated-tile route | No seam repair | Same trained seam repair |
|---|---:|---:|
| Mean Δ444 PSNR gain; positive is better | +0.01180 dB | +0.00940 dB |
| 95% interval from 10,000 image-cluster draws | [+0.00871, +0.01513] dB | [+0.00574, +0.01315] dB |
| Mean YUV 6:1:1 PSNR gain | +0.01408 dB | +0.00969 dB |
| Paired cases better / worse | 112 / 8 | 105 / 15 |
| Cases >0.1 dB below full-frame e15 | 35 / 120 | 36 / 120 |

The deployed route has 47/120 cases above that threshold; the exact-context
counterfactual has 34/120. The active-zero arm with existing repair recovers
0.00940/0.01488 = **63.2% of the mean deployed-to-exact Δ444 gap**. This is a
ratio of cohort means, not a per-image guarantee. Fifteen cases still lose
quality relative to deployed; the worst paired loss is 0.03776 dB.
The 47→36 threshold change consists of **11 previously failing cases rescued
and zero newly failing cases** on this frozen cohort. Without repair it is
12 rescued and zero newly failing. This is a descriptive paired count, not a
guarantee that the rule never creates threshold violations on new images.

On identical maps and streams, the active-zero arm with repair is +0.07513 dB
better than the legacy stale canvas on average. The 23 uniform-depth maps
are identical between these two canvas rules; all difference comes from the
97 mixed-depth maps, where the mean is +0.09294 dB. The latter average is
affected by severe stale-canvas failures at QP63 and should not be read as
the expected gain over the deployed decoder. Against deployed, the active
arm's mean gain is positive at each of the five QPs (from +0.00608 dB at
QP0 to +0.01069 dB at QP16 with repair).

The current implementation assembles halo buffers in PyTorch. Its extra
3×3 depthwise arithmetic is only 0.01087 percentage points of full synthesis
conv-MAC on these frozen maps, but this excludes memory traffic and kernel
launches; no runtime advantage is established. Full-codec BD-rate, active
replicate comparison, and DIV2K transfer are required before a paper method
claim. The frozen zero-fallback prototype also has a synthetic NaN-masking
robustness issue; finite-input equivalence to a safe variant has been tested,
but the raw cohort provenance deliberately points to the unchanged replay
implementation.

Raw cases: [`kodak_active_canvas_20261005.json`](../../../proof/cpu_early_exit/results/kodak_active_canvas_20261005.json).
Clustered summary: [`kodak_active_canvas_20261005_summary.json`](../../../proof/cpu_early_exit/results/kodak_active_canvas_20261005_summary.json).
Protocol: [`PROTOCOL.md`](PROTOCOL.md).
