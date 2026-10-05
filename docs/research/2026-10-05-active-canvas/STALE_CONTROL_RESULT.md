# Frozen Kodak control: stale neighbour features fail

The legacy `CanvasCoupler` reuses a neighbour's last feature after that tile
has exited. On the frozen Kodak24 × five-QP FUFREF2 cohort, this worsens the
paired reconstruction relative to the deployed isolated-tile decoder. This is
a **negative control**, not a proposed method or a latency measurement.

| Measure | No seam repair | Existing seam repair |
|---|---:|---:|
| Mean Δ444 gain over deployed; positive is better | −0.0621 dB | −0.0657 dB |
| 95% interval, resampling 24 images with their five QPs | [−0.1184, −0.0280] dB | [−0.1228, −0.0310] dB |
| Mean YUV 6:1:1 PSNR gain over deployed | −0.0443 dB | −0.0495 dB |
| Cases with Δ444 > 0.1 dB | 55/120 | 56/120 |

The deployed decoder has 47/120 cases above that Δ444 threshold. The
full-frame exact-context counterfactual has 34/120, so context remains a
plausible source of recoverable quality; the legacy implementation does not
recover it reliably.

The distinction between exit maps is sharp. Among the 23 uniform-map cases,
legacy no-repair coupling gains 0.0167 dB on average. Among the 97 mixed-map
cases it loses 0.0808 dB. At QP63 the mean no-repair gain is −0.2863 dB;
`kodim23` loses 3.2128 dB. These are paired comparisons with identical
checkpoint, exit map, and bitstream, so the regression is attributable to the
changed inference path, though this cohort alone does not isolate which
individual halo reads cause each failure.

The uniform/mixed comparison is also visible **within each QP**, rather than
only in an aggregate where the QP mix differs:

| QP | Uniform maps: count, mean gain | Mixed maps: count, mean gain |
|---:|---:|---:|
| 0 | 8, +0.0078 dB | 16, +0.0023 dB |
| 16 | 7, +0.0169 dB | 17, +0.0039 dB |
| 32 | 5, +0.0176 dB | 19, −0.0005 dB |
| 48 | 1, +0.0071 dB | 23, −0.0415 dB |
| 63 | 2, +0.0542 dB | 22, −0.3172 dB |

The uniform strata at QP48/63 contain just one/two cases and should not be
interpreted as precise population effects. The pattern nonetheless identifies
mixed-depth boundaries, particularly at high QP, as the relevant stress test.

This confirms that unconditionally sharing a cached neighbour feature is
unsafe in a tile-adaptive early-exit decoder. The predeclared active-neighbour
arms must be assessed on the full Kodak cohort and then transferred unchanged
to the disjoint DIV2K validation cohort. No claim about speed follows from
these untimed CPU FP32 replays.

Raw paired cases: [`kodak_canvas_coupling_20261005.json`](../../../proof/cpu_early_exit/results/kodak_canvas_coupling_20261005.json).
Recomputed clustered summary: [`kodak_canvas_coupling_20261005_summary.json`](../../../proof/cpu_early_exit/results/kodak_canvas_coupling_20261005_summary.json).
Protocol: [`PROTOCOL.md`](PROTOCOL.md).
