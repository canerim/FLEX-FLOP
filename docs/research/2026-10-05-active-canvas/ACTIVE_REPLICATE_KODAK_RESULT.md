# Frozen Kodak active-neighbour fallback comparison

We replayed all 24 Kodak images at QP 0, 16, 32, 48 and 63 (120 cases)
through the same frozen FUFREF2 streams and DIV2K-calibrated exit maps.
Checkpoint e15, full-frame e15 reference and trained seam repair are fixed.
Only the one-pixel depthwise context supplied by a neighbour that has
already exited changes. The stage-synchronous active-zero rule supplies zero;
the active-replicate rule replicates the nearest still-active feature.
Neither arm trains new weights or measures execution time.

| Decoder path, with identical seam repair | Mean 4:4:4 PSNR gain versus deployed | Cases >0.1 dB below full-frame e15 | Mean YUV 6:1:1 BD-rate versus deployed | Mean BD-rate versus released D12 |
|---|---:|---:|---:|---:|
| Deployed isolated-tile path | anchor | 47/120 | anchor | +3.665% |
| Active-zero | +0.00940 dB | 36/120 | −0.2443% | +3.4111% |
| Active-replicate | **+0.01112 dB** | **36/120** | **−0.2852%** | +3.3678% |
| Exact-context diagnostic | see comparison JSON | 34/120 | see BD-rate JSON | +3.2032% |
| Legacy stale canvas | −0.06573 dB | 56/120 | +0.662% | +4.355% |

The direct paired active-replicate minus active-zero mean gain is
**+0.001727 dB** (24-image cluster-bootstrap 95% interval
[+0.000890,+0.002724] dB). It improves 81 of 120 cases, worsens 16 and
ties the 23 uniform-depth maps. Direct per-image PCHIP BD-rate relative to
active-zero is **−0.04097%** (24-image bootstrap 95% interval
[−0.06762,−0.02035]%), with 21 of 24 images improving. This is a small
incremental gain, and the count over the 0.1 dB quality threshold does not
improve further. Both active rules improve all 97 mixed-depth maps relative
to the legacy stale-feature rule; all 23 uniform-depth maps tie it.

All five QP streams per image are byte-identical across decoder arms; their
length and SHA-256 were checked before calculating the rates. Thus BD-rate
here summarizes reconstruction quality at fixed research-stream rates. It is
not an encoder saving, a native DCVC-UF bitstream result, or a latency
measurement. The fallback choices followed a pilot, so this Kodak scan is
exploratory. The frozen DIV2K validation24 transfer is still running; retain
both variants until that result is complete. The PyTorch halo packer is
untimed, and identical analytical convolution MAC does not imply identical
wall-clock cost.

Reproduce numerical comparisons with
`python proof/cpu_early_exit/compare_canvas_arms.py` and
`.venv/bin/python proof/cpu_early_exit/analyze_canvas_bdrate.py --arms stale active_zero active_replicate`.
The [full paired comparison](../../../proof/cpu_early_exit/results/kodak_canvas_arms_comparison_20261005.json),
[per-image BD-rate curves](../../../proof/cpu_early_exit/results/kodak_canvas_bdrate_20261005.json),
and [vector figure](../../figures/active-canvas-20261005/canvas_context.pdf)
retain source hashes and all case-level values.
