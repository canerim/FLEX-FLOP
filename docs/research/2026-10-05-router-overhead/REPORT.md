# Router arithmetic on the frozen shared-exit CTC archive

The main early-exit comparison counts executed synthesis convolutions on the padded frame, including adapters and seam repair. It omits the learned controller, whereas deterministic ordered dithering does not run that controller. We add the 53-frame QP32 Conv2d/Linear hook count to the routed arm without changing any selected maps or quality measurements. The controller executes 289.71484375 MAC per padded image pixel. Its architecture and padded geometry are unchanged across QPs, so the same count applies to each of the five archived quality indices. The released D12 synthesis denominator is computed from the same geometry.

| Frozen comparison | Routed saving with controller | Dither saving | Routed − dither |
|---|---:|---:|---:|
| Nominal 0.1 dB, all 263 feasible frame–QP pairs | 27.16% | 24.78% | +2.38 percentage points |
| Fixed QP32 mean control, 53 frames | 24.92% | 23.04% | +1.89 points |
| Fixed QP32 Q90 control, 53 frames | 16.13% | 13.88% | +2.25 points |

The controller costs **0.13344 percentage points** of released synthesis convolution MACs for every padded geometry in this cohort. The previously reported nominal 27.29% and +2.51-point values exclude it; adding the controller changes them to 27.16% and +2.38 points. Fixed-control rows above recompute **exact** synthesis convolution MACs from their frozen maps, so they should not be compared directly with the older normalized 25.675/23.802% and 16.741/14.204% estimates.

This closes only the router's Conv2d/Linear arithmetic gap. It still excludes control dispatch, pooling, memory traffic, entropy decoding, bitstream size and wall-clock latency. The nominal protocol uses source-calibrated controls, and the fixed-control paired intervals already include zero; the arithmetic correction does not turn either protocol into a deployable equal-quality routing claim.

Reproduce with `python3 proof/cpu_early_exit/audit_router_overhead.py`. [The analysis JSON](analysis.json) records input hashes and every nominal case.
