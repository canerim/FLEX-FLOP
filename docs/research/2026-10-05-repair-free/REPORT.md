# Frozen-weight repair-free early-exit ablation

The 53 QP32 CTC first frames already have a matched 2×2 component intervention: fixed e15 checkpoint, fixed Q90 router map and shared encoded representation, with only adapters and grid repair toggled. This audit compares the trained path to the repair-identity path. It does **not** retrain the model or measure latency.

| Same-route synthesis | Exact convolution MAC saving vs released D12 | Mean RGB PSNR loss from repair removal | Cases above 0.1 dB vs dense e15 |
|---|---:|---:|---:|
| Trained path, with repair | 16.26% | — | 4/53 |
| Repair identity | 17.35% | 0.00208 dB [0.00062, 0.00364] | 6/53 |

Omitting the full-frame grid-repair 1×1 plus 3×3 depthwise convolutions saves **1.086 percentage points** of exact conv-MACs for these maps. The largest single-image repair-removal loss is 0.0224 dB. This is a promising low-cost inference option because the mean quality change is small, but the extra threshold exceedances matter: it is not a free or certified 0.1 dB improvement. Memory traffic, dispatch and full-codec latency are unknown, and the 53 frames were already used in development.

This follows the matched-component protocol useful in the local Mosaic study: hold checkpoint, coded representation and route constant before attributing a change to one module. The next decisive test is a repair-free retraining ablation with equal training budget, followed by a held-out, matched-backend timing and quality comparison. A single-checkpoint identity substitution cannot establish whether the network would learn a stronger repair-free solution.

Run `python3 proof/cpu_early_exit/audit_repair_free_tradeoff.py` to regenerate [the per-frame record](../../../proof/cpu_early_exit/results/repair_free_tradeoff_20261005.json) from the frozen intervention bundle and exact convolution counter.
