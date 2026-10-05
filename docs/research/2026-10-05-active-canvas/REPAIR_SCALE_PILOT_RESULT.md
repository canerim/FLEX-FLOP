# Repair attenuation pilot: no useful interior optimum established

The queued CPU pilot completed its six frozen DIV2K **calibration** cases:
images `0807`, `0836`, `0869` at QP16 and QP63. Active-zero coupling,
bitstreams, exit maps and e15 weights were fixed. The only intervention was
`f + α(R(f)-f)`, scaling the previously trained grid seam repair from
α=0 (off) to α=1 (original). The raw file records every case and hashes the
exact source, manifest, release and e15 checkpoints plus the audit code.

| α | Mean Δ444 loss against full-frame e15 | Mean gain over α=1 |
|---:|---:|---:|
| 0 | 0.017043 dB | +0.005045 dB |
| 0.25 | **0.016781 dB** | +0.005307 dB |
| 0.50 | 0.017534 dB | +0.004553 dB |
| 0.75 | 0.019304 dB | +0.002784 dB |
| 1 | 0.022088 dB | anchor |

α=0.25 gains just **+0.000262 dB** over α=0, improving four cases and
worsening two. Best per-case α varies across 0, 0.25, 0.75 and 1. This
six-case, post-result pilot does not establish a useful interior optimum;
we will not promote α=0.25 into the method or run a large attenuation sweep
on this evidence. The important remaining ablation is the queued
isolated-tile **repair-off** control on all 120 DIV2K validation cases,
which separates context restoration from simply removing the old repair.

No quality metric here is YUV BD-rate, and no latency is measured. Source:
[`active_repair_scale_pilot.json`](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/active_repair_scale_pilot.json).
