# Matched context × repair audit: DIV2K validation24

The isolated-tile/no-repair control completed all 120 DIV2K validation
replays (24 images × five QPs). Each row is paired to the deployed and
active-neighbour arms by source, FUFREF2 stream hash, e15 checkpoint and
frozen exit map. The isolated/repair-on replay reproduces the archived
deployed loss within 1e-6 dB in every case. These CPU FP32 measurements
are untimed YCbCr 4:4:4 reconstruction comparisons against full-frame e15,
not native bitstream, YUV BD-rate or latency measurements.

| Decoder context | Trained grid repair | Mean gain over deployed | 24-image clustered 95% CI | Cases losing >0.1 dB vs full-frame e15 |
|---|---|---:|---:|---:|
| Isolated tiles | On (deployed) | 0 | — | 35/120 |
| Isolated tiles | Off | **−0.01131 dB** | [−0.01785, −0.00564] | 46/120 |
| Active-replicate | On | +0.00908 dB | [+0.00441, +0.01413] | 30/120 |
| Active-replicate | Off | **+0.01969 dB** | [+0.01460, +0.02523] | **24/120** |

This is a strong **interaction**, not two additive gains. Removing repair
with isolated tiles *hurts* mean quality by 0.01131 dB (111 worse, 9 better
cases). Under active-replicate context the same removal *helps* by 0.01062
dB (90 better, 30 worse). Their difference-in-differences is **+0.02193
dB** (image-cluster 95% CI **[+0.01083,+0.03514] dB**). The active-replicate
context effect is +0.00908 dB with repair on, but +0.03100 dB with repair
off (CI [+0.02140,+0.04234]); it is positive in all 120 individual cases
under repair-off. Active-zero exhibits the same direction: context effect
+0.00646 dB repair-on and +0.02826 dB repair-off, with a +0.02180 dB
interaction. The repair weights were trained for independent tiles, so this
pattern is consistent with a repair module that partly compensates missing
context and overcorrects after neighbours are supplied. That mechanism is
an inference from the factorial measurements, not a directly observed
feature attribution.

The four arms share the *original*, independently calibrated beta and route
maps. The repair-off arms were prompted by earlier exploratory results and
are therefore hypothesis-generating. DIV2K validation is disjoint from
the beta calibration24 cohort, but has informed other exploratory project
work. A separate calibration-only frontier will test whether the restored
quality margin permits more aggressive routing; that selection must be
frozen before looking at the replay outcomes here. Analytical MAC and
wall-clock time also require separate accounting because active-neighbour
halo reads add work while removing repair saves work.

Reproduction: run
`proof/cpu_early_exit/audit_isolated_no_repair_div2k.py`, then
`proof/cpu_early_exit/analyze_isolated_no_repair_div2k.py` and
`proof/cpu_early_exit/plot_context_repair_interaction.py`. The
[raw isolated control](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/isolated_no_repair_validation24.json),
[hash-linked decomposition](../../../proof/cpu_early_exit/results/div2k_beta/quality_floor/isolated_no_repair_decomposition.json)
and [vector figure](../../figures/active-canvas-transfer-20261005/context_repair_interaction.pdf)
are committed with the scripts.
