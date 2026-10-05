# Exploratory repair attenuation under active context

The completed active-zero DIV2K validation arm improves more when the
pre-existing grid seam repair is disabled. The repair weights were trained
for isolated tiles, so a fixed interpolation between no correction and the
trained correction is a cheap hypothesis to test before retraining.

The pilot uses frozen active-zero coupling, e15 weights, DIV2K calibration
bitstreams and exit maps. Images are the 0th, 8th and 16th names in sorted
calibration order (`0807`, `0836`, `0869`); QPs are 16 and 63. This gives six
cases, including both uniform- and mixed-depth maps. For the same pre-repair
feature `f`, the head receives `f + α(R(f)-f)` with α in
`{0, 0.25, 0.5, 0.75, 1}` and the trained repair `R`. α=0 and α=1 reproduce
the no-repair and original-repair endpoints. The response is assessed as
YCbCr 4:4:4 PSNR loss against full-frame e15 at fixed bitstream and map.

This study was designed **after** seeing the active-zero validation result.
It is exploratory; its six calibration cases cannot confirm a new method,
and tuning α on them then reporting them as a validation gain would be
invalid. If an interior α helps consistently, the next check must freeze
that choice and evaluate an unused cohort at identical maps and real rates.
The pilot runs on one low-priority CPU process after the ongoing DIV2K
fallback transfer completes, leaving all GPU training untouched. It does
not measure latency or change any trained parameter.

Run: `bash proof/cpu_early_exit/run_active_repair_scale_pilot.sh`.
Output: `proof/cpu_early_exit/results/div2k_beta/quality_floor/active_repair_scale_pilot.json`.
