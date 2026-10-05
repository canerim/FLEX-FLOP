# Matched repair-off control for the DIV2K active-context result

The active-zero DIV2K result compared a coupled, repair-off decoder with
the deployed isolated-tile decoder that retained its trained seam repair.
Both context and repair changed. The missing control holds repair **off**
on both paths, at the same e15 weights, frozen validation24 exit maps and
FUFREF2 bitstreams. This separates the contribution of active-neighbour
context from the contribution of disabling repair.

The runner recomputes the deployed repair-on reconstruction in every case
and requires its YCbCr 4:4:4 PSNR loss to match the archived deployed value
within 1e-6 dB. It then decodes the identical latent and map with repair
disabled. It checks each source, case, stream, release and e15 checkpoint
hash. The output is an incremental 120-case CPU FP32 audit with no timing
or native CUDA result. A one-case smoke check reproduced the archived
repair-on baseline before the full job was queued.

This control was identified after inspecting the active-zero validation
result, so it is an **exploratory causal ablation**, not a preregistered
confirmatory test. It is queued after the active-replicate transfer and the
six-case calibration repair-scale pilot, on one low-priority CPU process;
the GPU training jobs remain untouched.

Run: `bash proof/cpu_early_exit/run_isolated_no_repair_div2k.sh`.
Output: `proof/cpu_early_exit/results/div2k_beta/quality_floor/isolated_no_repair_validation24.json`.
