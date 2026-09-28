# Metric correction: shared-exit loss is YCbCr 4:4:4, not RGB

The 27 September source audit and two actual CPU replays identified a
misleading legacy name. `db_rgb` in the archived shared-exit evaluation
does **not** contain RGB PSNR loss. The code subtracts unclipped, centred
YCbCr 4:4:4 tensors directly and averages squared error equally over their
three channels. The reported loss is `10 log10(MSE_mixed / MSE_reference)`.
The padded `M` and `R` source-error tables use the same colour space.

Current text and plots call this **444-MSE loss** or **Δ444**. Raw keys such
as `db_rgb`, `rgb_loss`, `rgb_mean`, `rgb_over` and `padded_rgb_loss_db`, and
older descriptive annotations in bundled audit JSON, are preserved for
traceability. Their old “RGB” wording is superseded by this correction;
the numerical observations are unchanged. These values must not be
represented as RGB measurements or RGB quality guarantees.

The separate `db_611` / `yuv_loss` metric is the difference between
weighted averages of Y/U/V plane PSNRs, `(6 PSNR_Y + PSNR_U + PSNR_V)/8`,
after the recorded 4:2:0 downsampling and clipping path. It is not an
equal-channel MSE-ratio statistic.

Two QP32 fixed-map replays reproduce the archived direct-tensor ratio to
within 1.1e-6 dB. Explicit RGB conversion gives different values; these
two checks do not supply a replacement RGB result for all 265 frame–QP
pairs. Evidence, strict-load status, source hashes and both numerical
conventions are in [the metric audit](data/sharedmetric20260927/analysis.json).
The source reader, model and evaluator chain is described in supplement S2.

The independent epoch20 D2/D4/D6/released-D12 CPU reference study uses
explicit RGB conversion and reports RGB PSNR under its own manifest.
It is a separate experiment, with different checkpoints and protocol.

Synthesis MAC reductions in the shared-exit plots exclude entropy recovery,
router/signalling work and system overhead. No colour-space correction or
MAC relabelling changes the absence of a complete codec latency benchmark.

The 53-sequence collection is an archived evaluation corpus; “CTC” in local source names is not a claim of a single deduplicated official benchmark. The outcome-blind first-frame similarity audit flags two candidate pairs across the sequence folds. Their raw byte hashes differ. Future calibration should conservatively group related sources; the current sequence-disjoint diagnostics are not content-independent external-test evidence. See `data/research20260927/shared_crossfit_qp32/cohort_similarity_audit.json`.
