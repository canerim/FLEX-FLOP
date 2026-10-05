# Recalibrating early-exit budgets after restoring active context

The deployed DIV2K-calibrated beta map leaves slack under the 0.1 dB mean
YCbCr 4:4:4 target when decoded with active-replicate context and no old
seam repair. This suggests the same router can exit more aggressively at
unchanged bitstream rate. The next experiment tests that opportunity
without new weights or GPU work.

The input is the **pre-existing** 24–25-beta grid per QP and its candidate
exit maps for all 24 DIV2K calibration images at five QPs. For each of the
120 frozen FUFREF2 streams, the runner deduplicates equal maps, decodes
every distinct map once using active-replicate/no-repair e15, and records
its actual Delta444 loss and analytical full-synthesis convolution MAC
saving. The full grid contains 1,004 distinct per-case maps. Every source,
case, stream, release, e15 checkpoint and critical code hash is pinned.
The cost formula charges the one-cell depthwise halo and removes grid
repair. Candidate cached deployed MACs are recomputed as a cross-check.

**Selection protocol after the frontier completes:** For each QP, among
the archived global beta candidates whose 24-image calibration mean
Delta444 is at most 0.1 dB, select the one with the highest mean analytical
synthesis MAC saving; ties choose lower mean Delta444, then lower beta.
No Kodak or DIV2K validation outcome may enter this choice. Freeze the
five selected betas and replay their maps on the 24 separate DIV2K
validation images, then Kodak, at the same FUFREF2 streams. Report
achieved quality, threshold violations and convolution MAC; only a
matched timed implementation can support a runtime claim. If the new
calibration point fails the validation quality target, report the failure
and do not silently retune beta on validation.

This is **exploratory** because the idea follows the observed context
transfer. The DIV2K validation images have also informed earlier project
work, so they are disjoint from this calibration split but not an untouched
benchmark. The 1,004-map run is queued as one low-priority CPU process
after the ongoing isolated-tile repair-off control. No GPU training job
is modified or displaced.

Run: `bash proof/cpu_early_exit/run_active_replicate_beta_calibration.sh`.
Output: `proof/cpu_early_exit/results/div2k_beta/quality_floor/active_replicate_beta_calibration24.json`.
