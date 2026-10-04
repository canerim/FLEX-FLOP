# Independent beta-transfer figure data

`per_qp_policy_summary.json` and `global_policy_summary.json` are audited
24-image × 5-QP cohort summaries. Both policies and their SHA256 identifiers
are archived beside them. The figure is a vector PDF built from these summaries
and the fixed-QP16 Kodak replay records.

Full code, source/bitstream/result hashes, the direct outlier replay, and the
research report are in [FLEX-FLOP proof commit `c067781`](https://github.com/canerim/FLEX-FLOP/tree/c067781/proof/cpu_early_exit).
`beta_quality_floor.json` and `worst_case_direct_replay.json` add the
calibration-only all-deep tiled-floor audit and its direct decoder check.
The per-QP rule was fixed before DIV2K validation and Kodak transfer. The
global-budget rule was added after inspecting calibration aggregates, then
frozen before its validation and Kodak replays. Kodak had already been used
in earlier exploratory diagnostics, so this is independent beta fitting,
not a globally untouched Kodak evaluation.

Δ444 measures assembled YCbCr444 reconstruction against e15 full-frame
synthesis from the same research bitstream. MAC saving is analytical and
restricted to synthesis; router, entropy and complete-codec time are omitted.
