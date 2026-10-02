# Matched-rate Kodak depth comparison

The three JSON inputs are unmodified copies of the `kodak_final.json`
files for the 105-epoch D2 and D4 runs, and the archived Microsoft released
D12 `kodak.json`. Source paths are under
`/data10/shareddata/can_karsal/dcvcuf_depth_20260927/` (respectively
`runs/d2`, `runs/d4`, and `reference_d12`). Their SHA-256 values are in
`paired_kodak_analysis.json`. Both scratch runs report 2,525,145 scheduled
steps, zero nonfinite skips and `state=complete`. The released metric JSON
matches `reference_d12/source_manifest.json`; the archived checkpoint hash
was captured after the original reference evaluation, so historical metric
file provenance does not prove a contemporaneous checkpoint hash.

Reproduce the CPU-only analysis and figure with
`python3 scripts/depth_final_kodak_pair_20261002.py`. It validates unique
image/QP identities, 24 × 5 records, finite metrics, common image IDs,
positive ordered rates, and explicit interpolation support. It uses
per-image log estimated-bpp interpolation and 5,000 paired-image bootstrap
draws, seed 20261002. The rate is an entropy estimate rather than container
payload. The comparison to released D12 mixes depth and training history;
it is retained as a deployment anchor until a matched scratch D12 completes.
