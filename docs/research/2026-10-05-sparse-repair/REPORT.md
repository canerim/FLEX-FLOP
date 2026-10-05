# Sparse grid-repair gate: quality pilot and locked full-cohort scan

The e15 `GridSeamRepair` gate is learned on a 32×32 feature-cell period. Its minimum is 0.223, not zero, so computing correction only at the tile boundary is **not** equivalent to the trained network. Nevertheless, the gate is nearly flat over the interior: a threshold of 0.25 retains 13.57% of feature positions. A [prototype packed 1×1 path](../../../proof/cpu_early_exit/sparse_grid_repair.py) now computes only those positions while keeping the full 3×3 depthwise pass. It is isolated from the decoder, and has not been timed or integrated into inference.

The quality pilot reconstructs the smallest, middle-size and largest frame of the frozen 53-frame QP32 intervention cohort. It keeps the e15 checkpoint, encoded representation and Q90 route fixed. The pointwise convolution still runs densely; only the gate is thresholded before the existing head. Baseline output matches the saved trained path and the all-off limit matches the saved repair-identity path within numerical tolerance.

| Gate threshold | Pointwise positions retained | Projected extra conv-MAC saving vs released D12 | Mean additional RGB PSNR loss |
|---:|---:|---:|---:|
| 0 (trained) | 100% | 0 | 0 |
| 0.25 | 13.57% | +0.917 pp | 0.00017 dB |
| 0.30 | 12.11% | +0.933 pp | 0.00017 dB |
| 1.00 (repair identity) | 0% | +1.061 pp | 0.00161 dB |

The largest additional loss among the three frames at threshold 0.25 is 0.00071 dB. These are selected pilot images and projected MACs, not a validated cohort result or actual speedup. The case with all tiles at the deepest exit has negative net routed-vs-released MAC saving even after gate truncation; the projection must therefore be interpreted per map rather than as a universal win.

Thresholds 0.25 and 0.30 were locked after this pilot. A one-thread, low-priority CPU replay of **all 53** frozen maps is running in the `reglic-gate-20261005` tmux session, with per-case resumable records in `proof/cpu_early_exit/results/sparse_repair_gate_full_cases/`. The 50 frames unused in the three-frame threshold pilot are the threshold-selection extension cohort, though the codec and router themselves were developed on this CTC corpus. The complete scan will decide whether the packed pointwise path is worth integrating and benchmarking. The [CPU equivalence tests](../../../proof/cpu_early_exit/test_sparse_grid_repair.py) pass for four synthetic gate thresholds and for the frozen 384-channel e15 weights; at threshold 0.25 the packed and dense-thresholded outputs were bit-identical on a 32×32 feature tile. This is functional evidence, not a speed result.

The [stream-to-reconstruction smoke check](../../../proof/cpu_early_exit/verify_sparse_repair_decode.py) installs the packed wrapper *after* strict checkpoint loading and decodes BQSquare with its archived QP32 Q90 map. Its full reconstructed tensor is bit-identical to dense thresholded repair (`max_abs=0`); cropped RGB MSE also matches the pilot exactly. This checks one real end-to-end synthesis path, not other images, a native stream container, or latency.

Reproduce the pilot with `python3 proof/cpu_early_exit/audit_sparse_repair_gate_pilot.py`; the [pilot JSON](../../../proof/cpu_early_exit/results/sparse_repair_gate_pilot_20261005.json) contains all 18 measured outputs. The full scan uses `proof/cpu_early_exit/audit_sparse_repair_gate_full.py` and writes one case record atomically before moving to the next.
