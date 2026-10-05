# Predeclared active-context quality-budget sweep

The primary active-replicate/no-repair policy targets mean Δ444 ≤0.1 dB
on the 24-image DIV2K calibration partition. To test whether the result is
specific to one tolerance, freeze four calibration-only targets **0.075,
0.100, 0.125, 0.150 dB**, each separately at QP 0/16/32/48/63. These
thresholds were recorded before the 120-case active-context calibration
frontier or primary validation completed. The 0.100 policy must match the
independently scripted primary selection byte for byte.

For each QP and target, `select_active_beta_budget_sweep.py` chooses the
archived beta candidate with the highest mean analytical synthesis-conv
MAC saving among candidates meeting the calibration mean Δ444 target.
Ties minimize mean loss and then beta. A target without a feasible beta at
any QP is labelled infeasible for the five-QP curve, not relaxed or filled
from validation. No Kodak or DIV2K validation outcome is read by selection.

`audit_active_beta_budget_transfer.py` replays only feasible selected
policies on the separate 24-image DIV2K validation partition, using the
same FUFREF2 streams and e15 weights. It reuses the separately completed
0.100-dB replay as a checksum and computes each other distinct exit map
once per case. `analyze_active_beta_budget_transfer.py` reports actual
mean Δ444, 0.1-dB violation counts and analytical synthesis-conv MAC
savings at each target, with 24-image clustered uncertainty intervals.
The validation result will be reported even if a selected policy misses
its calibration quality target after transfer. No post-validation beta
adjustment is allowed.

This is an exploratory fixed-bitstream decoder study, not native DCVC-UF
BD-rate, runtime or a previously untouched benchmark. DIV2K validation
has informed other exploratory work. It is disjoint from the calibration
partition used for this particular selector.
