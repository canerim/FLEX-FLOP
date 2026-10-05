# Cheap continuation for exited tiles: arithmetic feasibility only

The local Mosaic paper keeps a skipped region spatially connected through a
cheap depthwise path. For the shared early-exit DCVC-UF decoder, a related
future ablation would let an exited tile advance a **context-only** feature
through one 3×3 depthwise operation per skipped suffix block. Its frozen exit
output would still be produced by the existing adapter; the continued feature
would serve only as a neighbour value for tiles executing deeper blocks.
This is a hypothesis, not an implemented decoder or a measured improvement.

The frozen Kodak24 × five-QP route maps skip 4.128 of eight suffix blocks per
tile on average. Using the project's full-synthesis convolution-MAC
normalisation, a full-tile 3×3 depthwise operation costs 0.02177 percentage
points per skipped block. A dense context-only continuation therefore adds
**0.08987 percentage points** of full synthesis conv-MAC on average. Together
with the one-cell active halo already audited, the extra arithmetic is
**0.10074 percentage points**; the corresponding *hypothetical* saving with
the existing seam repair would be **25.7830%**, versus 25.8837% for the
deployed isolated-tile route. These calculations exclude launching kernels,
moving full tile features, memory writes, entropy, and router cost, so they
do not predict runtime.

A depthwise-only path is not an exact continuation of a DCVC-UF block: it
omits the dominant 1×1 channel mixing and nonlinearities. It may provide a
better stage-aligned boundary feature, or it may corrupt the active tile.
The next meaningful test would freeze the current bitstreams and routes,
train only the cheap context path against full-depth boundary features on
training crops, then compare it with active-zero and active-replicate on a
disjoint validation cohort at identical maps and rates. No conclusion should
be drawn from the MAC estimate before that quality test and a matched latency
measurement. A separate production question is whether the needed dense
feature update erases the arithmetic saving through memory traffic.

Reproduce the arithmetic with
`python proof/cpu_early_exit/audit_ghost_context_cost.py`. The
[per-case output](../../../proof/cpu_early_exit/results/kodak_ghost_context_cost_20261005.json)
pins its input hash. Mosaic's full cheap path is a distinct model that can
re-enter the expensive branch; this proposal keeps irreversible early exits.
