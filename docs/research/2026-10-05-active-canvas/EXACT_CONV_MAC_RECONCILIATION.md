# Exact convolution MAC reconciliation for active context

The historical `flexuf.cost.frame_relative_cost` table is a useful router
control, but it is not the exact convolution count used by the released-D12
comparison. It normalises rounded block shares and uses an `8C²+9C`
block proxy; the implemented block has `7C²+9C` convolution MAC per
feature pixel. We therefore recounted the frozen old-β maps with the
architecture-level `mac_latency_audit.case_macs` arithmetic. The latter
has an existing Conv2d-hook cross-check in the project.

The active-replicate coupler reads a `(P+2)×(P+2)` halo but performs a
valid 3×3 depthwise convolution whose **output is `P×P`**. The halo does
not increase convolution MAC. Repair-free active context has the exact
routed convolution count minus one full-frame GridSeamRepair convolution
pass. Canvas construction, gathers, memory traffic and control scheduling
may still add runtime; this correction says nothing about latency.

For the frozen deployed old-β policy, mean exact synthesis-convolution
savings versus released D12 are **20.17%** on DIV2K calibration24,
**19.48%** on DIV2K validation24 and **25.19%** on Kodak24, each at five
QPs. Kodak's 24-image clustered 95% interval is **[22.74%, 27.49%]**.
The previously reported legacy cost-model means were 20.73%,
20.05% and 25.88%, respectively. Quality numbers and exit maps do not
change. On the Kodak old-β maps, active-replicate/no-repair has **26.27%**
exact saving, versus **26.82%** under the legacy formula; the mean
overstatement is 0.549 percentage points across 120 cases.

The active-replicate β calibration currently running was intentionally
left untouched to preserve its provenance. Its stored MAC fields and the
queued selector still use the historical proxy. After the policy is
locked, we must recompute exact savings for every candidate and check
whether the selected β would change under exact arithmetic. The
calibration-only choice cannot be revised using validation outcomes.
Future paper claims should use exact arithmetic for reported savings and
label the original policy's selection cost as the legacy proxy.

Machine-readable case rows and source hashes:
`proof/cpu_early_exit/results/kodak_active_canvas_exact_conv_cost_20261005.json`.
