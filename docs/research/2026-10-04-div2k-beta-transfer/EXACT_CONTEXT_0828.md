# Why DIV2K 0828 fails the 0.1 dB target: depth choice versus tile context

This is a **targeted diagnostic**, not a held-out mean result or a deployable
speed claim. We reused the five real FUFREF2 streams for the fixed 768×512
crop of DIV2K `0828.png`, the frozen e15 checkpoint, and the previously locked
per-QP β router. The reference is the e15 full-frame reconstruction from the
same latent. Every loss below is Δ444 = 10 log10(MSE / MSE_e15_full), measured
on the assembled image in centred YCbCr444. The original released decoder is
not the reference in this audit.

| QP | Frozen exit map | Deployed zero-halo + repair | Exact context, no repair | Depthwise canvas coupling, no repair |
|--:|:--|--:|--:|--:|
| 0 | 3/3/3/2/3/3 | 0.7918 dB | 0.7202 dB | 0.7338 dB |
| 16 | 5/5/5/3/5/5 | 0.1722 dB | 0.0107 dB | 0.0556 dB |
| 32 | 5/5/5/3/5/5 | 0.1622 dB | 0.0131 dB | 0.0667 dB |
| 48 | 5/5/5/3/5/5 | 0.1528 dB | 0.0124 dB | 0.0821 dB |
| 63 | 5/5/5/3/5/5 | 0.1565 dB | 0.0159 dB | 0.0991 dB |

With **all six tiles at the deepest exit**, exact-context and canvas-coupled
decoding without repair equal e15 full-frame **bit-for-bit at all five QPs**
(maximum absolute output difference 0). The deployed tiled all-deep path
instead loses 0.1148–0.1636 dB. That establishes a removable tile-boundary
floor for this case; it does not show that the frozen router can meet a 0.1 dB
budget at its chosen map. QP0 is decisive: its shallow map still loses 0.7202
dB even with exact context. At QP16–63, the fixed map has one D8 tile and five
D12 tiles, and the seam is the dominant loss.

The exact-context experiment extracts clipped overlapping windows from the
shared stem, with a depth-dependent feature halo equal to the number of
remaining 3×3 depthwise layers, then runs the *whole* suffix on each window.
It is numerically clean but expensive. Relative to a unit-cost full synthesis,
analytical MAC is **1.3975** for all-deep exact-context decoding (no repair)
versus **1.0095** for deployed all-deep with repair. For QP16–63's frozen mixed
map, cost is **1.3149** versus **0.9676**. At QP0 the shallower map costs
**0.8201** versus **0.7330**. These figures charge actual clipped window areas,
adapters, stem, and head; they exclude router, entropy, memory movement, and
latency. The suffix feature-cell multiplier is 1.67× for all deep and 1.65× for
QP16–63's fixed map. Full halo therefore sacrifices the desired compute saving
at the high-QP maps.

The existing depthwise-only `CanvasCoupler` avoids haloing the costly pointwise
operations. In this narrow audit its **all-deep no-repair** output is also
bit-for-bit the full-frame reference. For the mixed map it stays under 0.1 dB
at QP16/32/48/63, but its loss grows with QP and reaches 0.0991 dB at QP63.
This is close to the threshold and cannot support a general guarantee. Keeping
the old trained seam-repair module actually makes the coupled QP63 result worse
(0.1493 dB versus 0.0991 dB without it), so a decoder variant would need an
explicit repair decision or retraining. Mixed-depth coupling uses a shallow
neighbour's activation as context for a deeper tile, which explains why it
does not match the exact-context reconstruction even though both match the
uniform all-deep control.

Subsequent post-hoc held-out stress checks show that this apparent gain does
**not** generalise automatically: on `0898/QP48`, coupling adds 0.2001 dB
relative to the matched zero-pad/no-repair control. See
[MIXED_COUPLING_STRESS.md](MIXED_COUPLING_STRESS.md). The coupler is therefore
only a diagnostic here, not a safe drop-in deployment choice.

**Next controlled test:** replay coupled/no-repair and the frozen router on a
disjoint multi-image cohort, report QP-stratified quality tails and analytical
MAC, then benchmark decoder latency with a matched released/e15 backend when
the host is quiet. A frame-level fallback or risk-sensitive router is needed
for the QP0-style shallow-route failure; changing seam handling alone cannot
fix it. Any modified synthesis decoder must be treated as a new codec variant,
with encode/decode compatibility verified. Current PyTorch `CanvasCoupler`
creates halo buffers and gathers at every depthwise call, so its tiny arithmetic
overhead must **not** be equated to tiny wall-clock overhead.

Reproduce with `proof/cpu_early_exit/audit_exact_context_0828.py` for each
archived `0828_qp*.json`/`.fufref2` pair and
`proof/cpu_early_exit/summarize_exact_context_0828.py`. The five audited JSON
outputs, analytical cost summary, and vector figure are in
`proof/cpu_early_exit/results/div2k_beta/quality_floor/`.
