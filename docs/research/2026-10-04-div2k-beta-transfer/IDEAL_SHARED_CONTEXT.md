# How much compute remains if exact tile context is shared perfectly?

The static halo experiment repeats overlapping suffix work per tile. A
blockwise taper discards dependency rings as soon as they are no longer
needed and matches static-halo output exactly, yet costs 1.151× full synthesis
MAC on `0828/QP63`'s fixed map. We therefore computed an **optimistic
feature-cell sharing estimate**: at suffix block `i`, take the union of the
spatial cells required by all tile cores at their chosen exit depths. Charge
each distinct cell for that block only once. Add the actual shared stem,
full-frame head, and core-only exit adapter costs. This assumes a hypothetical
masked scheduler executes exactly those cells with zero packing, control,
memory, or launch overhead. It is **not measured latency**, nor a general
lower bound over all possible decoder designs.

| Real bitstream/crop | Fixed map | Deployed Δ444 | Exact-context Δ444 | Optimistic shared-context MAC / full e15 |
|:--|:--|--:|--:|--:|
| 0828/QP0 | 3/3/3/2/3/3 | 0.7918 dB | 0.7202 dB | 0.7257 |
| 0828/QP63 | 5/5/5/3/5/5 | 0.1565 dB | **0.0159 dB** | **0.9655** |
| 0844/QP48 | 2/2/2/2/3/3 | 0.4961 dB | 0.4647 dB | 0.6525 |
| 0898/QP48 | 3/5/5/5/5/3 | 0.1080 dB | **0.0613 dB** | **0.9309** |

All-deep maps require the full frame at all eight suffix blocks, giving
relative MAC **1.0000** and zero exact-context loss on each case. The result
reveals the central quality–compute conflict in these examples: the shallow
maps retain 27–35% ideal synthesis savings but have substantial quality loss
even with perfect context; the quality-preserving, mostly-D12 maps leave only
3.5–6.9% ideal synthesis saving. The real implementation can be slower than
this optimistic estimate, especially for narrow sparse regions on GPU/CPU.
The current zero-halo decoder's deployed cost is a different implementation
with seam-repair overhead and cannot be compared as measured latency.

For reproducibility, `proof/cpu_early_exit/analyze_ideal_shared_context.py`
reads the archived case-level exact-context JSONs and e15 config, constructs
one 64×96 Boolean feature mask for each suffix block, and archives the eight
union-cell counts and costs in
`proof/cpu_early_exit/results/div2k_beta/quality_floor/ideal_shared_context_bound.json`.
No GPU model execution or timing enters this analysis.
