# Mixed-depth canvas coupling can hurt

This is a post-hoc **stress audit**, not a cohort estimate. After finding the
DIV2K `0828` seam floor, we selected two previously held-out DIV2K validation
cases by inspecting the frozen primary-policy losses: `0844/QP48` (largest
validation loss, shallow map) and `0898/QP48` (a smaller threshold violation
with four of six tiles at the deepest exit). These cases were not used to fit
the β policy, but the selection here was informed by their outcomes.

Each row uses the same released-analysis FUFREF2 stream and e15 full-frame
reconstruction as reference. Numbers are assembled-image Δ444 in dB. The
padding/repair controls matter: coupling changes both if invoked directly
through its existing decoder mode, so comparing it only against the deployed
path would confound mechanisms.

| Case and fixed exit map | Deployed replicate + repair | Replicate, no repair | Zero pad, no repair | Coupled depthwise, no repair | Exact context, no repair |
|:--|--:|--:|--:|--:|--:|
| 0828/QP63 · 5/5/5/3/5/5 | 0.1565 | 0.1281 | 0.1587 | **0.0991** | **0.0159** |
| 0844/QP48 · 2/2/2/2/3/3 | 0.4961 | 0.5364 | 0.5732 | 0.5721 | **0.4647** |
| 0898/QP48 · 3/5/5/5/5/3 | 0.1080 | 0.1651 | 0.1651 | **0.3651** | 0.0613 |

For all three cases, uniform all-deep exact-context and coupled/no-repair
outputs equal e15 full-frame exactly (maximum absolute difference 0). The
coupler is therefore correct in its uniform-depth control. Its mixed-depth
behavior is the issue. Coupling helps `0828/QP63` relative to a matched
zero-pad/no-repair control, does essentially nothing on the shallow `0844`
map, and **adds 0.2001 dB of loss** on `0898` relative to that same matched
control. The latter has a majority of deep tiles, so a rule based only on
deep-tile fraction would be unsafe. The coupler serves a shallower tile's last
activation as context when a deeper neighbor advances; we infer that this
depth-mismatched context can be harmful, but this audit does not isolate the
specific feature channel or convolution responsible.

By contrast, depth-specific exact context reduces all three mixed-map losses,
but the full-window implementation expands expensive pointwise operations and
can erase compute savings. A promising next *implementation* ablation is to
propagate only the dependency band across an early-exit boundary (or taper
windows after each depthwise layer), charging pointwise work only on the band.
This requires an exactness control against the full-window counterfactual,
plus matched latency and full-cohort quality tests before it can be a paper
claim. A policy fallback remains necessary for `0844`'s shallow-route loss:
even perfect spatial context leaves 0.4647 dB.

The results are archived as `exact_context_0828_qp63.json`,
`exact_context_0844_qp48.json`, and `exact_context_0898_qp48.json` under
`proof/cpu_early_exit/results/div2k_beta/quality_floor/`. Reproduce with
`proof/cpu_early_exit/audit_exact_context_0828.py` and the frozen case JSON,
bitstream, policy, checkpoint, and upstream paths recorded in each result.
