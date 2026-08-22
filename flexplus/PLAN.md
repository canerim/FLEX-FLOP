# FLEX+ — raising the ceiling with K=6 and j=2 left alone

## Why the obvious route is closed

The ladder shape is fixed by instruction: six exits, split at depth two, the
first four exits as they are. That closes the two moves the cost model most
rewards — a finer ladder (K=12, j=2 → 61.5%) or a shallower split (K=6, j=1 →
54.0%) — so the ceiling has to come from somewhere else.

## What the ceiling is made of

A frame with every tile on the shallowest rung it may take costs 0.6088 of a
released decode. That number is the ceiling: 1 − 0.6088 = 39.12%.

| part | cost | share of the floor |
|---|---|---|
| **shared trunk, 4 blocks, always, full frame** | **0.2981** | **49.0%** |
| tiled trunk, 2 blocks | 0.1491 | 24.5% |
| upsample, always, full frame | 0.0816 | 13.4% |
| adapter at the shallowest exit | 0.0464 | 7.6% |
| head × halo, always | 0.0240 | 3.9% |
| seam repair | 0.0095 | 1.6% |

Halving any one part moves the ceiling to:

| lever | −50% | −75% |
|---|---|---|
| **shared trunk** | **54.0%** | **61.5%** |
| tiled trunk | 46.6% | 50.3% |
| upsample | 43.2% | 45.2% |
| adapter | 41.5% | 42.6% |
| head | 40.3% | 40.9% |
| seam repair | 39.6% | 39.8% |

Only one lever reaches 55%. Every other part of the floor is too small to
matter even if it were free. So the experiment is not a search — it is a
single question: **can the four always-on trunk blocks be made cheaper for
the content that leaves at the shallowest exit, without spending the budget?**

## Why depth cannot answer it

The exit ladder is one-dimensional. A tile chooses how many blocks to run,
and the four blocks below the split are not among them: they run before the
frame is tiled, on every pixel, whatever any tile chooses. Depth cannot reach
them by construction. A second axis can.

## The three axes the literature offers, and which one fits

**Width.** Slimmable networks (Yu et al., ICLR 2019) train one network
executable at several channel widths, sharing weights and switching at
runtime; Dynamic Slimmable Network (Li et al., CVPR 2021) makes the width
input-dependent with a gate; SLEXNet (ACM TECS, 2024) combines slimmable
widths with early exits, which is exactly the pairing here. Cost falls with
the square of the width, so half width is a quarter of the cost — the
strongest scaling of the three.

**Resolution.** RANet (Yang et al., CVPR 2020) routes easy inputs down a
low-resolution path and exits them early, keeping a high-resolution path for
hard ones, on the argument that spatial detail is only needed sometimes.
Half resolution is a quarter of the cost, the same scaling as half width.

**Spatial sparsity.** SACT extends adaptive computation time across spatial
positions; Dynamic Convolutions (Verelst and Tuytelaars, arXiv:1912.03203)
learn a spatial mask and execute convolutions only where it is set; focused
convolutions (arXiv:2310.07782) do the same for pretrained networks without
retraining. Cost falls linearly with the active fraction — weaker scaling,
but it is the only one of the three that does not change the tensor shapes
the rest of the decoder expects.

**The fit.** Width and resolution both want the stem tiled, so that different
regions can run at different settings — and tiling the stem is exactly what
lowering the split depth does, which is the move that is closed. Spatial
sparsity does not: a mask inside a full-frame convolution changes no shape,
creates no tile boundary, and therefore adds no seam. It is the axis that
composes with the constraint rather than fighting it.

**And the gate is already computed.** The exit map says which tiles leave
shallowest. Those are the tiles the decoder has already judged to need least
work. Using that same map to sparsify the stem costs nothing to produce and
needs no second router.

## What is measured, in order

1. **Oracle degradation.** Replace the stem over easy regions with a cheaper
   stem — half resolution first, since it is a few lines and needs no
   training — and measure what the budget buys. If a half-resolution stem
   over the shallow tiles costs more than 0.2 dB on its own, the axis is
   dead and nothing further is worth running.
2. **Where the tolerance is.** Per rate and per exit: the tiles that leave at
   exit 2 should tolerate far more stem degradation than the ones that go
   deep, and the size of that gap is the size of the opportunity.
3. **Cost accounting.** What a masked stem actually saves, hook-counted, not
   modelled — the same discipline as the paper.
4. **Training**, only if 1–3 say the ceiling moves.

Nothing here writes outside `flexplus/`. `runs/` is a symlink to the main
tree and is read-only by convention: no experiment in this branch may pass a
`--save_dir` beneath it.
