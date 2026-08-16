"""Canvas-coupled tiling: let each tile's depthwise see its real neighbours.

The seam, restated
------------------
A tile decoded on its own meets its border with invented values, and every 3x3
in the per-tile trunk pushes that invention one pixel further inward. Everything
tried so far treats the symptom -- better invention (padding schemes) or repair
after the fact (SeamRepair, GridSeamRepair).

But look at what actually needs a neighbour. A DepthConvBlock is

    1x1 -> WSiLU -> 3x3 depthwise -> 1x1        (dc)
    1x1 -> WSiLUChunkAdd -> 1x1                 (ffn)

and at C=384 that is 1,035,648 MAC/px, of which the 3x3 depthwise is 3,456 --
**0.334%**. Every other operator is pointwise and could not care less what its
neighbours hold. So exactly 0.334% of the trunk is the entire cause of the seam.

Why the earlier halo was rejected, and why that was wrong
--------------------------------------------------------
Section 11 measured a trunk halo at 1.745x a full decode and dropped it. That
measurement haloed the WHOLE BLOCK -- it paid the (P+2h)^2/P^2 multiplier on the
99.666% that gains nothing from a halo. Haloing only the depthwise costs

    128px tile: +0.089% per block -> +0.066% of the decode
    256px tile: +0.043% per block -> +0.032% of the decode

against GridSeamRepair's 0.951%. Fourteen to thirty times cheaper, and it removes
the cause instead of repairing the effect.

What this does
--------------
The pre-depthwise activation of every tile lives in one shared canvas. Each
depthwise runs on that canvas, so a tile's border reads its neighbour's real
value. Tiles that have already exited keep their last activation in place and
go on serving as context for free.

The property that makes it worth doing: **when neighbouring tiles are at the
same depth, the result is bit-exact the full-frame decode.** Not approximately --
identically, because a depthwise on the assembled canvas IS the depthwise the
full-frame decoder would run. The seam does not get smaller; for uniform-depth
regions it stops existing. What remains is only the boundary between tiles that
chose DIFFERENT depths, where the neighbour's activation is real but shallower --
still a genuine feature of genuine content, not an invention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanvasCoupler:
    """Shared canvas the per-tile depthwise convolutions read across.

    One instance per decode. `begin` sizes it, the wrapped convolutions call
    `apply`, and `end` releases it. Deliberately explicit rather than inferred
    from tensor shapes: a coupler that silently guesses the tile grid would
    produce a plausible wrong answer instead of an error.
    """

    def __init__(self):
        self.slots = None
        self.nh = self.nw = self.batch = 0

    def begin(self, tiles: torch.Tensor, nh: int, nw: int, batch: int):
        # Every tile starts present: the ones that later exit simply stop being
        # updated, and their last value keeps serving as neighbour context.
        self.slots = tiles.clone()
        self.nh, self.nw, self.batch = nh, nw, batch

    def end(self):
        self.slots = None

    def apply(self, conv: nn.Conv2d, x: torch.Tensor,
              active: torch.Tensor) -> torch.Tensor:
        """Run `conv` (a 3x3 depthwise) over the assembled canvas."""
        if self.slots is None:
            return conv(x)
        self.slots = self.slots.index_copy(0, active, x)
        canvas = _unpatch(self.slots, self.nh, self.nw, self.batch)
        # padding=1 with ZEROS, exactly as stock Conv2d does at the frame border,
        # so a uniform-depth decode lands on the full-frame answer bit-exactly.
        out = F.conv2d(canvas, conv.weight, conv.bias, stride=1, padding=1,
                       groups=conv.groups)
        return _patch(out, self.slots.shape[-1])[active]


def _unpatch(tiles: torch.Tensor, nh: int, nw: int, batch: int) -> torch.Tensor:
    n, c, p, _ = tiles.shape
    return (tiles.view(batch, nh, nw, c, p, p)
                 .permute(0, 3, 1, 4, 2, 5)
                 .reshape(batch, c, nh * p, nw * p))


def _patch(x: torch.Tensor, p: int) -> torch.Tensor:
    b, c, h, w = x.shape
    return (x.view(b, c, h // p, p, w // p, p)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(-1, c, p, p))


class CoupledDepthwise(nn.Module):
    """A 3x3 depthwise that reads its halo from the shared canvas."""

    def __init__(self, conv: nn.Conv2d, coupler: CanvasCoupler):
        super().__init__()
        self.conv = conv
        object.__setattr__(self, "_coupler", coupler)

    def forward(self, x):
        c = self._coupler
        return c.apply(self.conv, x, c.active) if c.slots is not None else self.conv(x)


def install(groups: nn.ModuleList, first: int, coupler: CanvasCoupler):
    """Wrap every 3x3 depthwise in groups[first:]. Returns an undo fn."""
    targets = [
        (parent, name, child)
        for g in range(first, len(groups))
        for parent in groups[g].modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3)
        and child.groups > 1
    ]
    if any(isinstance(c, CoupledDepthwise) for _, _, c in targets):
        raise RuntimeError("canvas coupling already installed; run undo() first")
    for parent, name, child in targets:
        setattr(parent, name, CoupledDepthwise(child, coupler))

    def undo():
        for parent, name, child in targets:
            setattr(parent, name, child)
    return undo
