"""Tile-border padding schemes, and a way to swap them inside a live decoder.

Under a j-split the per-tile trunk blocks meet the tile border with nothing
beyond it, and what gets invented there is the seam. PyTorch's Conv2d offers
only zeros / reflect / replicate / circular, so anything else has to wrap the
convolution: pre-pad by hand, then convolve with padding=0.

Only 3x3 depthwise convolutions matter — they are the sole operator in a
DepthConvBlock with any spatial extent (9C MAC/px against the block's 8C^2+9C,
i.e. 0.3%), so they are the only place a border can be met at all.

The schemes
-----------
zeros       stock. A terrible estimate of a neighbour.
replicate   copy the edge. Free, and measured to remove 28% of the seam at qp63.
linear      first-order extrapolation, pad = 2*x0 - x1. Continues the local
            gradient instead of flattening it; still free.
arls        per-channel AR(1) fitted by least squares over the tile, following
            Kaseva et al., "Per-channel autoregressive linear prediction padding
            in tiled CNN processing of 2D spatial data" (arXiv:2502.12300). Their
            own conclusion is worth repeating: it only "slightly reduced" the
            error against zero and replication padding, at "a moderate increase
            in time cost", and they suggest cropping the output with cheap
            padding may be preferable. Implemented here so the question is
            settled by measurement on THIS decoder rather than by their
            workload's answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extrapolate(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Pad [B,C,H,W] by one pixel on every side under the chosen scheme."""
    if mode == "zeros":
        return F.pad(x, (1, 1, 1, 1))
    if mode == "replicate":
        return F.pad(x, (1, 1, 1, 1), mode="replicate")

    if mode == "linear":
        # pad = 2*nearest - next: continues the local slope outward.
        left = 2 * x[:, :, :, :1] - x[:, :, :, 1:2]
        right = 2 * x[:, :, :, -1:] - x[:, :, :, -2:-1]
        x = torch.cat([left, x, right], dim=3)
        top = 2 * x[:, :, :1, :] - x[:, :, 1:2, :]
        bot = 2 * x[:, :, -1:, :] - x[:, :, -2:-1, :]
        return torch.cat([top, x, bot], dim=2)

    if mode == "arls":
        # Per-channel AR(1) along each axis, coefficient fitted by least squares
        # over the tile: a = <x_t, x_{t+1}> / <x_{t+1}, x_{t+1}>, so the
        # extrapolated value is a * nearest. Reduces to replicate when the
        # channel is perfectly correlated and to zeros when it is uncorrelated,
        # which is the property that makes it a principled interpolation between
        # the two cheap schemes.
        def coef(a, b):
            num = (a * b).sum(dim=(2, 3), keepdim=True)
            den = (b * b).sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            return (num / den).clamp(-1.5, 1.5)

        ax = coef(x[:, :, :, :-1], x[:, :, :, 1:])
        left = ax * x[:, :, :, :1]
        right = ax * x[:, :, :, -1:]
        x = torch.cat([left, x, right], dim=3)
        ay = coef(x[:, :, :-1, :], x[:, :, 1:, :])
        top = ay * x[:, :, :1, :]
        bot = ay * x[:, :, -1:, :]
        return torch.cat([top, x, bot], dim=2)

    raise ValueError(f"unknown tile padding mode {mode!r}")


class PaddedDepthwise(nn.Module):
    """Wrap a 3x3 depthwise conv so its border padding is ours, not PyTorch's."""

    def __init__(self, conv: nn.Conv2d, mode: str):
        super().__init__()
        self.conv = conv
        self.mode = mode

    def forward(self, x):
        return F.conv2d(_extrapolate(x, self.mode), self.conv.weight,
                        self.conv.bias, stride=1, padding=0,
                        dilation=1, groups=self.conv.groups)


def wrap_tile_padding(groups: nn.ModuleList, first: int, mode: str):
    """Swap every 3x3 depthwise in groups[first:] to `mode`. Returns an undo fn."""
    # Collect first, mutate after: setattr during a modules() walk inserts the
    # wrapper into the tree being walked, and the walk descends into it forever.
    targets = [
        (parent, name, child)
        for g in range(first, len(groups))
        for parent in groups[g].modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3)
        and child.groups > 1
    ]
    swapped = []
    for parent, name, child in targets:
        setattr(parent, name, PaddedDepthwise(child, mode))
        swapped.append((parent, name, child))

    def undo():
        for parent, name, child in swapped:
            setattr(parent, name, child)

    return undo
