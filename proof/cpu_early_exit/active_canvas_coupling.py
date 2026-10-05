"""Experimental depthwise canvas coupling with an active-neighbour mask.

The deployed decoder's experimental CanvasCoupler retains exited tiles as
context. Their features become stale as a neighbouring tile continues deeper.
This diagnostic coupler reads across a tile edge only while the source tile is
still active in the same suffix group; otherwise it uses the stock zero halo.
No model weights or route decisions are changed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flexuf.backbone.coupling import CanvasCoupler


class ActiveCanvasCoupler(CanvasCoupler):
    def apply(self, conv: torch.nn.Conv2d, x: torch.Tensor,
              active: torch.Tensor) -> torch.Tensor:
        if self.slots is None:
            return conv(x)
        n_tiles = self.batch * self.nh * self.nw
        assert x.shape[0] == active.numel()
        assert active.min() >= 0 and active.max() < n_tiles
        p = self.slots.shape[-1]
        self.slots = self.slots.index_copy(0, active, x)
        v = self.slots.view(self.batch, self.nh, self.nw, -1, p, p)
        live = torch.zeros(n_tiles, dtype=torch.bool, device=x.device)
        live[active] = True
        edge = live.view(self.batch, self.nh, self.nw, 1, 1)
        corner = edge[..., 0]
        pad = x.new_zeros(self.batch, self.nh, self.nw, v.shape[3], p + 2, p + 2)
        pad[..., 1:-1, 1:-1] = v
        pad[:, 1:, :, :, 0, 1:-1] = v[:, :-1, :, :, -1, :] * edge[:, :-1]
        pad[:, :-1, :, :, -1, 1:-1] = v[:, 1:, :, :, 0, :] * edge[:, 1:]
        pad[:, :, 1:, :, 1:-1, 0] = v[:, :, :-1, :, :, -1] * edge[:, :, :-1]
        pad[:, :, :-1, :, 1:-1, -1] = v[:, :, 1:, :, :, 0] * edge[:, :, 1:]
        pad[:, 1:, 1:, :, 0, 0] = v[:, :-1, :-1, :, -1, -1] * corner[:, :-1, :-1]
        pad[:, 1:, :-1, :, 0, -1] = v[:, :-1, 1:, :, -1, 0] * corner[:, :-1, 1:]
        pad[:, :-1, 1:, :, -1, 0] = v[:, 1:, :-1, :, 0, -1] * corner[:, 1:, :-1]
        pad[:, :-1, :-1, :, -1, -1] = v[:, 1:, 1:, :, 0, 0] * corner[:, 1:, 1:]
        haloed = pad.reshape(-1, v.shape[3], p + 2, p + 2)[active]
        return F.conv2d(haloed, conv.weight, conv.bias, stride=1, padding=0,
                        groups=conv.groups)
