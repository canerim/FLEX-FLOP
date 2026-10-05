"""Active-neighbour coupling with deployed replicate padding as the fallback.

An active neighbouring tile supplies its real feature. Once that neighbour
exits, its stale feature is never read; the active tile instead uses its own
edge value, matching the deployed decoder's internal-tile padding. At the
outer image boundary padding remains zero, as in the full-frame decoder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flexuf.backbone.coupling import CanvasCoupler


class ActiveCanvasReplicateCoupler(CanvasCoupler):
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
        pad = F.pad(self.slots, (1, 1, 1, 1), mode='replicate')
        pad = pad.view(self.batch, self.nh, self.nw, v.shape[3], p+2, p+2)
        # Full-frame convolutions use zero at the outside of the image.
        pad[:, 0, :, :, 0, :] = 0
        pad[:, -1, :, :, -1, :] = 0
        pad[:, :, 0, :, :, 0] = 0
        pad[:, :, -1, :, :, -1] = 0
        pad[:, 1:, :, :, 0, 1:-1] = torch.where(edge[:, :-1], v[:, :-1, :, :, -1, :], pad[:, 1:, :, :, 0, 1:-1])
        pad[:, :-1, :, :, -1, 1:-1] = torch.where(edge[:, 1:], v[:, 1:, :, :, 0, :], pad[:, :-1, :, :, -1, 1:-1])
        pad[:, :, 1:, :, 1:-1, 0] = torch.where(edge[:, :, :-1], v[:, :, :-1, :, :, -1], pad[:, :, 1:, :, 1:-1, 0])
        pad[:, :, :-1, :, 1:-1, -1] = torch.where(edge[:, :, 1:], v[:, :, 1:, :, :, 0], pad[:, :, :-1, :, 1:-1, -1])
        pad[:, 1:, 1:, :, 0, 0] = torch.where(corner[:, :-1, :-1], v[:, :-1, :-1, :, -1, -1], pad[:, 1:, 1:, :, 0, 0])
        pad[:, 1:, :-1, :, 0, -1] = torch.where(corner[:, :-1, 1:], v[:, :-1, 1:, :, -1, 0], pad[:, 1:, :-1, :, 0, -1])
        pad[:, :-1, 1:, :, -1, 0] = torch.where(corner[:, 1:, :-1], v[:, 1:, :-1, :, 0, -1], pad[:, :-1, 1:, :, -1, 0])
        pad[:, :-1, :-1, :, -1, -1] = torch.where(corner[:, 1:, 1:], v[:, 1:, 1:, :, 0, 0], pad[:, :-1, :-1, :, -1, -1])
        haloed = pad.reshape(-1, v.shape[3], p+2, p+2)[active]
        return F.conv2d(haloed, conv.weight, conv.bias, stride=1, padding=0,
                        groups=conv.groups)
