"""NaN-insulated active-only zero-halo prototype for post-replay validation.

This is separate from the frozen Kodak replay implementation so the running
experiment and its source hash remain unchanged. For finite slots it must
produce the same output as ActiveCanvasCoupler; inactive slots are selected
away rather than multiplied by zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from proof.cpu_early_exit.active_canvas_coupling import ActiveCanvasCoupler


class SafeActiveCanvasCoupler(ActiveCanvasCoupler):
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

        # torch.where excludes poisoned inactive slot values from the output.
        pad[:, 1:, :, :, 0, 1:-1] = torch.where(
            edge[:, :-1], v[:, :-1, :, :, -1, :], 0)
        pad[:, :-1, :, :, -1, 1:-1] = torch.where(
            edge[:, 1:], v[:, 1:, :, :, 0, :], 0)
        pad[:, :, 1:, :, 1:-1, 0] = torch.where(
            edge[:, :, :-1], v[:, :, :-1, :, :, -1], 0)
        pad[:, :, :-1, :, 1:-1, -1] = torch.where(
            edge[:, :, 1:], v[:, :, 1:, :, :, 0], 0)
        pad[:, 1:, 1:, :, 0, 0] = torch.where(
            corner[:, :-1, :-1], v[:, :-1, :-1, :, -1, -1], 0)
        pad[:, 1:, :-1, :, 0, -1] = torch.where(
            corner[:, :-1, 1:], v[:, :-1, 1:, :, -1, 0], 0)
        pad[:, :-1, 1:, :, -1, 0] = torch.where(
            corner[:, 1:, :-1], v[:, 1:, :-1, :, 0, -1], 0)
        pad[:, :-1, :-1, :, -1, -1] = torch.where(
            corner[:, 1:, 1:], v[:, 1:, 1:, :, 0, 0], 0)
        haloed = pad.reshape(-1, v.shape[3], p + 2, p + 2)[active]
        return F.conv2d(haloed, conv.weight, conv.bias, stride=1, padding=0,
                        groups=conv.groups)
