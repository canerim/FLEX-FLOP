"""Boundary-condition controls for the experimental active canvas."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from flexuf.backbone.coupling import CanvasCoupler
from proof.cpu_early_exit.active_canvas_coupling import ActiveCanvasCoupler
from proof.cpu_early_exit.active_canvas_replicate import ActiveCanvasReplicateCoupler


class ActiveCanvasBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261005)
        self.tiles = torch.randn(9, 4, 7, 7)
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4).eval()

    def run_coupler(self, kind: type[CanvasCoupler], active: torch.Tensor) -> torch.Tensor:
        coupler = kind()
        coupler.begin(self.tiles, 3, 3, 1)
        return coupler.apply(self.conv, self.tiles[active], active)

    def test_all_active_matches_full_canvas(self) -> None:
        active = torch.arange(9)
        reference = self.run_coupler(CanvasCoupler, active)
        for kind in (ActiveCanvasCoupler, ActiveCanvasReplicateCoupler):
            torch.testing.assert_close(self.run_coupler(kind, active), reference,
                                       rtol=0, atol=0)

    def test_only_center_active_uses_declared_fallback(self) -> None:
        active = torch.tensor([4])
        center = self.tiles[active]
        zero = self.conv(center)
        replicate = F.conv2d(F.pad(center, (1, 1, 1, 1), mode='replicate'),
                             self.conv.weight, self.conv.bias, groups=4)
        torch.testing.assert_close(self.run_coupler(ActiveCanvasCoupler, active),
                                   zero, rtol=0, atol=0)
        torch.testing.assert_close(self.run_coupler(ActiveCanvasReplicateCoupler, active),
                                   replicate, rtol=0, atol=0)

    def test_partial_active_matches_masked_full_canvas_across_batch(self) -> None:
        tiles = torch.cat((self.tiles, self.tiles.flip(-1)), dim=0)
        active = torch.tensor([0, 4, 5, 8, 9, 10, 13, 17])
        coupler = ActiveCanvasCoupler()
        coupler.begin(tiles, 3, 3, 2)
        actual = coupler.apply(self.conv, tiles[active], active)
        canvas = torch.zeros_like(tiles)
        canvas[active] = tiles[active]
        p = tiles.shape[-1]
        whole = (canvas.view(2, 3, 3, 4, p, p).permute(0, 3, 1, 4, 2, 5)
                 .reshape(2, 4, 3*p, 3*p))
        output = self.conv(whole)
        expected = (output.view(2, 4, 3, p, 3, p).permute(0, 2, 4, 1, 3, 5)
                    .reshape(18, 4, p, p))[active]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
