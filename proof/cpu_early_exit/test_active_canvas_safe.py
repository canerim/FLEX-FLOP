"""Finite equivalence and poisoned-inactive-slot control for safe masking."""
from __future__ import annotations

import unittest

import torch

from proof.cpu_early_exit.active_canvas_coupling import ActiveCanvasCoupler
from proof.cpu_early_exit.active_canvas_safe import SafeActiveCanvasCoupler


class SafeActiveCanvasTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261005)
        self.tiles = torch.randn(18, 4, 7, 7)
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4).eval()

    def run_case(self, kind, active: torch.Tensor, poison: bool) -> torch.Tensor:
        c = kind()
        c.begin(self.tiles, 3, 3, 2)
        if poison:
            c.slots[:] = float('nan')
        return c.apply(self.conv, self.tiles[active], active)

    def test_finite_matches_frozen_replay_implementation(self) -> None:
        for active in (torch.arange(18), torch.tensor([4, 13]),
                       torch.tensor([0, 1, 4, 5, 8, 9, 13, 17])):
            with self.subTest(active=active.tolist()):
                frozen = self.run_case(ActiveCanvasCoupler, active, False)
                safe = self.run_case(SafeActiveCanvasCoupler, active, False)
                torch.testing.assert_close(safe, frozen, rtol=0, atol=0)

    def test_inactive_poison_cannot_reach_active_outputs(self) -> None:
        for active in (torch.tensor([4, 13]),
                       torch.tensor([0, 1, 4, 5, 8, 9, 13, 17])):
            with self.subTest(active=active.tolist()):
                reference = self.run_case(SafeActiveCanvasCoupler, active, False)
                poisoned = self.run_case(SafeActiveCanvasCoupler, active, True)
                self.assertTrue(torch.isfinite(poisoned).all())
                torch.testing.assert_close(poisoned, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
