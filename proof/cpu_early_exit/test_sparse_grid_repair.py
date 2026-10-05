"""Check sparse pointwise repair against its dense thresholded definition."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flexuf.backbone.decoder import GridSeamRepair
from proof.cpu_early_exit.sparse_grid_repair import apply_sparse_repair, make_plan


class SparseRepairTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar").exists(),
                         "frozen e15 checkpoint unavailable")
    def test_frozen_e15_gate_and_pointwise(self):
        checkpoint = torch.load(ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar",
                                map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        prefix = "dec.seam_repair."
        module = GridSeamRepair(channels=384, patch=32).eval()
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)
        torch.manual_seed(71)
        torch.set_num_threads(1)
        x = torch.randn(1, 384, 32, 32)
        with torch.inference_mode():
            plan = make_plan(module, batch=1, height=32, width=32,
                             threshold=.25, device=x.device, dtype=x.dtype)
            sparse = apply_sparse_repair(x, module, plan)
            g = module.gate
            dense = x + g * (g >= .25) * module.pw(module.act(module.dw(x)))
            self.assertEqual(plan.active_fraction, 139 / 1024)
            self.assertTrue(torch.equal(sparse, dense))

    def test_dense_threshold_equivalence(self):
        torch.manual_seed(1941)
        torch.set_num_threads(1)
        module = GridSeamRepair(channels=8, patch=4).eval()
        with torch.no_grad():
            module.pw.weight.normal_(0, .02)
            module.pw.bias.normal_(0, .01)
            module.gate.copy_(torch.tensor([[[[.20, .32, .55, .78],
                                               [.21, .25, .42, .60],
                                               [.23, .26, .34, .50],
                                               [.24, .27, .31, .48]]]]))
            x = torch.randn(2, 8, 7, 11)
            original = x.clone()
            g = module.gate.repeat(1, 1, 2, 3)[:, :, :7, :11]
            correction = module.pw(module.act(module.dw(x)))
            for threshold in (0.0, .25, .40, 1.0):
                plan = make_plan(module, batch=2, height=7, width=11,
                                 threshold=threshold, device=x.device, dtype=x.dtype)
                sparse = apply_sparse_repair(x, module, plan)
                dense = x + g * (g >= threshold) * correction
                self.assertLess(float((sparse - dense).abs().max()), 2e-6)
                self.assertTrue(torch.equal(x, original))
            self.assertTrue(torch.equal(apply_sparse_repair(x, module,
                make_plan(module, batch=2, height=7, width=11,
                          threshold=1.0, device=x.device, dtype=x.dtype)), x))

    def test_invalidated_gate_is_rejected(self):
        module = GridSeamRepair(channels=8, patch=4).eval()
        x = torch.zeros(1, 8, 4, 4)
        plan = make_plan(module, batch=1, height=4, width=4,
                         threshold=.25, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            module.gate.add_(.01)
            with self.assertRaisesRegex(RuntimeError, "Gate changed"):
                apply_sparse_repair(x, module, plan)


if __name__ == "__main__":
    unittest.main()
