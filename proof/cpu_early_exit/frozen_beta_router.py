"""Recompute frozen stem/QP router maps for validation-only beta choices.

The original DIV2K validation archive captured only the old locked beta,
not every calibration-grid candidate. This helper derives a requested map
from the same decoded latent, e15 stem, router checkpoint and cost vector.
"""
from __future__ import annotations

from pathlib import Path


class FrozenBetaRouter:
    def __init__(self, cfg, router_path: Path, calibration_path: Path):
        import torch
        from bitstream_benchmark import load_router

        class Args:
            router = router_path
            calibration = calibration_path

        self.cfg = cfg
        self.head, costs, self.archived_beta, _ = load_router(
            Args(), cfg, torch.device('cpu'))
        self.cost = costs[cfg.split_depth:].double()

    def scores(self, decoder, latent, qp: int):
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        stem = decoder.upsample(latent)
        for group in decoder.groups[:cfg.split_depth]:
            stem = group(stem)
        logits = self.head(stem, latent, latent,
                           torch.tensor([qp], dtype=torch.int32),
                           cfg.feature_patch, cfg.latent_patch)
        return F.log_softmax(logits[:, cfg.split_depth:], dim=1).double()

    def route(self, scores, beta: float):
        cfg = self.cfg
        return (scores - float(beta) * self.cost[None, :]).argmax(1).cpu() + cfg.split_depth
