"""Router v2: a learned view of everything the decoder already has.

What v1 got wrong, measured
---------------------------
v1 read a 1x1 of rank 16 off the stem, pooled it, and passed 33 numbers through a
64-unit MLP. Three defects, each found by measurement rather than taste:

1. **It ignored the entropy model.** Of the six hand-made signals, the two with
   the strongest correlation to the oracle's choice were `scales_mean` (-0.117)
   and `scales_max` (-0.097) -- the entropy model's predicted Gaussian scales.
   They are computed during the decode and were thrown away. So was the latent
   itself. The router was denied the one quantity that is literally a per-tile
   difficulty estimate.

2. **Capacity was in the wrong place.** The MLP runs on ONE vector per tile --
   40 of them for a 1080p frame at 256px tiles -- so its width is free, while the
   1x1 runs per pixel and is the only thing that costs. v1 kept the MLP narrow
   and the 1x1 narrow, paying for caution twice.

3. **Its objective was indirect.** Expected regret is the right thing to
   MINIMISE, but agreement with the oracle is what was asked for, and regret only
   reaches it via consequences. Cross-entropy to the oracle's own choice supplies
   the direct signal; weighting each tile by how much the choice COSTS keeps it
   honest, so the router spends its capacity where a mistake matters.

Cost, still the binding constraint
----------------------------------
    stem 1x1 384->48, at 1/64 of the pixel count      288 MAC/RGB-px   0.133%
    latent+scales 1x1 512->32, at 1/256                64              0.030%
    MLP, one vector per tile                            ~0
    ------------------------------------------------------------------------
                                                                       0.162%

GridSeamRepair is 0.951% and a trunk block 7.45%, so this is still small change
against the thing it is deciding about.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool(z: torch.Tensor, patch: int) -> torch.Tensor:
    """[B,C,h,w] -> [B*nh*nw, 2C] as (mean, std) per tile, unpatchify order."""
    B, C, h, w = z.shape
    v = (z.view(B, C, h // patch, patch, w // patch, patch)
          .permute(0, 2, 4, 1, 3, 5)
          .reshape(-1, C, patch * patch))
    return torch.cat([v.mean(-1), v.std(-1)], dim=1)


class StemRouterHeadV2(nn.Module):
    def __init__(self, stem_ch: int, latent_ch: int, num_exits: int,
                 min_exit: int = 0, r_stem: int = 48, r_lat: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.min_exit = min_exit
        self.proj_stem = nn.Conv2d(stem_ch, r_stem, 1)
        self.proj_lat = nn.Conv2d(2 * latent_ch, r_lat, 1)
        d = 2 * r_stem + 2 * r_lat + 1
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_exits),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-2)
        nn.init.zeros_(self.mlp[-1].bias)
        # Loss-free load balancing (arXiv:2408.15664 / OpenReview y1iU5czYpE):
        # a per-exit bias added to the scores BEFORE the decision, nudged by usage
        # rather than by a gradient. An auxiliary balance loss injects
        # interference gradients into the shared objective; this does not touch
        # the gradient at all. Registered as a buffer because it is updated by
        # rule, not learned.
        self.register_buffer("bias", torch.zeros(num_exits))

    def forward(self, stem, y_hat, scales, qp, feature_patch, latent_patch):
        a = _pool(self.proj_stem(stem), feature_patch)
        if scales.shape[1] != y_hat.shape[1]:
            scales = scales[:, : y_hat.shape[1]]
        b = _pool(self.proj_lat(torch.cat([y_hat, scales], 1)), latent_patch)
        n_per = a.shape[0] // stem.shape[0]
        q = (qp.reshape(-1, 1).float() / 63.0).repeat_interleave(n_per, 0)
        logits = self.mlp(torch.cat([a, b, q], dim=1)) + self.bias[None, :]
        if self.min_exit > 0:
            logits = logits.clone()
            logits[:, : self.min_exit] = -1e4
        return logits

    @torch.no_grad()
    def rebalance(self, chosen: torch.Tensor, target: torch.Tensor, rate=1e-2):
        """Nudge the bias so usage drifts toward `target` (the oracle's own mix).

        Balanced toward the ORACLE's distribution, not toward uniform. Uniform is
        what MoE wants because its experts are interchangeable; ours are not --
        at a high lambda the oracle genuinely sends everything to one exit, and
        forcing spread there would be forcing mistakes.
        """
        used = torch.bincount(chosen, minlength=self.bias.numel()).float()
        used = used / used.sum().clamp_min(1)
        self.bias += rate * (target - used)
        self.bias -= self.bias.mean()


def oracle_ce_loss(logits, mses, costs, lam: float, label_smooth: float = 0.0):
    """Cost-sensitive cross-entropy to the oracle's choice.

    Plain CE treats every tile equally, but they are not: for many tiles two
    exits are within a hair of each other and picking either is fine, while for a
    few the wrong pick is most of the frame's error. Weighting each tile by the
    REGRET of its worst alternative concentrates capacity where the decision
    actually matters, and leaves the ties alone.

    Returns (loss, oracle_label, agreement).
    """
    lag = mses + lam * costs[None, :]
    best, k = lag.min(dim=1)
    # How much is at stake on this tile: spread between best and worst option.
    stake = (lag.max(dim=1).values - best)
    w = stake / stake.mean().clamp_min(1e-12)
    ce = F.cross_entropy(logits, k, reduction="none", label_smoothing=label_smooth)
    agree = (logits.argmax(1) == k).float().mean()
    return (w * ce).mean(), k, agree
