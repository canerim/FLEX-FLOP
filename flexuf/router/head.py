"""A router that LEARNS what to look at, instead of being told.

Why the hand-made signals had to go
-----------------------------------
The router chose an exit from six scalars computed off the stem: max, energy,
std, two entropy-model statistics and a latent energy. Measured on the warm-start
checkpoint, none of them correlates past |r| = 0.12 with the oracle's choice --
while the oracle itself varies (std 0.348 exits). The distinction is in the
content and the router cannot see it, which is exactly why the first regret
sweep produced a constant router.

The obvious repair -- give it more of the same, a 768-dim pooled stem -- was
tried and MEASURED WORSE at the operating point that matters (0.609 against
0.719 on held-out tiles). So the problem is not the width of the description. It
is that nobody ever asked the network what a routable description would be.

This head does. A 1x1 convolution learns an R-channel view of the stem, pooled
per tile into (mean, std), and an MLP maps that plus the QP to per-exit logits.
Trained jointly with the decoder, the stem is free to become something worth
routing on rather than something we hope already is.

Cost, the constraint that shaped the design
-------------------------------------------
The stem lives at 1/64 of the RGB pixel count, so a 1x1 of 384 -> 16 there is

    384 * 16 = 6,144 MAC per feature pixel = 96 MAC per RGB pixel

against the decode's 217,113 -- **0.044%**. For scale: GridSeamRepair is 0.951%,
canvas coupling 0.032%, one trunk block 7.45%. The pooling and the MLP act on
one vector per tile and do not register at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemRouterHead(nn.Module):
    """Per-tile exit logits, read from a learned view of the stem."""

    def __init__(self, in_ch: int, num_exits: int, min_exit: int = 0,
                 rank: int = 16, hidden: int = 64):
        super().__init__()
        self.min_exit = min_exit
        self.proj = nn.Conv2d(in_ch, rank, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * rank + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, num_exits),
        )
        # Start near-uniform rather than at zero: a zero-init head would make the
        # Gumbel sample uniform over exits, which is a fine starting policy, but
        # a *dead* one -- with identical logits every tile gets the same gradient
        # and there is nothing to break the symmetry. Small random logits give
        # the tiles somewhere to diverge from.
        nn.init.normal_(self.mlp[-1].weight, std=1e-2)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stem: torch.Tensor, qp: torch.Tensor,
                feature_patch: int) -> torch.Tensor:
        """stem [B,C,h,w] -> logits [B*nh*nw, K], tiles in unpatchify order."""
        z = self.proj(stem)
        B, R, h, w = z.shape
        p = feature_patch
        v = (z.view(B, R, h // p, p, w // p, p)
              .permute(0, 2, 4, 1, 3, 5)
              .reshape(-1, R, p * p))
        d = torch.cat([v.mean(-1), v.std(-1)], dim=1)          # [n_tiles, 2R]
        n_per = d.shape[0] // B
        q = (qp.reshape(-1, 1).float() / 63.0).repeat_interleave(n_per, 0)
        logits = self.mlp(torch.cat([d, q], dim=1))
        if self.min_exit > 0:
            # Exits shallower than the split are not decodable per tile; masking
            # here rather than clamping later keeps the probabilities a real
            # distribution over the choices that exist.
            #
            # KNOWN BUG, NOT FIXED HERE. -1e4 is a mask only while this head's
            # own logits stay well above it. head2.py's did not -- nothing in a
            # cross-entropy or a regret objective penalises a common offset, one
            # drifted in, and the raw outputs settled near -10000, at which point
            # the "mask" became the largest entry in every row (DECISIONS 89).
            # head2 now uses -inf. This file is imported by live training runs,
            # and a crash-restart would pick up an edit mid-experiment, so the
            # change waits until nothing is running. Callers should slice
            # logits[:, min_exit:] rather than trust this line.
            logits = logits.clone()
            logits[:, : self.min_exit] = -1e4
        return logits


def gumbel_exits(logits: torch.Tensor, tau: float = 1.0, hard: bool = True):
    """Sample one exit per tile, differentiably.

    Returns (exit_index [n], p_selected [n]). `p_selected` is the straight-
    through handle: multiplying a tile's decoded output by
    p / p.detach() leaves the forward pass untouched and lets the RD loss push
    on the router's logits, which a plain argmax would not.
    """
    y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
    idx = y.argmax(1)
    return idx, y.gather(1, idx[:, None]).squeeze(1)
