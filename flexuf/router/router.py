"""The router — ClassSR's Class-Module, adapted to a shared-weight exit ladder.

ClassSR (Kong et al., CVPR 2021) splits an image into sub-images, runs a small
Class-Module that scores each sub-image's restoration difficulty, and sends it to
one of three SR branches of differing capacity. We keep the structure and change
one thing: our "branches" are prefixes of a single decoder rather than three
separate networks, so routing costs no extra parameters and the deepest branch is
the unmodified codec.

Why the router reads the LATENT and gets a generous halo
--------------------------------------------------------
The router must decide a tile's depth *before* that tile is decoded, so its only
available input is y_hat. That turns out to be an advantage: the measured cost of
the whole routing path is ~0.009% of the decode, so context the router reads is
effectively free.

This matters because the halo cannot be free anywhere else. Measured
(`flexuf/cost.py`), carrying a 2-latent-pixel halo through the per-patch *trunk*
at j=2 with 128x128 tiles turns a +58.7% saving into **-74.5%** — the haloed
tiles compute 2.25x the pixels they keep, on the expensive part of the network.
The same halo on the *head* costs ~1.4% of the decode, and on the *router* costs
nothing measurable.

So the halo is placed where it is affordable and where it actually helps:
  - router input:  generous halo (default 4 latent px) — free, and it is what
                   keeps per-tile statistics from being contaminated by borders,
                   which was the concern that motivated a bigger halo.
  - head:          the configured halo — cheap, and FLEX measured it removes
                   99.1% of the seam.
  - per-patch trunk: no halo — it cannot be afforded.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FlexUFConfig


# ---------------------------------------------------------------------------
# Signals — closed-form, no parameters
# ---------------------------------------------------------------------------
@torch.no_grad()
def tile_signals(y_tiles: torch.Tensor) -> torch.Tensor:
    """Cheap statistics of a latent tile that predict how much decode it needs.

    Args:
        y_tiles: [P, C, h, w] latent tiles, halo included.
    Returns:
        [P, 4] float32 signals.

    The four are chosen to span the axes that make a tile hard:

      s1 rate surrogate   — a Gaussian code-length estimate per channel. Tiles
                            that cost more bits carry more structure and lose
                            more when decoded shallowly.
      s2 sparsity         — fraction of channels that are essentially dead. A
                            mostly-empty latent is flat content; flat content is
                            exactly what a shallow exit reconstructs well.
      s3 gradient energy  — mean absolute finite difference. Texture and edges.
      s4 spatial variance — variance of the per-position channel norm, i.e. how
                            *unevenly* energy is distributed inside the tile.

    All four are computed from the tile itself with no learned parameters, which
    is what keeps the router path at ~0.009% of the decode.
    """
    P, C, h, w = y_tiles.shape
    y = y_tiles.float()

    mu = y.mean(dim=(2, 3), keepdim=True)
    var = y.var(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    # (1/2ln2) * ( (y-mu)^2/var + ln(2*pi*var) ) summed over channels, meaned over space
    nll = 0.5 * ((y - mu) ** 2 / var + torch.log(2 * torch.pi * var)) / torch.log(
        torch.tensor(2.0, device=y.device)
    )
    s1 = nll.sum(dim=1).mean(dim=(1, 2))

    chan_energy = y.abs().mean(dim=(2, 3))
    s2 = (chan_energy < 1e-2).float().mean(dim=1)

    dw = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))
    dh = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
    s3 = (dw + dh) / (C * h * w)

    pos_norm = y.pow(2).sum(dim=1).sqrt()
    s4 = pos_norm.flatten(1).var(dim=1, unbiased=False)

    return torch.stack([s1, s2, s3, s4], dim=1)


# ---------------------------------------------------------------------------
# Class-Module
# ---------------------------------------------------------------------------
class ExitRouter(nn.Module):
    """Maps tile signals (+ the QP) to a distribution over the K exits.

    Deliberately tiny — a few thousand parameters — because anything larger would
    show up in the cost accounting it is supposed to be optimising.
    """

    def __init__(self, num_exits: int, n_signals: int = 4, hidden: int = 32):
        super().__init__()
        self.num_exits = num_exits
        # +1 input for the normalised QP: the same tile needs more decode at high
        # rate than at low rate, because the patch penalty grows with rate
        # (1.02 dB at qp30 vs 2.48 dB at qp63, FLEX measurement).
        self.norm = nn.BatchNorm1d(n_signals + 1)
        self.mlp = nn.Sequential(
            nn.Linear(n_signals + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_exits),
        )

    def forward(self, signals: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
        """Returns logits [P, K]."""
        q = qp.reshape(-1, 1).float() / 63.0
        if q.shape[0] != signals.shape[0]:
            q = q.expand(signals.shape[0], 1)
        return self.mlp(self.norm(torch.cat([signals, q], dim=1)))

    def probabilities(self, signals, qp, tau: float = 1.0) -> torch.Tensor:
        return F.softmax(self.forward(signals, qp) / tau, dim=1)

    @torch.no_grad()
    def assign(self, signals, qp) -> torch.Tensor:
        """Hard exit assignment used at inference: argmax, as in ClassSR."""
        return self.forward(signals, qp).argmax(dim=1)


# ---------------------------------------------------------------------------
# Extracting the router's view of a frame
# ---------------------------------------------------------------------------
def latent_tiles_with_halo(
    y_hat: torch.Tensor, cfg: FlexUFConfig, router_halo: Optional[int] = None
) -> torch.Tensor:
    """Split y_hat into the tiles the decoder will use, each with a halo.

    The halo here is generous by default (4 latent px against a 2 latent px
    decode halo) precisely because it is free: the router is ~0.009% of the
    decode, so context costs nothing, and uncontaminated tile statistics are what
    stop the router learning to route on border artefacts instead of on content.
    """
    h = cfg.latent_halo * 2 if router_halo is None else router_halo
    p = cfg.latent_patch
    b, c, H, W = y_hat.shape
    assert H % p == 0 and W % p == 0, f"latent {H}x{W} not divisible by patch {p}"
    nh, nw = H // p, W // p

    padded = F.pad(y_hat, (h,) * 4, mode="replicate")
    side = p + 2 * h
    tiles = padded.unfold(2, side, p).unfold(3, side, p)
    return tiles.permute(0, 2, 3, 1, 4, 5).reshape(b * nh * nw, c, side, side).contiguous()
