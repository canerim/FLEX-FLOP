"""Does the router do better when it can see a tile's neighbours?

The head reads five groups, and the ablation in the paper's Section F says the
one that carries most of the per-tile signal is the stem, with the entropy
coder's bits close behind. Every one of those groups is read at the tile: the
bits pathway normalises the rate by the FRAME mean, so a tile knows how it
compares with the picture and not with what is next to it.

Difficulty is spatially correlated -- that is why the exit maps in the paper
have contiguous regions rather than salt and pepper -- so the neighbourhood
mean is information the head does not have and could have for nothing. This
adds it: two more channels on the bits pathway, the rate box-filtered over a
three-tile window and its log, so the same 1x1 sees the local rate and the
neighbourhood it sits in.

Nothing under ~/FLEX-UF is modified. The head is subclassed here, the decoder
and the oracle are imported and read.

    python flexplus/neighbour_bits.py --device cuda:0 --steps 3000
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF))
sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.cost import exit_costs                          # noqa: E402
from flexuf.eval import tiled_exit_mses                     # noqa: E402
from flexuf.router.head2 import (StemRouterHeadV2, bits_features,  # noqa: E402
                                 _pool, oracle_ce_loss)
from flexuf.router.losses import regret_objective  # noqa: E402


def neighbour_bits_features(bits, tile_lat):
    """The four channels: the frame-normalised rate, its log, and both again
    box-filtered over a three-tile window centred on the tile."""
    base = bits_features(bits)                      # [B, 2, h, w]
    k = max(1, int(tile_lat))
    pad = k                                          # one tile on each side
    box = F.avg_pool2d(F.pad(base[:, :1], (pad,) * 4, mode="replicate"),
                       kernel_size=2 * k + 1, stride=1)
    box = box[..., :base.shape[-2], :base.shape[-1]]
    return torch.cat([base, box, box.clamp_min(1e-6).log()], dim=1)


class NeighbourHead(StemRouterHeadV2):
    """The paper's head with two more channels on the bits pathway."""

    def __init__(self, *a, tile_lat: int = 16, **kw):
        super().__init__(*a, **kw)
        self.tile_lat = int(tile_lat)
        r = self.r_bits
        old = self.proj_bits
        # Building the wider convolution draws from the global RNG, which
        # would leave the data loader shuffling in a different order than in
        # the sweep this run is compared against. The draws are made and the
        # state put back, so the batch order and every parameter the parent
        # built are the ones the "all" variant of the ablation saw.
        _st = torch.get_rng_state()
        self.proj_bits = nn.Conv2d(4, r, 1)
        torch.set_rng_state(_st)
        with torch.no_grad():
            # Start from the trained-from-scratch equivalent of the two-channel
            # head: the first two channels keep their initialisation, the two
            # new ones start at zero, so step zero is the paper's head exactly
            # and anything it gains is gained during training.
            self.proj_bits.weight.zero_()
            self.proj_bits.weight[:, :2] = old.weight
            self.proj_bits.bias.copy_(old.bias)

    def forward(self, stem, y_hat, scales, qp, feature_patch, latent_patch,
                bits=None):
        # Same as the parent, with the wider bits features. Copied rather than
        # hooked because the parent builds `parts` inline.
        B = stem.shape[0]
        n_tiles = B * (stem.shape[2] // feature_patch) * (stem.shape[3] // feature_patch)
        a = _pool(self.proj_stem(stem), feature_patch) if self.gate["stem"] else \
            stem.new_zeros(n_tiles, 2 * self.proj_stem.out_channels)
        if scales.shape[1] != y_hat.shape[1]:
            scales = scales[:, : y_hat.shape[1]]
        p = self.proj_lat(torch.cat([y_hat, scales], 1))
        b = _pool(p, latent_patch)
        q = (qp.reshape(-1, 1).float() / 63.0).repeat_interleave(n_tiles // B, 0)
        parts = [a, b]
        if self.with_bits:
            parts.append(_pool(self.proj_bits(
                neighbour_bits_features(bits, self.tile_lat)), latent_patch))
        parts.append(q)
        return self.mlp(torch.cat(parts, dim=1)) + self.bias[None, :]
