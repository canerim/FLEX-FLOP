"""What each exit actually costs, in units of the full decode — halo included.

Every "compute saved" number the project reports comes from here, so the model is
written out rather than buried in a constant, and the halo is charged honestly.

The measured base
-----------------
`scripts/mac_audit.py`, 1920x1088, forward hooks on every nn.Conv2d:

    opening upsample     37.01 GMAC    8.16%   always runs, full-frame
    12-block trunk      405.64 GMAC   89.44%   the routable part
    head                 10.89 GMAC    2.40%   always runs
    ------------------------------------------------------
    total               453.54 GMAC  100.00%

Shares are resolution-independent (every op runs at a fixed multiple of the
latent grid), so they are used as fractions.

Why the halo must be charged
----------------------------
A tile decoded in isolation has no neighbours, so its border pixels get convolved
against padding. The fix is to carry a halo of real context, but a haloed tile is
physically larger: a tile of side F carried with halo h computes
((F + 2h) / F)^2 times as many pixels as it keeps.

At F = 16 feature px (our 128x128 RGB tile) and h = 4 feature px (2 latent px):

    (16 + 8)^2 / 16^2 = 2.25x

If that multiplier were applied to the whole per-patch trunk it would consume
more than the routing saves — the arithmetic simply does not work. FLEX avoided
this by haloing **only the head**, which is 2.40% of MACs, so its 1.5625x cost
was +1.35% of the decode overall.

So the halo placement is a real design axis, not a detail, and this module prices
both placements so the experiments can be compared on true net saving:

    halo_scope="head"   — cheap; the per-patch trunk runs unhaloed and the
                          adapters are trained to absorb the residual seam.
    halo_scope="trunk"  — the per-patch trunk carries the halo too. Cleaner tile
                          statistics (which is what the router reads), but the
                          2.25x lands on the expensive part.
    halo_scope="taper"  — carry the halo into the trunk but crop one pixel per
                          block, since each 3x3 depthwise shrinks the valid
                          region by exactly 1. Cost is the mean of a shrinking
                          sequence rather than the worst case.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .config import N_TRUNK_BLOCKS, FlexUFConfig

# Measured — scripts/mac_audit.py.
SHARE_UPSAMPLE = 0.0816
SHARE_TRUNK = 0.8944
SHARE_HEAD = 0.0240

_C = 384
# Adapter cost relative to one DepthConvBlock (8C^2 + 9C MAC/px).
_BLOCK_MACPX = 8 * _C**2 + 9 * _C
ADAPTER_MACPX = {
    "conv1x1": _C**2,                 # one 1x1
    "ffn": 2 * _C**2,                 # PW C->4C then PW C->C, chunk-add in between
}


def adapter_vs_block(kind: str = "conv1x1") -> float:
    return ADAPTER_MACPX[kind] / _BLOCK_MACPX


# The seam-repair pass runs once, full-frame, on the stitched canvas.
#   depthwise 3x3 : 9C
#   pointwise 1x1 : C^2
# expressed as a share of the whole decode via the trunk's per-block share.
SEAM_REPAIR_MACPX = {"none": 0.0, "depthwise": 9 * _C, "full": 9 * _C + _C**2}


def seam_repair_share(kind: str) -> float:
    """Seam repair as a fraction of the full decode.

    Charged explicitly rather than left inside the tolerance of the cost control.
    Measured at +1.1% by tests/test_cost_matches_reality.py, which is what the
    arithmetic predicts — and a systematic under-charge is exactly how the
    earlier saving figures came out inflated.
    """
    return (SEAM_REPAIR_MACPX[kind] / _BLOCK_MACPX) * (SHARE_TRUNK / N_TRUNK_BLOCKS)


def halo_multiplier(cfg: FlexUFConfig, n_blocks: int, scope: str) -> float:
    """Cost multiplier the halo imposes on `n_blocks` of per-tile trunk work.

    "head"  -> 1.0 (the trunk carries no halo)
    "trunk" -> ((F+2h)/F)^2 for every block
    "taper" -> mean over blocks of ((F+2h-2i)/F)^2, i = 0..n_blocks-1, because a
               3x3 depthwise shrinks the valid region by 1 px per side per block
    """
    F, h = cfg.feature_patch, cfg.feature_halo
    if scope == "head" or h == 0 or n_blocks == 0:
        return 1.0
    if scope == "trunk":
        return ((F + 2 * h) / F) ** 2
    if scope == "taper":
        mults = []
        for i in range(n_blocks):
            side = F + 2 * max(h - i, 0)
            mults.append((side / F) ** 2)
        return sum(mults) / len(mults)
    raise ValueError(f"unknown halo scope {scope!r}")


def head_halo_multiplier(cfg: FlexUFConfig) -> float:
    """Cost multiplier on the head when it is applied per tile with a halo.

    1.0 when `full_frame_head` is True, because then the head runs once over the
    stitched canvas and pays nothing extra.
    """
    if cfg.full_frame_head or cfg.feature_halo == 0:
        return 1.0
    F, h = cfg.feature_patch, cfg.feature_halo
    return ((F + 2 * h) / F) ** 2


def exit_costs(
    cfg: Optional[FlexUFConfig] = None, halo_scope: str = "head"
) -> torch.Tensor:
    """Per-tile relative cost C_k, k = 0..K-1, in units of the full decode.

    Reported on the "as if everything were per-tile" basis, which is the right
    basis for comparing exits *to each other*. Frame-level numbers — the ones
    quoted as headline savings — come from :func:`frame_relative_cost`, which
    amortises the shared stem.
    """
    cfg = cfg or FlexUFConfig()
    b, j, K = cfg.blocks_per_exit, cfg.split_depth, cfg.num_exits
    per_block = SHARE_TRUNK / N_TRUNK_BLOCKS
    adapter = adapter_vs_block(cfg.adapter_kind) * per_block

    costs = []
    for k in range(K):
        # Charge what the DECODER actually runs. forward() does
        # exit_map.clamp(min=j), so a tile nominally assigned an exit shallower
        # than j still runs group j before leaving. The cost model must apply the
        # same clamp or it bills for a decode that never happens.
        #
        # This was wrong: k_eff = max(k, j-1) let an exit below j be charged as
        # "left at the split", omitting group j entirely. At j=2 a router that
        # assigned exit 0 to every tile was billed 58.70% saved when the real
        # figure is 42.9% — and that inflated number is what made the router
        # appear to beat the Lagrangian Pareto bound, which is impossible.
        k_run = max(k, j) if j < K else k
        blocks_run = (k_run + 1) * b
        shared_blocks = min(blocks_run, j * b)
        tiled_blocks = blocks_run - shared_blocks
        mult = halo_multiplier(cfg, tiled_blocks, halo_scope)
        c = (
            SHARE_UPSAMPLE
            + shared_blocks * per_block
            + tiled_blocks * per_block * mult
            + SHARE_HEAD * head_halo_multiplier(cfg)
            + seam_repair_share(cfg.seam_repair)
        )
        if k_run < K - 1:
            c += adapter * (mult if halo_scope != "head" else 1.0)
        costs.append(c)
    return torch.tensor(costs, dtype=torch.float32)


def frame_relative_cost(
    exit_map: torch.Tensor,
    cfg: Optional[FlexUFConfig] = None,
    halo_scope: str = "head",
) -> float:
    """Honest frame-level cost, amortising the shared stem over all tiles.

        cost = stem + head + mean_over_tiles( per-tile suffix + adapter )

    The stem (upsample + groups 0..j-1) runs once per frame regardless of tile
    count, and with `full_frame_head` the head does too.
    """
    cfg = cfg or FlexUFConfig()
    b, j, K = cfg.blocks_per_exit, cfg.split_depth, cfg.num_exits
    per_block = SHARE_TRUNK / N_TRUNK_BLOCKS
    adapter = adapter_vs_block(cfg.adapter_kind) * per_block

    stem = SHARE_UPSAMPLE + (j * b) * per_block
    head = SHARE_HEAD * head_halo_multiplier(cfg) + seam_repair_share(cfg.seam_repair)

    suffix = []
    for k in range(K):
        # Same clamp as forward(): a tile cannot leave before group j has run.
        k_eff = max(k, j) if j < K else k
        tiled_blocks = max((k_eff + 1) * b - j * b, 0)
        mult = halo_multiplier(cfg, tiled_blocks, halo_scope)
        c = tiled_blocks * per_block * mult
        if k_eff < K - 1:
            c += adapter * (mult if halo_scope != "head" else 1.0)
        suffix.append(c)

    s = torch.tensor(suffix, dtype=torch.float32, device=exit_map.device)
    return stem + head + s[exit_map.long()].mean().item()


def saving(
    exit_map: torch.Tensor,
    cfg: Optional[FlexUFConfig] = None,
    halo_scope: str = "head",
) -> float:
    """Fraction of decoder compute saved versus decoding every tile in full."""
    return 1.0 - frame_relative_cost(exit_map, cfg, halo_scope)


def mean_relative_cost(
    exit_map: torch.Tensor, costs: Optional[Sequence[float] | torch.Tensor] = None
) -> float:
    if costs is None:
        costs = exit_costs()
    costs = torch.as_tensor(costs, dtype=torch.float32, device=exit_map.device)
    return costs[exit_map.long()].mean().item()


def describe(cfg: Optional[FlexUFConfig] = None) -> str:
    """The cost table, printed into every experiment log."""
    cfg = cfg or FlexUFConfig()
    out = [
        f"cost model  K={cfg.num_exits} b={cfg.blocks_per_exit} j={cfg.split_depth} "
        f"tile={cfg.rgb_patch}px halo={cfg.latent_halo}lat/{cfg.feature_halo}feat "
        f"adapter={cfg.adapter_kind}",
        f"  measured base: upsample {SHARE_UPSAMPLE:.4f} | trunk {SHARE_TRUNK:.4f} "
        f"| head {SHARE_HEAD:.4f}",
        "",
        f"  {'scope':<7} {'exit':>5} {'C_k':>8} {'saved':>8}   (all-tiles-at-k)",
    ]
    for scope in ("head", "taper", "trunk"):
        costs = exit_costs(cfg, scope)
        for k, c in enumerate(costs.tolist()):
            out.append(f"  {scope:<7} {k:>5} {c:>8.4f} {100 * (1 - c):>7.1f}%")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    for j in (2, 4):
        for patch in (8, 4):
            cfg = FlexUFConfig(split_depth=j, latent_patch=patch, latent_halo=2)
            print(describe(cfg))
            print("=" * 70)
