"""Per-tile, per-exit distortion measured on the DEPLOYED decode path.

Why this module exists
----------------------
`dec.forward_all_exits` runs the trunk FULL FRAME and taps each exit. It is the
right thing for training -- one trunk pass, K heads, and the gradient reaches
every exit -- and it is the wrong thing for evaluation, because the deployed
decoder runs `groups[j:]` per tile and a tile border there meets padding instead
of a neighbour.

Every evaluation script in this repository built its Lagrangian table with the
full-frame form, so the reference was the release's full-frame decode and our
side was ALSO full-frame: the tiling penalty cancelled out of the reported dB and
the numbers described a decoder nobody ships. Measured on RECIPE512, every tile
at the deepest exit:

    qp   full-frame dB   deployed dB    missing
     0          0.0040        0.0385    +0.0345
    32          0.0120        0.0506    +0.0387
    63          0.0252        0.0615    +0.0362

and at the shallowest exit the gap is +0.008, because only two blocks have run
per tile there instead of eight. So the error is not a constant offset -- it
depends on the exit mix, which is exactly the thing being optimised.

The cost of doing it properly is K tiled decodes per frame instead of one trunk
pass. At 40 frames and five rates that is a few minutes, which is nothing next to
being wrong.

One approximation remains, and it is stated rather than hidden: a tile's MSE is
measured with every OTHER tile at the same exit, while in a routed decode its
neighbours may sit at different depths. `true_frame_mse` exists to close that --
it decodes the actual mixed map once and returns what the frame really costs, so
a caller can bisect on the cheap table and report the exact number.
"""

from __future__ import annotations

import torch


def per_tile_mse(img: torch.Tensor, target: torch.Tensor, nh: int, nw: int,
                 patch: int) -> torch.Tensor:
    """Mean squared error of each tile, as a flat [nh*nw] tensor.

    Averaged over channels first, then over the tile's pixels, so the tile grid
    matches `patchify`'s ordering exactly -- row-major, which is what `exit_map`
    is indexed by.
    """
    e = ((img - target) ** 2).mean(1)
    return (e.view(img.shape[0], nh, patch, nw, patch)
             .permute(0, 1, 3, 2, 4)
             .reshape(img.shape[0] * nh * nw, patch * patch)
             .mean(1))


@torch.no_grad()
def tiled_exit_mses(dec, y_hat: torch.Tensor, quant_step: torch.Tensor,
                    target: torch.Tensor, cfg) -> torch.Tensor:
    """[n_tiles, K] per-tile MSE with every tile decoded at each exit in turn.

    Exits shallower than the split depth are not reachable -- `decoder.forward`
    clamps `exit_map` to `k >= j` -- so their rows are whatever exit j produces.
    Returning them filled that way rather than dropping them keeps the shape K
    for every caller, and makes an argmin over all K land on a choice the decoder
    can actually carry out. The old table had genuinely different values there,
    from adapters the deployed path never invokes, so an argmin could select an
    exit that does not exist.
    """
    P = cfg.rgb_patch
    H, W = target.shape[-2:]
    nh, nw = H // P, W // P
    n_tiles = nh * nw
    cols = []
    for k in range(cfg.num_exits):
        em = torch.full((n_tiles,), max(k, cfg.split_depth),
                        dtype=torch.long, device=y_hat.device)
        cols.append(per_tile_mse(dec(y_hat, quant_step, exit_map=em), target,
                                 nh, nw, P))
    return torch.stack(cols, dim=1)


@torch.no_grad()
def true_frame_mse(dec, y_hat: torch.Tensor, quant_step: torch.Tensor,
                   target: torch.Tensor, exit_map: torch.Tensor) -> torch.Tensor:
    """The frame's MSE under the actual mixed exit map -- one real decode.

    This is the only number that is exactly what a decoder would produce. The
    table above predicts it well but not perfectly, because a tile's border sees
    whatever depth its neighbour chose.
    """
    return ((dec(y_hat, quant_step, exit_map=exit_map) - target) ** 2).mean()


@torch.no_grad()
def reference_frame_mse(ref_dec, y_hat: torch.Tensor, quant_step: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """The released decoder's FULL-FRAME decode of the same latent.

    Full frame on purpose: it is what the release does, and it is what every dB
    in this project is quoted against. The tiling penalty therefore belongs on
    our side of the ratio, which is the whole point of this module.
    """
    return ((ref_dec.forward_full(y_hat, quant_step) - target) ** 2).mean()
