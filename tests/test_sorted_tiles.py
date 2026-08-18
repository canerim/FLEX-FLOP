"""Sorting the tiles by depth must not change the picture.

The sorted suffix is a bookkeeping change: the same tiles run through the same
groups, only the order and the indexing differ. That claim is worth a test rather
than an argument, because the failure mode is silent -- a permutation applied in
one place and not another produces a plausible picture with tiles in the wrong
positions, and PSNR would merely look a little worse.

Two things this test is careful about, both learned the hard way:

* **Random exit maps, not uniform.** A uniform map makes the permutation the
  identity and tests nothing.
* **Random content, not zeros.** `scripts/sorted_exec.py` originally checked
  equality on a decode of `torch.zeros`, where every tile is identical and any
  permutation is trivially exact. That check could not fail.

On CUDA the result is bit-identical. On CPU it is not, and the reason is not
ours: oneDNN selects a different blocking for some batch sizes, so a plain
`DepthConvBlock` is already order-dependent at N = 13 by about 1e-7 while being
exact at N = 4, 9, 16 and 40. The tolerance below reflects the library, not the
algorithm.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.backbone.decoder import MultiExitIntraDecoder
from flexuf.config import FlexUFConfig

LATENT_CH = 256
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = {"cuda": 0.0, "cpu": 1e-6}


def _build(dev, **over):
    cfg = FlexUFConfig(**{"latent_patch": 4, "tile_pad_mode": "replicate",
                          "seam_repair": "none", **over})
    dec = MultiExitIntraDecoder(cfg).to(dev).eval()
    for p in dec.parameters():          # zero-init adapters would hide a swap
        torch.nn.init.normal_(p, std=0.02)
    return cfg, dec


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sorted_matches_masked(dev, seed):
    torch.manual_seed(seed)
    cfg, dec = _build(dev)
    n_lat, side = 4, cfg.latent_patch * 4
    y = torch.randn(1, LATENT_CH, side, side, device=dev)
    q = torch.ones(1, 1, 1, 1, device=dev)
    n = n_lat * n_lat
    em = torch.randint(cfg.split_depth, cfg.num_exits, (n,), device=dev)
    gate = torch.rand(n, device=dev) + 0.5

    with torch.no_grad():
        dec.cfg = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": False})
        a = dec(y, q, exit_map=em, tile_gate=gate)
        dec.cfg = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": True})
        b = dec(y, q, exit_map=em, tile_gate=gate)

    assert len(set(em.tolist())) > 1, "the exit map must actually mix depths"
    assert (a - b).abs().max().item() <= TOL[dev]


@pytest.mark.parametrize("dev", DEVICES)
def test_sorted_with_repair_and_tiled_head(dev):
    torch.manual_seed(7)
    cfg, dec = _build(dev, seam_repair="grid", full_frame_head=False)
    y = torch.randn(1, LATENT_CH, 16, 16, device=dev)
    q = torch.ones(1, 1, 1, 1, device=dev)
    em = torch.randint(cfg.split_depth, cfg.num_exits, (16,), device=dev)
    with torch.no_grad():
        dec.cfg = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": False})
        a = dec(y, q, exit_map=em)
        dec.cfg = FlexUFConfig(**{**cfg.__dict__, "sorted_tiles": True})
        b = dec(y, q, exit_map=em)
    assert (a - b).abs().max().item() <= TOL[dev]


def test_sorted_refuses_coupling():
    cfg = FlexUFConfig(latent_patch=4, sorted_tiles=True, tile_coupling=True)
    dec = MultiExitIntraDecoder(cfg).eval()
    with pytest.raises(AssertionError, match="not compatible"):
        dec(torch.randn(1, LATENT_CH, 16, 16), torch.ones(1, 1, 1, 1))
