"""The controls that must come out exact.

FLEX-FLOP's methodological note, adopted here verbatim: *a control that cannot
fail is not a control*. Three properties are asserted with zero tolerance,
because each one, if violated, silently invalidates every number the project
would go on to report.

  1. Warm-start round-trip. The ladder loaded from stock UF weights and run at
     its deepest exit must equal stock UF **bit-exactly**. If this drifts, the
     ladder is not a re-expression of UF and the "gap to full decode" metric is
     measuring the port's bugs instead of early exit.

  2. Untrained adapters are the identity. Because every adapter's output 1x1 is
     zero-initialised, running *any* exit before training must equal running the
     stock decoder truncated at that depth. This is what makes the warm start a
     genuine starting point rather than an initialisation.

  3. Patchify/unpatchify is a pure reshape. Round-tripping must be exactly the
     input. If it is not, the seams being measured are reshape bugs.

Run:  python -m pytest tests/test_equivalence.py -v
  or: python tests/test_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

from src.models.image_model import IntraDecoder  # noqa: E402

from flexuf.backbone.decoder import (  # noqa: E402
    MultiExitIntraDecoder,
    patchify,
    unpatchify,
)
from flexuf.backbone.warmstart import load_into_ladder, verify_bit_exact  # noqa: E402
from flexuf.config import LATENT_CH, TRUNK_CH, FlexUFConfig  # noqa: E402

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
# 16x28 latent -> 32x56 feature -> 512x896 RGB. Divisible by the 8-px feature
# tile, which patchify requires.
LAT_H, LAT_W = 16, 28


def _fixtures(cfg: FlexUFConfig | None = None):
    torch.manual_seed(0)
    cfg = cfg or FlexUFConfig()
    stock = IntraDecoder().to(DEVICE).eval()
    ladder = MultiExitIntraDecoder(cfg).to(DEVICE).eval()
    load_into_ladder(ladder, stock.state_dict())
    y = torch.randn(1, LATENT_CH, LAT_H, LAT_W, device=DEVICE)
    q = torch.rand(1, TRUNK_CH, 1, 1, device=DEVICE) + 0.5
    return stock, ladder, y, q


def test_reshape_roundtrip():
    x = torch.randn(1, TRUNK_CH, 32, 56, device=DEVICE)
    tiles, nh, nw = patchify(x, 8)
    assert tiles.shape == (nh * nw, TRUNK_CH, 8, 8), tiles.shape
    assert (nh, nw) == (4, 7)
    back = unpatchify(tiles, nh, nw)
    err = (back - x).abs().max().item()
    assert err == 0.0, f"patchify/unpatchify is not lossless: max|diff|={err}"
    print(f"  reshape round-trip           max|diff| = {err}")


def test_deepest_exit_is_stock_uf():
    stock, ladder, y, q = _fixtures()
    err = verify_bit_exact(ladder, stock, y, q)
    assert err == 0.0, f"deepest exit is not bit-exact stock UF: max|diff|={err}"
    print(f"  deepest exit == stock UF     max|diff| = {err}")


@torch.no_grad()
def test_untrained_adapters_are_identity():
    """Every exit, before training, equals stock UF truncated at that depth."""
    cfg = FlexUFConfig()
    stock, ladder, y, q = _fixtures(cfg)
    b = cfg.blocks_per_exit
    worst = 0.0
    for k in range(cfg.num_exits):
        # Stock decoder truncated after (k+1)*b trunk blocks, same head.
        feat = stock.dec_1[0](y)
        for n in range(1, (k + 1) * b + 1):
            feat = stock.dec_1[n](feat)
        ref = torch.nn.functional.pixel_shuffle(stock.dec_2(feat * q), 8)
        got = ladder.forward_full(y, q, exit_idx=k)
        err = (ref - got).abs().max().item()
        worst = max(worst, err)
        assert err == 0.0, f"exit {k} adapter is not the identity: max|diff|={err}"
    print(f"  all {cfg.num_exits} exits == truncated UF  max|diff| = {worst}")


@torch.no_grad()
def test_hybrid_at_deepest_equals_full():
    """The j-split path with everyone at the deepest exit == the full decode.

    This is the strongest structural control: it exercises patchify, the
    per-patch group loop, the canvas assembly, unpatchify and the full-frame
    head, and still has to land on the full-frame answer exactly.
    """
    for j in (0, 2, 4, 5):
        cfg = FlexUFConfig(split_depth=j, full_frame_head=True)
        stock, ladder, y, q = _fixtures(cfg)
        ref = ladder.forward_full(y, q, exit_idx=None)
        got = ladder(y, q, exit_map=None)
        err = (ref - got).abs().max().item()
        assert err == 0.0, f"j={j}: hybrid decode != full decode, max|diff|={err}"
        print(f"  hybrid j={j} == full decode   max|diff| = {err}")


@torch.no_grad()
def test_mixed_depth_runs_and_saves():
    """A genuinely mixed exit map must run, and shallower maps must cost less."""
    from flexuf.cost import exit_costs, mean_relative_cost

    cfg = FlexUFConfig(split_depth=2)
    _, ladder, y, q = _fixtures(cfg)
    n_tiles = (LAT_H * 2 // cfg.feature_patch) * (LAT_W * 2 // cfg.feature_patch)

    torch.manual_seed(1)
    em = torch.randint(cfg.split_depth, cfg.num_exits, (n_tiles,), device=DEVICE)
    out = ladder(y, q, exit_map=em)
    assert out.shape == (1, 3, LAT_H * 16, LAT_W * 16), out.shape
    assert torch.isfinite(out).all(), "mixed-depth decode produced non-finite values"

    costs = exit_costs(cfg)
    all_deep = torch.full((n_tiles,), cfg.num_exits - 1, device=DEVICE)
    c_mixed = mean_relative_cost(em, costs)
    c_deep = mean_relative_cost(all_deep, costs)
    assert c_mixed < c_deep, f"mixed map ({c_mixed:.3f}) should cost less than deep ({c_deep:.3f})"
    print(f"  mixed-depth decode ok        cost {c_mixed:.3f} vs deep {c_deep:.3f}")


if __name__ == "__main__":
    print(f"\ndevice: {DEVICE}\n")
    print("controls (all must be exactly 0.0):")
    test_reshape_roundtrip()
    test_deepest_exit_is_stock_uf()
    test_untrained_adapters_are_identity()
    test_hybrid_at_deepest_equals_full()
    test_mixed_depth_runs_and_saves()
    print("\nall controls passed\n")
