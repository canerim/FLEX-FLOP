"""The controls that must come out exact.

FLEX-FLOP's methodological rule, adopted verbatim: *a control that cannot fail is
not a control*. Each property below is asserted with zero tolerance, because each
one, if violated, silently invalidates every number the project would report.

  1. **Warm-start round-trip.** The ladder loaded from stock UF weights, run at
     its deepest exit, must equal stock UF bit-exactly. If this drifts, the
     ladder is not a re-expression of UF, and "gap to full decode" measures the
     port's bugs rather than early exit.

  2. **Untrained adapters are the identity.** Every adapter's output is
     zero-initialised, so before training, running *any* exit must equal the
     stock decoder truncated at that depth. This is what makes a warm start a
     genuine starting point rather than merely an initialisation.

  3. **j=K reproduces the full decode.** The strongest structural control: it
     exercises patchify-with-halo, the per-tile group loop, canvas assembly,
     halo cropping, unpatchify and the head, and must still land exactly on the
     full-frame answer.

  4. **Reshapes are lossless.** If patchify/unpatchify is not exact, the seams
     being measured are reshape bugs.

  5. **Shallower routing costs less.** A sanity check on the cost model itself.

Run:  python tests/test_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

from src.models.image_model import IntraDecoder  # noqa: E402

from flexuf.backbone.decoder import (  # noqa: E402
    MultiExitIntraDecoder,
    patchify,
    unpatchify,
)
from flexuf.backbone.warmstart import load_into_ladder  # noqa: E402
from flexuf.config import LATENT_CH, TRUNK_CH, FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs, saving  # noqa: E402

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
LAT_H, LAT_W = 16, 16   # -> 32x32 feature -> 512x512 RGB


def _fixtures(cfg=None):
    torch.manual_seed(0)
    cfg = cfg or FlexUFConfig()
    stock = IntraDecoder().to(DEVICE).eval()
    ladder = MultiExitIntraDecoder(cfg).to(DEVICE).eval()
    load_into_ladder(ladder, stock.state_dict())
    y = torch.randn(1, LATENT_CH, LAT_H, LAT_W, device=DEVICE)
    q = torch.rand(1, TRUNK_CH, 1, 1, device=DEVICE) + 0.5
    return stock, ladder, y, q


def test_reshape_roundtrip():
    x = torch.randn(1, TRUNK_CH, 32, 32, device=DEVICE)
    tiles, nh, nw = patchify(x, 16)
    assert tiles.shape == (nh * nw, TRUNK_CH, 16, 16), tiles.shape
    err = (unpatchify(tiles, nh, nw) - x).abs().max().item()
    assert err == 0.0, f"patchify/unpatchify not lossless: {err}"
    print(f"  reshape round-trip                 max|diff| = {err}")


@torch.no_grad()
def test_deepest_exit_is_stock_uf():
    stock, ladder, y, q = _fixtures()
    err = (stock(y, q) - ladder.forward_full(y, q)).abs().max().item()
    assert err == 0.0, f"deepest exit != stock UF: {err}"
    print(f"  deepest exit == stock UF           max|diff| = {err}")


@torch.no_grad()
def test_untrained_adapters_are_identity():
    for kind in ("conv1x1", "ffn"):
        cfg = FlexUFConfig(adapter_kind=kind)
        stock, ladder, y, q = _fixtures(cfg)
        b, worst = cfg.blocks_per_exit, 0.0
        for k in range(cfg.num_exits):
            feat = stock.dec_1[0](y)
            for n in range(1, (k + 1) * b + 1):
                feat = stock.dec_1[n](feat)
            ref = F.pixel_shuffle(stock.dec_2(feat * q), 8)
            err = (ref - ladder.forward_full(y, q, exit_idx=k)).abs().max().item()
            worst = max(worst, err)
        assert worst == 0.0, f"{kind}: adapters not identity: {worst}"
        print(f"  {kind:<8} all exits == truncated UF  max|diff| = {worst}")


@torch.no_grad()
def test_j_equals_K_reproduces_full_decode():
    cfg = FlexUFConfig(split_depth=6, latent_patch=8, latent_halo=2)
    _, ladder, y, q = _fixtures(cfg)
    err = (ladder.forward_full(y, q) - ladder(y, q, exit_map=None)).abs().max().item()
    assert err == 0.0, f"j=K hybrid != full decode: {err}"
    print(f"  j=K hybrid == full decode          max|diff| = {err}")


@torch.no_grad()
def test_forward_all_exits_matches_forward_full():
    """One trunk pass must give the same answers as K separate passes."""
    cfg = FlexUFConfig()
    _, ladder, y, q = _fixtures(cfg)
    outs = ladder.forward_all_exits(y, q)
    worst = max(
        (outs[k] - ladder.forward_full(y, q, exit_idx=k)).abs().max().item()
        for k in range(cfg.num_exits)
    )
    assert worst == 0.0, f"forward_all_exits diverges: {worst}"
    print(f"  all-exits pass == per-exit passes  max|diff| = {worst}")


@torch.no_grad()
def test_mixed_depth_runs_and_is_cheaper():
    cfg = FlexUFConfig(split_depth=2, latent_patch=8, latent_halo=2)
    _, ladder, y, q = _fixtures(cfg)
    n_tiles = (LAT_H * 2 // cfg.feature_patch) * (LAT_W * 2 // cfg.feature_patch)

    torch.manual_seed(1)
    em = torch.randint(cfg.split_depth, cfg.num_exits, (n_tiles,), device=DEVICE)
    out = ladder(y, q, exit_map=em)
    assert out.shape == (1, 3, LAT_H * 16, LAT_W * 16), out.shape
    assert torch.isfinite(out).all(), "mixed-depth decode produced non-finite values"

    deep = torch.full((n_tiles,), cfg.num_exits - 1, device=DEVICE)
    s_mix, s_deep = saving(em, cfg, "head"), saving(deep, cfg, "head")
    assert s_mix > s_deep, f"mixed ({s_mix:.3f}) should save more than deep ({s_deep:.3f})"
    print(f"  mixed-depth decode runs            saving {100*s_mix:.1f}% vs {100*s_deep:.1f}%")


def test_cost_model_is_monotone():
    """Deeper exits must cost more, and the deepest must cost exactly 1.0."""
    cfg = FlexUFConfig(split_depth=6)  # j=K: no tiling, so C_{K-1} is the full decode
    c = exit_costs(cfg, "head").tolist()
    assert all(c[i] < c[i + 1] for i in range(len(c) - 1)), f"not monotone: {c}"
    assert abs(c[-1] - 1.0) < 1e-6, f"deepest exit should cost 1.0, got {c[-1]}"
    print(f"  cost model monotone, C_max = {c[-1]:.6f}")


if __name__ == "__main__":
    print(f"\ndevice: {DEVICE}\n")
    print("zero-tolerance controls:")
    test_reshape_roundtrip()
    test_deepest_exit_is_stock_uf()
    test_untrained_adapters_are_identity()
    test_j_equals_K_reproduces_full_decode()
    test_forward_all_exits_matches_forward_full()
    print("\nbehavioural checks:")
    test_mixed_depth_runs_and_is_cheaper()
    test_cost_model_is_monotone()
    print("\nall controls passed\n")
