"""Our MAC count against fvcore's, on the decodes the paper prices.

Every saving in this paper is a ratio of two MAC counts taken by
flexuf.measure.MacMeter -- forward hooks on Conv2d and Linear, counted from
the output shape. That is 40 lines of our own code standing under the headline
number, so it is worth checking against a counter nobody here wrote.

fvcore's FlopCountAnalysis is the usual choice in vision work. Despite the
name it counts multiply-accumulates, not floating-point operations, which is
the same convention as MacMeter, so the two totals are directly comparable and
no factor of two is involved. Where they differ it will be over which modules
count at all: fvcore dispatches on aten operators and reports anything it has
no handler for, and MacMeter hooks two module types. The unhandled list is
printed rather than hidden, because an operator neither counter charges for is
a hole in both.

    python scripts/mac_crosscheck.py --device cpu --size 512x512
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexuf.config import FlexUFConfig                    # noqa: E402
from flexuf.measure import MacMeter                       # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402
from flexuf.reference import reference_for                # noqa: E402


class _Routed(nn.Module):
    """fvcore calls model(*inputs); the decoder wants an exit map by keyword."""

    def __init__(self, dec, exit_map=None, full=False):
        super().__init__()
        self.dec, self.exit_map, self.full = dec, exit_map, full

    def forward(self, y, q):
        if self.full:
            return self.dec.forward_full(y, q)
        return self.dec(y, q, exit_map=self.exit_map)


def _fvcore(mod, y, q):
    """(MACs, {unhandled operator: count}) for one forward."""
    from fvcore.nn import FlopCountAnalysis
    fa = FlopCountAnalysis(mod, (y, q))
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return float(fa.total()), dict(fa.unsupported_ops())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/mac_crosscheck.json")
    a = ap.parse_args(argv)

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))

    w0, h0 = (int(v) for v in a.size.split("x"))
    P = cfg.rgb_patch
    W, H = (w0 + P - 1) // P * P, (h0 + P - 1) // P * P
    n_tiles = (H // P) * (W // P)
    j, K = cfg.split_depth, cfg.num_exits

    out = {"ckpt": a.ckpt, "qp": a.qp, "size": f"{W}x{H}",
           "n_tiles": n_tiles, "split_depth": j, "num_exits": K,
           "counter_a": "flexuf.measure.MacMeter",
           "counter_b": "fvcore.nn.FlopCountAnalysis", "rows": []}

    unhandled = Counter()
    with torch.no_grad():
        x = torch.zeros(1, 3, H, W, device=a.device)
        qp = torch.full((1,), a.qp, dtype=torch.int32, device=a.device)
        y, q, _ = net._encode_to_latent(x, qp)

        # The denominator both counters divide by: the released decoder, full
        # depth, on this latent.
        with MacMeter(ref) as m:
            ref.dec.forward_full(y, q)
        rel_ours = m.total
        rel_fv, u = _fvcore(_Routed(ref.dec, full=True), y, q)
        unhandled.update(u)

        print(f"  {W}x{H}, {n_tiles} tiles, qp {a.qp}\n")
        print(f"  released decode   MacMeter {rel_ours/1e9:10.3f} GMAC   "
              f"fvcore {rel_fv/1e9:10.3f} GMAC   "
              f"ratio {rel_fv/rel_ours:.6f}")
        out["released"] = {"macmeter": rel_ours, "fvcore": rel_fv}

        print(f"\n  {'exit':>5}{'MacMeter':>12}{'fvcore':>12}"
              f"{'b/a':>10}{'saving (a)':>12}{'saving (b)':>12}")
        for k in range(K):
            em = torch.full((n_tiles,), k, dtype=torch.long,
                            device=a.device).clamp(min=j)
            with MacMeter(net.dec) as m:
                net.dec(y, q, exit_map=em)
            ours = m.total
            fv, u = _fvcore(_Routed(net.dec, exit_map=em), y, q)
            unhandled.update(u)
            sa, sb = 100 * (1 - ours / rel_ours), 100 * (1 - fv / rel_fv)
            out["rows"].append({"exit": k, "macmeter": ours, "fvcore": fv,
                                "saving_pct_macmeter": sa,
                                "saving_pct_fvcore": sb})
            print(f"  {k:>5}{ours/1e9:>12.3f}{fv/1e9:>12.3f}"
                  f"{fv/ours:>10.6f}{sa:>11.2f}%{sb:>11.2f}%")

    ca = out["rows"][j]["saving_pct_macmeter"]
    cb = out["rows"][j]["saving_pct_fvcore"]
    out["ceiling_pct_macmeter"], out["ceiling_pct_fvcore"] = ca, cb
    out["ceiling_diff_points"] = cb - ca
    out["unhandled_ops"] = dict(unhandled)
    print(f"\n  the ceiling is exit {j}: MacMeter {ca:.2f}%, "
          f"fvcore {cb:.2f}%, difference {cb - ca:+.2f} points")
    if unhandled:
        print("\n  operators fvcore has no handler for (neither counter "
              "charges for them):")
        for op, n in sorted(unhandled.items(), key=lambda t: -t[1]):
            print(f"     {op:<40} {n}")
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
