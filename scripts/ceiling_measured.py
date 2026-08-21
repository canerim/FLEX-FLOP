"""The architectural ceiling, counted off a decode instead of modelled.

The ceiling is what the method can save if every tile takes the shallowest exit
it is allowed. Three numbers for it have been in circulation, and this settles
which one the paper should print by measuring it the same way scripts/measure.py
measures everything else: hooks on the executed forward pass, divided by the
released decoder's own count on the same latent.

    python scripts/ceiling_measured.py --device cuda:0
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

from flexuf.config import FlexUFConfig            # noqa: E402
from flexuf.cost import exit_costs                # noqa: E402
from flexuf.measure import MacMeter               # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for        # noqa: E402



class _Routed(torch.nn.Module):
    """fvcore calls model(*inputs); the decoder wants an exit map by keyword."""

    def __init__(self, dec, exit_map=None, full=False):
        super().__init__()
        self.dec, self.exit_map, self.full = dec, exit_map, full

    def forward(self, y, q):
        if self.full:
            return self.dec.forward_full(y, q)
        return self.dec(y, q, exit_map=self.exit_map)


def _fvcore_macs(dec, y, q, exit_map=None, full=False):
    """MACs of one decode, counted by fvcore. None if it is not installed.

    Despite the name, FlopCountAnalysis counts multiply-accumulates, which is
    MacMeter's convention too, so the two are directly comparable with no
    factor of two involved.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    fa = FlopCountAnalysis(_Routed(dec, exit_map=exit_map, full=full), (y, q))
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return float(fa.total())



def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/ceiling_measured.json")
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))

    w0, h0 = (int(v) for v in a.size.split("x"))
    P = cfg.rgb_patch
    W = (w0 + P - 1) // P * P
    H = (h0 + P - 1) // P * P
    n_tiles = (H // P) * (W // P)
    j, K = cfg.split_depth, cfg.num_exits

    out = {"ckpt": a.ckpt, "qp": a.qp, "size": f"{W}x{H}", "n_tiles": n_tiles,
           "split_depth": j, "num_exits": K, "exits": []}
    with torch.no_grad():
        x = torch.zeros(1, 3, H, W, device=a.device)
        qp = torch.full((1,), a.qp, dtype=torch.int32, device=a.device)
        y, q, _ = net._encode_to_latent(x, qp)

        with MacMeter(ref) as rel:
            ref.dec.forward_full(y, q)
        released = rel.total
        # The same count from fvcore, which is nobody here's code. It agrees
        # to six decimals on every exit (results/mac_crosscheck.json), so this
        # changes no number; it is here so the ceiling the paper prints is a
        # number a reader can reproduce with a standard tool.
        released_fv = _fvcore_macs(ref.dec, y, q, full=True)

        model = exit_costs(cfg, "head")
        print(f"  {W}x{H}, {n_tiles} tiles, released decode "
              f"{released/1e9:.1f} GMAC\n")
        print(f"  {'exit':>5}{'modelled':>11}{'measured':>11}{'diff':>9}"
              f"{'ceiling if all tiles here':>28}")
        for k in range(K):
            em = torch.full((n_tiles,), k, dtype=torch.long,
                            device=a.device).clamp(min=j)
            with MacMeter(net.dec) as m:
                net.dec(y, q, exit_map=em)
            meas = m.total / released
            meas_fv = (_fvcore_macs(net.dec, y, q, exit_map=em) / released_fv
                       if released_fv else None)
            if meas_fv is not None and abs(meas_fv - meas) > 1e-4:
                raise SystemExit(
                    f"exit {k}: MacMeter says {meas:.6f} of a released "
                    f"decode, fvcore says {meas_fv:.6f}. The two counters "
                    f"have drifted and the ceiling is not trustworthy.")
            # exit_costs returns a tensor; json wants floats.
            mk = float(model[k])
            row = {"exit": k, "modelled": mk, "measured": float(meas),
                   "measured_fvcore": (float(meas_fv) if meas_fv is not None
                                       else None),
                   "ceiling_modelled_pct": 100 * (1 - mk),
                   "ceiling_measured_pct": 100 * (1 - float(meas))}
            out["exits"].append(row)
            print(f"  {k:>5}{mk:>11.4f}{meas:>11.4f}"
                  f"{meas - mk:>+9.4f}{100*(1-meas):>27.2f}%")

    ceil_m = out["exits"][j]["ceiling_measured_pct"]
    ceil_model = out["exits"][j]["ceiling_modelled_pct"]
    out["ceiling_measured_pct"] = ceil_m
    out["ceiling_modelled_pct"] = ceil_model
    print(f"\n  the ceiling is exit {j}, the shallowest a tile may take:")
    print(f"     modelled {ceil_model:.2f}%   measured {ceil_m:.2f}%")
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
