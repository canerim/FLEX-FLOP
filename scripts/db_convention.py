"""How much does the dB averaging convention move the number?

Two of this project's scripts report "dB below the released decoder" and compute
it differently, which was found by noticing they disagreed at the same budget:

    paper_curve.py      10*log10( mean_over_all_tiles(MSE) / mean_over_all_tiles(R) )
    signalled_curve.py  mean_over_frames( 10*log10( mean_tiles(MSE_f) / mean_tiles(R_f) ) )

The first pools every tile of every frame into one distortion, which is the
natural form for the Lagrangian the theory is about -- J = D + lambda*C with a
single D. The second averages a per-frame decibel, which is what
`~/DCVC/test_video.py` does and therefore what every published DCVC-UF number
means.

Neither is wrong. Quoting both in one document without saying they differ would
be, so the size of the difference is measured here rather than left for a
reader to find by comparing two tables.

They coincide only when the per-frame ratio MSE_f/R_f is constant across frames;
the gap is driven by how much that ratio varies with content, which is exactly
the heterogeneity the rest of this project is about -- so it is expected to
grow with rate.

    python scripts/db_convention.py --ckpt <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar")
    ap.add_argument("--ref", default="runs/warmstart/ckpt_warmstart.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--out", default="results/db_convention.json")
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(ROOT / a.ref, map_location="cpu",
                                      weights_only=False))
    cost = exit_costs(cfg, "head").to(dev)

    seqs, _ = C.discover([])
    frames = []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            frames.append(x[0:1])
    print(f"  {len(frames)} frames, {cfg.rgb_patch}px tile\n")
    print(f"  {'qp':>4}{'pooled dB':>12}{'per-frame dB':>15}{'difference':>13}"
          f"{'saving':>10}")

    out = {"ckpt": a.ckpt, "n_frames": len(frames), "target_db": a.target,
           "rows": []}
    with torch.no_grad():
        for qp_v in a.qps:
            per_frame = []
            for x in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                nh, nw = (H + ph) // P, (W + pw) // P

                def tl(img):
                    e = ((img - xp) ** 2).mean(1)
                    return (e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                             .reshape(nh * nw, P * P).mean(1))

                M = torch.stack([tl(o) for o in net.dec.forward_all_exits(y, q)], 1)
                R = tl(ref.dec.forward_full(y, q))
                per_frame.append((M, R))

            def both(lam):
                """(pooled dB, per-frame-averaged dB, saving) at one lambda."""
                mses, refs, dbs, svs = [], [], [], []
                for M, R in per_frame:
                    k = (M + lam * cost[None, :]).argmin(1)
                    m = M.gather(1, k[:, None]).squeeze(1).mean()
                    mses.append(m)
                    refs.append(R.mean())
                    dbs.append(10 * torch.log10(m / R.mean()))
                    svs.append(1 - cost[k].mean() / cost[-1])
                pooled = 10 * torch.log10(torch.stack(mses).mean()
                                          / torch.stack(refs).mean())
                return (pooled.item(), torch.stack(dbs).mean().item(),
                        100 * torch.stack(svs).mean().item())

            # Bisect on the PER-FRAME convention, then report what the pooled
            # one would have called the same allocation. Same allocation, two
            # descriptions -- which isolates the convention from everything else.
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if both(mid)[1] <= a.target:
                    lo = mid
                else:
                    hi = mid
            pooled, perframe, sv = both(lo)
            out["rows"].append({"qp": qp_v, "pooled_db": pooled,
                                "per_frame_db": perframe,
                                "difference_db": pooled - perframe,
                                "saving_pct": sv})
            print(f"  {qp_v:>4}{pooled:>12.4f}{perframe:>15.4f}"
                  f"{pooled - perframe:>+13.4f}{sv:>9.1f}%")

    d = [abs(r["difference_db"]) for r in out["rows"]]
    print(f"\n  the convention moves the reported dB by {min(d):.4f} to "
          f"{max(d):.4f} dB on the SAME allocation.")
    print(f"  Against a 0.10 dB budget that is "
          f"{100 * min(d) / a.target:.1f}-{100 * max(d) / a.target:.1f}% of the "
          f"budget, so it is not a rounding detail.")
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
