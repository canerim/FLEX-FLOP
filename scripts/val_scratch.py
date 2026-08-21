"""Per-exit quality on held-out images, cheaply enough to run every quarter hour.

The CTC evaluation chain takes twenty minutes and compares against the released
decoder. Neither is any use for the first days of a from-scratch run: the model
has no relation to the release yet, and a twenty-minute measurement every
fifteen minutes would take the card the run is training on.

This is the other measurement -- the one that says whether the ladder is alive.
Per-exit PSNR on a fixed held-out split, the same split beta_calibration uses,
at three rates. What it has to show is a monotone ladder with a spread that
opens as the trunk learns: exits that converge on each other are the known
failure, and the loss falls perfectly well while it happens.

    python scripts/val_scratch.py --run SCRATCH105 --device cuda:7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig            # noqa: E402
from flexuf.model import FlexUFIntra              # noqa: E402
from gpu import pick                              # noqa: E402
from why_qp_val import load_val                   # noqa: E402


def _budget_savings(cfg, ps, budgets=(0.1, 0.2, 0.3)):
    """For each budget, the shallowest UNIFORM exit within it and what it saves.

    Two things this number is not, both of which would make it look like the
    paper's headline and it is neither.

    It is a uniform allocation -- every tile at the same exit -- chosen off the
    set mean. The paper's 29.2% at q0 is a per-tile allocation, which is
    strictly better, so this is a lower bound on what this checkpoint could do
    and the gap between them is the thing the whole paper is about.

    And its reference is this run's own deepest exit, not the released decoder.
    SCRATCH105 has no relation to the release; the question it exists to answer
    is whether a ladder built from random initialisation buys the same kind of
    trade a warm-started one does, and that question is asked against itself.

    Frame-relative, so the stem and head are amortised once as they are at
    deployment.
    """
    # frame_relative_cost takes a per-tile exit MAP, so a uniform map at exit k
    # gives that exit's frame-level cost: the stem and head are amortised once
    # whatever the tiles do, which is the whole point of the frame-relative
    # basis and the reason a per-tile cost vector cannot be used here.
    from flexuf.cost import frame_relative_cost
    C = [frame_relative_cost(torch.full((cfg.tiles_for_crop(1080),), k,
                                        dtype=torch.long), cfg)
         for k in range(cfg.num_exits)]
    out = {}
    deep = ps[-1]
    for b in budgets:
        ok = [k for k in range(cfg.split_depth, len(ps)) if deep - ps[k] <= b]
        k = min(ok) if ok else len(ps) - 1
        out[f"uniform_exit_at_{b:g}db"] = k
        out[f"uniform_saving_pct_at_{b:g}db"] = round(100.0 * (1.0 - C[k]), 2)
        out[f"uniform_delivered_db_at_{b:g}db"] = round(deep - ps[k], 4)
    return out


def psnr(mse: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-10)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SCRATCH105")
    ap.add_argument("--device", default=pick("cuda:0"))
    ap.add_argument("--n", type=int, default=48, help="held-out images")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    args = ap.parse_args()

    d = ROOT / "runs" / args.run
    # A step snapshot when there is one, the resume file otherwise: over the
    # first 2500 steps there is no snapshot yet and the run is exactly when
    # somebody most wants to see whether it is learning.
    src = None
    for name in ("ckpt_step.pth.tar", "ckpt_eval.pth.tar", "status_latest.pth.tar"):
        if (d / name).exists():
            src = d / name
            break
    if src is None:
        print(f"  {args.run}: no checkpoint yet")
        return 0

    blob = torch.load(src, map_location="cpu", weights_only=False)
    cfg_d = blob.get("config") or json.loads((d / "meta.json").read_text())["config"]
    cfg = FlexUFConfig(**cfg_d)
    sd = blob.get("state_dict", blob)
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    dev = torch.device(args.device)
    net = FlexUFIntra(cfg).to(dev).eval()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    imgs = load_val(args.n, args.crop, dev)

    t0 = time.time()
    rows = []
    with torch.no_grad():
        for qp in args.qps:
            acc = None
            bpp_sum = 0.0
            for x in imgs:
                q = torch.full((x.shape[0],), qp, dtype=torch.long, device=dev)
                out = net.forward_all_exits(x, q)
                m = torch.stack([mm.mean() for mm in out["mses"]])
                acc = m if acc is None else acc + m
                bpp_sum += float(out["bpp"].mean())
            acc = acc / len(imgs)
            ps = [psnr(v) for v in acc]
            rows.append({
                "qp": qp,
                "psnr_per_exit": [round(v, 3) for v in ps],
                "bpp": round(bpp_sum / len(imgs), 4),
                "spread_dB": round(ps[-1] - ps[0], 3),
                # What the ladder is worth, on this run's own terms.
                #
                # SCRATCH105 has no released decoder to be measured against --
                # it IS a different decoder -- so its reference is its own
                # deepest exit, and the question the run exists to answer is
                # whether a ladder built from random initialisation buys the
                # same kind of trade a warm-started one does. This is that
                # number as it trains: the cheapest exit whose quality is
                # within the budget of the deepest, and what that exit costs.
                **_budget_savings(cfg, ps),
            })

    rec = {
        "t": time.strftime("%F %T"),
        "ckpt": src.name,
        "epoch": blob.get("epoch"),
        "step": blob.get("step"),
        "n_images": len(imgs),
        "crop": args.crop,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "sec": round(time.time() - t0, 1),
        "rows": rows,
    }
    with open(d / "val.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    for r in rows:
        print(f"  q{r['qp']:<3} psnr {r['psnr_per_exit']}  "
              f"bpp {r['bpp']:.4f}  spread {r['spread_dB']:+.3f} dB  "
              f"| 0.1 dB uniform: exit {r['uniform_exit_at_0.1db']}, "
              f"{r['uniform_saving_pct_at_0.1db']:+.1f}%")
    print(f"  {args.run} @ epoch {rec['epoch']} step {rec['step']} "
          f"({rec['sec']}s, {len(imgs)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
