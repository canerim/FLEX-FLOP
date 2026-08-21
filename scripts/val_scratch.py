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
            rows.append({
                "qp": qp,
                "psnr_per_exit": [round(psnr(v), 3) for v in acc],
                "bpp": round(bpp_sum / len(imgs), 4),
                "spread_dB": round(psnr(acc[-1]) - psnr(acc[0]), 3),
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
              f"bpp {r['bpp']:.4f}  spread {r['spread_dB']:+.3f} dB")
    print(f"  {args.run} @ epoch {rec['epoch']} step {rec['step']} "
          f"({rec['sec']}s, {len(imgs)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
