"""The released DCVC-UF against ours on Kodak, at 0.1 and 0.2 dB.

Every number in this project is measured on CTC intra frames, which are video
frames: 1080p, YUV 4:2:0, and PSNR weighted 6:1:1. Kodak is the other test set
learned compression is read on, and it is different in every one of those
respects. Whether an allocation trained on Vimeo patches and measured on 1080p
video frames still pays on 24 photographs at 768x512 is not something the CTC
tables can answer, so this asks it directly.

Nothing here is for the paper. The tiling is the paper's -- 256 px, so a Kodak
image carries six tiles against a 1080p frame's thirty-three -- and six tiles
is a coarse allocation, which is a property of the image size and not of the
method. That alone makes this an out-of-distribution check rather than a
headline.

Two conventions change and both are stated in the output. PSNR is computed in
RGB, which is what Kodak is reported in, not the 6:1:1 YUV of the CTC tables;
and the decibel a budget delivers is measured the same way as everywhere else,
as the ratio of our reconstruction's error to the released decoder's on the
same latent, so it is directly comparable to the CTC numbers even though the
absolute PSNRs are not.

The encoder, hyperprior and entropy model are untouched, which the assertion
below proves on the weights rather than trusting a flag, so both codecs read
one bitstream and one bpp serves both curves.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

UF = Path.home() / "FLEX-UF"
sys.path.insert(0, str(UF))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(UF / "scripts"))

from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb            # noqa: E402
from flexuf.config import FlexUFConfig                              # noqa: E402
from flexuf.cost import exit_costs                                  # noqa: E402
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,      # noqa: E402
                         true_frame_mse)
from flexuf.measure import measured_saving_pct                      # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state             # noqa: E402
from flexuf.reference import reference_for                          # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load_kodak(d, n=0):
    """Kodak in the exact domain the decoder was trained on.

    /255, RGB to YCbCr, minus 0.5 -- the chain in DCVC's image_dataset.py. The
    RGB original is kept alongside so PSNR can be reported where Kodak is read.
    """
    out = []
    for p in sorted(Path(d).glob("kodim*.png"))[: n or None]:
        rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.as_tensor(rgb2ycbcr_np(rgb) - 0.5,
                            dtype=torch.float32).permute(2, 0, 1)[None]
        g = torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1)[None]
        out.append((p.name, x, g))
    return out


def psnr_rgb(recon_ycbcr, rgb):
    """PSNR in RGB, on [0,1], after the inverse of the training transform."""
    r = ycbcr2rgb((recon_ycbcr + 0.5).clamp(0, 1))
    mse = F.mse_loss(r, rgb).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--data", default=str(HERE.parent / "data/kodak"))
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--out", default=str(RES / "kodak_curve.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
    assert worst == 0.0, f"encoder differs by {worst}: one bpp cannot serve both"
    cost = exit_costs(cfg, "head").to(dev)

    imgs = load_kodak(a.data, a.max_images)
    print(f"  {len(imgs)} Kodak goruntusu, {cfg.rgb_patch}px karo, "
          f"epoch {ck.get('epoch')}, {dev}", flush=True)

    rows, t0 = [], time.time()
    with torch.no_grad():
        for qp_v in a.qps:
            cache, bpp_s, ps_rel, ps_deep = [], 0.0, 0.0, 0.0
            for name, x, rgb in imgs:
                x = x.to(dev); rgb = rgb.to(dev)
                _, _, H, W = x.shape
                P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                bpp, _, _ = ref._rate(aux, qp, H * W)
                bpp_s += float(bpp.mean())
                ps_rel += psnr_rgb(ref.dec.forward_full(y, q)[:, :, :H, :W], rgb)
                ps_deep += psnr_rgb(net.dec.forward_full(y, q)[:, :, :H, :W], rgb)
                M = tiled_exit_mses(net.dec, y, q, xp, cfg)
                R = reference_frame_mse(ref.dec, y, q, xp)
                cache.append((M, R, H * W, y, q, xp))
            n = len(cache)

            def table_db(lam):
                t = 0.0
                for M, R, _px, _y, _q, _xp in cache:
                    k = (M + lam * cost[None, :]).argmin(1)
                    t += (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                return t / n

            def true_db(lam):
                t = 0.0
                for M, R, _px, y_, q_, xp_ in cache:
                    k = (M + lam * cost[None, :]).argmin(1).clamp(
                        min=cfg.split_depth)
                    t += (10 * torch.log10(
                        true_frame_mse(net.dec, y_, q_, xp_, k) / R)).item()
                return t / n

            floor_db = true_db(0.0)

            def bisect_to(t):
                if table_db(1.0) < t:
                    return 1.0
                lo, hi = 0.0, 1.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if table_db(mid) <= t:
                        lo = mid
                    else:
                        hi = mid
                return lo

            for target in a.budgets:
                if floor_db > target:
                    print(f"   q{qp_v:<3} {target:.2f} dB: taban {floor_db:.4f}, "
                          f"butce ulasilmaz", flush=True)
                    rows.append({"qp": qp_v, "budget_db": target,
                                 "budget_reachable": False,
                                 "floor_db": floor_db,
                                 "bpp": bpp_s / n,
                                 "psnr_release_rgb": ps_rel / n,
                                 "psnr_deepest_rgb": ps_deep / n})
                    continue
                # Bisect on the cheap table, correct with a real decode: the
                # table measures each tile with its neighbours at the SAME
                # exit, a routed frame is mixed, and only a decode settles it.
                inner, lam, td = target, None, None
                for _ in range(6):
                    lam = bisect_to(inner)
                    td = true_db(lam)
                    if abs(td - target) < 5e-4:
                        break
                    inner = max(1e-4, min(1.0, inner + (target - td)))
                svr = msv = 0.0
                for M, R, _px, y_, q_, xp_ in cache:
                    k = (M + lam * cost[None, :]).argmin(1).clamp(
                        min=cfg.split_depth)
                    svr += (1 - cost[k].mean()).item()
                    msv += measured_saving_pct(net.dec, ref.dec, y_, q_, k)
                rows.append({
                    "qp": qp_v, "budget_db": target, "budget_reachable": True,
                    "bpp": bpp_s / n,
                    "psnr_release_rgb": ps_rel / n,
                    "psnr_deepest_rgb": ps_deep / n,
                    "db_vs_uf": td, "floor_db": floor_db, "lam": lam,
                    "saving_pct_vs_release": 100 * svr / n,
                    "saving_pct_measured": msv / n})
                print(f"   q{qp_v:<3} {target:.2f} dB  tasarruf "
                      f"{100*svr/n:6.2f}% (olculen {msv/n:6.2f}%)  "
                      f"dB {td:+.4f}  bpp {bpp_s/n:.4f}  "
                      f"release {ps_rel/n:.3f}", flush=True)

    Path(a.out).write_text(json.dumps(
        {"script": "flexplus/kodak_curve.py", "dataset": "Kodak24",
         "n_images": len(imgs), "tile_px": cfg.rgb_patch,
         "split_depth": cfg.split_depth, "num_exits": cfg.num_exits,
         "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
         "psnr_convention": "RGB, [0,1], after ycbcr2rgb of the reconstruction",
         "db_convention": "per-image ratio to the released decoder, averaged",
         "note": "exploratory, out of distribution; not a paper number",
         "rows": rows}, indent=2))
    print(f"\n  {time.time()-t0:.0f} s, yazildi {a.out}")


if __name__ == "__main__":
    main()
