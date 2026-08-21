"""The rate-distortion curve in absolute units, ours against the real DCVC-UF.

Everything else in this paper is relative: decibels below the released decoder,
per cent of its multiply-accumulates. That is the right frame for the claim --
the method's whole point is that the bitstream does not change -- but it means
the paper never shows the curve a reader of a compression paper looks for
first, bits per pixel against PSNR, with the released codec on it.

This measures that curve on the CTC intra frames:

  * bpp, from the released encoder. It is the same number for both, because
    the encoder, hyperprior and entropy model are untouched and the assertion
    below proves it on the weights rather than trusting the flag.
  * PSNR of the released decoder, its full-depth decode of that latent.
  * PSNR of our deepest exit, which is what the drift is measured against.

The routed operating points are not measured here. They are the released PSNR
minus the decibels a budget delivered, which the signalled sweeps already
record on the same frames -- and taking them from there rather than
re-measuring keeps this curve on the same numbers as every table.

    python scripts/rd_absolute.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar \\
        --qps 0 16 32 48 63 --out results/rd_absolute_PAPER.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu import pick as _gpu                      # noqa: E402
import ctc_intra as C                             # noqa: E402
from flexuf.config import FlexUFConfig            # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402
from flexuf.reference import reference_for        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/rd_absolute_PAPER.json")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()

    dev = a.device
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    base = torch.load(reference_for(cfg, a.ref), map_location="cpu",
                      weights_only=False)

    stock = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(stock, base)
    ours = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ours, ck)

    sd_s, sd_o = stock.enc.state_dict(), ours.enc.state_dict()
    assert set(sd_s) == set(sd_o), "encoder keys differ"
    worst = max((sd_s[k] - sd_o[k]).abs().max().item() for k in sd_s)
    assert worst == 0.0, f"encoder differs by {worst}; one bpp for two curves " \
                         f"would be a fiction"

    seqs, _ = C.discover([])
    frames = []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            frames.append((s["cls"], x[0:1], pl[0]))
    print(f"  {len(frames)} CTC intra frames, encoder identical "
          f"(max|diff| = {worst})")

    rows = []
    for qp_v in a.qps:
        bpp_s = ps_s = ps_o = 0.0
        with torch.no_grad():
            for _, x, pl in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                ph, pw = (-H) % cfg.rgb_patch, (-W) % cfg.rgb_patch
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = stock._encode_to_latent(xp, qp)
                # bpp over the frame's own pixels, not the padded canvas: the
                # padding is a decoder-side convenience and charging the file
                # for it would flatter every rate in the plot.
                bpp, _, _ = stock._rate(aux, qp, H * W)
                bpp_s += float(bpp.mean())
                ps_s += C.psnr_611_420(stock.dec.forward_full(y, q)[:, :, :H, :W], pl)
                ps_o += C.psnr_611_420(ours.dec.forward_full(y, q)[:, :, :H, :W], pl)
        n = len(frames)
        rows.append({"qp": qp_v, "bpp": bpp_s / n,
                     "psnr_release": ps_s / n, "psnr_deepest": ps_o / n,
                     "drift_db": (ps_o - ps_s) / n})
        print(f"  q{qp_v:<3} bpp {bpp_s/n:.4f}  release {ps_s/n:.3f} dB  "
              f"ours {ps_o/n:.3f} dB  drift {(ps_o-ps_s)/n:+.3f}")

    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "ref": str(reference_for(cfg, a.ref)),
         "n_frames": len(frames), "frames_per_seq": a.frames,
         "psnr_convention": "6:1:1 weighted, YUV420, as scripts/ctc_intra.py",
         "rows": rows}, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
