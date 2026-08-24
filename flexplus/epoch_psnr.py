"""Every exit's PSNR, per epoch, in the codec's own metric.

The training log's psnr_per_exit is on OpenImages patches at a randomly drawn
qp, so its bpp wanders between 0.409 and 0.432 across epochs and the numbers
are not comparable epoch to epoch: part of every difference is a different
rate. This measures the 53 CTC frames instead, at fixed qp, in DCVC-UF's own
metric -- (6*Y + U + V)/8 computed in 4:2:0 on 0..255, which is what
ctc_intra.psnr_611_420 does and therefore what a published DCVC-UF PSNR
means.

Reported per exit: the absolute PSNR, and the decibels below the released
decoder on the same latent, which is the unit every claim in the paper is in.
The gap between the shallowest available exit and the deepest is the ladder's
spread on the test set -- the quantity the saving is a function of.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF)); sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                        # noqa: E402
from flexuf.config import FlexUFConfig                       # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state      # noqa: E402
from flexuf.reference import reference_for                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    K, j = cfg.num_exits, cfg.split_depth
    ep = ck.get("epoch")

    seqs, _ = C.discover([])
    frames = []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append((x[0:1], pl[0]))
    print(f"  epoch {ep}, {len(frames)} CTC karesi", flush=True)

    out = {"ckpt": a.ckpt, "ckpt_epoch": ep, "frames": len(frames), "rows": []}
    with torch.no_grad():
        for qp_v in a.qps:
            acc = [0.0] * K; accr = 0.0
            for x, pl in frames:
                x = x.to(dev); _, _, H, W = x.shape
                P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                recs = net.dec.forward_all_exits(y, q)
                rr = ref.dec.forward_full(y, q)
                for k in range(K):
                    acc[k] += C.psnr_611_420(recs[k][:, :, :H, :W], pl)
                accr += C.psnr_611_420(rr[:, :, :H, :W], pl)
            n = len(frames)
            psnr = [v / n for v in acc]; pr = accr / n
            row = {"qp": qp_v, "psnr_per_exit": psnr, "psnr_release": pr,
                   "db_below_release": [pr - v for v in psnr],
                   "spread_db": psnr[-1] - psnr[j]}
            out["rows"].append(row)
            print(f"   qp{qp_v:>3}  release {pr:.3f} dB   cikislar "
                  + " ".join(f"{v:.3f}" for v in psnr)
                  + f"   yayilim {row['spread_db']:.4f} dB", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  yazildi {a.out}")


if __name__ == "__main__":
    main()
