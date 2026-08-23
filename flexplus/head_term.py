"""The one approximation in the granularity sweep, measured instead of assumed.

granularity.py assembles the routed reconstruction in pixel space out of K
full-frame decodes: position x is read from the decode at x's own depth. For
the TRUNK that is exactly what the dilated receptive field buys -- block b
runs wherever some position within b blocks needs it, so every position sees
the neighbours a full-frame decode would have given it.

The head is not covered by that argument. It has its own 3x3, and at a depth
boundary it reads one cell across into a feature that left the trunk at a
different depth. Pixel-space assembly quietly gives the head a neighbourhood
of the RIGHT depth on both sides, which no real decoder can do; the real one
stitches the exit features and runs the head once over the seam.

So decode the same map both ways and difference them. If the term is small
the sweep stands as written; if it is not, the sweep is optimistic by exactly
this much and has to say so.
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

import ctc_intra as C                                          # noqa: E402
from flexuf.config import FlexUFConfig                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state        # noqa: E402
from flexuf.reference import reference_for                     # noqa: E402
from adaptive_depth import decode_adaptive                     # noqa: E402
from granularity import cell_mse                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--cells", type=int, nargs="+", default=[256, 128, 64, 32])
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--out", default=str(HERE / "results/head_term.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    K, bpe, j = cfg.num_exits, cfg.blocks_per_exit, cfg.split_depth

    seqs, _ = C.discover([])
    frames, names = [], []
    for s in seqs:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1]); names.append(s["name"].split("_")[0])
        if len(frames) >= a.seqs:
            break

    out = {"qp": a.qp, "target_db": a.target, "rows": []}
    with torch.no_grad():
        for cell in a.cells:
            asm, real, n = [], [], 0
            for x, nm in zip(frames, names):
                x = x.to(dev)
                P = max(a.cells + [cfg.rgb_patch])
                ph, pw = (-x.shape[-2]) % P, (-x.shape[-1]) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                recs = net.dec.forward_all_exits(y, q)
                rr = ref.dec.forward_full(y, q)
                base = ((rr - xp) ** 2).mean().item()

                M = torch.stack([cell_mse(r, xp, cell) for r in recs], 1)
                R = cell_mse(rr, xp, cell)
                nh, nw = xp.shape[-2] // cell, xp.shape[-1] // cell
                idx = torch.arange(K, device=dev, dtype=M.dtype)

                def at(lam):
                    k = (M[:, j:] + lam * idx[None, j:]).argmin(1) + j
                    return (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean()
                        / R.mean())).item(), k

                lo, hi = 0.0, 1e-4
                while at(hi)[0] <= a.target and hi < 1e4:
                    hi *= 4
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if at(mid)[0] <= a.target:
                        lo = mid
                    else:
                        hi = mid
                _, k = at(lo)

                # (a) pixel-space assembly: each position read from the
                # full-frame decode at its own depth. What the sweep assumes.
                grid = k.reshape(nh, nw)
                up = grid.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
                up = up[:xp.shape[-2], :xp.shape[-1]]
                asm_img = torch.zeros_like(xp)
                for g in range(K):
                    m = (up == g)[None, None].to(xp.dtype)
                    if m.any():
                        asm_img = asm_img + recs[g] * m

                # (b) the decoder that could actually be built: one trunk with
                # a dilated depth, exit features stitched, head run once.
                Hf, Wf = net.dec.upsample(y).shape[-2:]
                cf = cell // (xp.shape[-2] // Hf)
                dcell = grid.repeat_interleave(cf, 0).repeat_interleave(cf, 1)
                real_img, _ = decode_adaptive(net, y, q, dcell[:Hf, :Wf].float(),
                                              bpe)

                asm.append(10 * torch.log10(torch.tensor(
                    ((asm_img - xp) ** 2).mean().item() / base)).item())
                real.append(10 * torch.log10(torch.tensor(
                    ((real_img - xp) ** 2).mean().item() / base)).item())
                n += 1
            ma, mr = sum(asm) / n, sum(real) / n
            out["rows"].append({"cell_px": cell, "assembled_db": ma,
                                "real_db": mr, "head_term_db": mr - ma})
            print(f"  hucre {cell:>3}px: birlestirilmis {ma:+.4f} dB, "
                  f"gercek {mr:+.4f} dB  ->  baslik terimi {mr - ma:+.4f} dB",
                  flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
