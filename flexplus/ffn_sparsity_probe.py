"""Is the FFN hidden state sparse per position, even though it is balanced on average?

The first probe killed routing among the chunk-add's four groups: they carry
25% of the energy each and are anti-correlated at -0.30, so none is
dispensable. The second killed low-rank factorisation: the weights need rank
323-351 of 384 for 99% of their energy, and factorising costs 1.16x more than
it saves. Both are statements about STRUCTURE, and both say the same thing --
there is no redundancy in this FFN that holds for every input.

That leaves the axis early exit already uses: which parts matter for THIS
content. TEAL (ICLR 2025) does it inside the FFN and without training, by
thresholding hidden states on magnitude, and it works on SiLU models where
zero-counting does not. Whether it can work here is one measurement: at a
single feature position, is the magnitude spread over all 1536 hidden channels
or concentrated in a few hundred?

Reported per block: the fraction of hidden channels needed, at one position,
to carry 50%, 90% and 99% of the total magnitude, averaged over positions --
and the error the block's output actually takes when the smallest ones are
zeroed, which is what a threshold would do.

Read-only. CPU.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
sys.path.insert(0, str(UF)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(UF / "scripts"))
import ctc_intra as C                                            # noqa: E402
from flexuf.config import FlexUFConfig                           # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state          # noqa: E402

RES = Path(__file__).resolve().parent / "results"
KEEP = [0.75, 0.50, 0.35, 0.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PIN_e9.pth.tar"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 63])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RES / "ffn_sparsity.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    dec = net.dec
    blocks = [g for grp in dec.groups for g in grp]

    seqs, _ = C.discover([])
    frames = []
    for s in seqs[: a.frames]:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append((s["name"], x[0:1]))
    print(f"  {len(frames)} kare, {len(blocks)} blok, epoch {ck.get('epoch')}",
          flush=True)

    acc = {}
    with torch.no_grad():
        for name, x in frames:
            x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            for q in a.qps:
                qp = torch.full((1,), q, dtype=torch.int32, device=dev)
                y, qs, _ = net._encode_to_latent(xp, qp)
                feat = dec.upsample(y)
                for bi, blk in enumerate(blocks):
                    if blk.adaptor is not None:
                        feat = blk.adaptor(feat)
                    inner = blk.dc(feat) + feat
                    h = blk.ffn[0](inner)
                    g = blk.ffn[1].wsilu(h)                    # [1, 4C, h, w]
                    Cc = g.shape[1]
                    # magnitude profile at each position, sorted descending
                    v = g[0].abs().reshape(Cc, -1)             # [4C, N]
                    # subsample positions: the profile is what is wanted, not
                    # every pixel, and sorting 4C x N is the expensive part
                    idx = torch.randperm(v.shape[1])[:4000]
                    vs = v[:, idx].t()                         # [n, 4C]
                    srt, _ = torch.sort(vs, dim=1, descending=True)
                    cum = srt.cumsum(1) / srt.sum(1, keepdim=True).clamp_min(1e-9)
                    rec = acc.setdefault(bi, {"f50": 0.0, "f90": 0.0,
                                              "f99": 0.0, "n": 0,
                                              "err": np.zeros(len(KEEP))})
                    for tag, thr in (("f50", 0.50), ("f90", 0.90), ("f99", 0.99)):
                        rec[tag] += float(((cum < thr).sum(1) + 1).float().mean()
                                          / Cc)
                    # what a magnitude threshold would actually cost
                    out_full = blk.ffn[2](blk.ffn[1](h)) + inner
                    for ki, keep in enumerate(KEEP):
                        k = max(1, int(round(keep * Cc)))
                        thr = v.kthvalue(Cc - k + 1, dim=0).values
                        thr = thr.reshape(1, 1, g.shape[2], g.shape[3])
                        gm = g * (g.abs() >= thr).to(g.dtype)
                        x1 = gm[:, 0::4] + gm[:, 1::4] + gm[:, 2::4] + gm[:, 3::4]
                        out_k = blk.ffn[2](x1) + inner
                        rec["err"][ki] += float(((out_k - out_full) ** 2).mean()
                                                / (out_full ** 2).mean())
                    rec["n"] += 1
                    feat = out_full
            print(f"   {name[:34]:<36} bitti", flush=True)

    out = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "keep": KEEP,
           "what": "per-position magnitude concentration of the FFN hidden "
                   "state, and the block error a magnitude threshold costs",
           "blocks": []}
    print(f"\n{'blok':>5}{'%50 icin':>10}{'%90 icin':>10}{'%99 icin':>10}   "
          + "".join(f"{int(100*k)}% tut".rjust(11) for k in KEEP))
    for bi in sorted(acc):
        r = acc[bi]; n = r["n"]
        e = r["err"] / n
        out["blocks"].append({"block": bi, "frac_for_50": r["f50"] / n,
                              "frac_for_90": r["f90"] / n,
                              "frac_for_99": r["f99"] / n,
                              "rel_err": e.tolist()})
        print(f"{bi:>5}{100*r['f50']/n:>9.1f}%{100*r['f90']/n:>9.1f}%"
              f"{100*r['f99']/n:>9.1f}%   "
              + "".join(f"{x:>10.4f} " for x in e))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
