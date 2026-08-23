"""What routing granularity is worth, once the seam stops being the reason for it.

The tile is 256 px because that is the size at which a cut is affordable: the
decoder is trained with replicate padding at tile borders, and every border a
3x3 crosses reads invented values. Finer tiles mean more border per unit
area, so the seam grows as 1/side and the ladder's own operating point pays
it -- scripts/raterank_curve.py bisects on a cheap per-tile table and then
corrects against a real decode of the mixed map, and that correction pulls
the inner budget from 0.100 to about 0.085 dB. A sixth of the distortion
budget is spent on the cut itself.

Per-position depth with a dilated receptive field removes that term outright:
a position's output is what a full-frame decode to that position's own depth
would have produced, whatever its neighbours do. So granularity stops costing
distortion and starts costing only compute -- the band whose width is the
depth DIFFERENCE between neighbours. That trade has never been measured here,
and it is the one lever that does not need a single weight retrained.

This sweeps the allocation cell from 256 px down to 32 px on the pinned
checkpoint, and for each one reports the oracle saving at the paper's budget
in the paper's decibel, with the band charged honestly by dilating the
allocated map and counting the blocks that actually run.

Two approximations are stated rather than hidden. The distortion is assembled
in pixel space from K full-frame decodes, which is the definition of the
exactness the dilation buys for the TRUNK; the head has its own 3x3 and at a
depth boundary it reads across one. That term is measured separately against
the real per-position decode rather than assumed to be zero.

Nothing under ~/FLEX-UF is modified.
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
from flexuf.config import FlexUFConfig, N_TRUNK_BLOCKS         # noqa: E402
from flexuf.cost import (SHARE_UPSAMPLE, SHARE_TRUNK, SHARE_HEAD,  # noqa: E402
                         adapter_vs_block)
from flexuf.model import FlexUFIntra, load_flexuf_state        # noqa: E402
from flexuf.reference import reference_for                     # noqa: E402
from adaptive_depth import dilate_depth                        # noqa: E402

PER_BLOCK = SHARE_TRUNK / N_TRUNK_BLOCKS


def cell_mse(img, target, cell):
    """Mean squared error per `cell`-sized square, row major."""
    e = (img - target) ** 2
    return F.avg_pool2d(e.mean(1, keepdim=True), cell, cell)[0, 0].reshape(-1)


def frame_cost(depth_cells, cfg, bpe, K, feat_hw, cell_feat):
    """Cost of one frame's per-position decode, in units of a released decode.

    The trunk is charged on the blocks the dilation actually runs, not on the
    blocks the map asks for: a position that leaves at exit 1 but sits beside
    one that leaves at exit 5 keeps being computed until the deeper
    neighbour's receptive field no longer reaches it. That band is the whole
    price of fine granularity and it is what this counts.

    Upsample and head are frame-level and run once; with per-position depth
    there is no seam, so no seam repair is charged. The deepest map therefore
    costs exactly 1.0 -- a released decode -- which is what makes the saving
    directly comparable to the paper's.
    """
    Hf, Wf = feat_hw
    e = depth_cells.repeat_interleave(cell_feat, 0).repeat_interleave(cell_feat, 1)
    e = e[:Hf, :Wf].float()
    D = dilate_depth((e + 1.0) * bpe)
    ran = 0
    for g in range(K):
        first = g * bpe + 1
        ran += int((D >= float(first) - 1e-6).sum().item()) * bpe
    n = Hf * Wf
    trunk = SHARE_TRUNK * (ran / n) / N_TRUNK_BLOCKS
    ad = torch.tensor([0.0 if k == K - 1 else
                       adapter_vs_block(cfg.adapter_kind, k, cfg) * PER_BLOCK
                       for k in range(K)], device=e.device)
    adapter = ad[e.long().clamp(0, K - 1)].mean().item()
    flat = SHARE_TRUNK * ((e + 1).mean().item() * bpe) / N_TRUNK_BLOCKS
    return (SHARE_UPSAMPLE + trunk + adapter + SHARE_HEAD,
            SHARE_UPSAMPLE + flat + adapter + SHARE_HEAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--cells", type=int, nargs="+", default=[256, 128, 64, 32])
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--frames", type=int, default=53)
    ap.add_argument("--out", default=str(HERE / "results/granularity.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    K, bpe = cfg.num_exits, cfg.blocks_per_exit
    j = cfg.split_depth
    print(f"  K={K} j={j} bpe={bpe}  paper tile={cfg.rgb_patch}px", flush=True)

    seqs, _ = C.discover([])
    frames = []
    for s in seqs:
        x, _pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1])
        if len(frames) >= a.frames:
            break
    print(f"  {len(frames)} CTC karesi", flush=True)

    out = {"ckpt": a.ckpt, "target_db": a.target, "frames": len(frames),
           "rows": []}
    with torch.no_grad():
        for qp_v in a.qps:
            # Per-frame tables, one per granularity: cell MSE at every exit,
            # and the released decoder's own cell MSE from the same latent.
            cache = {c: [] for c in a.cells}
            for x in frames:
                x = x.to(dev); _, _, H, W = x.shape
                P = max(a.cells + [cfg.rgb_patch])
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                recs = net.dec.forward_all_exits(y, q)
                rr = ref.dec.forward_full(y, q)
                Hf, Wf = net.dec.upsample(y).shape[-2:]
                for c in a.cells:
                    M = torch.stack([cell_mse(r, xp, c) for r in recs], 1)
                    R = cell_mse(rr, xp, c)
                    nh, nw = xp.shape[-2] // c, xp.shape[-1] // c
                    cache[c].append((M, R, nh, nw, (Hf, Wf),
                                     c // (xp.shape[-2] // Hf)))
            row = {"qp": qp_v, "cells": []}
            for c in a.cells:
                dat = cache[c]

                def at(lam, dat=dat):
                    ks, dbs = [], []
                    for M, R, *_ in dat:
                        # argmin over the exits that exist. Below the split
                        # the decoder clamps, so those columns carry exit j's
                        # error at a lower billed cost; including them and
                        # clipping after prices the shallow option at a rate
                        # no decoder charges.
                        k = (M[:, j:] + lam * torch.arange(
                            j, K, device=dev, dtype=M.dtype)[None, :]
                             ).argmin(1) + j
                        ks.append(k)
                        dbs.append((10 * torch.log10(
                            M.gather(1, k[:, None]).squeeze(1).mean()
                            / R.mean())).item())
                    return sum(dbs) / len(dbs), ks

                lo, hi = 0.0, 1e-4
                while at(hi)[0] <= a.target and hi < 1e4:
                    hi *= 4
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if at(mid)[0] <= a.target:
                        lo = mid
                    else:
                        hi = mid
                db, ks = at(lo)
                tot = totflat = 0.0
                for (M, R, nh, nw, fhw, cf), k in zip(dat, ks):
                    cst, flat = frame_cost(k.reshape(nh, nw), cfg, bpe, K,
                                           fhw, cf)
                    tot += cst; totflat += flat
                n = len(dat)
                sv = 100 * (1 - tot / n)
                svflat = 100 * (1 - totflat / n)
                mix = torch.bincount(torch.cat(ks), minlength=K).tolist()
                row["cells"].append({"cell_px": c, "db": db,
                                     "saving_pct_vs_release": sv,
                                     "saving_no_band_pct": svflat,
                                     "band_cost_pct": svflat - sv,
                                     "hist": mix})
                print(f"    qp{qp_v:>3} hucre {c:>3}px: {sv:5.2f}% "
                      f"(bantsiz {svflat:5.2f}%, bant {svflat - sv:4.2f} puan) "
                      f"dB {db:.4f}  mix {mix}", flush=True)
            out["rows"].append(row)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
