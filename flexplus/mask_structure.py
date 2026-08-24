"""Would a real kernel get this saving, or only a MAC count?

Every saving in this project is counted in multiply-accumulates. That is the
right unit for a claim about arithmetic and the wrong one for a claim about
time: a mask that is 40 per cent off saves nothing if the 40 per cent is
scattered one position at a time, because a dense 3x3 over the whole plane
still runs. AdaDSR made exactly this trade for super-resolution and its own
authors note that pixel-wise sparse convolution is not hardware friendly.

The saving here is not scattered, and this measures how much it is not. The
allocation is piecewise constant on cells -- 8 feature positions across at a
64 px cell -- and the dilation only blurs the boundary by the depth
DIFFERENCE between neighbours. So at each trunk block the needed region
should be a union of large rectangles, and a block-sparse kernel that works
on B x B tiles and must run any tile containing a single needed position
should realise most of the ideal saving.

Reported: the ideal per-position saving, and the saving a B x B block-sparse
implementation would actually get, for B = 8, 16, 32 feature positions, which
is 32, 64 and 128 pixels of image. The gap between them is the honest cost of
making the mask implementable.

CPU only, from the dumped tables. Nothing under ~/FLEX-UF is touched.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices, allocate  # noqa: E402

RES = HERE / "results"


def block_needed(D, first_block, B):
    """Which B x B tiles contain at least one position needing this block."""
    h, w = D.shape
    ph, pw = (-h) % B, (-w) % B
    if ph or pw:
        D = np.pad(D, ((0, ph), (0, pw)), constant_values=-1e9)
    need = D >= first_block - 1e-6
    hb, wb = need.shape[0] // B, need.shape[1] // B
    t = need.reshape(hb, B, wb, B).any((1, 3))
    return t, hb * wb, need.shape[0] * need.shape[1]


def dilate(exit_map, cf, bpe):
    b = (exit_map.astype(np.float32) + 1.0) * bpe
    b = np.repeat(np.repeat(b, cf, 0), cf, 1)
    D = b.copy()
    for _ in range(int(b.max()) + 1):
        p = np.pad(D, 1, mode="edge")
        m = np.maximum.reduce([p[i:i + D.shape[0], jj:jj + D.shape[1]]
                               for i in range(3) for jj in range(3)]) - 1.0
        nxt = np.maximum(D, m)
        if np.array_equal(nxt, D):
            break
        D = nxt
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--out", default=str(RES / "mask_structure.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    te = load(a.test)
    M = te["M"].astype(np.float64); R = te["R"].astype(np.float64)
    K, j = int(te["K"][0]), int(te["split_depth"][0])
    cell = int(te["cell"][0]); cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    bpe = C.blocks_per_exit
    print(f"  hucre {cell}px = {cf} oznitelik konumu, hedef {a.target} dB",
          flush=True)

    out = {"cell_px": cell, "target_db": a.target, "rows": []}
    for q in a.qps:
        fr = frame_slices(te["img"], te["qp"], q)
        sel = np.concatenate(fr)
        loc = {v: i for i, v in enumerate(sel)}
        frl = [np.array([loc[v] for v in f]) for f in fr]
        Mq, Rq = M[sel], R[sel]
        shapes = {}
        for rec in te["grid"]:
            g_, q_, nh_, nw_, _ = str(rec).split(",", 4)
            if int(q_) == q:
                shapes[int(g_)] = (int(nh_), int(nw_))
        ids = sorted(shapes)

        def db_of(lam):
            k = allocate(Mq, cost_k, lam, j)
            t = 0.0
            for f in frl:
                t += 10 * np.log10(Mq[f, k[f]].mean() / Rq[f].mean())
            return t / len(frl), k
        lo, hi = 0.0, 1e-8
        while db_of(hi)[0] <= a.target and hi < 1e6:
            hi *= 4
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if db_of(mid)[0] <= a.target:
                lo = mid
            else:
                hi = mid
        _, k = db_of(lo)

        ideal = np.zeros(len(a.blocks) + 1)
        tot_pos = 0.0
        blk = {B: 0.0 for B in a.blocks}
        adapt = 0.0
        for f, i_ in zip(frl, ids):
            mp = k[f].reshape(*shapes[i_])
            D = dilate(mp, cf, bpe)
            n = D.size
            ran = 0
            for g in range(K):
                ran += int((D >= g * bpe + 1 - 1e-6).sum()) * bpe
            tot_pos += ran / n / C.N_TRUNK_BLOCKS
            for B in a.blocks:
                r2 = 0
                for g in range(K):
                    t, nb, npos = block_needed(D, g * bpe + 1, B)
                    r2 += int(t.sum()) * B * B * bpe
                blk[B] += r2 / n / C.N_TRUNK_BLOCKS
            adapt += C.adapter[mp.reshape(-1)].mean()
        nfr = len(frl)

        def sv(trunk_frac):
            c = (C.SHARE_UPSAMPLE + C.SHARE_TRUNK * trunk_frac
                 + adapt / nfr + C.SHARE_HEAD)
            return 100 * (1 - c)
        row = {"qp": q, "ideal_pct": sv(tot_pos / nfr),
               "blocks": {str(B): sv(blk[B] / nfr) for B in a.blocks}}
        out["rows"].append(row)
        print(f"   qp{q:>3}  ideal {row['ideal_pct']:.2f}%   "
              + "  ".join(f"B={B}: {row['blocks'][str(B)]:.2f}%"
                          for B in a.blocks), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
