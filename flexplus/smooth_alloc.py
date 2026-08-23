"""The cost stopped being separable, so the allocation should stop being.

With tiles, a tile's cost depends on its own exit and nothing else, so the
Lagrangian allocation is a per-tile argmin and that argmin is exactly
optimal. Per-position depth breaks that: the trunk runs wherever the dilation
D(x) = max_y (d(y) - dist(x, y)) reaches, so a cell's real cost depends on how
much DEEPER its neighbours are. Two allocations with the same histogram of
exits can differ by several points of saving depending on whether the deep
cells are clustered or scattered.

The separable argmin is blind to this. It pays the band and does not know it
is paying. So: keep the same distortion table and the same budget, and add a
term that prices depth disagreement with neighbours,

    argmin_k  m_k + lam * c_k + mu * sum_neighbours |k - d_j|

solved by coordinate descent from the separable solution. mu = 0 recovers
today's allocation exactly. The question is whether some mu > 0 buys more
band than it costs in distortion -- and the band is 1.6 points of saving at
64 px cells and 2.8 at 32 px, which is what a finer allocation currently
spends its winnings on.

CPU only, on the dumped tables. Nothing under ~/FLEX-UF is touched.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices, dilated_blocks  # noqa: E402

RES = HERE / "results"


def icm(m, cost_k, lam, mu, j, K, shape, iters=6, init=None):
    """Coordinate descent on m_k + lam c_k + mu * sum_nbr |k - d_nbr|.

    Sweeps in a fixed raster order rather than randomly: the objective is not
    convex and the order changes the fixed point, so a deterministic one keeps
    the number reproducible.
    """
    nh, nw = shape
    base = m[:, j:] + lam * cost_k[None, j:]          # [N, K-j]
    ks = np.arange(j, K)
    d = (base.argmin(1) + j).reshape(nh, nw) if init is None else init.copy()
    if mu <= 0:
        return d
    B = base.reshape(nh, nw, -1)
    for _ in range(iters):
        changed = 0
        nb = np.zeros((nh, nw, len(ks)))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(d, (dy, dx), (0, 1)).astype(np.float64)
            if dy == 1:
                sh[0] = d[0]
            elif dy == -1:
                sh[-1] = d[-1]
            elif dx == 1:
                sh[:, 0] = d[:, 0]
            else:
                sh[:, -1] = d[:, -1]
            nb += np.abs(ks[None, None, :] - sh[:, :, None])
        nxt = ks[np.argmin(B + mu * nb, axis=2)]
        changed = int((nxt != d).sum())
        d = nxt
        if changed == 0:
            break
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--mus", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--out", default=str(RES / "smooth_alloc.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    te = load(a.test)
    M = te["M"].astype(np.float64); R = te["R"].astype(np.float64)
    K, j = int(te["K"][0]), int(te["split_depth"][0])
    cell = int(te["cell"][0]); cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    print(f"  hucre {cell}px, oznitelik izgarasinda {cf}, "
          f"{M.shape[0]} hucre", flush=True)

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

        for mu in a.mus:
            def maps_at(lam):
                out_ = []
                for f, i_ in zip(frl, ids):
                    out_.append(icm(Mq[f], cost_k, lam, mu, j, K, shapes[i_]))
                return out_

            def db_of(maps):
                t = 0.0
                for f, mp in zip(frl, maps):
                    k = mp.reshape(-1)
                    t += 10 * np.log10(Mq[f, k].mean() / Rq[f].mean())
                return t / len(frl)

            lo, hi = 0.0, 1e-8
            while db_of(maps_at(hi)) <= a.target and hi < 1e6:
                hi *= 4
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                if db_of(maps_at(mid)) <= a.target:
                    lo = mid
                else:
                    hi = mid
            maps = maps_at(lo)
            db = db_of(maps)
            sv = C.saving(maps, cf)
            flat = 0.0
            for mp in maps:
                fr_ = C.SHARE_TRUNK * ((mp + 1).mean() * C.blocks_per_exit) \
                    / C.N_TRUNK_BLOCKS
                flat += (C.SHARE_UPSAMPLE + fr_
                         + C.adapter[mp.reshape(-1)].mean() + C.SHARE_HEAD)
            svflat = 100 * (1 - flat / len(maps))
            out["rows"].append({"qp": q, "mu": mu, "saving_pct": sv,
                                "saving_no_band_pct": svflat,
                                "band_pct": svflat - sv, "db": db})
            print(f"   qp{q:>3} mu={mu:<5}: {sv:6.2f}%  "
                  f"(bantsiz {svflat:6.2f}%, bant {svflat - sv:4.2f}) "
                  f"dB {db:.4f}", flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
