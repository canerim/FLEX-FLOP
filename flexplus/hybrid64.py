"""Configuration C on the new ground: 64 px cells, no clamp.

C already beats A in the paper -- 35.29% against 33.69% at the lowest rate --
by spending the map selectively rather than everywhere. The encoder computes
both the oracle map and the decoder's own prediction, ranks cells by the
Lagrangian regret of leaving each to the predictor,

    Delta(x) = [ m(x, k_hat) + lam c_k_hat ] - [ m(x, k*) + lam c_k* ]

and transmits the true exit for the top rho fraction. rho=0 is B, rho=1 is A.
That beats A because the resulting allocation is a MIXTURE -- signalled cells
follow the oracle, predicted cells follow the predictor -- and no single
multiplier reaches it.

The two levers this project added are orthogonal to that: they shrink the cell
and unlock the shallow rungs. Nothing about C's argument depends on the cell
size or on the split, so the question is only whether the gains compose. This
measures it.

Bits are charged the way scripts/hybrid_curve.py charges them: an
entropy-coded mask over cells, N*H2(rho), plus ceil(log2(K-j)) bits per
override. At 64 px there are sixteen times as many cells as at 256, so the
mask is sixteen times bigger and has to be priced rather than waved through.

CPU only, on the dumped tables.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices, allocate  # noqa: E402

RES = HERE / "results"
QPS = [0, 16, 32, 48, 63]


def h2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def fit_predictor(Xtr, Mtr, Xte, Mte, K, j):
    """gbm_smear: the best of the model classes tried, refitted for this split."""
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    base = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
             random_state=0).fit(Xtr, np.log(Mtr[:, -1]))
    lb = base.predict(Xte)
    sm = float(np.mean(np.exp(np.log(Mtr[:, -1]) - base.predict(Xtr))))
    P = np.full_like(Mte, np.inf)
    P[:, -1] = np.exp(lb) * sm
    acc = np.zeros(len(Xte)); acc_t = np.zeros(len(Xtr))
    for k in range(K - 2, j - 1, -1):
        step = np.log(Mtr[:, k] / Mtr[:, k + 1]).clip(min=0)
        m = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
              random_state=0).fit(Xtr, np.log1p(step))
        acc = acc + np.expm1(m.predict(Xte)).clip(min=0)
        acc_t = acc_t + np.expm1(m.predict(Xtr)).clip(min=0)
        res = np.log(Mtr[:, k]) - (base.predict(Xtr) + acc_t)
        P[:, k] = np.exp(lb + acc) * float(np.mean(np.exp(res)))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(RES / "cells_train64.npz"))
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    ap.add_argument("--out", default=str(RES / "hybrid64_j0.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    tr, te = load(a.train), load(a.test)
    K = int(tr["K"][0]); j = a.split
    cell = int(te["cell"][0]); cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    Xtr, Mtr = tr["X"].astype(np.float64), tr["M"].astype(np.float64)
    Xte, Mte, Rte = (te["X"].astype(np.float64), te["M"].astype(np.float64),
                     te["R"].astype(np.float64))
    print(f"  hucre {cell}px, j={j}, {Mte.shape[0]} test hucresi", flush=True)
    P = fit_predictor(Xtr, Mtr, Xte, Mte, K, j)
    print("  tahmin edici hazir", flush=True)
    bits_per_override = math.ceil(math.log2(max(2, K - j)))

    out = {"cell_px": cell, "split_depth": j, "target_db": a.target,
           "bits_per_override": bits_per_override, "rows": []}
    for q in QPS:
        fr = frame_slices(te["img"], te["qp"], q)
        sel = np.concatenate(fr)
        loc = {v: i for i, v in enumerate(sel)}
        frl = [np.array([loc[v] for v in f]) for f in fr]
        Pq, Mq, Rq = P[sel], Mte[sel], Rte[sel]
        shapes = {}
        for rec in te["grid"]:
            g_, q_, nh_, nw_, _ = str(rec).split(",", 4)
            if int(q_) == q:
                shapes[int(g_)] = (int(nh_), int(nw_))
        ids = sorted(shapes)

        for rho in a.rhos:
            def maps_at(lam):
                khat = allocate(Pq, cost_k, lam, j)
                kstar = allocate(Mq, cost_k, lam, j)
                out_ = []
                for f in frl:
                    kh, ks = khat[f], kstar[f]
                    # Lagrangian regret of leaving this cell to the predictor.
                    d = ((Mq[f, kh] + lam * cost_k[kh])
                         - (Mq[f, ks] + lam * cost_k[ks]))
                    n = len(f); s = int(round(rho * n))
                    k = kh.copy()
                    if s > 0:
                        idx = np.argpartition(-d, min(s, n) - 1)[:s]
                        k[idx] = ks[idx]
                    out_.append(k)
                return out_

            def db_of(ks):
                t = 0.0
                for f, k in zip(frl, ks):
                    t += 10 * np.log10(Mq[f, k].mean() / Rq[f].mean())
                return t / len(frl)

            lo, hi = 0.0, 1e-8
            while db_of(maps_at(hi)) <= a.target and hi < 1e6:
                hi *= 4
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if db_of(maps_at(mid)) <= a.target:
                    lo = mid
                else:
                    hi = mid
            ks = maps_at(lo)
            db = db_of(ks)
            maps = [k.reshape(*shapes[i_]) for k, i_ in zip(ks, ids)]
            sv = C.saving(maps, cf)
            n_cells = sum(len(f) for f in frl) / len(frl)
            bits = n_cells * h2(rho) + rho * n_cells * bits_per_override
            out["rows"].append({"qp": q, "rho": rho, "saving_pct": sv,
                                "db": db, "map_bits_per_frame": bits})
            print(f"   qp{q:>3} rho={rho:<5}: {sv:6.2f}%  dB {db:.4f}  "
                  f"harita {bits:7.0f} bit/kare", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
