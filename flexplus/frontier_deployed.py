"""The frontier a deployed router actually reaches, not the oracle's.

Every frontier in this project so far is the Lagrangian oracle's: it allocates
from the true per-exit error, which no decoder has. The paper reports that
number because the signalled configuration transmits the map and therefore
attains it, at 64 to 91 bits a frame. A decoder-side router does not, and the
gap it leaves is the thing an ablation section is for.

So sweep lambda with the router's PREDICTION driving the allocation and the
TRUE error driving the decibel -- the same split as router_fit.py, extended
from one budget to the whole curve, so the result can be BD-integrated
against the oracle frontiers on the same interval.

CPU only, on the dumped tables.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices, allocate  # noqa: E402

RES = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(RES / "cells_train64.npz"))
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--points", type=int, default=32)
    ap.add_argument("--out", default=str(RES / "frontier_dep64.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    tr, te = load(a.train), load(a.test)
    feats = list(tr["features"])
    K, j = int(tr["K"][0]), int(tr["split_depth"][0])
    cell = int(te["cell"][0]); cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    Xtr, Mtr = tr["X"].astype(np.float64), tr["M"].astype(np.float64)
    Xte, Mte, Rte = (te["X"].astype(np.float64), te["M"].astype(np.float64),
                     te["R"].astype(np.float64))

    # gbm_mono: monotone increments above a predicted deepest error. Best of
    # the model classes tried, by 0.68 points over the rank-1 surrogate.
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    base = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
             random_state=0).fit(Xtr, np.log(Mtr[:, -1]))
    lb = base.predict(Xte)
    P = np.full_like(Mte, np.inf)
    P[:, -1] = np.exp(lb)
    acc = np.zeros(len(Xte))
    for k in range(K - 2, j - 1, -1):
        step = np.log(Mtr[:, k] / Mtr[:, k + 1]).clip(min=0)
        m = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
              random_state=0).fit(Xtr, np.log1p(step))
        acc = acc + np.expm1(m.predict(Xte)).clip(min=0)
        P[:, k] = np.exp(lb + acc)
    print("  gbm_mono egitildi", flush=True)

    rows = []
    for q in a.qps:
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
        for lam in [0.0] + list(np.logspace(-9, -1, a.points - 1)):
            k = allocate(Pq, cost_k, lam, j)
            db = 0.0
            for f in frl:
                db += 10 * np.log10(Mq[f, k[f]].mean() / Rq[f].mean())
            db /= len(frl)
            maps = [k[f].reshape(*shapes[i_]) for f, i_ in zip(frl, ids)]
            sv = C.saving(maps, cf)
            rows.append({"qp": q, "lam": float(lam), "saving_pct": sv,
                         "saving_pct_vs_release": sv,
                         "db_vs_uf": db, "db_vs_uf_per_frame": db})
        sp = [r for r in rows if r["qp"] == q]
        print(f"   qp{q:>3}: dB {min(r['db_vs_uf'] for r in sp):+.4f} .. "
              f"{max(r['db_vs_uf'] for r in sp):+.4f}   tasarruf "
              f"{min(r['saving_pct'] for r in sp):.2f} .. "
              f"{max(r['saving_pct'] for r in sp):.2f}%", flush=True)

    Path(a.out).write_text(json.dumps(
        {"script": "flexplus/frontier_deployed.py", "cell_px": cell,
         "router": "gbm_mono, decoder-side features, trained on OpenImages",
         "ckpt": "runs/RECIPE512/ckpt_PAPER.pth.tar", "ckpt_epoch": 4,
         "rows": rows}, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
