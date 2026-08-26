"""The four per-frame decibel distributions, as raw vectors, for the figure.

per_frame_guarantee.py and safe_routing.py each report quantiles. A CDF needs
the samples, and re-deriving them inside the plotting code would put the
allocation logic in two places. This dumps them once, from the same functions
the tables came from.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices                 # noqa: E402
from safe_routing import raterank, allocate_safe                # noqa: E402
from per_frame_guarantee import alloc, bisect, frame_db         # noqa: E402

RES = HERE / "results"
QPS = [0, 16, 32, 48, 63]
B = 0.1

C = Cost(json.loads((RES / "cost_constants.json").read_text()))
tr, te = load(str(RES / "cells_train256.npz")), load(str(RES / "cells_ctc256.npz"))
M = te["M"].astype(np.float64); R = te["R"].astype(np.float64)
K, j = int(te["K"][0]), int(te["split_depth"][0])
ck = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
P, _ = raterank(tr, te, j)

out = {"oracle_global": [], "oracle_perframe": [],
       "pred_global": [], "pred_siglam": []}
for q in QPS:
    fr = frame_slices(te["img"], te["qp"], q)
    sel = np.concatenate(fr); loc = {v: i for i, v in enumerate(sel)}
    frl = [np.array([loc[v] for v in f]) for f in fr]
    Mq, Rq, Pq = M[sel], R[sel], P[sel]

    lg = bisect(lambda l: float(np.mean(
        [frame_db(Mq, Rq, alloc(Mq, ck, l, j), f) for f in frl])), B)
    kg = alloc(Mq, ck, lg, j)
    out["oracle_global"] += [frame_db(Mq, Rq, kg, f) for f in frl]

    lp = bisect(lambda l: float(np.mean(
        [10 * np.log10(Mq[f, allocate_safe(Pq, ck, l, j, K, 0.0)[f]].mean()
                       / Rq[f].mean()) for f in frl])), B)
    kp = allocate_safe(Pq, ck, lp, j, K, 0.0)
    out["pred_global"] += [float(10 * np.log10(Mq[f, kp[f]].mean() / Rq[f].mean()))
                           for f in frl]

    for f in frl:
        Mf, Rf, Pf = Mq[f], Rq[f], Pq[f]
        n = np.arange(len(f))
        lo = bisect(lambda l: float(10 * np.log10(
            Mf[n, alloc(Mf, ck, l, j)].mean() / Rf.mean())), B)
        ko = alloc(Mf, ck, lo if lo is not None else 0.0, j)
        out["oracle_perframe"].append(
            float(10 * np.log10(Mf[n, ko].mean() / Rf.mean())))
        ls = bisect(lambda l: float(10 * np.log10(
            Mf[n, allocate_safe(Pf, ck, l, j, K, 0.0)].mean() / Rf.mean())), B)
        ks = allocate_safe(Pf, ck, ls if ls is not None else 0.0, j, K, 0.0)
        out["pred_siglam"].append(
            float(10 * np.log10(Mf[n, ks].mean() / Rf.mean())))

(RES / "per_frame_dists.json").write_text(json.dumps(out))
for k_, v in out.items():
    v = np.array(v)
    print(f"  {k_:<18} n={v.size}  ort {v.mean():.4f}  p95 {np.percentile(v,95):.4f}"
          f"  max {v.max():.4f}  >0.1: {int((v>B+1e-9).sum())}")
