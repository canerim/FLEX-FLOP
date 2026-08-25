"""The compute-quality frontier of the per-position decoder, for BD integration.

The saving at 0.1 dB is one sample of a curve, and this project has already
been bitten once by quoting a curve at a point: reading a saving off the
nearest sweep sample rather than the curve made a test-set comparison look
twice as large as it was. scripts/bd_saving.py exists for that reason -- it
integrates the measured frontier over a stated interval -- and it wants rows
of (qp, saving_pct, db_vs_uf_per_frame).

So sweep lambda and emit that, for the per-position decode, at whatever cell
the dumped table was built with. The band is charged by dilating each
allocated map on the feature grid, which is the whole reason a finer cell is
not free.

Runs entirely on the dumped per-cell tables: no GPU, nothing under ~/FLEX-UF
touched, and the main experiment's cards are left alone.
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
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--points", type=int, default=32)
    ap.add_argument("--split", type=int, default=None,
                    help="override the split depth. The dumps carry every "
                         "exit's real error, so j=0 asks what the frontier "
                         "looks like once per-position decoding removes the "
                         "seam the split existed to control.")
    ap.add_argument("--out", default=str(RES / "frontier_pp64.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    te = load(a.test)
    M = te["M"].astype(np.float64); R = te["R"].astype(np.float64)
    K, j = int(te["K"][0]), int(te["split_depth"][0])
    if a.split is not None:
        j = a.split
    cell = int(te["cell"][0]); cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    print(f"  hucre {cell}px, {M.shape[0]} hucre, {a.points} lambda", flush=True)

    rows = []
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

        # A log grid wide enough to run from "everything deepest" to
        # "everything at the split", so the frontier spans whatever interval
        # the BD integration is asked for rather than stopping inside it.
        lams = [0.0] + list(np.logspace(-9, -1, a.points - 1))
        for lam in lams:
            k = allocate(Mq, cost_k, lam, j)
            db = 0.0
            for f in frl:
                db += 10 * np.log10(Mq[f, k[f]].mean() / Rq[f].mean())
            db /= len(frl)
            maps = [k[f].reshape(*shapes[i_]) for f, i_ in zip(frl, ids)]
            sv = C.saving(maps, cf)
            rows.append({"qp": q, "lam": float(lam),
                         "saving_pct": sv, "saving_pct_vs_release": sv,
                         "db_vs_uf": db, "db_vs_uf_per_frame": db,
                         "hist": np.bincount(k, minlength=K).tolist()})
        span = [r for r in rows if r["qp"] == q]
        print(f"   qp{q:>3}: dB {min(r['db_vs_uf'] for r in span):+.4f} .. "
              f"{max(r['db_vs_uf'] for r in span):+.4f}   "
              f"tasarruf {min(r['saving_pct'] for r in span):.2f} .. "
              f"{max(r['saving_pct'] for r in span):.2f}%", flush=True)

    Path(a.out).write_text(json.dumps(
        {"script": "flexplus/frontier.py", "cell_px": cell,
         "decode": "per-position, dilated receptive field",
         "split_depth": j,
         "ckpt": "runs/RECIPE512/ckpt_PAPER.pth.tar", "ckpt_epoch": 4,
         "n_sequences": len(shapes), "frames_per_seq": 1,
         "rows": rows}, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
