"""Killing the tail on the side that cannot bisect anything.

With the map signalled, a per-frame lambda is a hard guarantee and costs the
encoder nothing it was not already computing -- see per_frame_guarantee.py.
The decoder-only configuration has no such option: it never sees the source,
so it cannot measure the decibel it is delivering and cannot bisect on it. Its
tail is prediction error, and prediction error is worst exactly where the
prediction is nearly indifferent.

So: score every exit with the shipped rate-rank surrogate, and when the best
and second-best scores are within a relative margin of each other, take the
DEEPER of the two. A tile whose Lagrangian cannot separate two depths is a
tile the router has no real opinion about, and spending one extra exit there
is cheap insurance. The margin is swept so the compute paid per decibel of
tail removed is a measured exchange rate rather than a claim.

The rate-rank fit is the paper's own functional form, m_hat = exp(a log b + c)
* phi_k, refitted on the OpenImages training split so the test set is never
fitted to. Lambda is re-bisected for every margin, on the TRUE per-frame mean
decibel, so every row below sits on the same headline and differs only in how
its tail is shaped.

CPU only, on the dumped tables. Nothing under ~/FLEX-UF is touched.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from router_fit import Cost, load, frame_slices                    # noqa: E402

RES = HERE / "results"
QPS = [0, 16, 32, 48, 63]


def raterank(tr, te, j):
    """The shipped surrogate, fitted on the training split only."""
    feats = list(tr["features"]); bi = feats.index("bits_sum")
    Xtr, Mtr = tr["X"].astype(np.float64), tr["M"].astype(np.float64)
    Xte, Mte = te["X"].astype(np.float64), te["M"].astype(np.float64)
    b = np.maximum(Xtr[:, bi], 1e-6); b = b / b.mean()
    A = np.stack([np.log(b), np.ones_like(b)], 1)
    al, c0 = np.linalg.lstsq(A, np.log(Mtr[:, -1]), rcond=None)[0]
    lphi = np.log(Mtr[:, j:] / Mtr[:, -1:]).mean(0)
    be = np.maximum(Xte[:, bi], 1e-6); be = be / be.mean()
    P = np.full_like(Mte, np.inf)
    P[:, j:] = np.exp(al * np.log(be) + c0)[:, None] * np.exp(lphi)[None, :]
    return P, float(al)


def allocate_safe(P, ck, lam, j, K, delta):
    """argmin, then one exit deeper wherever the extra depth nearly pays.

    The first version measured the margin against the score itself. That is
    the wrong scale: at the lambdas this budget needs, a score is dominated by
    lambda*c_k, so the relative gap between adjacent exits is pinned near
    (c_{k+1}-c_k)/c_k ~ 0.26 for every tile at every rate. Below that the rule
    never fired and above it, it fired on everything -- which is exactly the
    step the sweep showed.

    The scale that means something is the price of the extra depth. Going one
    deeper costs lambda*(c_{k+1}-c_k) and buys m_k - m_{k+1} of distortion, so

        deepen  <=>  s_{k+1} - s_k  <=  delta * lambda * (c_{k+1} - c_k)

    reads as "the distortion the deeper exit buys covers at least (1-delta) of
    what it costs". delta = 0 never deepens and recovers the plain argmin;
    delta = 1 deepens whenever the deeper exit helps at all, which is always,
    so the useful range is the interior and the sweep prices it.
    """
    S = P[:, j:] + lam * ck[None, j:]
    k = S.argmin(1)
    if delta <= 0:
        return k + j
    n, W = S.shape
    idx = np.arange(n)
    deeper = np.minimum(k + 1, W - 1)
    has = k + 1 <= W - 1
    gap = S[idx, deeper] - S[idx, k]
    price = lam * (ck[j:][deeper] - ck[j:][k])
    close = has & (gap <= delta * price)
    return np.where(close, deeper, k) + j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(RES / "cells_train256.npz"))
    ap.add_argument("--test", default=str(RES / "cells_ctc256.npz"))
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95])
    ap.add_argument("--pred_targets", type=float, nargs="+",
                    default=[0.10, 0.085, 0.07, 0.06, 0.05, 0.04])
    ap.add_argument("--out", default=str(RES / "safe_routing.json"))
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    tr, te = load(a.train), load(a.test)
    M = te["M"].astype(np.float64); R = te["R"].astype(np.float64)
    K, j = int(te["K"][0]), int(te["split_depth"][0])
    ck = np.array([(k + 1) * C.blocks_per_exit for k in range(K)], float)
    P, alpha = raterank(tr, te, j)
    print(f"  rate-rank alpha={alpha:.4f}, K={K} j={j}, hedef {a.budget} dB\n",
          flush=True)

    per = {}
    for q in QPS:
        fr = frame_slices(te["img"], te["qp"], q)
        sel = np.concatenate(fr)
        loc = {v: i for i, v in enumerate(sel)}
        per[q] = ([np.array([loc[v] for v in f]) for f in fr],
                  M[sel], R[sel], P[sel])

    def run(margin):
        dbs, svs = [], []
        for q in QPS:
            frl, Mq, Rq, Pq = per[q]

            def setmean(lam):
                k = allocate_safe(Pq, ck, lam, j, K, margin)
                return float(np.mean([10 * np.log10(
                    Mq[f, k[f]].mean() / Rq[f].mean()) for f in frl]))

            lo, hi = 0.0, 1e-3
            while setmean(hi) <= a.budget and hi < 1e6:
                hi *= 4
            if setmean(0.0) > a.budget:
                lam = 0.0
            else:
                for _ in range(55):
                    mid = 0.5 * (lo + hi)
                    if setmean(mid) <= a.budget:
                        lo = mid
                    else:
                        hi = mid
                lam = lo
            k = allocate_safe(Pq, ck, lam, j, K, margin)
            for f in frl:
                dbs.append(10 * np.log10(Mq[f, k[f]].mean() / Rq[f].mean()))
                kk = k[f]
                c = (C.SHARE_UPSAMPLE
                     + C.SHARE_TRUNK * ((kk + 1).mean() * C.blocks_per_exit)
                     / C.N_TRUNK_BLOCKS + C.adapter[kk].mean() + C.SHARE_HEAD)
                svs.append(100 * (1 - c))
        return np.array(dbs), np.array(svs)

    rows = []
    print(f"{'delta':>7}{'ort dB':>9}{'p50':>8}{'p90':>8}{'p95':>8}{'enkotu':>9}"
          f"{'>0.1':>8}{'>0.15':>7}{'tasarruf':>10}")
    for mg in a.margins:
        db, sv = run(mg)
        row = {"margin": mg, "mean_db": float(db.mean()),
               "p50": float(np.percentile(db, 50)),
               "p90": float(np.percentile(db, 90)),
               "p95": float(np.percentile(db, 95)),
               "max": float(db.max()),
               "over_010": int((db > 0.1 + 1e-9).sum()),
               "over_015": int((db > 0.15).sum()),
               "n": int(db.size), "saving_pct": float(sv.mean())}
        rows.append(row)
        print(f"{mg:>7.2f}{db.mean():>9.4f}{row['p50']:>8.4f}{row['p90']:>8.4f}"
              f"{row['p95']:>8.4f}{row['max']:>9.4f}"
              f"{row['over_010']:>5}/{db.size}{row['over_015']:>7}"
              f"{row['saving_pct']:>10.2f}%", flush=True)

    # ---------------------------------------------------------------------
    # What the decoder CAN do about its own tail.
    #
    # The margin rule fails because the tail is not made of near-ties: on the
    # frames that blow the budget the surrogate is confidently wrong, and no
    # tie-break reaches them. But the decoder does hold m_hat, so it can form a
    # PREDICTED frame decibel and bisect its own lambda per frame against it.
    # That is not a guarantee -- the prediction is what is wrong -- yet it
    # removes the part of the tail that comes from spending one lambda across
    # frames of very different difficulty, which is the larger part.
    #
    # A safety offset is swept with it: bisecting the prediction to a stricter
    # target buys margin for the prediction's own error, and the sweep prices
    # that margin in saving.
    print("\n  kod-cozucu tarafi, kare basina lambda (TAHMIN edilen dB uzerinde):")
    print(f"{'hedef':>7}{'ort dB':>9}{'p50':>8}{'p90':>8}{'p95':>8}{'enkotu':>9}"
          f"{'>0.1':>8}{'>0.15':>7}{'tasarruf':>10}")
    rows2 = []
    for tgt in a.pred_targets:
        dbs, svs = [], []
        for q in QPS:
            frl, Mq, Rq, Pq = per[q]
            for f in frl:
                Pf, Mf, Rf = Pq[f], Mq[f], Rq[f]
                # The decoder's own reference: the deepest exit it can reach is
                # the best it knows, so the predicted decibel is measured
                # against the surrogate's deepest value, not against a truth it
                # does not have.
                ref = Pf[:, -1].mean()

                def pdb(lam):
                    k = allocate_safe(Pf, ck, lam, j, K, 0.0)
                    return float(10 * np.log10(
                        Pf[np.arange(len(k)), k].mean() / ref))
                lo, hi = 0.0, 1e-3
                while pdb(hi) <= tgt and hi < 1e6:
                    hi *= 4
                if pdb(0.0) > tgt:
                    lam = 0.0
                else:
                    for _ in range(45):
                        mid = 0.5 * (lo + hi)
                        if pdb(mid) <= tgt:
                            lo = mid
                        else:
                            hi = mid
                    lam = lo
                k = allocate_safe(Pf, ck, lam, j, K, 0.0)
                dbs.append(10 * np.log10(
                    Mf[np.arange(len(k)), k].mean() / Rf.mean()))
                c = (C.SHARE_UPSAMPLE
                     + C.SHARE_TRUNK * ((k + 1).mean() * C.blocks_per_exit)
                     / C.N_TRUNK_BLOCKS + C.adapter[k].mean() + C.SHARE_HEAD)
                svs.append(100 * (1 - c))
        db = np.array(dbs); sv = np.array(svs)
        r = {"pred_target_db": tgt, "mean_db": float(db.mean()),
             "p50": float(np.percentile(db, 50)),
             "p90": float(np.percentile(db, 90)),
             "p95": float(np.percentile(db, 95)), "max": float(db.max()),
             "over_010": int((db > 0.1 + 1e-9).sum()),
             "over_015": int((db > 0.15).sum()),
             "n": int(db.size), "saving_pct": float(sv.mean())}
        rows2.append(r)
        print(f"{tgt:>7.3f}{db.mean():>9.4f}{r['p50']:>8.4f}{r['p90']:>8.4f}"
              f"{r['p95']:>8.4f}{r['max']:>9.4f}{r['over_010']:>5}/{db.size}"
              f"{r['over_015']:>7}{r['saving_pct']:>10.2f}%", flush=True)

    # ---------------------------------------------------------------------
    # The mode that actually delivers a guarantee without an oracle map.
    #
    # Configuration C already has the encoder running a REPLICA of the
    # decoder's predictor, so the encoder can evaluate the true frame decibel
    # of the allocation the decoder is going to make, for any lambda, and
    # bisect on it. Then it sends the lambda -- one scalar, call it 16 bits
    # against a frame's 200,000 -- and the decoder routes with its own
    # predictor at a lambda that is known to land inside the budget.
    #
    # The map is still never transmitted. What is transmitted is the price.
    print("\n  kare basina lambda SINYALLENIRSE (harita yine gonderilmiyor):")
    dbs, svs, deep = [], [], []
    for q in QPS:
        frl, Mq, Rq, Pq = per[q]
        for f in frl:
            Pf, Mf, Rf = Pq[f], Mq[f], Rq[f]

            def tdb(lam):
                k = allocate_safe(Pf, ck, lam, j, K, 0.0)
                return float(10 * np.log10(
                    Mf[np.arange(len(k)), k].mean() / Rf.mean()))
            lo, hi = 0.0, 1e-3
            while tdb(hi) <= a.budget and hi < 1e6:
                hi *= 4
            lam = 0.0
            if tdb(0.0) <= a.budget:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if tdb(mid) <= a.budget:
                        lo = mid
                    else:
                        hi = mid
                lam = lo
            k = allocate_safe(Pf, ck, lam, j, K, 0.0)
            dbs.append(tdb(lam))
            c = (C.SHARE_UPSAMPLE
                 + C.SHARE_TRUNK * ((k + 1).mean() * C.blocks_per_exit)
                 / C.N_TRUNK_BLOCKS + C.adapter[k].mean() + C.SHARE_HEAD)
            svs.append(100 * (1 - c))
    db = np.array(dbs); sv = np.array(svs)
    sig = {"mean_db": float(db.mean()), "p50": float(np.percentile(db, 50)),
           "p90": float(np.percentile(db, 90)),
           "p95": float(np.percentile(db, 95)), "max": float(db.max()),
           "over_010": int((db > a.budget + 1e-9).sum()),
           "over_015": int((db > 0.15).sum()), "n": int(db.size),
           "saving_pct": float(sv.mean())}
    print(f"{'':>7}{db.mean():>9.4f}{sig['p50']:>8.4f}{sig['p90']:>8.4f}"
          f"{sig['p95']:>8.4f}{sig['max']:>9.4f}{sig['over_010']:>5}/{db.size}"
          f"{sig['over_015']:>7}{sig['saving_pct']:>10.2f}%", flush=True)

    Path(a.out).write_text(json.dumps(
        {"budget_db": a.budget, "raterank_alpha": alpha, "cell_px": int(te["cell"][0]),
         "split_depth": j, "rows": rows,
         "decoder_perframe": rows2,
         "signalled_lambda_perframe": sig}, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
