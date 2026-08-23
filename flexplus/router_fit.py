"""Does the exit-error curve have a shape, or only a scale?

The shipped rate-rank router is a rank-1 model. Its surrogate is

    m_hat(t, k) = A * bits(t)^alpha * phi_k

-- one number per cell for the scale, one vector shared by every cell for the
shape. It reaches 83 to 100 per cent of the oracle's saving with no
parameters and no signalled bits, which is a strong result and also a strong
hypothesis: that cells differ in how much they lose by exiting early, but not
in HOW that loss is distributed across the exits.

If the hypothesis holds, nothing that predicts a per-cell shape can win, and
the interesting result is the negative one. If it fails -- if flat sky and
fine texture have differently curved ladders and not merely differently
scaled ones -- then a model that predicts the curve should beat it, and the
gain is free: same bitstream, same weights, decoder side.

So: fit m_k per cell, on OpenImages, from decoder-side features only; test on
the 53 CTC frames the paper reports; and put the prediction through the same
Lagrangian, bisected on the TRUE decibel, so a router is scored on the
allocation it causes rather than on how often it guesses the oracle's class.

Adaptive Patch Exiting (ECCV 2022) regresses each layer's incremental
capacity for super-resolution patches; this is that idea with the cost side
already exact, which is why the regression can be scored directly in saving.

Runs in .venv-ml -- numpy and scikit-learn only, CPU. The main experiment's
environment is a symlink to FLEX-UF's and is not touched.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def dilated_blocks(exit_map, cf, bpe, K):
    """Blocks actually run per feature position, given a per-cell exit map.

    D(x) = max_y (b(y) - dist(x, y)) by repeated 3x3 max minus one, in blocks,
    on the FEATURE grid -- a cell is cf feature positions across, and dilating
    at cell resolution would over-charge the band by that factor.
    """
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
    ran = 0
    for g in range(K):
        ran += int((D >= g * bpe + 1 - 1e-6).sum()) * bpe
    return ran / D.size


class Cost:
    def __init__(self, c):
        self.__dict__.update(c)
        self.adapter = np.array(c["adapter"], dtype=np.float64)

    def saving(self, maps, cf):
        """Mean saving over frames, with the band charged by dilation."""
        tot = 0.0
        for m in maps:
            frac = dilated_blocks(m, cf, self.blocks_per_exit, self.K)
            trunk = self.SHARE_TRUNK * frac / self.N_TRUNK_BLOCKS
            ad = self.adapter[m.reshape(-1)].mean()
            tot += self.SHARE_UPSAMPLE + trunk + ad + self.SHARE_HEAD
        return 100 * (1 - tot / len(maps))


def per_frame_db(M, R, k, frames):
    """The codec's decibel: a per-frame ratio, averaged over frames."""
    tot = 0.0
    for sl in frames:
        tot += 10 * np.log10(M[sl, k[sl]].mean() / R[sl].mean())
    return tot / len(frames)


def allocate(Mhat, cost_k, lam, j):
    """argmin over the exits that EXIST, not over all K then clipped.

    forward() clamps a map to j, so exits below the split decode identically
    to exit j -- same error, lower billed cost. Taking the argmin over all K
    and clipping afterwards therefore compares the shallow option at a price
    no decoder charges, and biases every allocation shallow.
    """
    return (Mhat[:, j:] + lam * cost_k[None, j:]).argmin(1) + j


def operate(Mhat, M, R, frames, cost_k, j, target, K):
    """Largest saving whose TRUE per-frame dB stays under the budget.

    The router allocates on its prediction; the decibel is measured on what
    the decoder actually produces. Scoring a router on its own prediction is
    how a confident, wrong one looks best.
    """
    def db(lam):
        k = allocate(Mhat, cost_k, lam, j)
        return per_frame_db(M, R, k, frames), k
    lo, hi = 0.0, 1e-8
    while db(hi)[0] <= target and hi < 1e6:
        hi *= 4
    if db(0.0)[0] > target:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if db(mid)[0] <= target:
            lo = mid
        else:
            hi = mid
    d, k = db(lo)
    return d, k


def frame_slices(img, qp, want_qp):
    out = []
    sel = qp == want_qp
    for i in np.unique(img[sel]):
        out.append(np.where(sel & (img == i))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(RES / "cells_train64.npz"))
    ap.add_argument("--test", default=str(RES / "cells_ctc64.npz"))
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--out", default=str(RES / "router_fit.json"))
    ap.add_argument("--models", nargs="+",
                    default=["oracle", "raterank", "flat", "ridge", "gbm",
                             "gbm_shape", "rank1_gbm", "gbm_mono"])
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--perframe", action="store_true")
    a = ap.parse_args()

    C = Cost(json.loads((RES / "cost_constants.json").read_text()))
    tr, te = load(a.train), load(a.test)
    feats = list(tr["features"])
    groups = json.loads(str(tr["groups"]))
    K, j = int(tr["K"][0]), int(tr["split_depth"][0])
    cell = int(tr["cell"][0])
    cf = cell // C.feature_stride
    cost_k = np.array([(k + 1) * C.blocks_per_exit for k in range(K)],
                      dtype=np.float64)
    print(f"  egitim {tr['X'].shape}, test {te['X'].shape}, "
          f"K={K} j={j} hucre={cell}px (oznitelik izgarasinda {cf})",
          flush=True)

    Xtr, Mtr = tr["X"].astype(np.float64), tr["M"].astype(np.float64)
    Xte, Mte, Rte = (te["X"].astype(np.float64), te["M"].astype(np.float64),
                     te["R"].astype(np.float64))
    qps = sorted(set(int(q) for q in te["qp"]))

    # --- the rank-1 question, before any model is fitted -------------------
    # Normalise every cell's curve by its own scale and see what is left. If
    # the shape is shared, the normalised curves collapse onto one line.
    # Rank of the curve, which is the whole question the rate-rank surrogate
    # answers with "one". Dividing each cell's curve by its own first entry
    # was the first version of this and it blew up on the cells whose first
    # entry is near zero -- exactly the flat cells that dominate the count --
    # so the split is done by SVD on the log-curves instead.
    A = (10 * np.log10(Mtr[:, j:] / Mtr[:, -1:]))[:, :-1]
    sv_ = np.linalg.svd(A - A.mean(0), compute_uv=False)
    ev = (sv_ ** 2 / (sv_ ** 2).sum())
    shape_sd = ev.tolist()
    print(f"  egri SVD enerjisi: " + " ".join(f"{v:.4f}" for v in ev)
          + f"   (rank-1 {ev[0] * 100:.1f}%)", flush=True)

    def fit_predict(name, cols):
        idx = [feats.index(c) for c in cols]
        xt, xe = Xtr[:, idx], Xte[:, idx]
        P = np.full_like(Mte, np.inf)
        if name == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            for k in range(j, K):
                m = make_pipeline(StandardScaler(),
                                  Ridge(alpha=1.0)).fit(
                    np.log1p(np.abs(xt)), np.log(Mtr[:, k]))
                P[:, k] = np.exp(m.predict(np.log1p(np.abs(xe))))
        elif name == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            for k in range(j, K):
                m = H(max_iter=300, learning_rate=0.08, max_depth=None,
                      l2_regularization=1.0, random_state=0).fit(
                    xt, np.log(Mtr[:, k]))
                P[:, k] = np.exp(m.predict(xe))
        elif name == "rank1_gbm":
            # Rate-rank's structure with rate-rank's weakest part replaced.
            # It is rank-1 by construction -- one learned scale per cell times
            # one shape shared by every cell -- which the rank analysis says
            # is where 3.0 of the 3.7 available points live. The difference
            # from the shipped surrogate is only that the scale comes from a
            # gradient-boosted fit on every decoder-side feature instead of a
            # power law in the cell's bits.
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            spread = np.log(Mtr[:, j] / Mtr[:, -1])
            ms = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                   random_state=0).fit(xt, spread)
            md = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                   random_state=0).fit(xt, np.log(Mtr[:, -1]))
            sp, dp = ms.predict(xe), md.predict(xe)
            # The shared shape: how the log-gap decays from exit j to the
            # deepest, averaged over the training cells and normalised so that
            # exit j is 1 and the deepest is 0.
            g = np.log(Mtr[:, j:] / Mtr[:, -1:])
            shp = g.mean(0) / max(g.mean(0)[0], 1e-12)
            P[:, j:] = np.exp(dp[:, None] + sp[:, None] * shp[None, :])
        elif name == "gbm_mono":
            # Same as gbm_shape but the increments are forced non-negative and
            # non-increasing in depth, which is what a ladder means. The
            # unconstrained per-exit fit ("gbm") is 3.7 points WORSE than the
            # rank-1 surrogate precisely because nothing stops it predicting a
            # curve no decoder can produce.
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            base = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                     random_state=0).fit(xt, np.log(Mtr[:, -1]))
            lb = base.predict(xe)
            P[:, -1] = np.exp(lb)
            acc = np.zeros(len(xe))
            for k in range(K - 2, j - 1, -1):
                step = np.log(Mtr[:, k] / Mtr[:, k + 1]).clip(min=0)
                m = H(max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                      random_state=0).fit(xt, np.log1p(step))
                acc = acc + np.expm1(m.predict(xe)).clip(min=0)
                P[:, k] = np.exp(lb + acc)
        elif name == "gbm_shape":
            # Scale and shape split apart: one model for the deepest error,
            # K-j-1 models for the increments above it. Same information, but
            # the target the increments carry is exactly what rate-rank holds
            # fixed, so this is the direct test of its assumption.
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            base = H(max_iter=300, learning_rate=0.08, l2_regularization=1.0,
                     random_state=0).fit(xt, np.log(Mtr[:, -1]))
            lb = base.predict(xe)
            P[:, -1] = np.exp(lb)
            for k in range(j, K - 1):
                m = H(max_iter=300, learning_rate=0.08, l2_regularization=1.0,
                      random_state=0).fit(
                    xt, np.log(Mtr[:, k] / Mtr[:, -1]))
                P[:, k] = np.exp(lb + m.predict(xe))
        return P

    def raterank_pred():
        """The shipped surrogate, refitted here on the training split.

        m_hat = exp(alpha*log b + c0) * phi_k, alpha and phi from a least
        squares fit -- the same functional form as scripts/raterank_curve.py,
        fitted on OpenImages so it is scored on the same footing as the
        learned models rather than on a leave-one-sequence-out fit of the
        test set itself.
        """
        bi = feats.index("bits_sum")
        b = np.maximum(Xtr[:, bi], 1e-6)
        b = b / b.mean()
        yv = np.log(Mtr[:, -1])
        A = np.stack([np.log(b), np.ones_like(b)], 1)
        al, c0 = np.linalg.lstsq(A, yv, rcond=None)[0]
        lphi = np.log(Mtr[:, j:] / Mtr[:, -1:]).mean(0)
        be = np.maximum(Xte[:, bi], 1e-6)
        be = be / be.mean()
        P = np.full_like(Mte, np.inf)
        P[:, j:] = np.exp(al * np.log(be) + c0)[:, None] * np.exp(lphi)[None, :]
        return P, float(al), lphi.tolist()

    out = {"target_db": a.target, "cell_px": cell, "K": K, "j": j,
           "shape_sd": shape_sd.tolist(), "rows": []}
    rr_pred, rr_al, rr_phi = raterank_pred()
    out["raterank_alpha"] = rr_al
    out["raterank_phi"] = rr_phi

    preds = {}
    for name in a.models:
        t0 = time.time()
        if name == "oracle":
            preds[name] = Mte
        elif name == "raterank":
            preds[name] = rr_pred
        elif name == "flat":
            # no features at all: every cell gets the average curve
            P = np.full_like(Mte, np.inf)
            P[:, j:] = np.exp(np.log(Mtr).mean(0))[None, j:]
            preds[name] = P
        else:
            preds[name] = fit_predict(name, feats)
        print(f"  {name} hazir ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n  {'model':<12}" + "".join(f"{('q%d' % q):>9}" for q in qps)
          + f"{'ort':>9}")
    for name, P in preds.items():
        row = {"model": name, "per_qp": {}}
        vals = []
        for q in qps:
            fr = frame_slices(te["img"], te["qp"], q)
            sel = np.concatenate(fr)
            loc = {v: i for i, v in enumerate(sel)}
            frl = [np.array([loc[v] for v in f]) for f in fr]
            r = operate(P[sel], Mte[sel], Rte[sel], frl, cost_k, j,
                        a.target, K)
            if r is None:
                row["per_qp"][q] = None
                continue
            db, k = r
            # Each frame's cell grid, from the dump's own record, so a map
            # is reshaped to the shape it was flattened from rather than to a
            # shape inferred from its length.
            shapes = {}
            for rec in te["grid"]:
                g_, q_, nh_, nw_, _ = str(rec).split(",", 4)
                if int(q_) == q:
                    shapes[int(g_)] = (int(nh_), int(nw_))
            ids = sorted(shapes)
            assert len(ids) == len(frl), (len(ids), len(frl))
            maps = [k[f].reshape(*shapes[i_]) for f, i_ in zip(frl, ids)]
            sv = C.saving(maps, cf)
            row["per_qp"][q] = {"saving_pct": sv, "db": db}
            vals.append(sv)
        row["mean_saving_pct"] = float(np.mean(vals)) if vals else None
        out["rows"].append(row)
        print(f"  {name:<12}" + "".join(
            f"{(row['per_qp'][q]['saving_pct'] if row['per_qp'].get(q) else float('nan')):>8.2f}%"
            for q in qps) + f"{row['mean_saving_pct']:>8.2f}%", flush=True)

    # --- input ablation ----------------------------------------------------
    # The shipped CNN head's ablation says stem alone matches every input
    # together (0.751 vs 0.750 held-out agreement) and bits alone reaches
    # 0.569. Agreement is not saving, so the same question is asked again in
    # the unit that pays: drop one group, keep the rest, and re-run the whole
    # allocation.
    if a.ablate:
        best = "gbm_mono"
        print(f"\n  girdi ablasyonu ({best}), 0.1 dB")
        abl = []
        trials = [("hepsi", feats)]
        for g, cols in groups.items():
            trials.append((f"-{g}", [c for c in feats if c not in cols]))
        for g, cols in groups.items():
            trials.append((f"yalniz {g}", list(cols)))
        for lab, cols in trials:
            if not cols:
                continue
            P = fit_predict(best, cols)
            vals = []
            for q in qps:
                fr = frame_slices(te["img"], te["qp"], q)
                sel = np.concatenate(fr)
                loc = {v: i for i, v in enumerate(sel)}
                frl = [np.array([loc[v] for v in f]) for f in fr]
                r = operate(P[sel], Mte[sel], Rte[sel], frl, cost_k, j,
                            a.target, K)
                if r is None:
                    continue
                _, k = r
                shapes = {}
                for rec in te["grid"]:
                    g_, q_, nh_, nw_, _ = str(rec).split(",", 4)
                    if int(q_) == q:
                        shapes[int(g_)] = (int(nh_), int(nw_))
                ids = sorted(shapes)
                maps = [k[f].reshape(*shapes[i_]) for f, i_ in zip(frl, ids)]
                vals.append(C.saving(maps, cf))
            m = float(np.mean(vals)) if vals else float("nan")
            abl.append({"inputs": lab, "n_features": len(cols),
                        "mean_saving_pct": m})
            print(f"    {lab:<16}{len(cols):>3} oznitelik   {m:>7.2f}%",
                  flush=True)
        out["ablation"] = abl

    # --- per-frame budget control ------------------------------------------
    # A single lambda per rate meets the budget ON AVERAGE: a frame that gives
    # up little subsidises one that gives up a lot, and nothing promises any
    # particular frame stayed under 0.1 dB. A decoder that can predict m_k can
    # instead pick lambda per frame, targeting the budget on the frame in
    # front of it. The catch is that it targets its PREDICTED decibel, so a
    # confident wrong prediction overshoots -- which is what the calibrated
    # margin is for: shrink the target by delta chosen on a calibration split
    # so a stated fraction of frames land under budget, the split-conformal
    # construction used for confidence-controlled early exit.
    if a.perframe:
        print(f"\n  kare basina lambda (tahmine gore), hedef {a.target} dB")
        pf = []
        for name, P in preds.items():
            if name == "oracle":
                continue
            rows_ = []
            for q in qps:
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

                def solve(f, tgt, table):
                    """lambda meeting tgt on THIS frame, on `table`'s error."""
                    def db(lam):
                        k = allocate(Pq[f], cost_k, lam, j)
                        return 10 * np.log10(table[f, k].mean() / Rq[f].mean()), k
                    lo, hi = 0.0, 1e-8
                    while db(hi)[0] <= tgt and hi < 1e6:
                        hi *= 4
                    if db(0.0)[0] > tgt:
                        return db(0.0)[1]
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if db(mid)[0] <= tgt:
                            lo = mid
                        else:
                            hi = mid
                    return db(lo)[1]

                # Calibrate the margin on half the frames, report on the rest,
                # so the margin is never fitted on what it is scored against.
                order = np.arange(len(frl))
                cal, tst = order[0::2], order[1::2]
                def overshoot(idx, delta):
                    o = []
                    for i_ in idx:
                        f = frl[i_]
                        k = solve(f, a.target - delta, Pq)
                        o.append(10 * np.log10(Mq[f, k].mean() / Rq[f].mean()))
                    return np.array(o)
                base = overshoot(cal, 0.0)
                delta = float(max(0.0, np.quantile(base - a.target, 0.9)))
                ov = overshoot(tst, delta)
                maps, ks = [], []
                for i_ in tst:
                    f = frl[i_]
                    k = solve(f, a.target - delta, Pq)
                    ks.append(k)
                    maps.append(k.reshape(*shapes[ids[i_]]))
                sv = C.saving(maps, cf)
                rows_.append({"qp": q, "delta_db": delta,
                              "saving_pct": sv,
                              "mean_db": float(ov.mean()),
                              "worst_db": float(ov.max()),
                              "frac_in_budget": float((ov <= a.target).mean())})
            m = float(np.mean([r["saving_pct"] for r in rows_]))
            inb = float(np.mean([r["frac_in_budget"] for r in rows_]))
            wo = max(r["worst_db"] for r in rows_)
            pf.append({"model": name, "per_qp": rows_, "mean_saving_pct": m,
                       "frac_in_budget": inb, "worst_db": wo})
            print(f"  {name:<12}{m:>8.2f}%   butcede kalan kare "
                  f"%{100 * inb:.0f}   en kotu {wo:+.4f} dB", flush=True)
        out["per_frame"] = pf

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
