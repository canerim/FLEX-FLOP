"""The frontier's exact form, derived and then tested against the measurement.

The claim
---------
Let the ladder have exits k = j..K-1 with measured per-exit distortion D_k(qp)
and relative cost C_k. Assign a fraction p_k of tiles to exit k. Then BOTH the
cost and the distortion are LINEAR in p:

    C(p) = sum_k p_k C_k          D(p) = sum_k p_k D_k

because cost is additive over tiles and MSE is a mean over tiles. So the set of
achievable (cost, distortion) pairs is exactly the CONVEX HULL of the K points
(C_k, D_k), and the efficient frontier is its lower-left hull. Two consequences
follow immediately and neither needs an experiment:

  1. Between two adjacent hull vertices the frontier is a straight line in
     (cost, MSE) -- mixing, not curvature. It looks curved in (saving, dB) only
     because dB is a logarithm of MSE.
  2. At the optimum with multiplier lambda, the active vertices satisfy
     lambda = -(D_k - D_k') / (C_k - C_k'). The multiplier IS the slope, which is
     why sweeping lambda traces the hull and nothing else.

So a uniform-mixing frontier is fully determined by 2K numbers per QP. This
script predicts it from the measured D_k, C_k and overlays the ACTUAL Lagrangian
frontier measured tile-by-tile.

The gap between them is not error. It is the value of content adaptivity: the
prediction mixes tiles blindly at fixed proportions, while the real oracle picks
per tile, and by Jensen it can only do better. Measuring that gap is measuring
what routing is worth -- which is the project's whole premise.
"""
import json, sys
from pathlib import Path
import numpy as np

W = json.load(open("results/why_qp.json"))
C = np.array(W["cost"])
J, K = 2, len(C)

def hull(D, C):
    """Lower-left convex hull of (C_k, D_k) -- the achievable efficient set."""
    pts = sorted(zip(C[J:], D[J:]))
    h = []
    for c, d in pts:
        while len(h) >= 2:
            (c1, d1), (c2, d2) = h[-2], h[-1]
            # drop h[-1] if it sits on or above the line h[-2] -> (c,d)
            if (d2 - d1) * (c - c1) >= (d - d1) * (c2 - c1):
                h.pop()
            else:
                break
        h.append((c, d))
    return h

def predict(D, C, savings):
    """dB below released UF at each target saving, by mixing along the hull."""
    h = hull(D, C)
    cs = np.array([p[0] for p in h]); ds = np.array([p[1] for p in h])
    out = []
    for s in savings:
        c = (1 - s / 100) * C[-1]
        if c < cs[0] or c > cs[-1]:
            out.append(np.nan); continue
        i = np.searchsorted(cs, c).clip(1, len(cs) - 1)
        t = (c - cs[i - 1]) / (cs[i] - cs[i - 1])
        out.append(10 * np.log10(ds[i - 1] + t * (ds[i] - ds[i - 1])))
    return np.array(out)

print("  TEORI: frontier, (maliyet, MSE) duzleminde K noktanin alt konveks zarfi.\n")
print(f"  {'qp':>4}{'cikis 2':>10}{'cikis 3':>10}{'cikis 4':>10}{'cikis 5':>10}   zarf koseleri")
for r in W["rows"]:
    D = 10 ** (np.array(r["db_per_exit"]) / 10)      # dB -> relative MSE
    h = hull(D, C)
    verts = [k for k in range(J, K) if any(abs(C[k] - c) < 1e-9 for c, _ in h)]
    print(f"  {r['qp']:>4}" + "".join(f"{r['db_per_exit'][k]:>10.4f}" for k in range(J, K))
          + f"   {verts}")

# Test against the measured per-tile Lagrangian frontier.
meas = json.load(open("results/paper_curve_grid128.json"))["rows"]
print("\n  TEST: tekduze karisim TAHMINI vs olculen tile-basina ORACLE\n")
print(f"  {'qp':>4}{'tasarruf':>10}{'tahmin dB':>12}{'olculen dB':>12}{'oracle kazanci':>16}")
for r in W["rows"]:
    qp = r["qp"]
    D = 10 ** (np.array(r["db_per_exit"]) / 10)
    for s in (15, 20, 30):
        pred = predict(D, C, [s])[0]
        cand = [m for m in meas if m["qp"] == qp and m["saving_pct"] >= s]
        if not cand or np.isnan(pred):
            continue
        m = min(cand, key=lambda z: z["saving_pct"])
        print(f"  {qp:>4}{s:>9}%{pred:>12.4f}{m['db_vs_uf']:>12.4f}"
              f"{pred - m['db_vs_uf']:>+15.4f} dB")
print("\n  Pozitif kazanc = tile-basina secim, kor karisimdan iyi (Jensen). Yonlendirmenin degeri budur.")
