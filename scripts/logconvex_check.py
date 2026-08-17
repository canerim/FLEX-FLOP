"""Is dB(S) convex, and if so why? Derive the exact condition, then test it.

The theorems give D convex in C. The frontier is nearly always PLOTTED as
decibels against percentage saving, and convexity there is a different claim,
because dB is a logarithm of D and the logarithm of a convex function need be
neither convex nor concave. Writing it out,

    dB = (10/ln 10) ln D        (ln D)'' = D''/D - (D'/D)^2

so

    dB convex  <=>  D D'' >= (D')^2  <=>  D is LOG-CONVEX in C.

Along the Lagrangian frontier D'(C) = -lambda(C), so the condition reads
-dlambda/dC * D >= lambda^2, equivalently d(1/lambda)/dC >= 1/D. In words:
distortion must fall at least exponentially in compute -- diminishing dB returns
to compute. That is a property of the decoder, not a theorem, so it is measured
here rather than assumed.
"""
import json
import numpy as np

rows = json.load(open("results/paper_curve_grid128.json"))["rows"]
print("  dB(S) konveks  <=>  D log-konveks  <=>  (ln D)'' >= 0\n")
print(f"  {'qp':>4}{'nokta':>8}{'kosul saglanan':>17}{'en kotu deger':>16}")
out = {}
for qp in (0, 16, 32, 48, 63):
    pts = sorted({(r["saving_pct"], r["db_vs_uf"]) for r in rows if r["qp"] == qp})
    S = np.array([p[0] for p in pts]) / 100.0
    dB = np.array([p[1] for p in pts])
    keep = (S > 0.02) & (S < 0.44)
    S, dB = S[keep], dB[keep]
    if len(S) < 6:
        print(f"  {qp:>4}  yetersiz nokta")
        continue
    lnD = np.log(10 ** (dB / 10.0))
    # Degree-4 fit then differentiate twice: the frontier is sampled at 20-odd
    # lambdas, so finite differences on raw samples would measure the sampling
    # rather than the curve.
    d2 = np.poly1d(np.polyfit(S, lnD, 4)).deriv(2)
    xs = np.linspace(S.min(), S.max(), 300)
    v = d2(xs)
    frac, worst = 100 * float(np.mean(v >= 0)), float(v.min())
    out[qp] = {"frac_ok": frac, "worst": worst, "n": int(len(S))}
    print(f"  {qp:>4}{len(S):>8}{frac:>16.1f}%{worst:>16.4f}")
json.dump(out, open("results/logconvexity.json", "w"), indent=2)
print("\n  Sutun: frontier boyunca kosulun saglandigi noktalarin yuzdesi.")
