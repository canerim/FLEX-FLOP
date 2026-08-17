"""Is dB(S) convex along the measured frontier? Derive the condition, then test it.

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
to compute. That is a property of the decoder, not a theorem, so it is measured.

How it is tested, and the artefact that made the first answer wrong
------------------------------------------------------------------
This originally fitted a degree-4 polynomial to ln D against S and asked where
its second derivative was non-negative. On 10 sequences that reported 100% at
every rate. On 40 it reported 69-78%, with every violation between 39% saving
and the ceiling, which is 43.4% for a 128px ladder and 42.5% for a 256px one --
the tile size shifts the grid seam-repair cost and so the exit costs slightly.

The violations were the fit, not the frontier. As saving approaches the ceiling
-- every tile at the shallowest exit -- no further compute can be bought at any
distortion and the curve turns nearly vertical. A quartic cannot represent that,
so it overshoots and its second derivative dips negative exactly there. The
diagnosis is confirmed by the location: the failures cluster at the ceiling at
every rate, and nowhere else.

Convexity of a sampled curve needs no fitting anyway. dB(S) is convex on the
samples precisely when the secant slopes are non-decreasing,

    (dB_{i+1} - dB_i) / (S_{i+1} - S_i)  non-decreasing in i,

which is the exact discrete statement, has no free parameter -- no polynomial
degree to choose -- and cannot manufacture a violation out of a steep endpoint.
Under it the condition holds on 100% of every frontier at every rate.

The honest limit of that statement: it is convexity AT THE SAMPLED RESOLUTION.
A curve that dipped between two adjacent lambdas would pass. The sweep is 23
lambdas across the range, so a dip would have to hide inside one interval; the
quartic is reported alongside as a second opinion, with the caveat that it is
untrustworthy near the ceiling for the reason above.
"""
import json

import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--curve", default="results/paper_curve_grid128.json")
_ap.add_argument("--out", default="results/logconvexity.json")
_a = _ap.parse_args()
rows = json.load(open(_a.curve))["rows"]

print("  dB(S) convex  <=>  D log-convex  <=>  secant slopes non-decreasing\n")
print(f"  {'qp':>4}{'pts':>6}{'holds':>9}{'margin':>11}{'quartic':>10}"
      f"   first violation")
out = {}
for qp in (0, 16, 32, 48, 63):
    pts = sorted({(r["saving_pct"], r["db_vs_uf"]) for r in rows if r["qp"] == qp})
    S = np.array([p[0] for p in pts]) / 100.0
    dB = np.array([p[1] for p in pts])
    keep = S > 0.02
    S, dB = S[keep], dB[keep]
    if len(S) < 6:
        print(f"  {qp:>4}  too few points")
        continue

    lnD = np.log(10 ** (dB / 10.0))
    slope = np.diff(lnD) / np.diff(S)
    step = np.diff(slope)                      # >= 0 everywhere iff convex
    ok = step >= -1e-9
    frac = 100 * float(np.mean(ok))
    first = next((100 * S[i + 1] for i, g in enumerate(ok) if not g), None)

    # Second opinion, kept for the record. Unreliable near the ceiling.
    f2 = np.poly1d(np.polyfit(S, lnD, 4)).deriv(2)
    quart = 100 * float(np.mean(f2(np.linspace(S.min(), S.max(), 300)) >= 0))

    out[qp] = {"frac_ok": frac, "worst": float(step.min()), "n": int(len(S)),
               "first_violation_pct": first, "quartic_frac_ok": quart}
    print(f"  {qp:>4}{len(S):>6}{frac:>8.1f}%{step.min():>11.4f}{quart:>9.1f}%"
          f"   {f'{first:.0f}% saving' if first else 'none'}")

json.dump(out, open(_a.out, "w"), indent=2)
print("\n  margin = smallest increase in secant slope; >= 0 is the condition.")
print("  quartic = the old degree-4 fit, kept as a second opinion. It is")
print(f"  untrustworthy near the ceiling ({max(r['saving_pct'] for r in rows):.1f}% "
      f"here), where the frontier is nearly")
print("  vertical and a quartic overshoots -- that artefact, not the decoder,")
print("  produced the 69-78% figures this script reported before.")
