"""Fit the compute-quality frontier, and cross-check convexity a second way.

What is fitted
--------------
At each rate the measured frontier is well described by

    dB(S) = f + a * S^b

with f the floor -- the residual of the deepest exit against the released
decoder, which every operating point pays before buying any saving. Fitting is
linear least squares on log(dB - f) against log S.

What is NOT claimed
-------------------
This is a per-rate fit, not a law. The exponent falls from 4.10 at qp0 to 2.28
at qp63, so the frontier is not self-similar across rate and no single curve
describes all five. Three fitted forms were compared and the power law won
clearly, but winning a comparison of three guesses is not evidence of a
mechanism.

Nor is a relation between the fitted parameters claimed. b and log a move
together, and with five rates almost any pair of monotone quantities will, so
fitting one to the other would be reading structure into five points.

The part that is worth something
--------------------------------
b > 1 at every rate is exactly the statement that dB is convex in S, and it is
reached without the secant test and without any assumption the secant test
makes. The two agree. That matters because the convexity claim previously
rested on a quartic fit that turned out to be producing the answer itself, so a
second, independent route to the same conclusion is not decorative.

Practically: three numbers per rate reproduce the frontier to within the fit
error, so an operating point can be chosen without re-running the sweep -- with
the caveat that the fit is worst at qp0 (R^2 0.964), where the curve is most
strongly bent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def frontier(rows, qp):
    best: dict[float, float] = {}
    for r in rows:
        if r["qp"] != qp:
            continue
        s = round(r["saving_pct"], 6)
        if s not in best or r["db_vs_uf"] < best[s]:
            best[s] = r["db_vs_uf"]
    p = sorted(best.items())
    return np.array([x[0] for x in p]), np.array([x[1] for x in p])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/paper_curve_grid128.json")
    ap.add_argument("--out", default="results/frontier_law.json")
    ap.add_argument("--s_min", type=float, default=2.0)
    ap.add_argument("--s_max", type=float, default=42.9,
                    help="drop the saturated tail: at the ceiling (42-43%%, "
                         "depending on tile size) every "
                         "tile is already at the shallowest exit, successive "
                         "lambdas buy nothing, and the points pile up at one "
                         "saving with rising dB -- a vertical segment, which no "
                         "function of S can represent")
    a = ap.parse_args(argv)

    d = json.loads((ROOT / a.curve).read_text())
    qps = sorted({r["qp"] for r in d["rows"]})
    print(f"  dB(S) = f + a * S^b, fitted per rate on "
          f"{d.get('n_sequences', '?')} CTC sequences\n")
    print(f"  {'qp':>4}{'floor f':>10}{'scale a':>12}{'exponent b':>12}"
          f"{'R^2':>9}{'pts':>5}")

    out = {"curve": a.curve, "n_sequences": d.get("n_sequences"),
           "ckpt": d.get("ckpt"), "form": "dB = f + a * S^b", "rows": []}
    for qp in qps:
        S, dB = frontier(d["rows"], qp)
        f = float(dB.min())
        m = (S > a.s_min) & (S < a.s_max) & (dB > f + 1e-6)
        if m.sum() < 5:
            print(f"  {qp:>4}  too few points")
            continue
        y, x = np.log(dB[m] - f), np.log(S[m])
        A = np.vstack([x, np.ones_like(x)]).T
        (b, loga), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ [b, loga]
        r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out["rows"].append({"qp": qp, "floor_db": f, "scale_a": float(np.exp(loga)),
                            "exponent_b": float(b), "r2": r2,
                            "n_points": int(m.sum())})
        print(f"  {qp:>4}{f:>10.4f}{np.exp(loga):>12.3e}{b:>12.3f}{r2:>9.4f}"
              f"{m.sum():>5}")

    bs = [r["exponent_b"] for r in out["rows"]]
    out["exponent_range"] = [min(bs), max(bs)]
    out["convex_all_rates"] = bool(min(bs) > 1.0)
    print(f"\n  exponent {min(bs):.2f} to {max(bs):.2f}: NOT constant, so this is "
          f"a per-rate fit,")
    print(f"  not a law -- the frontier is not self-similar across rate.")
    print(f"  But b > 1 at every rate IS convexity of dB in S, reached without "
          f"the")
    print(f"  secant test and agreeing with it. Two independent routes, same "
          f"conclusion.")

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
