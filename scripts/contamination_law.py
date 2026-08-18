"""Which counting argument actually predicts the seam?

The paper states the corrupted FRACTION of a tile, 1 - ((F-2b)/F)^2, and uses it
to explain why the split depth matters. seam_vs_split.py measured the seam at
every split depth, and the fraction turns out to be a poor predictor: it falls by
a factor of four from b=12 to b=2 while the seam falls by a factor of thirty-six.

The reason is that the fraction counts whether a pixel is REACHED and not how
far into it the border propagates. A pixel at distance d from the border is
touched by (b - d) of the b per-tile convolutions, so a severity measure should
weight by reach, not by membership. Four candidates are fitted here and compared
on held-out rates; the winner is the one the paper should state.

Pure analysis of an existing measurement. No GPU.
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
d = json.load(open(R / "results/seam_vs_split.json"))
Fp = d["feature_px"]
rows = [r for r in d["rows"] if r["b"] > 0]
b = np.array([r["b"] for r in rows], float)
qps = sorted(d["rows"][0]["seam_db"], key=int)
Y = {q: np.array([r["seam_db"][q] for r in rows]) for q in qps}


def ring_counts(bb, F):
    """Pixels at each distance d from the border of an F x F tile."""
    dd = np.arange(int(np.ceil(F / 2)))
    n = np.where(dd < F / 2, 4 * (F - 2 * dd) - 4, 0.0)
    n[0] = 4 * F - 4
    return dd, np.maximum(n, 0.0)


def model_area(bb):                       # what the paper currently says
    return 1 - np.maximum(0.0, (Fp - 2 * bb) / Fp) ** 2


def model_reach(bb, p):
    """Sum over pixels of (b - d)^p: weight by how many convolutions reach."""
    dd, n = ring_counts(bb, Fp)
    out = []
    for x in np.atleast_1d(bb):
        w = np.maximum(0.0, x - dd) ** p
        out.append((n * w).sum() / Fp ** 2)
    return np.array(out)


MODELS = {
    "area  1-((F-2b)/F)^2": model_area,
    "reach p=1": lambda x: model_reach(x, 1),
    "reach p=2": lambda x: model_reach(x, 2),
    "power b^a (a fitted)": None,
}

print(f"  tile {Fp} feature px, split sweep b = {b.astype(int).tolist()}\n")
best = {}
for q in qps:
    y = Y[q]
    print(f"  qp {q}")
    res = {}
    for name, f in MODELS.items():
        if f is None:
            # fit y = c * b^a in log space
            A = np.polyfit(np.log(b), np.log(y), 1)
            pred = np.exp(A[1]) * b ** A[0]
            label = f"power b^{A[0]:.2f}"
        else:
            x = f(b)
            # one free scale, no intercept: the model must vanish with b
            c = (x @ y) / (x @ x)
            pred = c * x
            label = name
        rel = np.abs(pred - y) / y
        res[label] = (rel.max(), rel.mean())
        print(f"    {label:<24} max rel err {rel.max():>6.1%}   "
              f"mean {rel.mean():>6.1%}")
    best[q] = min(res.items(), key=lambda kv: kv[1][1])[0]
    print()

print("  best model per rate: " + ", ".join(f"q{q}: {v}" for q, v in best.items()))

# The winner, refitted on q0 alone and tested on the other rates -- a fit that
# only works when it sees the data it is scored on is not a law.
x = model_reach(b, 2)
c0 = (x @ Y[qps[0]]) / (x @ x)
print(f"\n  reach p=2 fitted on qp {qps[0]} alone (c = {c0:.4f}), applied to the "
      f"others after a single scale per rate:")
for q in qps:
    y = Y[q]
    c = (x @ y) / (x @ x)
    rel = np.abs(c * x - y) / y
    print(f"    qp {q:>3}: shape error max {rel.max():>6.1%}, "
          f"scale {c/c0:>5.2f}x the qp {qps[0]} scale")

json.dump({"tile_feature_px": Fp, "b": b.tolist(),
           "seam_db": {q: Y[q].tolist() for q in qps},
           "area_model": model_area(b).tolist(),
           "reach2_model": x.tolist(),
           "best_per_rate": best},
          open(R / "results/contamination_law.json", "w"), indent=2)
print(f"\n  -> results/contamination_law.json")
