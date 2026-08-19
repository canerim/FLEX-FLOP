"""What the Lagrangian sweep can and cannot reach.

Shoham and Gersho (1988) settled the structure of this allocation: sweeping the
multiplier over a finite set of per-unit operating points traces exactly the
LOWER CONVEX HULL of the achievable (cost, distortion) set, and no lambda
reaches an interior point. Every budget-targeting bisection in this project
inherits that, and the consequence has never been measured here: when the budget
falls between two hull vertices, the bisection cannot land on it. It lands on the
vertex below, and the compute between the two vertices is left on the table.

This measures the loss. For one frame it enumerates the exact Pareto set by
dynamic programming over tiles -- feasible because the cost alphabet is tiny --
and compares:

  hull      what the Lagrangian actually reaches at each budget
  pareto    the best allocation at that budget, hull or not
  gap       what the convexity of the sweep costs

No GPU: the per-tile table is already measured (dump_tile_table.py).
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
d = json.load(open(R / "results/tile_table.json"))
D = np.array(d["D"])                      # [T, K] MSE
cost = np.array(d["cost"])
j, K = d["j"], d["K"]
T = D.shape[0]
D = D[:, j:]
cost = cost[j:]
ref = d["ref_mse"]
NK = D.shape[1]

# ---- the Lagrangian sweep: every allocation any lambda can produce ----------
lams = np.concatenate([[0.0], np.geomspace(1e-10, 1e-1, 6000)])
seen, hull = set(), []
for lam in lams:
    k = np.argmin(D + lam * cost[None, :], axis=1)
    key = k.tobytes()
    if key in seen:
        continue
    seen.add(key)
    hull.append((cost[k].mean(), D[np.arange(T), k].mean(), lam))
hull.sort()

# ---- the exact Pareto set, by DP over tiles --------------------------------
# Cost is a sum of per-tile costs from a 4-symbol alphabet, so the reachable
# total costs are a small lattice; for each we keep the least distortion. Exact,
# not sampled.
SCALE = 10000
cq = np.round(cost * SCALE).astype(int)
best = {0: 0.0}
for t in range(T):
    nxt = {}
    for c0, d0 in best.items():
        for a in range(NK):
            c1, d1 = c0 + cq[a], d0 + D[t, a]
            if c1 not in nxt or d1 < nxt[c1]:
                nxt[c1] = d1
    # prune dominated states: a higher cost with no lower distortion is useless
    keep, bestd = {}, float("inf")
    for c1 in sorted(nxt):
        if nxt[c1] < bestd - 1e-18:
            keep[c1] = nxt[c1]; bestd = nxt[c1]
    best = keep
pareto = sorted((c / SCALE / T, dd / T) for c, dd in best.items())

db = lambda m: 10 * np.log10(m / ref)
print(f"  {d['seq'][:24]} q{d['qp']}, {T} tiles, {NK} reachable exits")
print(f"  lambda sweep reaches {len(hull)} distinct allocations")
print(f"  exact Pareto set has {len(pareto)} points\n")

rows = []
print(f"  {'budget dB':>10}{'hull save':>11}{'pareto save':>13}{'gap pts':>9}")
for B in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
    h = [(c, m) for c, m, _l in hull if db(m) <= B]
    p = [(c, m) for c, m in pareto if db(m) <= B]
    if not h or not p:
        print(f"  {B:>10.2f}{'infeasible':>11}")
        continue
    hs = 100 * (1 - min(c for c, _ in h))
    ps = 100 * (1 - min(c for c, _ in p))
    rows.append({"budget_db": B, "hull_saving": hs, "pareto_saving": ps,
                 "gap_pts": ps - hs})
    print(f"  {B:>10.2f}{hs:>10.2f}%{ps:>12.2f}%{ps-hs:>9.2f}")

json.dump({"seq": d["seq"], "qp": d["qp"], "n_tiles": T,
           "n_hull_allocations": len(hull), "n_pareto": len(pareto),
           "note": "hull = reachable by some lambda; pareto = best at that "
                   "budget whether or not any lambda reaches it",
           "rows": rows}, open(R / "results/hull_gap.json", "w"), indent=2)
print(f"\n  -> results/hull_gap.json")
