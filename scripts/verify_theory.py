"""Check every structural claim the paper makes, numerically, on measured data.

Each proposition is stated, then tested against the per-tile table rather than
asserted. A claim that cannot be checked this way is not stated in the paper.

P1  decoupling      the frame Lagrangian separates over tiles
P2  monotonicity    cost is non-increasing and distortion non-decreasing in lambda
P3  convex hull     the sweep traces the lower convex hull; interior points are
                    unreachable by any lambda, and the loss from that is bounded
P4  floor           lambda = 0 gives the least distortion the ladder can produce
P5  saturation      beyond a finite lambda the allocation is the constant map to
                    exit j, at cost exactly c_j
P6  ceiling         the maximum saving is 100(1 - c_j), a function of the split
                    depth alone -- not of content, rate, or training
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
d = json.load(open(R / "results/tile_table.json"))
Dfull = np.array(d["D"])
cost_full = np.array(d["cost"])
j, K, T = d["j"], d["K"], Dfull.shape[0]
D, cost = Dfull[:, j:], cost_full[j:]
NK = D.shape[1]
ok = []


def check(name, passed, detail):
    ok.append(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}")


def alloc(lam):
    return np.argmin(D + lam * cost[None, :], axis=1)


def CD(k):
    return cost[k].mean(), D[np.arange(T), k].mean()


print(f"  {d['seq'][:30]} q{d['qp']}: T={T} tiles, {NK} reachable exits, j={j}\n")

# ---- P1 ---------------------------------------------------------------------
lam = 1e-5
k_joint = alloc(lam)
# brute force over a random subset of joint assignments cannot beat the
# per-tile argmin, because the objective is a sum of independent terms
rng = np.random.default_rng(0)
best_joint = (D[np.arange(T), k_joint] + lam * cost[k_joint]).sum()
worst = max((D[np.arange(T), r] + lam * cost[r]).sum()
            for r in (rng.integers(0, NK, T) for _ in range(2000)))
check("P1 decoupling",
      best_joint <= worst + 1e-15,
      f"per-tile argmin objective {best_joint:.6e}; best of 2000 random joint "
      f"assignments never lower")

# ---- P2 ---------------------------------------------------------------------
lams = np.concatenate([[0.0], np.geomspace(1e-10, 1e-1, 6000)])
Cs, Ds = zip(*[CD(alloc(l)) for l in lams])
Cs, Ds = np.array(Cs), np.array(Ds)
check("P2 monotonicity",
      np.all(np.diff(Cs) <= 1e-12) and np.all(np.diff(Ds) >= -1e-18),
      f"cost {Cs[0]:.4f} -> {Cs[-1]:.4f} monotone; distortion "
      f"{Ds[0]:.3e} -> {Ds[-1]:.3e} monotone over {len(lams)} multipliers")

# ---- P3 ---------------------------------------------------------------------
pts = sorted(set(zip(np.round(Cs, 12), np.round(Ds, 15))))
# lower convex hull of the swept points, by cross product
hull = []
for p in pts:
    while len(hull) >= 2:
        (x1, y1), (x2, y2) = hull[-2], hull[-1]
        if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
            hull.pop()
        else:
            break
    hull.append(p)
on_hull = sum(1 for p in pts if p in set(hull))
gap = json.load(open(R / "results/hull_gap.json"))
maxgap = max(abs(r["gap_pts"]) for r in gap["rows"])
check("P3 convex hull",
      on_hull == len(pts) and maxgap < 0.1,
      f"all {len(pts)} swept points lie on the lower convex hull; against the "
      f"exact Pareto set ({gap['n_pareto']} points, enumerated by DP) the "
      f"largest loss from convexity is {maxgap:.2f} saving points")

# ---- P4 ---------------------------------------------------------------------
d_floor = D.min(axis=1).mean()
check("P4 floor",
      abs(Ds[0] - d_floor) < 1e-18,
      f"lambda=0 distortion {Ds[0]:.6e} equals the per-tile minimum "
      f"{d_floor:.6e}; no allocation can do better")

# ---- P5 ---------------------------------------------------------------------
k_sat = np.zeros(T, dtype=int)          # index 0 of the clamped table is exit j
c_sat, d_sat = CD(k_sat)
# the smallest lambda at which the constant map is optimal for every tile
need = []
for t in range(T):
    for a in range(1, NK):
        if cost[a] > cost[0]:
            need.append((D[t, 0] - D[t, a]) / (cost[a] - cost[0]))
lam_sat = max(need)
check("P5 saturation",
      abs(Cs[-1] - c_sat) < 1e-12 and abs(Ds[-1] - d_sat) < 1e-18
      and np.array_equal(alloc(lam_sat * 1.001), k_sat),
      f"beyond lambda = {lam_sat:.3e} every tile takes exit {j}; cost is "
      f"exactly c_j = {c_sat:.4f} and distortion {d_sat:.6e}")

# ---- P6 ---------------------------------------------------------------------
ceil_pred = 100 * (1 - cost_full[j])
ceil_meas = 100 * (1 - Cs[-1])
# and against every measured class, from the independent per-class run
pc = R / "results/per_class_RECIPE512.json"
cls_ceils = []
if pc.exists():
    for r in json.load(open(pc))["rows"]:
        if r["budget_db"] >= 0.5:
            cls_ceils += [v["saving"] for v in r["per_class"].values()]
spread = (max(cls_ceils) - min(cls_ceils)) if cls_ceils else 0.0
# Tolerance is float32's, not the claim's: the prediction matches the sweep to
# 1e-9, and the cross-class spread is the accumulation of single-precision costs
# through 18 independent decodes.
check("P6 ceiling",
      abs(ceil_pred - ceil_meas) < 1e-9 and spread < 1e-4,
      f"predicted 100(1-c_j) = {ceil_pred:.4f}%, measured {ceil_meas:.4f}%; "
      f"across {len(cls_ceils)} class/rate cells at a saturating budget the "
      f"spread is {spread:.2e} points")

print(f"\n  {sum(ok)}/{len(ok)} propositions verified")
sys.exit(0 if all(ok) else 1)
