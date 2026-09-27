"""Separate the exit histogram from its spatial assignment on frozen tables.

All computation is on CPU. The expectation over uniformly permuted assignments
is exact; it is not a new reconstruction or a quality guarantee. Every shuffle
preserves each frame's exit histogram and therefore the recorded MAC model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/data/refresh20260927"
RULES = ["oracle", "router", "dither", "uniform"]


def expected_error(M, k):
    # Under a uniform permutation, every location has the same marginal exit
    # probability. Linearity of expectation avoids Monte Carlo error here.
    p = np.bincount(k, minlength=M.shape[1]) / len(k)
    return float((M * p[None]).sum(1).mean())


def check_expectation():
    import itertools
    M = np.array([[1., 8., 3.], [4., 2., 1.], [5., 3., 4.]])
    k = np.array([0, 1, 1])
    permutations = list(itertools.permutations(k))
    brute = np.mean([M[np.arange(3), p].mean() for p in permutations])
    assert abs(expected_error(M, k) - brute) < 1e-12
    for depth in range(3):
        assert abs(expected_error(M, np.full(3, depth)) - M[:, depth].mean()) < 1e-12


def clustered_ci(rows, seed):
    names = sorted({r["sequence"] for r in rows})
    groups = [np.array([r["assignment_gain_db"] for r in rows if r["sequence"] == n]) for n in names]
    totals = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    ix = np.random.default_rng(seed).integers(0, len(groups), size=(5000, len(groups)))
    samples = totals[ix].sum(1) / sizes[ix].sum(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return float(lo), float(hi)


def main():
    torch.set_num_threads(2)
    check_expectation()
    dump_path = ROOT / "flexplus/results/router_dump_e15_ce_soft.pt"
    maps_path = ROOT / "flexplus/results/eval_rules_ctc_e15.json"
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    raw = json.loads(maps_path.read_text())
    lookup = {(int(qp), i): f for qp, frames in dump["frames"].items() for i, f in enumerate(frames)}
    rows = []
    for frame in raw["rows"]:
        source = lookup[(frame["qp"], frame["frame"])]
        M = source["M"].double().numpy()
        cost = np.array(raw["cost"])
        assert frame["seq"] == dump["names"][frame["frame"]]
        for b in raw["budgets"]:
            if not all(frame["rules"][rule][b] is not None for rule in RULES):
                continue
            for rule in RULES:
                value = frame["rules"][rule][b]
                k = np.array(value["map"], dtype=int)
                actual = float(M[np.arange(len(k)), k].mean())
                expected = expected_error(M, k)
                gain = float(10 * np.log10(expected / actual))
                # Duplicating one column for each assigned exit turns the
                # fixed-histogram optimum into a square assignment problem.
                # This is an exact table-space bound, not a free deployment
                # baseline: obtaining M still requires candidate decodes.
                ii, jj = linear_sum_assignment(M[:, k])
                optimum = float(M[ii, k[jj]].mean())
                assert optimum <= actual + 1e-12
                if rule == "oracle":
                    # A separable Lagrangian minimiser is also optimal when
                    # its histogram (and therefore total cost) is fixed.
                    assert abs(optimum - actual) < 1e-10
                saving = float(100 * (1 - cost[k].mean()))
                assert abs(saving - value["saving"]) < 1e-9
                if len(np.unique(k)) == 1:
                    assert abs(gain) < 1e-10
                rows.append(dict(sequence=frame["seq"], qp=frame["qp"], frame=frame["frame"],
                    budget=float(b), rule=rule, n_tiles=len(k), distinct_exits=len(np.unique(k)),
                    table_mse=actual, permuted_expected_mse=expected, assignment_gain_db=gain,
                    fixed_histogram_optimal_mse=optimum,
                    fixed_histogram_optimal_gain_db=float(10*np.log10(expected/optimum)),
                    assignment_regret_db=float(10*np.log10(actual/optimum)),
                    mac_saving=saving, histogram=value["hist"]))
    summary = []
    for b in map(float, raw["budgets"]):
        for rule in RULES:
            rr = [r for r in rows if r["budget"] == b and r["rule"] == rule]
            gains = np.array([r["assignment_gain_db"] for r in rr])
            lo, hi = clustered_ci(rr, 20260927)
            summary.append(dict(budget=b, rule=rule, n=len(rr), n_sequences=len({r["sequence"] for r in rr}),
                mean=float(gains.mean()), median=float(np.median(gains)), lo=lo, hi=hi,
                optimal_gain_mean=float(np.mean([r["fixed_histogram_optimal_gain_db"] for r in rr])),
                assignment_regret_mean=float(np.mean([r["assignment_regret_db"] for r in rr])),
                positive=int((gains > 1e-8).sum()), negative=int((gains < -1e-8).sum()),
                uniform_histogram=int(sum(r["distinct_exits"] == 1 for r in rr))))
    result = dict(
        estimand="10 log10(E_permutation[mean tile MSE] / recorded mean tile MSE), then averaged over paired frame-QP cases",
        interpretation="Positive values indicate better table distortion than a uniformly permuted assignment with exactly the same exit histogram and modelled MAC cost.",
        scope="Frozen padded source-MSE tables only; no mixed reconstruction rerun, no uncertainty over training seeds, and no runtime evidence. The logarithm of expected MSE is not expected PSNR.",
        dependence="The archived histogram and control were chosen using each test frame's source table. This is an association diagnostic, not validation of an autonomous router or causal effect of training.",
        optimum="Fixed-histogram source-table optimum by linear_sum_assignment over replicated exit columns; it requires the full M table and is not a decoder-available policy.",
        bootstrap="5000 sequence-cluster draws, seed 20260927; all available QPs stay together.",
        checks="Exact expectation agrees with exhaustive permutations on a small independent example; constant maps have zero gain; recorded MAC savings reproduced; optimum never worse; every source-search map is optimal at its own fixed histogram.",
        rows=rows, summary=summary,
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [dump_path, maps_path, Path(__file__)]})
    (OUT / "spatial_assignment_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([r for r in summary if r["budget"] in [.1, .3]], indent=2))


if __name__ == "__main__":
    main()
