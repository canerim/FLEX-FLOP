"""Enumerate the fixed router's scalar-control path on archived error tables.

This audits the monotonicity assumption behind distortion-based bisection.
It is a CPU table analysis, not a new mixed reconstruction or latency result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/data/refresh20260927"


def candidates(lp, cost, lo=-2e4, hi=2e4, extra=()):
    """All open score-order intervals, intersections and endpoints in [lo,hi].

    Score lines can change their argmax only at a pairwise intersection. We
    include exact float64 intersections and interval midpoints; extra archived
    controls preserve their precise tie behaviour. Identical maps are merged.
    """
    points = [lo, hi, *extra]
    for a in range(len(cost)):
        for b in range(a + 1, len(cost)):
            if cost[a] == cost[b]:
                continue
            cross = (lp[:, a] - lp[:, b]) / (cost[a] - cost[b])
            points.extend(cross[np.isfinite(cross) & (cross >= lo) & (cross <= hi)].tolist())
    edges = np.unique(np.array(points, dtype=np.float64))
    probes = np.unique(np.r_[edges, edges[:-1] + (edges[1:] - edges[:-1]) * .5])
    maps = (lp[None] - probes[:, None, None] * cost[None, None]).argmax(2)
    _, keep = np.unique(maps, axis=0, return_index=True)
    keep.sort()
    return probes[keep], maps[keep]


def synthetic_check():
    # The deepest model starts infeasible, a middle model is feasible, and the
    # cheapest is again infeasible. Rejecting immediately at beta=lo misses it.
    cost = np.array([1., 2., 3.])
    lp = np.array([[0., 2., 3.]])
    beta, maps = candidates(lp, cost, -5, 5)
    mse = np.array([[.06, .01, .04]])
    error = mse[np.arange(1)[None], maps].mean(1)
    assert error[0] > .02 and error[-1] > .02 and (error <= .02).any()
    assert (np.diff(cost[maps].mean(1)) <= 1e-12).all()
    # Fine-grid enumeration must be contained in the complete line-arrangement
    # candidates for this constructed case; it is an independent search check.
    dense = (lp[None] - np.linspace(-5, 5, 10001)[:, None, None] * cost[None, None]).argmax(2)
    assert set(map(tuple, dense)) <= set(map(tuple, maps))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    synthetic_check()
    t0 = time.perf_counter()
    dump_path = ROOT / "flexplus/results/router_dump_e15_ce_soft.pt"
    records_path = ROOT / "flexplus/results/null_models_uf_e15.json"
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    records = json.loads(records_path.read_text())
    prior = {(r["qp"], r["frame"]): r for r in records["per_frame"]}
    j = dump["j"]; cost = dump["cost"].double().numpy()[j:]
    assert (np.diff(cost) > 0).all()
    rows, nonmonotonic, transitions = [], [], []
    for qp, frames in dump["frames"].items():
        for fi, frame in enumerate(frames):
            lp = frame["lp"].double().numpy()
            M = frame["M"].double().numpy()[:, j:]
            R = frame["R"]
            old = prior[(int(qp), fi)]
            extra = [v["router"]["param"] for v in old["budgets"].values() if v["router"] is not None]
            beta, maps = candidates(lp, cost, extra=extra)
            error = M[np.arange(len(M))[None], maps].mean(1)
            db = 10 * np.log10(error / R)
            saving = 100 * (1 - cost[maps].mean(1))
            # Model cost is monotone under this logit penalty; source loss is
            # not assumed monotone, since it was not the score being optimised.
            assert (np.diff(saving) >= -1e-8).all()
            falls = np.flatnonzero(np.diff(db) < -1e-8)
            if len(falls):
                nonmonotonic.append(dict(qp=int(qp), frame=fi, sequence=dump["names"][fi],
                    n_decreases=len(falls), largest_decrease_db=float(-np.diff(db).min())))
            transitions.append(dict(qp=int(qp), frame=fi, sequence=dump["names"][fi],
                                    beta=beta.tolist(), db=db.tolist(), saving=saving.tolist()))
            for b in records["budgets"]:
                archived = old["budgets"][str(b)]["router"]
                if archived is not None:
                    k = (lp - archived["param"] * cost[None]).argmax(1)
                    reproduced_db = float(10 * np.log10(M[np.arange(len(M)), k].mean() / R))
                    reproduced_saving = float(100 * (1 - cost[k].mean()))
                    assert abs(reproduced_db - archived["db"]) < 1e-8
                    assert abs(reproduced_saving - archived["saving"]) < 1e-8
                valid = np.flatnonzero(db <= b + 1e-12)
                # Cost is the primary objective; use lower table loss to break
                # equal-cost ties. Record the actual map for later replay.
                best = min(valid, key=lambda i: (-saving[i], db[i])) if len(valid) else None
                rows.append(dict(qp=int(qp), frame=fi, sequence=dump["names"][fi], budget=b,
                    n_distinct_maps=len(maps), n_loss_decreases=len(falls),
                    archived_saving=None if archived is None else archived["saving"],
                    archived_db=None if archived is None else archived["db"],
                    enumerated_saving=None if best is None else float(saving[best]),
                    enumerated_db=None if best is None else float(db[best]),
                    enumerated_beta=None if best is None else float(beta[best]),
                    enumerated_map=None if best is None else (maps[best] + j).tolist(),
                    improvement_points=None if best is None or archived is None else float(saving[best] - archived["saving"])))
    summary = []
    for b in records["budgets"]:
        rr = [r for r in rows if r["budget"] == b]
        paired = [r for r in rr if r["improvement_points"] is not None]
        gains = [r["improvement_points"] for r in paired]
        assert min(gains, default=0) >= -1e-8
        summary.append(dict(budget=b, n=len(rr), paired_n=len(paired),
            improved=sum(v > 1e-8 for v in gains), mean_gain_points=float(np.mean(gains)), max_gain_points=max(gains, default=0),
            recovered_feasible=sum(r["archived_saving"] is None and r["enumerated_saving"] is not None for r in rr)))
    result = dict(method="Enumerate score-line intersections and interval representatives in beta [-20000,20000], including archived controls",
        scope="Source-error table surrogate and fixed router score family only; not final cropped-image quality, global allocation optimum, or GPU speed",
        checks="Synthetic nonmonotone example, fine-grid inclusion, all archived feasible controls reproduced, cost monotonicity, no loss of feasible saving",
        n_frame_qp=len(transitions), nonmonotonic_frame_qp=len(nonmonotonic),
        nonmonotonic_cases=nonmonotonic, summary=summary, rows=rows, paths=transitions,
        analysis_wall_seconds=time.perf_counter() - t0,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [dump_path, records_path, Path(__file__)]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "router_control_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(nonmonotonic_frame_qp=len(nonmonotonic), total=len(transitions), summary=summary), indent=2))


if __name__ == "__main__":
    main()
