"""Audit the blind baseline's mixture search on archived CPU error tables.

Enumerating the finite Bayer thresholds avoids assuming that independently
trained exits have nested per-tile distortion. No decoder is executed.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flexplus"))
from null_models_uf import bayer
OUT = ROOT / "paper/data/refresh20260927"


def map_at(t, rank, n_exits):
    rung = min(int(np.floor(t)), n_exits - 1)
    return np.where(rank < t - rung, min(rung + 1, n_exits - 1), rung)


def main():
    torch.set_num_threads(2)
    dp = ROOT / "flexplus/results/router_dump_e15_ce_soft.pt"
    rp = ROOT / "flexplus/results/null_models_uf_e15.json"
    dump = torch.load(dp, map_location="cpu", weights_only=False)
    raw = json.loads(rp.read_text())
    prior = {(r["qp"], r["frame"]): r for r in raw["per_frame"]}
    j = dump["j"]; cost = dump["cost"].double().numpy()[j:]; K = len(cost)
    rows, nonmonotonic = [], []
    for qp, frames in dump["frames"].items():
        for fi, frame in enumerate(frames):
            old = prior[(int(qp), fi)]
            rank = bayer(*frame["grid"]).double().numpy()
            breaks = np.unique(np.r_[np.arange(K), *(rank + k for k in range(K - 1))])
            extra = [v["dither"]["param"] for v in old["budgets"].values() if v["dither"] is not None]
            probes = np.unique(np.r_[breaks, (breaks[:-1]+breaks[1:])/2, extra])
            maps = np.array([map_at(t, rank, K) for t in probes])
            _, keep = np.unique(maps, axis=0, return_index=True)
            keep.sort(); maps, probes = maps[keep], probes[keep]
            M = frame["M"].double().numpy()[:,j:]
            db = 10*np.log10(M[np.arange(len(M))[None],maps].mean(1)/frame["R"])
            saving = 100*(1-cost[maps].mean(1))
            assert (np.diff(saving) <= 1e-8).all()
            # Increasing t deepens the map. A positive change in source loss
            # contradicts the monotonic quality assumption, even as cost rises.
            increases = np.flatnonzero(np.diff(db) > 1e-8)
            if len(increases):
                nonmonotonic.append(dict(qp=int(qp),frame=fi,n_increases=len(increases)))
            for budget in raw["budgets"]:
                archived = old["budgets"][str(budget)]["dither"]
                if archived is not None:
                    k = map_at(archived["param"], rank, K)
                    reproduced_db = 10*np.log10(M[np.arange(len(M)),k].mean()/frame["R"])
                    reproduced_saving = 100*(1-cost[k].mean())
                    assert abs(reproduced_db-archived["db"]) < 1e-8
                    assert abs(reproduced_saving-archived["saving"]) < 1e-8
                feasible = np.flatnonzero(db <= budget+1e-12)
                best = min(feasible,key=lambda i:(-saving[i],db[i])) if len(feasible) else None
                gain = None if archived is None or best is None else float(saving[best]-archived["saving"])
                assert gain is None or gain > -1e-8
                rows.append(dict(qp=int(qp),frame=fi,sequence=dump["names"][fi],budget=budget,
                    archived_saving=None if archived is None else archived["saving"],
                    enumerated_saving=None if best is None else float(saving[best]),
                    enumerated_db=None if best is None else float(db[best]),
                    enumerated_t=None if best is None else float(probes[best]),
                    improvement_points=gain))
    summary=[]
    for b in raw["budgets"]:
        rr=[r for r in rows if r["budget"]==b]
        paired=[r for r in rr if r["improvement_points"] is not None]
        gains=[r["improvement_points"] for r in paired]
        summary.append(dict(budget=b,n=len(rr),paired_n=len(paired),
            improved=sum(v>1e-8 for v in gains),mean_gain_points=float(np.mean(gains)),
            max_gain_points=max(gains,default=0),
            recovered_feasible=sum(r["archived_saving"] is None and r["enumerated_saving"] is not None for r in rr)))
    result=dict(method="Enumerate every Bayer rank threshold, rung endpoint and open interval, plus archived mixture parameters",
        scope="Fixed ordered-dither family and padded source-MSE tables only; no reconstruction or runtime rerun",
        checks="Every archived feasible control and MAC saving reproduced; monotone cost; no feasible saving lost",
        n_frame_qp=len(prior),nonmonotonic_frame_qp=len(nonmonotonic),nonmonotonic_cases=nonmonotonic,
        rows=rows,summary=summary,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [dp,rp,ROOT/"flexplus/null_models_uf.py",Path(__file__)]})
    (OUT/"dither_control_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(nonmonotonic_frame_qp=len(nonmonotonic),summary=summary),indent=2))


if __name__=="__main__":
    main()
