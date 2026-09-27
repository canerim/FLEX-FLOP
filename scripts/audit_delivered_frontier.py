"""Retrospective quality-cap selection among actually decoded archive maps.

The candidate pool is restricted to six archived nominal-budget plans per
policy. Each frame also has its evaluated fine-tuned full-frame reference,
with zero relative loss and zero MAC saving. No interpolation, unseen mixed
image, new GPU evaluation or deployment latency is introduced.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"paper/data/refresh20260927"
RULES=["oracle","router","dither","uniform"]
BUDGETS=[.05,.1,.15,.2,.3,.5]


def select(candidates, cap):
    # The zero-loss full-frame anchor was evaluated for every source pair.
    pool=[dict(saving=0.,loss=0.,nominal_budget=None,fallback=True,map=None)]
    seen=set()
    for b,value in candidates.items():
        if value is None:continue
        key=tuple(value["map"])
        if key in seen:continue
        seen.add(key)
        if value["db_rgb"]<=cap+1e-4:
            pool.append(dict(saving=value["saving"],loss=value["db_rgb"],
                nominal_budget=float(b),fallback=False,map=value["map"]))
    return min(pool,key=lambda r:(-r["saving"],r["loss"]))


def ci(values,groups):
    _,ix=np.unique(groups,return_inverse=True)
    sums=np.bincount(ix,weights=values); counts=np.bincount(ix)
    draws=np.random.default_rng(20260927).integers(0,len(counts),size=(5000,len(counts)))
    x=sums[draws].sum(1)/counts[draws].sum(1)
    lo,hi=np.quantile(x,[.025,.975])
    return dict(mean=float(np.mean(values)),lo=float(lo),hi=float(hi))


def main():
    # Independent constructed example: a high-saving plan over the achieved
    # cap must be rejected; an empty or negative-saving pool chooses reference.
    toy={"0.1":dict(map=[2],saving=20.,db_rgb=.12),"0.05":dict(map=[3],saving=10.,db_rgb=.04)}
    assert select(toy,.1)["saving"]==10.
    assert select({},.1)["fallback"]
    rawpath=ROOT/"flexplus/results/eval_rules_ctc_e15.json"
    raw=json.loads(rawpath.read_text()); rows=[]
    assert len(raw["rows"])==265
    duplicate_comparisons=0
    for frame in raw["rows"]:
        # Deduplication requires the same measured output and cost, including
        # repeated maps under different policy labels.
        seen_maps={}
        for candidates in frame["rules"].values():
            for value in candidates.values():
                if value is None:continue
                key=tuple(value["map"])
                if key in seen_maps:
                    old=seen_maps[key]
                    assert value["db_rgb"]==old["db_rgb"]
                    assert value["saving"]==old["saving"]
                    duplicate_comparisons+=1
                else:seen_maps[key]=value
        for rule in RULES:
            previous=-float("inf")
            for cap in BUDGETS:
                chosen=select(frame["rules"][rule],cap)
                assert chosen["loss"]<=cap+1e-4 and chosen["saving"]>=previous-1e-10
                previous=chosen["saving"]
                rows.append(dict(sequence=frame["seq"],qp=frame["qp"],cap=cap,rule=rule,**chosen))
    summary=[];contrasts=[]
    for cap in BUDGETS:
        for rule in RULES:
            rr=[r for r in rows if r["cap"]==cap and r["rule"]==rule]
            summary.append(dict(cap=cap,rule=rule,n=len(rr),
                **ci([r["saving"] for r in rr],[r["sequence"] for r in rr]),
                mean_loss=float(np.mean([r["loss"] for r in rr])),
                max_loss=max(r["loss"] for r in rr),fallbacks=sum(r["fallback"] for r in rr)))
        for a,b in [("router","dither"),("oracle","router")]:
            aa={(r["sequence"],r["qp"]):r for r in rows if r["cap"]==cap and r["rule"]==a}
            bb={(r["sequence"],r["qp"]):r for r in rows if r["cap"]==cap and r["rule"]==b}
            assert aa.keys()==bb.keys()
            contrasts.append(dict(cap=cap,contrast=f"{a}_minus_{b}",n=len(aa),
                **ci([aa[k]["saving"]-bb[k]["saving"] for k in aa],[k[0] for k in aa])))
    result=dict(
        method="For every frame, choose the cheapest actually decoded archived map satisfying a common delivered RGB-loss cap; include the evaluated full-frame e15 reference as explicit zero-loss/zero-saving fallback.",
        anchor="Fine-tuned e15 full-frame reconstruction, cropped to original pixels; not released D12",
        candidates="Up to six distinct nominal-budget plans per policy; maps deduplicated. No interpolation or newly reconstructed maps.",
        information="Retrospective source-aware choice after observing candidate final quality. This is a limited-pool diagnostic, not a trained held-out policy, a global oracle or an encoder-time claim.",
        accounting="Decoder MAC model only; fallback count explicit. The cost of obtaining/rejecting candidates is not included.",
        tolerance_db=1e-4,bootstrap="5000 paired sequence-cluster draws, seed 20260927",
        checks="Independent cap/fallback examples; all 265 pairs included; every selected loss satisfies cap; saving monotone as cap relaxes; repeated maps have exactly equal measured RGB loss and MAC saving.",
        identical_duplicate_map_comparisons=duplicate_comparisons,
        rows=rows,summary=summary,contrasts=contrasts,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [rawpath,Path(__file__)]})
    (OUT/"delivered_frontier_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(summary=[r for r in summary if r["cap"]==.1],contrasts=contrasts),indent=2))


if __name__=="__main__":main()
