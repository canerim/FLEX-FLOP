"""Descriptive sensitivity of the archived, source-aware delivered-cap result.

This analysis consumes already measured outputs. It does not rerun a codec,
create new maps, calibrate a held-out policy, or measure runtime. All five
quality indices of a sequence are removed together in the leave-one-out check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/refresh20260927/delivered_frontier_audit.json"
TOL = 1e-8


def summarise(rows):
    values = [r["saving_difference_points"] for r in rows]
    return dict(
        n=len(rows),
        mean_points=mean(values),
        median_points=median(values),
        router_higher=sum(v > TOL for v in values),
        dither_higher=sum(v < -TOL for v in values),
        tied=sum(abs(v) <= TOL for v in values),
    )


def main():
    data = json.loads(SOURCE.read_text())
    records = []
    for cap in sorted({r["cap"] for r in data["rows"]}):
        policies = {
            rule: {(r["sequence"], r["qp"]): r for r in data["rows"]
                   if r["cap"] == cap and r["rule"] == rule}
            for rule in ("router", "dither")
        }
        assert policies["router"].keys() == policies["dither"].keys()
        rows = []
        for key, router in sorted(policies["router"].items()):
            dither = policies["dither"][key]
            rows.append(dict(
                sequence=key[0], qp=key[1],
                saving_difference_points=router["saving"] - dither["saving"],
                loss_difference_db=router["loss"] - dither["loss"],
                any_fallback=router["fallback"] or dither["fallback"],
            ))
        seqs = sorted({r["sequence"] for r in rows})
        qps = sorted({r["qp"] for r in rows})
        assert len(seqs) == 53 and len(qps) == 5 and len(rows) == 265
        assert all(sum(r["sequence"] == s for r in rows) == 5 for s in seqs)
        leave_sequence = [dict(
            removed=s,
            mean_points=mean(r["saving_difference_points"] for r in rows
                             if r["sequence"] != s),
        ) for s in seqs]
        sequence_means = [dict(
            sequence=s,
            saving_difference_points=mean(r["saving_difference_points"] for r in rows
                                          if r["sequence"] == s),
        ) for s in seqs]
        records.append(dict(
            cap_db=cap,
            paired_frames=summarise(rows),
            sequence_means=summarise(sequence_means),
            mean_loss_difference_db=mean(r["loss_difference_db"] for r in rows),
            by_qp=[dict(qp=q, **summarise([r for r in rows if r["qp"] == q])) for q in qps],
            leave_one_sequence_out_range_points=[
                min(r["mean_points"] for r in leave_sequence),
                max(r["mean_points"] for r in leave_sequence),
            ],
            leave_one_sequence_out=leave_sequence,
            leave_one_qp_out=[dict(
                removed=q,
                mean_points=mean(r["saving_difference_points"] for r in rows if r["qp"] != q),
            ) for q in qps],
            both_nonfallback=summarise([r for r in rows if not r["any_fallback"]]),
            paired_rows=rows,
        ))
    print(json.dumps(dict(
        scope="Descriptive influence checks of retrospective common-cap choices; decoder MAC savings, not runtime. No new reconstruction or held-out policy.",
        interpretation="Leave-one-out ranges are sensitivity diagnostics, not confidence intervals. Removing fallback pairs changes the estimand and is not the primary result. Wins refer to MAC saving under a common per-output loss cap, not exact equal quality.",
        tie_tolerance_points=TOL,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=records,
    ), indent=2))


if __name__ == "__main__":
    main()
