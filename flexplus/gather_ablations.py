"""Every ablation this project ran, pulled into one file.

Fourteen families of them are scattered across two results directories in
half a dozen schemas, some of which no longer match the script that wrote
them. This reads what is readable, records where each number came from, and
says out loud which families it could not parse rather than quietly dropping
them -- a missing ablation and a failed one look identical in a table.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
UF = Path.home() / "FLEX-UF" / "results"
QPS = [0, 16, 32, 48, 63]
OUT, MISSING = {}, []


def J(p):
    return json.loads(Path(p).read_text())


def sv(r):
    return r.get("saving_pct_measured", r.get("saving_pct_vs_release"))


def at01(d):
    m = {r["qp"]: sv(r) for r in d["rows"]
         if r.get("budget_reachable") and abs(r.get("budget_db", 0) - 0.1) < 1e-9}
    return [m[q] for q in QPS if q in m]


def add(name, **kw):
    OUT[name] = kw


def attempt(name, fn):
    try:
        fn()
    except Exception as e:
        MISSING.append(f"{name}: {type(e).__name__} {e}")


# 1 --------------------------------------------------------------- granularity
def _gran():
    g = J(R / "granularity_ctc53_fixed.json")
    cells = sorted({c["cell_px"] for r in g["rows"] for c in r["cells"]},
                   reverse=True)
    rows = []
    for cp in cells:
        v, band = [], []
        for q in QPS:
            r = [x for x in g["rows"] if x["qp"] == q][0]
            c = [c for c in r["cells"] if c["cell_px"] == cp][0]
            v.append(c["saving_pct_vs_release"]); band.append(c["band_cost_pct"])
        rows.append({"cell_px": cp, "mean": float(np.mean(v)),
                     "per_qp": v, "band_pct": float(np.mean(band))})
    add("granularity", what="cell size, per-position decode, 0.1 dB",
        source="flexplus/results/granularity_ctc53_fixed.json",
        ckpt_epoch=4, rows=rows)
attempt("granularity", _gran)


# 2 ---------------------------------------------------------------- the clamp
def _clamp():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from budget_search import Sweep, QPS as Q2
    out = []
    for j in (2, 0):
        s = Sweep(str(R / "cells_ctc64.npz"), j)
        v = [s.operate(q, 0.1)[1] for q in Q2]
        out.append({"split_depth": j, "mean": float(np.mean(v)), "per_qp": v,
                    "ceiling_pct": s.ceiling_saving(0)})
    add("clamp", what="split depth, 64 px cells, per-position, 0.1 dB",
        source="flexplus/results/cells_ctc64.npz", ckpt_epoch=4, rows=out)
attempt("clamp", _clamp)


# 3 ------------------------------------------------------------ decode geometry
def _geom():
    paper = J(Path.home() / "FLEX-UF/results/signalled_RECIPE512_ctc53.json")
    rows = [{"decode": "tiled 256 px (the paper)", "mean": float(np.mean(at01(paper))),
             "per_qp": at01(paper)}]
    g = J(R / "granularity_ctc53_fixed.json")
    for cp, lab in ((256, "per-position, 256 px cells"),
                    (64, "per-position, 64 px cells")):
        v = [[c for c in [x for x in g["rows"] if x["qp"] == q][0]["cells"]
              if c["cell_px"] == cp][0]["saving_pct_vs_release"] for q in QPS]
        rows.append({"decode": lab, "mean": float(np.mean(v)), "per_qp": v})
    add("decode_geometry", what="tiled against per-position, 0.1 dB",
        source="signalled_RECIPE512_ctc53.json + granularity_ctc53_fixed.json",
        ckpt_epoch=4, rows=rows)
attempt("decode_geometry", _geom)


# 4 ------------------------------------------------------------- router models
def _models():
    d = J(R / "router_fit.json")
    rows = []
    for r in d["rows"]:
        pq = r.get("per_qp", {})
        v = [pq[str(q)]["saving_pct"] for q in QPS
             if str(q) in pq and pq[str(q)]]
        if v:
            rows.append({"model": r["model"], "mean": float(np.mean(v)),
                         "per_qp": v})
    add("router_models", what="what predicts the exit-error curve, 0.1 dB",
        source="flexplus/results/router_fit.json",
        cell_px=d.get("cell_px"), split_depth=d.get("j"), rows=rows)
attempt("router_models", _models)


# 5 ------------------------------------------------------------- router inputs
def _inputs():
    rows = []
    for f in sorted(UF.glob("router_ablation_*_e4.json")):
        d = J(f)
        ev = d.get("eval", d.get("router2_meta", {}).get("eval", {}))
        rows.append({"inputs": d.get("inputs", d.get("label")),
                     "agree_pct": 100 * ev.get("agree", float("nan")),
                     "constant_pct": 100 * ev.get("constant_best_agree",
                                                  float("nan")),
                     "file": f.name})
    rows = [r for r in rows if r["agree_pct"] == r["agree_pct"]]
    add("router_inputs", what="which input group carries the signal",
        source="FLEX-UF/results/router_ablation_*_e4.json", rows=rows)
attempt("router_inputs", _inputs)


# 6 --------------------------------------------------------- closed-form refit
def _calex():
    rows = []
    for f, lab in (("signalled_CALEX_A_survivor.json", "survivors, matched lambda"),
                   ("signalled_CALEX_A_pooled.json", "all tiles (control)"),
                   ("signalled_CALEX_A_survivor_fixedlam.json",
                    "survivors, WRONG lambda (control)")):
        d = J(R / f); v = at01(d)
        rows.append({"arm": lab, "mean": float(np.mean(v)), "per_qp": v})
    base = J(Path.home() / "FLEX-UF/results/signalled_RECIPE512_0825_2350.json")
    add("calex_refit", what="closed-form adapter refit, global lambda, 0.1 dB",
        baseline_mean=float(np.mean(at01(base))), ckpt_epoch=9,
        source="flexplus/results/signalled_CALEX_A_*.json", rows=rows)
attempt("calex_refit", _calex)


# 7 -------------------------------------------------------------- fine-tuning
def _tierb():
    rows = []
    for f, lab in (("signalled_perframe_e9.json", "baseline, no fine-tune"),
                   ("signalled_perframe_B2prior.json",
                    "6000 steps, usage-matched weights"),
                   ("signalled_perframe_B2control.json",
                    "6000 steps, uniform weights (control)"),
                   ("signalled_perframe_calexA.json",
                    "closed-form refit, no training")):
        d = J(R / f)
        v = [r["saving_pct_measured"] for r in d["rows"]]
        rows.append({"arm": lab, "mean": float(np.mean(v)), "per_qp": v})
    add("fine_tuning", what="does gradient descent add anything, per-frame "
        "guarantee, 0.1 dB", ckpt_epoch=9,
        source="flexplus/results/signalled_perframe_*.json", rows=rows)
attempt("fine_tuning", _tierb)


# 8 ------------------------------------------------------------- block sparsity
def _blocks():
    rows = []
    for f, lab in (("mask_structure_256_all.json", "256 px cells"),
                   ("mask_structure_64_all.json", "64 px cells")):
        d = J(R / f)
        ideal = float(np.mean([r["ideal_pct"] for r in d["rows"]]))
        blocks = {k: float(np.mean([r["blocks"][k] for r in d["rows"]]))
                  for k in d["rows"][0]["blocks"]}
        rows.append({"cells": lab, "ideal_pct": ideal, "blocks": blocks})
    add("block_sparsity", what="what a B x B block-sparse kernel would realise",
        source="flexplus/results/mask_structure_*_all.json", rows=rows)
attempt("block_sparsity", _blocks)


# 9 ------------------------------------------------------------ the guarantee
def _guar():
    t = J(R / "guarantee_tiled_e9.json")
    add("guarantee", what="where the multiplier is chosen, deployed tiled path",
        source="flexplus/results/guarantee_tiled_e9.json", ckpt_epoch=9,
        rows=[{"mode": "one lambda per rate", **t["global"]},
              {"mode": "one lambda per frame", **t["perframe"]}])
attempt("guarantee", _guar)


# 10 ----------------------------------------------------------------- budget
def _budget():
    rows = []
    for f, b in (("signalled_perframe_calexA.json", 0.100),
                 ("signalled_perframe_ideal.json", 0.121317),
                 ("signalled_perframe_b015.json", 0.150)):
        p = R / f
        if not p.exists():
            continue
        d = J(p)
        v = [r["saving_pct_measured"] for r in d["rows"]]
        sat = sum(r.get("saturated", 0) for r in d["rows"])
        rows.append({"budget_db": b, "mean": float(np.mean(v)),
                     "saturated": sat, "n": sum(r["n"] for r in d["rows"])})
    add("budget", what="the operating point, per-frame guarantee, epoch 9",
        source="flexplus/results/signalled_perframe_*.json", rows=rows)
attempt("budget", _budget)


# 11 -------------------------------------------------------------- safe routing
def _safe():
    d = J(R / "safe_routing.json")
    add("safe_routing", what="can a margin rule shorten the decoder-side tail",
        source="flexplus/results/safe_routing.json",
        margins=[{"delta": r["margin"], "saving": r["saving_pct"],
                  "max_db": r["max"], "over": r["over_010"]} for r in d["rows"]],
        throttle=[{"target": r["pred_target_db"], "saving": r["saving_pct"],
                   "max_db": r["max"]} for r in d["decoder_perframe"]],
        signalled_lambda=d["signalled_lambda_perframe"])
attempt("safe_routing", _safe)


# 12 --------------------------------------------------------------- generalisation
def _kodak():
    k = J(R / "kodak_curve.json"); pf = J(R / "kodak_perframe.json")
    rows = [{"qp": r["qp"], "budget_db": r["budget_db"],
             "saving": r.get("saving_pct_measured"), "db": r.get("db_vs_uf")}
            for r in k["rows"] if r.get("budget_reachable")]
    per = [{"qp": r["qp"], "saving": r["saving_pct_vs_release"],
            "infeasible": r["infeasible"], "n": r["n"],
            "mean_floor_db": r["mean_floor_db"]}
           for r in pf["rows"] if r.get("mode") == "per_image"]
    add("kodak", what="does it transfer to an image benchmark",
        source="flexplus/results/kodak_*.json", ckpt_epoch=4,
        global_lambda=rows, per_image=per)
attempt("kodak", _kodak)


# 13 -------------------------------------------------------------- epoch series
def _epochs():
    rows = []
    for f in sorted(UF.glob("signalled_RECIPE512_*.json")) + \
             [R / "signalled_RECIPE512_e10.json"]:
        try:
            d = J(f); e = d.get("ckpt_epoch"); v = at01(d)
        except Exception:
            continue
        if e is None or len(v) != len(QPS):
            continue
        rows.append({"epoch": e, "mean": float(np.mean(v))})
    seen, keep = set(), []
    for r in sorted(rows, key=lambda r: r["epoch"]):
        if r["epoch"] not in seen:
            seen.add(r["epoch"]); keep.append(r)
    add("epochs", what="does more training help", source="signalled_RECIPE512_*",
        rows=keep)
attempt("epochs", _epochs)


# 14 ------------------------------------------------------------ prior stability
def _prior():
    add("prior_stability", **J(R / "prior_stability.json"))
attempt("prior_stability", _prior)


(R / "ablations.json").write_text(json.dumps(
    {"families": OUT, "unparsed": MISSING}, indent=2))
print(f"  {len(OUT)} aile toplandi:")
for k, v in OUT.items():
    n = len(v.get("rows", v.get("margins", v.get("global_lambda", []))))
    print(f"    {k:<20} {v.get('what','')[:58]:<58} ({n} satir)")
if MISSING:
    print(f"\n  {len(MISSING)} aile okunamadi:")
    for m in MISSING:
        print("    ", m[:110])
