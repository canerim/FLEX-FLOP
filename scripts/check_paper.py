"""Verify the numbers written into the paper's PROSE against results/.

Tables and macros regenerate from the measurements and cannot drift. Prose
cannot: every sentence of the form "falls from 25.9% to 4.2%" is a number typed
by hand, and this project has shipped three of those wrong. This checks the
claims that a reader would act on, and exits non-zero if any has moved.

Add a claim here whenever one is written into the text.
"""
import json, re, sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def J(name):
    p = R / "results" / name
    return json.load(open(p)) if p.exists() else None


CLAIMS = []


def claim(label, expected, actual, tol=0.06):
    ok = actual is not None and abs(expected - actual) <= tol
    CLAIMS.append((ok, label, expected, actual, tol))


# ---- coupling ---------------------------------------------------------------
d = J("coupling_ablation.json")
if d:
    by = {r["qp"]: r for r in d["rows"]}
    for qp, pad, cpl in ((0, 33.44, 31.94), (32, 25.94, 4.15), (63, 19.25, 0.40)):
        if qp in by:
            claim(f"coupling q{qp} padded saving", pad, by[qp]["padded"]["saving"])
            claim(f"coupling q{qp} coupled saving", cpl,
                  by[qp]["coupled"]["saving"])
    for qp, pad, cpl in ((0, 0.0358, 0.0031), (63, 0.0564, 0.0300)):
        if qp in by:
            claim(f"coupling q{qp} floor padded", pad,
                  by[qp]["padded"]["floor_db"], 0.0006)
            claim(f"coupling q{qp} floor coupled", cpl,
                  by[qp]["coupled"]["floor_db"], 0.0006)

# ---- contamination ----------------------------------------------------------
d = J("contamination_law.json")
if d:
    import numpy as np
    b = np.array(d["b"])
    for q, exp in (("0", 2.38), ("32", 2.22), ("63", 1.93)):
        if q in d["seam_db"]:
            y = np.array(d["seam_db"][q])
            a = np.polyfit(np.log(b), np.log(y), 1)[0]
            claim(f"contamination exponent q{q}", exp, float(a), 0.02)

# ---- theory ---------------------------------------------------------------
t = J("theory_checks.json")
if t:
    claim("theory: propositions passed", 7, t["n_passed"], 0)
    claim("theory: tiles that switch together", 2, t["m_simultaneous"], 0)
    claim("theory: lattice bound (pts)", 0.745, t["lattice_bound_pts"], 3)
    claim("theory: measured spacing (pts)", 0.745, t["lattice_spacing_pts"], 3)

# ---- rate rank ------------------------------------------------------------
rr = J("raterank_RECIPE512_b01.json")
b1 = J("router_RECIPE512_b01.json")
if rr and b1:
    B = {r["qp"]: r["saving_pct_vs_release"] for r in b1["rows"]
         if r.get("budget_reachable")}
    for r in rr["rows"]:
        if not r.get("budget_reachable") or r["qp"] not in B:
            continue
        if r["qp"] == 63:
            claim("rate-rank q63", 12.1, r["saving_pct_vs_release"], 0.1)
            claim("rate-rank beats B by (q63)", 8.6,
                  r["saving_pct_vs_release"] - B[r["qp"]], 0.15)
        if r["qp"] == 0:
            claim("rate-rank q0", 29.9, r["saving_pct_vs_release"], 0.1)
    claim("rate-rank crossover qp", 32,
          min(r["qp"] for r in rr["rows"]
              if r.get("budget_reachable") and r["qp"] in B
              and r["saving_pct_vs_release"] > B[r["qp"]]), 0)

# ---- hull -------------------------------------------------------------------
d = J("hull_gap.json")
if d:
    claim("hull: swept allocations", 95, d["n_hull_allocations"], 0)
    claim("hull: Pareto points", 635, d["n_pareto"], 1)
    claim("hull: worst gap (pts)", 0.05,
          max(abs(r["gap_pts"]) for r in d["rows"]), 0.005)

# ---- map transfer -----------------------------------------------------------
d = J("map_transfer.json")
if d:
    rt = {(r["from"], r["to"]): r for r in d["rows"] if r["kind"] == "rate"}
    if (0, 63) in rt:
        claim("transfer q0->q63 delivered dB", 0.190, rt[(0, 63)]["transfer_db"],
              0.002)
        claim("transfer q0->q63 saving", 33.05, rt[(0, 63)]["transfer_saving"])
    if (63, 0) in rt:
        claim("transfer q63->q0 delivered dB", 0.075, rt[(63, 0)]["transfer_db"],
              0.002)
        claim("transfer q63->q0 saving", 19.84, rt[(63, 0)]["transfer_saving"])

# ---- encoder cost -----------------------------------------------------------
d = J("encoder_cost.json")
if d:
    claim("encoder: deployed table x decode", 4.6, d["x_deployed"], 0.25)
    claim("encoder: full-frame table x decode", 1.4, d["x_full_frame"], 0.1)
    claim("encoder: map agreement", 85, 100 * d["approx"]["agreement"], 2)

# ---- static baseline --------------------------------------------------------
d = J("static_RECIPE512_b01.json")
if d:
    rows = {r["qp"]: r for r in d["rows"]}
    if 0 in rows:
        u3 = next((u for u in rows[0]["uniform"] if u["exit"] == 3), None)
        if u3:
            claim("static: uniform exit 3 at q0", 27.0, u3["saving"], 0.1)
    best = [r["best_static"]["saving"] if r["best_static"] else 0.0
            for r in d["rows"]]
    claim("static: best static mean", 10.2, sum(best) / len(best), 0.1)
    r63 = rows.get(63)
    if r63 and r63.get("rate_rank"):
        claim("rate-rank q63 dB", 0.110, r63["rate_rank"]["db"], 0.002)
        claim("random q63 dB", 0.141, r63["random"]["db"], 0.002)
        gapv = r63["random"]["db"] - r63["oracle"]["db"]
        got = r63["random"]["db"] - r63["rate_rank"]["db"]
        claim("rate-rank recovers %", 75, 100 * got / gapv, 2)

# ---- adapters ---------------------------------------------------------------
d = J("adapter_ablation.json")
if d:
    r63 = next((r for r in d["rows"] if r["qp"] == 63), None)
    if r63:
        claim("adapters: exit 2 without, q63", 4.40, r63["all_off"]["2"], 0.01)
        claim("adapters: exit 2 gain, q63", 4.10,
              r63["all_off"]["2"] - r63["trained"]["2"], 0.01)
        claim("adapters: deepest exit control", 0.0,
              r63["all_off"]["5"] - r63["trained"]["5"], 1e-9)

# ---- per class --------------------------------------------------------------
d = J("per_class_RECIPE512.json")
e = J("per_class_BEST128.json")
if d:
    r0 = next((r for r in d["rows"]
               if r["qp"] == 0 and abs(r["budget_db"] - 0.1) < 1e-9), None)
    if r0:
        for c, exp in (("MCL-JCV", 36.3), ("HEVC_D", 11.3), ("HEVC_C", 20.9)):
            if c in r0["per_class"]:
                claim(f"per-class q0 {c}", exp, r0["per_class"][c]["saving"], 0.1)
    if e:
        e0 = next((r for r in e["rows"]
                   if r["qp"] == 0 and abs(r["budget_db"] - 0.1) < 1e-9), None)
        if e0:
            for c, exp in (("MCL-JCV", 34.3), ("HEVC_D", 13.7)):
                if c in e0["per_class"]:
                    claim(f"128px q0 {c}", exp, e0["per_class"][c]["saving"], 0.1)

bad = [c for c in CLAIMS if not c[0]]
w = max(len(c[1]) for c in CLAIMS) if CLAIMS else 10
for ok, label, exp, act, tol in CLAIMS:
    a = f"{act:.4f}" if isinstance(act, float) else str(act)
    print(f"  [{'ok ' if ok else 'BAD'}] {label:<{w}}  paper {exp:>8}   "
          f"measured {a:>9}")
# ---- lint: numbers typed into the prose rather than expanded from a macro --
# Not a failure. Some numbers legitimately belong in the text -- a tile size, a
# count of exits, a year -- and the point is to watch the list shrink rather
# than to forbid it. What it catches is a measurement retyped by hand, which is
# how every drift in this project started.
# Budgets and the qp grid are the axes of every table, not measurements.
ALLOW = {"0", "1", "2", "3", "4", "6", "8", "12", "16", "32", "40", "53", "64",
         "100", "105", "128", "240", "256", "416", "480", "512", "720", "832",
         "1080", "1280", "1920", "2048",
         "0.1", "0.3", "0.5", "1.0", "0.05", "0.2", "0.4"}
_tex = re.sub(r"(?<!\\)%.*", "", (R / "paper/main.tex").read_text())
_pat = re.compile(r"(?<![\\A-Za-z0-9])(\d+(?:\.\d+)?)\s*(?:\\%|\\dB\b|dB\b)")
typed = {m.group(1) for m in _pat.finditer(_tex)} - ALLOW
if typed:
    print(f"\n  lint: {len(typed)} numeric literals typed into main.tex rather "
          f"than expanded from a macro:")
    print("        " + ", ".join(sorted(typed, key=float)))

print(f"\n  {len(CLAIMS)-len(bad)}/{len(CLAIMS)} prose claims match the data")
# Recorded so make_paper_tables.py can quote the count without running this.
json.dump({"n_claims": len(CLAIMS), "n_passed": len(CLAIMS) - len(bad),
           "failed": [c[1] for c in CLAIMS if not c[0]]},
          open(R / "results/check_paper.json", "w"), indent=2)
sys.exit(1 if bad else 0)
