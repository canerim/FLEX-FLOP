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


def J(*names):
    """First of `names` that exists, so a corrected file supersedes an old one."""
    for name in names:
        p = R / "results" / name
        if p.exists():
            return json.load(open(p))
    return None


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
b1 = J("router_RECIPE512_b01_fixed.json", "router_RECIPE512_b01.json")
if rr and b1:
    B = {r["qp"]: r["saving_pct_vs_release"] for r in b1["rows"]
         if r.get("budget_reachable")}
    for r in rr["rows"]:
        if not r.get("budget_reachable") or r["qp"] not in B:
            continue
        if r["qp"] == 63:
            claim("rate-rank q63", 12.1, r["saving_pct_vs_release"], 0.1)
        if r["qp"] == 0:
            claim("rate-rank q0", 29.9, r["saving_pct_vs_release"], 0.1)
    d_ = [(r["qp"], r["saving_pct_vs_release"] - B[r["qp"]]) for r in rr["rows"]
          if r.get("budget_reachable") and r["qp"] in B]
    claim("rate-rank: rates it wins", 3, sum(1 for _, v in d_ if v > 0), 0)
    claim("rate-rank: best margin (pts)", 2.7, max(v for _, v in d_), 0.1)
    claim("rate-rank: worst deficit (pts)", 0.8,
          max(-v for _, v in d_ if v <= 0), 0.1)

# ---- hybrid C -------------------------------------------------------------
hy = J("hybrid_RECIPE512_b01_fixed.json", "hybrid_RECIPE512_b01.json")
b1f = J("router_RECIPE512_b01_fixed.json", "router_RECIPE512_b01.json")
sg = J("signalled_RECIPE512_ctc53.json")
if hy and b1f and sg:
    rws = [r for r in hy["rows"] if r.get("budget_reachable")]

    def _h(q, rho):
        return next((r["saving_pct_vs_release"] for r in rws
                     if r["qp"] == q and abs(r["rho"] - rho) < 1e-9), None)
    Bv = {r["qp"]: r["saving_pct_vs_release"] for r in b1f["rows"]
          if r.get("budget_reachable")}
    Av = {r["qp"]: r["saving_pct_vs_release"] for r in sg["rows"]
          if abs(r["budget_db"] - 0.1) < 1e-9 and r.get("budget_reachable")}
    ends = [abs(_h(q, 0.0) - Bv[q]) for q in Bv if _h(q, 0.0) is not None]
    claim("hybrid: rho=0 reproduces B (max |diff|)", 0.0, max(ends), 0.05)
    ends1 = [abs(_h(q, 1.0) - Av[q]) for q in Av if _h(q, 1.0) is not None]
    claim("hybrid: rho=1 reproduces A (max |diff|)", 0.0, max(ends1), 0.05)
    beat = [(_h(q, 0.5) - Av[q]) for q in Av if _h(q, 0.5) is not None]
    claim("hybrid: rates where half beats all", 4, sum(1 for d in beat if d > 0), 0)
    claim("hybrid: best margin over A (pts)", 0.42, max(beat), 0.05)
    # and the margin is not the bisection tolerance in disguise
    dbs = [abs(r["db_vs_uf"] - 0.1) for r in rws]
    claim("hybrid: worst |dB - budget|", 0.0, max(dbs), 1e-3)

# ---- hybrid at the loose budget -------------------------------------------
hy3 = J("hybrid_RECIPE512_b03_fixed.json")
if hy3:
    r3 = [r for r in hy3["rows"] if r.get("budget_reachable")]

    def _c3(q, rho):
        return next((r["saving_pct_vs_release"] for r in r3
                     if r["qp"] == q and abs(r["rho"] - rho) < 1e-9), None)
    rec = []
    for q in sorted({r["qp"] for r in r3}):
        b0, bh, ba = _c3(q, 0.0), _c3(q, 0.5), _c3(q, 1.0)
        if None in (b0, bh, ba) or ba - b0 < 0.5:
            continue
        rec.append(100 * (bh - b0) / (ba - b0))
    if rec:
        claim("hybrid 0.3 dB: unsaturated rates", 2, len(rec), 0)
        claim("hybrid 0.3 dB: half-map recovery, low", 68, min(rec), 1.0)
        claim("hybrid 0.3 dB: half-map recovery, high", 91, max(rec), 1.0)

# ---- which predictor C is built on ----------------------------------------
hyr = J("hybrid_raterank_b01.json")
if hy and hyr:
    def _t(d):
        return {(r["qp"], round(r["rho"], 4)): r["saving_pct_vs_release"]
                for r in d["rows"] if r.get("budget_reachable")}
    Th, Tr = _t(hy), _t(hyr)
    ends = [abs(Tr[k] - Th[k]) for k in Th if k in Tr and k[1] == 1.0]
    if ends:
        claim("C: both predictors meet A at rho=1", 0.0, max(ends), 0.02)
    mid = [(Tr[k] - Th[k]) for k in Th if k in Tr and k[1] < 1.0]
    if mid:
        claim("C: bits predictor's best margin", 2.6, max(mid), 0.1)
        claim("C: head predictor's best margin", 1.1, -min(mid), 0.1)
    best = max(Tr.items(), key=lambda t: t[1])
    claim("C: best measured saving", 33.0, best[1], 0.1)

# ---- the contamination law ------------------------------------------------
cl = J("contamination_law.json")
if cl:
    import numpy as _np
    _b = _np.array(cl["b"])
    errs = {"area": [], "b2": [], "fit": []}
    for q, y in cl["seam_db"].items():
        y = _np.array(y)
        for k, m in (("area", _np.array(cl["area_model"])),
                     ("b2", (_b / _b.max()) ** 2)):
            sc = (m @ y) / (m @ m)
            errs[k].append(100 * _np.mean(_np.abs(sc * m - y) / y))
        A_ = _np.vstack([_np.log(_b), _np.ones_like(_b)]).T
        al, c0 = _np.linalg.lstsq(A_, _np.log(y), rcond=None)[0]
        errs["fit"].append(100 * _np.mean(_np.abs(_np.exp(c0) * _b ** al - y) / y))
    claim("contamination: area error, worst", 264, max(errs["area"]), 1)
    claim("contamination: area error, best", 160, min(errs["area"]), 1)
    claim("contamination: fitted power, worst", 22, max(errs["fit"]), 1)
    claim("contamination: fitted power, best", 12, min(errs["fit"]), 1)
    claim("contamination: fixed square, worst", 38, max(errs["b2"]), 1)
    claim("contamination: fixed square, best", 31, min(errs["b2"]), 1)

# ---- the band collapse ----------------------------------------------------
bc = J("band_collapse.json")
if bc:
    claim("band: raw spread at 0.1 dB", 15.5, bc["raw_spread_at_tenth_db"], 0.2)
    claim("band: spread after rescaling, mean", 1.7, bc["band_spread_mean"], 0.2)
    claim("band: spread after rescaling, worst", 2.2,
          bc["band_spread_max_excl_edge"], 0.2)

# ---- blend ----------------------------------------------------------------
cb = J("combined_RECIPE512_b01.json")
if cb:
    rws = [r for r in cb["rows"] if r.get("budget_reachable")]

    def _c(q, w):
        return next((r["saving_pct_vs_release"] for r in rws
                     if r["qp"] == q and abs(r["gamma"] - w) < 1e-9), None)
    for q, w, exp in ((48, 0.25, 2.1), (63, 0.1, 0.9)):
        v, v0 = _c(q, w), _c(q, 0.0)
        if v is not None and v0 is not None:
            claim(f"blend gain q{q}", exp, v - v0, 0.15)

# ---- rate rank, loose budgets ---------------------------------------------
rr5 = J("raterank_RECIPE512_b05.json")
if rr5:
    rs5 = [r for r in rr5["rows"] if r.get("budget_reachable")]
    claim("rate-rank 0.5 dB: rates matching the oracle exactly", 5,
          sum(1 for r in rs5 if abs(r["saving_pct_vs_release"]
                                    - r["oracle_saving_pct_vs_release"]) < 1e-6), 0)
    claim("rate-rank 0.5 dB: agreement with the oracle map", 1.0,
          min(r["agreement"] for r in rs5), 1e-6)

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
    # A literal that equals a macro's value is almost certainly that macro,
    # typed out. Say so, because that is the actionable half of the list.
    mac = {}
    _mp = R / "paper/tables/macros.tex"
    if _mp.exists():
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             _mp.read_text()):
            mac.setdefault(m.group(2).strip(), m.group(1))
    hits = sorted(((v, mac[v]) for v in typed if v in mac), key=lambda t: float(t[0]))
    rest = sorted((v for v in typed if v not in mac), key=float)
    print(f"\n  lint: {len(typed)} numeric literals typed into main.tex rather "
          f"than expanded from a macro")
    if hits:
        print(f"        {len(hits)} of them equal an existing macro:")
        for v, n in hits:
            print(f"          {v}  ->  \\{n}")
    if rest:
        print("        no macro exists for: " + ", ".join(rest))

print(f"\n  {len(CLAIMS)-len(bad)}/{len(CLAIMS)} prose claims match the data")
# Recorded so make_paper_tables.py can quote the count without running this.
json.dump({"n_claims": len(CLAIMS), "n_passed": len(CLAIMS) - len(bad),
           "failed": [c[1] for c in CLAIMS if not c[0]]},
          open(R / "results/check_paper.json", "w"), indent=2)
sys.exit(1 if bad else 0)
