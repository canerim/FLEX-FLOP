"""Emit every table and inline number the paper uses, as LaTeX, from results/.

No number in paper/main.tex is typed. Tables land in paper/tables/*.tex and
scalars become macros in paper/tables/macros.tex, so \\MainLowRate expands to
whatever the current measurement says. A paper whose numbers are retyped from a
terminal is a paper that will disagree with its own repository within a week.

Every generator here prefers the full 53-sequence CTC file and falls back to the
40-sequence one, printing which it used.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
RES, OUT = R / "results", R / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
QPS = [0, 16, 32, 48, 63]
MACROS = {}


def pick(*names):
    for n in names:
        p = RES / n
        if p.exists():
            print(f"    using {n}")
            return json.load(open(p)), n
    print(f"    MISSING: {names[0]}")
    return None, None


def w(name, body):
    (OUT / name).write_text(body.rstrip() + "\n")
    print(f"  -> paper/tables/{name}")


def mac(k, v):
    MACROS[k] = v


# ---------------------------------------------------------------- main result
print("main results")
d, src = pick("signalled_RECIPE512_ctc53.json", "signalled_RECIPE512_b135.json")
if d:
    by = {b: {r["qp"]: r for r in d["rows"]
              if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
          for b in d["budgets"]}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Budget & " + " & ".join(f"$q{q}$" for q in QPS) + r" & mean \\",
             r"\midrule"]
    for b in sorted(by):
        row = by[b]
        vs = [row[q]["saving_pct_vs_release"] for q in QPS if q in row]
        lines.append(f"{b:.2f}\\,dB & " +
                     " & ".join(f"{row[q]['saving_pct_vs_release']:.1f}" if q in row
                                else "--" for q in QPS) +
                     f" & \\textbf{{{sum(vs)/len(vs):.1f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("main_results.tex", "\n".join(lines))
    r01 = by[0.1]
    mac("MainLowRate", f"{r01[0]['saving_pct_vs_release']:.1f}")
    mac("MainHighRate", f"{r01[63]['saving_pct_vs_release']:.1f}")
    mac("MainMean", f"{sum(r01[q]['saving_pct_vs_release'] for q in QPS)/len(QPS):.1f}")
    mac("NumSeq", str(d["n_sequences"]))
    _mb = [r01[q]["map_bits"] for q in QPS
           if q in r01 and r01[q].get("map_bits")]
    mac("MapBits", f"{sum(_mb)/len(_mb):.0f}")
    mac("MapBitsLo", f"{min(_mb):.0f}")
    mac("MapBitsHi", f"{max(_mb):.0f}")
    mac("RouterParams", "144\\,K")
    mac("RouterCostPct", "0.163")
rl, _ = pick("router_latency.json")
if rl:
    mac("RouterTimePct", f"{rl['router_share_pct_time_median']:.2f}")
    mac("RouterTimeFactor", f"{rl['time_over_mac_factor']:.1f}")
    mac("RouterTimeExtra",
        f"{rl['router_share_pct_time_median'] - rl['router_share_pct_macs']:.2f}")
    mac("MapOverheadLow", f"{100*r01[0]['bpp_added']/0.40:.3f}")

# --------------------------------------------------------- operating structure
print("operating structure")
d, _ = pick("saturation_RECIPE512_ctc53.json", "saturation_RECIPE512.json")
if d:
    S = {r["qp"]: r for r in d["rows"]}
    # Five rates, not the nine measured: a nine-column table does not fit a
    # two-column page, and the figure carries the full sweep anyway.
    qs = [q for q in QPS if q in S] or sorted(S)
    lines = [r"\begin{tabular}{l" + "r" * len(qs) + "}", r"\toprule",
             r"$q$ & " + " & ".join(str(q) for q in qs) + r" \\", r"\midrule",
             r"Floor $D_{\min}$ & " +
             " & ".join(f"{S[q]['floor_db']:.3f}" for q in qs) + r" \\",
             r"Saturation $D_{\mathrm{sat}}$ & " +
             " & ".join(f"{S[q]['saturation_db']:.3f}" for q in qs) + r" \\",
             r"Usable band & " +
             " & ".join(f"{S[q]['saturation_db']-S[q]['floor_db']:.3f}"
                        for q in qs) + r" \\",
             r"0.1\,dB uses & " +
             " & ".join(f"{100*(0.1-S[q]['floor_db'])/(S[q]['saturation_db']-S[q]['floor_db']):.0f}\\%"
                        for q in qs) + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    w("operating.tex", "\n".join(lines))
    mac("Ceiling", f"{d['ceiling_pct']:.1f}")
    mac("FloorLow", f"{S[0]['floor_db']:.3f}")
    mac("FloorHigh", f"{S[max(qs)]['floor_db']:.3f}")
    mac("SatLow", f"{S[0]['saturation_db']:.3f}")
    mac("SatHigh", f"{S[max(qs)]['saturation_db']:.3f}")
    mac("BandUseLow", f"{100*(0.1-S[0]['floor_db'])/(S[0]['saturation_db']-S[0]['floor_db']):.0f}")
    mac("BandUseHigh", f"{100*(0.1-S[max(qs)]['floor_db'])/(S[max(qs)]['saturation_db']-S[max(qs)]['floor_db']):.0f}")
    if 8 in S:
        mac("SatWindowHi", f"{S[8]['saturation_db']:.3f}")
        mac("SatWindowMb", f"{1000*(S[8]['saturation_db']-S[0]['saturation_db']):.0f}")

# ------------------------------------------------------------------- padding
print("padding ablation")
d256, _ = pick("ctc_seam_p256.json")
d128, _ = pick("ctc_seam_p128.json")
if d256:
    q = d256["qps"]; r = d256["res"]
    nice = {"zeros": "Zeros (stock)", "replicate": "Replicate",
            "linear": "Linear extrap.", "arls": "AR(1) per channel~\\cite{arls}"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Border estimator & " + " & ".join(f"$q{x}$" for x in q) + r" \\",
             r"\midrule"]
    for m in ("zeros", "replicate", "linear", "arls"):
        if m in r:
            bold = m == "replicate"
            vals = " & ".join((f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}")
                              for v in r[m])
            lines.append(f"{nice[m]} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("padding.tex", "\n".join(lines))
    mac("SeamZerosHigh", f"{r['zeros'][-1]:.3f}")
    mac("SeamReplHigh", f"{r['replicate'][-1]:.3f}")
    mac("SeamArlsHigh", f"{r['arls'][-1]:.3f}")
    mac("SeamLinearHigh", f"{r['linear'][-1]:.3f}")
if d256 and d128:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Estimator, $q63$ & 128\,px & 256\,px & ratio \\", r"\midrule"]
    for m in ("zeros", "replicate", "arls"):
        a, b = d128["res"][m][-1], d256["res"][m][-1]
        lines.append(f"{m} & {a:.3f} & {b:.3f} & $\\times${b/a:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tilesize.tex", "\n".join(lines))

# ------------------------------------------------------------------ latency
print("latency")
d, _ = pick("latency_RECIPE512_sorted.json", "latency_BEST.json")
if d:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"$q$ & Released & Routed & \multicolumn{2}{c}{Saved (\%)} \\",
             r"\cmidrule(lr){4-5}",
             r" & (ms) & sorted (ms) & measured & MACs \\", r"\midrule"]
    for r_ in d["rows"]:
        srt = r_.get("ms_routed_sorted")
        meas = r_.get("realised_saving_sorted_pct", r_["realised_saving_pct"])
        lines.append(f"{r_['qp']} & {r_['ms_stock']:.0f} & "
                     f"{(srt if srt else r_['ms_routed']):.0f} & "
                     f"\\textbf{{{meas:.1f}}} & {r_['predicted_saving_pct']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("latency.tex", "\n".join(lines))
    r0 = d["rows"][0]
    if r0.get("realised_saving_sorted_pct"):
        mac("WallMasked", f"{r0['realised_saving_pct']:.1f}")
        mac("WallSorted", f"{r0['realised_saving_sorted_pct']:.1f}")
        mac("WallPredicted", f"{r0['predicted_saving_pct']:.1f}")
        mac("TilingOverhead", f"{r0['overhead_pct']:.1f}")
        mac("WallSortedGain",
            f"{r0['realised_saving_sorted_pct'] - r0['realised_saving_pct']:.1f}")
        rhi = d["rows"][-1]
        mac("WallHighMeasured",
            f"{rhi.get('realised_saving_sorted_pct', rhi['realised_saving_pct']):.1f}")
        mac("WallHighPredicted", f"{rhi['predicted_saving_pct']:.1f}")

# -------------------------------------------------------------- run compare
print("run comparison")
CFG = {"RECIPE512": ("K6 j2 256px", 41.91),
       "BEST": ("K6 j2 256px", 41.91),
       "BEST128": ("K6 j2 128px", 41.91),
       "FINE12": ("K12 j4 128px", 50.29)}
rows = []
for tag in ("RECIPE512", "BEST", "BEST128", "FINE12"):
    d, _ = pick(f"signalled_{tag}_ctc53.json", f"signalled_{tag}_b135.json")
    if not d:
        continue
    means = {}
    for b in d["budgets"]:
        r = {x["qp"]: x for x in d["rows"]
             if abs(x["budget_db"] - b) < 1e-9 and x.get("budget_reachable")}
        vs = [r[q]["saving_pct_vs_release"] for q in QPS if q in r]
        means[b] = (sum(vs) / len(vs), len(vs)) if vs else (None, 0)
    rows.append((tag, d.get("ckpt_epoch"), means))
if rows:
    buds = sorted({b for _, _, m in rows for b in m})
    lines = [r"\begin{tabular}{llr" + "r" * len(buds) + "}", r"\toprule",
             r"Ladder & Config & Ceiling & " +
             " & ".join(f"{b:.1f}\\,dB" for b in buds) + r" \\",
             r"\midrule"]
    best_at = {b: max((m[b][0] or -1) for _, _, m in rows) for b in buds}
    for tag, ep, m in rows:
        cfgs, ceil = CFG.get(tag, ("--", 0))
        cells = []
        for b in buds:
            v, n = m.get(b, (None, 0))
            if v is None:
                cells.append("--")
            else:
                t = f"{v:.1f}" + ("" if n == len(QPS) else r"$^{\ast}$")
                cells.append(f"\\textbf{{{t}}}" if abs(v - best_at[b]) < 1e-9 else t)
        lines.append(f"{tag} & {cfgs} & {ceil:.1f} & " + " & ".join(cells) +
                     r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("runs.tex", "\n".join(lines))
    fm = dict(rows).get if False else None
    for tag, ep, m in rows:
        if tag == "FINE12":
            mac("FineHalfDb", f"{m.get(0.5,(0,0))[0]:.1f}")
        if tag == "RECIPE512":
            mac("CoarseHalfDb", f"{m.get(0.5,(0,0))[0]:.1f}")

# ------------------------------------------------------------- static baseline
print("static baseline")
d, _ = pick("static_RECIPE512_b01.json")
if d:
    B = d["budget_db"]
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"$q$ & Allocation & $\Delta$PSNR (dB) & MACs saved (\%) \\",
             r"\midrule"]
    means = {"static": [], "oracle": []}
    for row in d["rows"]:
        q, first = row["qp"], True
        for u in row["uniform"]:
            dag = "" if u["db"] <= B else r"$^{\dagger}$"
            lines.append((f"{q}" if first else "") +
                         f" & uniform, exit {u['exit']}{dag} & {u['db']:.3f} & "
                         f"{u['saving']:.1f} \\\\")
            first = False
        lines.append(f" & random (matched mix) & {row['random']['db']:.3f} & "
                     f"{row['random']['saving']:.1f} \\\\")
        rr = row.get("rate_rank")
        if rr:
            lines.append(f" & rate-ranked (free) & {rr['db']:.3f} & "
                         f"{rr['saving']:.1f} \\\\")
        o = row["oracle"]
        lines.append(f" & \\textbf{{oracle}} & \\textbf{{{o['db']:.3f}}} & "
                     f"\\textbf{{{o['saving']:.1f}}} \\\\")
        lines.append(r"\midrule")
        bs = row.get("best_static")
        means["static"].append(bs["saving"] if bs else 0.0)
        means["oracle"].append(o["saving"])
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    w("static.tex", "\n".join(lines))
    mac("BestStaticMean", f"{sum(means['static'])/len(means['static']):.1f}")
    r0 = d["rows"][-1]
    if r0.get("rate_rank"):
        mac("RateRankDb", f"{r0['rate_rank']['db']:.3f}")
        mac("RandomDb", f"{r0['random']['db']:.3f}")
        mac("OracleDb", f"{r0['oracle']['db']:.3f}")
        gap = r0["random"]["db"] - r0["oracle"]["db"]
        got = r0["random"]["db"] - r0["rate_rank"]["db"]
        mac("RateRankRecovers", f"{100*got/gap:.0f}")

# ---------------------------------------------------------- per resolution
print("per class")
d, _ = pick("per_class_RECIPE512.json")
if d:
    ORDER = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
    at = {}
    for r in d["rows"]:
        if abs(r["budget_db"] - 0.1) < 1e-9:
            at[r["qp"]] = r["per_class"]
    qs = sorted(at)
    if at:
        lines = [r"\begin{tabular}{llrr" + "r" * len(qs) + "}", r"\toprule",
                 r"Class & Resolution & Tiles & $n$ & " +
                 " & ".join(f"$q{q}$" for q in qs) + r" \\", r"\midrule"]
        for c in ORDER:
            if c not in at[qs[0]]:
                continue
            v = at[qs[0]][c]
            lines.append(f"{c.replace('_', chr(92)+'_')} & {v['res']} & "
                         f"{v['tiles']} & {v['n']} & " +
                         " & ".join(f"{at[q][c]['saving']:.1f}" for q in qs) +
                         r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        w("perclass.tex", "\n".join(lines))
        lo = at[qs[0]]
        mac("SmallResLow", f"{lo['HEVC_D']['saving']:.1f}")
        mac("BigResLow", f"{lo['MCL-JCV']['saving']:.1f}")
        mac("MidResLow", f"{lo['HEVC_C']['saving']:.1f}")
        hi = at[qs[-1]]
        mac("SmallResHigh", f"{hi['HEVC_D']['saving']:.1f}")
        mac("BigResHigh", f"{hi['MCL-JCV']['saving']:.1f}")

# ------------------------------------------------------------- complexity
print("complexity")
INTRA_GMAC = 453.5          # DMCI at 1080p, scripts/mac_audit.py
d, _ = pick("signalled_RECIPE512_ctc53.json", "signalled_RECIPE512_b135.json")
lat, _ = pick("latency_RECIPE512_sorted.json")
if d and lat:
    by = {b: {r["qp"]: r for r in d["rows"]
              if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
          for b in d["budgets"]}
    L = {r["qp"]: r for r in lat["rows"]}
    lq = sorted(L)[0]
    r01 = by[0.1]
    mean01 = sum(r01[q]["saving_pct_vs_release"] for q in QPS if q in r01) / \
        len([q for q in QPS if q in r01])
    rows = [
        ("Released DCVC-UF", INTRA_GMAC, 100.0,
         L[lq]["ms_stock"], 0.0),
        ("FLEX-UF, all tiles deepest", INTRA_GMAC * 1.0095, 100.95,
         L[lq]["ms_deep"], None),
        (f"FLEX-UF, {0.1:.1f}\\,dB budget",
         INTRA_GMAC * (1 - mean01 / 100), 100 - mean01,
         L[lq].get("ms_routed_sorted", L[lq]["ms_routed"]), None),
        ("FLEX-UF, architectural ceiling",
         INTRA_GMAC * (1 - 41.9114 / 100), 100 - 41.9114, None, None),
    ]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Decoder & GMAC & \% of release & ms \\", r"\midrule"]
    for name, gm, pc, ms, _ in rows:
        lines.append(f"{name} & {gm:.0f} & {pc:.1f} & "
                     + (f"{ms:.0f}" if ms else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("complexity.tex", "\n".join(lines))
    mac("IntraGmac", f"{INTRA_GMAC:.1f}")
    mac("GmacAtBudget", f"{INTRA_GMAC*(1-mean01/100):.0f}")

# ------------------------------------------------------------------ A vs B
print("A versus B")
sa, _ = pick("signalled_RECIPE512_ctc53.json")
b1, _ = pick("router_RECIPE512_b01_fixed.json", "router_RECIPE512_b01.json")
b3, _ = pick("router_RECIPE512_b03_fixed.json", "router_RECIPE512_b03.json")
if sa and b1:
    def Aat(bud):
        return {r["qp"]: r["saving_pct_vs_release"] for r in sa["rows"]
                if abs(r["budget_db"] - bud) < 1e-9 and r.get("budget_reachable")}

    def Bat(d_):
        return {r["qp"]: r.get("saving_pct_vs_release") for r in (d_ or {}).get("rows", [])
                if r.get("budget_reachable")}
    A1, B1 = Aat(0.1), Bat(b1)
    A3, B3 = Aat(0.3), Bat(b3)
    qs = [q for q in QPS if q in A1 and B1.get(q) is not None]
    if qs:
        both = b3 is not None and all(q in A3 and B3.get(q) is not None for q in qs)
        head = (r"& \multicolumn{3}{c}{0.1\,dB}" +
                (r" & \multicolumn{3}{c}{0.3\,dB}" if both else "") + r" \\")
        sub_ = (r"$q$ & A & B & gap" + (r" & A & B & gap" if both else "") + r" \\")
        lines = [r"\begin{tabular}{l" + "rrr" * (2 if both else 1) + "}",
                 r"\toprule", head, sub_, r"\midrule"]
        for q in qs:
            row = [f"{q}", f"{A1[q]:.1f}", f"{B1[q]:.1f}",
                   f"\\textbf{{{A1[q]-B1[q]:.1f}}}"]
            if both:
                row += [f"{A3[q]:.1f}", f"{B3[q]:.1f}",
                        f"\\textbf{{{A3[q]-B3[q]:.1f}}}"]
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        w("ab.tex", "\n".join(lines))
        mac("GapLow", f"{A1[qs[0]]-B1[qs[0]]:.1f}")
        mac("GapHigh", f"{A1[qs[-1]]-B1[qs[-1]]:.1f}")
        mac("BLow", f"{B1[qs[0]]:.1f}")
        mac("BHigh", f"{B1[qs[-1]]:.1f}")
        gaps = {q: A1[q] - B1[q] for q in qs}
        mac("GapMin", f"{min(gaps.values()):.1f}")
        mac("GapMax", f"{max(gaps.values()):.1f}")
        mac("GapMinQp", str(min(gaps, key=gaps.get)))
        betas = {r["qp"]: r.get("beta") for r in (b1 or {}).get("rows", [])
                 if r.get("beta") is not None}
        if betas:
            mac("BetaMinQp", str(min(betas, key=lambda q: abs(betas[q]))))
            mac("BetaAtMin", f"{betas[min(betas, key=lambda q: abs(betas[q]))]:.0f}")
            mac("BetaLow", f"{betas[qs[0]]:.0f}")
        if both:
            mac("SigLooseHigh", f"{A3[qs[-1]]:.1f}")
            mac("SigLooseLow", f"{A3[qs[0]]:.1f}")
            mac("GapLooseLow", f"{A3[qs[0]]-B3[qs[0]]:.1f}")
            mac("GapLooseHigh", f"{A3[qs[-1]]-B3[qs[-1]]:.1f}")

# ------------------------------------------------------------------ hybrid C
print("hybrid C")
hy, _ = pick("hybrid_RECIPE512_b01_fixed.json", "hybrid_RECIPE512_b01.json")
if hy:
    rows_ = [r for r in hy["rows"] if r.get("budget_reachable")]
    qs = [q for q in QPS if any(r["qp"] == q for r in rows_)]
    rhos = sorted({r["rho"] for r in rows_})
    show = [r for r in rhos if r in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)] or rhos

    def cell(q, r_):
        for r in rows_:
            if r["qp"] == q and abs(r["rho"] - r_) < 1e-9:
                return r
        return None

    lines = [r"\begin{tabular}{l" + "r" * len(show) + "}", r"\toprule",
             r"$q$ & " + " & ".join(
                 ("B" if r_ == 0 else "A" if r_ == 1 else f"{100*r_:.0f}\\%")
                 for r_ in show) + r" \\",
             r"bits/frame & " + " & ".join(
                 f"{(cell(qs[0], r_) or {}).get('map_bits', 0):.0f}"
                 for r_ in show) + r" \\", r"\midrule"]
    for q in qs:
        vals = [cell(q, r_) for r_ in show]
        lines.append(f"{q} & " + " & ".join(
            f"{v['saving_pct_vs_release']:.1f}" if v else "---"
            for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("hybrid.tex", "\n".join(lines))
    def recov(q, r_):
        b0, bx, ba = cell(q, 0.0), cell(q, r_), cell(q, 1.0)
        if not (b0 and bx and ba):
            return None
        span = ba["saving_pct_vs_release"] - b0["saving_pct_vs_release"]
        if span <= 1e-9:
            return None
        return 100 * (bx["saving_pct_vs_release"]
                      - b0["saving_pct_vs_release"]) / span

    qlo, qhi = qs[0], qs[-1]
    for tag, q in (("Low", qlo), ("High", qhi)):
        for rt, r_ in (("Tenth", 0.1), ("Fifth", 0.2), ("Half", 0.5)):
            v = recov(q, r_)
            if v is not None:
                mac(f"HybridRecover{rt}{tag}", f"{v:.0f}")
            c_ = cell(q, r_)
            if c_ and tag == "High":
                mac(f"HybridSave{rt}High", f"{c_['saving_pct_vs_release']:.1f}")
    for rt, r_ in (("Tenth", 0.1), ("Fifth", 0.2), ("Full", 1.0)):
        c_ = cell(qhi, r_)
        if c_:
            mac(f"HybridBits{rt}", f"{c_['map_bits']:.0f}")
    # The hybrid file carries its own Lorenz statistics now; the separate
    # lorenz run is only a fallback for files written before that.
    lo_ = hy if any(r.get("gini_regret") is not None for r in rows_) else None
    if lo_ is None:
        lo_, _ = pick("hybrid_lorenz_b01_fixed.json", "hybrid_lorenz_b01.json")
    if lo_:
        lrows = [r for r in lo_["rows"] if r.get("budget_reachable")]

        def gini(q):
            return next((r["gini_regret"] for r in lrows
                         if r["qp"] == q and r.get("gini_regret") is not None),
                        None)
        for tag, q in (("Low", qlo), ("High", qhi)):
            g = gini(q)
            if g is not None:
                mac(f"Gini{tag}", f"{g:.2f}")
        gs = [gini(q) for q in qs if gini(q) is not None]
        if gs:
            mac("GiniMin", f"{min(gs):.2f}")
            mac("GiniMax", f"{max(gs):.2f}")
        # how tight the Lorenz bound is: measured recovery over the bound
        ratios = []
        for r in lrows:
            if r.get("lorenz_at_rho") in (None, 0):
                continue
            m = recov(r["qp"], r["rho"])
            if m is None:
                continue
            ratios.append(m / (100 * r["lorenz_at_rho"]))
        if ratios:
            mac("LorenzTightMin", f"{100*min(ratios):.0f}")
            mac("LorenzTightMax", f"{100*min(1.0, max(ratios)):.0f}")

# --------------------------------------------------------------- transfer
print("map transfer")
mt, _ = pick("map_transfer.json")
if mt:
    tr = [r for r in mt["rows"] if r.get("kind") == "time"]
    q0 = [r for r in tr if r["qp"] == 0]
    if q0:
        far = max(q0, key=lambda r: r["offset"])
        mac("TransferOffset", str(far["offset"]))
        mac("TransferDbCost",
            f"{far['transfer_db'] - q0[0]['in_place_db']:.3f}")
        mac("TransferSaving", f"{far['transfer_saving']:.2f}")
        mac("TransferInPlaceLo",
            f"{min(r['in_place_saving'] for r in q0):.2f}")
        mac("TransferInPlaceHi",
            f"{max(r['in_place_saving'] for r in q0):.2f}")
    rr_ = [r for r in mt["rows"] if r.get("kind") == "rate"]
    cross = next((r for r in rr_ if r.get("from") == 0 and r.get("to") == 63),
                 None)
    if cross:
        mac("TransferCrossDb", f"{cross['transfer_db']:.3f}")

# --------------------------------------------------------------- coupling
print("coupling")
cp, _ = pick("coupling_ablation.json")
if cp:
    rows_ = cp["rows"]
    lo, hi = rows_[0], rows_[-1]
    for tag, r in (("Low", lo), ("High", hi)):
        mac(f"CoupPadded{tag}", f"{r['padded']['saving']:.1f}")
        mac(f"CoupCoupled{tag}", f"{r['coupled']['saving']:.1f}")
        mac(f"CoupFloorDrop{tag}",
            f"{100*(1 - r['coupled']['floor_db']/r['padded']['floor_db']):.0f}")
    mid = next((r for r in rows_ if r["qp"] == 32), None)
    if mid:
        mac("CoupPaddedMid", f"{mid['padded']['saving']:.1f}")
        mac("CoupCoupledMid", f"{mid['coupled']['saving']:.1f}")
    drops = [100*(1 - r['coupled']['floor_db']/r['padded']['floor_db'])
             for r in rows_]
    mac("CoupFloorDropLo", f"{min(drops):.0f}")
    mac("CoupFloorDropHi", f"{max(drops):.0f}")

# ------------------------------------------------------------------- blend
print("blend")
cb, _ = pick("combined_RECIPE512_b01.json")
if cb:
    rws = [r for r in cb["rows"] if r.get("budget_reachable")]
    ws = sorted({r["gamma"] for r in rws})
    qs = [q for q in QPS if any(r["qp"] == q for r in rws)]

    def cell(q, w_):
        return next((r for r in rws
                     if r["qp"] == q and abs(r["gamma"] - w_) < 1e-9), None)

    lines = [r"\begin{tabular}{l" + "r" * len(ws) + "}", r"\toprule",
             r"$q$ & " + " & ".join(
                 ("bits" if w_ == 0 else "head" if w_ == 1 else f"{w_:g}")
                 for w_ in ws) + r" \\", r"\midrule"]
    for q in qs:
        vals = [(cell(q, w_) or {}).get("saving_pct_vs_release") for w_ in ws]
        best = max((v for v in vals if v is not None), default=None)
        lines.append(f"{q} & " + " & ".join(
            "---" if v is None else
            (f"\\textbf{{{v:.1f}}}" if v == best else f"{v:.1f}")
            for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("blend.tex", "\n".join(lines))
    for tag, q in (("Low", qs[0]), ("High", qs[-1])):
        c0, c1 = cell(q, 0.0), cell(q, 1.0)
        vals = [(w_, cell(q, w_)) for w_ in ws]
        vals = [(w_, c) for w_, c in vals if c]
        bw, bc = max(vals, key=lambda t: t[1]["saving_pct_vs_release"])
        mac(f"Blend{tag}Best", f"{bc['saving_pct_vs_release']:.1f}")
        mac(f"Blend{tag}BestW", f"{bw:g}")
        if c1:
            mac(f"BlendHeadOnly{tag}", f"{c1['saving_pct_vs_release']:.1f}")
        if c0:
            mac(f"BlendBitsOnly{tag}", f"{c0['saving_pct_vs_release']:.1f}")

# ---------------------------------------------------------------- BD-Rate
print("BD-Rate")
bdj, _ = pick("bdrate.json")
if bdj:
    def _bd(cfg, bud):
        return next((r["bd_rate_pct"] for r in bdj["rows"]
                     if r["config"].startswith(cfg)
                     and abs(r["budget_db"] - bud) < 1e-9), None)
    for tag, cfg in (("A", "A"), ("B", "B")):
        for bt, bud in (("Low", 0.1), ("Mid", 0.3), ("High", 0.5)):
            v = _bd(cfg, bud)
            if v is not None:
                mac(f"BdRate{tag}{bt}", f"{v:.2f}")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"configuration & budget & BD-Rate (\%) & saved (\%) & bits/frame \\",
             r"\midrule"]
    for r in bdj["rows"]:
        lines.append(f"{r['config']} & {r['budget_db']:.1f}\\,dB & "
                     f"{r['bd_rate_pct']:.2f} & "
                     f"{r['saving_pct_vs_release']:.1f} & "
                     f"{r['map_bits']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("bdrate.tex", "\n".join(lines))

# ------------------------------------------------------------- rate rank
print("rate rank")
rr, _ = pick("raterank_RECIPE512_b01.json")
rr3, _ = pick("raterank_RECIPE512_b03.json")
if rr3:
    rs3 = [r for r in rr3["rows"] if r.get("budget_reachable")]
    if rs3:
        mac("RateRankLoose", f"{rs3[-1]['saving_pct_vs_release']:.1f}")
        mac("RateRankLooseCeil",
            f"{sum(1 for r in rs3 if r['saving_pct_vs_release'] > 41.9)}")
if rr:
    rs = [r for r in rr["rows"] if r.get("budget_reachable")]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"$q$ & rate-rank & B router & A signalled & "
             r"$\rho$ depth & $\rho$ spread \\",
             r"\midrule"]
    for r in rs:
        b = (B1 or {}).get(r["qp"])
        beats = b is not None and r["saving_pct_vs_release"] > b
        v = (f"\\textbf{{{r['saving_pct_vs_release']:.1f}}}" if beats
             else f"{r['saving_pct_vs_release']:.1f}")
        lines.append(f"{r['qp']} & {v} & "
                     + (f"{b:.1f}" if b is not None else "---")
                     + f" & {r['oracle_saving_pct_vs_release']:.1f} & "
                     f"{-r['spearman_bits_vs_exit']:+.2f} & "
                     f"{r['spearman_bits_vs_spread']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("raterank.tex", "\n".join(lines))
    if rs:
        mac("RateRankLow", f"{rs[0]['saving_pct_vs_release']:.1f}")
        mac("RateRankHigh", f"{rs[-1]['saving_pct_vs_release']:.1f}")
        if B1:
            d_ = [(r["qp"], r["saving_pct_vs_release"] - B1[r["qp"]])
                  for r in rs if r["qp"] in B1]
            win = [q for q, v in d_ if v > 0]
            if win:
                mac("RateRankBeatsFrom", str(min(win)))
                mac("RateRankBeatsUpTo", str(max(win)))
                mac("RateRankNWins", str(len(win)))
                mac("RateRankBeatsBy", f"{max(v for _, v in d_):.1f}")
            if any(v <= 0 for _, v in d_):
                mac("RateRankLosesBy", f"{max(-v for _, v in d_ if v <= 0):.1f}")
                mac("RateRankLosesFrom",
                    str(min(q for q, v in d_ if v <= 0)))
        mac("RateRankSpreadLo",
            f"{min(r['spearman_bits_vs_spread'] for r in rs):.2f}")
        mac("RateRankSpreadHi",
            f"{max(r['spearman_bits_vs_spread'] for r in rs):.2f}")
        mac("RateRankAgreeLo", f"{min(r['agreement'] for r in rs):.2f}")
        mac("RateRankAgreeHi", f"{max(r['agreement'] for r in rs):.2f}")

# --------------------------------------------------------- adapter ablation
print("adapter ablation")
d, _ = pick("adapter_ablation.json")
if d:
    qs = [r["qp"] for r in d["rows"]]
    exits = sorted(int(e) for e in d["rows"][0]["trained"])
    lines = [r"\begin{tabular}{l" + "rr" * len(qs) + "}", r"\toprule",
             r"& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$q{q}$}}" for q in qs)
             + r" \\",
             r"Exit & " + " & ".join(["with & without"] * len(qs)) + r" \\",
             r"\midrule"]
    for e in exits:
        cells = []
        for r in d["rows"]:
            t, o = r["trained"][str(e)], r["all_off"][str(e)]
            cells += [f"{t:.3f}", f"{o:.3f}"]
        tag = r" $^{\dagger}$" if e == exits[-1] else ""
        lines.append(f"{e}{tag} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("adapters_ablation.tex", "\n".join(lines))
    r63 = next((r for r in d["rows"] if r["qp"] == 63), d["rows"][-1])
    e0 = str(exits[0])
    mac("AdapterGainHigh", f"{r63['all_off'][e0] - r63['trained'][e0]:.2f}")
    mac("AdapterNoneHigh", f"{r63['all_off'][e0]:.2f}")
    r0 = d["rows"][0]
    mac("AdapterGainLow", f"{r0['all_off'][e0] - r0['trained'][e0]:.2f}")

# ------------------------------------------------------------- positioning
print("positioning")
INTRA_GMAC_ = 453.5
PX = 1920 * 1080
d, _ = pick("signalled_RECIPE512_ctc53.json")
if d:
    by = {r["qp"]: r for r in d["rows"]
          if abs(r["budget_db"] - 0.1) < 1e-9 and r.get("budget_reachable")}
    mean01 = sum(by[q]["saving_pct_vs_release"] for q in QPS if q in by) / \
        len([q for q in QPS if q in by])
    ours_kmac = INTRA_GMAC_ * (1 - mean01 / 100) * 1e9 / PX / 1e3
    rel_kmac = INTRA_GMAC_ * 1e9 / PX / 1e3
    rows = [
        ("SlimCAE~\\cite{slimcae}", "width", "per stream", "yes"),
        ("Slimmable video~\\cite{slimvc}", "width", "per stream", "yes"),
        ("EVC~\\cite{evc}", "mask / pruning", "per model", "yes"),
        ("Spatial competition~\\cite{spatialcompetition}", "which codec",
         "per region", "yes"),
        ("DCVC-RT~\\cite{dcvcrt}", "architecture", "fixed", "yes"),
        ("\\textbf{FLEX-UF}", "\\textbf{decoder depth}",
         "\\textbf{per region}", "\\textbf{no}"),
    ]
    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"Method & What varies & Granularity & New bitstream \\",
             r"\midrule"]
    for a_, b_, c_, e_ in rows:
        lines.append(f"{a_} & {b_} & {c_} & {e_} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("positioning.tex", "\n".join(lines))
    mac("OursKMacPx", f"{ours_kmac:.0f}")
    mac("RelKMacPx", f"{rel_kmac:.0f}")

# how many claims check_paper.py re-reads out of the prose; it records its own
# count so this file does not have to run it
_cc, _ = pick("check_paper.json")
if _cc:
    mac("NumClaims", str(_cc["n_claims"]))

# ------------------------------------------------------------------ macros
w("macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                          for k, v in sorted(MACROS.items())))
print(f"\n  {len(MACROS)} macros")
