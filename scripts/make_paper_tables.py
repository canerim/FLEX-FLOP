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
    mac("MapBits", f"{r01[0]['map_bits']:.0f}")
    mac("MapOverheadLow", f"{100*r01[0]['bpp_added']/0.40:.3f}")

# --------------------------------------------------------- operating structure
print("operating structure")
d, _ = pick("saturation_RECIPE512_ctc53.json", "saturation_RECIPE512.json")
if d:
    S = {r["qp"]: r for r in d["rows"]}
    qs = sorted(S)
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

# -------------------------------------------------------------- run compare
print("run comparison")
CFG = {"RECIPE512": ("$K{=}6$, $j{=}2$, 256\\,px", 41.91),
       "BEST": ("$K{=}6$, $j{=}2$, 256\\,px", 41.91),
       "BEST128": ("$K{=}6$, $j{=}2$, 128\\,px", 41.91),
       "FINE12": ("$K{=}12$, $j{=}4$, 128\\,px", 50.29)}
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

# ---------------------------------------------------------- per resolution
print("per dataset")
d, _ = pick("curve_RECIPE512_ctc53.json")
if d and d.get("op_points"):
    import sys as _s
    _s.path.insert(0, str(Path.home() / "DCVC")); _s.path.insert(0, str(R))
    import ctc_intra as _C
    seqs, _m = _C.discover([])
    FAM = {x["name"]: x["cls"] for x in seqs}
    RESO = {"UVG": "1920$\\times$1080", "MCL-JCV": "1920$\\times$1080",
           "HEVC\\_B": "1920$\\times$1080", "HEVC\\_E": "1280$\\times$720",
           "HEVC\\_C": "832$\\times$480", "HEVC\\_D": "416$\\times$240"}
    TILES = {"1920$\\times$1080": 40, "1280$\\times$720": 18,
             "832$\\times$480": 8, "416$\\times$240": 2}
    D = 1.0095
    per, realised = {}, {}
    for q in (0, 63):
        ops = [o for o in d["op_points"]
               if o["qp"] == q and o.get("per_sequence")]
        if not ops:
            continue
        o = min(ops, key=lambda o: abs(o.get("target_db", 9) - 0.1))
        realised[q] = o.get("db_vs_uf_per_frame")
        for sq in o["per_sequence"]:
            per.setdefault(FAM.get(sq["seq"], "?"), {}).setdefault(q, []).append(
                100 - (100 - sq["saving_pct"]) * D)
    order = ["UVG", "MCL-JCV", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
    if per:
        lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                 r"Class & Resolution & Tiles & $q0$ & $q63$ \\", r"\midrule"]
        for f in order:
            if f not in per:
                continue
            esc = f.replace("_", r"\_")
            res = RESO.get(esc, "--")
            a = sum(per[f].get(0, [0])) / max(1, len(per[f].get(0, [1])))
            b = sum(per[f].get(63, [0])) / max(1, len(per[f].get(63, [1])))
            lines.append(f"{esc} & {res} & {TILES.get(res,'--')} & "
                         f"{a:.1f} & {b:.1f} \\\\")
        mac("SmallResHigh", f"{sum(per['HEVC_D'].get(63,[0]))/max(1,len(per['HEVC_D'].get(63,[1]))):.1f}")
        mac("BigResHigh", f"{sum(per['MCL-JCV'].get(63,[0]))/max(1,len(per['MCL-JCV'].get(63,[1]))):.1f}")
        mac("PerClassDb", f"{realised.get(63, 0):.3f}")
        lines += [r"\bottomrule", r"\end{tabular}"]
        w("perclass.tex", "\n".join(lines))

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
        ("Released DCVC-UF~\\cite{dcvcuf}", INTRA_GMAC, 100.0,
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

# ------------------------------------------------------------------ macros
w("macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                          for k, v in sorted(MACROS.items())))
print(f"\n  {len(MACROS)} macros")
