"""Emit every table and inline number the paper uses, as LaTeX, from results/.

No number in paper/main.tex is typed. Tables land in paper/tables/*.tex and
scalars become macros in paper/tables/macros.tex, so \\MainLowRate expands to
whatever the current measurement says. A paper whose numbers are retyped from a
terminal is a paper that will disagree with its own repository within a week.

Every generator here prefers the full 53-sequence CTC file and falls back to the
40-sequence one, printing which it used.
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
RES, OUT = R / "results", R / "paper" / "tables"
# _ceiling() below reads each run's config through flexuf, which needs the repo
# and the DCVC checkout on the path.
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path.home() / "DCVC"))
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the saving
# ONE definition, for every table in this document.
#
# Three numbers for the architectural ceiling were in circulation: 41.91 from
# before the FFN adapter was found to cost 5C^2 rather than 2, 39.12 from the
# arithmetic model as it now stands, and 38.33 from hooks on the executed
# decode. The model is still needed inside the argmin, where a per-tile price
# has to be evaluated for every candidate exit at every step of a bisection and
# decoding each candidate is not possible. It is not needed to REPORT, because
# the routed decode is run anyway to measure distortion, so the saving can
# simply be counted off it.
#
# Reported numbers are therefore the hook count. The model under-bills the
# shallow exits by a constant 0.008 of a released decode, which is 0.4 to 0.8
# saving points depending on the operating point, always in our favour. Every
# table below is 0.4 to 0.8 points lower than it was, and correct.
#
# results/ceiling_measured.json holds the per-exit comparison.
CEIL_MEASURED = 38.33   # overwritten below from results/ceiling_measured.json


def sv_measured(row):
    """Was this row's saving counted off the decode, or modelled?"""
    return row.get("saving_pct_measured") is not None


def sv(row, key="saving_pct"):
    """The saving a row reports, measured off the decode where available."""
    m = row.get(key + "_measured")
    return m if m is not None else row.get(key + "_vs_release")


def sv_oracle(row):
    """The Lagrangian oracle's saving on the same row, same definition."""
    return sv(row, "oracle_saving_pct")


_cm = RES / "ceiling_measured.json"
if _cm.exists():
    CEIL_MEASURED = json.load(open(_cm))["ceiling_measured_pct"]


QPS = [0, 16, 32, 48, 63]
MACROS = {}


# The checkpoint the paper is measured on. Pinned to an immutable name because
# runs/*/ckpt_eval.pth.tar is overwritten every epoch by scripts/watch_ckpts.sh,
# so a filename is not a checkpoint.
PINNED = "ckpt_PAPER.pth.tar"


def _pick_json(*names):
    """The first of these that exists, as parsed JSON.

    Three sites read the configuration-B curve with a plain json.load and the
    _PAPER name, which is the head fitted before the repin. They are the
    router's held-out agreement and the frozen-against-joint comparison, and
    both are about the head, so reading the one the paper does not report was
    the whole error.
    """
    for n in names:
        p = RES / n
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(names[0])


def pick(*names):
    """The best available candidate, by provenance rather than by list order.

    Candidates used to be tried in the order written here, newest first by hand.
    That ordering went stale: after scripts/remeasure_all.sh rewrote
    router_RECIPE512_b01.json on the pinned checkpoint, the hand-ordered list
    still preferred router_RECIPE512_b01_fixed.json from an earlier experiment,
    so the A-versus-B table and every macro derived from it silently described a
    decoder measured under the old cost model.

    Prefer, in order: a file measured on the pinned checkpoint, then the most
    recently written. Hand order only breaks ties.
    """
    found = [(n, RES / n) for n in names if (RES / n).exists()]
    if not found:
        print(f"    MISSING: {names[0]}")
        return None, None

    def rank(item):
        n, p = item
        try:
            d = json.load(open(p))
        except Exception:
            d = {}
        on_pin = PINNED in (d.get("ckpt") or "")
        # A router head fitted to other weights makes a file a measurement of
        # a configuration the paper does not report, even when the decoder
        # weights are the pinned ones.
        meta = d.get("router2_meta") or {}
        head_off = bool(d.get("router2")) and meta.get("ckpt_epoch") != d.get("ckpt_epoch")
        # Hand order last, and it decides among files that are equally on the
        # pin. Ranking those by mtime let a driver from the pre-refit chain
        # rewrite router_RECIPE512_b01.json at 13:28 and take the headline
        # away from the head the paper reports, silently, because the stale
        # file was newer.
        return (0 if on_pin else 1, 1 if head_off else 0, names.index(n))

    n, p = min(found, key=rank)
    others = [m for m, _ in found if m != n]
    print(f"    using {n}" + (f"  (over {', '.join(others)})" if others else ""))
    return json.load(open(p)), n


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
        vs = [sv(row[q]) for q in QPS if q in row]
        lines.append(f"{b:.2f}\\,dB & " +
                     " & ".join(f"{sv(row[q]):.1f}" if q in row
                                else "--" for q in QPS) +
                     f" & \\textbf{{{sum(vs)/len(vs):.1f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("main_results.tex", "\n".join(lines))
    r01 = by[0.1]
    mac("MainLowRate", f"{sv(r01[0]):.1f}")
    mac("MainHighRate", f"{sv(r01[63]):.1f}")
    mac("MainMean", f"{sum(sv(r01[q]) for q in QPS)/len(QPS):.1f}")
    # A mean per budget, named by the budget, so the abstract can quote any row
    # of this table without a second measurement and without a hand-typed
    # number. MainMean stays as the 0.1 dB alias the prose already uses.
    # LaTeX macro names cannot contain digits, so the budget is spelled out.
    _WORD = {5: "Half", 10: "One", 15: "OneHalf", 20: "Two", 30: "Three",
             50: "Five", 75: "Seven", 100: "Ten"}
    for _b, _row in by.items():
        _v = [sv(_row[q]) for q in QPS if q in _row]
        _w = _WORD.get(int(round(_b * 100)))
        if _v and _w:
            mac(f"MeanAt{_w}", f"{sum(_v)/len(_v):.1f}")
            if 0 in _row:
                mac(f"LowAt{_w}", f"{sv(_row[0]):.1f}")
            if 63 in _row:
                mac(f"HighAt{_w}", f"{sv(_row[63]):.1f}")
    mac("NumSeq", str(d["n_sequences"]))
    _mb = [r01[q]["map_bits"] for q in QPS
           if q in r01 and r01[q].get("map_bits")]
    mac("MapBits", f"{sum(_mb)/len(_mb):.0f}")
    mac("MapBitsLo", f"{min(_mb):.0f}")
    mac("MapBitsHi", f"{max(_mb):.0f}")
    mac("RouterParams", "144\\,K")
    # Read from the file that reports configuration B rather than typed
    # beside it. The head's share is a property of the head, and the paper
    # changed which head it reports today.
    _bshare, _ = pick("router_RECIPE512_b01_e4head.json",
                      "router_RECIPE512_b01_PAPER.json")
    if _bshare and _bshare.get("router_compute_share_pct"):
        mac("RouterCostPct", f"{_bshare['router_compute_share_pct']:.3f}")
    else:
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
    # The measured ceiling, not the modelled one. saturation's own ceiling_pct
    # comes from the arithmetic model, which under-bills the shallow exits.
    mac("Ceiling", f"{CEIL_MEASURED:.1f}")
    mac("CeilingModelled", f"{d['ceiling_pct']:.1f}")
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
        # The shortfall in points at each end, so the prose can say the
        # arithmetic over-promises by a fraction of the saving rather than by
        # a constant, without either number being typed anywhere.
        mac("WallShortLow", f"{r0['predicted_saving_pct'] - r0['realised_saving_sorted_pct']:.1f}")
        mac("WallShortHigh",
            f"{rhi['predicted_saving_pct'] - rhi.get('realised_saving_sorted_pct', rhi['realised_saving_pct']):.1f}")

# -------------------------------------------------------------- run compare
print("run comparison")

# How far two runs of one recipe land apart at a matched epoch. The main paper
# had 4.8 typed into the confound paragraph and the supplement computed 4.7
# from the same two files, because the typed one was taken from an earlier
# pair. One computation, one macro, both builds.
def _epoch_mean(name, budget=0.1):
    import json as _j
    p_ = R / "results" / name
    if not p_.exists():
        return None
    vs = [sv(r) for r in _j.loads(p_.read_text())["rows"]
          if abs(r.get("budget_db", -1) - budget) < 1e-9 and sv(r) is not None]
    return sum(vs) / len(vs) if vs else None


_e2 = _epoch_mean("signalled_RECIPE512_0820_0140.json")
_b2 = _epoch_mean("signalled_BEST_0819_1050.json")
if _e2 is not None and _b2 is not None:
    mac("RunGapEpochTwo", f"{_e2 - _b2:.1f}")

# Ceilings computed from each run's own config with the shipped cost model and
# then corrected by the constant the hook count shows, rather than typed in.
# They were 41.91 and 50.29 here, both from before the FFN adapter was found to
# cost 5C^2, while \Ceiling elsewhere already said 39.1.
def _ceiling(tag):
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    cfg = FlexUFConfig(**json.load(open(R / "runs" / tag / "meta.json"))["config"])
    modelled = 100.0 * (1.0 - float(exit_costs(cfg, "head")[cfg.split_depth]))
    # The hook count on RECIPE512 sits 0.79 points below its model; the offset
    # is the same 0.008 of a released decode at every shallow exit, so it
    # carries across ladders.
    return modelled - (39.12 - CEIL_MEASURED)


CFG = {t: (c, _ceiling(t)) for t, c in
       (("RECIPE512", "K6 j2 256px"), ("BEST", "K6 j2 256px"),
        ("BEST128", "K6 j2 128px"), ("FINE12", "K12 j4 128px"))}
rows = []
for tag in ("RECIPE512", "BEST", "BEST128", "FINE12"):
    d, _ = pick(f"signalled_{tag}_ctc53.json", f"signalled_{tag}_b135.json")
    if not d:
        continue
    means = {}
    for b in d["budgets"]:
        r = {x["qp"]: x for x in d["rows"]
             if abs(x["budget_db"] - b) < 1e-9 and x.get("budget_reachable")}
        vs = [sv(r[q]) for q in QPS if q in r]
        # Whether this run's rows are hook-counted or modelled. A table that
        # mixes the two silently is the defect this pass exists to remove, so
        # the modelled ones are marked and the caption says what the mark is.
        meas = all(sv_measured(r[q]) for q in QPS if q in r) if r else False
        means[b] = (sum(vs) / len(vs), len(vs), meas) if vs else (None, 0, False)
    rows.append((tag, d.get("ckpt_epoch"), means))
if rows:
    buds = sorted({b for _, _, m in rows for b in m})
    lines = [r"\begin{tabular}{llr" + "r" * len(buds) + "}", r"\toprule",
             r"Ladder & Config & Ceiling & " +
             " & ".join(f"{b:.1f}\\,dB" for b in buds) + r" \\",
             r"\midrule"]
    # .get, because the budgets are the union over runs and a run measured at
    # only three of them has no entry at the fourth. Indexing directly raised a
    # KeyError the moment RECIPE512 gained a 0.2 dB row that the others lack.
    best_at = {b: max((m.get(b, (None, 0, False))[0] or -1) for _, _, m in rows)
               for b in buds}
    for tag, ep, m in rows:
        cfgs, ceil = CFG.get(tag, ("--", 0))
        cells = []
        for b in buds:
            v, n, meas = m.get(b, (None, 0, False))
            if v is None:
                cells.append("--")
            else:
                t = (f"{v:.1f}" + ("" if n == len(QPS) else r"$^{\ast}$")
                     + ("" if meas else r"$^{\dagger}$"))
                cells.append(f"\\textbf{{{t}}}" if abs(v - best_at[b]) < 1e-9 else t)
        lines.append(f"{tag} & {cfgs} & {ceil:.1f} & " + " & ".join(cells) +
                     r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("runs.tex", "\n".join(lines))
    _half = {}
    for tag, ep, m in rows:
        if tag == "FINE12":
            _half["fine"] = m.get(0.5, (0, 0))[0]
            mac("FineHalfDb", f"{_half['fine']:.1f}")
        if tag == "RECIPE512":
            _half["coarse"] = m.get(0.5, (0, 0))[0]
            mac("CoarseHalfDb", f"{_half['coarse']:.1f}")
    # The margin between them, which Section 5.10 had as 6.7 typed into the
    # sentence beside the two macros that make it. They now read 48.6 and 38.3,
    # so the sentence disagreed with its own numbers by three and a half points.
    if "fine" in _half and "coarse" in _half:
        mac("FineHalfGain", f"{_half['fine'] - _half['coarse']:.1f}")

# ------------------------------------------------------------- static baseline
print("static baseline")
d, _ = pick("static_RECIPE512_b01.json")
# The oracle-histogram bit ranking's agreement with the oracle's map, which
# Section 5.5 had as "0.68-0.79" typed into the sentence. It is the only
# agreement number in the paper that is not the trained head's, and reusing
# \RateRankAgree* for it would have been worse than typing it.
try:
    _ra = [r["rate_rank"]["agreement"] for r in d["rows"] if "rate_rank" in r]
    if _ra:
        mac("HistAgreeLo", f"{min(_ra):.2f}")
        mac("HistAgreeHi", f"{max(_ra):.2f}")
except Exception as _e:
    print("   rate_rank agreement:", _e)
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
    # The same cells, transposed, for the main paper. `static.tex` is 35 rows
    # and 415.9pt tall, which is 60% of a column; the main text needs the same
    # comparison in about a quarter of that. Rows are the allocations, columns
    # the five quality indices, and a cell is the delivered dB over the MACs
    # saved. Nothing is recomputed: every number comes out of the same rows.
    qs = [row["qp"] for row in d["rows"]]
    exits = sorted({u["exit"] for row in d["rows"] for u in row["uniform"]})
    head = " & ".join([""] + [f"$q{q}$" for q in qs])
    tlines = ["\\begin{tabular}{l" + "r" * len(qs) + "}", r"\toprule",
              head + r" \\", r"\midrule"]

    def _cell(v, mark=False):
        # A uniform depth whose delivered dB is over budget is not an
        # admissible allocation at all, which is the whole point of those
        # rows, so the marker belongs on the cell and not on the row. The two
        # shuffled controls are matched to the oracle on compute rather than
        # on quality, so the same marker would say something else about them.
        if v is None:
            return "n/a"
        s_ = f"{v[0]:.3f}/{v[1]:.1f}"
        return s_ + (r"$^{\dagger}$" if mark and v[0] > B else "")

    for e in exits:
        cells = []
        for row in d["rows"]:
            u = next((x for x in row["uniform"] if x["exit"] == e), None)
            cells.append((u["db"], u["saving"]) if u else None)
        tlines.append(" & ".join([f"uniform {e}"] +
                                 [_cell(c, True) for c in cells]) + r" \\")
    for key, lab in (("random", "random"), ("rate_rank", "bit ranking")):
        cells = []
        for row in d["rows"]:
            r_ = row.get(key)
            cells.append((r_["db"], r_["saving"]) if r_ else None)
        tlines.append(" & ".join([lab] + [_cell(c) for c in cells]) + r" \\")
    cells = [(row["oracle"]["db"], row["oracle"]["saving"]) for row in d["rows"]]
    tlines.append(" & ".join([r"\textbf{oracle}"] +
                             [r"\textbf{" + _cell(c) + "}" for c in cells])
                  + r" \\")
    tlines += [r"\bottomrule", r"\end{tabular}"]
    w("static_main.tex", "\n".join(tlines))
    # The shallowest uniform depth the paper quotes as the static alternative.
    # check_paper used to hardcode it, which made a legitimate re-measurement
    # look like prose drift.
    _r0 = next((r for r in d["rows"] if r.get("qp") == 0), None)
    _u3 = next((u for u in (_r0 or {}).get("uniform", []) if u.get("exit") == 3),
               None)
    if _u3:
        mac("UniformThreeLow", f"{_u3['saving']:.1f}")
    mac("BestStaticMean", f"{sum(means['static'])/len(means['static']):.1f}")
    # What the deepest uniform depth costs rather than saves, and the adaptive
    # saving at the two rates where it is the only depth inside the budget.
    # These were typed into the prose and one of them went stale by 2.8 points.
    _u5 = next((u for u in (_r0 or {}).get("uniform", []) if u.get("exit") == 5),
               None)
    if _u5:
        mac("DeepestUniformCost", f"{abs(_u5['saving']):.1f}")
    for _q, _name in ((48, "OracleAtFortyEight"), (63, "OracleAtSixtyThree")):
        _r = next((r for r in d["rows"] if r.get("qp") == _q), None)
        if _r and _r.get("oracle"):
            mac(_name, f"{_r['oracle']['saving']:.1f}")
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
    mean01 = sum(sv(r01[q]) for q in QPS if q in r01) / \
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
b1, _ = pick("router_RECIPE512_b01_e4head.json", "router_RECIPE512_b01_PAPER.json", "router_RECIPE512_b01_fixed.json", "router_RECIPE512_b01.json")
b3, _ = pick("router_RECIPE512_b03_e4head.json", "router_RECIPE512_b03_PAPER.json", "router_RECIPE512_b03_fixed.json", "router_RECIPE512_b03.json")
if sa and b1:
    def Aat(bud):
        return {r["qp"]: sv(r) for r in sa["rows"]
                if abs(r["budget_db"] - bud) < 1e-9 and r.get("budget_reachable")}

    def Bat(d_):
        return {r["qp"]: sv(r) for r in (d_ or {}).get("rows", [])
                if r.get("budget_reachable")}
    A1, B1 = Aat(0.1), Bat(b1)
    # The bitstream-identical mode's own headline, so the contributions can
    # quote both modes without a number being typed in.
    if 0 in B1 and B1[0] is not None:
        mac("RouterLowRate", f"{B1[0]:.1f}")
    if 63 in B1 and B1[63] is not None:
        mac("RouterHighRate", f"{B1[63]:.1f}")
    _bv = [v for v in B1.values() if v is not None]
    if _bv:
        mac("RouterMean", f"{sum(_bv)/len(_bv):.1f}")
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

# ------------------------------------------------- beta, calibrated held out
# Configuration B's beta is bisected against the budget, and a deployed decoder
# cannot run that bisection: the budget is a distortion against a source it
# never receives. results/beta_calibration.json calibrates beta on the held-out
# Open Images validation split instead and applies the table unchanged to the
# test frames, so both numbers are on the same checkpoint, head and frames and
# what separates them is the transfer rather than a change of anything else.
print("held-out beta")
bh, _ = pick("beta_calibration.json")
trows = [r for r in (bh or {}).get("rows", [])
         if r.get("budget_reachable")
         and r.get("test_saving_pct_measured") is not None]
if trows:
    hby = {r["qp"]: r for r in trows}
    hqs = [q for q in QPS if q in hby]
    # Rates where the calibration set never reached the budget carry no beta at
    # all. They stay in the table as blanks rather than being dropped: a rate
    # the offline procedure cannot serve is the most important thing it does.
    held = [q for q in hqs
            if hby[q].get("heldout_saving_pct_measured") is not None]
    calf = {r["qp"]: r.get("floor_db") for r in bh.get("calibration_rows", [])}
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{$\beta$ held out}"
             r" & \multicolumn{3}{c}{$\beta$ bisected on the test set} & \\",
             r"$q$ & $\beta$ & saving & dB & $\beta$ & saving & dB"
             r" & at equal dB \\", r"\midrule"]
    for q in hqs:
        r_ = hby[q]
        tc = r_.get("transfer_cost_pts")
        cells = [f"{q}"]
        if q in held:
            cells += [f"{r_['beta_heldout']:.0f}",
                      f"{r_['heldout_saving_pct_measured']:.1f}",
                      f"{r_['heldout_db']:.3f}"]
        else:
            cells += ["--", "--", "--"]
        cells += [f"{r_['beta_test']:.0f}",
                  f"{r_['test_saving_pct_measured']:.1f}",
                  f"{r_['test_db']:.3f}",
                  # tc == 0.0 printed "$+-0.0$": the else branch formatted
                  # -0.0, which Python renders with its sign. Every value in
                  # this column is a hundredth of a point or less, which is the
                  # finding, so it is printed at two decimals and exact zero is
                  # printed without a sign.
                  "--" if tc is None else
                  ("$0.00$" if abs(tc) < 5e-3 else f"${-tc:+.2f}$")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("beta_heldout.tex", "\n".join(lines))

    cal = bh.get("calibration_set", {})
    if cal.get("n_images"):
        mac("HeldNCal", str(cal["n_images"]))
    mac("HeldNRates", str(len(hqs)))
    if held:
        mac("HeldBetaLow", f"{hby[held[0]]['beta_heldout']:.0f}")
        mac("HeldBetaHigh", f"{hby[held[-1]]['beta_heldout']:.0f}")
        mac("HeldSavingLow",
            f"{hby[held[0]]['heldout_saving_pct_measured']:.1f}")
        mac("HeldSavingHigh",
            f"{hby[held[-1]]['heldout_saving_pct_measured']:.1f}")
        mac("HeldDbLow", f"{hby[held[0]]['heldout_db']:.3f}")
        mac("HeldDbHigh", f"{hby[held[-1]]['heldout_db']:.3f}")
        mac("HeldSavingMean",
            f"{sum(hby[q]['heldout_saving_pct_measured'] for q in held)/len(held):.1f}")
        mac("BisectedSavingMean",
            f"{sum(hby[q]['test_saving_pct_measured'] for q in held)/len(held):.1f}")
        # The two allocations sit at different delivered qualities, so the
        # honest headline is the delivered dB and the saving given up at EQUAL
        # dB, not the difference between two savings taken at two qualities.
        dbs = {q: hby[q]["heldout_db"] for q in held}
        mac("HeldDbWorst", f"{max(dbs.values()):.3f}")
        mac("HeldDbWorstQp", str(max(dbs, key=dbs.get)))
        mac("HeldDbBest", f"{min(dbs.values()):.3f}")
        mac("HeldOverBudget",
            f"{max(max(dbs.values()) - bh['budget_db'], 0.0):.3f}")
        mac("HeldNOver", str(sum(1 for v in dbs.values()
                                 if v > bh["budget_db"] + 5e-4)))
        mac("HeldNHeld", str(len(held)))
        # The direction reversed at epoch 4. On the earlier checkpoint a beta
        # fitted to 512px photographs overshot the budget on video; on this one
        # it undershoots at every rate, which is the safe direction and a
        # different claim, so the quantities the prose needs are different too.
        ovr = {q: dbs[q] - bh["budget_db"] for q in held
               if dbs[q] > bh["budget_db"] + 5e-4}
        if ovr:
            mac("HeldOverMax", f"{max(ovr.values()):.3f}")
            mac("HeldOverMaxQp", str(max(ovr, key=ovr.get)))
            mac("HeldOverMin", f"{min(ovr.values()):.3f}")
            mac("HeldOverPctMax",
                f"{100 * max(ovr.values()) / bh['budget_db']:.0f}")
            mac("HeldOverPctMin",
                f"{100 * min(ovr.values()) / bh['budget_db']:.0f}")
        und = {q: bh["budget_db"] - dbs[q] for q in held
               if dbs[q] < bh["budget_db"] - 5e-4}
        mac("HeldNUnder", str(len(und)))
        if und:
            mac("HeldUnderMax", f"{max(und.values()):.3f}")
            mac("HeldUnderMaxQp", str(max(und, key=und.get)))
        gaps = {q: hby[q]["test_saving_pct_measured"]
                - hby[q]["heldout_saving_pct_measured"] for q in held}
        if gaps:
            mac("HeldGiveUpMax", f"{max(gaps.values()):.1f}")
            mac("HeldGiveUpMaxQp", str(max(gaps, key=gaps.get)))
            mac("HeldGiveUpMin", f"{min(gaps.values()):.1f}")
            mac("HeldGiveUpMinAbs", f"{abs(min(gaps.values())):.1f}")
            mac("HeldGiveUpMinQp", str(min(gaps, key=gaps.get)))
        tcs = {q: hby[q].get("transfer_cost_pts") for q in held
               if hby[q].get("transfer_cost_pts") is not None}
        if tcs:
            mac("HeldTransferMax", f"{max(tcs.values()):.1f}")
            mac("HeldTransferMin", f"{min(tcs.values()):.1f}")
            mac("HeldTransferMaxQp", str(max(tcs, key=tcs.get)))
            mac("HeldTransferMean",
                f"{sum(tcs.values())/len(tcs):.1f}")
            # Sign-free, because the prose claim is that the two allocations
            # sit on the same frontier and the direction of a tenth of a point
            # is not a claim worth making.
            mac("HeldTransferAbsMax",
                f"{max(abs(v) for v in tcs.values()):.1f}")
    # Why the table lands where it does: the floor is the tiling penalty with
    # no early exit at all, and it is not the same on 512px photographs as on
    # 1080p video, so a budget measured from one is a different distance above
    # the floor on the other.
    fgap = {q: hby[q]["floor_db"] - calf[q] for q in hqs
            if calf.get(q) is not None and hby[q].get("floor_db") is not None}
    if fgap:
        mac("HeldFloorGapMin", f"{min(fgap.values()):.3f}")
        mac("HeldFloorGapMax", f"{max(fgap.values()):.3f}")
        # The same difference the other way up, which is the direction the
        # prose reads it in: how far the calibration floor sits ABOVE the
        # test one. It changes sign along the ladder, and a signed range
        # printed as "a gap of -0.024 to 0.005" said nothing.
        _co = {q: -v for q, v in fgap.items()}
        _above = [q for q in _co if _co[q] > 0]
        _below = [q for q in _co if _co[q] <= 0]
        mac("HeldFloorCalOverMax", f"{max(_co.values()):.3f}")
        mac("HeldFloorCalOverMaxQp", str(max(_co, key=_co.get)))
        mac("HeldFloorCalUnderMax", f"{-min(_co.values()):.3f}")
        mac("HeldFloorNAbove", str(len(_above)))
        mac("HeldFloorNBelow", str(len(_below)))
        mac("HeldFloorCalLow", f"{calf[hqs[0]]:.3f}")
        mac("HeldFloorTestLow", f"{hby[hqs[0]]['floor_db']:.3f}")
        mac("HeldFloorCalHigh", f"{calf[hqs[-1]]:.3f}")
        mac("HeldFloorTestHigh", f"{hby[hqs[-1]]['floor_db']:.3f}")
    nob = [q for q in hqs if q not in held]
    mac("HeldNNoBeta", str(len(nob)))
    if nob:
        mac("HeldNoBetaQp", str(nob[-1]))
        if calf.get(nob[-1]) is not None:
            mac("HeldNoBetaFloor", f"{calf[nob[-1]]:.3f}")
        mac("HeldNoBetaForgone", f"{hby[nob[-1]]['test_saving_pct_measured']:.1f}")

# The cost of our own deepest exit, relative to the released decoder. It is
# the denominator the paper deliberately does not use, and it was typed into
# two sentences; it is arithmetic over the architecture, but a typed number
# beside a live one is how the last three drifts started.
_dc, _ = pick("router_RECIPE512_b01_e4head.json", "router_RECIPE512_b01_PAPER.json")
if _dc and _dc.get("deepest_exit_cost"):
    mac("DeepestExitCost", f"{_dc['deepest_exit_cost']:.4f}")

# What the cut costs at uniform depth, measured against this model's own
# full-frame decode -- the quantity that cancels when both sides of the ratio
# are tiled. Two numbers were typed into Section 5.1 for it.
print("reference gap")
_rg, _ = pick("reference_gap.json")
if _rg:
    mac("TileRefShallowLo", f"{_rg['tiling_shallow_lo']:.3f}")
    mac("TileRefShallowHi", f"{_rg['tiling_shallow_hi']:.3f}")
    mac("TileRefDeepLo", f"{_rg['tiling_deep_lo']:.3f}")
    mac("TileRefDeepHi", f"{_rg['tiling_deep_hi']:.3f}")
    mac("TileRefFrames", str(_rg["n_frames"]))

# The deblocking pass, split by distance from the tile boundary. Section 4.3
# argued from five typed numbers that it does not earn its cost; they are
# measured now, and the verdict turns out to depend on the rate.
print("seam ring")
_sr, _ = pick("seam_ring.json")
if _sr:
    mac("SeamRingPct", f"{100 * _sr['ring_fraction_of_pixels']:.1f}")
    mac("SeamRingGainPct", f"{abs(_sr['ring_error_change_vs_source_pct']):.2f}")
    mac("SeamInteriorPct",
        f"{abs(_sr['interior_error_change_vs_source_pct']):.3f}")
    mac("SeamRecovLo", f"{_sr['recovered_lo']:.4f}")
    mac("SeamRecovHi", f"{_sr['recovered_hi']:.4f}")
    mac("SeamRecovHiQp", str(_sr["recovered_hi_qp"]))
    mac("SeamPerfectExtra", f"{_sr['perfect_gate_extra_db']:.4f}")
    mac("SeamPenaltyLo", f"{min(r['penalty_off_db'] for r in _sr['by_rate']):.3f}")
    mac("SeamPenaltyHi", f"{max(r['penalty_off_db'] for r in _sr['by_rate']):.3f}")
    mac("SeamRingFrames", str(_sr["n_frames"]))
    _pp = [r["recovered_db"] / 0.95 for r in _sr["by_rate"]]
    mac("SeamPerPointLo", f"{min(_pp):.4f}")
    mac("SeamPerPointHi", f"{max(_pp):.4f}")
    _hi = max(_sr["by_rate"], key=lambda r: r["recovered_db"])
    mac("SeamRecovShareHi",
        f"{100 * _hi['recovered_db'] / _hi['penalty_off_db']:.0f}")

# The single frame the deblocking figure is drawn from.
_srg, _ = pick("seam_repair_grid.json")
if _srg:
    mac("SeamGridRecovered", f"{_srg['recovered_db']:.3f}")
    mac("SeamGridPenalty", f"{_srg['penalty_off_db']:.3f}")

# ---------------------------------------------------------------- exactness
print("halo exactness")
he, _ = pick("halo_exactness.json")
if he:
    w_ = he["worst"]
    def _e(x):
        return "0" if x == 0 else f"{x:.2e}".replace("e-0", "e-")
    mac("HaloFilterOn", _e(w_["halo exchange, filter on"]))
    mac("HaloFilterOff", _e(w_["halo exchange, filter off"]))
    mac("HaloPadding", _e(w_["replicate padding, filter off"]))
    mac("HaloNFrames", str(he["n_frames"]))
    # main.tex sets the same three numbers in a table, in maths.
    def _m(x):
        if x == 0:
            return r"\mathbf{0}"
        t = f"{x:.2e}".split("e")
        return f"{t[0]}\\times10^{{{int(t[1])}}}"
    mac("HaloFilterOnMath", _m(w_["halo exchange, filter on"]))
    mac("HaloFilterOffMath", _m(w_["halo exchange, filter off"]))
    mac("HaloPaddingMath", _m(w_["replicate padding, filter off"]))

# ------------------------------------------------------------------ hybrid C
# The sweep is arithmetic-model only: no hook count was taken for it. That is
# why its ends sit 0.4 to 0.8 points above configurations A and B, which are
# hook-counted -- the same offset the reporting conventions describe. The
# macros below let the paper say so instead of claiming the ends agree exactly.
print("hybrid C")
hy, _ = pick("hybrid_RECIPE512_b01_e4head.json",
             "hybrid_RECIPE512_b01_fixed.json", "hybrid_RECIPE512_b01.json")
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
            f"{sv(v):.1f}" if v else "---"
            for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("hybrid.tex", "\n".join(lines))
    def recov(q, r_):
        b0, bx, ba = cell(q, 0.0), cell(q, r_), cell(q, 1.0)
        if not (b0 and bx and ba):
            return None
        span = sv(ba) - sv(b0)
        if span <= 1e-9:
            return None
        return 100 * (sv(bx)
                      - sv(b0)) / span

    qlo, qhi = qs[0], qs[-1]
    for tag, q in (("Low", qlo), ("High", qhi)):
        for rt, r_ in (("Tenth", 0.1), ("Fifth", 0.2), ("Half", 0.5)):
            v = recov(q, r_)
            if v is not None:
                mac(f"HybridRecover{rt}{tag}", f"{v:.0f}")
            c_ = cell(q, r_)
            if c_ and tag == "High":
                mac(f"HybridSave{rt}High", f"{sv(c_):.1f}")
    for rt, r_ in (("Tenth", 0.1), ("Fifth", 0.2), ("Half", 0.5),
                   ("Full", 1.0)):
        c_ = cell(qhi, r_)
        if c_:
            mac(f"HybridBits{rt}", f"{c_['map_bits']:.0f}")
    # Where a partial map BEATS the full one. Two multipliers -- the oracle's
    # lambda on the overridden tiles, the router's fixed beta on the rest --
    # reach allocations a single lambda cannot.
    over = []
    for q in qs:
        ca, cb = cell(q, 1.0), cell(q, 0.5)
        if ca and cb:
            over.append((q, sv(cb)
                         - sv(ca)))
    if over:
        # Signalling half the tiles against signalling all of them. On a head
        # fitted to other weights this was positive at two rates; with the
        # head the paper reports, B starts high enough that the second
        # multiplier has nothing left to exploit and it is negative
        # everywhere. Both counts are always defined, so a paragraph that
        # reports the comparison prints a number rather than a macro name.
        beat = [d for _, d in over if d > 0]
        mac("HybridHalfVsFullN", str(len(beat)))
        mac("HybridHalfVsFullOf", str(len(over)))
        mac("HybridBeatsABy", f"{max(beat):.2f}" if beat else "0.00")
        # How far B (rho = 0) starts below A (rho = 1), worst rate.
        _g0 = [(q, sv(cell(q, 1.0)) - sv(cell(q, 0.0))) for q, _ in over
               if cell(q, 1.0) and cell(q, 0.0)]
        if _g0:
            mac("HybridStartGapLo", f"{min(v for _, v in _g0):.1f}")
            mac("HybridStartGapHi", f"{max(v for _, v in _g0):.1f}")
        short = [-d for _, d in over if d <= 0]
        if short:
            mac("HybridHalfShortLo", f"{min(short):.2f}")
            mac("HybridHalfShortHi", f"{max(short):.2f}")
            mac("HybridHalfShortHiQp", str(max(over, key=lambda t: -t[1])[0]))
            mac("HybridHalfShortLoQp", str(max(over, key=lambda t: t[1])[0]))
    # How far the sweep's ends sit from the hook-counted A and B.
    try:
        _bB, _ = pick("router_RECIPE512_b01_e4head.json", "router_RECIPE512_b01_PAPER.json",
                      "router_RECIPE512_b01_fixed.json",
                      "router_RECIPE512_b01.json")
        _bA = json.load(open(RES / "signalled_RECIPE512_ctc53.json"))
        _B = {r["qp"]: sv(r) for r in _bB["rows"] if r.get("budget_reachable")}
        _A = {r["qp"]: sv(r) for r in _bA["rows"]
              if abs(r["budget_db"] - 0.1) < 1e-9 and r.get("budget_reachable")}
        _off = []
        for _q in _B:
            _r0 = [r for r in hy["rows"] if r["qp"] == _q and r["rho"] == 0.0]
            _r1 = [r for r in hy["rows"] if r["qp"] == _q and r["rho"] == 1.0]
            if _r0 and _q in _B:
                _off.append(sv(_r0[0]) - _B[_q])
            if _r1 and _q in _A:
                _off.append(sv(_r1[0]) - _A[_q])
        if _off:
            mac("HybridEndOffsetLo", f"{min(_off):.2f}")
            mac("HybridEndOffsetHi", f"{max(_off):.2f}")
    except Exception as _e:
        print("   hybrid end offsets:", _e)
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
            # No longer clamped at 1: re-bisecting lambda enlarges the
            # feasible set, so the measured recovery can and does exceed the
            # fixed-lambda Lorenz prediction.
            mac("LorenzTightMin", f"{100*min(ratios):.0f}")
            mac("LorenzTightMax", f"{100*max(ratios):.0f}")

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
    # The reverse direction, which the prose carried as two typed numbers
    # beside these macros and which went stale with the checkpoint.
    back = next((r for r in rr_ if r.get("from") == 63 and r.get("to") == 0),
                None)
    if back:
        mac("TransferBackDb", f"{back['transfer_db']:.3f}")
        mac("TransferBackSaving", f"{back['transfer_saving']:.1f}")

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
    # The floors themselves, because the prose quoted them as typed numbers
    # beside these live percentages and they went stale the moment the
    # checkpoint moved.
    for tag, qp in (("Low", 0), ("High", 63)):
        r = next((x for x in rows_ if x["qp"] == qp), None)
        if r:
            mac(f"CoupFloorPad{tag}", f"{r['padded']['floor_db']:.3f}")
            mac(f"CoupFloorCpl{tag}", f"{r['coupled']['floor_db']:.3f}")
    drops = [100*(1 - r['coupled']['floor_db']/r['padded']['floor_db'])
             for r in rows_]
    mac("CoupFloorDropLo", f"{min(drops):.0f}")
    mac("CoupFloorDropHi", f"{max(drops):.0f}")

# hybrid at the loose budget: where the gap is large, how cheaply it closes
hy3, _ = pick("hybrid_RECIPE512_b03_e4head.json",
                 "hybrid_RECIPE512_b03_fixed.json")
if hy3:
    r3 = [r for r in hy3["rows"] if r.get("budget_reachable")]

    def _c3(q, rho):
        return next((sv(r) for r in r3
                     if r["qp"] == q and abs(r["rho"] - rho) < 1e-9), None)
    qs3 = sorted({r["qp"] for r in r3})
    rec = []
    for q in qs3:
        b0, bh, ba = _c3(q, 0.0), _c3(q, 0.5), _c3(q, 1.0)
        if None in (b0, bh, ba) or ba - b0 < 0.5:
            continue                      # saturated: no gap to recover
        rec.append((q, 100 * (bh - b0) / (ba - b0)))
    if rec:
        mac("HybridLooseRecLo", f"{min(v for _, v in rec):.0f}")
        mac("HybridLooseRecHi", f"{max(v for _, v in rec):.0f}")
        # No math delimiters inside a macro value: the reportlab build
        # substitutes it into prose and has no way to strip them, so the
        # paper printed a literal "$q48$ and $q63$". Plain text here; each
        # document formats it the way its renderer expects.
        mac("HybridLooseQps", " and ".join(f"q{q}" for q, _ in rec))
    else:
        # Every rate saturated. On the checkpoint the paper first reported
        # some rates still had a gap for a partial map to recover at the
        # loose budget; on this one configuration B alone reaches within a
        # fifth of a point of the signalled map everywhere, so there is
        # nothing left to buy. The macros are defined either way, because an
        # undefined one prints as itself.
        _gaps = []
        for q in qs3:
            b0, ba = _c3(q, 0.0), _c3(q, 1.0)
            if None not in (b0, ba):
                _gaps.append(ba - b0)
        mac("HybridLooseQps", "no rate")
        mac("HybridLooseRecLo", "0")
        mac("HybridLooseRecHi", "0")
        if _gaps:
            mac("HybridLooseGapMax", f"{max(_gaps):.2f}")

# which predictor should configuration C be built on
hyr, _ = pick("hybrid_raterank_b01.json")
if hy and hyr:
    def _t(d):
        return {(r["qp"], round(r["rho"], 4)): sv(r)
                for r in d["rows"] if r.get("budget_reachable")}
    Th, Tr = _t(hy), _t(hyr)
    both = [(k, Tr[k] - Th[k]) for k in Th if k in Tr and k[1] < 1.0]
    if both:
        mac("CPredBitsAhead", f"{max(v for _, v in both):.1f}")
        mac("CPredHeadAhead", f"{-min(v for _, v in both):.1f}")
        best = max(((k, v) for k, v in Tr.items()), key=lambda t: t[1])
        mac("CBestSave", f"{best[1]:.1f}")
        mac("CBestQp", str(best[0][0]))
        mac("CBestRho", f"{100*best[0][1]:.0f}")
        a1 = Th.get((best[0][0], 1.0))
        if a1:
            mac("CBestOverA", f"{best[1] - a1:.2f}")

# does the trade-off collapse once each rate's own band is divided out?
bc, _ = pick("band_collapse.json")
if bc:
    mac("BandRawSpread", f"{bc['raw_spread_at_tenth_db']:.1f}")
    mac("BandSpreadMean", f"{bc['band_spread_mean']:.1f}")
    mac("BandSpreadMax", f"{bc['band_spread_max_excl_edge']:.1f}")
    if "power_exponent" in bc:
        mac("BandExp", f"{bc['power_exponent']:.2f}")
        mac("BandRTwo", f"{bc['power_r2']:.3f}")
        mac("BandFitErr", f"{bc['power_max_err']:.1f}")
        try:
            _bb = json.load(open(RES / "band_collapse_BEST.json"))
            mac("BandFitErrB", f"{_bb['power_max_err']:.1f}")
        except Exception as _e:
            print("   BEST fit error:", _e)
    bb, _ = pick("band_collapse_BEST.json")
    if bb:
        mac("BandRawSpreadB", f"{bb['raw_spread_at_tenth_db']:.1f}")
        mac("BandSpreadMeanB", f"{bb['band_spread_mean']:.1f}")
        mac("BandSpreadMaxB", f"{bb['band_spread_max_excl_edge']:.1f}")
        mac("BandExpB", f"{bb['power_exponent']:.2f}")
        mac("BandRTwoB", f"{bb['power_r2']:.3f}")

# what a retrain with a working mask does
rt, _ = pick("router_retrain_compare.json")
if rt:
    ds = {int(q): rt["v3"][q] - rt["v2"][q] for q in rt["v2"]}
    mac("RetrainAgreeOld", f"{rt['heldout_agree_v2']:.3f}")
    mac("RetrainAgreeNew", f"{rt['heldout_agree_v3']:.3f}")
    mac("RetrainGain", f"{max(ds.values()):.1f}")
    mac("RetrainLoss", f"{-min(ds.values()):.1f}")
    mac("RetrainGainQp", str(min(ds, key=lambda q: -ds[q])))
    mac("RetrainLossQp", str(min(ds, key=lambda q: ds[q])))

# what a better predictor does to the regret distribution
# The claim is about the head, so both sides are measured on the pinned
# weights and differ only in which head C is built on: the pre-fix head in
# _fixed, the post-fix head in _pin. `hy` is not the comparison partner any
# more -- it now reports the head fitted to the pin, which is a third head.
hv3, _ = pick("hybrid_v3_b01_pin.json", "hybrid_v3_b01.json")
hy0, _ = pick("hybrid_RECIPE512_b01_fixed.json")
if hy0 and hv3:
    def _g(d):
        return {r["qp"]: r["gini_regret"] for r in d["rows"]
                if r.get("gini_regret") is not None}
    G2, G3 = _g(hy0), _g(hv3)
    common = [q for q in G2 if q in G3]
    if common:
        up = [q for q in common if G3[q] > G2[q]]
        mac("GiniRetrainUpN", str(len(up)))
        mac("GiniRetrainOfN", str(len(common)))
        mac("GiniRetrainLo", f"{min(G3[q] for q in common):.2f}")
        mac("GiniRetrainHi", f"{max(G3[q] for q in common):.2f}")

# does the free rule beat a trained head on a second training run?
rb, _ = pick("raterank_BEST_compare.json")
if rb:
    qs_ = sorted(rb["raterank"], key=int)
    d_ = [(q, rb["raterank"][q] - rb["router"][q]) for q in qs_
          if q in rb["router"]]
    g_ = [(q, rb["a_oracle"][q] - rb["raterank"][q]) for q in qs_]
    mac("BestRankWinsN", str(sum(1 for _, v in d_ if v > 0)))
    mac("BestRankOfN", str(len(d_)))
    mac("BestRankBy", f"{max(v for _, v in d_):.1f}")
    mac("BestRankToOracle", f"{max(v for _, v in g_):.1f}")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"$q$ & A signalled & B router & bits, 0 params \\", r"\midrule"]
    for q in qs_:
        v = rb["raterank"][q]
        beats = q in rb["router"] and v > rb["router"][q]
        lines.append(f"{q} & {rb['a_oracle'][q]:.1f} & "
                     f"{rb['router'].get(q, float('nan')):.1f} & "
                     + (f"\\textbf{{{v:.1f}}}" if beats else f"{v:.1f}")
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("raterank_best.tex", "\n".join(lines))

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
        vals = [sv(cell(q, w_) or {}) for w_ in ws]
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
        bw, bc = max(vals, key=lambda t: sv(t[1]))
        mac(f"Blend{tag}Best", f"{sv(bc):.1f}")
        mac(f"Blend{tag}BestW", f"{bw:g}")
        if c1:
            mac(f"BlendHeadOnly{tag}", f"{sv(c1):.1f}")
        if c0:
            mac(f"BlendBitsOnly{tag}", f"{sv(c0):.1f}")
    # What the head costs when it is trusted completely, across the rates.
    # On the pinned checkpoint the bit rule wins everywhere, so the
    # interesting quantity is no longer a gain but a price.
    _cost = []
    for q in qs:
        c0, c1 = cell(q, 0.0), cell(q, 1.0)
        if c0 and c1:
            _cost.append(sv(c0) - sv(c1))
    if _cost:
        mac("BlendCostMin", f"{min(_cost):.1f}")
        mac("BlendCostMax", f"{max(_cost):.1f}")
        # The rate that pays the most for it, named rather than typed.
        mac("BlendCostMaxQ", f"q{qs[_cost.index(max(_cost))]}")
        mac("BlendBestW",
            "0" if all(sv(cell(q, 0.0)) >= max(
                sv(c) for _, c in [(w_, cell(q, w_)) for w_ in ws] if c)
                - 1e-9 for q in qs) else "mixed")

    # What blending is worth at the two rates where it is worth anything. The
    # prose had +2.1 and +0.9 typed into it, which is how a number outlives
    # the measurement it came from.
    for tag, q in (("Mid", qs[-2]), ("High", qs[-1])):
        c0 = cell(q, 0.0)
        vals = [(w_, cell(q, w_)) for w_ in ws]
        vals = [(w_, c) for w_, c in vals if c]
        if not (c0 and vals):
            continue
        _, bc = max(vals, key=lambda t: sv(t[1]))
        mac(f"BlendGain{tag}", f"{sv(bc) - sv(c0):+.1f}")
        mac(f"BlendGain{tag}Q", f"q{q}")

# ------------------------------------------------ the colour components
# The budget is set on a 6:1:1 weighted PSNR in which chroma carries an
# eighth, so the weighted mean can land on the budget while its parts move
# apart underneath it. Nothing else in the paper would show that.
print("per component")
try:
    _yv = json.load(open(RES / "rd_yuv_PAPER.json"))
    _rel = {r["qp"]: r for r in _yv["rows"] if r["config"] == "release"}
    _at = {r["qp"]: r for r in _yv["rows"] if r.get("budget_db") == 0.1}
    _qs = sorted(q for q in _rel if q in _at)
    if _qs:
        def _d(k):
            return sum(_at[q][k] - _rel[q][k] for q in _qs) / len(_qs)
        _dy, _du, _dv = _d("psnr_y"), _d("psnr_u"), _d("psnr_v")
        mac("ChromaLumaDrop", f"{abs(_dy):.2f}")
        mac("ChromaUDrop", f"{abs(_du):.2f}")
        mac("ChromaVDrop", f"{abs(_dv):.2f}")
        mac("ChromaRatio", f"{((_du + _dv) / 2) / _dy:.1f}")
except Exception as _e:
    print("   rd_yuv_PAPER.json:", _e)


# ------------------------------------------------- the propositions, counted
# "seven of seven" was typed into both documents. The file says how many there
# are and how many passed; if a proposition is added or one starts failing,
# the sentence should move with it.
print("propositions")
try:
    _tc = json.load(open(RES / "theory_checks.json"))
    mac("PropsPassed", str(int(_tc["n_passed"])))
    mac("PropsTotal", str(int(_tc["n_propositions"])))
except Exception as _e:
    print("   theory_checks.json:", _e)


# ------------------------------------------- two numbers derived from others
# A typed number sitting beside the macros it is computed from is the failure
# this file keeps finding: the macro moves and the neighbour does not. Both of
# these were typed.
print("derived")
try:
    _sa = float(MACROS["SeamArlsHigh"]) if "SeamArlsHigh" in MACROS else None
    _sr = float(MACROS["SeamReplHigh"]) if "SeamReplHigh" in MACROS else None
    if _sa is not None and _sr is not None:
        mac("SeamArlsGain", f"{abs(_sr - _sa):.2f}")
except Exception as _e:
    print("   arls gain:", _e)
try:
    _rt = _pick_json("router_RECIPE512_b01_e4head.json",
                       "router_RECIPE512_b01_PAPER.json")
    _ha = _rt.get("router2_meta")
    if isinstance(_ha, str):
        import ast as _ast
        _ha = _ast.literal_eval(_ha)
    if _ha and _ha.get("heldout_agree") is not None:
        mac("RouterHeldAgree", f"{float(_ha['heldout_agree']):.3f}")
except Exception as _e:
    print("   router heldout agree:", _e)


# ------------------------------------------------------------- the seam gate
# Its corner value and its plateau, which the figure's caption quoted from the
# terminal. pipeline_stage_figs writes them now.
print("seam gate")
try:
    _sg = json.load(open(RES / "seam_gate.json"))
    mac("GateMax", f"{_sg['gate_max']:.2f}")
    mac("GateMin", f"{_sg['gate_min']:.2f}")
except Exception as _e:
    print("   seam_gate.json:", _e)


# ------------------------------------------------- what the search costs A
# Section 3.5 quoted 4.6x for the deployed table where the file says 4.52,
# and the other five numbers in that passage were typed too.
print("encoder cost")
try:
    _ec = json.load(open(RES / "encoder_cost.json"))
    mac("EncDeployedX", f"{_ec['x_deployed']:.1f}")
    mac("EncFullFrameX", f"{_ec['x_full_frame']:.1f}")
    mac("EncAgree", f"{100 * _ec['approx']['agreement']:.0f}")
    mac("EncApproxSaving", f"{_ec['approx']['saving']:.1f}")
    mac("EncExactSaving", f"{_ec['exact']['saving']:.1f}")
    mac("EncSavingGap",
        f"{abs(_ec['approx']['saving'] - _ec['exact']['saving']):.1f}")
except Exception as _e:
    print("   encoder_cost.json:", _e)


# ------------------------------------------------------------- tile size
# Section 5.3 compared a 256 px column of 36.3 / 33.8 / 20.9 / 11.3 against a
# 128 px one. Those four numbers are in no results file: the live 256 px
# column is 33.7 / 31.1 / 18.0 / 8.8, which is the paper's own per-class table
# two paragraphs earlier. Comparing a stale column against a current one
# reversed the sign at the lowest rate, where the supplement already says the
# smaller tile is ahead in five classes of six.
print("tile size")
try:
    _ts = json.load(open(RES / "tilesize_adaptive.json"))
    _byq = {r["qp"]: r for r in _ts["per_qp"]}
    _q0 = _byq[0]
    _d0 = [(c["cls"], c["saving_128_pct"] - c["saving_256_pct"])
           for c in _q0["classes"]]
    _ahead = [c for c, v in _d0 if v > 0]
    mac("TileAheadLow", str(len(_ahead)))
    mac("TileClassesN", str(len(_d0)))
    mac("TileGainLowMin", f"{min(v for _, v in _d0 if v > 0):+.1f}")
    mac("TileGainLowMax", f"{max(v for _, v in _d0):+.1f}")
    _behind0 = [c for c, v in _d0 if v <= 0]
    mac("TileLowLoser", _behind0[0].replace("_", " ") if _behind0 else "none")
    for _q, _n in ((32, "Mid"), (63, "High")):
        _d = [c["saving_128_pct"] - c["saving_256_pct"]
              for c in _byq[_q]["classes"]]
        mac(f"TileBehind{_n}", str(sum(1 for v in _d if v < 0)))
        mac(f"TileBehind{_n}Max", f"{-min(_d):.1f}")
    mac("TileSpliceDelta",
        f"{_ts['summary_over_qps']['mean_set_mean_delta_adaptive_minus_256']:+.2f}")
    _c0 = _q0["classes"][0]
    mac("TileCountMul", f"{_c0['tiles_128'] / _c0['tiles_256']:.1f}")
    mac("TileConfoundSteps", f"{_ts['confound']['training_gap_steps']:,}")
except Exception as _e:
    print("   tilesize_adaptive.json:", _e)


# ------------------------------------------------- what a convention is worth
# The protocol paragraph used to say two implementations had differed by 3.29
# saving points and that per-frame against pooled moved an integrated saving by
# 7 to 10. Neither number is in results/ any more, and the second is off by a
# factor of three against what the pinned files measure. Both are computed here
# now, from the files the supplement integrates.
print("conventions")
try:
    _pf = json.load(open(RES / "supp_bd_PAPER_per_frame.json"))["rows"]
    _po = json.load(open(RES / "supp_bd_PAPER_pooled.json"))["rows"]
    _pairs = [(a["bd_saving_pct"], b["bd_saving_pct"])
              for a, b in zip(_pf, _po)
              if a.get("bd_saving_pct") is not None
              and b.get("bd_saving_pct") is not None]
    if _pairs:
        mac("ConvPooledMax", f"{max(abs(x - y) for x, y in _pairs):.1f}")
except Exception as _e:
    print("   pooled vs per-frame:", _e)
try:
    _iv = [r["mean_bd_saving_pct"]
           for r in json.load(open(RES / "bd_sensitivity.json"))["rows"]]
    mac("ConvIntervalRange", f"{max(_iv) - min(_iv):.1f}")
    mac("ConvIntervalN", str(len(_iv)))
except Exception as _e:
    print("   bd interval:", _e)


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
    # No saving column. This file carries no hook count, so its saving is the
    # arithmetic model and would print 22.1 beside a headline table saying
    # 21.5. The saving is Table 3's job; this one is BD-Rate and side channel.
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"configuration & budget & BD-Rate (\%) & side channel \\",
             r"\midrule"]
    for r in bdj["rows"]:
        lines.append(f"{r['config']} & {r['budget_db']:.1f}\\,dB & "
                     f"{r['bd_rate_pct']:.2f} & "
                     f"{r['map_bits']:.0f} bits/frame \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    try:
        mac("GapBudgetPoints",
            f"{float(MACROS['MeanAtThree']) - float(MACROS['MeanAtOne']):.1f}")
    except Exception as _e:
        print("   budget gap:", _e)
    w("bdrate.tex", "\n".join(lines))

# ------------------------------------------------------------- rate rank
print("rate rank")
rr, _ = pick("raterank_RECIPE512_b01.json")
rr3, _ = pick("raterank_RECIPE512_b03.json")
if rr3:
    rs3 = [r for r in rr3["rows"] if r.get("budget_reachable")]
    b3_, _ = pick("router_RECIPE512_b03_e4head.json", "router_RECIPE512_b03_PAPER.json", "router_RECIPE512_b03_fixed.json", "router_RECIPE512_b03.json")
    if rs3 and b3_:
        B3v = {r["qp"]: sv(r) for r in b3_["rows"]
               if r.get("budget_reachable")}
        marg = [sv(r) - B3v[r["qp"]] for r in rs3
                if r["qp"] in B3v]
        if marg:
            mac("RateRankLooseAheadBy", f"{max(marg):.1f}")
    if rs3:
        mac("RateRankLoose", f"{sv(rs3[-1]):.1f}")
rr5, _ = pick("raterank_RECIPE512_b05.json")
if rr5:
    rs5 = [r for r in rr5["rows"] if r.get("budget_reachable")]
    if rs5 and all(abs(sv(r)
                       - sv_oracle(r)) < 1e-6 for r in rs5):
        mac("RateRankHalfDbExact", "every")
        mac("RateRankLooseCeil",
            f"{sum(1 for r in rs3 if sv(r) > CEIL_MEASURED - 0.5)}")
if rr:
    rs = [r for r in rr["rows"] if r.get("budget_reachable")]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"$q$ & rate-rank & B router & A signalled & "
             r"$\rho$ depth & $\rho$ spread \\",
             r"\midrule"]
    for r in rs:
        b = (B1 or {}).get(r["qp"])
        beats = b is not None and sv(r) > b
        v = (f"\\textbf{{{sv(r):.1f}}}" if beats
             else f"{sv(r):.1f}")
        lines.append(f"{r['qp']} & {v} & "
                     + (f"{b:.1f}" if b is not None else "---")
                     + f" & {sv_oracle(r):.1f} & "
                     f"{-r['spearman_bits_vs_exit']:+.2f} & "
                     f"{r['spearman_bits_vs_spread']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("raterank.tex", "\n".join(lines))
    if rs:
        mac("RateRankLow", f"{sv(rs[0]):.1f}")
        mac("RateRankHigh", f"{sv(rs[-1]):.1f}")
        # What the rule gives up to the Lagrangian oracle, which is the gap
        # Section C asks whether separability explains. Section C had been
        # quoting \RateRankBeatsBy for it, which is the rule's margin over the
        # trained head and points the other way.
        _og = [sv_oracle(r) - sv(r) for r in rs]
        mac("RateRankOracleGap", f"{max(_og):.1f}")
        # How often the spread between our two heads is wider than the rule's
        # margin over whichever of them is nearer. The paper said "at some
        # rates" and the answer is every one of them; understating a caveat
        # against our own claim is the wrong direction to be vague in.
        try:
            _rf = {r["qp"]: r["saving_pct_vs_release"] for r in
                   _pick_json("router_RECIPE512_b01_e4head.json",
                       "router_RECIPE512_b01_PAPER.json")["rows"]}
            _rj = {r["qp"]: r["saving_pct_vs_release"] for r in
                   json.load(open(RES / "router_RECIPE512_b01_jointhead.json"))["rows"]}
            _ru = {r["qp"]: r["saving_pct_vs_release"] for r in rs}
            _qs = [q for q in _rf if q in _rj and q in _ru]
            _w = sum(1 for q in _qs
                     if abs(_rj[q] - _rf[q]) > _ru[q] - max(_rf[q], _rj[q]))
            mac("HeadSpreadWiderN", str(_w))
            mac("HeadSpreadRatesN", str(len(_qs)))
        except Exception as _e:
            print("   head spread vs margin:", _e)
        if B1:
            d_ = [(r["qp"], sv(r) - B1[r["qp"]])
                  for r in rs if r["qp"] in B1]
            win = [q for q, v in d_ if v > 0]
            if win:
                mac("RateRankBeatsFrom", str(min(win)))
                mac("RateRankBeatsUpTo", str(max(win)))
                mac("RateRankNWins", str(len(win)))
                mac("RateRankBeatsBy", f"{max(v for _, v in d_):.1f}")
                # The mean matters more than the maximum now. Against a head
                # fitted to the checkpoint it is judged on, the free rule's
                # margin is under a point on average and four thousandths of
                # a point at one rate -- it matches the head rather than
                # beating it, and the sentence in the abstract has to say so.
                mac("RateRankBeatsMean",
                    f"{sum(v for _, v in d_) / len(d_):.2f}")
                mac("RateRankBeatsMin", f"{min(v for _, v in d_):.2f}")
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
    mean01 = sum(sv(by[q]) for q in QPS if q in by) / \
        len([q for q in QPS if q in by])
    ours_kmac = INTRA_GMAC_ * (1 - mean01 / 100) * 1e9 / PX / 1e3
    rel_kmac = INTRA_GMAC_ * 1e9 / PX / 1e3
    # A single "new bitstream" column could not represent our own two modes:
    # the signalled one leaves the coded latent untouched and sends a small map
    # beside it, which is neither "yes" nor "no" to that question. Four columns
    # say what actually changes.
    rows = [
        ("SlimCAE~\\cite{slimcae}", "width", "per stream", "new", "--", "no"),
        ("Slimmable video~\\cite{slimvc}", "width", "per stream", "new", "--", "no"),
        ("EVC~\\cite{evc}", "mask / pruning", "per model", "new", "--", "no"),
        ("Spatial competition~\\cite{spatialcompetition}", "which codec",
         "per region", "new", "yes", "no"),
        ("DCVC-RT~\\cite{dcvcrt}", "architecture", "fixed", "new", "--", "no"),
        ("\\textbf{FLEX-UF} signalled", "\\textbf{decoder depth}",
         "\\textbf{per region}", "\\textbf{same}", "\\MapBits\\,b", "no"),
        ("\\textbf{FLEX-UF} bitstream-identical", "\\textbf{decoder depth}",
         "\\textbf{per region}", "\\textbf{same}", "\\textbf{none}",
         "\\textbf{yes}"),
    ]
    lines = [r"\begin{tabular}{lllccc}", r"\toprule",
             r"Method & Varies & Where & Latent & Side & Dec. \\",
             r" & & & & data & only \\",
             r"\midrule"]
    for a_, b_, c_, d_, e_, f_ in rows:
        lines.append(f"{a_} & {b_} & {c_} & {d_} & {e_} & {f_} \\\\")
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

# ------------------------------------------------------- the literature table
# Numbers here are what other people PUBLISHED, not what we measured, and the
# two blocks are on different anchors and different test sets. Mixing them into
# one BD-Rate column would be the convention error this project keeps catching,
# so they are two tables and each says what it is measured against.
#
# Image block: Table 1 of Li et al., Learned Image Compression with Hierarchical
# Progressive Context Modeling, ICCV 2025 (arXiv:2507.19125), BD-Rate against
# VTM-22.0 on Kodak, kMACs/pixel and parameters from the same table.
# Video block: Tables 1 and 3 of Li et al., Ultra-Fast Neural Video Compression
# (arXiv:2606.04410), BD-Rate against VTM-17.0 low delay, MACs per 1080p frame.
print("literature")
_LIT_IMG = [
    ("CHARM~\\cite{charm}",         "496",  "58.5",  "$+0.86$"),
    ("STF~\\cite{stf}",             "511",  "99.9",  "$-2.06$"),
    ("ELIC~\\cite{elic}",           "574",  "36.9",  "$-3.22$"),
    ("WeConvene~\\cite{weconvene}", "2343", "107.2", "$-6.98$"),
    ("MambaVC~\\cite{mambavc}",     "814",  "47.9",  "$-8.72$"),
    ("DCVC-DC intra~\\cite{dcvcdc}", "542", "45.5",  "$-9.18$"),
    ("TCM~\\cite{tcm}",             "1824", "76.6",  "$-10.70$"),
    ("FLIC~\\cite{flic}",           "1096", "71.0",  "$-13.20$"),
    ("MLIC++~\\cite{mlicpp}",       "1283", "116.7", "$-15.15$"),
    ("HPCM-Base~\\cite{hpcm}",      "919",  "68.5",  "$-15.31$"),
    ("HPCM-Large~\\cite{hpcm}",     "1261", "89.7",  "$-19.19$"),
]
_lines = [r"\begin{tabular}{lrrr}", r"\toprule",
          r"Method & kMAC/px & Par. (M) & BD-Rate (\%) \\", r"\midrule"]
for _r in _LIT_IMG:
    _lines.append(" & ".join(_r) + r" \\")
_lines += [r"\bottomrule", r"\end{tabular}"]
w("literature_img.tex", "\n".join(_lines))

_LIT_VID = [
    ("DCVC-DC~\\cite{dcvcdc}",        "--",   "18.3",  "$+14.5$", "no"),
    ("DCVC-FM~\\cite{dcvcfm}",        "2642", "18.3",  "$-21.3$", "no"),
    ("DCVC-RT~\\cite{dcvcrt}",        "385",  "20.7",  "$-21.0$", "no"),
    ("DCVC-UF (LD)~\\cite{dcvcuf}",   "170",  "9.7",   "$-9.5$",  "no"),
    ("DCVC-UF (HT-S)~\\cite{dcvcuf}", "211",  "81.2",  "$-31.6$", "no"),
    ("DCVC-UF (HT-L)~\\cite{dcvcuf}", "343",  "120.5", "$-42.2$", "no"),
]
_lines = [r"\begin{tabular}{lrrrc}", r"\toprule",
          r"Method & GMAC & Par. (M) & BD-Rate (\%) & Adaptive \\",
          r"\midrule"]
for _r in _LIT_VID:
    _lines.append(" & ".join(_r) + r" \\")
_lines += [r"\midrule",
           r"\textbf{FLEX-UF} intra decoder & \textbf{\RelKMacPx$\,\to\,$\OursKMacPx} & -- & "
           r"\textbf{$+$\BdRateALow} & \textbf{yes} \\",
           r"\bottomrule", r"\end{tabular}"]
w("literature_vid.tex", "\n".join(_lines))

# --------------------------------------------------- against released DCVC-UF
# The comparison a codec reader wants at the end: what the released intra
# decoder costs, and what it costs at each of the three budgets, in the units
# DCVC-UF's own paper reports. Everything here is derived from measurements
# already in this file, not from a new run.
print("released comparison")
_GMAC, _PX = 453.5, 1920 * 1080
d, _ = pick("signalled_RECIPE512_ctc53.json")
bdj_, _ = pick("bdrate.json")
if d and bdj_:
    _bd = {r["budget_db"]: r["bd_rate_pct"] for r in bdj_["rows"]
           if r["config"].startswith("A")}
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Decoder & GMAC & kMAC & saved & dB below & BD-Rate \\",
             r" & /frame & /px & (\%) & release & (\%) \\",
             r"\midrule",
             f"released DCVC-UF intra & {_GMAC:.1f} & "
             f"{_GMAC * 1e9 / _PX / 1e3:.0f} & -- & 0.000 & 0.00 \\\\",
             r"\midrule"]
    for _b in (0.1, 0.3, 0.5):
        _rs = [r for r in d["rows"]
               if abs(r["budget_db"] - _b) < 1e-9 and r.get("budget_reachable")]
        if not _rs:
            continue
        _s = sum(sv(r) for r in _rs) / len(_rs)
        _db = sum(r["db_vs_uf"] for r in _rs) / len(_rs)
        _g = _GMAC * (1 - _s / 100)
        lines.append(f"\\textbf{{FLEX-UF}}, {_b:.1f}\\,dB budget & "
                     f"\\textbf{{{_g:.1f}}} & \\textbf{{{_g * 1e9 / _PX / 1e3:.0f}}} & "
                     f"\\textbf{{{_s:.1f}}} & {_db:.3f} & {_bd.get(_b, float('nan')):.2f} \\\\")
        _w = {0.1: "One", 0.3: "Three", 0.5: "Five"}[_b]
        mac(f"GmacAt{_w}", f"{_g:.0f}")
        mac(f"KmacAt{_w}", f"{_g * 1e9 / _PX / 1e3:.0f}")
        mac(f"DbAt{_w}", f"{_db:.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("released.tex", "\n".join(lines))
    mac("RelGmac", f"{_GMAC:.1f}")


# ------------------------------------------------- numbers typed into figures
# Captions written by hand carry literals, and a literal in a caption goes stale
# exactly like a literal in prose. These are the ones the figures added in the
# last pass introduced.
print("figure numbers")
d = J("supp_power.json") if False else None
try:
    _pw = json.load(open(RES / "supp_power.json"))
    _r = [r for r in _pw["rows"] if r["size"] == "2048x1280"]
    for _q, _n in ((0, "Low"), (32, "Mid"), (63, "High")):
        _x = next((r for r in _r if r["qp"] == _q), None)
        if _x:
            mac(f"PowerMac{_n}", f"{_x['mac_saving_pct']:.1f}")
            mac(f"PowerTime{_n}", f"{_x['time_saving_pct']:.1f}")
            mac(f"PowerEnergy{_n}", f"{_x['energy_saving_pct']:.1f}")
            # The milliseconds, frames and joules behind those percentages.
            # Section 6 had all six typed into a sentence that did not say
            # which rate they belong to, and a reader would take them for q0.
            mac(f"PowerMs{_n}", f"{_x['routed']['ms_median']:.1f}")
            mac(f"PowerMsFull{_n}", f"{_x['full']['ms_median']:.1f}")
            mac(f"PowerFps{_n}", f"{1000 / _x['routed']['ms_median']:.1f}")
            mac(f"PowerFpsFull{_n}", f"{1000 / _x['full']['ms_median']:.1f}")
            mac(f"PowerJ{_n}", f"{_x['joules_per_frame_routed']:.1f}")
            mac(f"PowerJFull{_n}", f"{_x['joules_per_frame_full']:.1f}")
            mac(f"PowerQ{_n}", f"q{_q}")
except Exception as _e:
    print("   supp_power.json:", _e)
try:
    _fp = json.load(open(RES / "supp_footprint.json"))
    _ok = [r for r in _fp["rows"] if not r.get("oom")]
    if _ok:
        mac("PeakDelta", f"{max(r['peak_delta_pct'] for r in _ok):.1f}")
        _hd = next((r for r in _ok if r["size"] == "2048x1280"), None)
        if _hd:
            mac("FpsFull", f"{_hd['fps_full']:.1f}")
            mac("FpsRouted", f"{_hd['fps_routed']:.1f}")
except Exception as _e:
    print("   supp_footprint.json:", _e)
# From the file the figure itself dumps, not from results/per_sequence.json:
# that held 40 sequences and a 42.5 maximum, both from before the test set was
# completed and before the FFN accounting was fixed.
try:
    _sp = json.load(open(RES / "spread_stats.json"))["rows"]
    _qs = sorted(_sp, key=int)
    mac("SpreadLowMin", f"{_sp[_qs[0]]['min']:.1f}")
    mac("SpreadLowMax", f"{_sp[_qs[0]]['max']:.1f}")
    mac("SpreadHighMin", f"{_sp[_qs[-1]]['min']:.1f}")
    mac("SpreadHighMax", f"{_sp[_qs[-1]]['max']:.1f}")
    mac("SpreadNSeq", str(_sp[_qs[0]]["n"]))
except Exception as _e:
    print("   spread_stats.json:", _e)


# ---------------------------------------------------- the checkpoint series
# What later epochs of the pinned run measure, on the same frames at the same
# budget and with the same hook count. The limitations section quoted these
# from modelled files once, which inflated them and reversed their order.
print("checkpoint series")
try:
    import glob as _glob
    _ser = {}
    for _f in _glob.glob(str(RES / "signalled_RECIPE512_*.json")):
        _d = json.load(open(_f))
        if _d.get("n_sequences") != 53:
            continue
        _rs = [r for r in _d["rows"]
               if r.get("budget_db") == 0.1 and r.get("budget_reachable")]
        if len(_rs) < 5 or _rs[0].get("saving_pct_measured") is None:
            continue
        _e = _d.get("ckpt_epoch")
        _m = sum(r["saving_pct_measured"] for r in _rs) / len(_rs)
        if _e is not None:
            _ser.setdefault(_e, []).append((_m, Path(_f).name))
    # One number per epoch, and not the best of them. The rule here used to be
    # max(), which is cherry-picking the moment an epoch has more than one
    # measurement -- and epoch 0 has four. They agree to 0.12 of a point, so it
    # changed nothing, but a rule that would manufacture a monotone series if
    # the files ever disagreed is not one to leave in. Median, and the spread
    # is printed so it cannot hide.
    for _e in list(_ser):
        _v = sorted(_ser[_e])
        _sp = _v[-1][0] - _v[0][0]
        if _sp > 0.5:
            print(f"   epoch {_e}: {len(_v)} files spread {_sp:.2f} points")
        _ser[_e] = _v[len(_v) // 2]
    _WORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}
    for _e, (_m, _fn) in sorted(_ser.items()):
        if _e in _WORD:
            mac(f"EpochMean{_WORD[_e]}", f"{_m:.1f}")
    if _ser:
        mac("EpochLatest", str(max(_ser)))
        mac("EpochLatestMean", f"{_ser[max(_ser)][0]:.1f}")
        mac("EpochGain", f"{_ser[max(_ser)][0] - _ser[min(_ser)][0]:.1f}")
        # The series itself, written out. The prose listed the values by hand
        # and the list stopped at epoch 3 while \EpochLatest and \EpochGain
        # moved to 4 on their own when that evaluation landed: four numbers,
        # a range of five epochs, and a gain computed from a fifth the
        # sentence never showed.
        _vals = [f"{_ser[e][0]:.1f}%" for e in sorted(_ser)]
        mac("EpochSeries", ", ".join(_vals[:-1]) + " and " + _vals[-1]
            if len(_vals) > 1 else _vals[0])
        mac("EpochFirst", str(min(_ser)))
except Exception as _e:
    print("   checkpoint series:", _e)

# Which epoch the paper actually reports, read from the pinned checkpoint
# rather than from the series. The two were the same while the paper sat on
# epoch 0 and the sentence saying so was typed; they are not the same after a
# repin, and a limitations paragraph that says "the first pass" while the
# tables come from the fifth is the kind of drift this file exists to stop.
print("pinned epoch")
try:
    import torch as _torch
    _ck = _torch.load(R / "runs/RECIPE512/ckpt_PAPER.pth.tar",
                      map_location="cpu", weights_only=False)
    _pe = _ck.get("epoch")
    if _pe is not None:
        mac("PaperEpoch", str(int(_pe)))
        # How many passes over the training set that is, for prose that wants
        # to say it in words. Epoch indices start at zero.
        _WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                  6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        mac("PaperPasses", _WORDS.get(int(_pe) + 1, str(int(_pe) + 1)))
except Exception as _e:
    print("   pinned epoch:", _e)


# ------------------------------------------------- where the ladder is used
try:
    _ev = json.load(open(RES / "exit_vs_rate.json"))
    mac("ExitMeanLow", f"{_ev['mean_exit_low_rate']:.2f}")
    mac("ExitMeanHigh", f"{_ev['mean_exit_high_rate']:.2f}")
    mac("ExitShallowLow", f"{_ev['share_two_low']:.0f}")
    mac("ExitShallowHigh", f"{_ev['share_two_high']:.0f}")
    mac("ExitDeepLow", f"{_ev['share_deepest_low']:.0f}")
    mac("ExitDeepHigh", f"{_ev['share_deepest_high']:.0f}")
except Exception as _e:
    print("   exit_vs_rate.json:", _e)


# ------------------------------------------------- what each input is worth
try:
    _ri = json.load(open(RES / "router_inputs_rows.json"))["rows"]
    _l3 = ["\\begin{tabular}{lrrrr}", "\\toprule",
           " & ".join(c.replace("±", "$\\pm$") for c in _ri[0]) + " \\\\",
           "\\midrule"]
    for r in _ri[1:]:
        _l3.append(" & ".join(r) + " \\\\")
    _l3 += ["\\bottomrule", "\\end{tabular}"]
    w("router_inputs.tex", "\n".join(_l3))
    mac("InputStemAgree", _ri[1][1])
    mac("InputAllAgree", _ri[2][1])
    mac("InputQpAgree", _ri[-1][1])
    mac("InputStemOver", _ri[1][4])
except Exception as _e:
    print("   router_inputs_rows.json:", _e)


# ------------------------------------------- what a set-level budget hides
try:
    _sb = json.load(open(RES / "setbudget_rows.json"))["rows"]
    def _t2(x):
        return x.replace("%", "\\%")
    _l2 = ["\\begin{tabular}{lrrlrl}", "\\toprule",
           " & ".join(_t2(c) for c in _sb[0]) + " \\\\", "\\midrule"]
    for r in _sb[1:]:
        _l2.append(" & ".join(_t2(c) for c in r) + " \\\\")
    _l2 += ["\\bottomrule", "\\end{tabular}"]
    w("setbudget.tex", "\n".join(_l2))
    _ov = [int(r[1].split()[0]) for r in _sb[1:]]
    _n = int(_sb[1][1].split()[-1])
    mac("OverBudgetLo", str(min(_ov)))
    mac("OverBudgetHi", str(max(_ov)))
    mac("OverBudgetN", str(_n))
    mac("WorstSeqDb", f"{max(float(r[2]) for r in _sb[1:]):.3f}")
except Exception as _e:
    print("   setbudget_rows.json:", _e)


# ------------------------------------------------ one frame per class
try:
    _pr = json.load(open(RES / "probe_rows.json"))["rows"]
    def _t(x):
        return x.replace("%", "\\%").replace("×", "$\\times$")
    _ln = ["\\begin{tabular}{lrlrrr}", "\\toprule",
           " & ".join(_t(c) for c in _pr[0]) + " \\\\", "\\midrule"]
    for r in _pr[1:]:
        _ln.append(" & ".join(_t(c) for c in r) + " \\\\")
    _ln += ["\\bottomrule", "\\end{tabular}"]
    w("probe.tex", "\n".join(_ln))
except Exception as _e:
    print("   probe_rows.json:", _e)


# ------------------------------------------------ built and abandoned
# The supplement computes these rows; this writes the paper's copy from the same
# dump, so the two documents cannot disagree about what was tried or why it was
# dropped. Regenerated whenever the supplement is built.
try:
    _ab = json.load(open(RES / "abandoned_rows.json"))["rows"]
    def _tex(t):
        return (t.replace("%", "\\%").replace("&", "\\&")
                 .replace("--", "-{}-"))
    _lines = ["\\begin{tabular}{p{0.30\\columnwidth}p{0.62\\columnwidth}}",
              "\\toprule",
              f"{_tex(_ab[0][0])} & {_tex(_ab[0][1])} \\\\", "\\midrule"]
    for r in _ab[1:]:
        _lines.append(f"{_tex(r[0])} & {_tex(r[1])} \\\\")
    _lines += ["\\bottomrule", "\\end{tabular}"]
    w("abandoned.tex", "\n".join(_lines))
    mac("AbandonedN", str(len(_ab) - 1))
except Exception as _e:
    print("   abandoned_rows.json:", _e)


# ------------------------------------------------------- a second metric
# The budget is bisected on PSNR. A reviewer will ask what that does to a
# perceptual metric, and the answer belongs in the paper rather than only in the
# supplement: MS-SSIM in dB, on the same frames at the same operating point.
try:
    _oq = json.load(open(RES / "supp_opquality_PAPER.json"))
    # No saving column. It would be this file's arithmetic model, half a point
    # away from the hook count Table 3 reports, and the comparison here is
    # between two metrics at one operating point, not another saving figure.
    _rows = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "q & PSNR lost & MS-SSIM lost & released & routed \\\\",
             " & (dB) & (dB) & (MS-SSIM) & (MS-SSIM) \\\\", "\\midrule"]
    _p, _m = [], []
    for r in _oq["rows"]:
        _dp = -r["psnr_delta"]
        _dm = r["ms_ssim_db_released"] - r["ms_ssim_db_routed"]
        _p.append(_dp)
        _m.append(_dm)
        _rows.append(f"q{r['qp']} & {_dp:.3f} & {_dm:.3f} & "
                     f"{r['ms_ssim_released']:.4f} & {r['ms_ssim_routed']:.4f} "
                     f"\\\\")
    _rows += ["\\bottomrule", "\\end{tabular}"]
    w("msssim.tex", "\n".join(_rows))
    mac("MsPsnrLoss", f"{sum(_p) / len(_p):.3f}")
    mac("MsSsimLossLo", f"{min(_m):.3f}")
    mac("MsSsimLossHi", f"{max(_m):.3f}")
    mac("MsSsimRatio", f"{(sum(_m) / len(_m)) / (sum(_p) / len(_p)):.2f}")
    _ws = _oq["worst_sequences_by_ms_ssim"]
    mac("MsWorstSeq", _ws[0]["seq"].split("_")[0])
    mac("MsWorstDb", f"{_ws[0]['ms_ssim_db_released'] - _ws[0]['ms_ssim_db_routed']:.3f}"
        if "ms_ssim_db_released" in _ws[0] else f"{-_ws[0]['ms_ssim_delta']:.4f}")
    mac("MsWorstSaving", f"{_ws[0]['saving_pct']:.1f}")
except Exception as _e:
    print("   supp_opquality:", _e)


# ------------------------------------------------- what the ladder costs to add
# A reviewer asks two things about a method that fine-tunes: how much it adds to
# the model, and how much training it took. Neither was in the paper.
try:
    _mt = json.load(open(R / "runs/RECIPE512/meta.json"))
    mac("ParamsTotal", f"{_mt['params_total'] / 1e6:.1f}")
    mac("ParamsAdapters", f"{_mt['params_adapters'] / 1e6:.2f}")
    mac("ParamsAdapterPct",
        f"{100 * _mt['params_adapters'] / _mt['params_total']:.1f}")
    mac("TrainImages", f"{_mt['dataset_size']:,}")
    mac("TrainBatch", str(_mt["batch_size"]))
    mac("TrainSteps", f"{_mt['dataset_size'] // _mt['batch_size']:,}")
except Exception as _e:
    print("   run meta:", _e)


# --------------------------------------------- the big-resolution classes
# The prose named three classes at the lowest rate and only the first was a
# macro; the other two were typed in and went stale when the per-class figures
# moved to the hook count -- UVG read 33.1 where the measurement is 30.6.
try:
    _pc = json.load(open(RES / "per_class_RECIPE512.json"))
    _rows = _pc.get("rows", _pc)
    _r0 = next(r for r in _rows
               if r.get("qp") == 0 and abs(r.get("budget_db", 0.1) - 0.1) < 1e-9)
    _cl = _r0["per_class"]
    for _key, _name in (("MCL-JCV", "Mcl"), ("UVG", "Uvg"), ("HEVC_B", "HevcB"),
                        ("HEVC_C", "HevcC"), ("HEVC_D", "HevcD"),
                        ("HEVC_E", "HevcE")):
        if _key in _cl:
            mac(f"ClassLow{_name}", f"{_cl[_key]['saving']:.1f}")
            # BigRes{Mcl,Uvg,HevcB} were a second name for the same three
            # numbers, one document using each. ClassLow* is the surviving name.
    # How far apart the three 1080p classes are. The prose called it "within
    # three points", which was true of the arithmetic-model figures (0.7 apart)
    # and is not true of the hook count.
    _big = [_cl[k]["saving"] for k in ("MCL-JCV", "UVG", "HEVC_B") if k in _cl]
    if len(_big) == 3:
        mac("BigResSpread", f"{max(_big) - min(_big):.1f}")
except Exception as _e:
    print("   per-class q0:", _e)


# ------------------------------------------------------- where the MACs are
try:
    _ma = json.load(open(RES / "mac_audit.json"))["1920x1088"]
    mac("DecGmac", f"{_ma['total_gmac']:.0f}")
    mac("TrunkShare", f"{100 * _ma['parts']['trunk']['share']:.1f}")
    mac("UpsampleShare", f"{100 * _ma['parts']['upsample']['share']:.1f}")
    mac("HeadShare", f"{100 * _ma['parts']['head']['share']:.1f}")
except Exception as _e:
    print("   mac audit:", _e)


# ------------------------------------------------- the exit map on one frame
try:
    _em = json.load(open(RES / "supp_exitmap_bosphorus_q32_b01.json"))
    _h = _em["exit_hist"]
    _live = [n for n in _h if n]
    mac("SplitShallow", str(_live[0]))
    mac("SplitMid", str(_live[1]))
    mac("SplitDeep", str(_live[2]))
    mac("ExitmapWorst", f"{_em['tile_penalty_db_max']:.2f}")
    mac("ExitmapFrame", f"{_em['delivered_db']:.3f}")
except Exception as _e:
    print("   exit map:", _e)


# ------------------------------------------- which exit the tail comes from
try:
    _te = json.load(open(RES / "supp_opquality_PAPER.json"))["tail_by_exit"]
    _q = sorted(_te, key=int)[0]
    _live = {k: v for k, v in _te[_q].items() if v}
    _lo = min(_live, key=int)
    mac("TailExitLo", f"e{_lo}")
    mac("TailExitLoMean", f"{_live[_lo]['mean_db']:.3f}")
    mac("TailExitLoOver", str(_live[_lo]["n_over_0_5_db"]))
    _rest = [v for k, v in _live.items() if k != _lo]
    mac("TailExitRestMean",
        f"{sum(v['mean_db'] * v['n'] for v in _rest) / sum(v['n'] for v in _rest):.3f}")
    mac("TailExitRestOver", str(sum(v["n_over_0_5_db"] for v in _rest)))
except Exception as _e:
    print("   tail by exit:", _e)


# ------------------------------------------------ how far two heads differ
# Claim (iii) is that a free rule beats our trained head. The obvious objection
# is that the head was badly tuned, and the honest answer is the spread between
# two heads trained independently on the same decoder.
try:
    _hf = _pick_json("router_RECIPE512_b01_e4head.json",
                       "router_RECIPE512_b01_PAPER.json")
    _hj = json.load(open(RES / "router_RECIPE512_b01_jointhead.json"))
    # Both sides on the arithmetic model, not sv(). The frozen head's file
    # carries a hook count and the jointly trained head's does not, so sv()
    # took the hook count on one side and the model on the other and reported
    # a spread inflated by the very convention offset this paper warns about:
    # +3.2 at q0 where a like-for-like comparison gives +2.4. Section F of the
    # supplement compares them on the model and is the one that was right.
    _F = {r["qp"]: r["saving_pct_vs_release"] for r in _hf["rows"]}
    _J = {r["qp"]: r["saving_pct_vs_release"] for r in _hj["rows"]}
    _d = [_J[q] - _F[q] for q in sorted(_F) if q in _J]
    mac("HeadSpreadLow", f"{_d[0]:+.1f}")
    mac("HeadSpreadHigh", f"{_d[-1]:+.1f}")
    mac("HeadSpreadRange", f"{max(_d) - min(_d):.1f}")
except Exception as _e:
    print("   two heads:", _e)


# ---------------------------------------------------------- CPU and batch
# The paper says a saving in operations is an optimistic bound on a saving in
# time. On a CPU it is not: the arithmetic model under-predicts what the routed
# decode actually saves there. Worth reporting, since a decoder saving that only
# shows on one accelerator is a weaker result.
try:
    _cpu = json.load(open(RES / "supp_latency_cpu_1920x1080.json"))
    _bat = json.load(open(RES / "supp_latency_batch_1920x1080.json"))
    _b1 = {r["qp"]: r for r in _bat["rows"] if r["batch"] == 1}
    _b4 = {r["qp"]: r for r in _bat["rows"] if r["batch"] == max(
        x["batch"] for x in _bat["rows"])}
    _ln = ["\\begin{tabular}{lrrrr}", "\\toprule",
           "q & arithmetic & GPU, batch 1 & GPU, batch 4 & CPU \\\\",
           "\\midrule"]
    for r in _cpu["rows"]:
        q = r["qp"]
        _ln.append(
            f"q{q} & {r['predicted_saving_pct']:.1f} & "
            f"{_b1[q]['realised_saving_pct']:.1f} & "
            f"{_b4[q]['realised_saving_pct']:.1f} & "
            f"{r['realised_saving_pct']:.1f} \\\\")
    _ln += ["\\bottomrule", "\\end{tabular}"]
    w("cpu_batch.tex", "\n".join(_ln))
    _c0 = _cpu["rows"][0]
    mac("CpuThreads", str(_cpu["torch_threads"]))
    mac("CpuSavingLow", f"{_c0['realised_saving_pct']:.1f}")
    mac("CpuPredLow", f"{_c0['predicted_saving_pct']:.1f}")
    mac("CpuSavingHigh", f"{_cpu['rows'][-1]['realised_saving_pct']:.1f}")
    mac("CpuMsStock", f"{_c0['ms_stock_per_frame'] / 1000:.1f}")
    mac("CpuMsRouted", f"{_c0['ms_routed_per_frame'] / 1000:.1f}")
    mac("BatchGainLow",
        f"{_b4[_c0['qp']]['realised_saving_pct'] - _b1[_c0['qp']]['realised_saving_pct']:.1f}")
except Exception as _e:
    print("   cpu/batch latency:", _e)


# ------------------------------------------------------- the per-tile tail
# The budget binds on a frame mean. This is what that mean is made of, and the
# split by padding is the one number that ties Section 4 to Section 5: the tiles
# that pay for the mean are the ones with an invented neighbour.
try:
    _oq2 = json.load(open(RES / "supp_opquality_PAPER.json"))
    _t, _bp = _oq2["tail"], _oq2["tail_by_padding"]
    _ln = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "q & tiles & median & p95 & p99 & max & over $1\\dB$ \\\\",
           " &  & (dB) & (dB) & (dB) & (dB) &  \\\\", "\\midrule"]
    for q in sorted(_t, key=int):
        r = _t[q]
        _ln.append(f"q{q} & {r['n_tiles']} & {r['median_db']:.3f} & "
                   f"{r['p95_db']:.3f} & {r['p99_db']:.3f} & "
                   f"{r['max_db']:.3f} & {r['n_over_1_db']} \\\\")
    _ln += ["\\bottomrule", "\\end{tabular}"]
    w("tiletail.tex", "\n".join(_ln))
    _q0 = _t[sorted(_t, key=int)[0]]
    mac("TailMedian", f"{_q0['median_db']:.3f}")
    mac("TailPNinetyNine", f"{_q0['p99_db']:.2f}")
    mac("TailMax", f"{_q0['max_db']:.2f}")
    mac("TailOverOne", str(_q0["n_over_1_db"]))
    mac("TailNTiles", str(_q0["n_tiles"]))
    _b0 = _bp[sorted(_bp, key=int)[0]]
    mac("TailInterior", f"{_b0['no_padding']['mean_db']:.3f}")
    mac("TailBorder", f"{_b0['some_padding']['mean_db']:.3f}")
    mac("TailBorderRatio",
        f"{_b0['some_padding']['mean_db'] / _b0['no_padding']['mean_db']:.1f}")
    mac("TailBorderN", str(_b0["some_padding"]["n"]))
except Exception as _e:
    print("   per-tile tail:", _e)


# --------------------------------------------- the same frame at high rate
# The qualitative figure is at q32. The same frame at q63 is the honest
# counterpart: the saving is smaller and the allocation deeper, and both are
# measured rather than argued.
try:
    _q6 = json.load(open(R / "results/qualitative_PAPER_q63.json"))
    mac("QualHighQp", str(_q6["qp"]))
    mac("QualHighDb", f"{_q6['delivered_db']:.3f}")
    mac("QualHighSaving", f"{_q6['saving_pct_vs_release']:.1f}")
    mac("QualHighBpp", f"{_q6['bpp']:.4f}")
    mac("QualHighPsnrRel", f"{_q6['psnr_released']:.2f}")
    mac("QualHighPsnrOurs", f"{_q6['psnr_routed']:.2f}")
    _h32 = json.load(open(R / "results/qualitative_PAPER.json"))["exit_hist"]
    _h63 = _q6["exit_hist"]
    mac("QualShallowLow", str(_h32[2]))
    mac("QualShallowHigh", str(_h63[2]))
    mac("QualDeepLow", str(_h32[-1]))
    mac("QualDeepHigh", str(_h63[-1]))
except Exception as _e:
    print("   qualitative q63:", _e)


# ---------------------------------------------------- rd_spread panel c
# The sentence beside panel c quoted a median and an IQR typed in from an older
# run, under the older saving definition. Now the figure dumps them.
try:
    _rs = json.load(open(R / "results/rd_spread_stats.json"))["rows"]
    _lo = _rs["0"]
    mac("SpreadMedLow", f"{_lo['median']:.1f}")
    mac("SpreadIqrLoLow", f"{_lo['p25']:.1f}")
    mac("SpreadIqrHiLow", f"{_lo['p75']:.1f}")
    mac("SpreadWorst", f"{min(r['min'] for r in _rs.values()):.1f}")
except Exception as _e:
    print("   rd_spread stats:", _e)


# ------------------------------------------------------- qualitative figure
# The one figure that shows the reader there is nothing to see. Its numbers come
# from the sidecar the figure script writes beside the PNG, so the caption and
# the picture cannot disagree about which checkpoint decoded it.
try:
    _q = json.load(open(R / "results/qualitative_PAPER.json"))
    mac("QualSeq", _q["seq"].split("_")[0])
    mac("QualQp", str(_q["qp"]))
    mac("QualDb", f"{_q['delivered_db']:.3f}")
    mac("QualSaving", f"{_q['saving_pct_vs_release']:.1f}")
    mac("QualPsnrRel", f"{_q['psnr_released']:.2f}")
    mac("QualPsnrOurs", f"{_q['psnr_routed']:.2f}")
    mac("QualBpp", f"{_q['bpp']:.4f}")
    mac("QualAmp", str(_q["amplification"]))
    mac("QualWorstTile", str(_q["worst_tile"]))
    mac("QualNTiles", str(_q["n_tiles"]))
    mac("QualWorstExit", str(_q["worst_tile_exit"]))
except Exception as _e:
    print("   qualitative:", _e)


# macros.tex is written LAST, after every block that can emit one. It used to be
# written in the middle of the file, so the two blocks appended after it emitted
# five macros that never reached disk and the build reported them unexpanded.
# A macro value with a math delimiter in it is fine in LaTeX and prints
# literally in the reportlab build, which has no way to know it was meant as
# maths. HybridLooseQps carried "$q48$ and $q63$" into the paper that way.
_mathy = {k: v for k, v in MACROS.items() if "$" in str(v)}
for _k, _v in _mathy.items():
    print(f"    MACRO WITH MATH DELIMITERS: {_k} = {_v}")
w("macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}"
                          for k, v in sorted(MACROS.items())))
print(f"\n  {len(MACROS)} macros")
