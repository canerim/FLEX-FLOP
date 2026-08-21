"""Failure cases, limitations and open problems.

The supplement's limitations section, and the place the project's negative
results are kept. Three things in order: the content the method does worst
on and what a frame looks like when it does, every mechanism that was built and then dropped
with the measurement that ended it, and the problems that are open at
submission with no proposal attached to them.

Every number is read from a named file in `results/` at build time, so the
section cannot drift from the measurements it describes. Two exceptions are
typed, and each is in the block below with the file it was typed from, because
the file is a log rather than JSON and `k.J` cannot parse it. Where a saving is
quoted it is the hook count wherever the file carries one, by the same rule
`scripts/make_paper_tables.py` uses; the modelled figure is used only for the
single-frame probes, which carry no hook count, and is marked where it appears.

This is where the supplement's open problems are collected, so that they are
stated once and in the place a reviewer looks for them. What this section
leaves to its neighbours: section D names five failure cases from the
operating-point files and tabulates the per-sequence and per-tile spread behind
every mean; section E measures how little of the budget-band exponent
replicates; section F reports what the router head cannot do. None of that is
repeated. What is here is the content axis rather than the rate axis, the
abandoned work, the state of the checkpoints, and the list of what is open.
"""

# ---------------------------------------------------------------------------
# Typed constants. Each block names the file it was read from. These are the
# only numbers in the section that k.J cannot supply, because the file is a
# log rather than JSON.
# ---------------------------------------------------------------------------

# results/first_ckpt_headsonly.log, the first checkpoint of
# runs/heads_only_j2_p256, 10 CTC frames. The abandoned-mechanism table
# uses these.
HEADS_ONLY = {
    # drift of the deepest exit from the released decoder, in dB
    "drift": {0: 0.000, 32: 0.000, 63: 0.000},
    # the Lagrangian oracle's saving at a 0.1 dB budget, per cent, and the
    # best single depth at the same quality
    "oracle": {0: 11.5, 32: 0.4, 63: -1.0},
    "uniform": {0: 4.7, 32: -0.7, 63: -1.0},
}

# results/shrink_ablation.log, 10 CTC frames, j = 2 at 128 px. Seam in dB with
# a fixed shrink coefficient in place of the fitted AR(1) one.
SHRINK = {
    "replicate": {0: 0.1288, 32: 0.1771, 63: 0.2160},
    "shrink0.95": {0: 0.1228, 32: 0.1774, 63: 0.2138},
    "shrink0.80": {0: 0.1356, 32: 0.2009, 63: 0.2279},
    "arls": {0: 0.1082, 32: 0.1473, 63: 0.1797},
}

PINNED = "runs/RECIPE512/ckpt_PAPER.pth.tar"

#: The five single-frame exit-map probes, largest tile count first. Each pairs
#: a results file with the figure drawn from it.
PROBES = [
    ("supp_exitmap_bosphorus_q32_b01.json", "Bosphorus"),
    ("supp_exitmap_basketballdrive_q32_b01.json", "BasketballDrive"),
    ("supp_exitmap_fourpeople_q32_b01.json", "FourPeople"),
    ("supp_exitmap_partyscene_q32_b01.json", "PartyScene"),
    ("supp_exitmap_racehorses_416x240_q32_b01.json", "RaceHorses"),
]

#: The checkpoints of one epoch of one run, all on \NumSeq sequences at the
#: same budget, all carrying a hook count. Named rather than globbed: the
#: watcher that wrote them is still running and a glob would change the table
#: between one build and the next.
FINE12_EPOCH2 = [
    "signalled_FINE12_0819_0923.json",
    "signalled_FINE12_0819_1239.json",
    "signalled_FINE12_0819_1555.json",
    "signalled_FINE12_0819_1911.json",
    "signalled_FINE12_0819_2157.json",
    "signalled_FINE12_0819_2227.json",
]

#: FINE12 reaches the 0.1 dB budget at four of the five rates and the fifth is
#: below its floor, so a mean over it is a mean over these four.
FINE12_QPS = [0, 16, 32, 48]

QPS = [0, 16, 32, 48, 63]


# --------------------------------------------------------------- arithmetic
def _sv(row, key="saving_pct"):
    """The saving a row reports, counted off the decode where it can be.

    The same rule scripts/make_paper_tables.py applies, so a figure here and
    the same figure in a generated table cannot differ by the cost model's
    offset. Returns None where the row reports neither, which is what a budget
    below that rate's floor looks like.
    """
    m = row.get(key + "_measured")
    return m if m is not None else row.get(key + "_vs_release")


def _modelled(doc, budget=0.1):
    """{rate: saving} in the cost model's denominator alone.

    Two files that disagree about whether they carry a hook count cannot be
    subtracted through _at without the model's offset landing in the answer.
    Where the point of a comparison is something other than the saving, both
    sides are read here instead, and the note says so.
    """
    return {r["qp"]: r["saving_pct_vs_release"] for r in doc["rows"]
            if abs(r.get("budget_db", -1) - budget) < 1e-9
            and r.get("saving_pct_vs_release") is not None}


def _at(doc, budget=0.1):
    """{rate: saving} for one budget of a signalled sweep, unreachable rates
    dropped."""
    out = {}
    for r in doc["rows"]:
        if abs(r.get("budget_db", -1) - budget) > 1e-9:
            continue
        v = _sv(r)
        if v is not None:
            out[r["qp"]] = v
    return out


def _mean(d, qps):
    vs = [d[q] for q in qps if q in d]
    return sum(vs) / len(vs) if vs else None


def _msssim_db(x):
    """MS-SSIM as a decibel, the convention section D states: −10 log10(1−x)."""
    import math
    return -10.0 * math.log10(max(1.0 - x, 1e-12))


def _powerfit(xs, ys):
    """Least squares on log y against log x. Returns exponent, correlation and
    the mean relative error of the fitted law."""
    import math
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    sxx = sum((a - mx) ** 2 for a in lx)
    syy = sum((b - my) ** 2 for b in ly)
    a = sxy / sxx
    c = my - a * mx
    r = sxy / math.sqrt(sxx * syy)
    err = sum(abs(math.exp(c) * x ** a - y) / y for x, y in zip(xs, ys)) / n
    return a, r, err


def _scalefit(xs, ys):
    """One free scale on a fixed shape. Returns the mean relative error."""
    s = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    return sum(abs(s * x - y) / y for x, y in zip(xs, ys)) / len(xs)


def _f(x, n=2):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ section
def content(k):
    k.h1("Failure cases, limitations and open problems")

    ps = k.J("supp_per_sequence_PAPER_b010.json")
    oq = k.J("supp_opquality_PAPER.json")
    cm = k.J("ceiling_measured.json")
    sig = k.J("signalled_RECIPE512_ctc53.json")
    anc = k.J("supp_anchor_PAPER.json")

    model_offset = (cm["ceiling_modelled_pct"] - cm["ceiling_measured_pct"])

    k.par(
        "The method removes decoder compute by letting easy tiles leave the "
        "trunk early, and there are frames where no tile is easy, frames with "
        "too few tiles to allocate over, and content where the quality a "
        "budget buys is spent in a place a viewer would look. This section "
        "measures those cases rather than describing them, then lists the "
        "mechanisms that were built for this work and dropped, each with the "
        "measurement that ended it, and closes on the problems that are open "
        "at submission. Section D names five failure cases read off the "
        "operating-point files and is not repeated; what follows is the "
        "content axis, the abandoned work, and the state of the checkpoints "
        "every table in this document rests on.")

    # ------------------------------------------------------------------ H.1
    k.h2("The frames it does worst on")

    probes = [(k.J(f), name) for f, name in PROBES]
    small = probes[-1][0]
    big = probes[0][0]

    k.par(
        "A frame is cut into tiles of 256 px, so the number of decisions the "
        "allocation has to make is fixed by the resolution. At 1920×1080 a "
        "padded frame carries 40 tiles and at 416×240 it carries two. Table " +
        str(k.peek_tbl()) + " takes one frame from each of the five test "
        "classes at the same rate and the same budget and prints what the "
        "allocation did with it. Reading down the tile column, the exit map "
        "goes from a mixture over three exits to a constant, and the saving "
        "falls with it.")

    rows = [["frame", "tiles", "exits taken", "saved %", "delivered dB",
             "worst tile dB"]]
    for d, name in probes:
        hist = d["exit_hist"]
        used = " ".join(f"e{i}:{n}" for i, n in enumerate(hist) if n)
        rows.append([f"{name}, {d['resolution'][0]}×{d['resolution'][1]}",
                     str(d["n_tiles"]), used,
                     _f(d["saving_pct_vs_release"], 1),
                     _f(d["delivered_db"], 3),
                     _f(d["tile_penalty_db_max"], 3)])
    # Dumped for the main paper's copy of this table; see the note beside the
    # abandoned-mechanism dump below.
    import json as _j
    from pathlib import Path as _Pp
    _j.dump({"rows": rows},
            open(_Pp(__file__).resolve().parents[2] / "results/probe_rows.json",
                 "w"), indent=2)

    t_probe = k.rows(rows,
           "<b>One frame from each class at q32 and a 0.1 dB budget.</b> The "
           "exits taken column is the histogram over the ladder, so "
           "<i>e4:2</i> is two tiles at exit 4. Take from it that the "
           "allocation stops being an allocation at the bottom of the table: "
           "at 416×240 both tiles take the same exit, and the frame receives "
           "whatever that one rung happens to cost. The worst tile column is "
           "the largest per-tile loss in the frame and is between two and "
           "three times the frame's own delivered decibel on the three "
           "largest frames, which is the tail section D measures over the "
           "whole set.")
    k.note("The five results/supp_exitmap_*_q32_b01.json files, all on " +
           PINNED + ", one frame each. These probes carry the modelled "
           f"saving and no hook count, so each is optimistic by up to "
           f"{model_offset:.1f} points, the constant recorded in "
           f"results/ceiling_measured.json; they are printed for the "
           f"allocation and the per-tile loss rather than for the headline, "
           f"which section D reports hook-counted over the whole set.")

    k.fig("exitmap_PAPER_racehorses_416x240_q32_b01.png",
          "<b>The smallest frame in the test set, and the whole of its "
          "allocation.</b> <b>a</b>, the decoded frame with the single tile "
          "boundary drawn, tinted by the exit each tile took. <b>b</b>, the "
          "exit map, two cells. <b>c</b>, the loss each tile pays. Both tiles "
          "take exit 4 and the map has nothing to trade; the legend gives "
          "what each rung would have saved had a tile taken it.",
          maxh=80)
    k.note("paper/figures/exitmap_PAPER_racehorses_416x240_q32_b01.png, drawn "
           "from results/supp_exitmap_racehorses_416x240_q32_b01.json on " +
           PINNED + ". RaceHorses at 416×240, q32, 0.1 dB.")

    k.par(
        f"The two frames at either end of that table are the same decoder at "
        f"the same budget. Bosphorus at 1080p spreads 40 tiles over exits "
        f"{', '.join(str(i) for i, n in enumerate(big['exit_hist']) if n)} and "
        f"saves {big['saving_pct_vs_release']:.1f}% of the modelled decode; "
        f"RaceHorses at 416×240 puts both of its tiles on exit "
        f"{small['exit_map_row_major'][0]} and saves "
        f"{small['saving_pct_vs_release']:.1f}%. Nothing about the content "
        f"separates them. What separates them is that a two-cell map has "
        f"three admissible states above the split depth and a 40-cell map has "
        f"more states than the sweep could enumerate, so the Lagrangian has "
        f"somewhere to put the budget in one case and not in the other.")

    def _msloss(r):
        return _msssim_db(r["ms_ssim_released"]) - _msssim_db(r["ms_ssim_routed"])

    ms = sorted(oq["worst_sequences_by_ms_ssim"], key=_msloss,
                reverse=True)[:3]
    q0row = [r for r in ps["rows"] if r["qp"] == 0][0]
    q0sav = sorted(x["saving_pct"] for x in q0row["per_sequence"])
    setrow = [r for r in oq["rows"] if r["qp"] == 0][0]
    setloss = setrow["ms_ssim_db_released"] - setrow["ms_ssim_db_routed"]

    def _pct(v):
        return 100.0 * sum(1 for a in q0sav if a < v) / len(q0sav)

    rows = [["sequence", "q", "tiles", "PSNR lost dB", "MS-SSIM lost dB",
             "saved %", "its rank"]]
    for r in ms:
        rows.append([r["seq"].split("_")[0], f"q{r['qp']}", str(r["tiles"]),
                     _f(-r["psnr_delta"], 3), _f(_msloss(r), 3),
                     _f(r["saving_pct"], 1), f"{_pct(r['saving_pct']):.0f}th"])
    k.rows(rows,
           "<b>The three sequences that lose most on the perceptual "
           "metric.</b> All three are 1080p frames at the lowest rate and "
           "carry the full 40 tiles. The MS-SSIM column is "
           "−10 log10(1 − MS-SSIM), the unit section D puts the two "
           f"metrics in, and on these sequences it is two to three times the "
           f"set mean of {setloss:.3f} dB at this rate. The PSNR column is "
           f"the same file's own difference in PSNR and runs from "
           f"{min(-r['psnr_delta'] for r in ms):.3f} to "
           f"{max(-r['psnr_delta'] for r in ms):.3f} dB, over the nominal "
           f"budget on all three, which is the per-sequence overshoot the "
           f"next subsection measures. The last column is where each sits in "
           f"the set by compute saved, from the "
           f"{min(_pct(r['saving_pct']) for r in ms):.0f}th percentile to "
           f"the {max(_pct(r['saving_pct']) for r in ms):.0f}th: these are "
           f"not the sequences that save least.")
    k.note("The worst_sequences_by_ms_ssim block of "
           "results/supp_opquality_PAPER.json, on " + PINNED + ", \\NumSeq "
           "sequences at one frame each, at the 0.1 dB operating point of "
           "results/supp_paper_curve_PAPER.json. That block is the file's own "
           "shortlist of the fifteen largest MS-SSIM losses over the whole "
           "sweep, reordered here by the decibel rather than by the raw "
           "difference. The saving column is the same file's per-sequence "
           "saving against our own deepest exit and the rank is its "
           "percentile among the \\NumSeq sequences of "
           "results/supp_per_sequence_PAPER_b010.json at the same rate.")

    k.par(
        "The ordering there is the one to take away. A budget in PSNR does "
        "not price the second metric evenly across content, and the "
        "sequences where the two disagree most are ordinary 1080p clips "
        "from the middle and upper part of the set by saving rather than the "
        "hard cases further down this section. On a clip with 40 tiles the "
        "multiplier can move a third of the frame two rungs down the ladder "
        "and stay inside 0.1 dB of PSNR, and the tiles it moves are the flat, "
        "low-texture ones a structural metric weighs differently from a mean "
        "squared error. The remedy is a budget written in the metric a "
        "deployment cares about, and no allocation in this work was bisected "
        "on anything but PSNR.")

    # ------------------------------------------------------------------ H.2
    k.h2("What a set-level budget hides")

    per = {r["qp"]: r for r in ps["rows"]}
    rows = [["q", "over budget", "worst dB", "on", "least saved %", "on"]]
    for q in QPS:
        seqs = per[q]["per_sequence"]
        over = [s for s in seqs if s["db_vs_uf"] > 0.1]
        wq = max(seqs, key=lambda s: s["db_vs_uf"])
        lq = min(seqs, key=lambda s: s["saving_pct_vs_release"])
        rows.append([f"q{q}", f"{len(over)} of {len(seqs)}",
                     _f(wq["db_vs_uf"], 3), wq["seq"].split("_")[0],
                     _f(lq["saving_pct_vs_release"], 1),
                     lq["seq"].split("_")[0]])
    import json as _j2
    from pathlib import Path as _P2
    _j2.dump({"rows": rows},
             open(_P2(__file__).resolve().parents[2] / "results/setbudget_rows.json",
                  "w"), indent=2)

    k.rows(rows,
           "<b>The 0.1 dB budget, sequence by sequence.</b> The multiplier is "
           "bisected once for the whole set, so the budget binds on the set "
           "mean and on nothing else. Rather more than half the sequences "
           "receive a decode worse than the budget they were nominally given, "
           "and the worst of them receives between two and three and a half "
           "times it. Section D.4 gives the spread on the compute axis; this "
           "is the same allocation seen on the quality axis.")
    k.note("results/supp_per_sequence_PAPER_b010.json, the rows block for the "
           "delivered decibel and rows_vs_release arithmetic for the saving, "
           "on " + PINNED + ". Decibels are pooled, the convention "
           "results/supp_paper_curve_PAPER.json bisects in.")

    fl = {r["qp"]: r["floor_db"] for r in sig["rows"]
          if abs(r["budget_db"] - 0.1) < 1e-9}
    dr = {r["qp"]: abs(r["drift_db"]) for r in anc["rows"]}
    k.par(
        f"The other thing a set mean hides is how much of the budget is gone "
        f"before any tile exits early. With every tile at full depth the "
        f"decode already differs from the released decoder by {fl[0]:.3f} dB "
        f"at q0 and {fl[63]:.3f} dB at q63, which is two thirds of the "
        f"headline budget at the high rate. Two things make it up, and they "
        f"are measured separately rather than subtracted from one another: "
        f"the deepest exit's own departure from the released decoder, "
        f"{dr[63]:.3f} dB at q63 on the same checkpoint, and the seam left by "
        f"cutting the frame into tiles. Both are charged to the method, "
        f"because the saving is "
        f"measured against the released decoder rather than against our own "
        f"full-depth decode, and both grow with rate while the budget does "
        f"not.")
    k.note("Floors from results/signalled_RECIPE512_ctc53.json and drift from "
           "results/supp_anchor_PAPER.json, both on " + PINNED + " over "
           "\\NumSeq sequences. The two files do not pool their decibels "
           "alike, which is why the seam is described rather than given as "
           "the difference. Section C derives why a floor above the budget "
           "makes the feasible set empty and section E tabulates the three "
           "ways the floor itself can be measured.")

    sp = k.J("supp_seam_problem_bosphorus_q63.json")
    k.fig("seam_PAPER_bosphorus_q63.png",
          f"<b>The seam, at frame scale.</b> <b>a</b>, the decoded frame. "
          f"<b>b</b>, the absolute error against a full-frame decode of the "
          f"same latent, amplified {sp['amplification']:.0f}×, with every "
          f"tile at the deepest exit so that nothing here is early exiting. "
          f"<b>c</b>, a zoom on one junction. The tile grid is the error: "
          f"{sp['penalty_db']:.3f} dB at q63, paid before the allocation "
          f"begins.", maxh=80)
    k.note("paper/figures/seam_PAPER_bosphorus_q63.png, drawn from "
           "results/supp_seam_problem_bosphorus_q63.json on " + PINNED + ". "
           "The panel is the worst case rather than the shipped one: the "
           "file records tile_pad_mode zeros with the deblocking filter "
           "removed, which is how the artefact is made visible. The floors "
           "quoted above are the shipped configuration, replicate padding "
           "with the filter on.")

    # ------------------------------------------------------------------ H.3
    k.h2("Built, measured and abandoned")

    seam = k.J("ctc_seam_p256.json")
    cpl = k.J("coupling_ablation.json")
    ssp = k.J("seam_spatial_published.json")
    shapes = k.J("supp_module_shapes.json")
    tsz = k.J("tilesize_adaptive.json")
    svs = k.J("supp_seam_vs_split.json")

    sq = seam["qps"]
    sr = seam["res"]
    C = {r["qp"]: r for r in cpl["rows"]}
    rep_share = shapes["share_of_released_decode"]["seam_repair"]

    # The ceiling on a perfect gate for the deblocking filter: give it the
    # whole of its gain in the boundary band and none of its loss anywhere
    # else, then read the result as a decibel.
    import math
    tot = sum(s / 100.0 * m for s, m in zip(ssp["share_pct"], ssp["mse_off"]))
    gain = (ssp["share_pct"][0] / 100.0 * ssp["mse_off"][0]
            * abs(ssp["change_pct"][0]) / 100.0)
    rep_ceiling = 10.0 * math.log10(1.0 / (1.0 - gain / tot))

    # The area fraction against a fitted power law, refitted here on the
    # pinned checkpoint rather than quoted from the warm-start measurement
    # the claim was originally withdrawn on.
    svr = [r for r in svs["rows"] if r["b"] > 0]
    zero = [r for r in svs["rows"] if r["b"] == 0][0]
    fits, aerrs = {}, {}
    for q in ("0", "32", "63"):
        bs = [r["b"] for r in svr]
        ys = [r["seam_db"][q] for r in svr]
        fits[q] = _powerfit(bs, ys)
        aerrs[q] = _scalefit([r["predicted_fraction"] for r in svr], ys)

    ts = {p["qp"]: p for p in tsz["per_qp"]}
    ts_delta = tsz["summary_over_qps"]["mean_set_mean_delta_adaptive_minus_256"]

    k.par(
        "Seven mechanisms were built, measured and then left out of the "
        "reported system. Two of them replace the border a tile is convolved "
        "against, one repairs the seam afterwards, one removes the seam by "
        "letting neighbouring tiles exchange features, one freezes the trunk "
        "so that the deepest exit cannot move, one gives the small "
        "resolutions a smaller tile, and the last is an explanation of the "
        "tiling penalty rather than a mechanism. Table " +
        str(k.peek_tbl()) + " gives what settled each, and the file it was "
        "settled in. The cost of building them is not otherwise recoverable "
        "from the paper, and two of them are the first thing a reader would "
        "propose.")

    rows = [["what was built", "the measurement that ended it"]]
    rows.append([
        "First-order border extrapolation",
        f"worse than replication at all three rates: {sr['linear'][0]:.3f}, "
        f"{sr['linear'][1]:.3f} and {sr['linear'][2]:.3f} dB against "
        f"{sr['replicate'][0]:.3f}, {sr['replicate'][1]:.3f} and "
        f"{sr['replicate'][2]:.3f}"])
    rows.append([
        "Fitted AR(1) border padding",
        f"ahead of replication by {sr['replicate'][2] - sr['arls'][2]:.3f} dB "
        f"at q63, and bought with wall clock that no file in results/ "
        f"records"])
    rows.append([
        "A learned deblocking filter",
        f"{rep_ceiling:.4f} dB even with a perfect gate, against "
        f"{100 * rep_share:.2f}% of a decode"])
    rows.append([
        "Halo exchange between tiles",
        f"exact at uniform depth, and the saving falls from "
        f"{C[32]['padded']['saving']:.1f}% to "
        f"{C[32]['coupled']['saving']:.1f}% at q32 and from "
        f"{C[63]['padded']['saving']:.1f}% to "
        f"{C[63]['coupled']['saving']:.1f}% at q63"])
    rows.append([
        "Freezing the trunk",
        f"drift exactly {HEADS_ONLY['drift'][63]:.3f} dB, and the oracle "
        f"ceiling collapses with it to {HEADS_ONLY['oracle'][0]:.1f}% at q0, "
        f"{HEADS_ONLY['oracle'][32]:.1f}% at q32 and "
        f"{HEADS_ONLY['oracle'][63]:.1f}% at q63"])
    rows.append([
        "A tile size chosen per resolution",
        f"worth {ts_delta:+.2f} points on the set mean over three rates, with "
        f"the training confound running in its favour"])
    rows.append([
        "The affected-area fraction",
        f"wrong by {100 * min(aerrs.values()):.0f} to "
        f"{100 * max(aerrs.values()):.0f}%, against {100 * min(f[2] for f in fits.values()):.0f} "
        f"to {100 * max(f[2] for f in fits.values()):.0f}% for a power law in "
        f"the per-tile block count"])
    # The main paper carries a two-column version of this table, generated by
    # make_paper_tables from this dump. One place computes the rows; the paper
    # and the supplement cannot disagree about what was abandoned or why.
    import json as _json
    from pathlib import Path as _P
    _out = _P(__file__).resolve().parents[2] / "results/abandoned_rows.json"
    _json.dump({"rows": rows}, open(_out, "w"), indent=2)

    t_aband = k.rows(rows,
                     "<b>Seven mechanisms and what settled each.</b> Five "
                     "were built and dropped, one is a proposed extension "
                     "measurement did not support, and the last is an "
                     "explanation of the tiling penalty that the measurement "
                     "does not carry. None of the seven is in the reported "
                     "system, whose configuration is replicate padding at "
                     "256 px with the deblocking filter left on, as the "
                     "ablation table of section A records.")
    k.note("Rows in order: results/ctc_seam_p256.json (40 sequences, three "
           "rates); the same file with results/shrink_ablation.log; "
           "results/seam_spatial_published.json with the module's share of a "
           "decode from results/supp_module_shapes.json; "
           "results/coupling_ablation.json; results/first_ckpt_headsonly.log; "
           "results/tilesize_adaptive.json; results/supp_seam_vs_split.json, "
           "refitted in the build. The coupling and heads-only files are on "
           "runs/RECIPE512/ckpt_eval.pth.tar and "
           "runs/heads_only_j2_p256/ckpt_epo0.pth.tar; the seam-repair "
           "spatial file is on a runs/BEST checkpoint whose epoch it does not "
           "record. The rest are on " + PINNED + ".")

    k.par(
        f"Three of those rows need more than a line. The halo exchange does "
        f"two of the three things asked of it: it is exact at uniform depth, "
        f"and it removes "
        f"{100 * (1 - C[0]['coupled']['floor_db'] / C[0]['padded']['floor_db']):.0f}% "
        f"of the floor at q0 and "
        f"{100 * (1 - C[63]['coupled']['floor_db'] / C[63]['padded']['floor_db']):.0f}% "
        f"at q63. The third, that closing the floor is worth several points "
        f"of saving, has the wrong sign at every rate above the lowest. "
        f"A tile whose neighbour ran six more blocks is reading an activation "
        f"the trained weights have never seen, and the error that introduces "
        f"is far larger than the seam it removed. Exactness holds when every "
        f"tile is at the same depth, and routing is the deliberate violation "
        f"of that condition, so the two are in tension by construction. The "
        f"one caveat that keeps this from being final is that the model was "
        f"trained with replicate padding, so switching the exchange on at "
        f"inference is a distribution shift; a run trained with it could "
        f"reverse the sign, and no such run exists.")

    k.par(
        f"The deblocking filter is the reverse case, where the mechanism does "
        f"what it was built to do and the amount is too small to pay for. "
        f"Error falls monotonically with distance from the nearest tile "
        f"boundary, {ssp['mse_off'][0]:.1e} in the innermost "
        f"{ssp['bands_px'][0][1]} px against {ssp['mse_off'][-1]:.1e} more "
        f"than {ssp['bands_px'][-1][0]} px away, so the seam is real and it is "
        f"local. The filter's gate leaks: it gains in the boundary band and "
        f"loses slightly everywhere else. Giving it a perfect gate, with its "
        f"boundary gain kept and its interior loss set to zero, bounds it at "
        f"{rep_ceiling:.4f} dB for {100 * rep_share:.2f}% of a decode. "
        f"Tightening the gate cannot rescue the module because there is not "
        f"enough in the band for it to win. It is still switched on in the "
        f"reported system, because it was trained jointly with the decoder "
        f"and switching it off at inference is not the same experiment as "
        f"training without it, and no run has been trained without it at "
        f"256 px.")

    k.par(
        f"The affected-area fraction is an explanation rather than a "
        f"mechanism, and it is the weaker of the two available. It is a "
        f"correct statement about which pixels a tile border can reach and a "
        f"poor predictor of what the border costs, because it saturates: at "
        f"{svr[0]['b']:.0f} blocks per "
        f"tile it is already {svr[0]['predicted_fraction']:.2f} and cannot "
        f"rise further, while the penalty still climbs by a factor of "
        f"{svr[0]['seam_db']['63'] / svr[2]['seam_db']['63']:.1f} from "
        f"{svr[2]['b']:.0f} blocks to {svr[0]['b']:.0f}. Fitted with one free "
        f"scale it is wrong by {100 * min(aerrs.values()):.0f} to "
        f"{100 * max(aerrs.values()):.0f}% on average. A power law in the block count "
        f"is wrong by {100 * min(f[2] for f in fits.values()):.0f} to "
        f"{100 * max(f[2] for f in fits.values()):.0f}%, with exponents "
        f"{fits['0'][0]:.2f}, {fits['32'][0]:.2f} and {fits['63'][0]:.2f} at "
        f"q0, q32 and q63 and log-log correlations from {min(f[1] for f in fits.values()):.3f} "
        f"to {max(f[1] for f in fits.values()):.3f}. An exponent near two has "
        f"a reading: the number of affected pixels grows with the border's "
        f"reach and the damage in each grows with how many convolutions "
        f"reached it, and the two are the same quantity. The drift from "
        f"{fits['0'][0]:.2f} to {fits['63'][0]:.2f} with rate is not "
        f"explained here. The control is the row that licenses reading any of "
        f"it as seam: at zero blocks per tile the penalty is "
        f"{zero['seam_db']['0']:.4f} dB at every rate.")

    k.par(
        f"The tile-size row is the one a reader is most likely to propose, "
        f"since the exit maps of Table {t_probe} say the method is "
        f"governed by tile count. Splicing 128 px tiles into the two smallest "
        f"classes and leaving 256 px elsewhere is worth "
        f"{ts_delta:+.2f} points on the set mean, averaged over three rates. "
        f"The direction differs by rate rather than being uniformly flat: at "
        f"q0 the smaller tile is ahead in five of six classes and the splice "
        f"gains {ts[0]['set_mean_delta_adaptive_minus_256']:+.2f} points, "
        f"while at q63 the smaller tile is behind on every class and the "
        f"splice loses {ts[63]['set_mean_delta_adaptive_minus_256']:+.2f}. "
        f"Halving the tile side doubles the seam at fixed depth, and above "
        f"the lowest rate the extra seam costs more than the extra "
        f"granularity returns. The comparison is not matched for training "
        f"time: the 128 px column carries "
        f"{tsz['confound']['training_gap_steps']:,} more steps than the "
        f"pinned checkpoint, an advantage that runs in the smaller tile's "
        f"favour, which is why the conclusion is that the extension is not "
        f"supported rather than that it is harmful.")
    k.note("results/tilesize_adaptive.json, which splices "
           "results/per_class_RECIPE512.json on " + PINNED + " against "
           "results/per_class_BEST128.json on runs/BEST128/ckpt_step.pth.tar. "
           "The file brackets the training confound with a third measurement "
           "on runs/RECIPE512/ckpt_PIN_e1.pth.tar and records the verdict per "
           "class and per rate.")

    k.par(
        f"The AR(1) padding row is the one this section cannot close from "
        f"results/. It is the best border estimator measured, "
        f"{sr['arls'][2]:.3f} dB at q63 against {sr['replicate'][2]:.3f} for "
        f"replication, and it was rejected on wall clock, which no file in "
        f"results/ holds. What results/ does hold is why nothing cheaper "
        f"works. Its coefficient is fitted per channel and per tile, and "
        f"replacing the fit with a fixed shrinkage recovers none of the gain: "
        f"at a coefficient of 0.95 the seam is "
        f"{SHRINK['shrink0.95'][63]:.4f} dB against replication's "
        f"{SHRINK['replicate'][63]:.4f} and the fit's "
        f"{SHRINK['arls'][63]:.4f}, and every coefficient below that is worse "
        f"than replication at every rate. The gain is in the adaptivity, and "
        f"the adaptivity is what costs.")
    k.note("results/shrink_ablation.log, 10 CTC frames at j = 2 and 128 px, "
           "typed into this module because it is a log rather than JSON; the "
           "seam figures it is compared against are "
           "results/ctc_seam_p256.json, 40 sequences at 256 px, so the two "
           "sets of absolute values are not on one protocol and only the "
           "ordering within each is read. The wall-clock measurement that "
           "decided the estimator is recorded in the project's decision log "
           "and in no file in results/.")

    # ------------------------------------------------------------------ H.4
    k.fig("seam_module.png",
          "<b>The learned deblocking filter, and where it wins.</b> It is "
          "gated by position within a tile and applied once to the stitched "
          "frame, for "
          f"{100 * rep_share:.2f}% of the decode. <b>a</b>, the trained gate "
          "against distance from the boundary, beside its initialisation. "
          "<b>b</b>, the change in error by that distance; bar width is the "
          "share of pixels, so bar area is the contribution to the frame. It "
          "wins on the pixels nearest a boundary and loses on the rest, which "
          "is why a perfect gate still cannot pay for it.")

    k.par(
        "The exactness claim for the halo exchange is worth writing out, "
        "because it is the condition and not the number that carries the "
        "argument. At uniform depth the largest absolute difference between a "
        "tiled decode with the exchange and a full-frame decode of the same "
        "latent is exactly zero once the deblocking pass is switched off; "
        "with the pass on it is "
        f"{cpl['rows'][0]['uniform_depth_max_diff']:.2e}, because that pass "
        "runs on the stitched frame and has no full-frame counterpart, and "
        "with replicate padding instead of the exchange it is larger again. "
        "The condition in that sentence is <i>uniform depth</i>, and routing "
        "is the deliberate violation of it, which is the whole of the "
        "tension.")

    k.fig("contamination.png",
          "<b>The seam against per-tile depth.</b> <b>a</b>, the measurement "
          "with the fitted power law. <b>b</b>, both models against q63, one "
          "free scale each. <b>c</b>, mean relative error. The area fraction "
          "saturates once the border reaches every pixel and the penalty does "
          "not, which is where the area law fails as a predictor.")

    k.h2("The checkpoints, and what a later one measures")

    e0_40 = k.J("signalled_RECIPE512_0818_0618.json")
    e1 = k.J("signalled_RECIPE512_0819_0556.json")
    e2 = k.J("signalled_RECIPE512_0820_0140.json")
    b1 = k.J("signalled_BEST_ctc53.json")
    b2 = k.J("signalled_BEST_0819_1050.json")
    b3 = k.J("signalled_BEST_0820_0841.json")

    P = _at(sig)
    E1, E2 = _at(e1), _at(e2)
    B1, B2, B3 = _at(b1), _at(b2), _at(b3)

    k.par(
        "Every table in this supplement is measured on " + PINNED + ", which "
        "is the first epoch boundary of a run that has since passed three. "
        "The per-checkpoint watcher measured the epochs in between on the "
        "same \\NumSeq sequences at the same budget through the same code "
        "path, and Table " + str(k.peek_tbl()) + " is what they say.")

    rows = [["checkpoint"] + [f"q{q}" for q in QPS] + ["mean"]]
    for lab, d in (("RECIPE512, epoch 0, pinned", P),
                   ("RECIPE512, epoch 1", E1),
                   ("RECIPE512, epoch 2", E2),
                   ("BEST, epoch 1", B1),
                   ("BEST, epoch 2", B2),
                   ("BEST, epoch 3", B3)):
        rows.append([lab] + [_f(d[q], 1) if q in d else "-" for q in QPS]
                    + [_f(_mean(d, QPS), 1)])
    t_epoch = k.rows(
        rows,
        "<b>Compute saved at the 0.1 dB budget as training continues.</b> "
        "RECIPE512 and BEST are two runs of one configuration, same "
        "architecture and same recipe, differing only in the draw. The "
        "pinned row is what the paper reports. Epoch 2 of the same run is "
        f"hook-counted like the pinned row and is "
        f"{_mean(E2, QPS) - _mean(P, QPS):.1f} points ahead of it; epoch 1 "
        f"reads {_mean(E1, QPS) - _mean(P, QPS):.1f} ahead in a file that "
        f"carries no hook count and is therefore about {model_offset:.1f} "
        f"points optimistic against the other rows. Either way the paper is "
        f"quoting a checkpoint behind the run that produced it. Two "
        "measurements are not enough to call epoch 1 a peak, and moving the "
        "headline there would be choosing the better of two rather than "
        "reporting the latest.")
    k.note("results/signalled_RECIPE512_ctc53.json on " + PINNED + ", and "
           "results/signalled_RECIPE512_0819_0556.json, "
           "results/signalled_RECIPE512_0820_0140.json, "
           "results/signalled_BEST_ctc53.json, "
           "results/signalled_BEST_0819_1050.json and "
           "results/signalled_BEST_0820_0841.json on the ckpt_eval and "
           "ckpt_step files of their runs, which the watchers overwrite; each "
           "records its own epoch. Savings are hook-counted where the file "
           f"carries a hook count and modelled where it does not, which "
           f"applies to the two epoch-1 rows and makes them optimistic by "
           f"about {model_offset:.1f} points against the rest. A difference "
           f"between two rows that agree about the convention carries none "
           f"of that offset, and the two comparisons this section draws "
           f"conclusions from, epoch 0 against epoch 2 and RECIPE512 against "
           f"BEST at epoch 2, are both of that kind.")

    k.par(
        f"One condition governs any comparison between measurements made at "
        f"different times. The test set grew from {e0_40['n_sequences']} "
        f"sequences to {sig['n_sequences']} when MCL-JCV and the two smallest "
        f"HEVC classes were added, and eight tiles at 832×480 or two at "
        f"416×240 save nothing like a 1080p frame does, so a saving measured "
        f"on the smaller set is several points above one measured on the "
        f"larger at the same checkpoint. Every row of this section is on "
        f"\\NumSeq sequences.")

    # ------------------------------------------------------------------ H.5
    k.h2("How far two runs of one recipe are apart")

    f12 = [k.J(f) for f in FINE12_EPOCH2]
    means = []
    for d in f12:
        at = _at(d)
        means.append(_mean(at, FINE12_QPS))
    per_rate = {}
    for q in FINE12_QPS:
        vs = [_at(d)[q] for d in f12]
        per_rate[q] = (min(vs), max(vs))

    # One computation for this number, in make_paper_tables, because the
    # main paper quotes it too and had a stale 4.8 typed into it.
    gap2 = float(k.macro("RunGapEpochTwo"))

    k.par(
        f"RECIPE512 and BEST are the same K, the same split depth, the same "
        f"tile size, the same adapters and the same 45,445,398 parameters. At "
        f"epoch 2 they are {gap2:.1f} points apart on the set mean. The "
        f"ladder table of the main paper reports a 256 px run at "
        f"{_mean(B1, QPS):.1f} against a 128 px run at "
        f"{_mean(_at(k.J('signalled_BEST128_ctc53.json')), QPS):.1f} and a "
        f"twelve-exit run at "
        f"{_mean(_at(k.J('signalled_FINE12_ctc53.json')), FINE12_QPS):.1f}, "
        f"the last over the four rates it reaches rather than five, which "
        f"are differences of about five points. The gap between two runs of "
        f"one configuration is the same size as the architectural "
        f"differences the table is ranking.")
    k.note("results/signalled_BEST_ctc53.json, "
           "results/signalled_BEST128_ctc53.json and "
           "results/signalled_FINE12_ctc53.json, the three files behind the "
           "main paper's ladder table, each on the ckpt_eval or ckpt_step "
           "file of its own run; none of the three carries a hook count, so "
           "the three are on one basis with each other and about "
           f"{model_offset:.1f} points optimistic against the pinned row of "
           f"Table {t_epoch}. The epoch-2 comparison above is between two "
           "files that both carry one.")

    k.par(
        "The spread is not only between runs. The watcher writes a "
        f"measurement every time it finds a new checkpoint, so one epoch of "
        f"one run yields several, and Table {k.peek_tbl()} prints six of them "
        f"from a single epoch. They span {max(means) - min(means):.1f} points "
        f"on the four-rate mean and up to "
        f"{max(hi - lo for lo, hi in per_rate.values()):.1f} at a single "
        f"rate, on one test set, one budget and one code path, with every row "
        f"hook-counted.")

    rows = [["checkpoint, in order"] + [f"q{q}" for q in FINE12_QPS]
            + ["mean"]]
    for f, d, m in zip(FINE12_EPOCH2, f12, means):
        at = _at(d)
        rows.append([f.replace("signalled_FINE12_", "").replace(".json", "")]
                    + [_f(at[q], 1) for q in FINE12_QPS] + [_f(m, 2)])
    k.rows(rows,
           "<b>Six checkpoints from one epoch of one run.</b> FINE12 is the "
           "twelve-exit ladder, whose highest rate has no admissible "
           "allocation at this budget, so the mean is over the four rates "
           "that do. Take from it that a comparison at a matched epoch is not "
           "a comparison at a matched checkpoint, and that the difference "
           "between two rows here is as large as several of the differences "
           "this work reads as results.")
    k.note("The six results/signalled_FINE12_0819_*.json files listed in this "
           "module, all at epoch 2 of runs/FINE12 on \\NumSeq sequences at a "
           "0.1 dB budget, all carrying a hook count. The checkpoints "
           "themselves are the run's own per-epoch files, which it rewrites, "
           "so these measurements are what preserve them. A seventh epoch-2 "
           "measurement is excluded because it carries no hook count.")

    k.par(
        "What follows from the two tables is a constraint on how this work "
        "may be read. No configuration was trained twice at two seeds, and no "
        "seed is set in the decoder's trainer, so there is no interval on any "
        "architectural comparison in this document. Section A says so and "
        "gives the sources of variation; what these files add is a magnitude. "
        "Until a configuration is trained twice, a difference of a few points "
        "between two rows of the ladder table is not separable from the "
        "difference between two runs of either of them.")

    # ------------------------------------------------------------------ H.6
    k.h2("Anchor drift")

    A = {
        "RECIPE512, pinned": k.J("supp_anchor_PAPER.json"),
        "RECIPE512, later": k.J("anchor_RECIPE512.json"),
        "BEST": k.J("anchor_BEST.json"),
        "BEST128": k.J("anchor_BEST128.json"),
        "FINE12": k.J("anchor_FINE12.json"),
        "VERBATIM, no anchor term": k.J("anchor_VERBATIM.json"),
    }
    rows = [["run", "q0", "q32", "q63", "frames"]]
    drifts = {}
    for lab, d in A.items():
        m = {r["qp"]: abs(r["drift_db"]) for r in d["rows"]}
        drifts[lab] = m
        rows.append([lab] + [_f(m[q], 4) if q in m else "-"
                             for q in (0, 32, 63)] + [str(d["n_frames"])])
    vb = drifts["VERBATIM, no anchor term"]
    pn = drifts["RECIPE512, pinned"]
    fn = drifts["FINE12"]
    k.rows(rows,
           "<b>How far the deepest exit has moved from the released "
           "decoder</b>, in decibels, one checkpoint per run. The deepest "
           "exit is supposed to be the released decoder and the training "
           "objective carries a term that holds it there. The term does real "
           f"work: the run trained without it sits {vb[0]:.3f} to "
           f"{vb[63]:.3f} dB away, more than the whole headline budget at "
           f"every rate before a single tile has exited early. The term does "
           f"not hold it exactly, and what it leaves behind grows with rate: "
           f"{100 * pn[63] / 0.1:.0f}% of a 0.1 dB budget at q63 on the "
           f"pinned checkpoint and {100 * fn[63] / 0.1:.0f}% on the "
           f"twelve-exit run, whose floor exceeds that budget outright at "
           f"the highest rate.")
    k.note("results/supp_anchor_PAPER.json on " + PINNED + "; "
           "results/anchor_RECIPE512.json, results/anchor_BEST.json, "
           "results/anchor_BEST128.json, results/anchor_FINE12.json and "
           "results/anchor_VERBATIM.json on the ckpt_eval or ckpt_step file "
           "of their runs, at the epoch each happened to be at, which none of "
           "them records. Frame counts differ by file and are printed, "
           "because two of these are 40 frames and are not comparable with "
           "the \\NumSeq-frame rows at the fourth decimal.")

    lat = A["RECIPE512, later"]
    latm = {r["qp"]: abs(r["drift_db"]) for r in lat["rows"]}
    pinm = {r["qp"]: abs(r["drift_db"]) for r in anc["rows"]}
    k.par(
        f"Whether it grows with training is not settled by what is in "
        f"results/. The two RECIPE512 rows are the pinned checkpoint and a "
        f"later one on the same \\NumSeq frames, and at q63 they read "
        f"{pinm[63]:.4f} and {latm[63]:.4f} dB, which is no movement at all; "
        f"at q0 the later checkpoint is the closer of the two, at "
        f"{latm[0]:.4f} against {pinm[0]:.4f}. Neither file records the epoch "
        f"of the checkpoint it read, so the pair does not bracket a known "
        f"amount of training. The measured direction is with rate rather than "
        f"with time, sevenfold from q0 to q63 on the pinned checkpoint. The "
        f"open problem is the size, not the trend: at q63 a third of the "
        f"budget is spent before the allocation begins, and any move to a "
        f"later checkpoint has to remeasure this and report it as its own "
        f"row rather than assume it carries over.")

    k.par(
        "There is a structural fix and it was measured and rejected, which is "
        f"row five of Table {t_aband}. Freezing the trunk makes the deepest "
        "exit the released decoder by construction, with no weight left that "
        "could move, and the drift measures exactly zero. The shallow exits "
        "then cannot improve either, because the exit heads alone cannot "
        "recover what six skipped blocks contained, and the oracle ceiling "
        "collapses to nothing above the lowest rate. Training the trunk is "
        "what makes the ladder worth having, and the drift is what training "
        "the trunk costs.")

    # ------------------------------------------------------------------ H.7
    k.h2("What is open")

    cp = k.J("check_paper.json")

    k.par(
        "Every open problem in this supplement is listed here, so that a "
        "reader meets each of them once. Section A gives the provenance of "
        "each table and section D the spread behind each mean; what follows "
        "is what those two cannot close.")

    k.bullets([
        f"<b>The reported checkpoint is early.</b> Table {t_epoch} says a "
        "later checkpoint of the same run saves more at every rate. Nothing "
        "in this document is measured on one, because \\NumClaims checked "
        "claims, every table and every figure rest on the pinned checkpoint, "
        "and moving it means remeasuring all of them. What a later "
        "checkpoint would change is the size of the headline and the anchor "
        "drift beneath it, in opposite directions.",

        "<b>No interval exists on any architectural comparison.</b> Two runs "
        f"of one configuration are {gap2:.1f} points apart at a matched "
        "epoch, and six checkpoints inside one epoch of one run span "
        f"{max(means) - min(means):.1f}. Both are the size of the "
        "differences the ladder table ranks. The remedy is repeated runs, "
        "which is a training cost this work has not spent.",

        "<b>The anchor is held by a loss term and not by construction.</b> "
        f"The deepest exit is {pinm[63]:.3f} dB from the released decoder at "
        "q63 on the pinned checkpoint, about a third of the headline budget, "
        "and the one construction that removes the drift removes the ladder's "
        "value with it.",

        "<b>Everything here is one decoder's intra path.</b> The ladder is "
        "built inside the intra decoder of one released codec and measured on "
        "intra frames of video sequences at 1080p and below. No inter frame, "
        "no second codec rearranged into a ladder, and no resolution above "
        "1080p, the last because the evaluation card has about 5 GB free and "
        "a 4K decode does not fit. The construction needs a residual trunk "
        "with a shared head and nothing else, which is an argument rather "
        "than a measurement.",

        "<b>The window is measured with the losses known.</b> The allocations "
        "in this section are the exact per-tile search, which bounds any "
        "decoder-side predictor. Section F measures how much of it a trained "
        "head reaches and how much a rule with no learned parameters "
        "reaches, and neither closes the gap at high rate.",

        "<b>Part of the recipe is pinned outside results/.</b> Nothing that "
        "writes to results/ runs inside the trainer, so the hyperparameter "
        "table of A.11, the dataset statistics of A.8 and the training wall "
        "clock are read from the launcher, the trainer and the run's own log "
        "rather than from a measurement file. The warm-start report on disk "
        "is the K = 12 rebuild rather than the K = 6 one the pinned run "
        "descends from; its two zero-difference controls are asserted at "
        "every build instead, so a failure would stop the run.",

        "<b>One accelerator class.</b> Every millisecond and every joule in "
        "this work is an NVIDIA RTX A6000, all eight cards in this machine "
        "being that model, and the CPU rows of section B are the only second "
        "device class. Power comes from the driver counter, with no external "
        "meter to calibrate it against.",
    ])

    k.par(
        f"No internal check is outstanding. They all hold, including the one "
        f"this list would otherwise have carried: the "
        f"partially signalled configuration reduces to the fully signalled "
        f"one to four decimals when every tile is overridden, with both sides "
        f"read in the same saving convention, so the cost model's "
        f"{model_offset:.1f}-point offset cannot present itself as a failure "
        f"of the interpolation. As recorded in results/check_paper.json, "
        f"{cp['n_passed']} of {cp['n_passed'] + len(cp.get('failed') or [])} "
        f"checked claims pass.")
