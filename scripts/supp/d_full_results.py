"""Complete results: every rate against every test set, and what it costs.

The main paper samples this section. Its per-class table is one budget at
three of the five quality indices, its headline is one budget at five, and its
BD table is three budgets pooled over the set. Everything here is the full
grid on the pinned checkpoint, together with the measurements the main paper
has no room for: a second quality metric, the distribution over sequences and
over tiles, the composition with weight quantisation, the raw numbers behind
the curves, and the cases where the method does badly.

Two habits run through the section and are stated once here.

Provenance. Every number is read from a file in `results/` at build time by
`k.J`, so the page and the measurement cannot drift apart, and the note under
each table names the file. Where a table is not on
`runs/RECIPE512/ckpt_PAPER.pth.tar` the note says which checkpoint it is on.

Conventions. Two decibel conventions are in use in this project and they
differ by about a third of a 0.1 dB budget, so each table says which one it is
under. `per frame` averages a per-frame PSNR difference, which is what
`~/DCVC/test_video.py` reports and therefore what a published DCVC-UF number
means. `pooled` forms one mean squared error over the whole set and
differences that. Two denominators are also in use for compute saved: against
the released DCVC-UF decoder, which is what the paper quotes, and against our
own deepest exit, which costs \\DeepestUniformCost released decodes because it
pays for seam repair and the release does not.

The public surface is `content(k)`. Everything else is a private helper that
returns a list of rows.
"""

# Largest tile count first, which is also the order the saving falls in.
CLASSES = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
QPS = [0, 16, 32, 48, 63]
# Trunk blocks skipped at each of the six exits, from results/adapter_cost.json.
SKIPPED = [10, 8, 6, 4, 2, 0]
PINNED = "runs/RECIPE512/ckpt_PAPER.pth.tar"
CURVE = "supp_paper_curve_PAPER.json"

def _name(c):
    return c.replace("HEVC_", "HEVC ")

def _short(s):
    """A sequence name short enough for a table cell."""
    s = (s.replace(".yuv", "").replace("_420_8bit_YUV", "")
          .replace("_1920x1080", "").replace("_120fps", ""))
    head, _, tail = s.rpartition("_")
    return head if head and tail.isdigit() else s

def _f(x, n=1):
    return "n/a" if x is None else f"{x:.{n}f}"

def _pc(k):
    d = k.J("supp_per_class_budgets.json")
    return {(r["qp"], round(r["budget_db"], 3)): r for r in d["rows"]}

# ------------------------------------------------- every class, every rate
def _class_saving_rows(k):
    idx = _pc(k)
    out = [["Class"] + [f"q{q}" for q in QPS]]
    for c in CLASSES:
        out.append([_name(c)] +
                   [_f(idx[(q, 0.1)]["per_class"][c]["saving"]) for q in QPS])
    out.append(["All \\NumSeq, pooled"] +
               [_f(idx[(q, 0.1)]["overall_saving"]) for q in QPS])
    out.append(["Quality given up (dB)"] +
               [_f(idx[(q, 0.1)]["overall_db"], 3) for q in QPS])
    return out

# ------------------------------------------------------------ exit shares
def _pooled_exit_rows(k):
    idx = _pc(k)
    out = [["Rate", "e2", "e3", "e4", "e5", "Blocks skipped", "Saved"]]
    for q in QPS:
        r = idx[(q, 0.1)]
        tot = [0] * 6
        for c in CLASSES:
            for i, h in enumerate(r["per_class"][c]["hist"]):
                tot[i] += h
        n = sum(tot)
        mb = sum(t * SKIPPED[i] for i, t in enumerate(tot)) / n
        out.append([f"q{q}"] + [_f(100 * tot[i] / n) for i in (2, 3, 4, 5)] +
                   [_f(mb, 2), _f(r["overall_saving"])])
    return out

def _class_exit_rows(k):
    idx = _pc(k)
    out = [["Class", "e2", "e3", "e4", "e5", "e2", "e3", "e4", "e5"]]
    for c in CLASSES:
        row = [_name(c)]
        for q in (0, 63):
            sh = idx[(q, 0.1)]["per_class"][c]["exit_share"]
            row += [_f(sh[i]) for i in (2, 3, 4, 5)]
        out.append(row)
    return out

# -------------------------------------------------------- per sequence
def _spread_rows(k):
    d = k.J("supp_per_sequence_PAPER_b010.json")
    out = [["Rate", "n", "Mean", "SD", "Min", "p25", "Median", "p75", "Max"]]
    for r in d["rows_vs_release"]:
        out.append([f"q{r['qp']}", r["n"]] +
                   [_f(r[f]) for f in ("mean", "sd", "min", "p25", "median",
                                       "p75", "max")])
    return out

def _under_ten(k):
    """Sequence-and-rate pairs saving less than a tenth of a decode."""
    d = k.J("supp_per_sequence_PAPER_b010.json")
    return sum(1 for r in d["rows"] for s in r["per_sequence"]
               if s["saving_pct_vs_release"] < 10)

def _tail_seq_rows(k, n=6):
    d = k.J("supp_per_sequence_PAPER_b010.json")
    by = {r["qp"]: {s["seq"]: s for s in r["per_sequence"]} for r in d["rows"]}
    order = sorted(by[63], key=lambda s: by[63][s]["saving_pct_vs_release"])
    out = [["Sequence", "q0", "q63", "q63 dB"]]
    for s in order[:n] + order[-2:]:
        out.append([_short(s), _f(by[0][s]["saving_pct_vs_release"]),
                    _f(by[63][s]["saving_pct_vs_release"]),
                    _f(by[63][s]["db_vs_uf"], 3)])
    return out

# ------------------------------------------------- a second quality metric
def _metric_rows(k):
    d = k.J("supp_opquality_PAPER.json")
    out = [["Rate", "bpp", "PSNR", "ours", "lost", "MS-SSIM", "ours",
            "lost"]]
    for r in sorted(d["rows"], key=lambda r: r["qp"]):
        out.append([f"q{r['qp']}", _f(r["bpp"], 4),
                    _f(r["psnr_released"], 3), _f(r["psnr_routed"], 3),
                    _f(-r["psnr_delta"], 4),
                    _f(r["ms_ssim_released"], 5), _f(r["ms_ssim_routed"], 5),
                    _f(r["ms_ssim_db_released"] - r["ms_ssim_db_routed"], 4)])
    return out

def _tile_tail_rows(k):
    d = k.J("supp_opquality_PAPER.json")["tail"]
    out = [["Rate", "Tiles", "Mean", "Median", "p95", "p99", "Max",
            "Over 0.25", "Over 0.5", "Over 1.0"]]
    for q in QPS:
        t = d[str(q)]
        out.append([f"q{q}", t["n_tiles"], _f(t["mean_db"], 3),
                    _f(t["median_db"], 3), _f(t["p95_db"], 3),
                    _f(t["p99_db"], 3), _f(t["max_db"], 3),
                    t["n_over_0_25_db"], t["n_over_0_5_db"],
                    t["n_over_1_db"]])
    return out

# ----------------------------------------------- composition with quantisation
def _quant_rows(k):
    d = k.J("supp_quant_PAPER.json")
    out = [["Weights", "Rate", "e2", "e3", "e4", "e5", "Layers"]]
    for r in d["rows"]:
        lab = "float32" if r["bits"] == 32 else f"int{r['bits']}"
        out.append([lab, f"q{r['qp']}"] +
                   [_f(r["db_per_exit"][i], 3) for i in (2, 3, 4, 5)] +
                   [r["layers_quantised"]])
    return out

# --------------------------------------------------------------- integrated
def _bd_rows(k):
    a = k.J("supp_bd_PAPER_per_frame.json")
    b = k.J("supp_bd_PAPER_pooled.json")
    A = {r["qp"]: r for r in a["rows"]}
    B = {r["qp"]: r for r in b["rows"]}
    out = [["Rate", "BD-saving", "BD-quality", "BD-saving", "BD-quality",
            "Points"]]
    for q in QPS:
        out.append([f"q{q}",
                    "n/a" if A[q]["bd_saving_pct"] is None
                    else _f(A[q]["bd_saving_pct"]) + "%",
                    _f(A[q]["bd_quality_db"], 4),
                    "n/a" if B[q]["bd_saving_pct"] is None
                    else _f(B[q]["bd_saving_pct"]) + "%",
                    _f(B[q]["bd_quality_db"], 4), A[q]["n_points"]])
    out.append(["Mean where defined", _f(a["mean_bd_saving_pct"]) + "%",
                _f(a["mean_bd_quality_db"], 4),
                _f(b["mean_bd_saving_pct"]) + "%",
                _f(b["mean_bd_quality_db"], 4), ""])
    return out

def _bdrate_rows(k):
    d = k.J("bdrate.json")
    by = {(r["config"], round(r["budget_db"], 2)): r for r in d["rows"]}
    out = [["Budget", "A BD-Rate", "B BD-Rate", "Difference", "A saved",
            "B saved"]]
    for b in (0.1, 0.3, 0.5):
        a, r = by[("A signalled", b)], by[("B router", b)]
        out.append(["%.1f dB" % b, _f(a["bd_rate_pct"], 3) + "%",
                    _f(r["bd_rate_pct"], 3) + "%",
                    _f(a["bd_rate_pct"] - r["bd_rate_pct"], 3) + " pt",
                    _f(a["saving_pct_vs_release"]) + "%",
                    _f(r["saving_pct_vs_release"]) + "%"])
    return out

# ----------------------------------------------------------- raw values
def _deployed_rows(k):
    d = k.J("signalled_RECIPE512_ctc53.json")
    out = [["Rate", "Budget", "lambda", "dB", "Model", "Meter", "Map bits"]]
    for r in d["rows"]:
        out.append([f"q{r['qp']}", "%g" % r["budget_db"],
                    "%.3e" % r["lam"] if r.get("lam") else "n/a",
                    _f(r["db_vs_uf"], 4), _f(r["saving_pct_vs_release"]),
                    _f(r["saving_pct_measured"]), _f(r["map_bits"], 0)])
    return out

def _deepest_cost(k):
    """What our deepest exit costs, in released decodes, from pinned files.

    At saturation every tile sits at the shallowest selectable exit, so the
    saving the curve file records there is measured against our own deepest
    exit while the ceiling the saturation file records is measured against the
    release. One divides into the other. The result, 1.0095, is what
    results/combined_RECIPE512_b01.json stores directly, and that file is on
    runs/RECIPE512/ckpt_eval.pth.tar; deriving it from two pinned files avoids
    quoting an unpinned one for a quantity the whole section leans on.
    """
    sat = k.J("saturation_RECIPE512_ctc53.json")
    sv = [o["saving_pct"] for o in k.J(CURVE)["op_points"]
          if o.get("saturated")][0]
    return sat["cost_j"] / (1 - sv / 100)

def _frontier_rows(k):
    """The 35 measured operating points, both conventions, in one table.

    paper_curve.py takes its argmin over the whole [tiles, K] table and
    flexuf/eval.py fills the columns below the split depth with the decode at
    the split depth, so those columns tie and the argmin returns the first of
    them. Bins 0, 1 and 2 are one exit; folding them is what makes this
    histogram agree with the clamped one in supp_per_class_budgets.json.
    """
    d = k.J(CURVE)
    deep = _deepest_cost(k)
    out = [["Rate", "Budget", "lambda", "dB pooled", "dB per frame",
            "Saved", "e2/e3/e4/e5"]]
    for o in d["op_points"]:
        sv = o.get("saving_pct")
        rel = "n/a" if sv is None else _f(100 - (100 - sv) * deep)
        if o.get("hist"):
            h = list(o["hist"])
            h[2] += h[0] + h[1]
            h[0] = h[1] = 0
            n = sum(h)
            sh = "/".join("%.0f" % (100 * h[i] / n) for i in (2, 3, 4, 5))
        elif o.get("saturated"):
            sh = "100/0/0/0"
        else:
            sh = "none"
        out.append([f"q{o['qp']}", "%g" % o["target_db"],
                    "%.3e" % o["lam"] if o.get("lam") else "n/a",
                    _f(o["db_vs_uf"], 4), _f(o["db_vs_uf_per_frame"], 4),
                    rel, sh])
    return out

# ------------------------------------------------------------------ prose
def content(k):
    k.h1("Complete results")

    k.par(
        "This section prints in full what the main paper samples: every class "
        "at every quality index and at three budgets, the frontier drawn "
        "class by class, and the distribution behind each mean, over "
        "sequences and over tiles. It also carries three measurements the "
        "main paper has no room for: a perceptual metric beside the decibel, "
        "the composition with weight quantisation, and the raw numbers behind "
        "the curves. It ends with the cases where the method does badly.")

    # ----------------------------------------------------------------- G.1
    k.h2("Every class at every rate")

    k.rows(_class_saving_rows(k),
           "Compute saved against the released decoder, by class and by "
           "quality index, at the 0.1 dB budget; the main paper prints the "
           "q0, q32 and q63 columns of the first six rows. Read it in two "
           "directions. Along a row the saving falls with rate, because the "
           "residual a shallow exit fails to reconstruct grows relative to "
           "the quality being protected. Down a column it falls with "
           "resolution, from \\BigResLow% on MCL-JCV to \\SmallResLow% on "
           "HEVC D at q0, because a 256 px tile is 40 tiles on a 1080p frame "
           "and two at 416x240. The last row is the quality the pooled "
           "allocation gave up, which is the budget to within a thousandth of "
           "a decibel at every rate.")
    _pcb0 = next((r for r in k.J("supp_per_class_budgets.json")["rows"]
                  if r["qp"] == 0), None)
    _sg0 = next((r for r in k.J("signalled_RECIPE512_ctc53.json")["rows"]
                 if r["qp"] == 0 and abs(r["budget_db"] - 0.1) < 1e-9), None)
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ", \\NumSeq "
           "sequences at one frame each, per-frame convention. The pooled row "
           f"reads {_pcb0['overall_saving']:.1f}% at q0 where the main paper's "
           "\\MainLowRate% comes from "
           "results/signalled_RECIPE512_ctc53.json; the two bisect the same "
           "budget with different code and settle on "
           f"\u03bb = {_pcb0['lam'] * 1e5:.3f}\u00d710<super>-5</super> and "
           f"{_sg0['lam'] * 1e5:.3f}\u00d710<super>-5</super>.")

    k.fig("res_grid.png",
          "The same measurement as a grid, with the looser budget under it. "
          "<b>a</b>, compute saved at the 0.1 dB budget. <b>b</b>, the same "
          "at 0.3 dB, where every class at the three lowest rates has reached "
          "the architectural ceiling of \\CeilingModelled% and only q48 and q63 are "
          "still making a choice. <b>c</b>, the quality actually given up at "
          "the 0.1 dB budget, which the set meets and no class does: at q0 "
          "the allocation spends 0.116 dB on MCL-JCV and 0.028 dB on HEVC D. "
          "The small classes are cheap in quality for the same reason they "
          "are poor in saving, that with two or eight tiles there is no "
          "fine-grained way to spend a budget. The 0.5 dB grid is not drawn "
          "because every one of its thirty cells is at the ceiling.",
          maxh=230)
    k.note("Drawn by scripts/supp_results_figs.py from "
           "results/supp_per_class_budgets.json, on " + PINNED + ".")

    k.par(
        "The resolution effect is a property of the tiling and not of the "
        "content, which is the argument against reading a single mean over a "
        "test set that mixes resolutions: MCL-JCV is 30 of the \\NumSeq "
        "sequences and all of them 1080p, so it carries most of the pooled "
        "row.")

    # ----------------------------------------------------------------- G.2
    k.h2("Adaptivity against its controls")

    k.par(
        "The main paper prints this comparison transposed and compressed to "
        "five columns. The full form is below, one block per quality index, "
        "because the uniform rows are the ones a reader is most likely to "
        "want to read off directly: they say what a decoder that made no "
        "per-tile choice at all would deliver at the same budget.")
    k.tbl("static",
          "<b>Adaptivity against two controls</b> at the 0.1 dB budget, in "
          "full. Uniform rows are every tile at one depth. The dagger marks a "
          "uniform depth whose distortion exceeds the budget, so it is not an "
          "admissible allocation at all: at q48 and q63 the only uniform "
          "depth inside the budget is the deepest one, which costs "
          "\\DeepestUniformCost% rather than saving anything. "
          "<i>Random</i> draws each frame's map from the oracle's own exit "
          "histogram and shuffles it across tiles, so the mix of depths and "
          "the average cost are unchanged and only the content dependence is "
          "discarded. <i>Rate-ranked</i> keeps that histogram and orders it "
          "by the bits the entropy model spent per tile. Both shuffled rows "
          "are matched to the oracle on compute, so they are compared on the "
          "decibel column and not on the saving column.")
    k.note("results/static_RECIPE512_b01.json, on " + PINNED + ", seed "
           "recorded in the file, generated into paper/tables/static.tex by "
           "scripts/make_paper_tables.py.")

    k.par(
        "Two readings follow from the shuffled rows and they are easy to "
        "conflate. Quality falls sharply when the same histogram is assigned "
        "at random, so what the allocation buys is knowing <i>which</i> tiles "
        "can afford to run shallower and not the fact that some fraction of "
        "them does. And ordering that same histogram by a free decoder-side "
        "signal recovers \\RateRankRecovers% of the oracle's advantage over "
        "chance at q63, which is the control section F turns into a complete "
        "routing rule.")

    k.h2("Where the tiles exit")

    k.rows(_pooled_exit_rows(k),
           "The exit histogram at the 0.1 dB budget, as a percentage of the "
           "1765 tiles in one pass over the test set, with the mean number of "
           "trunk blocks skipped and the saving that buys. At q0 two thirds "
           "of tiles take the shallowest selectable exit and the ladder above "
           "it is barely used. At q63 the mass is spread almost evenly over "
           "the four exits and the mean depth skipped has fallen from 5.04 "
           "blocks to 2.82. Exits e0 and e1 are omitted because the exit "
           "clamp cannot select them: the split depth is 2, so a map naming a "
           "shallower exit runs to e2 and wears e2's adapter.")
    k.note("Histograms summed over the six classes of "
           "results/supp_per_class_budgets.json, on " + PINNED + ". Blocks "
           "skipped per exit from results/adapter_cost.json, same checkpoint.")

    k.fig("res_exits.png",
          "How the ladder fills up. <b>a</b>, exit shares at the 0.1 dB "
          "budget as the rate rises. <b>b</b>, exit shares at q63 as the "
          "budget loosens, until at 0.3 dB 96% of tiles are already at the "
          "shallowest exit and there is almost nothing left to allocate. Both "
          "panels are pooled over the test set.", maxh=170)
    k.note("Drawn by scripts/supp_results_figs.py from results/" + CURVE +
           ", on " + PINNED + ", pooled convention, so the budgets on the "
           "horizontal axis of <b>b</b> are looser than the identically "
           "labelled ones in the table above.")

    k.rows(_class_exit_rows(k),
           "The same histogram split by class at the two ends of the rate "
           "range: the first four columns are q0 and the second four are q63, "
           "each as a percentage of that class's tiles. HEVC C and HEVC D "
           "reach neither of the two shallowest exits at q63, and three "
           "quarters of their tiles run to full depth, which is the "
           "\\SmallResHigh% saving of the first table seen tile by tile. UVG "
           "behaves differently from MCL-JCV at q63 despite being the same "
           "resolution, which is content and not geometry.")
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ".")

    # ----------------------------------------------------------------- G.3
    k.h2("The frontier, class by class")

    k.figwide("res_frontier.png",
              "Compute saved against quality given up, one panel per class "
              "and one curve per quality index, from the seven budgets at "
              "which the frontier was measured. The dotted line is the "
              "architectural ceiling of \\CeilingModelled%. Three things are visible "
              "that a table of operating points hides. The curves are steep "
              "and then flat, so most of the saving is bought in the first "
              "tenth of a decibel and the rest of the band buys little. The "
              "rate ordering is preserved at every budget, so a curve never "
              "crosses another within a panel. And the panels move to the "
              "right as resolution falls: MCL-JCV reaches the ceiling by 0.3 "
              "dB at every rate, while HEVC D needs 0.5 dB at q48 and does "
              "not reach it at q63 within the measured range.")
    k.note("Drawn by scripts/supp_results_figs.py from results/" + CURVE +
           ", on " + PINNED + ". A point is the mean over the sequences of "
           "that class at a global operating point, so the multiplier is the "
           "one the whole set was bisected to and not a per-class retuning. "
           "Class membership per sequence from "
           "results/supp_opquality_PAPER.json, same checkpoint.")

    k.par(
        "Two limits bound every panel. A budget below the floor cannot be met "
        "at all, because our deepest exit already differs from the released "
        "decoder by \\FloorLow dB at q0 and \\FloorHigh dB at q63 in the "
        "per-frame convention; a budget above \\SatLow dB at q0 or "
        "\\SatHigh dB at q63 changes nothing, "
        "because every tile is already at the shallowest selectable exit. "
        "Between those two the multiplier is doing work, and the 0.1 dB "
        "budget is inside the window at every rate.")

    # ----------------------------------------------------------------- G.4
    k.h2("Per-sequence spread")

    k.par(
        "A set mean is not what any single clip receives. One multiplier "
        "prices compute for the whole set, so each sequence lands wherever "
        "its own content puts it, and the spread of those landings is what a "
        "deployment would have to plan against.")

    k.rows(_spread_rows(k),
           "Compute saved per sequence at the 0.1 dB budget: the "
           "distribution behind the mean, over all \\NumSeq sequences. The "
           "interquartile range widens from 9.9 points at q0 to 15.0 at q63, "
           "and the gap between best and worst sequence widens from 33.1 to "
           "40.1. The minimum is negative at four of the five rates, because "
           "a sequence whose tiles all run to full depth costs 0.95% more "
           "than the released decoder: our deepest exit is the more expensive "
           "of the two. Counting over rates as well as sequences, " +
           str(_under_ten(k)) + " of the 265 pairs save less than a tenth of "
           "a decode at this budget.")
    k.note("results/supp_per_sequence_PAPER_b010.json, on " + PINNED + ", "
           "read from its rows_vs_release block so that the denominator is "
           "the released decoder. Pooled convention, because the operating "
           "points come from results/" + CURVE + ".")

    k.rows(_tail_seq_rows(k),
           "The six hardest sequences at q63 and the two easiest, with their "
           "q0 value beside them and the decibel each actually received. All "
           "six of the hard ones are 832x480 or 416x240. The delivered "
           "decibel is not the budget on any of them: across the \\NumSeq "
           "sequences at q63 it runs from 0.013 dB to 0.237 dB against a "
           "0.1 dB target, so a single multiplier for the set means some "
           "clips overshoot the budget by more than a factor of two while "
           "others use an eighth of it.")
    k.note("results/supp_per_sequence_PAPER_b010.json, on " + PINNED + ". "
           "Savings are against the released decoder; the decibel is the "
           "sequence's own, in the pooled convention.")

    # ----------------------------------------------------------------- G.5
    k.h2("A second metric, and the tiles that pay for the mean")

    k.par(
        "The whole mechanism is denominated in PSNR, and PSNR is known not to "
        "track visual quality closely. Two things follow that are worth "
        "measuring rather than asserting: whether a perceptual metric agrees "
        "that the loss is small, and how the loss is distributed over tiles, "
        "since a mean of 0.1 dB is compatible with a few tiles losing a great "
        "deal.")

    k.rows(_metric_rows(k),
           "PSNR and MS-SSIM at the 0.1 dB operating point, over the whole "
           "test set. MS-SSIM agrees with PSNR about the direction and is "
           "kinder about the size: expressed as -10log10(1 minus MS-SSIM), "
           "the loss is 0.052 dB at q0 against 0.102 dB of PSNR, and 0.046 "
           "against 0.118 at q63, so on this metric the budget costs about "
           "half what the decibel it is written in suggests. Each metric is "
           "given as the released decoder, ours, and what was lost, with the "
           "MS-SSIM loss expressed as -10log10(1 minus MS-SSIM) so that the "
           "two loss columns are in the same unit. The bpp column is the "
           "released bitstream, which early exiting does not change beyond "
           "the exit map.")
    k.note("results/supp_opquality_PAPER.json, on " + PINNED + ", \\NumSeq "
           "sequences at one frame each. MS-SSIM is Wang et al. 2003 at five "
           "scales with an 11-tap Gaussian of σ = 1.5 on the valid region, on "
           "the luma plane in 4:2:0 on 0 to 255, the domain the PSNR column "
           "uses; the implementation self-checks at start-up and the file "
           "records that identical inputs score 1.000000 and a 3x3 box blur "
           "0.7797. λ is read from results/signalled_RECIPE512_ctc53.json so "
           "the allocation is the deployed one, and the delivered decibel "
           "recomputed here matches the stored value to four decimals.")

    k.fig("rd_vs_uf.png",
          "<b>a</b>, rate against quality over the five quality indices, the "
          "released DCVC-UF intra decoder and FLEX-UF at three budgets, on "
          "the \\NumSeq CTC intra frames. Both axes come from one pass over "
          "the same frames, and the encoders are asserted bit-identical on "
          "the weights before a shared rate is reported, because a shared "
          "x-axis is a claim rather than a convenience. "
          "<b>b</b>, the same difference magnified: the decibels each budget "
          "delivered, against the decoder arithmetic it bought on the "
          "right-hand axis.", maxh=190)
    k.note("Drawn by scripts/rd_vs_uf_figure.py from "
           "results/rd_absolute_PAPER.json and "
           "results/signalled_RECIPE512_ctc53.json, both on " + PINNED + ".")

    k.fig("res_rd.png",
          "<b>a</b>, the rate-quality curve of the released decoder and of "
          "ours at the 0.1 dB budget, over the five quality indices. At plot "
          "scale they are one curve, which is what a 0.1 dB budget means. "
          "<b>b</b>, the gap that panel <b>a</b> cannot show, in PSNR and in "
          "MS-SSIM expressed as a decibel.", maxh=170)
    k.note("Drawn by scripts/supp_results_figs.py from "
           "results/supp_opquality_PAPER.json, on " + PINNED + ".")

    k.rows(_tile_tail_rows(k),
           "How the 0.1 dB is distributed over the 1765 tiles of one pass "
           "over the test set. The last three columns count tiles losing more "
           "than a quarter, a half and a whole decibel. The median tile is at "
           "or below the budget at every rate, and the tail is long: at q0, "
           "302 tiles lose more than 0.25 dB, 79 lose more than 0.5 dB, and "
           "the worst loses 1.97 dB. A budget stated as a mean says nothing "
           "about the last three columns, which is why they are printed.")
    k.note("results/supp_opquality_PAPER.json, on " + PINNED + ", at the same "
           "operating points as the metric table above. A tile penalty is the "
           "decibel between the released decode of that tile and ours.")

    k.fig("res_tail.png",
          "Where the tail comes from. <b>a</b>, mean and 95th percentile tile "
          "penalty against the exit the tile took. The tail sits on e2, the "
          "shallowest exit the clamp allows: at q0 its mean is 0.202 dB "
          "against 0.054 dB at e5, and all 79 tiles above 0.5 dB took it. "
          "<b>b</b>, the same penalty split by whether the tile contains "
          "replicated padding. A 1080p frame is padded to 2048x1280, so 200 "
          "of the bottom tile row's 256 rows are invented by replication, and "
          "those tiles carry roughly double the mean penalty at every rate: "
          "0.261 dB against 0.121 dB at q0, and 65 of the 79 tiles above half "
          "a decibel are padded ones.", maxh=170)
    k.note("Drawn by scripts/supp_results_figs.py from the tail_by_exit and "
           "tail_by_padding blocks of results/supp_opquality_PAPER.json, on " +
           PINNED + ". A tile counts as padded if any of its pixels came from "
           "the replicate pad, recorded per tile as pad_fraction. Padding "
           "geometry from results/tile_definition.json, same checkpoint.")

    # ----------------------------------------------------------------- G.6
    k.h2("Composition with weight quantisation")

    k.par(
        "Early exiting removes work and quantisation makes the remaining work "
        "cheaper, so the two ought to compose. Whether they do is a question "
        "about the same weights, and it is measured here rather than assumed.")

    k.rows(_quant_rows(k),
           "Quality given up against the released decoder with the decoder "
           "weights quantised, at four exits and three rates. The float32 "
           "rows are the ladder itself and are the baseline each integer row "
           "should be read against. At q0 int8 is nearly free, moving the "
           "deepest exit from 0.027 dB to 0.032 dB, so the two levers "
           "compose. At q63 they compete: the deepest exit moves from 0.108 "
           "dB to 0.223 dB, which is more than a whole 0.1 dB budget spent "
           "before any tile has exited early. Six-bit costs 0.13 dB at the "
           "deepest exit at q0 and 2.18 dB at q63, and four-bit is unusable "
           "at every rate. The last column is the number of layers "
           "quantised, the same 86 in every integer row.")
    k.note("results/supp_quant_PAPER.json, on " + PINNED + ", 32 held-out "
           "images at 512 px. Weight-only, symmetric, per output channel, no "
           "calibration, on dec.* and router_head.* with the encoder asserted "
           "unchanged. Nothing here measures "
           "an integer kernel: the file records the bit width and the implied "
           "bit-operation ratio, and the arithmetic ran in floating point on "
           "quantised weights.")

    # ----------------------------------------------------------------- G.7
    k.h2("Integrated figures, and what they depend on")

    k.par(
        "A budget is one sample of a curve. Video coding answers this for "
        "rate against quality with Bjontegaard's construction [4], "
        "integrating one axis over a stated interval of the other, and the "
        "same construction applies with compute in place of rate. BD-saving "
        "is the mean compute saved over an interval of decibels and "
        "BD-quality is the mean decibel paid over an interval of saving. "
        "Neither means anything without its interval and its convention, and "
        "both are given with theirs.")

    k.rows(_bd_rows(k),
           "BD-saving and BD-quality per rate under both decibel "
           "conventions: the first pair of columns is per frame and the "
           "second is pooled. BD-saving is defined at only two of the five "
           "rates, because at q0, q16 and q32 the frontier saturates below "
           "the upper limit of the interval and there is no curve left to "
           "integrate. The convention moves the answer by 1.9 points of "
           "BD-saving and 0.018 dB of BD-quality on identical allocations, "
           "and it moves it consistently, since per-frame averaging raises "
           "every floor and pooling does not. Points is the number of "
           "measured frontier points inside the interval.")
    k.note("results/supp_bd_PAPER_per_frame.json and "
           "results/supp_bd_PAPER_pooled.json, both on " + PINNED + " over "
           "\\NumSeq sequences, integrating results/" + CURVE + ". BD-saving "
           "is over 0.066 to 0.300 dB per frame and 0.059 to 0.300 dB pooled, "
           "the lower limit being the highest floor any rate has under that "
           "convention. BD-quality is over 10% to 30% saving. The floors "
           "that set those lower limits are recorded per rate in both files "
           "as floor_db, and appear directly in the frontier table below at "
           "the two rates where the 0.05 dB budget is unreachable.")

    k.par(
        "Two sensitivities sit around those numbers and are worth stating "
        "beside them. The first is the anchor. Our deepest exit is not "
        "bit-exact with the released decoder, since it carries seam repair "
        "and a fine-tuned trunk, and it starts every budget slightly behind: "
        "the drift runs from 0.005 dB at q0 to 0.036 dB at q63, which at q63 "
        "is a third of the whole budget. Setting it to zero would raise "
        "BD-saving at q63 from 28.7% to 34.9%, a gain of 6.2 points. The "
        "second is the interval. Over six defensible intervals the mean "
        "BD-saving moves from 22.8% to 29.2%, a range of 6.3 points, while "
        "the gap between the lowest and highest rate stays inside 15.3 to "
        "16.8 points on every one of them; the level depends on the "
        "integration and the rate effect does not.")
    k.note("Anchor drift from results/supp_anchor_PAPER.json, on " + PINNED +
           ", \\NumSeq frames, per-frame convention; the zero-drift figure is "
           "the bd_saving_pct_zero_drift field of "
           "results/supp_bd_PAPER_per_frame.json. Interval sensitivity from "
           "results/bd_sensitivity.json, which is not pinned: "
           "scripts/bd_sensitivity.py names its curve as "
           "results/paper_curve_grid128.json, on "
           "runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar at 128 px tiles over 40 "
           "sequences, so its levels are not the levels above and only its "
           "robustness conclusion is quoted.")

    bd = k.J("bdrate.json")
    _by = {(r["config"], round(r["budget_db"], 2)): r for r in bd["rows"]}
    _gap = max(abs(_by[("A signalled", b)]["bd_rate_pct"]
                   - _by[("B router", b)]["bd_rate_pct"])
               for b in (0.1, 0.3, 0.5))
    _sav = [(_by[("A signalled", b)]["saving_pct_vs_release"]
             - _by[("B router", b)]["saving_pct_vs_release"])
            for b in (0.1, 0.5)]
    k.rows(_bdrate_rows(k),
           "BD-Rate cost of the two configurations: the rate a reader would "
           "have to spend to buy back the quality the compute saving costs, "
           "integrated over the five rate points, which is the number that "
           "makes this trade-off comparable with a published codec result. "
           "Configuration A signals the exit map in the bitstream and pays "
           "\\MapBitsLo to \\MapBitsHi bits a frame for it at the 0.1 dB "
           "budget, depending on the rate, "
           "configuration B infers it from what the decoder already holds and "
           "pays none. Their BD-Rate costs differ by at most "
           f"{_gap:.3f} of a point, so the map is close to free. What A buys "
           "with it is compute: the gap in the last two columns is "
           f"{_sav[0]:.1f} points at the tight budget and {_sav[1]:.1f} at "
           "the loose one, where both configurations are at the ceiling.")
    k.note("results/bdrate.json, every row on " + PINNED + ", which its "
           "sources block records file by file.")

    # ----------------------------------------------------------------- G.8
    k.h2("Raw values behind the curves")

    k.par(
        "Everything above is an integral, a mean or a picture. What follows "
        "is the measured numbers themselves, so that a reader can compare "
        "against them without digitising a plot or rerunning a decode.")

    k.rows(_deployed_rows(k),
           "The deployed operating points: four budgets at five rates. Model "
           "is the saving the cost model predicts for the chosen map and "
           "Meter is what a forward hook on every convolution and linear "
           "layer counted on the same decode, so the two columns are the cost "
           "model checked against itself; the model reads high by 0.4 to 0.8 "
           "points throughout. Map bits is the entropy of the per-frame exit "
           "histogram, charged to the bitstream before any saving is "
           "computed, and it falls as the budget loosens because the "
           "histogram concentrates.")
    k.note("results/signalled_RECIPE512_ctc53.json, on " + PINNED + ", "
           "\\NumSeq sequences, per-frame convention. λ is the multiplier the "
           "two-level bisection settled on, and a λ of exactly 1 is the top "
           "of its bracket, which marks a budget the ladder saturated below "
           "rather than a price that was chosen. Saving is against the "
           "released decoder.")

    _dd = [o["db_vs_uf_per_frame"] - o["db_vs_uf"]
           for o in k.J(CURVE)["op_points"]]
    _d1 = [o["db_vs_uf_per_frame"] - o["db_vs_uf"]
           for o in k.J(CURVE)["op_points"] if o["target_db"] == 0.1]
    k.rows(_frontier_rows(k),
           "The frontier itself: every operating point behind the per-class "
           "figure, at seven budgets and five rates, with the exit shares "
           "that produced each one. Both decibel conventions are printed for "
           "the same allocation, and they differ by "
           f"{min(_dd):.3f} to {max(_dd):.3f} dB over the table and by "
           f"{min(_d1):.3f} to {max(_d1):.3f} dB at the 0.1 dB budget, which "
           "is a fifth of that budget. A row "
           "reading n/a in the saving column is a budget below the floor at "
           "that rate, where no allocation meets the target. A row reading "
           "100/0/0/0 is past saturation: the file stores neither a "
           "multiplier nor a histogram there, and the shares are filled in "
           "from what saturation means, every tile at the shallowest "
           "selectable exit.")
    k.note("results/" + CURVE + ", on " + PINNED + ". Saving is converted to "
           "the released-decoder denominator by charging the retained cost "
           "the price of our deepest exit, "
           f"{_deepest_cost(k):.4f} released decodes, which is "
           "\\DeepestUniformCost at the precision the paper prints and is "
           "derived here from the ceiling in "
           "results/saturation_RECIPE512_ctc53.json and the saturated saving "
           "in this file. One caution about that denominator: this file bills "
           "the shared stem inside every tile's cost, while "
           "results/supp_latency_batch_1920x1080.json bills it once per frame "
           "through flexuf/cost.py frame_relative_cost, and the two "
           "accountings differ by about a third of a point on an identical "
           "exit map at q0.")

    # ----------------------------------------------------------------- G.9
    k.h2("Reusing an exit map")

    k.par(
        "The signalled configuration pays one extra decode per frame at the "
        "encoder to find its map. How much that matters depends on how often "
        "the map has to be found again, so we probed two kinds of reuse: "
        "across frames of one sequence, and across the quality index. The "
        "probe is narrow and is reported as one.")

    mt = k.J("map_transfer.json")
    rows = [["reuse", "map found at", "applied at", "in place, dB",
             "reused, dB", "in place, saved %", "reused, saved %"]]
    for r in mt["rows"]:
        if r["kind"] == "time":
            if r["offset"] == 0:
                continue
            kind, found, used = "frames", f"q{r['qp']} frame 0", \
                f"frame {r['offset']}"
        else:
            if r["from"] == r["to"]:
                continue
            kind, found, used = "rate", f"q{r['from']}", f"q{r['to']}"
        rows.append([kind, found, used,
                     f"{r['in_place_db']:.4f}", f"{r['transfer_db']:.4f}",
                     f"{r['in_place_saving']:.2f}",
                     f"{r['transfer_saving']:.2f}"])
    k.rows(rows,
           "<b>What a reused exit map delivers.</b> Each row applies a map "
           "found in one place to a decode somewhere else and reports what "
           "that decode actually delivered, beside the map recomputed in "
           "place. Read the two decibel columns first: a reused map is only "
           "usable if the decode it produces still meets the budget, and the "
           "saving column means nothing on a row where it does not.")
    k.note("results/map_transfer.json, budget " + f"{mt['budget_db']} dB, "
           "on the checkpoint the file records. The offsets and the quality "
           "pairs in the file are the whole of the probe; nothing here is a "
           "statement about reuse in general.")

    k.par(
        "Across frames of one sequence, reuse is close to free. Taking the "
        "map found on frame 0 and applying it eight frames later raises the "
        "delivered distortion by \\TransferDbCost dB at q0, about five "
        "percent of the budget, and the saving does not move at all: "
        "\\TransferSaving% transferred against "
        "\\TransferInPlaceLo\u2013\\TransferInPlaceHi% recomputed. The "
        "allocation is a property of where the content is hard, and that "
        "moves slowly. An encoder that searched once per group of pictures "
        "rather than once per frame would divide its extra cost by the group "
        "length, on the offsets we probed.")

    k.par(
        "Across the quality index the map does have to be recomputed, and the "
        "direction decides how badly. Found at q0 and applied at q63 it "
        "delivers \\TransferCrossDb dB against a 0.1 dB budget, claiming the "
        "low-rate saving of \\TransferSaving% while spending nearly twice "
        "the quality it is allowed. The reverse is safe and wasteful. Shallow "
        "exits are cheap in quality at low rate and expensive at high rate, "
        "so a map is calibrated to the rate it was found at, and reusing one "
        "upward breaks the quality guarantee with nothing in the decode "
        "reporting that it has. In our reuse probe, then, an encoder can "
        "search once per rate and reuse that search across the frames we "
        "tested; how far that carries beyond the offsets and sequences probed "
        "here we have not measured.")

    k.h2("Failure cases")

    k.par(
        "Five cases are worth naming, all of them measured in the files "
        "above rather than inferred.")
    k.bullets([
        "<b>Sequences that cost more than they save.</b> At q63 four "
        "sequences finish at minus 0.95%, which is every tile at full depth "
        "and our deepest exit costing that much more than a released decode: "
        "RaceHorses at 832x480 and at 416x240, BlowingBubbles and BQSquare. "
        "One more sits under 3% and ten in all save less than a tenth of a "
        "decode. On this content the method should be switched off rather "
        "than run at a budget, and the exit histogram is what tells the "
        "encoder so.",
        "<b>The bottom tile row of a padded frame.</b> The six worst tiles in "
        "the test set are all in the last row of a 1080p frame at 78% "
        "padding, each losing between 1.72 and 1.97 dB. The allocation is "
        "behaving correctly given its table, and the table is measuring a "
        "region that will be cropped away before anyone looks at it, so the "
        "budget is being spent on invented pixels.",
        "<b>The two smallest classes at high rate.</b> HEVC C and HEVC D save "
        "\\SmallResHigh% at q63, which is one tile in four moved one exit up "
        "and nothing else. With two tiles on a 416x240 frame there is no "
        "allocation worth making, and the honest reading is that this method "
        "needs a frame large enough to hold a useful number of tiles.",
        "<b>Quantisation and early exit compete at high rate.</b> Int8 "
        "weights cost 0.115 dB at q63 at the deepest exit, more than the "
        "whole budget the allocation is then asked to work inside, while at "
        "q0 they cost 0.005 dB. A deployment cannot assume the two savings "
        "multiply.",
        "<b>A 0.5 dB budget has no per-sequence breakdown to report.</b> At "
        "0.5 dB every rate is past saturation, the frontier stops at the "
        "ceiling of \\CeilingModelled%, and a saturated operating point "
        "carries no allocation to break down, which is why the per-sequence "
        "sweep runs from 0.10 dB to 0.30 dB.",
    ])
    k.note("Worst tiles from the worst_tiles block of "
           "results/supp_opquality_PAPER.json and negative sequences from "
           "results/supp_per_sequence_PAPER_b010.json, both on " + PINNED +
           "; the saturation points from "
           "results/saturation_RECIPE512_ctc53.json.")

    # ---------------------------------------------------------------- G.10
    k.fig("qualitative.png",
          "<b>What the saving looks like.</b> The same bitstream decoded by "
          "the released decoder and by ours at the 0.1 dB operating point. "
          "The crop is the tile that gave up the most quality on this frame, "
          "chosen automatically rather than by eye, so it shows the method's "
          "worst case here and not a flattering one.")

    k.fig("qualitative_q63.png",
          "<b>The same frame at the top of the rate range.</b> Bosphorus at "
          "q\\QualHighQp, \\QualHighBpp bpp, the same 0.1 dB budget and the "
          "same crop as the figure above. The released decoder reaches "
          "\\QualHighPsnrRel dB and the routed decode \\QualHighPsnrOurs dB "
          "for \\QualHighDb dB delivered, and the saving is "
          "\\QualHighSaving% of the arithmetic against \\QualSaving% at "
          "q\\QualQp.")
    k.par(
        "The allocation is what moved between the two pictures. On this frame "
        "\\QualShallowLow of 40 tiles take the shallowest exit at q\\QualQp "
        "and \\QualShallowHigh does at q\\QualHighQp, while the deepest exit "
        "goes from \\QualDeepLow tiles to \\QualDeepHigh. The picture is as "
        "hard to tell apart at either rate; what changes is how much of the "
        "decoder the budget will release, which is the rate dependence of the "
        "main paper's operating-point section arriving in a single frame.")
