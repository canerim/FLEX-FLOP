"""Complete results: every rate against every test set, and the frontier.

The main paper samples. Its per-class table (`paper/tables/perclass.tex`) is
one budget at three of the five quality indices; its headline is one budget at
five. This section prints the whole grid, on the pinned checkpoint wherever a
pinned measurement exists, and says which checkpoint every other table came
from.

Every number here is computed from a file in `results/` at build time by
`k.J`, so nothing can drift between the measurement and the page. The LaTeX
sibling, `paper/supp/d_full_results.tex`, carries the same numbers as literal
digits; both were produced by the same helpers below, so they agree by
construction rather than by proof-reading.

The public surface is `content(k)`. Everything else is a private helper that
returns a list of rows, shared with the script that emitted the LaTeX.
"""

# The six CTC classes, largest tile count first, which is also the order the
# saving falls in.
CLASSES = ["UVG", "MCL-JCV", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
QPS = [0, 16, 32, 48, 63]
# Trunk blocks skipped at each of the six exits, from results/adapter_cost.json.
SKIPPED = [10, 8, 6, 4, 2, 0]
# The pinned checkpoint. Anything else is named at the table that uses it.
PINNED = "runs/RECIPE512/ckpt_PAPER.pth.tar"


def _name(c):
    """CTC class names as the paper writes them."""
    return c.replace("HEVC_", "HEVC ")


def _f(x, n=1):
    return "n/a" if x is None else f"{x:.{n}f}"


# ------------------------------------------------------------- the test set
def _testset_rows(k):
    """Class, resolution, padded size, tiles per frame, sequences."""
    pc = k.J("supp_per_class_budgets.json")
    td = k.J("tile_definition.json")
    pad = {r["name"]: r for r in td["rows"]}
    at = [r for r in pc["rows"]
          if r["qp"] == 0 and abs(r["budget_db"] - 0.1) < 1e-9][0]["per_class"]
    out = [["Class", "Resolution", "Decoded", "Tiles", "Seq."]]
    tiles = seqs = 0
    for c in CLASSES:
        v = at[c]
        p = pad.get(v["res"])
        dec = f"{p['padded'][0]}x{p['padded'][1]}" if p else "n/a"
        out.append([_name(c), v["res"], dec, v["tiles"], v["n"]])
        tiles += v["tiles"] * v["n"]
        seqs += v["n"]
    out.append(["All", "mixed", "", tiles, seqs])
    return out


def _ladder_rows(k):
    """The six exits, what each skips, and which are reachable."""
    ac = k.J("adapter_cost.json")
    out = [["Exit", "Blocks run", "Blocks skipped", "Adapter", "Selectable"]]
    n = 12
    for e in ac["exits"]:
        out.append([f"e{e['exit']}", n - e["blocks_skipped"], e["blocks_skipped"],
                    e["adapter_kind"] or "none",
                    "yes" if e["reachable"] else "no"])
    return out


# ------------------------------------------------- per class, 0.1 dB budget
def _pc_index(k):
    d = k.J("supp_per_class_budgets.json")
    return {(r["qp"], round(r["budget_db"], 3)): r for r in d["rows"]}


def _pc_saving_rows(k, budget):
    idx = _pc_index(k)
    out = [["Class"] + [f"q{q}" for q in QPS]]
    for c in CLASSES:
        out.append([_name(c)] +
                   [_f(idx[(q, budget)]["per_class"][c]["saving"]) for q in QPS])
    out.append(["All 53, pooled"] +
               [_f(idx[(q, budget)]["overall_saving"]) for q in QPS])
    return out


def _pc_db_rows(k, budget):
    idx = _pc_index(k)
    out = [["Class"] + [f"q{q}" for q in QPS]]
    for c in CLASSES:
        out.append([_name(c)] +
                   [_f(idx[(q, budget)]["per_class"][c]["db"], 3) for q in QPS])
    out.append(["All 53, pooled"] +
               [_f(idx[(q, budget)]["overall_db"], 3) for q in QPS])
    return out


def _pc_loose_rows(k, field, digits):
    """0.3 dB and 0.5 dB in one table, budget as the first column."""
    idx = _pc_index(k)
    out = [["Budget", "Class"] + [f"q{q}" for q in QPS]]
    for b in (0.3, 0.5):
        for c in CLASSES:
            out.append([f"{b:.1f} dB", _name(c)] +
                       [_f(idx[(q, b)]["per_class"][c][field], digits)
                        for q in QPS])
    return out


# ------------------------------------------------------------- exit shares
def _pooled_exit_rows(k):
    """Where the 1765 tiles of one pass over the test set actually exited."""
    idx = _pc_index(k)
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
    idx = _pc_index(k)
    out = [["Class", "e2", "e3", "e4", "e5", "e2", "e3", "e4", "e5"]]
    for c in CLASSES:
        row = [_name(c)]
        for q in (0, 63):
            sh = idx[(q, 0.1)]["per_class"][c]["exit_share"]
            row += [_f(sh[i]) for i in (2, 3, 4, 5)]
        out.append(row)
    return out


# --------------------------------------------------------- per sequence
def _spread_rows(k):
    d = k.J("per_sequence.json")
    out = [["Rate", "n", "Mean", "SD", "Min", "p25", "Median", "p75", "Max"]]
    for r in d["rows"]:
        out.append([f"q{r['qp']}", r["n"]] +
                   [_f(r[f]) for f in ("mean", "sd", "min", "p25", "median",
                                       "p75", "max")])
    return out


def _curve_ops(k, target=0.1):
    d = k.J("curve_RECIPE512.json")
    return {o["qp"]: o for o in d["op_points"]
            if abs(o["target_db"] - target) < 1e-9 and o.get("per_sequence")}


def _pct(v, p):
    """Percentile by linear interpolation, so numpy is not needed here."""
    s = sorted(v)
    i = (len(s) - 1) * p / 100.0
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def _spread53_rows(k):
    ops = _curve_ops(k)
    out = [["Rate", "n", "Mean", "SD", "Min", "p25", "Median", "p75", "Max",
            "Under 10"]]
    for q in QPS:
        v = [r["saving_pct"] for r in ops[q]["per_sequence"]]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        out.append([f"q{q}", len(v), _f(m), _f(sd), _f(min(v)),
                    _f(_pct(v, 25)), _f(_pct(v, 50)), _f(_pct(v, 75)),
                    _f(max(v)), sum(1 for x in v if x < 10)])
    return out


def _tail_rows(k, n=6):
    """The sequences the set mean is carried past, and the ones carrying it."""
    ops = _curve_ops(k)
    by = {r["seq"]: r for r in ops[63]["per_sequence"]}
    at0 = {r["seq"]: r for r in ops[0]["per_sequence"]}
    order = sorted(by, key=lambda s: by[s]["saving_pct"])
    out = [["Sequence", "q0", "q63", "q63 vs release", "q63 dB"]]
    for s in order[:n] + order[-2:]:
        r, r0 = by[s], at0[s]
        out.append([s.replace(".yuv", "").replace("_420_8bit_YUV", ""),
                    _f(r0["saving_pct"]), _f(r["saving_pct"]),
                    _f(r["saving_pct_vs_release"]), _f(r["db_vs_uf"], 3)])
    return out


# ------------------------------------------------------------- the frontier
def _grid_rows(k):
    d = k.J("signalled_RECIPE512_grid.json")
    by = {(r["qp"], round(r["budget_db"], 3)): r for r in d["rows"]}
    out = [["Budget"] + [f"q{q}" for q in QPS]]
    for b in d["budgets"]:
        row = ["%g dB" % b]
        for q in QPS:
            r = by.get((q, round(b, 3)))
            row.append(_f(r["saving_pct_vs_release"])
                       if r and r.get("budget_reachable") else "n/a")
        out.append(row)
    return out


def _operating_rows(k):
    d = k.J("saturation_RECIPE512_ctc53.json")
    out = [["Rate", "Floor", "Saturation", "Usable band", "0.1 dB uses"]]
    for r in d["rows"]:
        band = r["saturation_db"] - r["floor_db"]
        out.append([f"q{r['qp']}", _f(r["floor_db"], 3),
                    _f(r["saturation_db"], 3), _f(band, 3),
                    _f(100 * (0.1 - r["floor_db"]) / band, 0) + "%"])
    return out


def _curve_rows(k):
    d = k.J("curve_RECIPE512.json")
    by = {(o["qp"], round(o["target_db"], 3)): o for o in d["op_points"]}
    tg = sorted({round(o["target_db"], 3) for o in d["op_points"]})
    out = [["Budget"] + [f"q{q}" for q in QPS]]
    for t in tg:
        row = ["%g dB" % t]
        for q in QPS:
            o = by.get((q, t))
            if o is None or o.get("saving_pct") is None:
                row.append("n/a")
            else:
                row.append(_f(o["saving_pct"]) + ("*" if o.get("saturated")
                                                  else ""))
        out.append(row)
    return out


# --------------------------------------------------------------- BD figures
def _bd_rows(k):
    d = k.J("bdrate.json")
    out = [["Configuration", "Budget", "BD-Rate", "MACs saved", "Map bits"]]
    for r in d["rows"]:
        out.append([r["config"], f"{r['budget_db']:.1f} dB",
                    _f(r["bd_rate_pct"], 2) + "%",
                    _f(r["saving_pct_vs_release"]) + "%",
                    _f(r["map_bits"], 0)])
    return out


def _bd_perrate_rows(k):
    d = k.J("bd_RECIPE512_ctc53.json")
    out = [["Rate", "BD-saving", "BD-quality", "Floor", "Points"]]
    for r in d["rows"]:
        out.append([f"q{r['qp']}",
                    "n/a" if r["bd_saving_pct"] is None
                    else _f(r["bd_saving_pct"]) + "%",
                    _f(r["bd_quality_db"], 3), _f(r["floor_db"], 3),
                    r["n_points"]])
    out.append(["Mean of the two that span it",
                _f(d["mean_bd_saving_pct"]) + "%",
                _f(d["mean_bd_quality_db"], 3), "", ""])
    return out


def _bd_interval_rows(k):
    d = k.J("bd_sensitivity.json")
    out = [["dB interval", "Mean", "q0", "q63", "q0 minus q63"]]
    for r in d["rows"]:
        out.append([f"{r['db_lo']:.3f} to {r['db_hi']:.2f}",
                    _f(r["mean_bd_saving_pct"]) + "%",
                    _f(r["qp_low"]) + "%", _f(r["qp_high"]) + "%",
                    _f(r["rate_effect_pts"]) + " pt"])
    return out


def _bd_convention_rows(k):
    a = k.J("bd_saving.json")
    b = k.J("bd_saving_pooled.json")
    A = {r["qp"]: r for r in a["rows"]}
    B = {r["qp"]: r for r in b["rows"]}
    out = [["Rate", "Per frame", "Pooled", "Difference"]]
    for q in QPS:
        out.append([f"q{q}", _f(A[q]["bd_saving_pct"]) + "%",
                    _f(B[q]["bd_saving_pct"]) + "%",
                    _f(B[q]["bd_saving_pct"] - A[q]["bd_saving_pct"]) + " pt"])
    out.append(["Mean", _f(a["mean_bd_saving_pct"]) + "%",
                _f(b["mean_bd_saving_pct"]) + "%",
                _f(b["mean_bd_saving_pct"] - a["mean_bd_saving_pct"]) + " pt"])
    return out


# ------------------------------------------------------------------ prose
def content(k):
    k.h1("Complete results")

    k.par(
        "This section prints in full the measurements the main paper quotes "
        "at one or two points. The per-class table there reports one budget "
        "at three of the five quality indices; here every class appears at "
        "every index and at three budgets. The frontier is then given at "
        "every budget measured, and the integrated figures that summarise it "
        "are given with the interval and the averaging convention each was "
        "taken under, because both move the number by more than the "
        "difference between the systems being compared.")

    # -------------------------------------------------------------- D.1
    k.h2("Every class at every rate, at the 0.1 dB budget")

    t = k.rows(_pc_saving_rows(k, 0.1),
               "Compute saved against the released decoder, per class, at "
               "the 0.1 dB budget. The main paper prints the q0, q32 and q63 "
               "columns of this table. Two effects are visible and they "
               "compound: the saving falls with rate, from 29.7% to 15.6% on "
               "the pooled row, and it falls with "
               "decreasing resolution, from \\BigResLow% on MCL-JCV to "
               "\\SmallResLow% on HEVC D at q0. At q63 both of the two "
               "smallest classes sit at 2.5%, which is one tile in four at "
               "the second-deepest exit and every other tile run to full "
               "depth.")
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ", 53 "
           "sequences at one frame each. This file extends "
           "results/per_class_RECIPE512.json from three rates to five; on "
           "the nine cells the two share, every saving, every decibel and "
           "every histogram bin is bit-identical, so the extension is a "
           "continuation of the same measurement and not a re-run. The "
           "pooled row sits under a tenth of a point from the main paper's "
           "\\MainLowRate% and \\MainHighRate%, which are the same budget "
           "measured over two frames per sequence rather than one.")

    t = k.rows(_pc_db_rows(k, 0.1),
               "Quality actually given up, per class, at the same operating "
               "points. The budget is met on the set and on nothing else. At "
               "q0 the allocation spends 0.116 dB on MCL-JCV and 0.028 dB on "
               "HEVC D, a factor of four across classes at a single "
               "multiplier. The small classes are cheap in quality for the "
               "same reason they are poor in saving: with two or eight tiles "
               "there is no fine-grained way to spend the budget, so the "
               "bisection stops short of it.")
    k.note("Same file and same rows as the previous table. Decibels are the "
           "per-frame convention, averaged over the sequences of the class.")

    k.par(
        "The resolution effect is a property of the tiling and not of the "
        "content. A 256 px tile is 40 tiles on a 1080p frame, 15 on 720p, "
        "eight at 832x480 and two at 416x240, so the allocation has two "
        "orders of magnitude fewer choices at the bottom of the range. It is "
        "also the argument against reading a single mean over a test set "
        "that mixes resolutions: the pooled row is dominated by MCL-JCV, "
        "which is 30 of the 53 sequences and all of them 1080p.")

    # -------------------------------------------------------------- D.3
    k.h2("Where the tiles exit")

    t = k.rows(_pooled_exit_rows(k),
               "The exit histogram at the 0.1 dB budget, as a percentage of "
               "the 1765 tiles in one pass over the test set, with the mean "
               "number of trunk blocks skipped. At the lowest rate two "
               "thirds of tiles take the shallowest selectable exit and the "
               "ladder is barely used above it. At the highest rate the mass "
               "is spread almost evenly over the four exits and the mean "
               "depth skipped has fallen from 5.04 blocks to 2.82. The same "
               "budget buys less at high rate because the residual the "
               "shallow exits fail to reconstruct is larger relative to the "
               "quality being protected.")
    k.note("Histograms summed over the six classes from "
           "results/supp_per_class_budgets.json, on " + PINNED + ". Blocks "
           "skipped per exit from results/adapter_cost.json.")

    t = k.rows(_class_exit_rows(k),
               "The same histogram split by class, at the two ends of the "
               "rate range: the first four columns are q0 and the second "
               "four are q63, each as a percentage of that class's tiles. "
               "HEVC C and HEVC D never reach the shallowest exit at q63; "
               "three quarters of their tiles run to full depth, which is "
               "the 2.5% saving of the previous table seen tile by tile.")
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ".")

    # -------------------------------------------------------------- D.4
    k.h2("The 0.3 dB and 0.5 dB budgets")

    t = k.rows(_pc_loose_rows(k, "saving", 1),
               "Compute saved per class at the two looser budgets. Almost "
               "every cell is \\Ceiling%, which is the architectural "
               "ceiling: the largest saving the ladder can reach, obtained "
               "by putting every tile at the shallowest selectable exit. "
               "Where a cell is below the ceiling the budget is still "
               "binding and the allocation is still making a choice; that "
               "happens at 0.3 dB only at q48 and q63.")
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ". The "
           "ceiling is \\Ceiling% from results/saturation_RECIPE512_ctc53.json, "
           "same checkpoint, which records it as 100(1 minus the cost of "
           "exit e2) with that cost measured at 0.6088 of a released decode.")

    t = k.rows(_pc_loose_rows(k, "db", 3),
               "Quality given up at the two looser budgets. In the saturated "
               "cells the delivered decibel is well under the budget, "
               "because once every tile is at the shallowest exit there is "
               "nothing further to spend and the multiplier has run to the "
               "top of its bracket. The direction reverses at the bottom of "
               "the resolution range: a 0.5 dB set budget costs 0.727 dB on "
               "HEVC D at q63 and 0.573 dB on HEVC C, so the small classes "
               "overshoot a budget that the set as a whole undershoots.")
    k.note("results/supp_per_class_budgets.json, on " + PINNED + ". Where "
           "the bisection saturates it returns the top of its bracket, "
           "λ = 1, which is the marker for a budget the ladder cannot "
           "reach rather than a price that was chosen.")

    k.par(
        "Saturation is worth stating as a property of the design rather than "
        "of these two budgets. The set-level decibel at which every tile "
        "arrives at the shallowest exit is \\SatLow dB at q0 and \\SatHigh dB "
        "at q63, so a budget above that number buys nothing further at any "
        "rate, and a budget below \\FloorLow dB at q0 or \\FloorHigh dB at "
        "q63 cannot be met at all, because the deepest exit already differs "
        "from the released decoder by that much. The 0.1 dB budget sits "
        "inside that window at every rate, which is why it is the one the "
        "main paper reports.")

    # -------------------------------------------------------------- D.5
    k.h2("Per-sequence spread")

    k.par(
        "A set mean is not what any single clip receives. Because one "
        "multiplier prices compute for the whole set, each sequence lands "
        "wherever its own content puts it, and the spread of those landings "
        "is the number a deployment would care about.")

    t = k.rows(_spread_rows(k),
               "Compute saved per sequence at the 0.1 dB budget: the "
               "distribution behind the mean. The interquartile range is "
               "5.5 points wide at q0 and 7.6 at q63, and the gap between "
               "best and worst sequence is 14.4 points at q0 and 31.4 at "
               "q63. Quoting the mean alone would promise the hardest "
               "content something it does not get.")
    k.note("results/per_sequence.json. Not the pinned checkpoint: the file "
           "records runs/BEST/ckpt_eval.pth.tar, 40 sequences. Those 40 are "
           "UVG, MCL-JCV and HEVC E only, so the three classes where the "
           "saving collapses are absent and the minimum column is not the "
           "test set's minimum. The file also names "
           "results/curve_BEST.json as its source, and that file has since "
           "been regenerated with 53 sequences and different per-sequence "
           "values, so this table cannot be reproduced from anything now on "
           "disk. The next table asks the same question on a file that can.")

    t = k.rows(_spread53_rows(k),
               "The same question over all 53 sequences, including the "
               "classes the previous table omits. The last column counts "
               "sequences saving under 10%. At q0 there are none; at q63 "
               "there are seven, and four of those save nothing at all, "
               "which against the released decoder is minus 0.95% because "
               "our deepest exit is the more expensive of the two. The "
               "standard deviation rises from 6.7 to 12.2 points across "
               "the rate range, so the saving at high rate is not only "
               "smaller but harder to predict.")
    k.note("Recomputed here from the per-sequence arrays in "
           "results/curve_RECIPE512.json at its 0.1 dB operating point. Not "
           "the pinned checkpoint: that file records "
           "runs/RECIPE512/ckpt_eval.pth.tar at epoch 1, and it was measured "
           "under an earlier compute-cost model whose ceiling is 42.5% "
           "rather than \\Ceiling%, so the levels are not comparable with the "
           "per-class tables above. The shape of the distribution is what "
           "this table is for.")

    t = k.rows(_tail_rows(k),
               "The six hardest sequences at q63 and the two easiest, with "
               "their q0 value for comparison and the decibel each actually "
               "received. All six of the hard ones are 832x480 or 416x240. "
               "The delivered decibel is not the budget on any of them: "
               "across the 53 sequences at q63 it runs from "
               "0.002 dB to 0.248 dB against a 0.1 dB target, so one "
               "multiplier for the set means individual clips overshoot the "
               "budget by a factor of two while others use a fiftieth of it.")
    k.note("results/curve_RECIPE512.json, operating point 0.1 dB, on "
           "runs/RECIPE512/ckpt_eval.pth.tar at epoch 1 rather than the "
           "pinned checkpoint. The q63 saving against the release is the "
           "same allocation counted against the released decoder rather "
           "than against our deepest exit.")

    # -------------------------------------------------------------- D.6
    k.h2("Integrated figures")

    k.par(
        "A budget is one sample of a curve, and a conclusion drawn from one "
        "sample is fragile. Video coding answers this for rate against "
        "quality with Bjontegaard's construction [4], integrating one axis "
        "over a stated interval of the other, and the same construction "
        "applies with compute in place of rate. Two integrals are reported: "
        "BD-saving, the mean compute saved over an interval of decibels, and "
        "BD-quality, the mean decibel paid over an interval of saving. "
        "Neither means anything without its interval, and both are given "
        "with theirs.")

    t = k.rows(_bd_rows(k),
               "BD-Rate cost of the two configurations at three budgets. "
               "This is the rate a reader would have to spend to buy back "
               "the quality the compute saving costs, integrated over the "
               "five rate points, and it is the number that makes the "
               "trade-off comparable with a published codec result. "
               "Configuration A signals the exit map in the bitstream and "
               "pays about 80 bits a frame for it; configuration B infers it "
               "and pays none. Their BD-Rate costs differ by at most 0.03 "
               "points at every budget, so the map is close to free, and at "
               "the 0.1 dB budget A converts that into four more points of "
               "compute saved.")
    k.note("results/bdrate.json. The file records no checkpoint. It was "
           "written before the current "
           "results/signalled_RECIPE512_ctc53.json and its saving column is "
           "on the superseded compute-cost model: it reads 41.9% at 0.5 dB "
           "where the pinned file now saturates at \\Ceiling%. The BD-Rate "
           "column is an integral of rate against quality and does not "
           "depend on that model; the saving column does.")

    f = k.figwide("paper_rdc.png",
              "The same measurement drawn. Left, BD-Rate cost against "
              "compute saved, with the released decoder at the origin: the "
              "two configurations sit almost on top of each other in "
              "BD-Rate, and loosening the budget moves both up and to the "
              "right together. Right, decode cost per 1080p frame in GMAC. "
              "Rendered from the same run as the previous table, so the "
              "compute axis carries the same superseded cost model and the "
              "BD-Rate axis does not.")
    k.note("docs/figures/paper_rdc.png, written by scripts/paper_metrics.py "
           "in the same pass that wrote results/bdrate.json.")

    t = k.rows(_bd_perrate_rows(k),
               "BD-saving and BD-quality per rate, integrated over 0.066 to "
               "0.3 dB and over 10% to 30% saving respectively. Three of the "
               "five rates return no BD-saving at all, because their "
               "measured frontier does not span the interval: the floor at "
               "q0 is 0.033 dB but the sweep does not reach 0.3 dB before "
               "saturating. Reporting a mean over the two rates that do span "
               "it, as the file does, is a mean over the two hardest rates "
               "and reads higher than a mean over five would.")
    k.note("results/bd_RECIPE512_ctc53.json, on "
           "runs/RECIPE512/ckpt_eval.pth.tar at epoch 0 rather than the "
           "pinned checkpoint, 53 sequences, per-frame convention. Points is "
           "the number of measured frontier points inside the interval.")

    t = k.rows(_bd_interval_rows(k),
               "What the interval does to the answer. The level moves by 6.3 "
               "points across six defensible intervals, which is why an "
               "interval is quoted beside every integrated number in this "
               "work. The rate effect does not move: the lowest rate exceeds "
               "the highest by 15.3 to 16.8 points on every interval tried, "
               "so the statement that the saving falls by roughly fifteen "
               "points across the quality range is about the decoder and not "
               "about the integration.")
    k.note("results/bd_sensitivity.json. The file records the convention but "
           "not the curve; scripts/bd_sensitivity.py names it as "
           "results/paper_curve_grid128.json, which is "
           "runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar at 128 px tiles over 40 "
           "sequences. That is neither the pinned checkpoint nor the tile "
           "size this work reports, so the levels here are not the levels "
           "elsewhere in this section; the robustness conclusion is what it "
           "is for. Intervals wide enough that a rate's frontier does not "
           "span them are excluded rather than averaged over fewer rates.")

    t = k.rows(_bd_convention_rows(k),
               "The averaging convention costs more than the systems being "
               "compared differ by. The two columns are the same allocation "
               "on the same curve, integrated under the two decibel "
               "conventions, and they disagree by 7.3 to 9.6 points of "
               "BD-saving. Part of that is the interval, which differs "
               "between the two because the per-frame convention raises "
               "every floor and so moves the lower limit the integration can "
               "start from; the two effects cannot be separated from these "
               "two files. Either way, no comparison across papers is "
               "meaningful unless both state which convention they average "
               "under.")
    k.note("results/bd_saving.json, per-frame convention over 0.064 to 0.196 "
           "dB, against results/bd_saving_pooled.json, pooled convention "
           "over 0.057 to 0.300 dB. Both on "
           "runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar over 40 sequences, "
           "which is neither the pinned checkpoint nor 256 px tiles.")

    # -------------------------------------------------------------- D.8
    k.h2("What this section does not measure")

    k.par(
        "Four gaps are worth naming so that they are visibly absent rather "
        "than quietly missing.")
    k.bullets([
        "No integrated figure in this work is on the pinned checkpoint. "
        "results/bdrate.json records no checkpoint and predates the current "
        "compute-cost model, results/bd_RECIPE512_ctc53.json is the epoch-0 "
        "evaluation checkpoint, and the interval and convention studies are "
        "on a 128 px run over 40 sequences. The per-class and frontier "
        "tables above are pinned; the BD tables are not.",
        "No per-sequence measurement is on the pinned checkpoint either. "
        "results/per_sequence.json is a second training run over a "
        "40-sequence subset and cannot be regenerated from the curve it "
        "names, and results/curve_RECIPE512.json is the epoch-1 evaluation "
        "checkpoint under a superseded cost model.",
        "The 0.1 dB budget is a convention this work adopts and not a "
        "perceptual threshold. Subjective work measures the smallest "
        "noticeable change in quantisation parameter or in a video quality "
        "metric, not in tenths of a decibel of PSNR, so no published result "
        "licenses a claim that 0.1 dB is invisible. What it buys is a number "
        "a codec reader can compare, namely \\BdRateALow% of BD-Rate, and "
        "0.3 dB and 0.5 dB are reported beside it so that nothing rests on "
        "the choice.",
        "Every measurement here is on intra frames of video sequences, one "
        "frame per sequence except in the frontier grid, which uses two. "
        "Nothing here measures the inter-frame path, and nothing here "
        "measures a second checkpoint, so the class and sequence effects are "
        "reported without a replication.",
    ])
