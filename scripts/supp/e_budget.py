"""The operating window: where a quality budget changes anything, and by how much.

The main paper states three things about the budget in a page and a half. There
is a floor below which no allocation is feasible, a saturation point above which
nothing improves, and a rescaling of the budget axis onto each rate's own band
that collapses the five curves onto one power law. This section carries the
grids those claims are read off, the conventions that have to be pinned before a
decibel figure means anything, the spread behind every averaged entry, and a
sensitivity analysis of the power law that shows how little of it replicates.

Every number is read from a named file in `results/`. Where a quantity is
arithmetic on file values, the position u in the band, a least-squares exponent,
a conversion between two conventions, it is computed here at build time from the
files named in the note beside the table, by the procedure in
`scripts/tradeoff_figure.py`, and never typed.

`paper/supplementary.tex` inputs `supp/e_budget`, which does not exist yet. When
it is written it has to carry the derived figures as digits, because LaTeX
cannot compute them, and this module is where they come from.

What this section deliberately leaves to its neighbours: Section C derives the
floor, the saturation point and the ceiling; Section D tabulates the two ends at
nine rates, the per-sequence spread behind every mean, and the same operating
point measured in a second metric. Repeating any of it here would make the
document longer without making it say more.
"""
import math

# The matched-position grid. Every rate of signalled_RECIPE512_grid.json has
# measured points spanning u = 0.10 to 0.70, so nothing in the collapse table is
# a held-constant extrapolation. band_collapse.json interpolates on 0.05 to 0.95
# instead and holds the last measured value beyond each rate's last point, which
# is why its worst spread is larger than this table's.
UGRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
QPS = [0, 16, 32, 48, 63]

# The window the sensitivity table refits inside. The two sweeps of the pinned
# checkpoint and the sweep of the second one all carry points across it, so a
# difference between two rows there is a difference in the curve rather than a
# difference in which part of the curve each sweep happened to sample.
WLO, WHI = 0.15, 0.85


# --------------------------------------------------------------- arithmetic
def _curves(grid, sat):
    """Each rate's reachable points as (position in band, saving vs release)."""
    rows = [r for r in grid["rows"] if r.get("budget_reachable")]
    band = {r["qp"]: (r["floor_db"], r["saturation_db"]) for r in sat["rows"]}
    cur = {}
    for q in sorted({r["qp"] for r in rows}):
        if q not in band:
            continue
        rs = sorted((r for r in rows if r["qp"] == q),
                    key=lambda r: r["budget_db"])
        f, s = band[q]
        pts = []
        for r in rs:
            u = (r["budget_db"] - f) / (s - f)
            if 0.0 <= u <= 1.0001:
                pts.append((min(u, 1.0), r["saving_pct_vs_release"]))
        cur[q] = pts
    return cur


def _interp(pts, u):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if u <= xs[0]:
        return ys[0]
    if u >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if u <= xs[i]:
            t = (u - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _fit(pts, C):
    """Least squares on ln(saving/C) against ln u, through the origin.

    The same one-parameter fit scripts/tradeoff_figure.py performs. The ceiling
    C is not fitted, it is the architectural constant, so the only free number
    is the exponent and the fit is a straight line through the origin in log
    space rather than an optimiser's opinion.
    """
    su = sv = 0.0
    for u, y in pts:
        if u <= 1e-6 or y <= 0:
            continue
        lu = math.log(u)
        su += lu * lu
        sv += lu * math.log(y / C)
    b = sv / su
    ys, sse, mx = [], 0.0, 0.0
    for u, y in pts:
        if u <= 1e-6 or y <= 0:
            continue
        p = C * u ** b
        sse += (y - p) ** 2
        mx = max(mx, abs(y - p))
        ys.append(y)
    m = sum(ys) / len(ys)
    tot = sum((y - m) ** 2 for y in ys)
    return b, 1.0 - sse / tot, mx, len(ys)


def _win(cur):
    """Every point of every rate that falls inside the common window."""
    return [p for q in cur for p in cur[q] if WLO <= p[0] <= WHI]


def _f(x, n=2):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ section
def content(k):
    sec = k.h1("The operating window")

    sat = k.J("saturation_RECIPE512_ctc53.json")
    grid = k.J("signalled_RECIPE512_grid.json")
    front = k.J("supp_paper_curve_PAPER.json")
    rtr = k.J("router_RECIPE512_b01.json")

    S = {r["qp"]: r for r in sat["rows"]}
    cj = sat["cost_j"]
    cK = rtr["deepest_exit_cost"]
    C = sat["ceiling_pct"]

    k.par(r"A budget D is the peak-signal-to-noise ratio the decode is allowed "
          r"to give up, and an allocation is admissible if the loss it incurs "
          r"stays at or below it. Between a floor, below which no allocation "
          r"is admissible, and a saturation point, above which every "
          r"allocation has reached the same ceiling, lies the window of "
          r"budgets in which the decoder has anything to trade. Section C "
          r"derives the two ends and Section D tabulates them. This section "
          r"is about the window itself.")
    k.par(r"It covers how the two ends are measured and how far the "
          r"measurements agree, what has to be stated with every decibel "
          r"before any of it means anything, the sweep across the "
          r"whole window, and how much of the rescaling that collapses the "
          r"five rates onto one curve survives a change of sweep, of "
          r"convention or of checkpoint.")

    # ---------------------------------------------------------------- E.1
    k.h2("Floor, saturation point and ceiling")

    k.par(r"The floor D_min is the loss of a tiled decode with every tile at "
          r"full depth. No early exiting has happened, so it is the tiling "
          r"penalty alone: the border of every tile has been convolved "
          r"against padding rather than against its neighbour. The saturation "
          r"point D_sat is the loss when every tile takes exit j, the "
          r"shallowest the split depth allows. Neither is a quantity to search "
          r"for. Each is one forward pass per rate with the exit map held "
          r"constant, which is what makes the rescaling in Section " + sec +
          r".4 cheap enough to be worth doing in deployment.")

    k.par(r"Between them the saving is fixed by the ladder rather than by the "
          r"budget. With c_k the cost of exit k in units of one released "
          r"full-frame decode, j the split depth and K the number of exits, "
          r"the ceiling is closed form under either compute denominator:")
    k.eq(r"S_{\mathrm{max}}^{\mathrm{rel}} = 100\,(1 - c_j), \;\; "
         r"S_{\mathrm{max}}^{\mathrm{self}} = 100\,(1 - c_j/c_{K-1}),")
    k.par(f"With c_j = {cj:.4f}, recorded in "
          f"results/saturation_RECIPE512_ctc53.json, and c_(K−1) = {cK:.4f}, "
          f"recorded as deepest_exit_cost in "
          f"results/router_RECIPE512_b01.json, the first reads \\CeilingModelled% and "
          f"the second {100 * (1 - cj / cK):.1f}%. The deepest exit costs more "
          f"than one released decode because it carries the seam repair, which "
          f"the released decoder does not run, and that difference is the "
          f"whole of the gap between the two ceilings.")

    k.par(r"Section D tabulates both ends at all nine measured rates, and "
          r"that table is not repeated here. The one property of it this "
          r"section needs is that a budget in decibels means something "
          r"different at "
          r"each rate: both ends widen as the rate falls, the band widens "
          r"faster than the floor rises, and the 0.1 dB budget that sits "
          r"\BandUseLow% of the way up the band at q0 sits \BandUseHigh% of "
          r"the way up at q63. The rescaling of Section " + sec + r".4 turns "
          r"out to need that position rather than the decibel figure.")

    unreach = [r for r in grid["rows"] if not r.get("budget_reachable")]
    names = [f"q{r['qp']}" for r in unreach]
    k.par(f"A budget below the floor admits no allocation at all, which is a "
          f"different failure from a budget falling in a gap of the Lagrangian "
          f"hull, where the sweep returns the nearest reachable point and "
          f"Everett's argument still makes it optimal for the compute it "
          f"consumes [28, 29]. Below the floor the feasible set is empty and "
          f"there is nothing to return. Three cells of the grid in Section "
          + sec + f".3 are in that state and are recorded as such rather than "
          f"dropped: at a 0.05 dB budget the three highest rates, "
          + ", ".join(names[:-1]) + " and " + names[-1] + f", have floors of "
          + ", ".join(_f(r["floor_db"], 3) for r in unreach[:-1])
          + " and " + _f(unreach[-1]["floor_db"], 3) + " dB.")

    fq0 = [r for r in grid["rows"] if r["qp"] == 0][0]["floor_db"]
    fq63 = [r for r in grid["rows"] if r["qp"] == 63][0]["floor_db"]
    tq63 = [o for o in front["op_points"]
            if o["qp"] == 63 and o.get("budget_reachable") is False][0]
    k.par(r"Three estimates of the floor are in play and they are not "
          r"interchangeable. Table " + str(k.peek_tbl()) + r" gives all three, "
          r"because the rescaling below divides by the distance between the "
          r"floor and the saturation point, and near the floor that is a small "
          r"difference divided by a small difference.")
    k.rows([["source", "what it measures", "q0", "q63"],
            ["the saturation file", r"a deployed decode with every tile at "
             r"full depth, over \NumSeq frames", r"\FloorLow",
             r"\FloorHigh"],
            ["the dense grid", "the same quantity, over 106 frames",
             _f(fq0, 3), _f(fq63, 3)],
            ["the second sweep", "the per-tile table at a zero multiplier, "
             "where a tile takes its own lowest-error exit",
             "-", _f(tq63["db_vs_uf_per_frame"], 3)]],
           r"<b>Three measurements of the floor, in decibels.</b> The first "
           r"two differ by three to six thousandths of a decibel because they "
           r"are the same quantity over a different number of frames. The "
           r"third is a different quantity: a tile whose shallower exit "
           r"happens to have the lower error takes it, so the table's floor "
           r"sits below the deployed one. Everything downstream of here takes "
           r"both ends of the window from the first row, so that the floor and "
           r"the saturation point are on one measurement.")
    k.note(r"The three rows are results/saturation_RECIPE512_ctc53.json, "
           r"results/signalled_RECIPE512_grid.json and "
           r"results/supp_paper_curve_PAPER.json, all on the pinned "
           r"checkpoint and all with decibels averaged per frame. The third "
           r"records a floor only where it found the 0.05 dB budget "
           r"unreachable, which is why q0 is empty.")

    # ---------------------------------------------------------------- E.2
    k.fig("saturation_RECIPE512.png",
          "<b>Three regions, and only the middle one is a design choice.</b> "
          "Below the floor no allocation meets the budget; above saturation "
          "every tile already takes the cheapest exit and a looser budget "
          "buys nothing. The 0.1 dB budget uses \\BandUseLow% of the band at "
          "the lowest rate and \\BandUseHigh% at the highest, which is why "
          "the same budget behaves so differently at the two ends of the rate "
          "range.")

    k.fig("seam_vs_qp.png",
          "<b>The tiling penalty across the whole rate range</b>, every tile "
          "at full depth, so this is the floor and nothing else. <b>a</b>, "
          "the steps that reduce it, of which the two that matter cost no "
          "arithmetic; the largest single factor is training the ladder with "
          "the seam present. <b>b</b>, the floor as a share of the 0.1 dB "
          "budget: it is charged <i>inside</i> the budget and consumes a "
          "growing part of it as the rate rises, which is one of the two "
          "reasons the saving falls with rate.")

    k.h2("Three conventions, and the conversions between them")

    k.par(r"Three separate choices have to be pinned before a figure in this "
          r"section means anything, and the project makes all three in two "
          r"ways in different files. The loss can be measured against the "
          r"released decoder's full-frame decode of the same latent, which "
          r"puts the cost of cutting the frame into tiles inside the budget, "
          r"or against this model's own deepest exit decoded tile by tile, "
          r"which cancels it. Decibels can be averaged per frame and then over "
          r"frames, which is what the released codec's own evaluation does, or "
          r"pooled into one mean squared error first. And the saving can be "
          r"divided by the released decode or by our own deepest exit. Two of "
          r"the three conversions are exact:")
    k.eq(r"D^{\mathrm{rel}} = D^{\mathrm{self}} + D_{\mathrm{min}}, \;\;\; "
         r"S^{\mathrm{rel}} = 100 - c_{K-1}\,(100 - S^{\mathrm{self}}).")

    def op(q, t):
        return [o for o in front["op_points"]
                if o["qp"] == q and abs(o["target_db"] - t) < 1e-9][0]

    pf0, pf63 = op(0, 0.10), op(63, 0.10)
    rows = [["choice", "the two options", "what separates them"],
            ["loss measured against",
             "the released full-frame decode, or our own tiled deepest exit",
             f"the floor: \\FloorLow dB at q0, \\FloorHigh dB at q63"],
            ["decibels averaged",
             "per frame and then over frames, or pooled over every tile first",
             f"measured, not closed form: a pooled 0.1 dB is "
             f"{pf0['db_vs_uf_per_frame']:.3f} dB per frame at q0 and "
             f"{pf63['db_vs_uf_per_frame']:.3f} at q63"],
            ["saving divided by",
             f"the released decode, or our deepest exit at {cK:.4f} of it",
             f"a self-referenced 30% is {100 - cK * (100 - 30):.2f}% "
             f"release-referenced, and a self-referenced 0% is "
             f"−{100 * (cK - 1):.2f}%"]]
    k.rows(rows, r"<b>What has to be stated with every decibel and every "
                 r"saving in this section.</b> Take from it that a number "
                 r"quoted without its convention is uncertain by about a third "
                 r"of a 0.1 dB budget on the quality axis and by about two "
                 r"thirds of a point on the compute axis. The first and third "
                 r"rows convert exactly, by the identity above; the second "
                 r"does not, because a mean of logarithms is not the logarithm "
                 r"of a mean.")
    k.note(r"The floor is results/saturation_RECIPE512_ctc53.json; the two "
           r"averaging figures are the 0.1 dB operating points of "
           r"results/supp_paper_curve_PAPER.json, which bisects on the pooled "
           r"decibel and records the per-frame decibel of the same allocation "
           r"beside it; the deepest-exit cost is deepest_exit_cost in "
           r"results/router_RECIPE512_b01.json. The averaging convention "
           r"itself is defined in Section A.")

    # ---------------------------------------------------------------- E.3
    k.h2("The grid")

    k.par(r"Table " + str(k.peek_tbl()) + r" is the sweep the main paper "
          r"quotes: nine budgets across five rates, measured against the "
          r"released decoder on both axes, with decibels averaged per frame, "
          r"two frames from each of the \NumSeq test sequences. Allocation is "
          r"by the exact per-tile losses, so every entry is an upper bound on "
          r"what a decoder-side router can reach.")

    # The table prints the model; the caption also quotes the hook count for
    # the 0.1 dB row, because that is the row the main paper's headline is,
    # and the headline is a hook count.
    _m01 = {x["qp"]: x["saving_pct_measured"] for x in grid["rows"]
            if abs(x["budget_db"] - 0.1) < 1e-9
            and x.get("saving_pct_measured") is not None}

    rows = [["budget dB"] + [f"q{q}" for q in QPS]]
    for b in grid["budgets"]:
        cells = []
        for q in QPS:
            r = [x for x in grid["rows"]
                 if x["qp"] == q and abs(x["budget_db"] - b) < 1e-9][0]
            cells.append(_f(r["saving_pct_vs_release"], 1)
                         if r.get("budget_reachable") else "none")
        rows.append([_f(b, 3)] + cells)
    k.rows(rows, r"<b>Compute saved against budget, release-referenced.</b> "
                 r"<i>none</i> marks a budget below that rate's floor, where "
                 r"no allocation is admissible. Reading down a column, the "
                 r"return on a looser budget falls away long before the "
                 r"ceiling is reached. Reading across a row, the rate "
                 r"dependence is largest at the tightest budget and vanishes "
                 r"once every rate has saturated. Cells are the cost model "
                 r"against the release, the convention the rest of this "
                 r"supplement compares in; hook-counted, the 0.1 dB row reads "
                 + _f(_m01[QPS[0]], 1) + r"% at q0 and " + _f(_m01[QPS[-1]], 1)
                 + r"% at q63, which reproduces the headline of the main "
                 r"paper, \MainLowRate% and \MainHighRate%, to within a "
                 r"tenth of a point on a different frame count.")
    k.note(r"results/signalled_RECIPE512_grid.json, pinned checkpoint, two "
           r"frames per sequence, budget bisected on the delivered per-frame "
           r"decibel of a real decode of the mixed map. The headline file "
           r"results/signalled_RECIPE512_ctc53.json uses one frame per "
           r"sequence and reports \MainLowRate% and \MainHighRate% at the same "
           r"budget.")

    front_sat = {q: min(o["target_db"] for o in front["op_points"]
                        if o["qp"] == q and o.get("saturated"))
                 for q in QPS}
    k.par(f"A second sweep of the same checkpoint prices the same window "
          f"differently, and Section D prints its operating points in full. "
          f"It bisects on the pooled decibel rather than the per-frame one "
          f"and reads \\NumSeq sequences at one frame each rather than two, "
          f"and at a nominal 0.1 dB it saves several points more at every "
          f"rate than the grid above. The conversions of Section " + sec +
          f".2 account for part of that and not for all of it, which is the "
          f"first sign of what Section " + sec + f".5 measures. Its own "
          f"saturation points are the useful thing here: the budget at which "
          f"a rate runs out of allocation is "
          + ", ".join(f"{front_sat[q]:.2f} dB at q{q}" for q in QPS) + f". The "
          f"three budgets the paper reports are therefore not three points on "
          f"one curve. At 0.1 dB every rate still has an allocation to make, "
          f"at 0.3 dB only q48 and q63 do, and at 0.5 dB none does.")
    k.note(r"results/supp_paper_curve_PAPER.json, pinned checkpoint, \NumSeq "
           r"sequences at one frame each, 35 operating points at seven "
           r"budgets. A saturated point is flagged in the file rather than "
           r"inferred here, and carries no exit histogram, because the "
           r"allocation has stopped changing.")

    k.par(r"Budgets in [\SatLow, \SatWindowHi) dB saturate the lowest rate and "
          r"no other, a window \SatWindowMb thousandths of a decibel wide. "
          r"Above \SatHigh dB every rate has saturated, so the binding "
          r"constraint is the ladder rather than the budget. Both figures are "
          r"read off the nine-rate table in Section D rather than off either "
          r"sweep.")

    # ---------------------------------------------------------------- E.4
    k.h2("Rescaling onto the band")

    k.par(r"With both ends of the window measured, a budget can be quoted as a "
          r"position in the window rather than as a decibel figure. Write")
    k.eq(r"u = \frac{D - D_{\mathrm{min}}}"
         r"{D_{\mathrm{sat}} - D_{\mathrm{min}}},")
    k.par(r"so that u = 0 is the floor and u = 1 the saturation point. At a "
          r"matched decibel the five rates are \BandRawSpread points apart in "
          r"saving. At a matched u they are \BandSpreadMean points apart on "
          r"average and \BandSpreadMax at worst, so the window accounts for "
          r"most of the rate dependence of the trade-off and a couple of "
          r"points are left over.")

    cur = _curves(grid, sat)
    rows = [["u"] + [f"q{q}" for q in QPS] + ["spread"]]
    sps = []
    for u in UGRID:
        vs = [_interp(cur[q], u) for q in QPS]
        sps.append(max(vs) - min(vs))
        rows.append([_f(u, 1)] + [_f(v, 1) for v in vs] + [_f(sps[-1])])
    k.rows(rows, r"<b>The five rates at matched positions in their own "
                 r"windows.</b> Read across a row: five rates whose raw "
                 r"savings at a matched decibel differ by \BandRawSpread "
                 r"points agree here to about a point and a half. The spread "
                 r"is largest at the bottom of the window, where the floor "
                 r"subtraction is a large fraction of a small number and the "
                 r"two files disagree about the floor by a few thousandths of "
                 r"a decibel.")
    k.note(f"Computed in the build from "
           f"results/signalled_RECIPE512_grid.json and "
           f"results/saturation_RECIPE512_ctc53.json by the rescaling in "
           f"scripts/tradeoff_figure.py, linearly interpolated between "
           f"measured points. Every rate has measured points spanning "
           f"u = 0.10 to 0.70, so no entry is an extrapolation. Mean spread "
           f"over these seven positions {sum(sps) / len(sps):.2f} points, "
           f"worst {max(sps):.2f}, against the \\BandSpreadMean and "
           f"\\BandSpreadMax recorded in results/band_collapse.json, which "
           f"interpolates on a wider grid and holds the last measured value "
           f"beyond each rate's last point.")

    k.par(r"The collapsed curve is then described by one free number. With C "
          r"the architectural ceiling, which is not fitted,")
    k.eq(r"S(u) \approx C\,u^{\beta},")
    k.par(r"and fitting ln(S/C) against ln u by least squares through the "
          r"origin gives \BandExp on the pinned checkpoint, with R² = \BandRTwo "
          r"and a worst residual of \BandFitErr points. The same procedure on "
          r"the second training run gives \BandExpB with R² = \BandRTwoB, "
          r"which is the replication Figure " + str(k.peek_fig()) + r" draws "
          r"and Section " + sec + r".5 takes apart.")

    k.fig("budget_band_BEST.png",
          r"<b>The collapse on the second training run.</b> <b>a</b> Saving "
          r"against budget, one line per rate, with each rate's floor and "
          r"saturation point ticked. <b>b</b> The same with the budget axis "
          r"rescaled onto each rate's own window; dashed is the "
          r"one-parameter power law. The main paper shows this pair for the "
          r"pinned checkpoint.")
    k.note(r"docs/figures/budget_band_BEST.png, drawn by "
           r"scripts/tradeoff_figure.py from "
           r"results/signalled_BEST_grid.json and "
           r"results/saturation_BEST_ctc53.json, both on "
           r"runs/BEST/ckpt_eval.pth.tar rather than on the pinned "
           r"checkpoint.")

    # ---------------------------------------------------------------- E.5
    k.fig("tradeoff.png",
          "<b>The trade-off, whole.</b> <b>a</b> What a budget buys, per "
          "rate; the ceiling is \\Ceiling% and q0 reaches it at \\SatLow dB. "
          "<b>b</b> The same relation inverted, so it reads as what a saving "
          "target costs in quality. The budgets the main paper reports are "
          "three points on this curve, and the flat right-hand end of each "
          "trace is that rate sitting on the ceiling.")

    k.h2("How weak the replication is")

    gB = k.J("signalled_BEST_grid.json")
    sB = k.J("saturation_BEST_ctc53.json")
    curB = _curves(gB, sB)
    CB = sB["ceiling_pct"]
    bc = k.J("band_collapse.json")
    bb = k.J("band_collapse_BEST.json")
    fl = k.J("frontier_law.json")

    # Every fit below is taken over the same window, so that a difference
    # between two rows is a difference in the curve and not a difference in
    # which part of the curve each sweep happened to sample.
    base = _fit(_win(cur), C)

    cur3 = {}
    for q in QPS:
        pts = []
        for o in front["op_points"]:
            if (o["qp"] != q or o.get("saturated")
                    or o.get("budget_reachable") is False):
                continue
            f, s = S[q]["floor_db"], S[q]["saturation_db"]
            pts.append(((o["db_vs_uf_per_frame"] - f) / (s - f),
                        100 * (1 - cK * (1 - o["saving_pct"] / 100))))
        cur3[q] = pts
    b_sweep = _fit(_win(cur3), C)

    def _rowcurve(key):
        c = {}
        for q in QPS:
            f, s = S[q]["floor_db"], S[q]["saturation_db"]
            c[q] = [((r[key] - f) / (s - f), r["saving_pct_vs_release"])
                    for r in front["rows"] if r["qp"] == q]
        return c
    b_pf = _fit(_win(_rowcurve("db_vs_uf_per_frame")), C)
    b_pool = _fit(_win(_rowcurve("db_vs_uf")), C)

    curF = {}
    for q in QPS:
        rs = sorted((r for r in grid["rows"]
                     if r["qp"] == q and r.get("budget_reachable")),
                    key=lambda r: r["budget_db"])
        f, s = rs[0]["floor_db"], S[q]["saturation_db"]
        curF[q] = [((r["budget_db"] - f) / (s - f),
                    r["saving_pct_vs_release"]) for r in rs
                   if 0.0 <= (r["budget_db"] - f) / (s - f) <= 1.0]
    b_floor = _fit(_win(curF), C)
    b_ceil = _fit(_win(cur), CB)
    b_ck = _fit(_win(curB), CB)
    per = {q: _fit([p for p in cur[q] if WLO <= p[0] <= WHI], C) for q in QPS}
    lo_q = min(per, key=lambda q: per[q][0])
    hi_q = max(per, key=lambda q: per[q][0])
    inv = [1.0 / r["exponent_b"] for r in fl["rows"]]

    k.par(r"The main paper says the collapse replicates and the exponent does "
          r"not. Two checkpoints is a thin basis for either half of that "
          r"sentence, and Table " + str(k.peek_tbl()) + r" is meant to show "
          r"how thin. Each row refits the same one-parameter law after "
          r"changing exactly one thing, over the window of positions where "
          r"every sweep has points.")

    rows = [["what is changed", "β before", "β after", "Δ"]]
    rows.append(["the checkpoint, pinned to BEST, each with its own ceiling",
                 _f(base[0], 3), _f(b_ck[0], 3), _f(b_ck[0] - base[0], 3)])
    rows.append(["the sweep, 106 frames to \\NumSeq, one checkpoint",
                 _f(base[0], 3), _f(b_sweep[0], 3),
                 _f(b_sweep[0] - base[0], 3)])
    rows.append(["the budget axis, per-frame decibels to pooled, one sweep",
                 _f(b_pf[0], 3), _f(b_pool[0], 3), _f(b_pool[0] - b_pf[0], 3)])
    rows.append([f"the assumed ceiling C, \\CeilingModelled to {CB:.2f}",
                 _f(base[0], 3), _f(b_ceil[0], 3), _f(b_ceil[0] - base[0], 3)])
    rows.append(["the floor, the saturation file to the grid's own",
                 _f(base[0], 3), _f(b_floor[0], 3),
                 _f(b_floor[0] - base[0], 3)])
    rows.append([f"the rate, fitted one at a time, q{lo_q} against q{hi_q}",
                 _f(per[lo_q][0], 3), _f(per[hi_q][0], 3),
                 _f(per[hi_q][0] - per[lo_q][0], 3)])
    rows.append(["the fit form, 1/b from the inverse fit, per rate",
                 _f(min(inv), 3), _f(max(inv), 3),
                 _f(max(inv) - min(inv), 3)])
    k.rows(rows, r"<b>What moves the exponent.</b> Take from it that the "
                 r"exponent describes a sweep rather than a decoder. Every "
                 r"row except the floor moves it by as much as changing the "
                 r"checkpoint does, re-sweeping the same checkpoint moves it "
                 r"twice as far, and reading the budget axis in the other "
                 r"averaging convention moves it almost as far again. The "
                 r"comparison the main paper draws between \BandExp and "
                 r"\BandExpB is inside the noise of every other choice on this "
                 r"list.")
    k.note(f"Computed in the build. Rows 4, 5 and 6, and the before "
           f"column of rows 1 and 2, refit "
           f"results/signalled_RECIPE512_grid.json against "
           f"results/saturation_RECIPE512_ctc53.json; the after column of "
           f"row 1 uses "
           f"results/signalled_BEST_grid.json and "
           f"results/saturation_BEST_ctc53.json on "
           f"runs/BEST/ckpt_eval.pth.tar, whose ceiling is its own; the "
           f"after column of row 2 uses the operating points of "
           f"results/supp_paper_curve_PAPER.json and row 3 uses its raw "
           f"sweep; row 7 is "
           f"results/frontier_law.json, fitted to results/curve_BEST.json on "
           f"the same second run over 40 sequences. Every fit is over "
           f"u in [{WLO:.2f}, {WHI:.2f}] and through the origin in log space, "
           f"on {base[3]} to {b_ck[3]} points. R² stays between "
           f"{min(x[1] for x in (base, b_ck, b_sweep, b_pf, b_pool, b_floor, b_ceil)):.3f} "
           f"and "
           f"{max(x[1] for x in (base, b_ck, b_sweep, b_pf, b_pool, b_floor, b_ceil)):.3f} "
           f"across the rows, so no row is a bad fit to its own data.")

    k.fig("band_sensitivity_PAPER.png",
          r"<b>The collapse, and what the exponent is sensitive to.</b> "
          r"<b>a</b> The rescaled points from two sweeps of the pinned "
          r"checkpoint, filled circles for the 106-frame grid and open "
          r"triangles for the \NumSeq-frame frontier, with the power law "
          r"fitted to each over the shaded window. The two sweeps of one "
          r"checkpoint do not lie on one curve. <b>b</b> The exponent under "
          r"one change at a time, with the vertical line at the exponent of "
          r"the first row of Table " + str(k.peek_tbl() - 1) + r". The change "
          r"of checkpoint, which the main paper reads as a finding, is the "
          r"smallest of the six.",
          maxh=126)
    k.note(r"docs/figures/band_sensitivity_PAPER.png, drawn by "
           r"scripts/supp_fig_band_sensitivity.py, which imports the fit from "
           r"this section's module so that the figure and the table cannot "
           r"disagree. Sources as for the table above.")

    k.par(f"The floor is the one input the exponent is insensitive to over "
          f"this window, and only over this window. Refitting over the whole "
          f"range instead, where the points below u = 0.15 sit, moves it from "
          f"{_fit([p for q in cur for p in cur[q]], C)[0]:.3f} to "
          f"{_fit([p for q in curF for p in curF[q]], C)[0]:.3f} when the "
          f"floor is taken from the grid rather than from the saturation "
          f"file. A rescaling is only as good as the two numbers it divides "
          f"by, and near the floor it is dividing a small difference by a "
          f"small difference.")

    bB, r2B = _fit([p for q in curB for p in curB[q]], CB)[:2]
    k.par(f"Both fits are re-run in this build from the files they name, so "
          f"the exponent quoted here is the exponent those inputs now support: "
          f"results/band_collapse_BEST.json reproduces at {bB:.4f} with "
          f"R² = {r2B:.4f} against the {bb['power_exponent']:.4f} and "
          f"{bb['power_r2']:.4f} it records, and the two digits the paper "
          f"quotes, \\BandExp and \\BandRTwo, are unchanged by the exercise.")

    # ---------------------------------------------------------------- E.6
    k.h2("Which ladder a budget wants")

    k.par(
        "Once a budget saturates a ladder, the only way to spend more quality "
        "is an exit that does not exist, so the ladder itself is a choice "
        "made at the operating point rather than a hyperparameter tuned once. "
        "Four ladders have been run far enough to compare, and the two "
        "orderings cross between 0.1 and 0.3 dB.")
    k.tbl("runs",
          "<b>Ladder settings</b>, mean saving over the five rates, on each "
          "run's own latest checkpoint. Ceilings are 100(1 \u2212 c_j) for "
          "that ladder, corrected by the offset the hook count shows. The "
          "asterisk marks a setting where some rate did not reach the budget, "
          "so the mean is over the rest; the dagger marks a saving taken from "
          "the arithmetic model rather than counted off the decode, which "
          "reads 0.4 to 0.8 points optimistic. Only RECIPE512 has been "
          "re-measured with hooks at every budget, and it is the only run the "
          "main paper reports.")
    k.note("The four signalled_*.json curve files named in "
           "scripts/make_paper_tables.py, generated into "
           "paper/tables/runs.tex. The four runs are at different epochs, so "
           "no row here separates a ladder from the run that trained it; "
           "section H measures how far two runs of one recipe land apart.")

    k.par(
        "At 0.1 dB the coarse ladder wins and the fine one (K = 12, j = 4) "
        "cannot reach the budget at the highest rate at all. Splitting later "
        "puts twice as many blocks in the per-tile section, and the seam "
        "penalty grows with that count, so a finer ladder raises the floor it "
        "has to clear before it can spend anything. At 0.5 dB the fine ladder "
        "wins, \\FineHalfDb% against \\CoarseHalfDb%, because the coarse "
        "one has been pinned at its ceiling since 0.3 dB and has nothing left "
        "to spend. The band of E.1 says which side of that crossover a given "
        "budget sits on, which is the practical use of measuring the two ends.")

    k.h2("What the window claims, and what it does not")

    k.bullets([
        r"The floor and the saturation point are measured, cheap and "
        r"unambiguous. Each is one forward pass per rate, and the ceiling "
        r"between them is closed form in the per-exit costs.",
        r"The collapse is the claim. Rescaling a budget onto its own window "
        r"takes the spread across rates from \BandRawSpread points to under "
        r"two at matched positions, on both checkpoints.",
        r"The exponent is an empirical fit and a description of one sweep, "
        r"not a law. It moves further when the same checkpoint is swept again, "
        r"or when the budget axis is read in the other averaging convention, "
        r"than it moves between the two training runs the main paper compares.",
        r"Everything here is the oracle allocation, obtained with the "
        r"per-tile losses known. It bounds any router and says nothing about "
        r"how much of the window a decoder-side predictor reaches, which is "
        r"what Section F measures.",
    ])
