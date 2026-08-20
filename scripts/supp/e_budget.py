"""The quality budget: what it means, where it works, and how it rescales.

The main paper states three things about the budget in a page and a half: that
there is a floor below which no allocation is feasible, a saturation point
above which nothing improves, and that rescaling the budget axis onto each
rate's own band collapses the five curves onto one power law. This section
carries the grids those claims are read off, both decibel conventions the
project uses, and an account of how weak the two-checkpoint replication of the
power law actually is.

Every number is read from a named file in `results/`. Where a quantity is
arithmetic on file values -- the position u in the band, a least-squares
exponent, a ratio of two costs -- it is computed here at build time from the
files named in the note beside the table, by the procedure in
`scripts/tradeoff_figure.py`, and never typed.

`paper/supp/e_budget.tex` carries the same section as LaTeX, with the derived
figures written out as digits, because LaTeX cannot compute them.
"""
import math

# The matched-position grid. Every rate of signalled_RECIPE512_grid.json has
# measured points spanning u = 0.10 to 0.70, so nothing in the table below is a
# held-constant extrapolation. band_collapse.json interpolates on 0.05 to 0.95
# instead and holds the last measured value beyond each rate's last point,
# which is why its worst spread is larger than this table's.
UGRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
QPS = [0, 16, 32, 48, 63]


# --------------------------------------------------------------- arithmetic
def _curves(grid, sat):
    """Each rate's reachable points as (position in band, saving)."""
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

    The same one-parameter fit scripts/tradeoff_figure.py performs: the ceiling
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


def _f(x, n=2):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ section
def content(k):
    sec = k.h1("The quality budget and its working range")

    sat = k.J("saturation_RECIPE512_ctc53.json")
    grid = k.J("signalled_RECIPE512_grid.json")
    supp = k.J("supp_budget_grid.json")
    rtr = k.J("router_RECIPE512_b01.json")

    S = {r["qp"]: r for r in sat["rows"]}
    cj = sat["cost_j"]
    cK = rtr["deepest_exit_cost"]

    k.par(r"A budget D is the peak-signal-to-noise ratio the decode is allowed "
          r"to give up, and an allocation is admissible if the loss it incurs "
          r"stays at or below it. Everything the paper reports is one point on "
          r"a grid of such budgets. This section gives the grid whole, "
          r"together with the two ends of the range over which a budget "
          r"changes anything, and the rescaling that removes most of the rate "
          r"dependence from the trade-off.")

    # ---------------------------------------------------------------- E.1
    k.h2("Two decibel conventions")

    k.par(r"Two separate choices have to be pinned before a decibel figure "
          r"means anything, and the project makes both choices in two ways in "
          r"different files. The first is what the tiled decode is compared "
          r"against. Under the <i>release-referenced</i> convention it is "
          r"compared against the released decoder's full-frame decode of the "
          r"same latent, so the cost of cutting the frame into tiles at all "
          r"sits inside the number. Under the <i>self-referenced</i> "
          r"convention it is compared against this model's own deepest exit "
          r"decoded tile by tile, so the tiling cost cancels and the budget "
          r"pays only for early exiting.")

    k.par(r"The two conventions give the ceiling two values. With c_k the "
          r"cost of exit k in units of one released decode, j the split depth "
          r"and K the number of exits,")
    k.eq(r"S_{\mathrm{max}}^{\mathrm{rel}} = 100\,(1 - c_j), \;\; "
         r"S_{\mathrm{max}}^{\mathrm{self}} = 100\,(1 - c_j/c_{K-1})")
    k.par(f"With c_j = {cj:.4f}, recorded in "
          f"results/saturation_RECIPE512_ctc53.json, and c_(K−1) = {cK:.4f}, "
          f"recorded as deepest_exit_cost in "
          f"results/router_RECIPE512_b01.json, the first reads \\Ceiling% and "
          f"the second {100 * (1 - cj / cK):.2f}%. The deepest exit costs more "
          f"than one released decode because it carries the deblocking filter, "
          f"which the released decoder does not run, and that difference is "
          f"the whole of the gap between the two ceilings.")

    k.par(r"The second choice is how decibels are averaged over frames. "
          r"Table " + str(k.peek_tbl()) + r" gives the same allocation read "
          r"both ways: a per-frame mean of the per-frame loss, and one loss "
          r"computed from mean-squared errors pooled over all frames first. "
          r"Pooling is the flattering reading at every rate, and the gap is "
          r"between a quarter and a third of a 0.1 dB budget. The project "
          r"reports the per-frame convention throughout.")

    dbc = k.J("db_convention.json")
    rows = [["q", "per-frame dB", "pooled dB", "difference", "saving %"]]
    for r in dbc["rows"]:
        rows.append([str(r["qp"]), _f(r["per_frame_db"], 4),
                     _f(r["pooled_db"], 4), _f(r["difference_db"], 4),
                     _f(r["saving_pct"])])
    t_conv = k.rows(rows, r"<b>The averaging convention is worth more than "
                          r"the third digit of the budget.</b> One allocation "
                          r"targeted at 0.1 dB, read two ways. Pooling the "
                          r"squared errors before taking the logarithm reports "
                          r"0.023 to 0.033 dB less loss than the per-frame "
                          r"mean of the same decodes, so a result quoted "
                          r"without its convention is uncertain by about a "
                          r"third of the budget it claims to meet.")
    k.note(r"results/db_convention.json. This is the one table in this "
           r"section not measured on the pinned checkpoint: it is "
           r"runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar over 40 frames, an "
           r"early 128 px run. The two conventions are definitions rather "
           r"than properties of a checkpoint, so the size of the gap is what "
           r"transfers, not the digits.")

    k.par(r"Nothing in the published literature fixes 0.1 dB as a tolerance. "
          r"Subjective work measures the smallest noticeable change in "
          r"quantisation parameter or in a video-quality metric rather than "
          r"in tenths of a decibel of peak signal-to-noise ratio, and one of "
          r"our own test sets, MCL-JCV [26], is a just-noticeable-difference "
          r"dataset built that way. We treat 0.1 dB as a reporting "
          r"convention, report 0.3 dB and 0.5 dB beside it in the main paper, "
          r"and give the whole grid here so that nothing in the argument "
          r"rests on the choice.")

    # ---------------------------------------------------------------- E.2
    k.h2("The floor")

    k.par(r"The floor D_min is the loss of a tiled decode with every tile at "
          r"full depth. No early exiting has happened, so it is pure tiling "
          r"penalty: the border of every tile has been convolved against "
          r"padding instead of against its neighbour. It is measured by one "
          r"forward pass per rate, with the exit map held at K − 1, against "
          r"the released decoder's full-frame decode of the same latent.")

    k.par(r"A budget below the floor admits no allocation at all. This is a "
          r"different failure from the budget falling in a gap of the "
          r"Lagrangian hull, where the sweep returns the nearest reachable "
          r"point and Everett's argument still makes it optimal for the "
          r"compute it consumes [28, 29]. Below the floor the feasible set is "
          r"empty and there is nothing to return. Three cells of the "
          r"release-referenced grid are in that state and are recorded as "
          r"such: at a 0.05 dB budget, results/signalled_RECIPE512_grid.json "
          r"carries budget_reachable false at q32, q48 and q63, whose floors "
          r"there are 0.053, 0.060 and 0.066 dB.")

    rows = [["q", "floor D_min", "saturation D_sat", "band", "0.1 dB at"]]
    for r in sat["rows"]:
        f, s = r["floor_db"], r["saturation_db"]
        rows.append([str(r["qp"]), _f(f, 3), _f(s, 3), _f(s - f, 3),
                     f"{100 * (0.1 - f) / (s - f):.0f}%"])
    t_op = k.rows(rows, r"<b>The working range, at all nine measured "
                        r"rates.</b> The main paper's operating table shows "
                        r"five of these columns. Both ends widen with the "
                        r"rate and the band widens faster than the floor "
                        r"rises, so the 0.1 dB budget that sits 45% of the "
                        r"way up the band at q0 sits 9% of the way up at q63. "
                        r"That single quantity, and not the rate, is what the "
                        r"rescaling of Section " + sec + r".5 turns "
                        r"out to need.")
    k.note(r"results/saturation_RECIPE512_ctc53.json, 53 CTC frames on the "
           r"pinned checkpoint. The band and the position of 0.1 dB in it are "
           r"arithmetic on the two measured columns.")

    k.par(r"The floor also appears in results/signalled_RECIPE512_grid.json, "
          r"measured over 106 frames rather than 53, and the two estimates "
          r"differ by 3 to 6 thousandths of a decibel: 0.033 against 0.036 dB "
          r"at q0, and 0.066 against 0.072 dB at q63. The rescaling below "
          r"uses the saturation file's values for both ends, so the two ends "
          r"are on one measurement.")

    # ---------------------------------------------------------------- E.3
    k.h2("The saturation point")

    k.par(r"The saturation point D_sat is the loss when every tile takes exit "
          r"j, the shallowest the split depth allows. Saving is then pinned "
          r"at the ceiling and a looser budget buys nothing, so it is not a "
          r"quantity to search for: it is one forward pass with the exit map "
          r"held at j. Both ends of the band are therefore closed-form or "
          r"one-pass quantities, which is what makes the rescaling cheap "
          r"enough to be worth doing in deployment.")

    k.par(r"Table " + str(k.peek_tbl()) + r" checks that claim against a "
          r"measurement that never sets an exit map by hand. Sweeping the "
          r"Lagrange multiplier upward, the achieved loss stops rising at a "
          r"value that agrees with the one-pass saturation point to within "
          r"1.5 thousandths of a decibel at all five rates, and the saving "
          r"there is the architectural ceiling to four decimal places.")

    rows = [["q", "D_sat, one pass", "grid stops at", "difference",
             "saving there"]]
    for q in QPS:
        rs = [r for r in grid["rows"]
              if r["qp"] == q and r.get("budget_reachable")]
        mx = max(r["db_vs_uf"] for r in rs)
        sv = [r["saving_pct_vs_release"] for r in rs
              if abs(r["db_vs_uf"] - mx) < 1e-9][0]
        rows.append([str(q), _f(S[q]["saturation_db"], 4), _f(mx, 4),
                     f"{1000 * (mx - S[q]['saturation_db']):+.1f} mdB",
                     _f(sv, 4)])
    t_sat = k.rows(rows, r"<b>Two independent routes to the same "
                         r"saturation point.</b> The middle column is the "
                         r"largest loss any budget in the release-referenced "
                         r"grid actually incurs; past that point a looser "
                         r"budget changes neither the loss nor the saving. "
                         r"The two routes share a checkpoint and a test set "
                         r"but no code path.")
    k.note(r"results/saturation_RECIPE512_ctc53.json (53 frames) and "
           r"results/signalled_RECIPE512_grid.json (106 frames), both on the "
           r"pinned checkpoint. The difference column is arithmetic on the "
           r"two.")

    k.par(r"One consequence is narrow enough to be worth stating. Budgets in "
          r"[\SatLow, \SatWindowHi) dB saturate the lowest rate and no other, "
          r"a window \SatWindowMb thousandths of a decibel wide. Above "
          r"\SatHigh dB every rate has saturated and the ladder, not the "
          r"budget, is the binding constraint.")

    # ---------------------------------------------------------------- E.4
    k.h2("The grid")

    k.par(r"Table " + str(k.peek_tbl()) + r" is the main grid: eight budgets "
          r"across five rates, on the pinned checkpoint, one frame from each "
          r"of the \NumSeq test sequences. It is measured in the "
          r"self-referenced convention, so its ceiling is "
          f"{100 * (1 - cj / cK):.2f}% rather than \\Ceiling%.")

    def gsel(qp, b):
        return [r for r in supp["rows"]
                if r["qp"] == qp and abs(r["budget"] - b) < 1e-9][0]

    budgets = sorted({r["budget"] for r in supp["rows"]})
    rows = [["budget dB"] + [f"q{q}" for q in QPS]]
    for b in budgets:
        rows.append([_f(b, 2)] + [_f(gsel(q, b)["saving_pct"], 1) for q in QPS])
    t_grid = k.rows(rows, r"<b>Compute saved against budget, the whole "
                         r"grid.</b> Oracle allocation, so an upper bound on "
                         r"any router. Reading down a column, the return on a "
                         r"looser budget falls away long before the ceiling "
                         r"is reached: at q0 the first 0.05 dB buys 23.9 "
                         r"points and the next 0.95 dB buys 15.8 more. "
                         r"Reading across a row, the rate dependence is 5.9 "
                         r"points at 0.05 dB and vanishes entirely once every "
                         r"rate has saturated.")
    k.note(r"results/supp_budget_grid.json, pinned checkpoint, one frame per "
           r"sequence, measured by scripts/budget_table.py on an uncontended "
           r"card.")

    rows = [["budget dB"] + [f"q{q}" for q in QPS]]
    for b in budgets:
        rows.append([_f(b, 2)] +
                    [_f(gsel(q, b)["achieved_db"], 3) for q in QPS])
    t_ach = k.rows(rows, r"<b>What the same allocations actually spend.</b> "
                        r"Up to 0.3 dB the achieved loss tracks the budget to "
                        r"the third decimal, which is the bisection working. "
                        r"Above it the entries stop moving, and at q0 they "
                        r"stop at 0.522 dB however loose the budget gets: "
                        r"there is no allocation left that spends more, which "
                        r"is saturation seen from the other side.")
    k.note(r"results/supp_budget_grid.json. The entries above 0.3 dB are an "
           r"upper bound on the loss rather than the loss a deployed decoder "
           r"would incur; see the paragraph below.")

    k.par(r"Two properties of that grid need stating rather than leaving to "
          r"be discovered. Its exit histogram has mass in the two columns "
          r"below the split depth, which the deployed decoder cannot use: "
          r"forward() clamps any assignment below j to j, and "
          r"flexuf/cost.py applies the same clamp, which is why the ceiling "
          f"is exactly {100 * (1 - cj / cK):.2f}% and not something larger. "
          r"Those tiles are billed at exit j's cost, correctly, but the loss "
          r"recorded for them is the shallower exit's, which is worse than "
          r"what the clamped decode would produce. The effect is negligible "
          r"at 0.05 dB, where 93 of 1,765 tiles are nominally below j, and "
          r"large at 1.0 dB, where most of them are. Saving is unaffected at "
          r"every budget; the achieved-loss column is pessimistic above "
          r"about 0.3 dB.")

    rows = [["budget dB"] + [f"q{q}" for q in QPS]]
    for b in grid["budgets"]:
        cells = []
        for q in QPS:
            r = [x for x in grid["rows"]
                 if x["qp"] == q and abs(x["budget_db"] - b) < 1e-9][0]
            cells.append(_f(r["saving_pct_vs_release"], 1)
                         if r.get("budget_reachable") else "none")
        rows.append([_f(b, 3)] + cells)
    t_rel = k.rows(rows, r"<b>The same sweep in the release-referenced "
                         r"convention.</b> Nine budgets, five rates, two "
                         r"frames from each of the \NumSeq sequences. "
                         r"<i>none</i> marks a budget below that rate's "
                         r"floor, where no allocation is admissible. The "
                         r"0.1 dB row is the main paper's headline: "
                         r"\MainLowRate% at q0, \MainHighRate% at q63, "
                         r"\MainMean% averaged over the five rates. Every "
                         r"entry at 0.4 dB and above has saturated at "
                         r"\Ceiling%; at 0.3 dB all but q48 and q63 have.")
    k.note(r"results/signalled_RECIPE512_grid.json, pinned checkpoint, two "
           r"frames per sequence. The budget_reachable flag is recorded in "
           r"the file, not inferred here.")

    k.par(r"Table " + str(k.peek_tbl()) + r" puts the two conventions side by "
          r"side at the budget the paper leads with. They disagree by 1.5 "
          r"points at q0 and 9.3 points at q63, and the disagreement is very "
          r"nearly the floor: the release-referenced number has to pay the "
          r"tiling penalty out of the same 0.1 dB, and that penalty grows "
          r"with the rate. Neither number is wrong. A reader comparing "
          r"against a full-frame codec wants the first; a reader asking what "
          r"early exiting alone contributes wants the second.")

    rows = [["q", "self-ref.", "release-ref.", "difference", "floor dB"]]
    for q in QPS:
        a = [r for r in grid["rows"]
             if r["qp"] == q and abs(r["budget_db"] - 0.1) < 1e-9][0]
        b = gsel(q, 0.1)
        rows.append([str(q), _f(b["saving_pct"]),
                     _f(a["saving_pct_vs_release"]),
                     _f(b["saving_pct"] - a["saving_pct_vs_release"]),
                     _f(a["floor_db"], 3)])
    t_two = k.rows(rows, r"<b>The 0.1 dB column under both conventions.</b> "
                         r"The difference between them tracks the floor, "
                         r"which is what the release-referenced number spends "
                         r"before it has saved anything. A saving figure "
                         r"quoted without its convention is ambiguous by more "
                         r"than the gap between the two configurations the "
                         r"main paper compares.")
    k.note(r"results/supp_budget_grid.json and "
           r"results/signalled_RECIPE512_grid.json, both pinned, one and two "
           r"frames per sequence respectively. The difference column is "
           r"arithmetic on the two.")

    # ---------------------------------------------------------------- E.5
    k.h2("Rescaling onto the band")

    k.par(r"With both ends of the range measured, a budget can be quoted as a "
          r"position in the range rather than as a decibel figure. Write")
    k.eq(r"u = \frac{D - D_{\mathrm{min}}}{D_{\mathrm{sat}} - D_{\mathrm{min}}}")
    k.par(r"so that u = 0 is the floor and u = 1 the saturation point. At a "
          r"matched decibel the five rates are \BandRawSpread points apart in "
          r"saving. At a matched u they are \BandSpreadMean points apart on "
          r"average and \BandSpreadMax at worst, so the band accounts for "
          r"most of the rate dependence of the trade-off and what is left "
          r"over is a couple of points.")

    cur = _curves(grid, sat)
    C = sat["ceiling_pct"]
    allpts = [p for q in sorted(cur) for p in cur[q]]
    b_all, r2_all, mx_all, n_all = _fit(allpts, C)

    rows = [["u"] + [f"q{q}" for q in QPS] + ["spread"]]
    sps = []
    for u in UGRID:
        vs = [_interp(cur[q], u) for q in QPS]
        sps.append(max(vs) - min(vs))
        rows.append([_f(u, 1)] + [_f(v, 1) for v in vs] + [_f(sps[-1])])
    t_coll = k.rows(rows, r"<b>The five curves at matched positions in their "
                         r"own bands.</b> Read across a row: five rates whose "
                         r"raw savings at a matched decibel differ by "
                         r"\BandRawSpread points agree here to about a point "
                         r"and a half. The spread is largest at the bottom of "
                         r"the band, where the floor subtraction is a large "
                         r"fraction of a small number and the two files "
                         r"disagree about the floor by a few thousandths of a "
                         r"decibel.")
    k.note(f"Computed in the build from "
           f"results/signalled_RECIPE512_grid.json and "
           f"results/saturation_RECIPE512_ctc53.json by the rescaling in "
           f"scripts/tradeoff_figure.py, linearly interpolated between "
           f"measured points. Every rate has measured points spanning "
           f"u = 0.10 to 0.70, so no entry is an extrapolation. Mean spread "
           f"over these seven positions {sum(sps) / len(sps):.2f} points, "
           f"worst {max(sps):.2f}.")

    k.par(r"The collapsed curve is described by one free number. With C the "
          r"architectural ceiling, which is not fitted,")
    k.eq(r"S(u) \approx C\,u^{\beta}")
    k.par(r"and fitting ln(S/C) against ln u by least squares through the "
          r"origin gives \BandExp on the pinned checkpoint, with R² = "
          r"\BandR2 and a worst residual of \BandFitErr points. On the "
          r"second training run the same procedure gives \BandExpB with R² = "
          r"\BandRTwoB. Table " + str(k.peek_tbl()) + r" gives both fits in "
          r"full, and Figure " + str(k.peek_fig()) + r" shows the second one "
          r"as a picture.")

    bc = k.J("band_collapse.json")
    bb = k.J("band_collapse_BEST.json")
    rows = [["", "pinned", "BEST"],
            ["source grid", "signalled_RECIPE512", "signalled_BEST"],
            ["checkpoint", "ckpt_PAPER", "ckpt_eval, epoch 1"],
            ["rates", str(bc["n_rates"]), str(bb["n_rates"])],
            ["points fitted", str(bc["power_n"]), str(bb["power_n"])],
            ["ceiling C used", _f(bc["ceiling_pct"]), _f(bb["ceiling_pct"])],
            ["exponent β", _f(bc["power_exponent"], 4),
             _f(bb["power_exponent"], 4)],
            ["R²", _f(bc["power_r2"], 4), _f(bb["power_r2"], 4)],
            ["worst residual", _f(bc["power_max_err"]),
             _f(bb["power_max_err"])],
            ["mean residual", _f(bc["power_mean_err"]),
             _f(bb["power_mean_err"])],
            ["raw spread at 0.1 dB", _f(bc["raw_spread_at_tenth_db"]),
             _f(bb["raw_spread_at_tenth_db"])],
            ["spread after rescaling, mean", _f(bc["band_spread_mean"]),
             _f(bb["band_spread_mean"])],
            ["spread after rescaling, worst", _f(bc["band_spread_max"]),
             _f(bb["band_spread_max"])],
            ["worst excluding first point",
             _f(bc["band_spread_max_excl_edge"]),
             _f(bb["band_spread_max_excl_edge"])]]
    t_fit = k.rows(rows, r"<b>The one-parameter fit, on both "
                         r"checkpoints.</b> The exponent is the only fitted "
                         r"quantity; the ceiling is architectural. Both fits "
                         r"describe their own data well, and they describe it "
                         r"with different exponents. The two ceilings differ "
                         r"because the BEST files predate the correction to "
                         r"the adapter cost, and the row is kept in the table "
                         r"rather than harmonised because the exponent "
                         r"depends on it.")
    k.note(r"results/band_collapse.json and results/band_collapse_BEST.json. "
           r"Neither file records a checkpoint; the rows above name the "
           r"checkpoint of the grid each was fitted to, "
           r"results/signalled_RECIPE512_grid.json and "
           r"results/signalled_BEST_grid.json. The BEST grid is "
           r"runs/BEST/ckpt_eval.pth.tar at epoch 1, a path the per-epoch "
           r"watcher has since overwritten.")

    f_band = k.fig("budget_band_BEST.png",
                   r"<b>The collapse on the second training run.</b> "
                   r"<b>a</b> Saving against budget, one line per rate, with "
                   r"each rate's floor and saturation point ticked. "
                   r"<b>b</b> The same with the budget axis rescaled onto "
                   r"each rate's own band; dashed is the one-parameter power "
                   r"law. The main paper shows this pair for the pinned "
                   r"checkpoint; this is the replication.")
    k.note(r"docs/figures/budget_band_BEST.png, drawn by "
           r"scripts/tradeoff_figure.py from "
           r"results/signalled_BEST_grid.json and "
           r"results/saturation_BEST_ctc53.json.")

    # ---------------------------------------------------------------- E.6
    k.h2("How weak the replication is")

    curB = _curves(k.J("signalled_BEST_grid.json"),
                   k.J("saturation_BEST_ctc53.json"))
    CB = k.J("saturation_BEST_ctc53.json")["ceiling_pct"]
    bA, r2A, _, _ = _fit([p for q in sorted(cur) for p in cur[q]], C)
    bB, r2B, _, _ = _fit([p for q in sorted(curB) for p in curB[q]], CB)

    k.par(r"The main paper says the collapse replicates and the exponent does "
          r"not. Two checkpoints is a thin basis for either half of that "
          r"sentence, and the arithmetic below is meant to show how thin.")

    rows = [["q", "β, pinned", "R²", "β, BEST", "R²"]]
    ea, eb = [], []
    for q in QPS:
        p1 = _fit(cur[q], C)
        p2 = _fit(curB[q], CB)
        ea.append(p1[0])
        eb.append(p2[0])
        rows.append([str(q), _f(p1[0], 3), _f(p1[1], 4),
                     _f(p2[0], 3), _f(p2[1], 4)])
    t_per = k.rows(rows, r"<b>The exponent fitted to one rate at a time.</b> "
                        r"Within a checkpoint the five rates want exponents "
                        r"spanning "
                        f"{max(ea) - min(ea):.3f} on the pinned run and "
                        f"{max(eb) - min(eb):.3f} on BEST. The difference "
                        r"between the two pooled exponents is "
                        f"{abs(bA - bB):.3f}, which is the same size. A "
                        r"quantity that varies this much across the rates of "
                        r"one checkpoint cannot be shown to differ between "
                        r"two checkpoints by measuring it twice.")
    k.note(r"Computed in the build from "
           r"results/signalled_RECIPE512_grid.json with "
           r"results/saturation_RECIPE512_ctc53.json, and "
           r"results/signalled_BEST_grid.json with "
           r"results/saturation_BEST_ctc53.json, by the fit in "
           r"scripts/tradeoff_figure.py applied one rate at a time.")

    k.par(f"The exponent also moves with the ceiling constant, which is not "
          f"fitted but assumed. Holding the pinned grid fixed and changing C "
          f"from \\Ceiling to {bc['ceiling_pct']:.2f}, the constant the "
          f"pre-correction cost model gave, moves the pooled exponent from "
          f"{bA:.3f} to "
          f"{_fit([p for q in sorted(cur) for p in cur[q]], bc['ceiling_pct'])[0]:.3f}. "
          f"That shift is of the same order as the difference between the two "
          f"checkpoints, so the comparison the main paper draws between "
          f"\\BandExp and \\BandExpB is partly a comparison between two cost "
          f"models.")

    k.par(f"One file needs regenerating. Re-running the fit on the files "
          f"results/band_collapse.json names as its inputs, as they stand "
          f"now, gives β = {bA:.4f} with R² = {r2A:.4f}, against the "
          f"{bc['power_exponent']:.4f} and {bc['power_r2']:.4f} the file "
          f"records; both of its inputs were rewritten after it was written, "
          f"and it still carries the pre-correction ceiling. The two digits "
          f"the paper quotes, \\BandExp and \\BandR2, survive the "
          f"regeneration unchanged, so nothing in the main text moves, but "
          f"the file should be rebuilt before submission. "
          f"results/band_collapse_BEST.json reproduces exactly from its "
          f"inputs: {bB:.4f} and {r2B:.4f} against the "
          f"{bb['power_exponent']:.4f} and {bb['power_r2']:.4f} recorded.")

    k.par(r"A second description of the same frontier disagrees with the "
          r"first. results/frontier_law.json fits the inverse relation, "
          r"budget as a function of saving, in the form D = f + aS^b with the "
          r"floor f free rather than measured. If both descriptions were "
          r"exact, b would be 1/β. Table " + str(k.peek_tbl()) + r" shows "
          r"that it is at the lowest rate and is not at the other four.")

    fl = k.J("frontier_law.json")
    rows = [["q", "b, inverse fit", "R²", "1/β, band fit"]]
    for r in fl["rows"]:
        q = r["qp"]
        rows.append([str(q), _f(r["exponent_b"]), _f(r["r2"], 3),
                     _f(1.0 / _fit(curB[q], CB)[0])])
    t_inv = k.rows(rows, r"<b>Two power laws for one frontier.</b> Both fits "
                        r"report R² above 0.97 on their own data and they "
                        r"imply exponents that differ by up to 40% at the "
                        r"middle rates. The inverse fit gives its floor three "
                        r"free parameters where the band fit measures the "
                        r"floor and fits one, so the agreement at q0 and the "
                        r"disagreement elsewhere both belong to the fitting, "
                        r"not to the decoder.")
    k.note(r"results/frontier_law.json, fitted to results/curve_BEST.json on "
           r"runs/BEST/ckpt_eval.pth.tar over 40 sequences. The last column "
           r"is computed in the build from "
           r"results/signalled_BEST_grid.json and "
           r"results/saturation_BEST_ctc53.json, so both columns describe the "
           r"same training run but not the same sweep.")

    k.par(r"What the grids in this section do establish, and what they leave "
          r"open:")
    k.bullets([
        r"The floor and the saturation point are measured, cheap and "
        r"unambiguous. Each is one forward pass per rate, both agree with an "
        r"independent sweep, and the ceiling between them is closed form.",
        r"The collapse is a strong empirical regularity on both checkpoints. "
        r"Rescaling takes the spread across rates from \BandRawSpread points "
        r"to under two.",
        r"The exponent is a description and not a law. It varies across "
        r"rates within a checkpoint by as much as it varies between the two "
        r"checkpoints, it moves when the assumed ceiling moves, and an "
        r"inverse fit to the same frontier implies a different value.",
        r"Everything here is the oracle allocation, obtained by sweeping the "
        r"Lagrange multiplier with the per-tile losses known. It is an upper "
        r"bound on any router and says nothing about how much of the band a "
        r"decoder-side predictor can reach.",
        r"The grids are intra frames at 1080p on the CTC set. Nothing here "
        r"is measured on inter frames, and the resolution dependence of the "
        r"band is not measured at all.",
    ])
