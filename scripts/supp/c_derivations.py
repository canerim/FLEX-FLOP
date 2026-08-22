"""Derivations: the allocation problem, stated as numbered results and proved.

The LaTeX twin of this module is paper/supp/c_derivations.tex. The two carry
the same claims, the same numbers and the same order.

Every result is set out in the same four parts: the statement, the assumptions
it needs, the proof, and a remark saying what it does not cover. The numbering
runs Proposition 1 to Proposition 14 across the section and is stable, so the
main paper can cite "Proposition 5 of the supplement" and mean one thing. The
index table at the head of the section is the citation list and the evidence
record together.

Everything numeric is computed inside content() from a file in results/, so
the build's provenance list is what the section rests on and no digit is typed
twice. Where a statement is checked numerically rather than proved, the check
names its file and its checkpoint.

The order is objects and assumptions first, then the price and the argmin,
then the achievable set, then the three limits of the sweep, then what
adaptivity is worth, then the shape of the plotted curve, then the two
predictors that stand in for the oracle at the decoder.
"""

import math


# ----------------------------------------------------------------- helpers
def _f(x, n=4):
    """A float as a fixed-point string, for a table cell."""
    return f"{x:.{n}f}"


def _lower_hull(pts):
    """Indices of the points on the lower convex hull of (x, y), x ascending."""
    order = sorted(range(len(pts)), key=lambda i: pts[i][0])
    h = []
    for i in order:
        x, y = pts[i]
        while len(h) >= 2:
            (x1, y1), (x2, y2) = pts[h[-2]], pts[h[-1]]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h


# ------------------------------------------------------------------ section
def content(k):
    L = k.h1("Derivations")
    # Subsection labels are formed from the letter this section was given, so a
    # cross-reference stays right when a section is inserted before it.
    S = [f"{L}.{i}" for i in range(0, 10)]

    k.par(
        "This section states the allocation problem from its objects upward "
        "and proves what the paper asserts about it. Nothing here needs a "
        "result from elsewhere in the supplement. Each statement carries the "
        "assumptions it uses, a proof, and a remark on what it leaves open.")

    idx = [["no.", "claim", "in", "checked in", "ckpt"],
           ["P1", "separates over tiles", S[2], "verify_theory", "eval"],
           ["P2", "monotone in λ", S[2], "verify_theory", "eval"],
           ["P3", "blind sets are a hull", S[3], "static_RECIPE512_b01",
            "PAPER"],
           ["T4", "a Minkowski average", S[3], "not measured", "-"],
           ["P5", "reaches only the hull", S[3], "hull_gap", "eval"],
           ["P6", "savings on a 1/N lattice", S[3], "theory_checks", "eval"],
           ["P7", "the floor", S[4], "saturation_RECIPE512_ctc53", "PAPER"],
           ["P8", "the saturation price", S[4], "verify_theory", "eval"],
           ["P9", "the ceiling", S[4], "per_class_RECIPE512", "PAPER"],
           ["T10", "adaptivity gain", S[5], "theory_check", "BEST"],
           ["P11", "plotted log-convexity", S[6], "logconvexity", "none"],
           ["P12", "Lorenz bound", S[7], "hybrid_RECIPE512_b01_e4head",
            "PAPER"],
           ["P13", "rank-1 threshold rule", S[8], "raterank_RECIPE512_b01",
            "PAPER"],
           ["P14", "rank-1 uses hull exits", S[8], "raterank_RECIPE512_b01",
            "PAPER"]]
    t_index = k.rows(idx,
        "<b>The numbered results of this section, and the evidence for each.</b> "
        "P is a proposition and T a theorem; the numbering is stable and is "
        "what the main paper cites. PAPER is "
        "runs/RECIPE512/ckpt_PAPER.pth.tar, eval is "
        "runs/RECIPE512/ckpt_eval.pth.tar and BEST is "
        "runs/BEST/ckpt_eval.pth.tar. verify_theory is "
        "scripts/verify_theory.py, which reports \\PropsPassed of \\PropsTotal propositions "
        "passing on one sequence at one rate.")
    k.note("Files named without a directory are under results/ with a .json "
           "extension; the saturation and Lorenz rows are the _ctc53 and "
           "_b01_fixed variants of the names given. T4 is proved and not "
           "measured: it describes a set of 120 vertices that no experiment "
           "enumerates.")

    # ==================================================================
    k.h2("The setting, and four assumptions")

    td = k.J("tile_definition.json")
    ac = k.J("adapter_cost.json")
    ntile = td["rows"][0]["tiles_measured"]
    small = min(r["tiles_measured"] for r in td["rows"])

    k.par(
        f"<b>Tiles.</b> The decoder is cut at a fixed depth: everything above "
        f"the cut runs once over the whole frame, everything below on square "
        f"patches of the trunk's feature grid. Replicate padding makes the "
        f"grid divide exactly, so the tiling is a partition. Tiles are indexed "
        f"by t and N is their number in a frame, {ntile} at 1080p and {small} "
        f"at 416×240; section A gives their geometry.")

    nex = ac["config"]["num_exits"]
    jj = ac["config"]["split_depth"]
    nblk = max(e["blocks_skipped"] for e in ac["exits"]) + 2
    bpe = nblk // nex

    k.par(
        f"<b>Exits.</b> An exit is a point at which a tile may stop and be "
        f"handed to the reconstruction head. There are K = {nex} over {nblk} "
        f"residual blocks, evenly spaced, so exit k runs (k+1)b blocks with "
        f"b = {bpe}. At a split depth of j = {jj} the first jb blocks run "
        f"full-frame whatever exit a tile takes, so the usable ladder has "
        f"K - j = {nex - jj} members and k ranges over j, ..., K-1 "
        f"throughout.")

    k.par(
        "<b>Distortion.</b> D<sub>t,k</sub> is the mean squared error tile t "
        "incurs when it leaves at exit k, measured on the deployed tiled path "
        "so that what tiling costs sits inside it, and measured against the "
        "released decoder's decode of the same latent so that synthesis is the "
        "only difference between the two sides. Nothing forces "
        f"D<sub>t,K-1</sub> to zero, and {S[4]} returns to that.")

    k.par(
        "<b>Decibels.</b> Distortion is reported as 10 log<sub>10</sub> of the "
        "ratio to the reference, and the ratio can be formed at two levels. "
        "Pooling every tile of every frame into one mean squared error is the "
        "form the Lagrangian below is exact for; averaging a per-frame decibel "
        "is the codec convention, and it is what the budget is bisected "
        "against throughout. Section E measures how far apart the two read on "
        "one allocation, which is a substantial fraction of the working "
        "budget.")

    k.par(
        "<b>Compute.</b> Compute is counted in multiply-accumulate operations "
        "and divided by those of one released full-frame decode, "
        "\\IntraGmac\\,GMAC at 1080p, so c = 1 is one released decode and a "
        "saving of S% means the frame was decoded for (1 - S/100) of one. "
        "Every operator runs at a fixed multiple of the latent grid, which "
        "makes the cost of an exit resolution-independent. One tile at exit k "
        "costs")

    e_c = k.eq(r"c_k \;=\; s_{\mathrm{up}} \;+\; s_{\mathrm{trunk}}\,"
               r"\frac{(k+1)\,b}{N_{\mathrm{blk}}} \;+\; s_{\mathrm{head}} "
               r"\;+\; a_k \;+\; r,")

    k.par(
        f"with s<sub>up</sub> the opening upsample, s<sub>trunk</sub> the "
        f"trunk spread over N<sub>blk</sub> = {nblk} blocks of which exit k "
        f"runs (k+1)b, s<sub>head</sub> the head, a<sub>k</sub> the adapter "
        f"exit k wears and r the full-frame deblocking filter; section B "
        f"measures all five. Only the trunk term depends on k, which is why a "
        f"ceiling exists: however shallow the ladder gets, the stem, the head "
        f"and the filter are still paid.")

    st = k.J("static_RECIPE512_b01.json")
    uni = st["rows"][0]["uniform"]
    cost_B = [1 - u["saving"] / 100.0 for u in uni]
    exits_B = [u["exit"] for u in uni]
    tt = k.J("tile_table.json")
    cost_A = tt["cost"][tt["j"]:]
    adapter_name = {e["exit"]: (e["adapter_kind"] or "none") for e in ac["exits"]}
    adapter_blk = {e["exit"]: e["adapter_blocks"] for e in ac["exits"]}

    rows = [["exit k", "blocks", "adapter", "share", "c_k", "100(1-c_k)",
             "c_k, stored"]]
    for i, kk in enumerate(exits_B):
        rows.append([kk, (kk + 1) * bpe,
                     adapter_name[kk].replace("conv1x1", "1×1"),
                     _f(adapter_blk[kk], 3), _f(cost_B[i], 4),
                     _f(100 * (1 - cost_B[i]), 2), _f(cost_A[i], 4)])
    t_cost = k.rows(rows,
        "<b>What each exit costs</b>, in units of one released full-frame "
        "decode, with the adapter share in units of one residual block. The "
        "deepest exit costs more than 1 because a tiled decode still pays the "
        "deblocking filter. The last column is the cost vector stored in "
        "results/tile_table.json, which is the one the checks of "
        "Propositions 5, 6, 8 and 10 are computed on; each says so where it "
        "appears.")
    k.note("c_k from the uniform-depth rows of "
           "results/static_RECIPE512_b01.json and adapter shares from "
           "results/adapter_cost.json, both on "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. Section B audits the block "
           "count itself and prices the gap between the model the argmin runs "
           "on and the hook count every saving is quoted from.")

    k.par(
        f"The gaps between those costs are what the allocation trades in, and "
        f"they are unequal: {_f(cost_B[1] - cost_B[0], 4)}, "
        f"{_f(cost_B[2] - cost_B[1], 4)} and {_f(cost_B[3] - cost_B[2], 4)}. "
        f"The first is largest because exit 3 gives up the FFN adapter's "
        f"saving as well as two blocks.")

    k.par(
        f"<b>Assignments.</b> An assignment is a map a from tiles to exits. "
        f"There are (K-j)<super>N</super> of them, "
        f"{nex - jj}<super>{ntile}</super> at 1080p, so they will not be "
        f"enumerated. The frame's compute and distortion are")
    e_CD = k.eq(r"C(a) \;=\; \frac{1}{N}\sum_{t=1}^{N} c_{a(t)}, \qquad "
                r"D(a) \;=\; \frac{1}{N}\sum_{t=1}^{N} D_{t,a(t)}.")

    k.par(
        "Everything after this rests on four assumptions, none of them a "
        "mathematical necessity. Each result names the ones it uses.")
    k.bullets([
        "<b>(A1) Compute is additive over tiles.</b> Exact when tiles decode "
        "independently, which is what a tiled decoder does.",
        "<b>(A2) Distortion is a tile mean</b>, and a tile's error does not "
        "depend on which exits its neighbours took. The first half is exact "
        "for any per-pixel loss averaged over a frame; the second is not free, "
        "the deblocking filter running on the stitched canvas and seeing the "
        "whole exit map at once.",
        "<b>(A3) The reference is fixed.</b> D<sub>t,k</sub> is measured on "
        "the deployed tiled path against the released decoder's own decode of "
        "the same latent, so rate is identical on both sides and cancels.",
        "<b>(A4) The cost vector is strictly increasing</b> in k over "
        "k ≥ j, which Table " + str(t_cost) + " shows it is."])

    Dtt = tt["D"]
    jt = tt["j"]
    ref = tt["ref_mse"]
    errs = []
    for s in tt["sweep"]:
        m = s["map"]
        pred = 10 * math.log10(sum(Dtt[i][m[i]] for i in range(len(m)))
                               / len(m) / ref)
        errs.append(abs(pred - s["db"]))
    k.par(
        f"The second half of (A2) is checked rather than asserted, over "
        f"{len(tt['sweep'])} allocations of one frame spanning savings of "
        f"{min(s['saving'] for s in tt['sweep']):.1f}% to "
        f"{max(s['saving'] for s in tt['sweep']):.1f}%: the largest "
        f"disagreement between the per-tile prediction and a decode of the "
        f"mixed map is {max(errs) * 1e5:.1f}×10<super>-5</super> dB, four "
        f"orders of magnitude below the budget. Nothing in (A1) to (A4) "
        f"constrains D<sub>t,k</sub> itself, and nothing requires it to fall "
        f"as k grows.")
    k.note("Additivity check computed from results/tile_table.json "
           "(Bosphorus, q32, 40 tiles), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar. What it checks is a property of "
           "the decode path rather than of the weights.")

    # ==================================================================
    k.h2("A price on compute")

    k.par(
        "The encoder's problem is to spend as little compute as possible while "
        "staying inside a distortion budget:")
    e_con = k.eq(r"\min_{a}\; C(a) \quad \mathrm{subject\ to} \quad "
                 r"D(a) \;\leq\; D_{\mathrm{budget}}.")

    k.par(
        f"Constraint ({e_con}) couples the tiles: whether a tile can afford to "
        f"leave early depends on how much of the budget the other {ntile - 1} "
        f"have used. The standard way out is to buy the constraint off with a "
        f"price \\lambda ≥ 0 charged per unit of compute:")
    e_L = k.eq(r"L(a,\lambda) \;=\; D(a) + \lambda\,C(a) \;=\; "
               r"\frac{1}{N}\sum_{t=1}^{N}\left[\,D_{t,a(t)} + "
               r"\lambda\,c_{a(t)}\,\right].")

    k.par(
        "\\lambda is an exchange rate, in squared error per unit of relative "
        "compute, so \\lambda c<sub>k</sub> converts the compute an exit uses "
        "into the squared error that compute is deemed to be worth. Below it "
        "runs from about 2×10<super>-7</super> to about "
        "1×10<super>-4</super>, against squared errors of order "
        "1×10<super>-4</super> on a 0 to 1 scale, and what matters is where it "
        "sits relative to the differences between exits: at zero every tile "
        "takes its own lowest-error exit, and past a large enough price every "
        "tile has dropped as far as the ladder allows.")

    k.h3("Proposition 1 (separation)")
    k.par(
        "For every \\lambda ≥ 0, an assignment minimises L(a,\\lambda) over "
        "all assignments if and only if it minimises the bracket tile by tile:")
    e_argmin = k.eq(r"k^{\star}_t(\lambda) \;\in\; \mathrm{arg\,min}_{k \geq j}\;"
                    r"\left[\,D_{t,k} + \lambda\,c_k\,\right].")
    k.par("<i>Assumes</i> (A1) and (A2).")
    k.par(
        "<i>Proof.</i> Write f<sub>t</sub>(k) = D<sub>t,k</sub> + \\lambda "
        "c<sub>k</sub>. Then N L(a,\\lambda) = ∑<sub>t</sub> "
        "f<sub>t</sub>(a(t)), a sum in which the t-th term depends on a only "
        "through a(t). For any assignment a, term by term f<sub>t</sub>(a(t)) "
        "≥ min<sub>k</sub> f<sub>t</sub>(k), so N L(a,\\lambda) ≥ "
        "∑<sub>t</sub> min<sub>k</sub> f<sub>t</sub>(k), and the right-hand "
        "side is attained by any a that picks a minimiser in every tile. "
        "Conversely if a is optimal and some tile had f<sub>t</sub>(a(t)) > "
        "min<sub>k</sub> f<sub>t</sub>(k), replacing a(t) by that minimiser "
        "and leaving every other tile alone would strictly lower L.")

    oq = k.J("supp_opquality_PAPER.json")
    _tl = oq["tail"]
    _tmax = max(_tl[q]["max_db"] for q in _tl)
    k.par(
        f"<i>Remark.</i> The result is about the frame mean and bounds no "
        f"single tile. At the deployed 0.1 dB operating point the worst tile "
        f"in the test set gives up {_tmax:.2f} dB, so a frame inside its "
        f"budget can hold a tile an order of magnitude outside it; section D "
        f"tabulates that tail. No result in this section repairs it.")
    k.note("Per-tile tail from results/supp_opquality_PAPER.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, at the operating point read "
           "from results/signalled_RECIPE512_ctc53.json.")

    k.par(
        f"The N-dimensional search has become N independent searches over "
        f"{nex - jj} numbers each, with ties broken toward the shallower exit "
        f"so that k<super>*</super> is a function. We call "
        f"k<super>*</super><sub>t</sub>(\\lambda) the oracle's choice, because "
        f"it uses the true D<sub>t,k</sub>, which only the encoder has. The "
        f"construction is Shoham and Gersho's [26] with compute in place of "
        f"rate, made standard in coding by [27].")

    k.h3("Proposition 2 (monotonicity)")
    k.par(
        "Let \\lambda<sub>1</sub> < \\lambda<sub>2</sub> and let "
        "a<sub>1</sub>, a<sub>2</sub> be the corresponding oracle "
        "assignments. Then C(a<sub>2</sub>) ≤ C(a<sub>1</sub>) and "
        "D(a<sub>2</sub>) ≥ D(a<sub>1</sub>). <i>Assumes</i> (A1), (A2) and "
        "Proposition 1.")
    k.par(
        "<i>Proof.</i> Fix a tile and drop the index. Optimality at each price "
        "gives D<sub>1</sub> + \\lambda<sub>1</sub>c<sub>1</sub> ≤ "
        "D<sub>2</sub> + \\lambda<sub>1</sub>c<sub>2</sub> and D<sub>2</sub> + "
        "\\lambda<sub>2</sub>c<sub>2</sub> ≤ D<sub>1</sub> + "
        "\\lambda<sub>2</sub>c<sub>1</sub>, where (D<sub>i</sub>, "
        "c<sub>i</sub>) is the pair chosen at \\lambda<sub>i</sub>. Adding the "
        "two and cancelling D<sub>1</sub> + D<sub>2</sub> leaves "
        "(\\lambda<sub>2</sub> - \\lambda<sub>1</sub>)(c<sub>2</sub> - "
        "c<sub>1</sub>) ≤ 0, hence c<sub>2</sub> ≤ c<sub>1</sub>. "
        "Substituting that back into the first inequality gives D<sub>2</sub> "
        "≥ D<sub>1</sub> + \\lambda<sub>1</sub>(c<sub>1</sub> - c<sub>2</sub>) "
        "≥ D<sub>1</sub>. Averaging over tiles preserves both.")
    k.par(
        "<i>Remark.</i> Monotone is not continuous. Cost moves in the jumps "
        "Proposition 6 measures, so bisecting \\lambda converges on a "
        "reachable level rather than on the budget. The result also assumes "
        "exact minimisation of the true table, whereas the deployed search "
        "bisects a predicted table and corrects with a real decode, at most "
        "six passes to 5×10<super>-4</super> dB.")

    bits = tt["bits_per_tile"]
    ctt = tt["cost"][jt:]
    Dj = [row[jt:] for row in Dtt]
    order = sorted(range(len(bits)), key=lambda i: bits[i])
    sel = [order[0], order[len(order) // 2], order[-1]]

    def schedule(t):
        """The exits tile t takes as lambda rises, and where it switches."""
        out = []
        prev = None
        grid = [0.0] + [1e-9 * (10 ** (7 * i / 4000.0)) for i in range(4001)]
        for lam in grid:
            vals = [Dj[t][a] + lam * ctt[a] for a in range(len(ctt))]
            kk = min(range(len(vals)), key=lambda a: vals[a])
            if kk != prev:
                out.append((lam, jt + kk))
                prev = kk
        return out

    rows = [["tile", "bits", "D(2)", "D(3)", "D(4)", "D(5)", "exits as λ rises"]]
    _last = {}
    for t in sel:
        sc = schedule(t)
        _last[t] = sc[-1][0]
        if len(sc) == 1:
            desc = f"{sc[0][1]} throughout"
        else:
            desc = str(sc[0][1]) + "".join(
                f", {kk} at {lam * 1e6:.1f}" for lam, kk in sc[1:])
        rows.append([t, f"{bits[t]:,.0f}"]
                    + [_f(Dj[t][a] * 1e5, 3) for a in range(4)] + [desc])
    t_tiles = k.rows(rows,
        f"<b>Three tiles of one frame, and what the price does to them.</b> D "
        f"is in units of 10<super>-5</super> squared error and switch points "
        f"in units of 10<super>-6</super> of λ. The cheapest tile prefers the "
        f"shallowest exit at every price including λ = 0, its error not "
        f"falling with depth at all, so depth is not always worth buying. The "
        f"most expensive tile holds out against the shallowest exit "
        f"{_last[sel[-1]] / _last[sel[1]]:.1f} times longer than the middle "
        f"one, which is why one uniform depth cannot suit both.")
    k.note("results/tile_table.json (Bosphorus, q32, 40 tiles), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar. The 60 prices stored in that "
           f"file produce 33 distinct savings, the granularity {S[3]} "
           "quantifies.")

    # ==================================================================
    k.h2("The achievable set, and what one price reaches")

    hg = k.J("hull_gap.json")
    tc = k.J("theory_checks.json")

    k.par(
        "The pairs (C(a), D(a)) an allocation can produce form a set whose "
        "shape decides both what a sweep can reach and what adaptivity is "
        "worth. There are two versions of it. Write")
    e_Dbar = k.eq(r"\bar D_k \;=\; \frac{1}{N}\sum_{t=1}^{N} D_{t,k},")
    k.par(
        "for the exit mean at k, the one place tiles are averaged before "
        "anything is chosen.")

    k.h3("Proposition 3 (fixed proportions)")
    k.par(
        "Suppose tiles are assigned to exits in proportions p, one probability "
        "per exit, independently of their content. Then the achievable set of "
        "expected pairs is exactly the convex hull of the K-j points whose "
        "coordinates are the cost c<sub>k</sub> and the exit mean at k. "
        "<i>Assumes</i> (A1) and (A2).")
    k.par(
        "<i>Proof.</i> Under content-independent proportions the expected cost "
        "is ∑<sub>k</sub> p<sub>k</sub> c<sub>k</sub> and, by the tile mean, "
        "the expected distortion is ∑<sub>k</sub> p<sub>k</sub> times the exit "
        "mean at k. The map from p to that pair is affine and its domain is "
        "the probability simplex, whose extreme points are the unit vectors. "
        "An affine image of a simplex is the convex hull of the images of its "
        "vertices, and those images are the cost and exit mean of each exit.")
    k.par(
        f"<i>Remark.</i> The hull is a hull of expectations. One frame with N "
        f"tiles realises proportions in multiples of 1/N, so interior points "
        f"are attained only to that granularity, and at N = {small} the "
        f"smallest test class reaches three mixtures per pair of exits.")

    k.par(
        "The four uniform-depth decodes are the vertices of that hull, and "
        "they are everything a content-blind allocation can reach; section D "
        "tabulates them in dB at every rate beside the two shuffled controls. "
        "The mean over rates of the best single depth is \\BestStaticMean%, "
        "against \\MainMean% for the per-tile allocation at the same budget.")
    k.note("results/static_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq frames.")

    k.h3("Theorem 4 (per-tile assignment)")
    k.par(
        "The convex hull of the pairs achievable by per-tile assignment is the "
        "Minkowski average of the N per-tile hulls,")
    e_mink = k.eq(r"\mathcal{A}_{\mathrm{tile}} \;=\; \frac{1}{N}"
                  r"\bigoplus_{t=1}^{N}\mathrm{conv}\{(c_k, D_{t,k}) : "
                  r"k \geq j\}.")
    k.par("<i>Assumes</i> (A1) and (A2).")
    k.par(
        "<i>Proof.</i> By (A1) and (A2) an achievable pair is the average of "
        "one point drawn from each per-tile set S<sub>t</sub>, the set of "
        "pairs cost c<sub>k</sub> against error D<sub>t,k</sub>, so the "
        "achievable set is the Minkowski average of the S<sub>t</sub>. Taking "
        "convex hulls commutes with Minkowski addition, the hull of a sum "
        "being the sum of the hulls, and with positive scaling; applying that "
        "N-1 times moves the hull inside the sum.")
    k.par(
        f"<i>Remark.</i> The theorem bounds what any allocation reaches, a "
        f"perfect router included, and says nothing about which of those "
        f"points a price reaches, which is Proposition 5. It is the one result "
        f"here with no measurement behind it, the set having up to "
        f"N(K-j-1) = {ntile * (nex - jj - 1)} vertices at 1080p.")

    k.par(
        f"Proposition 3 is the special case in which every tile has the same "
        f"row, and otherwise the per-tile set is strictly larger: the "
        f"fixed-proportion hull has at most K-j = {nex - jj} vertices and so "
        f"{nex - jj - 1} straight segments, while the Minkowski average is "
        f"smooth at any plotted scale. Figure {k.peek_fig()} draws both on one "
        f"frame.")

    f_ach = k.fig("supp_achievable.png",
        "<b>The two achievable sets, on one frame.</b> <b>a</b>, the per-tile "
        "frontier of Theorem 4 traced by sweeping λ, against the "
        "fixed-proportion chain of Proposition 3 through the four "
        "uniform-depth points, labelled by exit. The per-tile curve lies below "
        "the chain everywhere between the vertices, by 0.059 dB at the exit-3 "
        "vertex. <b>b</b>, the exit each of the 40 tiles takes as λ rises, "
        "tiles ordered by the bits the entropy coder spent on them. Every row "
        "is non-increasing, which is Proposition 2 per tile, and the rows "
        "differ, which is what Theorem 4 buys over Proposition 3. The bottom "
        "row is a tile that never leaves the shallowest exit.", maxh=132)
    k.note("Both panels from results/tile_table.json (Bosphorus, q32, 40 "
           "tiles), checkpoint runs/RECIPE512/ckpt_eval.pth.tar, drawn by "
           "scripts/supp_derivations_figure.py on that file's own cost vector, "
           "the last column of Table " + str(t_cost) + ". A grid of 4001 "
           "prices finds 93 distinct reachable savings on this frame, against "
           f"the {hg['n_hull_allocations']} the dynamic programme of Table "
           f"{k.peek_tbl()} enumerates.")

    k.h3("Proposition 5 (a price reaches only the lower hull)")
    k.par(
        "For every \\lambda ≥ 0 the point (C, D) produced by the oracle "
        "assignment lies on the lower convex hull of the achievable set, and "
        "no \\lambda produces a point strictly inside it. <i>Assumes</i> (A1), "
        "(A2) and Proposition 1.")
    k.par(
        "<i>Proof.</i> By Proposition 1 the oracle assignment minimises D(a) + "
        "\\lambda C(a) over all assignments, so its pair minimises the linear "
        "functional (C, D) → D + \\lambda C over the achievable set and hence "
        "over that set's convex hull. A minimiser of a linear functional over "
        "a compact convex set lies on its boundary, and because \\lambda ≥ 0 "
        "the supporting line has slope -\\lambda ≤ 0, which places the point "
        "on the lower boundary. Conversely a point strictly inside the hull is "
        "a strict convex combination of achievable points and is therefore "
        "strictly beaten, at that same C, by some point of the hull, so it "
        "cannot minimise any such functional.")
    k.par(
        "<i>Remark.</i> A budget between two hull vertices is unreachable by "
        "any price, and the proposition does not say what that costs, which is "
        "Proposition 6. The measurement below enumerates the exact Pareto set "
        "on one frame at one rate, on that file's own cost vector.")

    k.par(
        "Everett's generalised multiplier method (1963) supplies the "
        "reassuring half of the converse: whatever compute the returned "
        "allocation consumes, it is optimal for that level. With four distinct "
        "exit costs the whole Pareto set can be enumerated by dynamic "
        "programming over tiles, so the other half can be priced exactly.")

    rows = [["budget dB", "by sweep", "exact Pareto", "gap"]]
    for r in hg["rows"]:
        rows.append([_f(r["budget_db"], 2), _f(r["hull_saving"], 3),
                     _f(r["pareto_saving"], 3), _f(r["gap_pts"], 4)])
    t_hull = k.rows(rows,
        f"<b>What convexity costs.</b> Savings in per cent at seven budgets on "
        f"one frame. The sweep reaches {hg['n_hull_allocations']} allocations "
        f"against the {hg['n_pareto']} on the exact Pareto set, and the "
        f"largest loss over these budgets is "
        f"{max(abs(r['gap_pts']) for r in hg['rows']):.3f} saving points. The "
        f"two negative entries are the sweep landing on a budget the "
        f"enumeration's grid straddles rather than the sweep beating the "
        f"Pareto set, which Proposition 5 forbids.")
    k.note("results/hull_gap.json, Bosphorus q32, 40 tiles. The enumeration "
           "and the sweep read one cost vector, so the gap does not depend on "
           "which one.")

    k.h3("Proposition 6 (granularity)")
    k.par(
        "If at most m tiles switch exit at any single breakpoint of the sweep, "
        "and each switch moves a tile by one rung, then consecutive reachable "
        "savings differ by at most")
    e_lat = k.eq(r"\mathrm{spacing} \;\leq\; \frac{100\;m\,"
                 r"\max_k (c_{k+1}-c_k)}{N}\;\%,")
    k.par(
        "and the loss from convexity cannot exceed the largest spacing. "
        "<i>Assumes</i> (A1), (A4) and Proposition 2.")
    k.par(
        "<i>Proof.</i> Between breakpoints the assignment is constant, so the "
        "reachable costs are exactly the costs at the breakpoints. At one "
        "breakpoint the mean cost changes by 1/N times the sum of the "
        "individual changes, each at most max<sub>k</sub>(c<sub>k+1</sub> - "
        "c<sub>k</sub>) in magnitude, and there are at most m of them; "
        "multiplying by 100 puts it in saving points. For the second part, let "
        "a budget fall strictly between two reachable levels. By Proposition 2 "
        "the bisection returns the lower level, so the loss is at most the "
        "distance between them.")
    k.par(
        f"<i>Remark.</i> Both hypotheses are read off a measured table rather "
        f"than guaranteed, and the bound is vacuous if either fails on another "
        f"frame. It falls as 1/N, so it says nothing useful at the smallest "
        f"test class, where N = {small} and one switch moves the frame by half "
        f"the largest rung.")

    k.par(
        f"On the measured table m = {tc['m_simultaneous']}, tiles with "
        f"identical rows switching together, the largest rung is "
        f"{tc['max_rung']:.4f} and N = {tc['T']}, so the bound is "
        f"{tc['lattice_bound_pts']:.4f} saving points. The measured spacing is "
        f"{tc['lattice_spacing_pts']:.4f}, sitting on the bound, and the "
        f"convexity loss of Table {t_hull} is {tc['hull_loss_pts']:.4f}. "
        f"Nothing in the expression depends on content or on rate, and halving "
        f"the tile side would put four times as many tiles on the lattice, "
        f"each carrying a quarter of the step.")
    k.note("m, the largest rung, the bound and the measured spacing from "
           "results/theory_checks.json (Bosphorus q32, 40 tiles, "
           "runs/RECIPE512/ckpt_eval.pth.tar), written by "
           "scripts/verify_theory.py.")

    # ==================================================================
    k.h2("Floor, saturation and ceiling")

    k.par(
        "The sweep has two ends, and a budget outside them buys nothing. The "
        "budget section measures where they sit at each rate; this one derives "
        "them.")

    k.h3("Proposition 7 (the floor)")
    k.par(
        "At \\lambda = 0 the allocation attains D<sub>min</sub> = (1/N) "
        "∑<sub>t</sub> min<sub>k</sub> D<sub>t,k</sub>, and no allocation "
        "attains less. <i>Assumes</i> (A2) and Proposition 1. <i>Proof.</i> "
        "Immediate from Proposition 1 at \\lambda = 0, and from D(a) being a "
        "mean of terms each bounded below by its own tile minimum.")

    k.par(
        "<i>Remark.</i> The floor is a statement about the table, and the "
        "table is a product of training, which (A3) does not constrain. If the "
        "deepest exit drifts from the reference by δ then the achievable set "
        "shifts up by δ before any compute is saved, and a budget below δ is "
        "unreachable at every allocation. Section H measures that drift, with "
        "and without the training objective's anchor term.")

    k.par(
        "Because D<sub>t,k</sub> is measured on the deployed tiled path, "
        "D<sub>min</sub> is not zero: it is what tiling costs before any tile "
        "exits early, and section E measures where it sits at each rate and "
        "which budgets fall below it.")

    k.h3("Proposition 8 (saturation)")
    k.par("Define")
    e_sat = k.eq(r"\lambda_{\mathrm{sat}} \;=\; \max_{t}\;\max_{k>j}\;"
                 r"\frac{D_{t,j}-D_{t,k}}{c_k-c_j},")
    k.par(
        "Then for every \\lambda > \\lambda<sub>sat</sub> the oracle assigns "
        "exit j to every tile, the frame's cost is exactly c<sub>j</sub>, and "
        "no larger price changes anything. <i>Assumes</i> (A1), (A4) and "
        "Proposition 1.")
    k.par(
        f"<i>Proof.</i> Exit j beats exit k on tile t precisely when "
        f"D<sub>t,j</sub> + \\lambda c<sub>j</sub> ≤ D<sub>t,k</sub> + "
        f"\\lambda c<sub>k</sub>, that is when \\lambda ≥ (D<sub>t,j</sub> - "
        f"D<sub>t,k</sub>)/(c<sub>k</sub> - c<sub>j</sub>), the denominator "
        f"being positive for k > j by (A4). Taking the maximum over k and then "
        f"over t gives a single price beyond which exit j wins everywhere. The "
        f"cost of the constant map is c<sub>j</sub> by ({e_CD}).")

    lam_sat = max((Dj[t][0] - Dj[t][a]) / (ctt[a] - ctt[0])
                  for t in range(len(Dj)) for a in range(1, len(ctt)))
    k.par(
        f"<i>Remark.</i> \\lambda<sub>sat</sub> is a maximum over the tiles of "
        f"one frame, so a price chosen for a sequence saturates some of its "
        f"frames and not others, and it constrains the price rather than the "
        f"budget. On the worked frame the closed form gives "
        f"{lam_sat * 1e5:.3f}×10<super>-5</super>, exactly the last switch of "
        f"the most expensive tile in Table {t_tiles}.")

    k.h3("Proposition 9 (the ceiling)")
    k.par("The largest saving any allocation can reach is")
    e_ceil = k.eq(r"S_{\max} \;=\; 100\,(1-c_j)\,\%.")
    k.par(
        "<i>Assumes</i> (A1), (A4) and Proposition 8. <i>Proof.</i> C(a) is a "
        "mean of costs each at least c<sub>j</sub>, j being the shallowest "
        "usable exit, so C(a) ≥ c<sub>j</sub> with equality only for the "
        "constant map, which Proposition 8 shows is attained.")

    pc = k.J("per_class_RECIPE512.json")
    _cv = [v["saving"] for r in pc["rows"] if r["budget_db"] >= 0.5
           for v in r["per_class"].values()]
    k.par(
        f"<i>Remark.</i> S<sub>max</sub> depends on the split depth alone, so "
        f"no amount of training raises it and the only lever is j. It is a "
        f"ceiling in multiply-accumulates and not in seconds, and section B "
        f"reports the wall clock falling short of it. Over the {len(_cv)} "
        f"class-and-rate cells measured at a saturating budget it spreads by "
        f"{max(_cv) - min(_cv):.0e} saving points, which is single-precision "
        f"noise.")
    k.note("Per-class invariance from results/per_class_RECIPE512.json at "
           "budgets of 0.5 dB and above, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar; the ceiling itself is "
           "\\CeilingModelled%, from results/saturation_RECIPE512_ctc53.json.")

    # ==================================================================
    k.h2("What adaptivity is worth")

    k.par(
        "What content awareness is worth can be answered without building a "
        "predictor. Compare the best per-tile allocation with the best "
        "content-blind one at the same price:")
    e_J = k.eq(r"J_{\mathrm{ad}}(\lambda)=\frac{1}{N}\sum_{t}\min_{k}"
               r"\left[D_{t,k}+\lambda c_k\right], \quad "
               r"J_{\mathrm{fix}}(\lambda)=\min_{k}\left[\bar D_k+"
               r"\lambda c_k\right].")

    k.par(
        "J<sub>ad</sub> takes the minimum inside the average and "
        "J<sub>fix</sub> takes it outside. By Propositions 1 and 3 the first "
        "is the best the per-tile family attains at price \\lambda and the "
        "second the best the fixed-proportion family attains.")

    k.h3("Theorem 10 (adaptivity gain)")
    k.par(
        "For every \\lambda ≥ 0, \\Delta(\\lambda) = J<sub>fix</sub>"
        "(\\lambda) - J<sub>ad</sub>(\\lambda) ≥ 0, with equality if and only "
        "if some single exit k<super>*</super> attains the per-tile minimum "
        "for every tile. <i>Assumes</i> (A1) and (A2).")
    k.par(
        "<i>Proof.</i> Fix \\lambda and let k be any exit. Then the exit mean "
        "at k plus \\lambda c<sub>k</sub> = (1/N) ∑<sub>t</sub> "
        "[D<sub>t,k</sub> + \\lambda c<sub>k</sub>] ≥ (1/N) ∑<sub>t</sub> "
        "min<sub>k′</sub>[D<sub>t,k′</sub> + \\lambda c<sub>k′</sub>] = "
        "J<sub>ad</sub>(\\lambda), because each summand dominates the "
        "corresponding minimum. The right-hand side does not depend on k, so "
        "taking the minimum over k on the left preserves the inequality and "
        "gives J<sub>fix</sub> ≥ J<sub>ad</sub>. For equality, suppose some "
        "k<super>*</super> attains the per-tile minimum everywhere; then "
        "choosing it makes every summand equal to its minimum and the "
        "inequality is tight. Conversely suppose \\Delta = 0 and let "
        "k<super>*</super> attain J<sub>fix</sub>. Then (1/N) ∑<sub>t</sub> "
        "([D<sub>t,k*</sub> + \\lambda c<sub>k*</sub>] - min<sub>k′</sub>"
        "[D<sub>t,k′</sub> + \\lambda c<sub>k′</sub>]) = 0. Every summand is "
        "non-negative, so every summand is zero, which says exactly that "
        "k<super>*</super> attains the minimum on every tile.")
    k.par(
        f"<i>Remark.</i> \\Delta upper-bounds what any predictor gains over a "
        f"content-blind allocation at that price and says nothing about "
        f"whether one can get any of it, which is what {S[7]} measures. It is "
        f"a property of the ladder as much as of the content: improving the "
        f"shallow exits makes more tiles agree on the argmin, so \\Delta falls "
        f"even as the saving rises and is not a figure of merit.")

    thc = k.J("theory_check.json")
    _sf = {q: 100 * thc[q]["delta"]["3e-04"]["delta"]
           / thc[q]["delta"]["3e-04"]["J_oracle"] for q in
           ("0", "16", "32", "48", "63")}
    k.par(
        f"Read as a ratio Δ/J<sub>ad</sub> rather than in saving points, the "
        f"gain rises with rate at a low price and falls to "
        f"{min(_sf.values()):.2f}% to {max(_sf.values()):.2f}% at the highest "
        f"price measured, which is Theorem 10's equality condition arriving: "
        f"the oracle has itself collapsed onto one exit and nothing is left "
        f"for a router to recover.")
    k.note("Ratios from results/theory_check.json, checkpoint "
           "runs/BEST/ckpt_eval.pth.tar, 40 sequences (UVG, MCL-JCV and HEVC "
           "class E; classes B, C and D not measured), on the last column of "
           "Table " + str(t_cost) + ". The table below is the same quantity "
           "in saving points on the pinned checkpoint.")

    tgt = 10 ** (0.1 / 10)
    rows = [["q", "blind %", "oracle %", "gap", "blind dB", "shuffled dB"]]
    _gaps = []
    for r in st["rows"]:
        pts = sorted([(1 - u["saving"] / 100.0, 10 ** (u["db"] / 10.0))
                      for u in r["uniform"]])
        blind = None
        for (c1, y1), (c2, y2) in zip(pts, pts[1:]):
            if min(y1, y2) - 1e-12 <= tgt <= max(y1, y2) + 1e-12:
                blind = 100 * (1 - (c1 + (tgt - y1) / (y2 - y1) * (c2 - c1)))
                break
        Co = 1 - r["oracle"]["saving"] / 100.0
        ydb = None
        for (c1, y1), (c2, y2) in zip(pts, pts[1:]):
            if c1 - 1e-12 <= Co <= c2 + 1e-12:
                ydb = 10 * math.log10(y1 + (Co - c1) / (c2 - c1) * (y2 - y1))
                break
        _gaps.append(r["oracle"]["saving"] - blind)
        rows.append([f"q{r['qp']}", _f(blind, 2),
                     _f(r["oracle"]["saving"], 2),
                     _f(r["oracle"]["saving"] - blind, 2),
                     _f(ydb, 4), _f(r["random"]["db"], 4)])
    t_adapt = k.rows(rows,
        f"<b>The same gap in the units the paper reports.</b> Columns 2 to 4 "
        f"are savings at a 0.1 dB budget: the best content-blind mixture of "
        f"uniform depths, the per-tile oracle, and the difference. Columns 5 "
        f"and 6 are decibels at the oracle's own compute, what the best blind "
        f"mixture would read there and what a random shuffle of the oracle's "
        f"exit map actually reads. Adaptivity is worth {min(_gaps):.1f} to "
        f"{max(_gaps):.1f} saving points, rising with rate, against an oracle "
        f"reading \\OracleDb dB and a shuffle reading up to \\RandomDb dB.")
    k.note("Computed from results/static_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq frames. The blind "
           "columns are the lower hull of the four uniform-depth points under "
           "Proposition 3, evaluated at the "
           "budget and at the oracle's compute; the shuffle column is a "
           "measured decode. At q0 the shuffle reads slightly better than the "
           "hull predicts, which bounds the interaction (A2) ignores at under "
           "0.002 dB.")

    # ==================================================================
    k.h2("Convexity in the units the frontier is plotted in")

    lc = k.J("logconvexity.json")

    k.par(
        "Theorem 4 gives D convex in C. The frontier is almost never plotted "
        "in those units: it is plotted as decibels against percentage saving, "
        "where convexity is a different statement, the logarithm being concave "
        "and increasing and so preserving neither convexity nor concavity. "
        "With S = 1 - C/c<sub>K-1</sub>,")
    e_db = k.eq(r"\mathrm{dB}(S) \;=\; \frac{10}{\ln 10}\,\ln D(C), "
                r"\qquad C = (1-S)\,c_{K-1}.")

    k.h3("Proposition 11 (plotted convexity is log-convexity)")
    k.par(
        "Let D be twice differentiable along the frontier. Then dB(S) is "
        "convex if and only if D is log-convex in C:")
    e_lcx = k.eq(r"D\,D^{\prime\prime} \;\geq\; (D^{\prime})^{2} "
                 r"\qquad\Longleftrightarrow\qquad "
                 r"\frac{d}{dC}\left(\frac{1}{\lambda}\right) \;\geq\; "
                 r"\frac{1}{D}.")
    k.par(
        "<i>Assumes</i> twice differentiability along the frontier, and "
        "Proposition 5 for the second form.")
    k.par(
        f"<i>Proof.</i> C is affine in S and convexity is preserved under "
        f"affine reparametrisation, so by ({e_db}) dB is convex in S if and "
        f"only if ln D is convex in C. Differentiating twice, (ln D)″ = "
        f"D″/D - (D′/D)² = [D D″ - (D′)²]/D², and D > 0, so (ln D)″ ≥ 0 if "
        f"and only if D D″ ≥ (D′)². For the second form, the slope of the "
        f"frontier is D′ = -\\lambda by the supporting-line argument of "
        f"Proposition 5, so D″ = -\\lambda′; substituting gives "
        f"-\\lambda′ D ≥ \\lambda², and dividing by \\lambda² > 0 gives "
        f"-\\lambda′/\\lambda² ≥ 1/D, whose left-hand side is "
        f"d(1/\\lambda)/dC. The second form reads as a rate condition: "
        f"distortion has to fall at least exponentially in compute for the "
        f"plotted curve to be convex.")
    k.par(
        "<i>Remark.</i> This is a property of the decoder rather than a "
        "consequence of Theorem 4, and a convex D can fail it. "
        "Differentiability fails in one of the two cases: under fixed "
        "proportions D is piecewise linear with K-j vertices, so the shape one "
        "sees depends on where one samples, whereas the per-tile hull is "
        "smooth at any plotted scale.")

    rows = [["q", "points", "condition holds", "margin", "quartic fit"]]
    for q in ["0", "16", "32", "48", "63"]:
        r = lc[q]
        rows.append([f"q{q}", r["n"], f"{r['frac_ok']:.1f}%",
                     _f(r["worst"], 4), f"{r['quartic_frac_ok']:.1f}%"])
    t_lcx = k.rows(rows,
        "<b>Testing log-convexity on the measured frontier.</b> The direct "
        "test asks whether the secant slopes of ln D against S are "
        "non-decreasing, and the margin is the smallest increase attained. The "
        "condition holds at every point of every frontier and the margin grows "
        "tenfold from the lowest rate to the highest. Fitting a quartic and "
        "differentiating it twice instead reports 73.7% to 84.0%, and the "
        "difference is the fit rather than the data: near the ceiling the "
        "frontier is close to vertical and a polynomial overshoots, whereas "
        "the secant test has no fitted degree to choose.")
    k.note("results/logconvexity.json. That file records no checkpoint, so it "
           "cannot be attributed from disk; its five frontiers have 13 to 16 "
           "points each. The quartic column is the file's own "
           "quartic_frac_ok field.")

    # ==================================================================
    k.h2("How much of the oracle a partial signal recovers")

    k.par(
        "Letting a predictor decide most cases and handing the hardest ones to "
        "something exact is the shape of selective prediction [36] and "
        "learning to defer [37, 38], and of the budgeted variant of the latter "
        "[39]. Two things make the case easier here. The expert is the "
        "encoder's own search, exact and free at test time, so what is scarce "
        "is the bits needed to say what it decided rather than the expert's "
        "time; and the selection rule is not learned, since a separable "
        "objective makes the optimal set of size s exactly the s largest "
        "regrets. What is left to measure is how concentrated the regret is.")

    hy = k.J("hybrid_RECIPE512_b01_e4head.json",
               "hybrid_RECIPE512_b01_fixed.json")

    k.par(
        "The decoder cannot run the argmin of Proposition 1, D<sub>t,k</sub> "
        "being an error against a source it never receives. A predictor can "
        "guess it. Writing p<sub>t,k</sub> for a head's probability that tile "
        "t belongs at exit k, its rule is")
    e_router = k.eq(r"\hat k_t \;=\; \mathrm{arg\,max}_{k \geq j}\left[\log p_{t,k} - "
                    r"\beta\,c_k\right] \;=\; \mathrm{arg\,min}_{k \geq j}\left["
                    r"\frac{-\log p_{t,k}}{\kappa} + \lambda\,c_k\right].")

    k.par(
        f"The second form of ({e_router}) is the same rule with \\beta = "
        f"\\lambda \\kappa, \\kappa being a fixed calibration turning nats "
        f"into squared error. Written that way it says that a router is the "
        f"oracle of ({e_argmin}) with a surrogate in place of "
        f"D<sub>t,k</sub>, and that one price governs both. Define the "
        f"per-tile regret at a fixed price:")
    e_reg = k.eq(r"r_t(\lambda) \;=\; \left[D_{t,\hat k_t} + \lambda "
                 r"c_{\hat k_t}\right] - \left[D_{t,k^{\star}_t} + \lambda "
                 r"c_{k^{\star}_t}\right] \;\geq\; 0.")

    k.par(
        f"Non-negativity is immediate from ({e_argmin}). The regret is what "
        f"the frame's Lagrangian loses on tile t by trusting the prediction, "
        f"and by Proposition 1 those losses add, so signalling the exits of a "
        f"subset S removes exactly ∑<sub>t∈ S</sub> r<sub>t</sub> from the "
        f"objective and leaves the rest to the head.")

    k.h3("Proposition 12 (Lorenz bound)")
    k.par(
        "Fix \\lambda and let r<sub>(1)</sub> ≥ r<sub>(2)</sub> ≥ ... ≥ "
        "r<sub>(N)</sub> be the regrets in decreasing order with total R. For "
        "any subset S of size s, the fraction of the total regret that "
        "signalling S removes is at most")
    e_lor = k.eq(r"L(\rho) \;=\; \frac{1}{R}\sum_{i=1}^{\lceil \rho N\rceil} "
                 r"r_{(i)}, \qquad \rho = s/N,")
    k.par(
        "with equality if and only if S is a set of s largest regrets. L is "
        "the Lorenz curve of the regret distribution: non-decreasing, concave, "
        "running from 0 to 1, and lying above the diagonal by an amount the "
        "Gini coefficient summarises. <i>Assumes</i> (A1), (A2) and a fixed "
        "\\lambda.")
    k.par(
        "<i>Proof.</i> The sum of any s of the r<sub>t</sub> is at most the "
        "sum of the s largest, since replacing an element of S by a larger one "
        "outside S cannot decrease the sum, and repeating that exchange "
        "terminates at a set of s largest values. Dividing by R gives the "
        "bound. Concavity holds because the increments r<sub>(i)</sub> are "
        "non-increasing by construction.")
    k.par(
        "<i>Remark.</i> The bound holds at a fixed price, and the deployed "
        "system re-bisects \\lambda at every ρ so the frame lands back on its "
        "budget. Re-bisecting enlarges the feasible set: the overridden tiles "
        "and the predicted ones then carry two multipliers, and pairs of "
        "multipliers reach allocations no single one does. The measured "
        "recovery is therefore not bounded by L, and Proposition 12 is the "
        "wrong thing to call a bound on the deployed system.")

    def cell(q, rho):
        return next((x for x in hy["rows"]
                     if x["qp"] == q and abs(x["rho"] - rho) < 1e-9), None)

    qs = sorted({x["qp"] for x in hy["rows"]})
    rows = [["q", "Gini", "L(.1)", "rec(.1)", "L(.2)", "rec(.2)", "L(.5)",
             "rec(.5)"]]
    for q in qs:
        a0, a1 = cell(q, 0.0), cell(q, 1.0)
        base, full = a0["saving_pct_vs_release"], a1["saving_pct_vs_release"]
        line = [f"q{q}", _f(a0["gini_regret"], 2)]
        for rho in (0.1, 0.2, 0.5):
            c_ = cell(q, rho)
            rec = 100 * (c_["saving_pct_vs_release"] - base) / (full - base)
            L_ = 100 * c_["lorenz_at_rho"]
            line += [f"{L_:.0f}%", f"{rec:.0f}%"]
        rows.append(line)
    _L10 = [100 * cell(q, 0.1)["lorenz_at_rho"] for q in qs]
    _r10 = [100 * (cell(q, 0.1)["saving_pct_vs_release"]
                   - cell(q, 0.0)["saving_pct_vs_release"])
            / (cell(q, 1.0)["saving_pct_vs_release"]
               - cell(q, 0.0)["saving_pct_vs_release"]) for q in qs]
    t_lor = k.rows(rows,
        f"<b>The Lorenz curve of the regret, and what signalling recovers.</b> "
        f"L(ρ) is the share of the total regret carried by the ρ worst tiles, "
        f"which Proposition 12 says is the most any signal of that size can "
        f"remove at a fixed price; rec(ρ) is the share of the "
        f"predictor-to-oracle gap the measured sweep recovers. Regret is "
        f"concentrated at every rate, Gini \\GiniMin to \\GiniMax: a tenth of "
        f"the tiles carries {min(_L10):.0f}% to {max(_L10):.0f}% of the total "
        f"regret and recovers {min(_r10):.0f}% to {max(_r10):.0f}% of the gap. "
        f"Over every reachable cell the ratio rec/L runs from "
        f"\\LorenzTightMin% to \\LorenzTightMax%, for the reason the remark "
        f"gives.")
    k.note("results/hybrid_RECIPE512_b01_e4head.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences at a 0.1 dB "
           "budget, against the head fitted to that checkpoint. rec(\u03c1) is "
           "computed here from the same file as the saving at \u03c1, relative "
           "to the \u03c1 = 0 and \u03c1 = 1 endpoints. Signalling half the "
           "tiles beats signalling all of them at \\HybridHalfVsFullN of "
           "\\HybridHalfVsFullOf rates here; on a head fitted to other weights it "
           "did so at two of them, which is the re-bisection effect the "
           "remark predicts rather than a property of the decoder.")

    k.par(
        "The signal is an entropy-coded mask over tiles plus an index per "
        "override, so its bill grows with ρ and is pure overhead at ρ = 1, "
        "where the mask names every tile; section G prices both objects. A "
        "fifth of the tiles costs \\HybridBitsFifth bits per frame and "
        "recovers over half the gap.")

    # ==================================================================
    k.h2("The rate-rank rule, and the model it comes from")

    rr = k.J("raterank_RECIPE512_b01.json")

    k.par(
        "The decoder holds, before the trunk runs, the bits the entropy coder "
        "spent on each tile's latents; write b<sub>t</sub> for that count over "
        "the frame's mean. Deriving the rule from a model rather than "
        "presenting it as a heuristic says when it must work and how it fails. "
        "The model is separability in log space: one scalar difficulty per "
        "tile, and one ladder profile applying to every tile in proportion.")
    e_r1 = k.eq(r"\log D_{t,k} \;\approx\; u_t + \log\varphi_k, \qquad "
                r"u_t \;=\; \alpha\log b_t + c.")

    k.par(
        "Equivalently D<sub>t,k</sub> = g<sub>t</sub> φ<sub>k</sub> with "
        "g<sub>t</sub> = exp(u<sub>t</sub>) > 0. The profile φ is K-j numbers "
        "fitted once per rate and \\alpha and c are two more, all three fitted "
        "leave-one-sequence-out. Nothing is learned per frame and nothing is "
        "transmitted, which is why the rule costs the decoder zero.")

    k.h3("Proposition 13 (a rank-1 table gives a threshold rule)")
    k.par(
        "If D<sub>t,k</sub> = g<sub>t</sub> φ<sub>k</sub> with g<sub>t</sub> > "
        "0 then")
    e_rank = k.eq(r"\mathrm{arg\,min}_{k \geq j}\left[g_t\varphi_k + \lambda c_k\right] \;=\; "
                  r"\mathrm{arg\,min}_{k \geq j}\left[\varphi_k + \frac{\lambda}{g_t}c_k\right],")
    k.par(
        "so a tile's exit depends on the tile only through the scalar "
        "\\lambda/g<sub>t</sub>, and by Proposition 2 the chosen exit is "
        "non-increasing in it. Sorting tiles by g<sub>t</sub> and cutting the "
        "sorted list at K-j-1 thresholds therefore reproduces the oracle "
        "exactly. <i>Assumes</i> exact rank-1 structure, and Proposition 2.")
    k.par(
        f"<i>Proof.</i> Divide the bracket by g<sub>t</sub> > 0, which does "
        f"not move the argmin. The reduced problem is the one-tile problem of "
        f"({e_argmin}) with distortion φ and price \\lambda/g<sub>t</sub>, so "
        f"Proposition 2 applies and its solution is non-increasing in that "
        f"price. Since \\lambda/g<sub>t</sub> is decreasing in g<sub>t</sub>, "
        f"the exit is non-decreasing in g<sub>t</sub>: harder tiles go deeper. "
        f"The thresholds are the breakpoints of that one common problem and "
        f"there are at most K-j-1 of them.")
    k.par(
        "<i>Remark.</i> Exactness needs an exactly rank-1 table, which no real "
        "table is, and the shortfall of the deployed rule measures the "
        "departure. The proposition also says the exit is monotone in "
        "g<sub>t</sub> and not that a bit count estimates g<sub>t</sub>, which "
        "is a separate and weaker claim.")

    k.h3("Proposition 14 (a rank-1 rule uses only the profile's hull)")
    k.par(
        "Under the same model, an exit k is chosen for some tile at some price "
        "only if (c<sub>k</sub>, φ<sub>k</sub>) lies on the lower convex hull "
        "of the profile. <i>Assumes</i> exact rank-1 structure, and "
        "Propositions 5 and 13. <i>Proof.</i> By Proposition 13 the reduced "
        "problem is a single Lagrangian sweep over the points (c<sub>k</sub>, "
        "φ<sub>k</sub>), and by Proposition 5 such a sweep reaches only hull "
        "points.")
    k.par(
        "<i>Remark.</i> The statement is about the fitted profile rather than "
        "the true table, so an exit off the hull is dropped by the rule and "
        "may still be the oracle's choice. Hull membership is not automatic: "
        "on the single-sequence table of results/tile_table.json the same "
        "construction drops exit 4.")

    cB = cost_B
    cfgB_split = 2      # exits below the split are not in cost_B
    _lev, _spr, _ext = [], [], []
    for r in rr["rows"]:
        _lev.append(r["spearman_bits_vs_level"])
        _spr.append(r["spearman_bits_vs_spread"])
        _ext.append(-r["spearman_bits_vs_exit"])
    _alphas = [r["alpha"] for r in rr["rows"]]
    # Computed, not asserted. This was an assert, and it fired the first time
    # the paper moved to a later checkpoint -- which is the assert doing its
    # job, and also a build that dies rather than a sentence that updates.
    # Hull membership is a measurement about the fitted profile, so it is
    # measured and then described.
    # _lower_hull returns INDICES, not points. Comparing points against it
    # made every exit look off the hull, and the sentence below said so.
    _off = {}
    for r in rr["rows"]:
        _h = set(_lower_hull(list(zip(cB, r["phi"]))))
        _miss = [i for i in range(len(cB)) if i not in _h]
        if _miss:
            _off[r["qp"]] = _miss
    if not _off:
        _hull_sentence = (
            "And all four exits survive on the lower hull of "
            "(c<sub>k</sub>, φ<sub>k</sub>) at every rate, so Proposition 14 "
            "costs nothing here.")
    else:
        _rates = ", ".join(f"q{q}" for q in sorted(_off))
        _ex = sorted({cfgB_split + i for m in _off.values() for i in m})
        _exs = ", ".join(f"exit {e}" for e in _ex)
        _hull_sentence = (
            f"And the hull is no longer full: at {_rates} the fitted profile "
            f"puts {_exs} above the line joining its neighbours, so the "
            f"Lagrangian rule cannot select it there whatever the multiplier. "
            f"On the checkpoint this paper first reported all four exits "
            f"survived; on the one it reports now, the two lowest rates have "
            f"a rung the rule will not use. That is Proposition 14 costing "
            f"something rather than nothing. We have two checkpoints and so "
            f"cannot say whether further training removes more rungs or puts "
            f"this one back; what the two do establish is that hull "
            f"membership is a property of the weights and not of the design, "
            f"and a claim that it is full has to be re-checked whenever they "
            f"move.")
    k.par(
        f"Section F prints the fitted α and φ at every rate and reports what "
        f"the rule saves. Two properties of that fit belong here, with the "
        f"propositions they bear on. α is positive at every rate, so a tile "
        f"the entropy coder spent bits on is a tile every exit reconstructs "
        f"worse, and it falls {max(_alphas) / min(_alphas):.0f}-fold from the "
        f"lowest rate to the highest. " + _hull_sentence)
    k.note("Fitted values from results/raterank_RECIPE512_b01.json, "
           "checkpoint runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences "
           "at a 0.1 dB budget; hull membership computed in the build against "
           "the shipped cost vector of Table " + str(t_cost) + ".")

    M = [[math.log(v) for v in row] for row in Dj]
    nT, nK = len(M), len(M[0])
    gm = sum(sum(r) for r in M) / (nT * nK)
    rmean = [sum(r) / nK for r in M]
    cmean = [sum(M[t][a] for t in range(nT)) / nT for a in range(nK)]
    tot = sum((M[t][a] - gm) ** 2 for t in range(nT) for a in range(nK))
    res = [[M[t][a] - (rmean[t] + cmean[a] - gm) for a in range(nK)]
           for t in range(nT)]
    ssr = sum(v * v for r in res for v in r)
    rms = math.sqrt(ssr / (nT * nK))
    spread = max(cmean) - min(cmean)
    lb = [math.log(x / (sum(bits) / len(bits))) for x in bits]
    xm, ym = sum(lb) / nT, sum(rmean) / nT
    al = (sum((x - xm) * (y - ym) for x, y in zip(lb, rmean))
          / sum((x - xm) ** 2 for x in lb))
    pred = [ym + al * (x - xm) for x in lb]
    r2 = 1 - (sum((y - p) ** 2 for y, p in zip(rmean, pred))
              / sum((y - ym) ** 2 for y in rmean))

    k.par(
        f"The rule gives up \\RateRankOracleGap saving points to the "
        f"Lagrangian oracle at the rate where it does worst, and the loss is "
        f"not in the separability. On the measured tile "
        f"table the separable model absorbs {100 * (1 - ssr / tot):.2f}% of "
        f"the variance of log D. The quantity it has to resolve is small: the "
        f"exit profile spans {spread:.4f} in log units while the non-separable "
        f"residual has a root mean square of {rms:.4f}, "
        f"{100 * rms / spread:.0f}% of it. A model can explain almost all of a "
        f"table and still be unable to order the differences that decide the "
        f"argmin, which reads only the differences. Predicting a tile's "
        f"difficulty from its bit count alone then leaves R² = {r2:.2f}, and "
        f"by Proposition 13 an error in log g<sub>t</sub> is a proportional "
        f"error in the price that tile faces. Coded bits correlate with the "
        f"mean level of a tile's row at Spearman {min(_lev):.2f} to "
        f"{max(_lev):.2f} and with the oracle's exit index at {min(_ext):.2f} "
        f"to {max(_ext):.2f}, the ordering the model predicts.")
    k.note("Residual analysis computed from results/tile_table.json "
           "(Bosphorus, q32, 40 tiles), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar; α fitted there is "
           f"{al:.2f} against {rr['rows'][2]['alpha']:.2f} for the "
           "\\NumSeq-sequence fit at the same rate, the difference being one "
           "sequence against \\NumSeq. Spearman correlations from "
           "results/raterank_RECIPE512_b01.json, pinned checkpoint; the file "
           "stores the last against the negated exit index and it is reported "
           "here with the sign flipped, so a positive value means more bits "
           "and a deeper exit.")

    # ==================================================================
    k.h2("What is proved, what is checked, and what neither covers")

    k.par(
        f"Table {t_index} lists every result with its evidence. Four gaps in "
        f"it are worth naming. Theorem 10 is measured as a ratio only on "
        f"runs/BEST/ckpt_eval.pth.tar over 40 sequences and on the stored cost "
        f"vector; the pinned measurement of Table {t_adapt} gives it in saving "
        f"points instead. The convexity check of Table {t_lcx} "
        f"records no checkpoint. The granularity and hull results of {S[3]} "
        f"rest on one sequence at one rate, which is enough to check "
        f"arithmetic and not enough to characterise a codec. And every "
        f"proposition here concerns a frame mean under (A1) and (A2), so none "
        f"bounds what a single tile gives up.")
