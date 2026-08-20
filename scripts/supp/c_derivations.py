"""Derivations: the mathematics of the allocation, stated and proved.

The LaTeX twin of this module is paper/supp/c_derivations.tex. The two carry
the same claims, the same numbers and the same order.

Everything numeric here is computed inside content() from a file in results/,
so the build's provenance list is the record of what this section rests on and
no digit is typed twice. Where a statement is checked numerically rather than
proved, the check names its file and its checkpoint.

The order is deliberate. Objects first (tile, exit, distortion, reference,
compute), then the price on compute and what moves as it moves, then the
argmin, then the achievable set, then what the sweep can and cannot reach,
then the value of adaptivity, then the two rules that stand in for the oracle
when the decoder has to guess.
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


def _spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra)
                    * sum((y - mb) ** 2 for y in rb))
    return num / den


# ------------------------------------------------------------------ section
def content(k):
    L = k.h1("Derivations")
    # Subsection labels are formed from the letter this section was given,
    # so a cross-reference stays right when a section is inserted before it.
    S = [f"{L}.{i}" for i in range(0, 14)]

    k.par(
        "This section states the allocation problem from its objects upward and "
        "proves what the paper asserts about it. Nothing here needs a result "
        "from elsewhere in the supplement, and a reader who has not met a "
        "Lagrangian rate-distortion argument before should be able to follow "
        "every line. Each statement is either proved in place or marked as "
        "checked numerically, in which case the file that checks it and the "
        "checkpoint it was measured on are named at that point.")

    # ==================================================================
    k.h2("Tiles, exits, and what is measured against what")

    td = k.J("tile_definition.json")
    ac = k.J("adapter_cost.json")

    k.par(
        "<b>A tile.</b> The decoder is cut in two at a fixed depth. Everything "
        "above the cut runs once over the whole frame; everything below it runs "
        "separately on square patches of the trunk's feature grid. Those "
        "patches are the tiles. The frame is padded on the bottom and the right "
        "before the encoder sees it, by replication, so that the feature grid "
        "divides exactly into patches, and the tiling is then a partition: "
        "every feature position belongs to exactly one tile and no tile "
        "overlaps another. A tile is 32 feature positions on a side, which is "
        "16 latent positions and 256 pixels. We index tiles by t and write N "
        "for their number in a frame.")

    rows = [["frame", "padded to", "latent grid", "tiles", "N"]]
    for r in td["rows"]:
        rows.append([r["name"],
                     f"{r['padded'][0]}×{r['padded'][1]}",
                     f"{r['latent'][0]}×{r['latent'][1]}",
                     f"{r['nh']}×{r['nw']}",
                     r["tiles_measured"]])
    t_geom = k.rows(rows,
        "<b>The tiling, at every resolution measured.</b> N is fixed by the "
        "frame size and the tile side, not chosen per frame. Read it as the "
        "granularity available to the allocation: 40 tiles at 1080p, and only "
        "2 at 416×240, which is why the arguments below that improve as 1/N "
        "improve very little at the smallest class.")
    k.note("results/tile_definition.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. The pad is applied to the RGB "
           "frame before the encoder, by replication, on the bottom and right "
           "only.")

    nex = ac["config"]["num_exits"]
    jj = ac["config"]["split_depth"]
    nblk = max(e["blocks_skipped"] for e in ac["exits"]) + 2
    bpe = nblk // nex

    k.par(
        f"<b>An exit.</b> The per-tile part of the decoder is a stack of "
        f"{nblk} residual blocks. An exit is a point in that stack at which a "
        f"tile may stop and be handed to the reconstruction head. There are "
        f"K = {nex} of them, evenly spaced, so exit k runs (k+1)b blocks with "
        f"b = {bpe}. The split depth is j = {jj}: blocks 0 to jb-1 run "
        f"full-frame for every tile whatever exit it takes, so exits below j "
        f"are not distinguishable from exit j and the usable ladder has "
        f"K - j = {nex - jj} distinct members. Throughout, k ranges over "
        f"j, ..., K-1 unless stated otherwise. Because a tile leaving early "
        f"hands the head a feature the head was not trained to read, each "
        f"exit below the deepest carries a small learned adapter, "
        f"zero-initialised so that the ladder starts life as the released "
        f"decoder [31, 32].")

    k.par(
        "<b>Distortion, and the reference it is measured against.</b> D<sub>t,k"
        "</sub> is the mean squared error tile t incurs when it leaves at exit "
        "k. Two choices in that definition do real work. The error is measured "
        "on the path actually deployed, a tiled decode with replicate padding "
        "at tile borders, so what tiling costs is already inside "
        "D<sub>t,k</sub> rather than being accounted for separately. And it is "
        "measured against the released decoder's full-frame decode of the same "
        "latent, not against the source image. The encoder, the hyperprior and "
        "the entropy model are therefore byte-identical on both sides of the "
        "comparison and synthesis is the only thing that differs. A consequence "
        "worth stating now: D<sub>t,k</sub> = 0 would mean the exit reproduces "
        "the released decoder exactly, and nothing in the construction forces "
        f"the deepest exit to achieve it. Section {S[8]} returns to that.")

    dbc = k.J("db_convention.json")
    _dd = sorted(abs(r["difference_db"]) for r in dbc["rows"])
    k.par(
        "<b>Decibels, and which convention.</b> Distortion enters the "
        "optimisation as a mean squared error and is reported as "
        "10 log<sub>10</sub> of the ratio to the reference. The two are not "
        "interchangeable, because the ratio can be formed at two levels. "
        "Pooling every tile of every frame into one mean squared error and "
        "taking one logarithm is the form the Lagrangian below is exact for. "
        "Averaging a per-frame decibel instead is the codec convention and is "
        "what the budget is bisected against in every measurement this project "
        f"reports. On an identical allocation the two read {_dd[0]:.3f} to "
        f"{_dd[-1]:.3f} dB apart, which is {100 * _dd[0] / 0.1:.0f}% to "
        f"{100 * _dd[-1] / 0.1:.0f}% of the headline 0.1 dB budget, so a "
        f"decibel figure is meaningless without its convention.")
    k.note("Convention gap from results/db_convention.json, checkpoint "
           "runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar, 40 frames. That is an "
           "old checkpoint for a claim about arithmetic rather than about a "
           "model, but it is the only file in results/ that measures both "
           "conventions on one allocation.")

    # ==================================================================
    k.h2("Compute, and what an exit costs")

    st = k.J("static_RECIPE512_b01.json")
    sat = k.J("saturation_RECIPE512_ctc53.json")

    k.par(
        "<b>The unit.</b> Compute is counted in multiply-accumulate operations "
        "and then divided by the multiply-accumulates of one full-frame decode "
        "by the released decoder, \\IntraGmac\\,GMAC at 1080p. Everything below "
        "is in those units, so c = 1 means one released decode and a saving of "
        "S% means the frame was decoded for (1 - S/100) of one. Working in a "
        "ratio rather than in GMAC is what makes the cost of an exit "
        "resolution-independent: every operator in this decoder runs at a fixed "
        "multiple of the latent grid, so the shares below do not move with "
        "frame size.")
    k.note("\\IntraGmac\\,GMAC is produced by scripts/mac_audit.py, which "
           "prints and writes no JSON, so it is the one quantity in this "
           "section with no file in results/ behind it. It enters only as a "
           "normaliser and cancels from every ratio quoted here.")

    e_c = k.eq(r"c_k \;=\; s_{\mathrm{up}} \;+\; s_{\mathrm{trunk}}\,"
               r"\frac{(k+1)\,b}{N_{\mathrm{blk}}} \;+\; s_{\mathrm{head}} "
               r"\;+\; a_k \;+\; r")

    k.par(
        f"Equation ({e_c}) is the cost of one tile at exit k, and it is worth "
        f"reading term by term. s<sub>up</sub> = 0.0816 is the opening "
        f"upsample, which runs whatever happens. The middle term is the trunk: "
        f"s<sub>trunk</sub> = 0.8944 spread evenly over N<sub>blk</sub> = "
        f"{nblk} blocks, of which exit k runs (k+1)b. s<sub>head</sub> = "
        f"0.0240 is the reconstruction head, which also runs whatever happens. "
        f"a<sub>k</sub> is the adapter that exit k wears and r = 0.0095 is the "
        f"full-frame deblocking filter that repairs tile borders after the "
        f"tiles are stitched. Only the trunk term depends on k. That single "
        f"fact is the reason a ceiling exists at all: however shallow the "
        f"ladder gets, the stem, the head and the filter are still paid.")

    k.par(
        "Two of those shares are worth pinning down. The adapter share "
        "a<sub>k</sub> is measured with forward hooks on the real modules: one "
        "DepthConvBlock costs "
        f"{ac['block']['mac_per_px']:,.0f} MAC/px, the 1×1 adapter "
        f"{ac['adapters']['conv1x1']['mac_per_px']:,.0f} or "
        f"{ac['adapters']['conv1x1']['blocks']:.3f} of a block, and the FFN "
        f"adapter {ac['adapters']['ffn']['mac_per_px']:,.0f} or "
        f"{ac['adapters']['ffn']['blocks']:.3f} of a block. The exits skipping "
        "four or more blocks get the FFN and the rest the 1×1, and the deepest "
        "exit carries none, which is what makes it bit-identical to the "
        "released decoder in structure. The three whole-decoder shares are the "
        "output of the MAC audit and are not in results/, but they are not free "
        "parameters either: 0.0816 + 0.8944 + 0.0240 + 0.0095 = 1.0095072, "
        "which is exactly the deepest-exit cost recorded in every configuration "
        "file this section reads.")

    # cost vector: from the uniform rows of the static baseline
    uni = st["rows"][0]["uniform"]
    cost_B = [1 - u["saving"] / 100.0 for u in uni]
    exits_B = [u["exit"] for u in uni]
    tt = k.J("tile_table.json")
    cost_A = tt["cost"][tt["j"]:]
    adapter_name = {e["exit"]: (e["adapter_kind"] or "none") for e in ac["exits"]}
    adapter_blk = {e["exit"]: e["adapter_blocks"] for e in ac["exits"]}

    rows = [["exit k", "blocks", "adapter", "share", "c_k", "100(1-c_k)",
             "earlier c_k"]]
    for i, kk in enumerate(exits_B):
        rows.append([kk, (kk + 1) * bpe,
                     adapter_name[kk].replace("conv1x1", "1×1"),
                     _f(adapter_blk[kk], 3), _f(cost_B[i], 4),
                     _f(100 * (1 - cost_B[i]), 2), _f(cost_A[i], 4)])
    t_cost = k.rows(rows,
        "<b>What each exit costs</b>, in units of one released full-frame "
        "decode. The deepest exit costs more than 1 because a tiled decode "
        "still pays the deblocking filter and the tile borders. The last "
        "column is the earlier cost vector, built when the FFN adapter was "
        "billed at 2C² instead of its measured 5C²; the numerical checks in "
        f"{S[7]}, {S[9]} and {S[12]} were run against it and are labelled where they "
        "appear.")
    k.note("c_k and 100(1-c_k) from the uniform-depth rows of "
           "results/static_RECIPE512_b01.json, and c_j independently from "
           "results/saturation_RECIPE512_ctc53.json, both on "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. Adapter shares from "
           "results/adapter_cost.json, same checkpoint. Earlier vector from "
           "results/tile_table.json. The complexity section of this supplement "
           "audits the block count itself.")

    k.par(
        f"The four usable exits therefore cost {_f(cost_B[0], 4)}, "
        f"{_f(cost_B[1], 4)}, {_f(cost_B[2], 4)} and {_f(cost_B[3], 4)} of a "
        f"released decode. The gaps between them are what the whole allocation "
        f"trades in, and they are unequal: "
        f"{_f(cost_B[1] - cost_B[0], 4)}, {_f(cost_B[2] - cost_B[1], 4)} and "
        f"{_f(cost_B[3] - cost_B[2], 4)}. The first gap is the largest because "
        f"exit 3 gives up the FFN adapter's saving as well as two blocks.")

    # ==================================================================
    k.h2("An assignment, and the two things assumed about it")

    ntile = td["rows"][0]["tiles_measured"]
    k.par(
        f"An <i>assignment</i> is a map a from tiles to exits, a(t) being the "
        f"exit tile t takes. There are (K-j)<super>N</super> of them, which is "
        f"{nex - jj}<super>{ntile}</super> at 1080p, so they are not going to "
        f"be enumerated. "
        "The frame's compute and the frame's distortion are")
    e_CD = k.eq(r"C(a) \;=\; \frac{1}{N}\sum_{t=1}^{N} c_{a(t)}, \qquad "
                r"D(a) \;=\; \frac{1}{N}\sum_{t=1}^{N} D_{t,a(t)}")

    k.par(
        f"Equation ({e_CD}) carries two assumptions and it is worth being "
        f"explicit about both, because everything after this rests on them and "
        f"neither is a mathematical necessity. The first is that compute is "
        f"additive over tiles: the frame costs the mean of what its tiles cost. "
        f"That is exact when tiles are decoded independently, which is what a "
        f"tiled decoder does, and it is why the stem and the head appear inside "
        f"c<sub>k</sub> rather than as separate terms. The second is that "
        f"distortion is a tile mean, which is exact for any per-pixel loss "
        f"averaged over a frame provided a tile's error does not depend on "
        f"which exits its neighbours took. That second proviso is not free: the "
        f"deblocking filter runs on the stitched canvas and therefore sees the "
        f"whole exit map at once.")

    # additivity check on the sweep
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
        f"So the second assumption is checked rather than asserted. The "
        f"per-tile table and the measured decode of the corresponding mixed "
        f"map are both recorded for {len(tt['sweep'])} allocations of one "
        f"frame, spanning savings from "
        f"{min(s['saving'] for s in tt['sweep']):.2f}% to "
        f"{max(s['saving'] for s in tt['sweep']):.2f}%. Predicting the frame's "
        f"decibels from the tile means and comparing with the decode, the "
        f"largest disagreement over all {len(errs)} allocations is "
        f"{max(errs) * 1e5:.1f}×10<super>-5</super> dB, which is four orders "
        f"of magnitude below the "
        f"budget. Neither assumption constrains D<sub>t,k</sub> itself. In "
        f"particular nothing requires D<sub>t,k</sub> to fall as k grows, and "
        f"Table {t_cost + 2} below contains a tile for which it does not.")
    k.note("Additivity check computed from results/tile_table.json "
           "(Bosphorus, q32, 40 tiles), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar. That checkpoint has since been "
           "overwritten by the per-epoch watcher; the arithmetic it checks is "
           "a property of the decode path rather than of the weights.")

    # ==================================================================
    k.h2("A price on compute")

    k.par(
        "The problem the encoder faces is to spend as little compute as "
        "possible while staying inside a distortion budget:")
    e_con = k.eq(r"\min_{a}\; C(a) \quad \mathrm{subject\ to} \quad "
                 r"D(a) \;\leq\; D_{\mathrm{budget}}")

    k.par(
        f"Constraint ({e_con}) couples the tiles. A tile cannot be given an "
        f"exit on its own merits, because whether it can afford to leave early "
        f"depends on how much of the budget the other {int(td['rows'][0]['tiles_measured']) - 1} "
        f"tiles have used. The standard way out is to buy the constraint off "
        f"with a price. Introduce a number \\lambda ≥ 0, charge \\lambda "
        f"per unit of compute, and minimise the total bill:")
    e_L = k.eq(r"L(a,\lambda) \;=\; D(a) + \lambda\,C(a) \;=\; "
               r"\frac{1}{N}\sum_{t=1}^{N}\left[\,D_{t,a(t)} + "
               r"\lambda\,c_{a(t)}\,\right]")

    k.par(
        f"It is worth being slow about what \\lambda is. It is an exchange "
        f"rate. Its units are squared error per unit of relative compute, so "
        f"\\lambda c<sub>k</sub> converts the compute an exit uses into the "
        f"squared error that compute is deemed to be worth. In the numbers "
        f"below \\lambda runs from about 2×10<super>-7</super> to about "
        f"1×10<super>-4</super>, against squared errors of order "
        f"1×10<super>-4</super>, because the reference images are on a 0 to 1 "
        f"scale. Nothing depends on the scale itself; what matters is where "
        f"\\lambda sits relative to the differences between exits.")

    k.par(
        "Two extremes fix the picture. At \\lambda = 0 compute is free, the "
        "second term vanishes, and every tile takes whichever exit has the "
        "smallest error, whatever it costs. As \\lambda grows, the compute term "
        "grows for every exit but grows fastest for the deepest one, because "
        "c<sub>k</sub> is largest there. A tile therefore drops to a shallower "
        "exit at the moment the compute that exit saves, times the price, "
        "exceeds the error it adds. Past a large enough \\lambda every tile has "
        "dropped as far as the ladder allows. Moving \\lambda from 0 upward "
        "sweeps the decoder from most accurate and most expensive to cheapest "
        f"and worst, and does so without ever turning back, which {S[4]} proves.")

    k.h3("The Lagrangian separates")

    k.par(
        f"Nothing so far has removed the coupling; equation ({e_L}) is still a "
        f"minimisation over (K-j)<super>N</super> assignments. What removes it "
        f"is that the sum has no cross terms.")

    k.par(
        "<b>Proposition 1 (separation).</b> For every \\lambda ≥ 0, an "
        "assignment minimises L(a,\\lambda) over all assignments if and only if "
        "it minimises the bracket tile by tile:")
    e_argmin = k.eq(r"k^{\star}_t(\lambda) \;\in\; \arg\min_{k\,\geq\,j}\;"
                    r"\left[\,D_{t,k} + \lambda\,c_k\,\right]")

    k.par(
        "<i>Proof.</i> Write f<sub>t</sub>(k) = D<sub>t,k</sub> + \\lambda "
        "c<sub>k</sub>. Then N L(a,\\lambda) = ∑<sub>t</sub> "
        "f<sub>t</sub>(a(t)), a sum in which the t-th term depends on a only "
        "through a(t). For any assignment a, term by term f<sub>t</sub>(a(t)) "
        "≥ min<sub>k</sub> f<sub>t</sub>(k), so N L(a,\\lambda) ≥ "
        "∑<sub>t</sub> min<sub>k</sub> f<sub>t</sub>(k), and the "
        "right-hand side is attained by any a that picks a minimiser in every "
        "tile. Conversely if a is optimal and some tile t had f<sub>t</sub>"
        "(a(t)) > min<sub>k</sub> f<sub>t</sub>(k), replacing a(t) by that "
        "minimiser and leaving every other tile alone would strictly lower L, "
        "contradicting optimality.")

    k.par(
        f"The N-dimensional search has become N independent searches over "
        f"{nex - jj} numbers each. Ties are broken toward the shallower exit "
        f"throughout, which makes k<super>*</super> a function rather than "
        f"a set-valued map and costs nothing, because a tie means the two "
        f"choices have identical Lagrangian value. We call k<super>*</super>"
        f"<sub>t</sub>(\\lambda) the <i>oracle</i>'s choice, because it uses "
        f"the true D<sub>t,k</sub>, which only the encoder has. The "
        f"construction is Shoham and Gersho's [28], with compute in place of "
        f"rate, and Ortega and Ramchandran [29] made it standard in image and "
        f"video coding.")

    k.h3("The sweep is monotone, so bisection finds the budget")

    k.par(
        "<b>Proposition 2 (monotonicity).</b> Let \\lambda<sub>1</sub> < "
        "\\lambda<sub>2</sub> and let a<sub>1</sub>, a<sub>2</sub> be the "
        "corresponding oracle assignments. Then C(a<sub>2</sub>) ≤ "
        "C(a<sub>1</sub>) and D(a<sub>2</sub>) ≥ D(a<sub>1</sub>).")

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
        "≥ D<sub>1</sub> + \\lambda<sub>1</sub>(c<sub>1</sub> - "
        "c<sub>2</sub>) ≥ D<sub>1</sub>. Averaging over tiles preserves "
        "both.")

    k.par(
        "This is the standard exchange argument and it is what makes the "
        "budget searchable. D(a<sup>*</sup><sub>\\lambda</sub>) is "
        "non-decreasing in \\lambda, so bisecting \\lambda against the measured "
        "decibels converges on the largest price whose allocation still meets "
        "the budget, and by Proposition 2 that is also the cheapest such "
        "allocation the family contains. Every \\lambda quoted in this project "
        "is found that way, and so is the router's \\beta. Monotonicity is "
        "also checked numerically, over 6001 multipliers on a measured tile "
        "table, in scripts/verify_theory.py.")

    # ==================================================================
    k.h2("The sweep in one frame")

    bits = tt["bits_per_tile"]
    ctt = tt["cost"][jt:]
    Dj = [row[jt:] for row in Dtt]
    order = sorted(range(len(bits)), key=lambda i: bits[i])
    sel = [order[0], order[len(order) // 2], order[-1]]

    def schedule(t):
        """The exits tile t takes as lambda rises, and where it switches."""
        out = []
        prev = None
        lam = 0.0
        grid = [0.0] + [1e-9 * (10 ** (7 * i / 4000.0)) for i in range(4001)]
        for lam in grid:
            vals = [Dj[t][a] + lam * ctt[a] for a in range(len(ctt))]
            kk = min(range(len(vals)), key=lambda a: vals[a])
            if kk != prev:
                out.append((lam, jt + kk))
                prev = kk
        return out

    rows = [["tile", "bits", "D(2)", "D(3)", "D(4)", "D(5)", "exits as λ rises"]]
    for t in sel:
        sc = schedule(t)
        if len(sc) == 1:
            desc = f"{sc[0][1]} throughout"
        else:
            desc = str(sc[0][1]) + "".join(
                f", {kk} at {lam * 1e6:.1f}" for lam, kk in sc[1:])
        rows.append([t, f"{bits[t]:,.0f}"]
                    + [_f(Dj[t][a] * 1e5, 3) for a in range(4)] + [desc])
    t_tiles = k.rows(rows,
        "<b>Three tiles of one frame, and what the price does to them.</b> D "
        "is in units of 10<super>-5</super> squared error; switch points are "
        "in units of 10<super>-6</super> of λ. The cheapest tile prefers the "
        "shallowest exit at every price, including λ = 0, because its error "
        "does not fall with depth at all: depth is not always worth buying. "
        "The most expensive tile holds its deepest exit five times longer than "
        "the middle one. This is the whole mechanism, and it is why a single "
        "uniform depth cannot be right for both.")
    k.note("Computed from results/tile_table.json (Bosphorus, q32, 40 tiles, "
           "the cost vector recorded in that file), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar. Tiles chosen as the lowest, "
           "median and highest coded bit count, not for their behaviour.")

    # sweep summary
    sw = tt["sweep"]
    picks = [0, 10, 18, 24, 30, 34, 38, 42, 46, 52, 59]
    rows = [["λ", "n at 2", "3", "4", "5", "saved %", "dB"]]
    for i in picks:
        s = sw[i]
        h = {}
        for m in s["map"]:
            h[m] = h.get(m, 0) + 1
        rows.append([f"{s['lam']:.2e}"]
                    + [h.get(kk, 0) for kk in range(jt, jt + 4)]
                    + [_f(s["saving"], 2), _f(s["db"], 4)])
    t_sweep = k.rows(rows,
        "<b>The same frame swept.</b> Each row is one price and the exit "
        "histogram it produces over the frame's 40 tiles. The allocation moves "
        "continuously from almost everything deep to everything shallow, and "
        "both ends are flat: below the first row nothing more is gained and "
        "above the last row nothing more is available. The 60 prices in the "
        f"file produce only 33 distinct savings, which is the granularity {S[7]} "
        "quantifies.")
    k.note("results/tile_table.json, checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar; savings are on the earlier cost "
           "vector of Table " + str(t_cost) + ", where the ceiling is "
           "41.91% rather than \\Ceiling%.")

    # ==================================================================
    k.h2("The achievable set")

    k.par(
        "The pairs (C(a), D(a)) an allocation can produce form a set, and the "
        "shape of that set is what determines both what a sweep can reach and "
        "what content adaptivity is worth. There are two versions of it and "
        "they are genuinely different. Write")
    e_Dbar = k.eq(r"\bar D_k \;=\; \frac{1}{N}\sum_{t=1}^{N} D_{t,k}")
    k.par("for the mean distortion of exit k over the frame. We call it the "
          "<i>exit mean</i>, and it is the only place the tiles are averaged "
          "before anything is chosen.")

    k.par(
        "<b>Proposition 3 (fixed proportions).</b> Suppose tiles are assigned "
        "to exits in proportions p, one probability per exit, independently of "
        "their content. Then the achievable set of expected pairs is exactly "
        "the convex hull of the K-j points whose coordinates are the cost "
        "c<sub>k</sub> and the exit mean at k.")

    k.par(
        "<i>Proof.</i> Under content-independent proportions the expected cost "
        "is ∑<sub>k</sub> p<sub>k</sub> c<sub>k</sub> and, by the tile "
        "mean, the expected distortion is ∑<sub>k</sub> p<sub>k</sub> times "
        "the exit mean at k. The map from p to that pair is affine and its "
        "domain is the probability simplex, whose extreme points are the unit "
        "vectors. An affine image of a simplex is the convex hull of the "
        "images of its vertices, and those images are exactly the cost and "
        "exit mean of each exit.")

    rows = [["q", "exit 2", "exit 3", "exit 4", "exit 5", "best single"]]
    for r in st["rows"]:
        u = {x["exit"]: x for x in r["uniform"]}
        rows.append([f"q{r['qp']}"]
                    + [_f(u[kk]["db"], 4) for kk in exits_B]
                    + [f"{r['best_static']['exit']} at "
                       f"{r['best_static']['saving']:.1f}%"])
    t_vert = k.rows(rows,
        "<b>The vertices of the fixed-proportion set</b>, in dB below the "
        "released decoder, one uniform-depth decode per column. These four "
        "points and their hull are everything a content-blind allocation can "
        "reach. The last column is the shallowest uniform depth that still "
        "meets a 0.1 dB budget, and its saving: at the two highest rates that "
        "is the deepest exit, which saves nothing at all.")
    k.note("results/static_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq frames. The mean over "
           "rates of the best single depth is \\BestStaticMean%, against "
           "\\MainMean% for the per-tile allocation at the same budget.")

    k.par(
        "<b>Theorem 4 (per-tile assignment).</b> The convex hull of the pairs "
        "achievable by per-tile assignment is the Minkowski average of the N "
        "per-tile hulls,")
    e_mink = k.eq(r"\mathcal{A}_{\mathrm{tile}} \;=\; \frac{1}{N}"
                  r"\bigoplus_{t=1}^{N}\mathrm{conv}\{(c_k, D_{t,k}) : "
                  r"k \geq j\}")

    k.par(
        "<i>Proof.</i> By the two assumptions, an achievable pair is the "
        "average of one point drawn from each per-tile set S<sub>t</sub>, the "
        "set of pairs cost c<sub>k</sub> against error D<sub>t,k</sub>, so the "
        "achievable set is the Minkowski average of the S<sub>t</sub>. Taking "
        "convex hulls commutes with Minkowski addition, the hull of a sum "
        "being the sum of the hulls, and with positive scaling; applying that "
        "N-1 times moves the hull inside the sum.")

    k.par(
        f"Proposition 3 is the special case in which every tile has the same "
        f"row, so that each tile's error at k equals the exit mean there. "
        f"Otherwise the per-tile "
        f"set is strictly larger, and the difference is visible in the vertex "
        f"count. The fixed-proportion hull has at most K-j = {nex - jj} "
        f"vertices, so its lower boundary is a chain of {nex - jj - 1} straight "
        f"segments. The Minkowski average carries up to N(K-j-1) = "
        f"{ntile * (nex - jj - 1)} vertices at 1080p, and the sweep behind "
        f"Table "
        f"{t_sweep} finds 33 distinct cost levels on one frame. That is the "
        f"difference between a frontier with three kinks and a frontier that "
        f"looks smooth at any scale one plots, and {S[10]} shows why conflating "
        f"them leads to a confident statement about a shape that is an "
        f"artefact of sampling.")

    # ==================================================================
    k.h2("What the sweep reaches, and what it cannot")

    hg = k.J("hull_gap.json")
    tc = k.J("theory_checks.json")

    k.par(
        "<b>Proposition 5 (the sweep traces the lower convex hull).</b> For "
        "every \\lambda ≥ 0 the point (C, D) produced by the oracle "
        "assignment lies on the lower convex hull of the achievable set, and "
        "no \\lambda produces a point strictly inside it.")

    k.par(
        "<i>Proof.</i> By Proposition 1 the oracle assignment minimises D(a) + "
        "\\lambda C(a) over all assignments, so its pair minimises the linear "
        "functional (C, D) → D + \\lambda C over the achievable set and "
        "hence over that set's convex hull. A minimiser of a linear functional "
        "over a compact convex set lies on its boundary, and because \\lambda "
        "≥ 0 the supporting line has slope -\\lambda ≤ 0, which places "
        "the point on the lower boundary. Conversely a point strictly inside "
        "the hull is a strict convex combination of achievable points and is "
        "therefore strictly beaten, at that same C, by some point of the hull; "
        "it cannot minimise any such functional.")

    k.par(
        "The converse has a practical edge and it cuts both ways. Everett's "
        "generalised multiplier method (1963) supplies the reassuring half: "
        "whatever compute the returned allocation happens to consume, it is "
        "optimal for that level, so a sweep never returns something dominated. "
        "Proposition 5 supplies the other half: a budget whose optimum sits "
        "between two hull vertices is not reachable by any price at all, and "
        "the bisection returns the nearest reachable point below it. The "
        "question is how much that costs, and it can be answered exactly, "
        "because with only four distinct exit costs the whole Pareto set can "
        "be enumerated by dynamic programming over tiles.")

    rows = [["budget dB", "by sweep", "exact Pareto", "gap"]]
    for r in hg["rows"]:
        rows.append([_f(r["budget_db"], 2), _f(r["hull_saving"], 3),
                     _f(r["pareto_saving"], 3), _f(r["gap_pts"], 4)])
    t_hull = k.rows(rows,
        f"<b>What convexity costs.</b> Savings in per cent at seven budgets, "
        f"on one frame. The sweep reaches {hg['n_hull_allocations']} "
        f"allocations against the {hg['n_pareto']} on the exact Pareto set, "
        f"and the largest loss over these budgets is "
        f"{max(abs(r['gap_pts']) for r in hg['rows']):.3f} saving points. The "
        f"two negative entries are the sweep landing on a budget the "
        f"enumeration's grid straddles, not the sweep beating the Pareto set, "
        f"which Proposition 5 forbids.")
    k.note("results/hull_gap.json, Bosphorus q32, 40 tiles, on the earlier "
           "cost vector; the enumeration and the sweep use the same vector, so "
           "the gap is unaffected by the correction.")

    k.par(
        "Why is the loss that small? Not because the images cooperate. The "
        "reachable costs form a lattice, and the spacing of that lattice is "
        "set by how far the mean cost moves when \\lambda crosses a point at "
        "which some tile changes its mind.")

    k.par(
        "<b>Proposition 6 (granularity).</b> If at most m tiles switch exit at "
        "any single breakpoint of the sweep, and each switch moves a tile by "
        "one rung, then consecutive reachable savings differ by at most")
    e_lat = k.eq(r"\mathrm{spacing} \;\leq\; \frac{100\;m\,"
                 r"\max_k (c_{k+1}-c_k)}{N}\;\%")
    k.par(
        "and the loss from convexity cannot exceed the largest spacing.")

    k.par(
        "<i>Proof.</i> Between breakpoints the assignment is constant, so the "
        "reachable costs are exactly the costs at the breakpoints. At one "
        "breakpoint the mean cost changes by (1/N) times the sum of the "
        "individual changes, each of which is at most max<sub>k</sub>"
        "(c<sub>k+1</sub> - c<sub>k</sub>) in magnitude, and there are at most "
        "m of them; multiplying by 100 puts it in saving points. For the "
        "second part, let a budget fall strictly between two reachable levels. "
        "By Proposition 2 the bisection returns the lower level, so the loss "
        "is at most the distance between them.")

    k.par(
        f"On the measured table m = {tc['m_simultaneous']}, because tiles with "
        f"identical rows switch together, the largest rung is "
        f"{tc['max_rung']:.4f} and N = {tc['T']}, so the bound is "
        f"{tc['lattice_bound_pts']:.4f} saving points. The measured spacing is "
        f"{tc['lattice_spacing_pts']:.4f}, sitting exactly on the bound, and "
        f"the convexity loss of Table {t_hull} is {tc['hull_loss_pts']:.4f}, "
        f"comfortably under it. Nothing in the expression depends on content or "
        f"on rate, and it falls as 1/N: halving the tile side puts four times "
        f"as many tiles on the lattice, each carrying a quarter of the step. "
        f"Fine granularity, not a property of the pictures, is what makes the "
        f"Lagrangian relaxation lossless in practice, and by Table "
        f"{t_geom} the smallest test class has N = 2, where it is not.")
    k.note("m, the largest rung, the bound and the measured spacing from "
           "results/theory_checks.json (Bosphorus q32, 40 tiles, "
           "runs/RECIPE512/ckpt_eval.pth.tar), written by "
           "scripts/verify_theory.py, which reports seven of seven "
           "propositions passing.")

    # ==================================================================
    k.h2("Floor, saturation and ceiling")

    k.par(
        "The sweep has two ends and both are worth naming, because a budget "
        "outside them buys nothing.")

    k.par(
        "<b>Proposition 7 (the floor).</b> At \\lambda = 0 the allocation "
        "attains D<sub>min</sub> = (1/N) ∑<sub>t</sub> min<sub>k</sub> "
        "D<sub>t,k</sub>, and no allocation attains less. <i>Proof.</i> "
        "Immediate from Proposition 1 with \\lambda = 0, and from the fact "
        "that D(a) is a mean of terms each bounded below by its own tile "
        "minimum.")

    k.par(
        "Because D<sub>t,k</sub> is measured on the deployed tiled path, "
        "D<sub>min</sub> is not zero: it is what tiling costs before any tile "
        "exits early, and a budget below it admits no allocation at all.")

    k.par(
        "<b>Proposition 8 (saturation).</b> Define")
    e_sat = k.eq(r"\lambda_{\mathrm{sat}} \;=\; \max_{t}\;\max_{k>j}\;"
                 r"\frac{D_{t,j}-D_{t,k}}{c_k-c_j}")
    k.par(
        "Then for every \\lambda > \\lambda<sub>sat</sub> the oracle assigns "
        "exit j to every tile, the frame's cost is exactly c<sub>j</sub>, and "
        "no larger price changes anything.")

    k.par(
        "<i>Proof.</i> Exit j beats exit k on tile t precisely when "
        "D<sub>t,j</sub> + \\lambda c<sub>j</sub> ≤ D<sub>t,k</sub> + "
        "\\lambda c<sub>k</sub>, that is when \\lambda ≥ (D<sub>t,j</sub> - "
        "D<sub>t,k</sub>)/(c<sub>k</sub> - c<sub>j</sub>), the denominator "
        "being positive for k > j. Taking the maximum over k and then over t "
        "gives a single price beyond which exit j wins everywhere. The cost of "
        "the constant map is c<sub>j</sub> by equation "
        f"({e_CD}).")

    lam_sat = max((Dj[t][0] - Dj[t][a]) / (ctt[a] - ctt[0])
                  for t in range(len(Dj)) for a in range(1, len(ctt)))
    k.par(
        f"On the worked frame that closed form gives "
        f"\\lambda<sub>sat</sub> = {lam_sat * 1e5:.3f}×10<super>-5</super>, "
        f"which is exactly the last "
        f"switch of the most expensive tile in Table {t_tiles}: the price at "
        f"which the frame saturates is the price at which its most stubborn "
        f"tile gives up.")

    k.par(
        "<b>Proposition 9 (the ceiling).</b> The largest saving any allocation "
        "can reach is")
    e_ceil = k.eq(r"S_{\max} \;=\; 100\,(1-c_j)\,\%")
    k.par(
        "<i>Proof.</i> C(a) is a mean of costs each at least c<sub>j</sub>, "
        "because j is the shallowest usable exit, so C(a) ≥ c<sub>j</sub> "
        "with equality only for the constant map, which Proposition 8 shows is "
        "attained.")

    pc = k.J("per_class_RECIPE512.json")
    _cv = [v["saving"] for r in pc["rows"] if r["budget_db"] >= 0.5
           for v in r["per_class"].values()]
    k.par(
        f"S<sub>max</sub> = \\Ceiling% for the shipped ladder. It is a "
        f"function of the split depth alone, so it does not move with content, "
        f"with rate or with training, and that is checkable: over the "
        f"{len(_cv)} class-and-rate cells measured at a saturating budget the "
        f"reported ceiling spreads by {max(_cv) - min(_cv):.0e} saving points, "
        f"which is single-precision noise accumulated through {len(_cv)} "
        f"independent decodes.")
    k.note("Per-class invariance computed from results/per_class_RECIPE512.json "
           "at budgets of 0.5 dB and above, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. The ceiling itself is "
           "\\Ceiling%, from results/saturation_RECIPE512_ctc53.json.")

    rows = [["q", "floor dB", "saturation dB", "band dB", "0.1 dB uses"]]
    for r in sat["rows"]:
        band = r["saturation_db"] - r["floor_db"]
        rows.append([f"q{r['qp']}", _f(r["floor_db"], 4),
                     _f(r["saturation_db"], 4), _f(band, 4),
                     f"{100 * (0.1 - r['floor_db']) / band:.0f}%"])
    t_band = k.rows(rows,
        "<b>The band a budget works in.</b> Below the floor no allocation "
        "exists; at or above saturation every allocation is the same one and "
        "reaches \\Ceiling%. The last column places the headline budget inside "
        "that band: it uses \\BandUseLow% of it at the lowest rate and "
        "\\BandUseHigh% at the highest, so the same 0.1 dB is a loose budget at "
        "low rate and a tight one at high rate.")
    k.note("results/saturation_RECIPE512_ctc53.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq frames. Floor and "
           "saturation are \\FloorLow to \\FloorHigh dB and \\SatLow to "
           "\\SatHigh dB across the five reported rates.")

    _an = [abs(r["drift_db"]) for r in k.J("anchor_RECIPE512_ctc53.json")["rows"]]
    _av = [abs(r["drift_db"]) for r in k.J("anchor_VERBATIM.json")["rows"]]
    k.par(
        "The floor is also the one place where the derivation is exposed to "
        "training. Every quantity here is measured against a reference decoder "
        "and the deepest exit is assumed to reproduce it; nothing in "
        "Propositions 1 to 9 enforces that. If the deepest exit drifts from the "
        "reference by δ then D<sub>t,K-1</sub> is raised for every tile, "
        "the whole achievable set shifts upward by δ before any compute "
        "is saved, and a budget below δ becomes unreachable at every "
        "allocation. Routing has not failed there; the achievable set simply "
        "does not meet the constraint. This is why the training objective "
        f"carries an anchor term tying the deepest exit to the released "
        f"decoder, and the size of the effect is measurable: with the anchor "
        f"the deepest-exit drift is {min(_an):.3f} to {max(_an):.3f} dB over "
        f"q0 to q63, and on a run trained without it the same drift is "
        f"{min(_av):.3f} to {max(_av):.3f} dB, which on its own consumes more "
        f"than the headline budget.")
    k.note("Anchored drift from results/anchor_RECIPE512_ctc53.json, checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar, \\NumSeq frames. Unanchored drift "
           "from results/anchor_VERBATIM.json, checkpoint "
           "runs/VERBATIM/ckpt_eval.pth.tar, 40 frames, a different training "
           "run at three rates. Neither is the pinned checkpoint and the two "
           "are not a controlled pair; the direction is what the argument "
           "needs, and the interval is quoted as measured rather than as a "
           "difference.")

    grid = k.J("signalled_RECIPE512_grid.json")
    unreach = [r for r in grid["rows"] if not r.get("budget_reachable", True)]
    k.par(
        f"The floor is not an abstraction either. Over the "
        f"{len(grid['rows'])} budget-and-rate cells of the dense sweep, "
        f"{len(unreach)} are unreachable, all of them at the 0.05 dB budget and "
        f"all at rates whose floor already exceeds it.")
    k.note("results/signalled_RECIPE512_grid.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences, 2 frames "
           "per sequence, 9 budgets by 5 rates.")

    # ==================================================================
    k.h2("The value of adaptivity, and a bound on it")

    k.par(
        "Everything so far describes what a content-aware allocation does. The "
        "next question is what content awareness is worth, and it can be "
        "answered without building a predictor at all. Compare the best "
        "per-tile allocation with the best content-blind one at the same "
        "price:")
    e_J = k.eq(r"J_{\mathrm{ad}}(\lambda)=\frac{1}{N}\sum_{t}\min_{k}"
               r"\left[D_{t,k}+\lambda c_k\right], \quad "
               r"J_{\mathrm{fix}}(\lambda)=\min_{k}\left[\bar D_k+"
               r"\lambda c_k\right]")

    k.par(
        f"J<sub>ad</sub> takes the minimum inside the average and "
        f"J<sub>fix</sub> takes it outside. That is the entire difference, and "
        f"by Propositions 1 and 3 the first is the best value the per-tile "
        f"family can attain at price \\lambda and the second is the best the "
        f"fixed-proportion family can attain.")

    k.par(
        "<b>Theorem 10 (adaptivity gain).</b> For every \\lambda ≥ 0, "
        "\\Delta(\\lambda) = J<sub>fix</sub>(\\lambda) - "
        "J<sub>ad</sub>(\\lambda) ≥ 0, with equality if and only if some "
        "single exit k<super>*</super> attains the per-tile minimum for "
        "every tile.")

    k.par(
        "<i>Proof.</i> Fix \\lambda and let k be any exit. Then the exit mean "
        "at k plus \\lambda c<sub>k</sub> = (1/N) ∑<sub>t</sub> "
        "[D<sub>t,k</sub> + \\lambda c<sub>k</sub>] ≥ (1/N) "
        "∑<sub>t</sub> min<sub>k'</sub>[D<sub>t,k'</sub> + \\lambda "
        "c<sub>k'</sub>] = J<sub>ad</sub>(\\lambda), because each summand "
        "dominates the corresponding minimum. The right-hand side does not "
        "depend on k, so taking the minimum over k on the left preserves the "
        "inequality and gives J<sub>fix</sub> ≥ J<sub>ad</sub>. For "
        "equality, suppose some k<super>*</super> attains the per-tile "
        "minimum everywhere; then choosing it makes every summand equal to its "
        "minimum and the inequality is tight. Conversely suppose \\Delta = 0 "
        "and let k<super>*</super> attain J<sub>fix</sub>. Then (1/N) "
        "∑<sub>t</sub> ([D<sub>t,k*</sub> + \\lambda "
        "c<sub>k*</sub>] - min<sub>k'</sub>[D<sub>t,k'</sub> + \\lambda "
        "c<sub>k'</sub>]) = 0. Every summand is non-negative, so every summand "
        "is zero, which says exactly that k<super>*</super> attains the "
        "minimum on every tile.")

    k.par(
        "Three things follow that are worth separating. First, \\Delta depends "
        "only on the table D<sub>t,k</sub> and the costs, so it is computable "
        "before any router exists and it upper-bounds what any router can gain "
        "over a content-blind allocation at that operating point. Second, the "
        "equality condition says when adaptivity is worthless, and it is a "
        "condition on the data, not on the method: when every tile agrees on "
        "the best exit there is nothing to adapt to. Third, and less "
        "comfortably, \\Delta is a property of the ladder it is measured on. "
        "Improving the shallow exits compresses the differences between exits, "
        "more tiles then agree on the argmin, and the measured value of "
        "adaptivity falls even as the saving rises.")

    thc = k.J("theory_check.json")
    lams = ["1e-05", "3e-05", "1e-04", "3e-04"]
    rows = [["q", "λ=10<super>-5</super>", "3×10<super>-5</super>",
             "10<super>-4</super>", "3×10<super>-4</super>"]]
    for q in ["0", "16", "32", "48", "63"]:
        d_ = thc[q]["delta"]
        rows.append([f"q{q}"] + [f"{100 * d_[l]['delta'] / d_[l]['J_oracle']:.2f}%"
                                 for l in lams])
    t_delta = k.rows(rows,
        "<b>The adaptivity gain in scale-free form</b>, Δ(λ) as a percentage "
        "of J<sub>ad</sub>(λ). At the two smaller prices it rises with rate, "
        "which says adaptivity is worth most where the deep blocks are doing "
        "real work. At the largest price it is essentially zero at every rate, "
        "which is Theorem 10's equality condition arriving: the oracle itself "
        "has collapsed onto one exit, so there is nothing left for a router to "
        "recover.")
    k.note("results/theory_check.json, checkpoint "
           "runs/BEST/ckpt_eval.pth.tar, 40 sequences (UVG, MCL-JCV and HEVC "
           "class E; classes B, C and D not measured), on the earlier cost "
           "vector of Table " + str(t_cost) + ". This is the one table in the "
           "section that is not on the pinned checkpoint, and nothing on the "
           "pinned checkpoint currently reproduces it.")

    # value of adaptivity in the paper's own units, pinned
    tgt = 10 ** (0.1 / 10)
    rows = [["q", "blind %", "oracle %", "gap", "blind dB", "shuffled dB"]]
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
        rows.append([f"q{r['qp']}", _f(blind, 2),
                     _f(r["oracle"]["saving"], 2),
                     _f(r["oracle"]["saving"] - blind, 2),
                     _f(ydb, 4), _f(r["random"]["db"], 4)])
    t_adapt = k.rows(rows,
        "<b>The same gap in the units the paper reports.</b> Columns 2 to 4 "
        "are savings at a 0.1 dB budget: the best content-blind mixture of "
        "uniform depths, the per-tile oracle, and the difference. Columns 5 "
        "and 6 are decibels at the oracle's own compute: what the best blind "
        "mixture would read there, and what a random shuffle of the oracle's "
        "exit map actually reads. Adaptivity is worth 4.5 to 7.6 saving "
        "points, rising with rate, against an oracle reading \\OracleDb dB and "
        "a shuffle reading up to \\RandomDb dB.")
    k.note("Computed from results/static_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq frames. The blind "
           "columns are the lower hull of the four uniform-depth points of "
           "Table " + str(t_vert) + " under Proposition 3, evaluated at the "
           "budget and at the oracle's compute; the shuffle column is a "
           "measured decode. At q0 the shuffle reads slightly better than the "
           "hull predicts, which bounds the interaction the additivity "
           "assumption ignores at under 0.002 dB.")

    # ==================================================================
    k.h2("Convexity in the units the frontier is plotted in")

    lc = k.J("logconvexity.json")

    k.par(
        "Theorem 4 gives D convex in C. The frontier is almost never plotted "
        "in those units: it is plotted as decibels against percentage saving, "
        "and convexity there is a different statement, because the logarithm "
        "is concave and increasing and so preserves neither convexity nor "
        "concavity. With S = 1 - C/c<sub>K-1</sub>,")
    e_db = k.eq(r"\mathrm{dB}(S) \;=\; \frac{10}{\ln 10}\,\ln D(C), "
                r"\qquad C = (1-S)\,c_{K-1}")

    k.par(
        "<b>Proposition 11.</b> Let D be twice differentiable along the "
        "frontier. Then dB(S) is convex if and only if D is log-convex in C:")
    e_lcx = k.eq(r"D\,D^{\prime\prime} \;\geq\; (D^{\prime})^{2} "
                 r"\qquad\Longleftrightarrow\qquad "
                 r"\frac{d}{dC}\left(\frac{1}{\lambda}\right) \;\geq\; "
                 r"\frac{1}{D}")

    k.par(
        f"<i>Proof.</i> C is affine in S and convexity is preserved under "
        f"affine reparametrisation, so by ({e_db}) dB is convex in S if and "
        f"only if ln D is convex in C. Differentiating twice, (ln D)'' = "
        f"D''/D - (D'/D)² = [D D'' - (D')²]/D², and D > 0, so (ln D)'' ≥ 0 "
        f"if and only if D D'' ≥ (D')². For the second form, the slope of "
        f"the frontier is D' = -\\lambda by the supporting-line argument of "
        f"Proposition 5, so D'' = -\\lambda'; substituting gives -\\lambda' D "
        f"≥ \\lambda², and dividing by \\lambda² > 0 gives -\\lambda'/"
        f"\\lambda² ≥ 1/D, whose left-hand side is d(1/\\lambda)/dC. "
        f" Read the second form as a rate condition: distortion has "
        f"to fall at least exponentially in compute for the plotted curve to "
        f"be convex.")

    k.par(
        "This is a property of the decoder and not a consequence of anything "
        "above; a convex D can fail it. It is also directly checkable, so it "
        "is checked rather than assumed. Two remarks make the differentiability "
        "hypothesis harmless in one case and misleading in the other. Under "
        "fixed proportions D is piecewise linear with K-j vertices, so ln D is "
        "piecewise <i>concave</i> with upward jumps in its derivative at the "
        "vertices: neither convex nor concave, and the shape one sees depends "
        "entirely on where one samples. Under per-tile assignment the hull "
        "carries up to N(K-j-1) vertices and is smooth at any plotted scale. "
        "The two cases are genuinely different, and treating a "
        "fixed-proportion curve as though it were the per-tile one is how a "
        "confident claim about frontier shape turns out to be an artefact of "
        "vertex spacing.")

    rows = [["q", "points", "condition holds", "margin", "quartic fit"]]
    for q in ["0", "16", "32", "48", "63"]:
        r = lc[q]
        rows.append([f"q{q}", r["n"], f"{r['frac_ok']:.1f}%",
                     _f(r["worst"], 4), f"{r['quartic_frac_ok']:.1f}%"])
    t_lcx = k.rows(rows,
        "<b>Testing log-convexity on the measured frontier.</b> The direct "
        "test asks whether the secant slopes of ln D against S are "
        "non-decreasing; the margin is the smallest increase attained. The "
        "condition holds at every point of every frontier, and the margin "
        "grows tenfold from the lowest rate to the highest. Fitting a quartic "
        "and differentiating it twice instead reports 73.7% to 84.0%, and the "
        "difference is the fit rather than the data: near the ceiling the "
        "frontier is close to vertical and a polynomial overshoots, whereas "
        "the secant test has no fitted degree to choose.")
    k.note("results/logconvexity.json. That file records no checkpoint, so it "
           "cannot be attributed from disk; its five frontiers have 13 to 16 "
           "points each. The quartic column is the file's own "
           "quartic_frac_ok field.")

    # ==================================================================
    k.h2("How much of the oracle a partial signal recovers")

    hy = k.J("hybrid_RECIPE512_b01_fixed.json")

    k.par(
        "The decoder cannot run the argmin of Proposition 1, because "
        "D<sub>t,k</sub> is an error against a source the decoder never "
        "receives. A predictor can guess it. Writing p<sub>t,k</sub> for a "
        "head's probability that tile t belongs at exit k, its rule is")
    e_router = k.eq(r"\hat k_t \;=\; \arg\max_k\left[\log p_{t,k} - "
                    r"\beta\,c_k\right] \;=\; \arg\min_k\left["
                    r"\frac{-\log p_{t,k}}{\kappa} + \lambda\,c_k\right]")

    k.par(
        f"The second form of ({e_router}) is the same rule with \\beta = "
        f"\\lambda \\kappa, where \\kappa is a fixed calibration turning nats "
        f"into squared error. It is written that way to make the point that a "
        f"router is the oracle of ({e_argmin}) with a surrogate in place of "
        f"D<sub>t,k</sub>, and that the same price governs both. Given the two, "
        f"define the per-tile <i>regret</i> at a fixed price:")
    e_reg = k.eq(r"r_t(\lambda) \;=\; \left[D_{t,\hat k_t} + \lambda "
                 r"c_{\hat k_t}\right] - \left[D_{t,k^{\star}_t} + \lambda "
                 r"c_{k^{\star}_t}\right] \;\geq\; 0")

    k.par(
        f"Non-negativity is immediate from ({e_argmin}). The regret is exactly "
        f"what the frame's Lagrangian loses on tile t by trusting the "
        f"prediction, and by Proposition 1 those losses add. That gives the "
        f"encoder an option the two extremes do not have: signal the exits of "
        f"a subset S of the tiles and let the head decide the rest. Signalling "
        f"S removes exactly ∑<sub>t∈ S</sub> r<sub>t</sub> from the "
        f"objective, no more and no less, which is the whole reason a partial "
        f"signal is analysable at all.")

    k.par(
        "<b>Proposition 12 (Lorenz bound).</b> Fix \\lambda and let "
        "r<sub>(1)</sub> ≥ r<sub>(2)</sub> ≥ ... ≥ "
        "r<sub>(N)</sub> be the regrets in decreasing order with total R. For "
        "any subset S of size s, the fraction of the total regret that "
        "signalling S removes is at most")
    e_lor = k.eq(r"L(\rho) \;=\; \frac{1}{R}\sum_{i=1}^{\lceil \rho N\rceil} "
                 r"r_{(i)}, \qquad \rho = s/N")
    k.par(
        "with equality if and only if S is a set of s largest regrets. L is "
        "the Lorenz curve of the regret distribution: it is non-decreasing, "
        "concave, runs from 0 to 1, and lies above the diagonal by an amount "
        "the Gini coefficient summarises.")

    k.par(
        "<i>Proof.</i> The sum of any s of the r<sub>t</sub> is at most the sum "
        "of the s largest, since replacing an element of S by a larger one "
        "outside S cannot decrease the sum, and repeating that exchange "
        "terminates at a set of s largest values. Dividing by R gives the "
        "bound. Concavity holds because the increments r<sub>(i)</sub> are "
        "non-increasing by construction.")

    k.par(
        "The practical content is that a highly unequal regret distribution "
        "can be repaired cheaply. If a tenth of the tiles carry half the "
        "regret then signalling a tenth of them removes half of what the "
        "predictor gives up, at a tenth of the bits. Whether that holds is a "
        "measurement, and the Gini coefficient of the regret is the one number "
        "that says so.")

    def cell(q, rho):
        return next((x for x in hy["rows"]
                     if x["qp"] == q and abs(x["rho"] - rho) < 1e-9), None)

    qs = sorted({x["qp"] for x in hy["rows"]})
    rows = [["q", "Gini", "L(.1)", "rec(.1)", "L(.2)", "rec(.2)", "L(.5)",
             "rec(.5)"]]
    ratios = []
    for q in qs:
        a0, a1 = cell(q, 0.0), cell(q, 1.0)
        base, full = a0["saving_pct_vs_release"], a1["saving_pct_vs_release"]
        line = [f"q{q}", _f(a0["gini_regret"], 2)]
        for rho in (0.1, 0.2, 0.5):
            c_ = cell(q, rho)
            rec = 100 * (c_["saving_pct_vs_release"] - base) / (full - base)
            L = 100 * c_["lorenz_at_rho"]
            line += [f"{L:.0f}%", f"{rec:.0f}%"]
        rows.append(line)
        for x in hy["rows"]:
            if x["qp"] == q and x.get("lorenz_at_rho"):
                ratios.append(
                    (x["saving_pct_vs_release"] - base) / (full - base)
                    / x["lorenz_at_rho"])
    _L10 = [100 * cell(q, 0.1)["lorenz_at_rho"] for q in qs]
    _r10 = [100 * (cell(q, 0.1)["saving_pct_vs_release"]
                   - cell(q, 0.0)["saving_pct_vs_release"])
            / (cell(q, 1.0)["saving_pct_vs_release"]
               - cell(q, 0.0)["saving_pct_vs_release"]) for q in qs]
    t_lor = k.rows(rows,
        f"<b>The Lorenz curve of the regret, and what signalling actually "
        f"recovers.</b> L(ρ) is the share of the total regret carried by the ρ "
        f"worst tiles, which Proposition 12 says is the most any signal of "
        f"that size can remove at a fixed price; rec(ρ) is the share of the "
        f"predictor-to-oracle gap the measured sweep recovers. Regret is "
        f"concentrated at every rate, Gini \\GiniMin to \\GiniMax: a tenth of "
        f"the tiles carries {min(_L10):.0f}% to {max(_L10):.0f}% of the total "
        f"regret and recovers {min(_r10):.0f}% to {max(_r10):.0f}% of the "
        f"gap.")
    k.note("results/hybrid_RECIPE512_b01_fixed.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences at a 0.1 dB "
           "budget; the head it is measured against was trained on "
           "runs/RECIPE512/ckpt_eval.pth.tar. rec(ρ) is computed here from the "
           "same file as the saving at ρ, relative to the ρ = 0 and ρ = 1 "
           "endpoints.")

    k.par(
        f"The measured recovery is not bounded by L, and the reason is not a "
        f"failure of Proposition 12. Over every reachable cell of the file, "
        f"three ρ of which appear in Table {t_lor}, the ratio rec/L runs from "
        f"\\LorenzTightMin% to \\LorenzTightMax%. Proposition 12 is a "
        f"statement at a <i>fixed</i> "
        f"price, whereas the measurement re-bisects \\lambda at every ρ so "
        f"that the frame lands back on its budget. Re-bisecting enlarges the "
        f"feasible set: the overridden tiles and the predicted ones then "
        f"effectively carry two multipliers, and pairs of multipliers reach "
        f"allocations that no single one does. That is also why signalling half "
        f"the tiles beats signalling all of them at \\HybridBeatsAN of \\HybridBeatsAOf "
        f"rates. The Lorenz curve is the right prediction to hold in mind and "
        f"the wrong thing to call a bound on the deployed system.")

    bits_rows = [["ρ", "map bits per frame", "recovery at q0", "at q63"]]
    for rho in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        line = [f"{rho:g}"]
        c_ = cell(qs[-1], rho)
        line.append(_f(c_["map_bits"], 1))
        for q in (qs[0], qs[-1]):
            a0, a1 = cell(q, 0.0), cell(q, 1.0)
            cc = cell(q, rho)
            line.append(f"{100 * (cc['saving_pct_vs_release'] - a0['saving_pct_vs_release']) / (a1['saving_pct_vs_release'] - a0['saving_pct_vs_release']):.0f}%")
        bits_rows.append(line)
    t_bits = k.rows(bits_rows,
        "<b>What the signal costs.</b> The map is an entropy-coded mask over "
        "tiles plus an index for each override. A fifth of the tiles costs "
        "\\HybridBitsFifth bits per 1080p frame and recovers over half the "
        "gap; the ρ = 1 endpoint, which reproduces the oracle's own "
        "allocation, costs \\HybridBitsFull bits under this coder. For "
        "scale, a fixed-length code over the K exits would be 3 bits per tile, "
        "120 bits per frame at 1080p, and the usable alphabet of K-j = 4 exits "
        "needs only 2, so the full map is already coded below its fixed-length "
        "size.")
    k.note("results/hybrid_RECIPE512_b01_fixed.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. Configuration A's own map is "
           "\\MapBitsLo to \\MapBitsHi bits per frame across the five rates "
           "(results/signalled_RECIPE512_ctc53.json, same checkpoint), or "
           "\\MapOverheadLow% of a typical bitrate.")

    # ==================================================================
    k.h2("The rate-rank rule, and the model it comes from")

    rr = k.J("raterank_RECIPE512_b01.json")

    k.par(
        "A trained head is one way to guess the oracle. There is a cheaper "
        "one, and it is worth deriving rather than presenting as a heuristic, "
        "because the derivation says exactly when it must work and exactly how "
        "it fails. The decoder holds, before the trunk runs, the number of bits "
        "the entropy coder spent on each tile's latents. Write b<sub>t</sub> "
        "for that count divided by the frame's mean, so b<sub>t</sub> is "
        "around 1.")

    k.par(
        "The model is separability, in log space: a tile has one scalar "
        "difficulty, and the ladder has one profile that applies to every tile "
        "in proportion.")
    e_r1 = k.eq(r"\log D_{t,k} \;\approx\; u_t + \log\varphi_k, \qquad "
                r"u_t \;=\; \alpha\log b_t + c")

    k.par(
        f"Equivalently D<sub>t,k</sub> = g<sub>t</sub> φ<sub>k</sub> "
        f"with g<sub>t</sub> = exp(u<sub>t</sub>) > 0. The exit profile "
        f"φ is K-j numbers fitted once per rate; \\alpha and c are two "
        f"more. Nothing is learned per frame and nothing is transmitted, which "
        f"is why the rule costs the decoder exactly zero. \\alpha, c and the "
        f"profile are all fitted leave-one-sequence-out, so no sequence "
        f"contributes to the profile used to route it.")

    k.par(
        "<b>Proposition 13 (a rank-1 table gives a threshold rule).</b> If "
        "D<sub>t,k</sub> = g<sub>t</sub> φ<sub>k</sub> with "
        "g<sub>t</sub> > 0 then")
    e_rank = k.eq(r"\arg\min_k\left[g_t\varphi_k + \lambda c_k\right] \;=\; "
                  r"\arg\min_k\left[\varphi_k + \frac{\lambda}{g_t}c_k\right]")
    k.par(
        "so a tile's exit depends on the tile only through the single scalar "
        "\\lambda/g<sub>t</sub>, and by Proposition 2 the chosen exit is "
        "non-increasing in it. Sorting tiles by g<sub>t</sub> and cutting the "
        "sorted list at K-j-1 thresholds therefore reproduces the oracle "
        "exactly.")

    k.par(
        f"<i>Proof.</i> Divide the bracket by g<sub>t</sub> > 0, which does not "
        f"move the argmin. The reduced problem is the one-tile problem of "
        f"({e_argmin}) with distortion φ and price \\lambda/"
        f"g<sub>t</sub>, so Proposition 2 applies to it and its solution is "
        f"non-increasing in the price. Since \\lambda/g<sub>t</sub> is "
        f"decreasing in g<sub>t</sub>, the exit is non-decreasing in "
        f"g<sub>t</sub>: harder tiles go deeper. The thresholds are the "
        f"breakpoints of that one common problem and there are at most "
        f"K-j-1 of them.")

    k.par(
        "<b>Proposition 14 (a rank-1 rule uses only the profile's hull).</b> "
        "Under the same model, an exit k is chosen for some tile at some price "
        "only if (c<sub>k</sub>, φ<sub>k</sub>) lies on the lower convex "
        "hull of the profile. <i>Proof.</i> By Proposition 13 the reduced "
        "problem is a single Lagrangian sweep over the points (c<sub>k</sub>, "
        "φ<sub>k</sub>), and by Proposition 5 such a sweep reaches only "
        "hull points.")

    cB = cost_B
    rows = [["q", "α", "φ(2)", "φ(3)", "φ(4)", "φ(5)", "on hull"]]
    for r in rr["rows"]:
        phi = r["phi"]
        hull = _lower_hull(list(zip(cB, phi)))
        rows.append([f"q{r['qp']}", _f(r["alpha"], 2)]
                    + [_f(p, 4) for p in phi]
                    + [",".join(str(exits_B[i]) for i in sorted(hull))])
    t_fit = k.rows(rows,
        "<b>The fitted separable model.</b> α is the exponent relating a "
        "tile's coded bits to its difficulty and φ is the exit profile, both "
        "fitted leave-one-sequence-out. α is positive at every rate, so a tile "
        "the entropy coder spent bits on is a tile every exit reconstructs "
        "worse, and it falls sevenfold from the lowest rate to the highest. "
        "The profile spans only about 4% from its shallowest to its deepest "
        "entry, which is the margin the rule has to work in. All four exits "
        "survive on the profile's lower hull at every rate, so Proposition 14 "
        "costs nothing here.")
    k.note("results/raterank_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences at a 0.1 dB "
           "budget. Hull membership computed against the shipped cost vector "
           "of Table " + str(t_cost) + ". On the single-sequence table of "
           "results/tile_table.json the same construction drops exit 4 from "
           "the hull, so the property is not automatic.")

    rows = [["q", "rule %", "oracle %", "agree", "ρ level", "ρ spread",
             "ρ exit"]]
    for r in rr["rows"]:
        rows.append([f"q{r['qp']}", _f(r["saving_pct_vs_release"], 2),
                     _f(r["oracle_saving_pct_vs_release"], 2),
                     _f(r["agreement"], 3),
                     _f(r["spearman_bits_vs_level"], 3),
                     _f(r["spearman_bits_vs_spread"], 3),
                     _f(-r["spearman_bits_vs_exit"], 3)])
    t_rr = k.rows(rows,
        "<b>What the free rule achieves, and what the bit count carries.</b> "
        "Savings at a 0.1 dB budget against the oracle's, and the fraction of "
        "tiles on which the two agree. The last three columns are Spearman "
        "correlations of the tile's coded bits with the mean distortion level "
        "across the ladder, with the spread from shallowest to deepest, and "
        "with the oracle's own exit index. Bits track the level well, the "
        "spread almost as well, and the exit least well, which is the ordering "
        "the separable model predicts.")
    k.note("results/raterank_RECIPE512_b01.json, checkpoint "
           "runs/RECIPE512/ckpt_PAPER.pth.tar, \\NumSeq sequences. The file "
           "stores the last column as spearman_bits_vs_exit against the "
           "negated exit index; it is reported here with the sign flipped, so "
           "a positive value means more bits and a deeper exit.")

    # rank-1 residual analysis on the tile table
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
        f"So where does the rule lose its \\RateRankBeatsBy points? Not in the "
        f"separability. On the measured tile table the separable model absorbs "
        f"{100 * (1 - ssr / tot):.2f}% of the variance of log D, which by any "
        f"ordinary standard is an excellent fit. The trouble is that the "
        f"quantity it has to resolve is small. The whole exit profile spans "
        f"{spread:.4f} in log units, while the non-separable residual has a "
        f"root mean square of {rms:.4f}, which is {100 * rms / spread:.0f}% of "
        f"it. A model can explain almost all of a table and still be unable to "
        f"order the differences that decide the argmin, because the argmin "
        f"reads only the differences. Layered on that, predicting the tile's "
        f"difficulty from its bit count alone leaves R² = {r2:.2f}, and by "
        f"Proposition 13 an error in log g<sub>t</sub> is a proportional error "
        f"in the effective price that tile faces.")
    k.note("Residual analysis computed from results/tile_table.json "
           "(Bosphorus, q32, 40 tiles), checkpoint "
           "runs/RECIPE512/ckpt_eval.pth.tar; α fitted there is "
           f"{al:.2f} against {rr['rows'][2]['alpha']:.2f} for the "
           "\\NumSeq-sequence fit at the same rate, the difference being one "
           "sequence against \\NumSeq.")

    k.par(
        "Two consequences are worth stating plainly. The rule is not a "
        "weaker router: it is the exact oracle for a decoder whose distortion "
        "table happens to be rank-1, and its shortfall measures how far from "
        "rank-1 the real table is. And it is the reason a partial signal is "
        "attractive even without a trained head, since Proposition 12 applies "
        "to any predictor whose regret can be computed by the encoder, this "
        "one included.")

    # ==================================================================
    k.h2("What is proved and what is checked")

    rows = [["statement", "proof", "checked in", "checkpoint"],
            ["1 separation", f"{S[4]}", "verify_theory P1", "eval"],
            ["2 monotonicity", f"{S[4]}", "verify_theory P2", "eval"],
            ["3 fixed-proportion hull", f"{S[6]}", "static_RECIPE512_b01", "PAPER"],
            ["4 Minkowski average", f"{S[6]}", "not measured", "-"],
            ["5 hull, and interior unreachable", f"{S[7]}", "hull_gap", "eval"],
            ["6 lattice spacing", f"{S[7]}", "theory_checks", "eval"],
            ["7 floor", f"{S[8]}", "saturation_RECIPE512", "PAPER"],
            ["8 saturation", f"{S[8]}", "verify_theory P5", "eval"],
            ["9 ceiling", f"{S[8]}", "per_class_RECIPE512", "PAPER"],
            ["10 adaptivity gain", f"{S[9]}", "theory_check", "BEST"],
            ["11 log-convexity", f"{S[10]}", "logconvexity", "none recorded"],
            ["12 Lorenz bound", f"{S[11]}", "hybrid_RECIPE512_b01_fixed", "PAPER"],
            ["13 rank-1 threshold rule", f"{S[12]}", "raterank_RECIPE512_b01",
             "PAPER"],
            ["14 rank-1 uses hull exits", f"{S[12]}", "raterank_RECIPE512_b01",
             "PAPER"]]
    t_sum = k.rows(rows,
        "<b>Every statement in this section, and its evidence.</b> PAPER is "
        "runs/RECIPE512/ckpt_PAPER.pth.tar, eval is "
        "runs/RECIPE512/ckpt_eval.pth.tar, since overwritten by the per-epoch "
        "watcher, and BEST is runs/BEST/ckpt_eval.pth.tar. The seven "
        "propositions checked by scripts/verify_theory.py all pass, on one "
        "sequence at one rate. Theorem 4 is proved and not measured: it "
        "describes a set of 120 vertices that no experiment enumerates.")
    k.note("Files named without a directory are under results/ with a .json "
           "extension. verify_theory refers to scripts/verify_theory.py, whose "
           "output is results/theory_checks.json.")

    k.par(
        f"Three gaps in that table are worth naming rather than leaving to be "
        f"noticed. Theorem 10, the one result the paper leans on hardest, is "
        f"measured only on runs/BEST/ckpt_eval.pth.tar over 40 sequences and "
        f"on the earlier cost vector; the pinned-checkpoint version in Table "
        f"{t_adapt} measures the same quantity in saving points but not the "
        f"scale-free ratio. The convexity check of Table {t_lcx} records no "
        f"checkpoint at all. And the granularity and hull results of {S[7]} rest "
        f"on a single sequence at a single rate, which is enough to check "
        f"arithmetic and not enough to characterise a codec.")
