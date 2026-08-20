"""Architecture and complexity accounting.

Every number here is derived from a file in results/ and the file is named at
the table that uses it. The derivation runs from one DepthConvBlock upward:
per-operator MAC counts (results/adapter_cost.json), the two adapter kinds
(same file), the per-exit cost vector (results/static_RECIPE512_b01.json pins
it exactly, see `_vectors` below), and then the check against hook counts taken
off decodes that were actually executed (results/supp_power.json,
results/signalled_RECIPE512_grid.json).

Nothing in this module reads a checkpoint, opens a CUDA context or writes to
results/. The arithmetic below is reproducible from the JSON alone, which is
the point: a reader with the repository can rerun it.
"""


def _vectors(k):
    """Recover the cost model from results/ rather than from flexuf/cost.py.

    results/static_RECIPE512_b01.json records the frame-level saving of a map
    that sends every tile to exit e, for e = 2..5, on the pinned checkpoint.
    Those four numbers determine the whole model: with c_e = 1 - saving/100,

        c_3 - c_2 = 2p                    p   = one trunk block
        c_5 - c_4 = 2p - a_1              a_1 = the 1x1 adapter
        c_4 - c_3 = 2p + a_1 - a_ffn      a_ffn = the FFN adapter
        c_5 - 8p  = stem + head + repair

    so no share has to be taken on trust from the source. The adapters and the
    repair pass are then repriced on the measured block from
    results/adapter_cost.json, which is what the "derived" column is.
    """
    ac = k.J("adapter_cost.json")
    st = k.J("static_RECIPE512_b01.json")
    C = ac["trunk_channels"]
    blk_meas = ac["block"]["mac_per_px"]
    blk_closed = 8 * C ** 2 + 9 * C
    u = {r["exit"]: 1.0 - r["saving"] / 100.0 for r in st["rows"][0]["uniform"]}
    p = (u[3] - u[2]) / 2.0
    a1 = 2 * p - (u[5] - u[4])
    aff = 2 * p + a1 - (u[4] - u[3])
    repair_closed = (9 * C + C ** 2) / blk_closed * p
    stem = u[5] - 8 * p - 0.0240 - repair_closed        # upsample + j*b blocks

    def model(blk, ffn_in_c2):
        rep = (9 * C + C ** 2) / blk * p
        s1 = C ** 2 / blk * p
        sf = ffn_in_c2 * C ** 2 / blk * p
        fixed = stem + 0.0240 + rep
        out = []
        for e in range(6):
            run = max(e, 2)
            tiled = (run + 1) * 2 - 4
            ad = 0.0 if run == 5 else (sf if (5 - run) * 2 >= 4 else s1)
            out.append(fixed + tiled * p + ad)
        return out, rep, s1, sf

    A, _, _, _ = model(blk_closed, 2)      # the first shipped table
    B, repB, a1B, affB = model(blk_closed, 5)   # the table the paper reports
    D, repD, a1D, affD = model(blk_meas, 5)     # block from the hook count
    return dict(C=C, blk_meas=blk_meas, blk_closed=blk_closed, p=p,
                A=A, B=B, D=D, stem=stem,
                repB=repB, repD=repD, a1B=a1B, a1D=a1D, affB=affB, affD=affD,
                a1=a1, aff=aff)


def _maps(k):
    """The exit map each power condition actually decoded.

    results/supp_power.json records the pooled exit histogram and the tile
    count; scripts/power_profile.py turns the two into the map it decodes by
    rounding the histogram's proportions to the tile count and giving the
    remainder to the deepest exit. Reproduced here so the predicted column of
    Table 5 is a prediction about the decode that was timed.
    """
    out = []
    for r in k.J("supp_power.json")["rows"]:
        h, n = r["hist"], r["n_tiles"]
        tot = float(sum(h))
        cnt = [int(round(x / tot * n)) for x in h]
        cnt[-1] += n - sum(cnt)
        out.append((r, [cnt[0] + cnt[1] + cnt[2], cnt[3], cnt[4], cnt[5]]))
    return out


def _sci(x):
    """3.3e-14 -> "3×10<super>-14</super>". Times-Roman has no superscript
    digits, so the tag is the only way to set one that is not a black box."""
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}×10<super>{int(exp)}</super>"


def _recover(k):
    """Solve the six hook counts of results/supp_power.json for c_2..c_5.

    Each condition contributes one equation: the measured relative cost of a
    decode is the map-weighted mean of the per-exit costs, because the count
    is linear in the map. Six equations, four unknowns, and the residual tells
    us whether the linearity holds. Returns the solution and that residual, in
    points of saving.
    """
    import numpy as np
    A, b = [], []
    for r, m in _maps(k):
        n = float(r["n_tiles"])
        A.append([c / n for c in m])
        b.append(1.0 - r["mac_saving_pct"] / 100.0)
    A, b = np.array(A), np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = 100.0 * float(np.abs(A @ sol - b).max())
    return {e: float(sol[i]) for i, e in enumerate((2, 3, 4, 5))}, resid


def content(k):
    v = _vectors(k)
    C, p = v["C"], v["p"]
    ac = k.J("adapter_cost.json")
    td = k.J("tile_definition.json")

    sec = k.h1("Architecture and complexity accounting")

    k.par(r"This section derives what the decoder costs, operator by operator, "
          r"and then checks the derivation against a count taken off decodes "
          r"that were actually run. It closes with what the saving is worth in "
          r"the two units a deployment cares about, milliseconds and joules, "
          r"which are not the unit the allocation is optimised in.")

    k.par(r"Some vocabulary first. A <i>multiply-accumulate</i> (MAC) is one "
          r"multiply and one add; MAC/px is per pixel of the feature grid an "
          r"operator runs on. The <i>trunk</i> is the released decoder's stack "
          r"of N = 12 identical DepthConvBlocks at width C = " + f"{C}" +
          r". An <i>exit</i> is a point in that stack where a tile may stop; "
          r"the ladder has K = 6 exits, one every b = N/K = 2 blocks. The "
          r"<i>split depth</i> j = 2 is the number of exits' worth of blocks "
          r"that run once for the whole frame before any tile is allowed to "
          r"leave, so groups 0 and 1 are full-frame and groups 2 to 5 are "
          r"per-tile. An <i>adapter</i> is a small zero-initialised residual "
          r"module that stands in for the blocks an exit skips. A <i>tile</i> "
          r"is 256 RGB px square. The <i>exit map</i> is the per-tile "
          r"assignment. The <i>released decoder</i> is DCVC-UF's intra decoder "
          r"[14] with none of this added, and it is the unit all costs below "
          r"are quoted in.")

    # ------------------------------------------------------------------
    k.h2("One DepthConvBlock at C = 384")

    k.par(r"The block is five convolutions: a 1×1 projection, a 3×3 depthwise, "
          r"a second 1×1, then a feed-forward pair whose first 1×1 expands to "
          r"4C and whose gated activation folds 4C back to C for free, so the "
          r"closing 1×1 is C→C rather than 4C→C. Table " + str(k.peek_tbl()) +
          r" is the count, taken by forward hooks on the real modules at the "
          r"real tensor shapes.")

    prof = {r["name"]: r for r in ac["block"]["profile"]}
    rows = [["Operator", "Kernel", "Out ch.", "MAC/px", "in C²", "Share"]]
    label = {"dc.0": "1×1 in", "dc.2": "3×3 depthwise", "dc.3": "1×1 out",
             "ffn.0": "FFN 1×1 C→4C", "ffn.2": "FFN 1×1 C→C"}
    for nm in ("dc.0", "dc.2", "dc.3", "ffn.0", "ffn.2"):
        r = prof[nm]
        rows.append([label[nm], f"{r['kernel']}×{r['kernel']}",
                     f"{r['out_channels']}", f"{int(r['mac_per_px']):,}",
                     f"{r['mac_per_px'] / C ** 2:.4f}",
                     f"{r['share_of_block']:.4f}"])
    rows.append(["<b>Block</b>", "", "",
                 f"<b>{int(ac['block']['mac_per_px']):,}</b>",
                 f"<b>{ac['block']['in_C_squared']:.4f}</b>", "<b>1.0000</b>"])
    t_block = k.rows(rows,
        r"One DepthConvBlock at C = 384, by operator. The block is "
        r"7C² + 9C = 1,035,648 MAC/px. Take from it that the 3×3 depthwise, "
        r"the only operator in the whole trunk with a spatial footprint, is "
        r"0.33\% of the block: everything else is 1×1 and has no neighbours to "
        r"miss, which is why a tile boundary is cheap to repair and why the "
        r"trunk can be cut into tiles at all.")
    k.note(r"Source: results/adapter_cost.json, forward hooks on the modules of "
           r"runs/RECIPE512/ckpt\_PAPER.pth.tar. Shares are of the block.")

    k.eq(r"M_{\mathrm{block}} = C^2 + 9C + C^2 + 4C^2 + C^2 = 7C^2 + 9C")

    k.par(r"The cost model shipped in flexuf/cost.py writes the block in closed "
          r"form as 8C² + 9C = " + f"{v['blk_closed']:,}" + r" MAC/px, one C² "
          r"too many, which overstates it by " +
          f"{100 * (v['blk_closed'] / v['blk_meas'] - 1):.2f}" + r"\%. Every "
          r"adapter and the seam-repair pass are priced as fractions of a "
          r"block, so the error propagates to all of them; Section " +
          f"{sec}.5 " + r"measures exactly how far.")

    # ------------------------------------------------------------------
    k.h2("The two adapters")

    k.par(r"An exit that leaves early hands a feature map to the head that the "
          r"remaining blocks would have refined. The adapter is what stands in "
          r"for them. Two kinds are used. The 1×1 adapter is Ad(f) = f + Wf "
          r"with W a C×C matrix, so it costs C² MAC/px. The FFN adapter "
          r"repeats the block's feed-forward pair: a 1×1 expanding C→4C, the "
          r"same gated activation, and a 1×1 C→C. Both are zero-initialised, "
          r"so the ladder is the released decoder exactly at step zero and any "
          r"quality the adapters add is attributable to them.")

    ad = ac["adapters"]
    rows = [["Module", "Operators", "MAC/px", "in C²", "Blocks", "Of decode"]]
    rows.append(["1×1 adapter", "one 1×1 C→C",
                 f"{int(ad['conv1x1']['mac_per_px']):,}",
                 f"{ad['conv1x1']['in_C_squared']:.4f}",
                 f"{ad['conv1x1']['blocks']:.4f}",
                 f"{100 * v['a1D']:.3f}\\%"])
    for pr, lab in zip(ad["ffn"]["profile"],
                       ["FFN adapter, 1×1 C→4C", "FFN adapter, 1×1 C→C"]):
        rows.append([lab, "", f"{int(pr['mac_per_px']):,}",
                     f"{pr['mac_per_px'] / C ** 2:.4f}",
                     f"{pr['share_of_block']:.4f}", ""])
    rows.append(["<b>FFN adapter, total</b>", "",
                 f"<b>{int(ad['ffn']['mac_per_px']):,}</b>",
                 f"<b>{ad['ffn']['in_C_squared']:.4f}</b>",
                 f"<b>{ad['ffn']['blocks']:.4f}</b>",
                 f"<b>{100 * v['affD']:.3f}\\%</b>"])
    t_ad = k.rows(rows,
        r"The two adapter kinds. Take from it that the FFN adapter costs 5C², "
        r"not 2C²: the gated activation is free, so the pair is 4C² + C² and "
        r"not 4C² + 4C². It is 0.71 of a whole block, which makes it much the "
        r"largest single correction in this section. The 1×1 adapter is 0.14 "
        r"of a block, nearer one seventh than the one eighth the closed form "
        r"implies.")
    k.note(r"Source: results/adapter\_cost.json, same hooks and same "
           r"checkpoint as Table " + f"{t_block}" + r". The ``of decode'' column is the "
           r"adapter priced against a released decode, using the "
           r"per-block share derived in Section " + f"{sec}.3" + r".")

    k.par(r"The ladder assigns the FFN adapter to any exit skipping four or "
          r"more blocks and the 1×1 adapter otherwise, which at K = 6 and "
          r"b = 2 means the FFN at exits 0 to 3, the 1×1 at exit 4, and no "
          r"adapter at exit 5, whose feature is the released decoder's own. "
          r"Since the routed allocation lives at exits 2 and 3, the correction "
          r"from 2C² to 5C² lands on precisely the exits the reported saving "
          r"leans on.")

    # ------------------------------------------------------------------
    k.h2("From the block to the decoder")

    k.par(r"Write s<sub>up</sub>, s<sub>trunk</sub> and s<sub>head</sub> for "
          r"the shares of a released decode taken by the opening upsample, the "
          r"12-block trunk and the head; a<sub>k</sub> for the adapter at exit "
          r"k; and r for the seam-repair pass, which runs once over the "
          r"stitched canvas. A tile assigned an exit shallower than j still "
          r"runs group j, because the decoder clamps the map, so the cost of "
          r"exit k in units of one released decode is")

    k.eq(r"c_k = s_{\mathrm{up}} + s_{\mathrm{head}} + r + "
         r"s_{\mathrm{trunk}}\frac{(\max(k,j)+1)b}{N} + a_{\max(k,j)}")

    k.par(r"and the frame-level cost of an exit map a over T tiles amortises "
          r"the shared part over the frame rather than over the tile,")

    k.eq(r"C(a) = s_{\mathrm{up}} + s_{\mathrm{trunk}}\frac{jb}{N} + "
         r"s_{\mathrm{head}} + r + \frac{1}{T}\sum_{i=1}^{T} u_{a(i)}")

    k.par(r"with u<sub>k</sub> the per-tile suffix, that is the blocks after "
          r"the split plus the adapter. Only the last term depends on the map. "
          r"That is the structural reason a ladder over a decoder of this shape "
          r"has a ceiling well below 100\%: the stem, the head and the repair "
          r"pass run whatever the map says.")

    k.par(r"The shares themselves need not be taken on trust. "
          r"results/static\_RECIPE512\_b01.json records, on the pinned "
          r"checkpoint, the frame-level saving of a map that sends every tile "
          r"to exit e, for e = 2, 3, 4 and 5. Four numbers give three "
          r"differences and a constant, and those four quantities are the "
          r"whole model. One trunk block is " + f"{p:.6f}" + r" of a released "
          r"decode, so the twelve of them are " + f"{12 * p:.4f}" + r"; the "
          r"always-on remainder, upsample plus head plus repair, is " +
          f"{v['B'][5] - 8 * p:.6f}" + r"; and the two adapters come out at " +
          f"{v['a1']:.6f}" + r" and " + f"{v['aff']:.6f}" + r", which are the "
          r"prices the shipped model pays for them. Repricing those two and "
          r"the repair pass on the measured block of Table " + f"{t_block}" +
          r" raises them to " + f"{v['a1D']:.6f}" + r", " +
          f"{v['affD']:.6f}" + r" and " + f"{v['repD']:.6f}" + r", which is "
          r"the derived column throughout this section. Table " +
          str(k.peek_tbl()) +
          r" lays the stages out against the wall clock.")

    L = k.J("supp_latency_1920x1080.json")
    S = k.J("supp_latency_1280x720.json")
    lm = {s["stage"]: s["ms"] for s in L["stages"]}
    sm = {s["stage"]: s["ms"] for s in S["stages"]}
    grp = [("group 2  (40 tiles)", "group 2  (15 tiles)", 40, 15),
           ("group 3  (28 tiles)", "group 3  (10 tiles)", 28, 10),
           ("group 4  (14 tiles)", "group 4  (5 tiles)", 14, 5),
           ("group 5  (9 tiles)", "group 5  (3 tiles)", 9, 3)]
    rows = [["Stage", "MAC \\%", "1080p tiles", "1080p ms",
             "720p tiles", "720p ms"]]
    rows.append(["Stem: upsample + groups 0, 1", f"{100 * v['stem']:.2f}",
                 "frame", f"{lm['stem (upsample + groups 0..j-1)']:.2f}",
                 "frame", f"{sm['stem (upsample + groups 0..j-1)']:.2f}"])
    rows.append(["Patchify", "0.00", "", f"{lm['patchify']:.2f}", "",
                 f"{sm['patchify']:.2f}"])
    for i, (gl, gs, nl, ns) in enumerate(grp):
        rows.append([f"Group {i + 2} (2 blocks, per tile)",
                     f"{100 * 2 * p:.2f}", f"{nl}", f"{lm[gl]:.2f}",
                     f"{ns}", f"{sm[gs]:.2f}"])
    rows.append(["Unpatchify", "0.00", "", f"{lm['unpatchify']:.2f}", "",
                 f"{sm['unpatchify']:.2f}"])
    rows.append(["Seam repair (full-frame)", f"{100 * v['repD']:.2f}", "frame",
                 f"{lm['seam repair (384ch, full-frame)']:.2f}", "frame",
                 f"{sm['seam repair (384ch, full-frame)']:.2f}"])
    rows.append(["Head", "2.40", "frame", f"{lm['head']:.2f}", "frame",
                 f"{sm['head']:.2f}"])
    t_stage = k.rows(rows,
        r"Where a decode goes, in arithmetic and in milliseconds. The MAC "
        r"column is each stage's share of a released decode at full occupancy, "
        r"derived as above; it sums to 1.0109, which is the deepest exit's cost "
        r"and is why the ladder is about 1\% dearer than the release when "
        r"nothing is skipped. Take from the table that patchify and unpatchify "
        r"are free arithmetically and nearly free on the clock, that the "
        r"repair pass is 1.09\% of the arithmetic, and that the head takes "
        r"5.1\% of the routed clock for 2.40\% of the arithmetic, being "
        r"bandwidth-bound rather than arithmetic-bound.")
    k.note(r"Milliseconds: results/supp\_latency\_1920x1080.json and "
           r"results/supp\_latency\_1280x720.json, pinned checkpoint, q32, "
           r"0.1 dB budget, NVIDIA RTX A6000 taken under a lock so the card was "
           r"not shared. The tile counts are that decode's own exit map, which "
           r"leaves 28, 14 and 9 of 40 tiles alive in groups 3, 4 and 5. The "
           r"head's 2.40\% is the one share in the table that no file in "
           r"results/ pins; see Section " + f"{sec}.9" + r".")

    k.par(r"Patchify and unpatchify are reshapes and cost no arithmetic at "
          r"all. They require the feature map to divide by the 32 px feature "
          r"tile, which the encoder-side padding guarantees: the RGB frame is "
          r"replicate-padded at the bottom and right before encoding, so 1920×"
          r"1080 becomes 2048×1280, a 128×80 latent, a 256×160 feature map and "
          r"40 tiles of 32×32; 1280×720 becomes 1280×768 and 15 tiles "
          r"(results/tile\_definition.json). One feature pixel covers 8×8 RGB "
          r"pixels, so a 256 px tile is 32 feature px and 16 latent px.")

    rl = k.J("router_latency.json")
    k.par(r"The router head, which reads the shared stem map and predicts a "
          r"per-tile exit, is \RouterParams parameters and "
          r"\RouterCostPct\% of decode MACs. On the clock it is "
          r"\RouterTimePct\% of decode time, \RouterTimeFactor× its "
          r"arithmetic share, an extra \RouterTimeExtra points of decode "
          r"time bought by a launch of a small kernel over a large map. It is "
          r"small enough on either measure to leave the accounting unchanged, "
          r"and it does not run at all in the signalled configuration, where "
          r"the encoder chooses the map.")
    k.note(r"results/router\_latency.json: 40 iterations at 2048×1280 on an "
           r"NVIDIA RTX A6000. Measured on runs/RECIPE512/ckpt\_eval.pth.tar, "
           r"not the pinned checkpoint; the router head is the same size in "
           r"both.")

    # ------------------------------------------------------------------
    k.h2("The per-exit cost vector and the ceiling")

    rows = [["Exit k", "Blocks run", "Adapter", "c (2C²)", "c (5C²)",
             "c derived", "Saved \\%"]]
    for e in range(6):
        run = max(e, 2)
        kind = "none" if run == 5 else ("FFN" if (5 - run) * 2 >= 4 else "1×1")
        nm = f"{e}" if e >= 2 else f"{e} (clamped)"
        rows.append([nm, f"{(run + 1) * 2}", kind,
                     f"{v['A'][e]:.4f}", f"{v['B'][e]:.4f}",
                     f"{v['D'][e]:.4f}", f"{100 * (1 - v['D'][e]):.2f}"])
    t_exit = k.rows(rows,
        r"The per-exit cost vector, in units of one released decode, under "
        r"three prices for the same architecture: the first shipped table, "
        r"which put the FFN adapter at 2C²; the table the paper reports, which "
        r"corrected that to 5C² but kept the closed-form block; and the "
        r"derivation of Tables " + f"{t_block}" + r" and " + f"{t_ad}" +
        r", which prices both on the measured block. Take from it that the two "
        r"corrections move the shallowest reachable exit by 2.79 and then 0.80 "
        r"points, and that they leave exits 4 and 5 almost untouched, because "
        r"those wear the cheap adapter or none.")
    k.note(r"Exits 0 and 1 are unreachable: the decoder clamps the map at j = "
           r"2, so a tile nominally assigned them leaves through exit 2 and "
           r"wears exit 2's cost. The 2C² and 5C² columns are the vectors "
           r"recorded in results/why\_qp.json and "
           r"results/static\_RECIPE512\_b01.json respectively.")

    sat = k.J("saturation_RECIPE512_ctc53.json")
    band = k.J("band_collapse.json")
    k.par(r"The ceiling is 100(1 − c<sub>j</sub>), the saving when every tile "
          r"takes the shallowest reachable exit. It is a property of the "
          r"architecture and no allocation can pass it. Three values are in "
          r"circulation and they are the three columns of Table " +
          f"{t_exit}" + r": " + f"{band['ceiling_pct']:.2f}" +
          r"\% at 2C² (results/band\_collapse.json), \Ceiling\% at 5C² "
          r"(results/saturation\_RECIPE512\_ctc53.json, on the pinned "
          r"checkpoint, and the value the paper's tables use), and " +
          f"{100 * (1 - v['D'][2]):.2f}" + r"\% on the measured block. The "
          r"last of the three is the one a hook count agrees with, as the next "
          r"subsection shows.")

    # ------------------------------------------------------------------
    k.h2("Predicted against measured")

    k.par(r"A cost model and the decoder it models drift. The check that "
          r"catches it is a hook count: flexuf/measure.py attaches a forward "
          r"hook to every convolution and linear layer and accumulates MACs "
          r"from the output shape of each call. A group skipped by an early "
          r"exit never fires its hook and so costs nothing; a group run on 28 "
          r"of 40 tiles costs 28 tiles' worth, because the tiles are the batch "
          r"dimension and the output shape carries them. There is no "
          r"bookkeeping to get wrong, and the count is the ground truth "
          r"against which the model is judged.")

    rows = [["Condition", "Map 2/3/4/5", "Measured", "at 5C²", "diff",
             "Derived", "diff"]]
    lab = {"1280x768": "720p", "2048x1280": "1080p"}
    for r, m in _maps(k):
        n = float(r["n_tiles"])
        cB = v["B"][5] - 8 * p + sum(
            c * (v["B"][e] - (v["B"][5] - 8 * p)) for c, e in
            zip(m, (2, 3, 4, 5))) / n
        cD = v["D"][5] - 8 * p + sum(
            c * (v["D"][e] - (v["D"][5] - 8 * p)) for c, e in
            zip(m, (2, 3, 4, 5))) / n
        sB, sD, sm_ = 100 * (1 - cB), 100 * (1 - cD), r["mac_saving_pct"]
        rows.append([f"{lab[r['size']]} q{r['qp']}",
                     "/".join(str(x) for x in m),
                     f"{sm_:.4f}", f"{sB:.4f}", f"{sB - sm_:+.3f}",
                     f"{sD:.4f}", f"{sD - sm_:+.4f}"])
    t_pm = k.rows(rows,
        r"Predicted against measured, on six decodes with six different exit "
        r"maps at two resolutions. ``Measured'' is the hook count off the "
        r"decode that was executed and timed. Take from it that the derivation "
        r"of Tables " + f"{t_block}" + r" to " + f"{t_exit}" + r" reproduces "
        r"the count to four decimal places in every condition, while the "
        r"model the paper reports is optimistic by 0.47 to 0.80 points, always "
        r"in the same direction.")
    k.note(r"results/supp\_power.json, pinned checkpoint, 0.1 dB budget, "
           r"NVIDIA RTX A6000. The map is the file's pooled exit histogram "
           r"rounded to the frame's tile count, which is what "
           r"scripts/power\_profile.py decodes; the four columns of the map are "
           r"the tiles leaving at exits 2, 3, 4 and 5.")

    rec, resid = _recover(k)
    k.par(r"Because the count is exactly linear in the exit map, six mixtures "
          r"spanning four reachable exits over-determine the per-exit vector, "
          r"and it can simply be solved for. The six equations turn out to be "
          r"consistent to machine precision, the largest residual being " +
          _sci(resid) +
          r" points of saving, so the recovered vector in Table " +
          str(k.peek_tbl()) + r" is a measurement and not a fit.")

    rows = [["Exit k", "c recovered", "c derived", "diff (10<super>-6</super>)",
             "Saved \\%", "Reported \\%"]]
    for e in (2, 3, 4, 5):
        rows.append([f"{e}", f"{rec[e]:.6f}", f"{v['D'][e]:.6f}",
                     f"{(rec[e] - v['D'][e]) * 1e6:+.1f}",
                     f"{100 * (1 - rec[e]):.4f}",
                     f"{100 * (1 - v['B'][e]):.4f}"])
    t_rec = k.rows(rows,
        r"The per-exit cost vector recovered from the six hook counts of "
        r"Table " + f"{t_pm}" + r" by solving the six linear equations, "
        r"against the derivation. Take from it that the derivation and the "
        r"measurement agree to a few parts in a million at every exit, and "
        r"that the model the paper reports overstates the saving by 0.80 "
        r"points at exits 2 and 3, 0.27 at exit 4 and 0.14 at exit 5.")
    k.note(r"Recovered by least squares from the six (map, measured saving) "
           r"pairs of results/supp\_power.json; the system is consistent, so "
           r"the solution is exact rather than fitted. ``Reported'' is the "
           r"5C² column of Table " + f"{t_exit}" + r".")

    grid = k.J("signalled_RECIPE512_grid.json")
    qs = sorted({r["qp"] for r in grid["rows"]})
    bs = sorted({r["budget_db"] for r in grid["rows"]})
    tab = {(r["budget_db"], r["qp"]): r for r in grid["rows"]}
    rows = [["Budget (dB)"] + [f"q{q}" for q in qs]]
    for b in bs:
        line = [f"{b:g}"]
        for q in qs:
            r = tab.get((b, q))
            d = r.get("model_minus_measured") if r else None
            line.append(f"{d:.3f}" if d is not None else "n/a")
        rows.append(line)
    t_grid = k.rows(rows,
        r"Model minus hook count, in points of saving, over the whole budget "
        r"grid: 9 budgets by 5 quality indices, 53 sequences, 2 frames each. "
        r"Take from it that the residual is positive everywhere, that it grows "
        r"with the budget, and that it saturates at exactly 0.797, which is "
        r"the residual Table " + f"{t_rec}" + r" predicts for an exit-2 map. "
        r"Every entry lies inside the interval [0.135, 0.797] that the "
        r"per-exit residuals bound it to. n/a marks a budget below the floor, "
        r"where no allocation is admissible.")
    k.note(r"results/signalled\_RECIPE512\_grid.json, pinned checkpoint. Both "
           r"columns are in the file: saving\_pct\_vs\_release is the model, "
           r"saving\_pct\_measured is the hook count, and "
           r"model\_minus\_measured is their difference.")

    k.par(r"The residual is therefore not noise and not a tolerance. It is the "
          r"one C² of block that the closed form counts twice, propagated "
          r"through the adapters and the repair pass, and it is largest "
          r"exactly where the saving is largest. Reading the paper's headline "
          r"figures as hook counts means subtracting between 0.43 and 0.80 "
          r"points: at the 0.1 dB budget on the pinned checkpoint the model "
          r"reports \MainLowRate\% at the lowest rate against a measured "
          r"29.13\%, and \MainHighRate\% at the highest against a measured "
          r"15.16\% (results/signalled\_RECIPE512\_ctc53.json).")

    # ------------------------------------------------------------------
    k.h2("What the saving is worth in seconds")
    k.fig("power.png",
          "<b>Three units for one saving.</b> <b>a</b>, the same routed "
          "decode measured as arithmetic, as wall clock and as joules "
          "per frame. <b>b</b>, why they differ: the board draws the "
          "same power either way, because the later groups run on a "
          "shrinking set of tiles and a partly idle GPU still draws its "
          "static power. <b>c</b>, peak memory, which routing raises "
          "rather than lowers.")

    k.par(r"Arithmetic is the right unit to allocate in, because it is the "
          r"only one independent of the machine, the driver and whatever else "
          r"is resident on the card. It is not the unit a deployment cares "
          r"about. DCVC-RT [33] makes the point for neural codecs generally, "
          r"arguing that non-computational operational cost rather than "
          r"arithmetic is the primary bottleneck, and the complexity-aware "
          r"line of work [35, 36] treats decode cost as an axis to be "
          r"measured rather than inferred. Table " + str(k.peek_tbl()) +
          r" is the stage evidence for why the two diverge here.")

    rows = [["Group", "1080p tiles", "1080p ms/tile", "720p tiles",
             "720p ms/tile"]]
    for i, (gl, gs, nl, ns) in enumerate(grp):
        rows.append([f"{i + 2}", f"{nl}", f"{lm[gl] / nl:.4f}",
                     f"{ns}", f"{sm[gs] / ns:.4f}"])
    t_occ = k.rows(rows,
        r"Milliseconds per tile in each per-tile group, as the group's tile "
        r"set shrinks. Take from it that a tile costs more to decode in a "
        r"group that has fewer tiles left: 5.5\% more at 1080p from group 2 to "
        r"group 5, 8.0\% more at 720p from group 2 to group 4. The last groups "
        r"launch kernels over a handful of 32×32 feature maps, and a GPU sized "
        r"for the first group is partly idle inside them.")
    k.note(r"Derived from results/supp\_latency\_1920x1080.json and "
           r"results/supp\_latency\_1280x720.json by dividing each group's "
           r"measured time by the tiles alive in it, both recorded in the "
           r"files.")

    pw = k.J("supp_power.json")
    rows = [["Condition", "MACs \\%", "Time \\%", "Gap", "Energy \\%",
             "Above idle \\%"]]
    for r in pw["rows"]:
        rows.append([f"{lab[r['size']]} q{r['qp']}",
                     f"{r['mac_saving_pct']:.2f}",
                     f"{r['time_saving_pct']:.2f}",
                     f"{r['mac_saving_pct'] - r['time_saving_pct']:.2f}",
                     f"{r['energy_saving_pct']:.2f}",
                     f"{r['energy_saving_above_idle_pct']:.2f}"])
    t_three = k.rows(rows,
        r"The three units side by side, all as savings of the routed decode "
        r"against the released decode on the same latent. Take from it the "
        r"result of this subsection: energy saved tracks time saved to within "
        r"0.5 points in every condition, while arithmetic saved runs 3.5 to "
        r"5.0 points ahead of both. Subtracting idle power changes nothing, "
        r"which is the check that the agreement is not an artefact of the "
        r"static draw.")
    k.note(r"results/supp\_power.json, pinned checkpoint, 0.1 dB budget, "
           r"NVIDIA RTX A6000 with a 300 W limit, taken under the evaluation "
           r"lock. 720p is padded to 1280×768 and 15 tiles, 1080p to 2048×1280 "
           r"and 40 tiles.")

    k.par(r"The gap between arithmetic and the clock is the shrinking tile "
          r"set of Table " + f"{t_occ}" + r", plus the fixed cost of the "
          r"tiled path itself: patchify, unpatchify and the repair pass "
          r"together are 2.9\% of the routed clock at 1080p and cost nothing "
          r"in the released decode. The gap is largest where the map is most "
          r"lopsided, which is the low-rate end, and that is also where the "
          r"arithmetic saving is largest.")

    # ------------------------------------------------------------------
    k.h2("What the saving is worth in joules")

    k.par(r"Board power was sampled from the driver at 100 ms, which is about "
          r"the rate the sensor updates. Each condition therefore runs for a "
          r"fixed wall time of 20 s rather than a fixed iteration count, so "
          r"that a 30 ms decode yields two hundred power samples rather than "
          r"three. Idle is measured for 4 s immediately before and after every "
          r"timed run on the same card, so a neighbouring process waking up "
          r"mid-measurement appears as a drift between the two rather than as "
          r"a saving. Energy per frame is the mean board power over the run "
          r"times the median decode time.")

    rows = [["Condition", "ms full", "ms routed", "W full", "W routed",
             "J full", "J routed"]]
    for r in pw["rows"]:
        rows.append([f"{lab[r['size']]} q{r['qp']}",
                     f"{r['full']['ms_median']:.2f}",
                     f"{r['routed']['ms_median']:.2f}",
                     f"{r['full']['watts_mean']:.1f}",
                     f"{r['routed']['watts_mean']:.1f}",
                     f"{r['joules_per_frame_full']:.3f}",
                     f"{r['joules_per_frame_routed']:.3f}"])
    t_abs = k.rows(rows,
        r"The measurement behind Table " + f"{t_three}" + r", in absolute "
        r"units. Take from it why energy tracks time: the routed decode draws "
        r"the same board power as the released one, within 3 W in the worst "
        r"case and within 1 W in four of the six conditions, and at 720p q0 it "
        r"draws slightly <i>more</i>. Routing does not lower the instantaneous "
        r"draw of the card. It shortens the job.")
    k.note(r"results/supp\_power.json. Watts are the mean of about 200 driver "
           r"samples per run; times are medians over 181 to 664 iterations. "
           r"Idle over the six conditions in the order run: 74.6, 78.5, 91.2, "
           r"100.5, 106.5 and 98.6 W before, and 122.4, 120.0, 126.2, 128.4, "
           r"125.0 and 121.9 W after, on a card whose decode draw is 284 to "
           r"298 W.")

    k.par(r"That is the honest engineering result and it should be stated "
          r"plainly: the energy saving of this method tracks its time saving, "
          r"not its arithmetic saving. A partly idle GPU still draws most of "
          r"its static power, so the joules follow the seconds almost exactly, "
          r"and the seconds fall short of the multiply-accumulates for the "
          r"occupancy reason of Table " + f"{t_occ}" + r". At 1080p and the "
          r"lowest rate the three read 32.98\%, 29.07\% and 29.56\%; at the "
          r"highest rate they read 17.85\%, 14.14\% and 14.17\%. Anyone "
          r"quoting the arithmetic figure as an energy figure overstates it by "
          r"three and a half to five points.")

    k.par(r"Two limits on these numbers. The idle baseline drifts upward "
          r"across the six conditions as the card warms, from 74.6 W before "
          r"the first to 128.4 W after the fourth, so the above-idle column of "
          r"Table " + f"{t_three}" + r" is the weaker of the two energy "
          r"figures; that it agrees with the unadjusted column to within 0.4 "
          r"points is the reason the conclusion survives. And power comes from "
          r"the driver counter, not from an external meter, so the absolute "
          r"joules inherit whatever bias that sensor has. The savings are "
          r"ratios of two measurements taken the same way on the same card "
          r"minutes apart, which is the quantity the bias mostly cancels in.")

    # ------------------------------------------------------------------
    k.h2("Memory, and what the encoder pays")

    fp = k.J("supp_footprint.json")
    rows = [["Frame (padded)", "Tiles", "Peak MB, full", "Peak MB, routed",
             "Change \\%", "Time saved \\%"]]
    for r in fp["rows"]:
        if r.get("oom"):
            rows.append([r["size"], f"{r['n_tiles']}", "out of memory", "", "",
                         ""])
        else:
            rows.append([r["size"], f"{r['n_tiles']}",
                         f"{r['peak_mb_full']:.1f}",
                         f"{r['peak_mb_routed']:.1f}",
                         f"+{r['peak_delta_pct']:.2f}",
                         f"{r['time_saving_pct']:.2f}"])
    t_mem = k.rows(rows,
        r"Peak decoder memory, released against routed. Take from it that "
        r"routing costs memory rather than saving it, by a constant 13.33\% at "
        r"every resolution that fits, because the tiled path holds the whole "
        r"tile batch and the stitched canvas alive together. The 4K row is a "
        r"genuine failure and is reported as one: the job ran out of memory on "
        r"a 48 GB card that four other processes were sharing, so no "
        r"measurement above 1080p exists anywhere in this work.")
    k.note(r"results/supp\_footprint.json, pinned checkpoint, q32, 0.1 dB "
           r"budget, NVIDIA RTX A6000. The 4K stage profile queued alongside it "
           r"failed the same way (results/supp\_queue.log).")

    ec = k.J("encoder_cost.json")
    k.par(r"The signalled configuration moves the choice of exit map to the "
          r"encoder, which has to price the tiles before it can allocate them. "
          r"On one 1080p sequence at the highest rate, one decode takes " +
          f"{ec['ms_one_decode']:.1f}" + r" ms, building the cost table "
          r"full-frame takes " + f"{ec['ms_full_frame_table']:.1f}" + r" ms "
          r"(" + f"{ec['x_full_frame']:.3f}" + r"× one decode), and building "
          r"it the way the deployed encoder does, per tile, takes " +
          f"{ec['ms_deployed_table']:.1f}" + r" ms (" +
          f"{ec['x_deployed']:.3f}" + r"×). The two searches agree on " +
          f"{ec['approx']['agreement']:.3f}" + r" of tiles and reach savings "
          r"of " + f"{ec['exact']['saving']:.2f}" + r"\% and " +
          f"{ec['approx']['saving']:.2f}" + r"\% at the same budget. The "
          r"asymmetry is the design: a decoder-side saving is bought with "
          r"encoder-side work, which suits a compress-once, decode-many "
          r"deployment and does not suit live encoding.")
    k.note(r"results/encoder\_cost.json: one sequence (Bosphorus), q63, 40 "
           r"tiles. The file records no checkpoint field, so this is the one "
           r"measurement in this section whose checkpoint cannot be stated "
           r"from the file.")

    # ------------------------------------------------------------------
    k.h2("What these files do not pin")

    k.par(r"Four gaps, listed so that they are visible rather than quietly "
          r"absent.")

    k.bullets([
        r"<b>The split of the always-on remainder.</b> Table " +
        f"{t_stage}" + r" gives the head 2.40\% and the opening upsample "
        r"8.16\%. results/ pins their sum, " +
        f"{100 * (v['stem'] - 4 * p + 0.0240):.2f}" + r"\%, and pins "
        r"everything else in the model, but the split between the two is "
        r"produced by scripts/mac\_audit.py, which prints to stdout and writes "
        r"no JSON. The same is true of \IntraGmac GMAC, the absolute figure "
        r"every row of the main paper's complexity table is normalised by. "
        r"Neither affects any saving, which is a ratio, but neither is "
        r"reproducible from results/ as it stands.",

        r"<b>The paper's tables are on the 5C² column.</b> Table " +
        f"{t_exit}" + r" and Table " + f"{t_grid}" + r" together say by how "
        r"much: 0.14 to 0.80 points of saving, always optimistic, largest "
        r"where the saving is largest. The measured-block column is the one a "
        r"hook count agrees with, and the honest ceiling is " +
        f"{100 * (1 - v['D'][2]):.2f}" + r"\% rather than \Ceiling\%.",

        r"<b>Nothing above 1080p.</b> Both 4K jobs ran out of memory (Table " +
        f"{t_mem}" + r"). The resolution trend in this section rests on three "
        r"points, of which two carry power measurements.",

        r"<b>One card, one budget, one card-level power sensor.</b> Every "
        r"millisecond and joule here is an NVIDIA RTX A6000 at a 0.1 dB "
        r"budget; the stage profiles are q32 only, and the power runs cover "
        r"three quality indices at two resolutions. Nothing here separates "
        r"the decoder's energy from the board's, and no external meter was "
        r"available to calibrate the driver counter against.",
    ])
