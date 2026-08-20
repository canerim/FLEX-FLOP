"""Architecture and complexity accounting.

The section derives what a decode costs from one convolution upward, and then
checks the derivation against MAC counts taken by forward hooks off decodes
that were actually executed. Three sources carry it:

  results/mac_audit.json          every leaf convolution of the released
                                  DCVC-UF intra decoder, by forward hook, with
                                  kernel, channels, groups, input and output
                                  grid and MAC, at three resolutions. Written by
                                  scripts/mac_audit_json.py, which is
                                  scripts/mac_audit.py with its stdout report
                                  replaced by a JSON dump; CPU only, because a
                                  MAC count needs shapes and not weights
  results/adapter_cost.json       the same treatment for one DepthConvBlock and
                                  for the two adapter kinds, on the pinned
                                  checkpoint
  results/supp_power.json,        hook counts and wall clock off real decodes,
  results/signalled_*_grid.json   which is what the derivation is judged against

Nothing in this module reads a checkpoint, opens a CUDA context or writes to
results/. Every number below is arithmetic on those JSON files, so a reader
with the repository can rerun it. The three figures are drawn from the same
files by scripts/supp_b_figures.py.

The one structural fact that makes the section short: because a MAC count is
linear in the exit map and the hook meter counts a module only when it fires,
the per-module counts of results/mac_audit.json reassemble into the per-decode
counts of results/supp_power.json exactly, not approximately. The residual is
floating point, and the check is therefore an identity rather than a fit.
"""

C_TRUNK = "trunk_channels"


# --------------------------------------------------------------------------
# the pieces, all read rather than typed
# --------------------------------------------------------------------------
def _parts(k):
    """Every quantity the cost model is built from, in one dictionary.

    MACs are absolute, at the padded 1920x1088 grid, so that a share is a
    ratio of two counts taken under the same convention rather than a
    calibrated constant. `mac_audit.json` records the same three shares at
    1280x768 and 832x512 as well and they agree to every digit, because every
    operator in the decoder runs at a fixed multiple of the latent grid.
    """
    ma = k.J("mac_audit.json")["1920x1088"]
    ac = k.J("adapter_cost.json")
    C = ac[C_TRUNK]
    gh, gw = ma["layers"][1]["out_hw"]          # the feature grid, 136 x 240
    fpx = gh * gw
    tot = float(ma["total_mac"])
    blk = ac["block"]["mac_per_px"]
    return dict(
        ma=ma, ac=ac, C=C, gh=gh, gw=gw, fpx=fpx, tot=tot,
        nb=ma["n_trunk_blocks"],
        blk_macpx=blk,
        blk_closed=8 * C * C + 9 * C,
        rep_macpx=9 * C + C * C,
        a1_macpx=ac["adapters"]["conv1x1"]["mac_per_px"],
        aff_macpx=ac["adapters"]["ffn"]["mac_per_px"],
        s_up=ma["parts"]["upsample"]["share"],
        s_trunk=ma["parts"]["trunk"]["share"],
        s_head=ma["parts"]["head"]["share"],
        p=ma["parts"]["trunk"]["share"] / ma["n_trunk_blocks"],
        share=lambda macpx, fpx=fpx, tot=tot: macpx * fpx / tot,
    )


def _hook_vector(P):
    """The per-exit cost vector, in units of one released decode.

    Assembled from the per-module MAC counts of results/mac_audit.json and
    results/adapter_cost.json under the same hook convention the meter uses, so
    this is not a model of the count: it is the count, written as a sum.
    """
    up, hd = P["s_up"], P["s_head"]
    rep = P["share"](P["rep_macpx"])
    a1 = P["share"](P["a1_macpx"])
    aff = P["share"](P["aff_macpx"])
    out = []
    for e in range(6):
        run = max(e, 2)
        ad = 0.0 if run == 5 else (aff if (5 - run) * 2 >= 4 else a1)
        out.append(up + hd + rep + (run + 1) * 2 * P["p"] + ad)
    return out, dict(rep=rep, a1=a1, aff=aff)


def _maps(k):
    """The exit map each power condition actually decoded.

    results/supp_power.json records a pooled exit histogram and a tile count;
    scripts/power_profile.py turns the two into the map it decodes by rounding
    the histogram's proportions to the tile count and giving the remainder to
    the deepest exit. Reproduced here so that a prediction is a prediction
    about the decode that was timed, and the four returned counts are the tiles
    leaving at exits 2, 3, 4 and 5.
    """
    out = []
    for r in k.J("supp_power.json")["rows"]:
        h, n = r["hist"], r["n_tiles"]
        tot = float(sum(h))
        cnt = [int(round(x / tot * n)) for x in h]
        cnt[-1] += n - sum(cnt)
        out.append((r, [cnt[0] + cnt[1] + cnt[2], cnt[3], cnt[4], cnt[5]]))
    return out


def _recover(k, D):
    """Solve the six hook counts of results/supp_power.json for c_2 to c_5.

    A hook count is linear in the exit map, so each condition contributes one
    equation and six conditions over-determine four unknowns. The residual is
    the test: if it is at the level of floating point, the recovered vector is
    a measurement rather than a fit.
    """
    import numpy as np
    A, b = [], []
    for r, m in _maps(k):
        n = float(r["n_tiles"])
        A.append([c / n for c in m])
        b.append(1.0 - r["mac_saving_pct"] / 100.0)
    A, b = np.array(A), np.array(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return ({e: float(sol[i]) for i, e in enumerate((2, 3, 4, 5))},
            float(np.abs(A @ sol - b).max()),
            max(abs(sol[i] - D[e]) for i, e in enumerate((2, 3, 4, 5))))


def _sci(x):
    """3.3e-16 -> "3×10<super>-16</super>". Times-Roman has no superscript
    digits, so the tag is the only way to set one that is not a black box."""
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}×10<super>{int(exp)}</super>"


def _n(x):
    return f"{int(round(x)):,}"


# --------------------------------------------------------------------------
def content(k):
    P = _parts(k)
    ma, ac, C = P["ma"], P["ac"], P["C"]
    D, shr = _hook_vector(P)
    B = k.J("why_qp_PAPER.json")["cost"]
    A2 = k.J("why_qp.json")["cost"]
    td = k.J("tile_definition.json")

    sec = k.h1("Architecture and complexity accounting")

    k.par(r"This section states what the decoder is, module by module and "
          r"convolution by convolution, derives what a decode costs from those "
          r"shapes, and then checks the derivation against counts taken off "
          r"decodes that were run. It closes with what the saving is worth in "
          r"the units a deployment measures, which are milliseconds, joules, "
          r"megabytes and frames per second, and none of which is the unit the "
          r"allocation is optimised in.")

    k.par(r"Some vocabulary first, because the rest of the supplement uses it. "
          r"A <i>multiply-accumulate</i> (MAC) is one multiply and one add, and "
          r"MAC/px is per pixel of the grid the operator runs on, which is not "
          r"the same grid for every operator. The <i>trunk</i> is the released "
          r"decoder's stack of N = " + f"{P['nb']}" + r" identical "
          r"DepthConvBlocks at width C = " + f"{C}" + r". An <i>exit</i> is a "
          r"depth at which a tile may stop; the ladder has K = 6 of them, one "
          r"every b = N/K = 2 blocks. The <i>split depth</i> j = 2 is how many "
          r"exits' worth of blocks run once for the whole frame before any tile "
          r"is allowed to leave, so the first jb = 4 blocks are full-frame and "
          r"the last 8 are per-tile. An <i>adapter</i> is a small "
          r"zero-initialised residual module that stands in for the blocks an "
          r"exit skips. A <i>tile</i> is " + f"{td['rgb_patch']}" + r" RGB px "
          r"square, which is " + f"{td['feature_patch']}" + r" px of the "
          r"feature grid and " + f"{td['latent_patch']}" + r" px of the latent. "
          r"The <i>exit map</i> is the per-tile assignment of exits. The "
          r"<i>released decoder</i> is DCVC-UF's intra decoder [14] with none "
          r"of this added, and it is the unit every cost below is quoted in.")

    # ------------------------------------------------------------------
    k.h2("The decoder, module by module")

    k.par(r"The released intra decoder is " + _n(ma["decoder_params"]) +
          r" parameters in " + f"{len(ma['layers'])}" + r" convolutions and "
          r"nothing else: no attention, no normalisation with parameters of its "
          r"own on the critical path, and no operator that changes the spatial "
          r"grid except two pixel shuffles. It has three parts. An "
          r"<i>upsample</i> takes the decoded latent, widens it with a 1×1 and "
          r"doubles its grid with a pixel shuffle, then runs one DepthConvBlock "
          r"at C = " + f"{C}" + r". A <i>trunk</i> of " + f"{P['nb']}" +
          r" further DepthConvBlocks runs at that width and grid throughout. A "
          r"<i>head</i> narrows to " + f"{ma['ch_preshuffle']}" + r" channels "
          r"with a 1×1, runs one more DepthConvBlock at that narrower width, "
          r"and pixel-shuffles by 8 straight to RGB, with no convolution after "
          r"the shuffle. Table " + str(k.peek_tbl()) + r" is that pipeline with "
          r"its shapes.")

    def mrow(name, where, out_c, gh, gw, macpx, gmac):
        return [name, where, f"{gh}×{gw}", f"{out_c}",
                _n(macpx) if macpx else "0",
                f"{100 * gmac / P['tot']:.2f}"]

    lay = {L["name"]: L for L in ma["layers"]}
    lh, lw = ma["latent"][1], ma["latent"][2]
    blk_gmac = P["blk_macpx"] * P["fpx"]
    rows = [["Module", "Where", "Grid", "Out ch.", "MAC/px", "\\% dec."]]
    rows.append(mrow("Latent y, decoder input", "entropy dec.", ma["ch_latent"],
                     lh, lw, 0, 0))
    rows.append(mrow("1×1, widen", "upsample", 4 * C, lh, lw,
                     lay["dec_1.0.up.conv.0"]["mac"] / (lh * lw),
                     lay["dec_1.0.up.conv.0"]["mac"]))
    rows.append(mrow("PixelShuffle 2", "upsample", C, P["gh"], P["gw"], 0, 0))
    rows.append(mrow("DepthConvBlock", "upsample", C, P["gh"], P["gw"],
                     P["blk_macpx"], blk_gmac))
    rows.append(mrow(f"DepthConvBlock × {P['nb']}", "trunk", C,
                     P["gh"], P["gw"], P["blk_macpx"],
                     P["nb"] * blk_gmac))
    rows.append(mrow("1×1, narrow", "head", ma["ch_preshuffle"],
                     P["gh"], P["gw"], lay["dec_2.adaptor"]["mac"] / P["fpx"],
                     lay["dec_2.adaptor"]["mac"]))
    hd_blk = ma["parts"]["head"]["gmac"] * 1e9 - lay["dec_2.adaptor"]["mac"]
    rows.append(mrow("DepthConvBlock", "head", ma["ch_preshuffle"],
                     P["gh"], P["gw"], hd_blk / P["fpx"], hd_blk))
    rows.append(mrow("PixelShuffle 8", "head", 3, ma["recon"][1], ma["recon"][2],
                     0, 0))
    rows.append(["<b>Ours, added</b>", "", "", "", "", ""])
    rows.append(mrow("Conv1x1Adapter", "exit 4", C, td["feature_patch"],
                     td["feature_patch"], P["a1_macpx"],
                     P["a1_macpx"] * P["fpx"]))
    rows.append(mrow("FFNAdapter", "exits 0 to 3", C, td["feature_patch"],
                     td["feature_patch"], P["aff_macpx"],
                     P["aff_macpx"] * P["fpx"]))
    rows.append(mrow("GridSeamRepair", "post-stitch", C, P["gh"], P["gw"],
                     P["rep_macpx"], P["rep_macpx"] * P["fpx"]))
    rows.append(["Router head", "stem", "per tile", "K = 6", "n/a",
                 "\\RouterCostPct"])
    t_shape = k.rows(rows,
        r"The decode, module by module, at a 1920×1080 frame padded to "
        r"1920×1088. Grids are height × width, and ``Where'' names the released "
        r"decoder's three parts: the upsample dec\_1[0], the trunk "
        r"dec\_1[1..12] and the head dec\_2. The last column is the module's "
        r"share of one released decode, and for the two adapters it is the "
        r"share if every tile wears that adapter. Take from the table that the "
        r"grid changes exactly twice and never inside the trunk, so an exit "
        r"ladder cut into the trunk needs no resampling; that the head is "
        r"cheap because it runs at " + f"{ma['ch_preshuffle']}" + r" channels "
        r"rather than " + f"{C}" + r", which is a quarter of the arithmetic per "
        r"pixel; and that the FFN adapter is the most expensive thing this "
        r"work adds, at " + f"{100 * shr['aff']:.2f}" + r"\% of a decode, more "
        r"than half a trunk block.")
    k.note(r"Released modules and their shapes: results/mac\_audit.json, "
           r"forward hooks on every convolution of an "
           r"IntraDecoder at the real tensor shapes. A MAC count depends on "
           r"shapes and not on values, so no weights are loaded for it. "
           r"Adapters and seam repair: results/adapter\_cost.json, hooks on the "
           r"modules of runs/RECIPE512/ckpt\_PAPER.pth.tar. Router: "
           r"results/router\_latency.json, measured on "
           r"runs/RECIPE512/ckpt\_eval.pth.tar; the head is the same size in "
           r"both checkpoints.")

    k.fig("supp_b_cost.png",
          "<b>Where the arithmetic is, and what an exit saves.</b> "
          "<b>a</b>, one released decode by module: the opening upsample, the "
          "twelve trunk blocks, the head, and the seam-repair pass this work "
          "adds (hatched). Light bars always run; dark bars are the eight "
          "blocks after the split that a tile can skip. <b>b</b>, the saving "
          "at each reachable exit under three prices for the same "
          "architecture, with each price's ceiling dotted. Section " +
          f"{sec}.5 " + "is the table behind it.",
          maxh=1.45 * 72)

    # ------------------------------------------------------------------
    k.h2("Every convolution in the decode")

    shapes = {(L["kernel"], L["in_ch"], L["out_ch"], L["stride"], L["groups"])
              for L in ma["layers"]}
    n_dw = sum(1 for L in ma["layers"] if L["kernel"] > 1)
    k.par(r"Table " + str(k.peek_tbl()) + r" opens up the modules of Table " +
          f"{t_shape}" + r", giving every convolution in the decode in the "
          r"(K, C<sub>in</sub>, C<sub>out</sub>, S) convention "
          r"that the architecture figures of this literature use, with the "
          r"group count added because two of the shapes are depthwise. The " +
          f"{len(ma['layers'])}" + r" convolutions of the released decoder have "
          r"only " + f"{len(shapes)}" + r" distinct shapes between them, "
          r"because the DepthConvBlock repeats.")

    def crow(nm, L, used, denom):
        return [nm, f"{L['kernel']}", f"{L['in_ch']}", f"{L['out_ch']}",
                f"{L['stride']}", f"{L['groups']}",
                _n(L["mac"] / denom), used]

    rows = [["Convolution", "K", "C<sub>in</sub>", "C<sub>out</sub>", "S",
             "Grp", "MAC/px", "Used"]]
    rows.append(crow("up.conv.0", lay["dec_1.0.up.conv.0"], "1", lh * lw))
    for nm, lab in (("dc.0", "dc.0, 1×1 in"), ("dc.2", "dc.2, 3×3 depthwise"),
                    ("dc.3", "dc.3, 1×1 out"), ("ffn.0", "ffn.0, 1×1 C→4C"),
                    ("ffn.2", "ffn.2, 1×1 4C→C")):
        rows.append(crow(lab, lay[f"dec_1.1.{nm}"], f"×{P['nb'] + 1}", P["fpx"]))
    rows.append(crow("dec_2.adaptor", lay["dec_2.adaptor"], "1", P["fpx"]))
    for nm, lab in (("dc.0", "head dc.0"), ("dc.2", "head dc.2, depthwise"),
                    ("dc.3", "head dc.3"), ("ffn.0", "head ffn.0"),
                    ("ffn.2", "head ffn.2")):
        rows.append(crow(lab, lay[f"dec_2.{nm}"], "1", P["fpx"]))
    a1p = ac["adapters"]["conv1x1"]["profile"][0]
    affp = ac["adapters"]["ffn"]["profile"]
    rows.append(["Conv1x1Adapter", "1", f"{C}", f"{C}", "1", "1",
                 _n(a1p["mac_per_px"]), "1"])
    rows.append(["FFNAdapter pw_in", "1", f"{C}", f"{4 * C}", "1", "1",
                 _n(affp[0]["mac_per_px"]), "×4"])
    rows.append(["FFNAdapter pw_out", "1", f"{C}", f"{C}", "1", "1",
                 _n(affp[1]["mac_per_px"]), "×4"])
    rows.append(["Seam repair, 3×3 dw", "3", f"{C}", f"{C}", "1", f"{C}",
                 _n(9 * C), "1"])
    rows.append(["Seam repair, 1×1", "1", f"{C}", f"{C}", "1", "1",
                 _n(C * C), "1"])
    t_layer = k.rows(rows,
        r"Every distinct convolution in the decode. K is the kernel side, S "
        r"the stride, Grp the group count; MAC/px is per pixel of that "
        r"operator's own grid, which is the latent grid for the first row and "
        r"the feature grid for the rest. ``Used'' counts the instances in one "
        r"decode: the DepthConvBlock appears " + f"{P['nb'] + 1}" + r" times, "
        r"once in the upsample and " + f"{P['nb']}" + r" times in the trunk, "
        r"and there are K − 1 = 5 adapters of which four are of the FFN kind. "
        r"Take from it that only two of the shapes have a kernel wider than "
        r"1×1 and both are depthwise, so the " + f"{n_dw}" + r" instances of "
        r"them in one decode carry " +
        f"{100 * ma['spatial_share']:.2f}" + r"\% of the arithmetic between "
        r"them.")
    k.note(r"results/mac\_audit.json for the released rows, "
           r"results/adapter\_cost.json for ours. The two depthwise "
           r"convolutions are the whole of the decoder's spatial footprint, "
           r"which is why a tile boundary is cheap to repair.")

    k.par(r"That last figure is the premise of the whole method, so it is worth "
          r"stating separately: " + f"{100 * ma['pointwise_share']:.2f}" +
          r"\% of the released decoder's arithmetic is 1×1 convolution, which "
          r"has no neighbours to miss. The receptive field of the trunk plus "
          r"head grows by exactly one feature pixel per block, from 1 feature "
          r"pixel with no trunk blocks to 13 with all " + f"{P['nb']}" +
          r", which is 104 RGB px per tile edge (results/dmc\_ld\_rf\_probe.json, "
          r"measured on the released DCVC checkpoints rather than ours). A tile "
          r"decoded alone is therefore wrong only in a band, and the band is "
          r"narrower for a tile that exits early.")

    k.par(r"The two DepthConvBlock widths obey the same closed form. Writing "
          r"the block as a 1×1 projection, a 3×3 depthwise, a second 1×1 and a "
          r"feed-forward pair whose first 1×1 expands to 4C and whose gated "
          r"activation folds 4C back to C for free, the closing 1×1 is C→C "
          r"rather than 4C→C and the block is")

    k.eq(r"M_{\mathrm{block}}(C) = C^2 + 9C + C^2 + 4C^2 + C^2 = 7C^2 + 9C")

    k.par(r"which is " + _n(P["blk_macpx"]) + r" MAC/px at C = " + f"{C}" +
          r" and " + _n(7 * ma["ch_preshuffle"] ** 2 + 9 * ma["ch_preshuffle"]) +
          r" at C = " + f"{ma['ch_preshuffle']}" + r", both matching Table " +
          f"{t_layer}" + r" row by row. The cost model shipped in "
          r"flexuf/cost.py writes it as 8C² + 9C = " + _n(P["blk_closed"]) +
          r" instead, one C² too many, which overstates a block by " +
          f"{100 * (P['blk_closed'] / P['blk_macpx'] - 1):.2f}" + r"\%. "
          r"Sections " + f"{sec}.5 " + r"and " + f"{sec}.6 " + r"follow that "
          r"one term to the end.")

    # ------------------------------------------------------------------
    k.h2("What the ladder adds")

    k.par(r"Three modules are new. An exit that stops early hands the head a "
          r"feature map the remaining blocks would have refined, and an "
          r"<i>adapter</i> stands in for them. The 1×1 adapter is f + Wf with W "
          r"a C×C matrix. The FFN adapter repeats the block's feed-forward "
          r"pair, a 1×1 expanding C→4C, the same gated activation, and a 1×1 "
          r"C→C. Both are zero-initialised, so the ladder is the released "
          r"decoder exactly at step zero and any quality the adapters add is "
          r"attributable to them. Neither has a spatial footprint, so neither "
          r"widens the seam. The <i>seam repair</i> is a gated depthwise 3×3 "
          r"followed by a zero-initialised 1×1, run once on the stitched canvas "
          r"after unpatchify and before the head, which is the only place a 3×3 "
          r"can see across a tile boundary. The <i>router head</i> reads the "
          r"shared stem and predicts a per-tile exit; Section F is its "
          r"architecture and this section only prices it.")

    rows = [["Module", "Operators", "MAC/px", "In C²", "Blocks", "Of decode"]]
    rows.append(["1×1 adapter", "1×1 C→C", _n(P["a1_macpx"]),
                 f"{ac['adapters']['conv1x1']['in_C_squared']:.2f}",
                 f"{ac['adapters']['conv1x1']['blocks']:.4f}",
                 f"{100 * shr['a1']:.3f}\\%"])
    rows.append(["FFN adapter", "1×1 C→4C, 1×1 C→C", _n(P["aff_macpx"]),
                 f"{ac['adapters']['ffn']['in_C_squared']:.2f}",
                 f"{ac['adapters']['ffn']['blocks']:.4f}",
                 f"{100 * shr['aff']:.3f}\\%"])
    rows.append(["Seam repair", "3×3 dw, 1×1 C→C", _n(P["rep_macpx"]),
                 f"{P['rep_macpx'] / C ** 2:.2f}",
                 f"{P['rep_macpx'] / P['blk_macpx']:.4f}",
                 f"{100 * shr['rep']:.3f}\\%"])
    rows.append(["Router head", "two 1×1, then an MLP", "n/a", "n/a", "n/a",
                 "\\RouterCostPct\\%"])
    t_add = k.rows(rows,
        r"What the ladder adds, priced against one released decode. ``Blocks'' "
        r"is the module in units of a trunk block. Take from it that the FFN "
        r"adapter costs 5C² and not 2C²: the gated activation is free, so the "
        r"pair is 4C² + C² rather than 4C² + 4C², and the module is 0.71 of a "
        r"whole block. Since the routed allocation lives at exits 2 and 3, "
        r"which both wear the FFN adapter, that correction lands on precisely "
        r"the exits the reported saving leans on.")
    k.note(r"results/adapter\_cost.json for the two adapters, on the pinned "
           r"checkpoint. The seam-repair row is 9C + C² per pixel; Section " +
           f"{sec}.6 " + r"confirms that figure against a hook count rather "
           r"than taking it from the source. Router: "
           r"results/router\_latency.json.")

    k.par(r"The ladder assigns the FFN adapter to any exit skipping four or "
          r"more blocks and the 1×1 adapter otherwise, which at K = 6 and "
          r"b = 2 means the FFN at exits 0 to 3, the 1×1 at exit 4, and no "
          r"adapter at exit 5, whose feature is the trunk's own output with "
          r"nothing standing in for a skipped block. The arithmetic at exit 5 "
          r"is therefore the released decoder's, plus the repair pass, which "
          r"is why the deepest exit costs slightly more than the release "
          r"rather than the same.")

    # ------------------------------------------------------------------
    k.h2("From the block to the decoder")

    k.par(r"Write s<sub>up</sub>, s<sub>trunk</sub> and s<sub>head</sub> for "
          r"the shares of a released decode taken by the upsample, the trunk "
          r"and the head; a<sub>k</sub> for the adapter at exit k; and r for "
          r"the seam-repair pass. A tile assigned an exit shallower than j "
          r"still runs group j, because the decoder clamps the map, so the cost "
          r"of exit k in units of one released decode is")

    k.eq(r"c_k = s_{\mathrm{up}} + s_{\mathrm{head}} + r + "
         r"s_{\mathrm{trunk}}\frac{(\max(k,j)+1)b}{N} + a_{\max(k,j)}")

    k.par(r"and the frame-level cost of an exit map a over T tiles amortises "
          r"the shared part over the frame rather than over the tile,")

    k.eq(r"C(a) = s_{\mathrm{up}} + s_{\mathrm{trunk}}\frac{jb}{N} + "
         r"s_{\mathrm{head}} + r + \frac{1}{T}\sum_{i=1}^{T} u_{a(i)}")

    k.par(r"with u<sub>k</sub> the per-tile suffix, meaning the blocks after "
          r"the split plus the adapter. Only the last term depends on the map. "
          r"That is the structural reason a ladder over a decoder of this shape "
          r"has a ceiling well below 100\%: at the split depth the upsample, "
          r"the first jb = 4 blocks, the head and the repair pass are already " +
          f"{100 * (P['s_up'] + P['s_head'] + shr['rep'] + 4 * P['p']):.1f}" +
          r"\% of a released decode, and they run whatever the map says.")

    k.par(r"None of the four shares has to be taken on trust. "
          r"results/mac\_audit.json gives s<sub>up</sub> = " +
          f"{P['s_up']:.4f}" + r", s<sub>trunk</sub> = " +
          f"{P['s_trunk']:.4f}" + r" and s<sub>head</sub> = " +
          f"{P['s_head']:.4f}" + r" by hook count, so one trunk block is " +
          f"{P['p']:.6f}" + r" of a decode. That figure has an independent "
          r"check. results/static\_RECIPE512\_b01.json records the frame-level "
          r"saving of a map that sends every tile to exit e, for e = 2 to 5, on "
          r"the pinned checkpoint; differencing consecutive entries gives one "
          r"trunk block as " + f"{_block_from_static(k):.6f}" + r", which "
          r"agrees with the audit to " +
          _sci(abs(_block_from_static(k) - P["p"])) + r" of a decode. One is "
          r"a shape audit of the released decoder with no weights loaded and "
          r"the other is a set of measured savings on ours, which is the "
          r"reason to report both.")

    # ------------------------------------------------------------------
    k.h2("The per-exit cost vector and the ceiling")

    rows = [["Exit k", "Blocks", "Adapter", "c (2C²)", "c (5C²)",
             "c hook", "Saved \\%"]]
    for e in range(6):
        run = max(e, 2)
        kind = "none" if run == 5 else ("FFN" if (5 - run) * 2 >= 4 else "1×1")
        nm = f"{e}" if e >= 2 else f"{e} (clamped)"
        rows.append([nm, f"{(run + 1) * 2}", kind, f"{A2[e]:.4f}",
                     f"{B[e]:.4f}", f"{D[e]:.4f}",
                     f"{100 * (1 - D[e]):.2f}"])
    t_exit = k.rows(rows,
        r"The per-exit cost vector, in units of one released decode, under "
        r"three prices for the same architecture: the first shipped table, "
        r"which put the FFN adapter at 2C²; the table the paper reports, which "
        r"corrected that to 5C² but kept the closed-form block; and the hook "
        r"count assembled in Section " + f"{sec}.4" + r". Take from it that "
        r"the two corrections move the shallowest reachable exit by 2.79 and "
        r"then 0.80 points and leave exits 4 and 5 nearly untouched, because "
        r"those wear the cheap adapter or none, and that the deepest exit "
        r"costs " + f"{100 * (D[5] - 1):.2f}" + r"\% <i>more</i> than the "
        r"release, which is the seam-repair pass.")
    k.note(r"Exits 0 and 1 are unreachable: the decoder clamps the map at "
           r"j = 2, so a tile nominally assigned them leaves through exit 2 and "
           r"wears exit 2's cost. The 2C² and 5C² vectors are recorded in "
           r"results/why\_qp.json and results/why\_qp\_PAPER.json; a cost "
           r"vector is a property of the model rather than of the weights, so "
           r"the first being measured on runs/BEST/ckpt\_eval.pth.tar does not "
           r"affect it.")

    k.par(r"The ceiling is 100(1 − c<sub>j</sub>), the saving when every tile "
          r"takes the shallowest reachable exit. It is a property of the "
          r"architecture and no allocation can pass it. Three values are in "
          r"circulation and they are the three columns of Table " +
          f"{t_exit}" + r": " + f"{100 * (1 - A2[2]):.2f}" + r"\% at 2C², "
          r"\Ceiling\% at 5C², which is the value the paper's tables use, and " +
          f"{100 * (1 - D[2]):.2f}" + r"\% by hook count. The last of the "
          r"three is the one a meter agrees with, and the next subsection is "
          r"the demonstration.")

    # ------------------------------------------------------------------
    k.h2("Predicted against hook-measured")

    k.par(r"A cost model and the decoder it models drift apart. The check that "
          r"catches it is a hook count: flexuf/measure.py attaches a forward "
          r"hook to every convolution and linear layer and accumulates MACs "
          r"from the output shape of each call. A group skipped by an early "
          r"exit never fires its hook and so costs nothing; a group run on 28 "
          r"of 40 tiles costs 28 tiles' worth, because tiles are the batch "
          r"dimension and the output shape carries them. There is no "
          r"bookkeeping to get wrong, and the count is the ground truth the "
          r"model is judged against.")

    rows = [["Condition", "Map 2/3/4/5", "Measured", "5C²", "diff",
             "Hook", "diff"]]
    lab = {"1280x768": "720p", "2048x1280": "1080p"}
    for r, m in _maps(k):
        n = float(r["n_tiles"])
        cB = sum(c * B[e] for c, e in zip(m, (2, 3, 4, 5))) / n
        cD = sum(c * D[e] for c, e in zip(m, (2, 3, 4, 5))) / n
        sB, sD, sm_ = 100 * (1 - cB), 100 * (1 - cD), r["mac_saving_pct"]
        rows.append([f"{lab[r['size']]} q{r['qp']}",
                     "/".join(str(x) for x in m),
                     f"{sm_:.4f}", f"{sB:.4f}", f"{sB - sm_:+.3f}",
                     f"{sD:.4f}", f"{sD - sm_:+.4f}"])
    t_pm = k.rows(rows,
        r"Predicted against measured, on six decodes with six different exit "
        r"maps at two resolutions. ``Measured'' is the hook count off the "
        r"decode that was executed and timed. Take from it that the assembly "
        r"of Section " + f"{sec}.4" + r" reproduces the count exactly in every "
        r"condition, to four decimal places and in fact to floating point, "
        r"while the model the paper reports is optimistic by 0.47 to 0.80 "
        r"points and always in the same direction.")
    k.note(r"results/supp\_power.json, pinned checkpoint, 0.1 dB budget, "
           r"NVIDIA RTX A6000. The map is the file's pooled exit histogram "
           r"rounded to the frame's tile count, which is what "
           r"scripts/power\_profile.py decodes; its four columns are the tiles "
           r"leaving at exits 2, 3, 4 and 5.")

    rec, resid, gap = _recover(k, D)
    k.par(r"Because a hook count is exactly linear in the exit map, six "
          r"mixtures spanning four reachable exits over-determine the per-exit "
          r"vector and it can simply be solved for. The six equations are "
          r"consistent to " + _sci(resid) + r", and the solution differs from "
          r"the assembled vector by at most " + _sci(gap) + r" at any exit. "
          r"The identity is worth stating plainly: the per-module counts of "
          r"results/mac\_audit.json and results/adapter\_cost.json are not a "
          r"model of what the meter reports, they are what the meter reports, "
          r"rearranged. Where the paper's tables differ from the meter, the "
          r"difference has a closed form and Table " + str(k.peek_tbl()) +
          r" gives it term by term.")

    rows = [["Term", "Model price", "Hook price", "Points of saving"]]
    rows.append(["Seam repair", f"{P['rep_macpx'] / P['blk_closed'] * P['p']:.6f}",
                 f"{shr['rep']:.6f}",
                 f"{100 * (shr['rep'] - P['rep_macpx'] / P['blk_closed'] * P['p']):.4f}"])
    rows.append(["1×1 adapter", f"{C * C / P['blk_closed'] * P['p']:.6f}",
                 f"{shr['a1']:.6f}",
                 f"{100 * (shr['a1'] - C * C / P['blk_closed'] * P['p']):.4f}"])
    rows.append(["FFN adapter", f"{5 * C * C / P['blk_closed'] * P['p']:.6f}",
                 f"{shr['aff']:.6f}",
                 f"{100 * (shr['aff'] - 5 * C * C / P['blk_closed'] * P['p']):.4f}"])
    rows.append(["<b>Exit 5 residual</b>", "", "",
                 f"<b>{100 * (D[5] - B[5]):.4f}</b>"])
    rows.append(["<b>Exit 4 residual</b>", "", "",
                 f"<b>{100 * (D[4] - B[4]):.4f}</b>"])
    rows.append(["<b>Exits 2, 3 residual</b>", "", "",
                 f"<b>{100 * (D[2] - B[2]):.4f}</b>"])
    t_dec = k.rows(rows,
        r"The residual of Table " + f"{t_pm}" + r", term by term. Three "
        r"modules are priced as a fraction of a block, and the shipped model's "
        r"block is 8C² + 9C where the meter's is 7C² + 9C. Take from it that "
        r"the residual is not noise and not a tolerance: exit 5 wears the "
        r"repair mispricing alone, exit 4 wears the repair plus the 1×1 "
        r"adapter, and exits 2 and 3 wear the repair plus the FFN adapter, "
        r"which is why the error is largest exactly where the saving is "
        r"largest.")
    k.note(r"Model prices: the module's MAC/px divided by 8C² + 9C, times one "
           r"trunk block. Hook prices: the module's MAC/px times the feature "
           r"grid, divided by the decode total, both from "
           r"results/mac\_audit.json and results/adapter\_cost.json, and they "
           r"are the last column of Table " + f"{t_add}" + r". The three "
           r"residual rows are the corresponding columns of Table " +
           f"{t_exit}" + r".")

    grid = k.J("signalled_RECIPE512_grid.json")
    gv = [r["model_minus_measured"] for r in grid["rows"]
          if r.get("model_minus_measured") is not None]
    ctc = {r["qp"]: r for r in k.J("signalled_RECIPE512_ctc53.json")["rows"]
           if abs(r["budget_db"] - 0.1) < 1e-9}
    k.fig("supp_b_check.png",
          "<b>The model against the meter.</b> <b>a</b>, six decodes at two "
          "resolutions: the hook-count assembly lies on the identity line and "
          "the model the paper reports lies above it. <b>b</b>, the same "
          "residual over the whole budget grid, " +
          f"{len(sorted(set(r['budget_db'] for r in grid['rows'])))}" +
          " budgets by " +
          f"{len(sorted(set(r['qp'] for r in grid['rows'])))}" +
          " quality indices on \\NumSeq sequences. It is positive everywhere, "
          "grows with the budget, and saturates at the value an all-exit-2 map "
          "implies.",
          maxh=1.45 * 72)

    k.par(r"Figure " + f"{k.peek_fig() - 1}" + r"b runs the same comparison "
          r"over the whole operating range rather than six points of it. The "
          r"residual is positive in all " + f"{len(gv)}" + r" cells, from " +
          f"{min(gv):.3f}" + r" to " + f"{max(gv):.3f}" + r" points, and it "
          r"saturates at exactly the exits-2 value of Table " + f"{t_dec}" +
          r". Reading the paper's headline figures as hook counts therefore "
          r"means subtracting between " +
          f"{min(ctc[q]['model_minus_measured'] for q in ctc):.2f}" +
          r" and " +
          f"{max(ctc[q]['model_minus_measured'] for q in ctc):.2f}" +
          r" points: at the 0.1 dB budget the model reports \MainLowRate\% at "
          r"the lowest rate against a measured " +
          f"{ctc[0]['saving_pct_measured']:.2f}" + r"\%, and \MainHighRate\% "
          r"at the highest against a measured " +
          f"{ctc[63]['saving_pct_measured']:.2f}" + r"\%.")
    k.note(r"results/signalled\_RECIPE512\_grid.json and "
           r"results/signalled\_RECIPE512\_ctc53.json, both on the pinned "
           r"checkpoint. Both columns are in the files: "
           r"saving\_pct\_vs\_release is the model, saving\_pct\_measured is "
           r"the hook count, and model\_minus\_measured is their difference.")

    # ------------------------------------------------------------------
    k.h2("How the timings were taken")

    lb = k.J("supp_latency_batch_1920x1080.json")
    lc = k.J("supp_latency_cpu_1920x1080.json")
    rl = k.J("router_latency.json")

    k.par(r"The rest of this section leaves arithmetic for the clock, so the "
          r"protocol comes first. Everything below follows the same rules and "
          r"they are stated rather than implied.")

    k.bullets([
        r"<b>Batch 1, and why.</b> Every wall clock here is one frame at a "
        r"time, which is what a decoder does. A batch is genuinely different "
        r"work for this decoder and not a repetition, because exit\_map is "
        r"indexed over the whole batch's tiles, so B frames of 40 tiles form "
        r"one shrinking active set of 40B and the groups run wider. Section " +
        f"{sec}.8 " + r"measures what the choice costs.",

        r"<b>Warm-up, then interleaving, then the median.</b> " +
        f"{lb['warmup']}" + r" untimed iterations, then " + f"{lb['iters']}" +
        r" rounds of one call of each variant in turn. This is a shared card, "
        r"and timing all of the released decoder and then all of ours puts any "
        r"drift in a neighbouring job's load straight into the ratio, which is "
        r"the only quantity these runs exist to produce.",

        r"<b>What the timer wraps.</b> time.perf\_counter around a "
        r"synchronised call, not CUDA events, because events do not exist on "
        r"the CPU and the two device classes have to be timed alike. The timer "
        r"starts at the decoded latent and stops at the reconstruction, so it "
        r"includes patchify, every group, unpatchify, the seam-repair pass and "
        r"the head.",

        r"<b>What is deliberately outside it.</b> Entropy decoding, because "
        r"the bitstream and therefore the rate do not depend on the exit map: "
        r"including it would divide both sides of every ratio by the same "
        r"constant and flatter the method. The map itself is charged, in bits "
        r"rather than in time, at \MapBits bits per frame.",

        r"<b>The router is charged in MACs and not in seconds.</b> Its "
        r"arithmetic is \RouterCostPct\% of a decode but "
        r"\RouterTimePct\% of decode time, \RouterTimeFactor× its arithmetic "
        r"share, an extra \RouterTimeExtra points bought by launching a small "
        r"kernel over a large map. That is the general reason a MAC count "
        r"cannot see a kernel launch, in miniature (results/router\_latency.json, " +
        f"{rl['iters']}" + r" iterations at " +
        f"{rl['resolution'][1]}×{rl['resolution'][0]}" + r" on an NVIDIA RTX "
        r"A6000, measured on runs/RECIPE512/ckpt\_eval.pth.tar). The router "
        r"does not run at all in the signalled configuration, where the encoder "
        r"chooses the map.",

        r"<b>The masked path, not the sorted one.</b> The shipped "
        r"configuration leaves sorted\_tiles off, so every group is a masked "
        r"gather over the full tile batch. Sorting the tiles by depth turns "
        r"each group into a contiguous slice and recovers \WallSortedGain "
        r"points at 1080p for bit-identical output "
        r"(results/latency\_RECIPE512\_sorted.json, on "
        r"runs/RECIPE512/ckpt\_eval.pth.tar). None of the timings below use "
        r"it, so they are the slower of the two implementations.",
    ])

    # ------------------------------------------------------------------
    k.h2("Where the seconds go, and on what device")

    L = k.J("supp_latency_1920x1080.json")
    S = k.J("supp_latency_1280x720.json")
    lm = {s["stage"]: s["ms"] for s in L["stages"]}
    sm = {s["stage"]: s["ms"] for s in S["stages"]}
    grp = [("group 2  (40 tiles)", "group 2  (15 tiles)", 40, 15),
           ("group 3  (28 tiles)", "group 3  (10 tiles)", 28, 10),
           ("group 4  (14 tiles)", "group 4  (5 tiles)", 14, 5),
           ("group 5  (9 tiles)", "group 5  (3 tiles)", 9, 3)]
    tot_ms = sum(lm.values())
    rows = [["Stage", "MAC \\%", "1080p tiles", "1080p ms", "720p tiles",
             "720p ms"]]
    rows.append(["Stem: upsample + groups 0, 1",
                 f"{100 * (P['s_up'] + 4 * P['p']):.2f}", "frame",
                 f"{lm['stem (upsample + groups 0..j-1)']:.2f}", "frame",
                 f"{sm['stem (upsample + groups 0..j-1)']:.2f}"])
    rows.append(["Patchify", "0.00", "", f"{lm['patchify']:.2f}", "",
                 f"{sm['patchify']:.2f}"])
    for i, (gl, gs, nl, ns_) in enumerate(grp):
        rows.append([f"Group {i + 2} (2 blocks, per tile)",
                     f"{100 * 2 * P['p']:.2f}", f"{nl}", f"{lm[gl]:.2f}",
                     f"{ns_}", f"{sm[gs]:.2f}"])
    rows.append(["Unpatchify", "0.00", "", f"{lm['unpatchify']:.2f}", "",
                 f"{sm['unpatchify']:.2f}"])
    rows.append(["Seam repair (full-frame)", f"{100 * shr['rep']:.2f}", "frame",
                 f"{lm['seam repair (384ch, full-frame)']:.2f}", "frame",
                 f"{sm['seam repair (384ch, full-frame)']:.2f}"])
    rows.append(["Head", f"{100 * P['s_head']:.2f}", "frame",
                 f"{lm['head']:.2f}", "frame", f"{sm['head']:.2f}"])
    t_stage = k.rows(rows,
        r"Where a routed decode goes, in arithmetic and in milliseconds. The "
        r"MAC column is each stage's share of a released decode at full "
        r"occupancy and sums to " + f"{100 * D[5]:.2f}" + r"\%, which is the "
        r"deepest exit's cost. Take from the table that patchify and "
        r"unpatchify are free arithmetically and nearly free on the clock, "
        r"that the repair pass is " + f"{100 * shr['rep']:.2f}" + r"\% of the "
        r"arithmetic and " +
        f"{100 * lm['seam repair (384ch, full-frame)'] / tot_ms:.1f}" +
        r"\% of this decode's clock, and that the head takes " +
        f"{100 * lm['head'] / tot_ms:.1f}" + r"\% of the clock for " +
        f"{100 * P['s_head']:.2f}" + r"\% of the arithmetic, being "
        r"bandwidth-bound rather than arithmetic-bound.")
    k.note(r"results/supp\_latency\_1920x1080.json and "
           r"results/supp\_latency\_1280x720.json, pinned checkpoint, q32, "
           r"0.1 dB budget, NVIDIA RTX A6000 taken under the evaluation lock so "
           r"the card was not shared. Tile counts are that decode's own exit "
           r"map, which leaves 28, 14 and 9 of 40 tiles alive in groups 3, 4 "
           r"and 5.")

    k.par(r"Patchify and unpatchify are reshapes and cost no arithmetic at "
          r"all. They require the feature map to divide by the " +
          f"{td['feature_patch']}" + r" px feature tile, which the "
          r"encoder-side replicate padding guarantees: 1920×1080 becomes "
          r"2048×1280 and 40 tiles, 1280×720 becomes 1280×768 and 15 "
          r"(results/tile\_definition.json).")

    k.fig("supp_b_units.png",
          "<b>What the arithmetic is worth.</b> <b>a</b>, one exit map "
          "measured three ways at 1080p. <b>b</b>, the same map on two device "
          "classes against what the MAC model predicts: the A6000 falls short "
          "of the prediction at every rate, the CPU does not. <b>c</b>, why a "
          "GPU falls short: a tile costs more to decode in a group that has "
          "fewer tiles left in it.",
          maxh=1.45 * 72)

    k.par(r"Two effects separate seconds from multiply-accumulates and the "
          r"figure isolates both. The first is the tiled path itself, which the "
          r"released decoder does not pay: patchify, unpatchify and the repair "
          r"pass are " +
          f"{100 * (lm['patchify'] + lm['unpatchify'] + lm['seam repair (384ch, full-frame)']) / tot_ms:.1f}" +
          r"\% of the routed clock at 1080p. Timing our own decoder with every "
          r"tile at the deepest exit isolates it: that variant does the same "
          r"arithmetic as the release plus the tiling machinery, and it is " +
          f"{min(r['overhead_pct'] for r in lb['rows']):.1f}" + r" to " +
          f"{max(r['overhead_pct'] for r in lb['rows']):.1f}" + r"\% slower "
          r"across nine measurements on the pinned checkpoint, against "
          r"\TilingOverhead\% for the figure the main paper quotes. The second "
          r"is occupancy. A group late in the ladder launches kernels over a "
          r"handful of 32×32 feature maps, and a GPU sized for the first group "
          r"is partly idle inside them. Dividing each group of Table " +
          f"{t_stage}" + r" by the tiles alive in it, at 1080p a tile costs " +
          f"{100 * ((lm['group 5  (9 tiles)'] / 9) / (lm['group 2  (40 tiles)'] / 40) - 1):.1f}" +
          r"\% more in group 5, on 9 tiles, than in group 2 on 40. At 720p the "
          r"same curve rises by " +
          f"{100 * ((sm['group 4  (5 tiles)'] / 5) / (sm['group 2  (15 tiles)'] / 15) - 1):.1f}" +
          r"\% from 15 tiles to 5 and then falls back at 3, so the trend is "
          r"clear at 1080p and not monotone at 720p.")

    k.par(r"Device class and batch size are the two knobs a reader would ask "
          r"about next, and Table " + str(k.peek_tbl()) + r" sweeps both. All "
          r"eight cards in this machine are RTX A6000s, so the only second "
          r"device class reachable here is the CPU. It is also the "
          r"unflattering comparison to make, in the sense that it is the one "
          r"our own explanation predicts we will look better on: a CPU decode "
          r"is closer to arithmetic-bound, so if the shortfall on the GPU is "
          r"scheduling rather than arithmetic it should shrink.")

    rows = [["Device", "Rate", "Batch", "ms/frame rel.", "ms/frame ours",
             "fps ours", "Saved \\%", "MAC model \\%"]]
    for r in lb["rows"]:
        rows.append(["A6000", f"q{r['qp']}", f"{r['batch']}",
                     f"{r['ms_stock_per_frame']:.1f}",
                     f"{r['ms_routed_per_frame']:.1f}",
                     f"{1000 / r['ms_routed_per_frame']:.1f}",
                     f"{r['realised_saving_pct']:.2f}",
                     f"{r['predicted_saving_pct']:.2f}"])
    for r in lc["rows"]:
        rows.append(["CPU, 8 thr.", f"q{r['qp']}", f"{r['batch']}",
                     f"{r['ms_stock_per_frame']:.0f}",
                     f"{r['ms_routed_per_frame']:.0f}",
                     f"{1000 / r['ms_routed_per_frame']:.3f}",
                     f"{r['realised_saving_pct']:.2f}",
                     f"{r['predicted_saving_pct']:.2f}"])
    t_dev = k.rows(rows,
        r"Wall clock at 1080p by device class and batch size, on one exit map "
        r"per rate. ``rel.'' is the released decoder timed in the same loop; "
        r"fps is one over our per-frame time. Take from it that batch 1 costs "
        r"about a point of saving against batch 4 and nothing beyond it, so "
        r"the batch-1 convention this work uses is close to free; and that the "
        r"A6000 falls short of the MAC model by 3.2 to 4.7 points at every "
        r"rate and every batch size, while the CPU meets or beats it at the "
        r"two lower rates.")
    k.note(r"results/supp\_latency\_batch\_1920x1080.json and "
           r"results/supp\_latency\_cpu\_1920x1080.json, pinned checkpoint, "
           r"0.1 dB budget, exit maps read from "
           r"results/supp\_paper\_curve\_PAPER.json. GPU rows: " +
           f"{lb['warmup']}" + r" warm-up and " + f"{lb['iters']}" +
           r" interleaved iterations. CPU rows: " + f"{lc['torch_threads']}" +
           r" threads, " + f"{lc['warmup']}" + r" warm-up and " +
           f"{lc['iters']}" + r" iterations only, on a shared machine.")

    st = [r["ms_stock_per_frame"] for r in lc["rows"]]
    k.par(r"The CPU rows carry a noise floor and it should be read before the "
          r"conclusion. The released decode does not depend on the rate, so its "
          r"three timings should be equal; they range from " +
          f"{min(st) / 1000:.2f}" + r" to " + f"{max(st) / 1000:.2f}" +
          r" s, a spread of " + f"{100 * (max(st) / min(st) - 1):.1f}" +
          r"\%, which is what " + f"{lc['iters']}" + r" iterations on a shared "
          r"machine buys. That is enough to support the gap at q0 and q32, "
          r"where the CPU is 6.5 and 4.5 points above the A6000 on the same "
          r"map, and not enough to support anything at q63, where the two "
          r"agree to a tenth of a point. The direction is the one the "
          r"occupancy explanation predicts, and a firmer statement needs a "
          r"machine that is not shared.")

    # ------------------------------------------------------------------
    k.h2("What the saving is worth in joules")

    pw = k.J("supp_power.json")
    k.par(r"Board power was sampled from the driver at 100 ms, which is about "
          r"the rate the sensor updates. Each condition therefore runs for a "
          r"fixed wall time of 20 s rather than a fixed iteration count, so "
          r"that a 30 ms decode yields two hundred power samples rather than "
          r"three. Idle is measured for 4 s immediately before and after every "
          r"timed run on the same card, so a neighbouring process waking up "
          r"mid-measurement appears as a drift between the two rather than as a "
          r"saving. Energy per frame is the mean board power over the run times "
          r"the median decode time.")

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
           r"NVIDIA RTX A6000 with a " + f"{pw['power_limit_w']:.0f}" +
           r" W limit, taken under the evaluation lock. 720p is padded to "
           r"1280×768 and 15 tiles, 1080p to 2048×1280 and 40 tiles.")

    rows = [["Condition", "ms rel.", "ms ours", "W rel.", "W ours", "J rel.",
             "J ours"]]
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

    hi = pw["rows"][3]
    lo = pw["rows"][5]
    k.par(r"The engineering result should be stated plainly: the energy saving "
          r"of this method tracks its time saving, not its arithmetic saving. "
          r"A partly idle GPU still draws most of its static power, so the "
          r"joules follow the seconds almost exactly, and the seconds fall "
          r"short of the multiply-accumulates for the occupancy reason of "
          r"Figure " + f"{k.peek_fig() - 1}" + r"c, at the board powers of "
          r"Table " + f"{t_abs}" + r". At 1080p and the lowest "
          r"rate the three read " + f"{hi['mac_saving_pct']:.2f}" + r"\%, " +
          f"{hi['time_saving_pct']:.2f}" + r"\% and " +
          f"{hi['energy_saving_pct']:.2f}" + r"\%; at the highest rate they "
          r"read " + f"{lo['mac_saving_pct']:.2f}" + r"\%, " +
          f"{lo['time_saving_pct']:.2f}" + r"\% and " +
          f"{lo['energy_saving_pct']:.2f}" + r"\%. Quoting the arithmetic "
          r"figure as an energy figure overstates it by three and a half to "
          r"five points.")

    k.par(r"Two limits. The idle baseline drifts upward as the card warms, "
          r"from 74.6 W before the first condition to 128.4 W after the fourth, "
          r"so the above-idle column of Table " + f"{t_three}" + r" is the "
          r"weaker of the two energy figures, and it is only because it agrees "
          r"with the unadjusted column to within 0.4 points that the conclusion "
          r"survives. Power also comes from the driver counter rather than an "
          r"external meter, so the absolute joules inherit whatever bias that "
          r"sensor has; the savings are ratios of two measurements taken the "
          r"same way on the same card minutes apart, which is what the bias "
          r"mostly cancels in.")

    # ------------------------------------------------------------------
    k.h2("Memory, frame rate, and what the encoder pays")

    fp = k.J("supp_footprint.json")
    rows = [["Frame (padded)", "Tiles", "MB rel.", "MB ours", "Change \\%",
             "fps rel.", "fps ours"]]
    for r in fp["rows"]:
        if r.get("oom"):
            rows.append([r["size"], f"{r['n_tiles']}", "out of memory", "", "",
                         "", ""])
        else:
            rows.append([r["size"], f"{r['n_tiles']}",
                         f"{r['peak_mb_full']:.1f}",
                         f"{r['peak_mb_routed']:.1f}",
                         f"+{r['peak_delta_pct']:.2f}",
                         f"{r['fps_full']:.2f}", f"{r['fps_routed']:.2f}"])
    t_mem = k.rows(rows,
        r"Peak decoder memory and frame rate, released against ours, at every "
        r"resolution that fitted. Take from it that routing costs memory "
        r"rather than saving it, by a constant 13.33\% at every resolution, "
        r"because the tiled path holds the whole tile batch and the stitched "
        r"canvas alive together; and that the frame-rate gain grows with the "
        r"frame, from 43.9 to 49.9 fps at 832×480 up to 9.0 to 11.2 fps at "
        r"1080p, because a larger frame has more tiles to allocate across. The "
        r"4K row is a genuine failure and is reported as one: the job ran out "
        r"of memory on a 48 GB card that four other processes were sharing, so "
        r"no measurement above 1080p exists anywhere in this work.")
    k.note(r"results/supp\_footprint.json, pinned checkpoint, q32, 0.1 dB "
           r"budget, NVIDIA RTX A6000. Frame rates here are lower than in "
           r"Table " + f"{t_dev}" + r" because that table's maps are at "
           r"different rates and this one is q32 throughout. The 4K stage "
           r"profile queued alongside it failed the same way "
           r"(results/supp\_queue.log).")

    ec = k.J("supp_encoder_cost_PAPER.json")
    k.par(r"The signalled configuration moves the choice of exit map to the "
          r"encoder, which has to price the tiles before it can allocate them, "
          r"and that cost belongs in a complexity section even though it falls "
          r"on the other side. On one 1080p sequence at the highest rate, one "
          r"tiled decode takes " + f"{ec['ms_one_decode']:.1f}" + r" ms; "
          r"building the per-tile cost table full-frame takes " +
          f"{ec['ms_full_frame_table']:.1f}" + r" ms, or " +
          f"{ec['x_full_frame']:.3f}" + r"× one decode; and building it the "
          r"way the deployed encoder does, per tile, takes " +
          f"{ec['ms_deployed_table']:.1f}" + r" ms, or " +
          f"{ec['x_deployed']:.3f}" + r"×. The two searches agree on " +
          f"{100 * ec['approx']['agreement']:.1f}" + r"\% of tiles and reach " +
          f"{ec['exact']['saving']:.2f}" + r"\% and " +
          f"{ec['approx']['saving']:.2f}" + r"\% saving at the same budget. "
          r"The asymmetry is the design rather than an accident: a decoder-side "
          r"saving is bought with encoder-side work, which suits a "
          r"compress-once decode-many deployment and does not suit live "
          r"encoding.")
    k.note(r"results/supp\_encoder\_cost\_PAPER.json: pinned checkpoint, one "
           r"sequence (Bosphorus), q63, 40 tiles, " + f"{ec['iters']}" +
           r" iterations on an NVIDIA RTX A6000.")

    # ------------------------------------------------------------------
    k.h2("What no file in results/ pins")

    k.par(r"Four gaps, listed so that they are visible rather than quietly "
          r"absent.")

    k.bullets([
        r"<b>The paper's tables are on the 5C² column.</b> Tables " +
        f"{t_exit}" + r" and " + f"{t_dec}" + r" together say by how much: "
        r"between 0.14 and 0.80 points of saving, in the optimistic direction "
        r"at every exit. The hook-count column is the one a meter agrees "
        r"with, and the honest ceiling is " + f"{100 * (1 - D[2]):.2f}" +
        r"\% rather than \Ceiling\%. The BD-Rate figures are unaffected, "
        r"being integrals of rate against quality with no compute axis in "
        r"them; the BD-saving figures would shift by the same fraction of a "
        r"point as the savings they integrate.",

        r"<b>Nothing above 1080p.</b> Both 4K jobs ran out of memory (Table " +
        f"{t_mem}" + r"). GPU 2 has about 5 GB free on a card four other "
        r"processes are sharing, and a 3840×2304 decode does not fit, so the "
        r"resolution trend in this section rests on three points of which two "
        r"carry power measurements. This is a limit of the machine and not of "
        r"the method.",

        r"<b>One GPU model, one power sensor, and no second card class.</b> "
        r"Every millisecond and joule on a GPU here is an NVIDIA RTX A6000, "
        r"because all eight cards in this machine are; the CPU rows of Table " +
        f"{t_dev}" + r" are the only second device class, and they carry the "
        r"noise floor stated beside them. Nothing here separates the decoder's "
        r"energy from the board's, and no external meter was available to "
        r"calibrate the driver counter against.",

        r"<b>One budget for the clock.</b> Every timing, power and memory "
        r"measurement in this section is at the 0.1 dB budget, and the stage "
        r"profiles are at q32 only. The arithmetic is swept across the whole "
        r"grid in Figure " + f"{k.peek_fig() - 1}" + r"b; the seconds are not, "
        r"and a deeper budget moves more tiles to exit 2, where the occupancy "
        r"penalty of Figure " + f"{k.peek_fig() - 1}" + r"c is smallest.",
    ])


def _block_from_static(k):
    """One trunk block as a fraction of a released decode, from measured savings.

    results/static_RECIPE512_b01.json records the frame-level saving of a map
    that sends every tile to exit e. Consecutive entries differ by exactly two
    trunk blocks wherever the adapter does not change, which is between exits 2
    and 3, so the difference halved is one block. Independent of the hook audit
    in every respect except the definition of a MAC.
    """
    st = k.J("static_RECIPE512_b01.json")
    u = {r["exit"]: 1.0 - r["saving"] / 100.0 for r in st["rows"][0]["uniform"]}
    return (u[3] - u[2]) / 2.0
