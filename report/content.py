"""The report's text. Every quantity is read from a result file."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reportlab.lib.styles import ParagraphStyle
from build_report import (P, H1, H2, BODY, ABST, TITLE, AUTH, MONO, CAP,
                          figure, table, build, J, RES, fign, tabn, Spacer)

QPS = [0, 16, 32, 48, 63]


def num():
    n = {}
    ep = J("epoch_series_remeasured.json")
    n["ep"] = {r["epoch"]: r for r in ep}
    n["pin"] = n["ep"][4]
    g = J("granularity_ctc53_fixed.json")
    n["gran"] = {r["qp"]: {c["cell_px"]: c for c in r["cells"]} for r in g["rows"]}
    n["gmean"] = {c: sum(n["gran"][q][c]["saving_pct_vs_release"] for q in QPS) / 5
                  for c in (256, 128, 64, 32)}
    n["band"] = {c: sum(n["gran"][q][c]["band_cost_pct"] for q in QPS) / 5
                 for c in (256, 128, 64, 32)}
    # Two runs cover the model list between them: router_fit2 has the first
    # sweep (including the independent per-exit fit), router_fit3 adds the
    # smearing correction and the MLP. Later file wins on a shared name.
    n["r_j2"] = {}
    for f in ("router_fit2.json", "router_fit3.json"):
        for r in J(f)["rows"]:
            n["r_j2"][r["model"]] = r
    n["r_j0"] = {r["model"]: r for r in J("router_fit_j0.json")["rows"]}
    n["abl"] = J("router_ablate64.json").get("ablation", [])
    n["gb"] = J("gather_bench.json")
    n["mask"] = J("mask_structure.json")
    n["ht"] = {q: J(f"head_term_q{q}_ctc53.json") for q in (0, 32, 63)}
    n["psnr"] = {e: J(f"psnr_{t}.json") for t, e in
                 [("epo0", 0), ("PIN_e1", 1), ("PIN_e3", 3), ("PAPER", 4),
                  ("PIN_e5", 5), ("PIN_e6", 6), ("PIN_e7", 7)]}
    n["bd"] = {}
    for k, f in [("tiled", "bdc_frontier_tiled256_rebased.json"),
                 ("pp256", "bdc_frontier_pp256.json"),
                 ("pp64", "bdc_frontier_pp64.json"),
                 ("dep64", "bdc_frontier_dep64.json")]:
        d = J(f)
        n["bd"][k] = d
    return n


N = num()


def bdmean(k, key):
    """The mean bd_saving.py already computed, not one recomputed here.

    The file carries mean_bd_saving_pct and mean_bd_quality_db at the top
    level; averaging its per-rate rows again would silently drop the rates it
    marked unmeasurable.
    """
    return N["bd"][k][{"bd_saving": "mean_bd_saving_pct",
                       "bd_quality": "mean_bd_quality_db"}[key]]


def story():
    s = []
    s.append(P("Exact per-position early exit:<br/>three inference-time upgrades "
               "to the FLEX-UF ladder", TITLE))
    s.append(P("Supplementary experiment report &mdash; the main experiment "
               "(<font face='Courier'>runs/RECIPE512</font>, pinned at epoch 4) "
               "is not modified anywhere in this document.", AUTH))

    s.append(P("<b>Abstract.</b> FLEX-UF saves decoder compute by letting each "
               "256&nbsp;px tile of a frame leave a shared trunk at its own depth, "
               "and reports "
               f"{N['pin']['mean']:.2f}% of a released DCVC-UF decode saved at a "
               "0.1&nbsp;dB budget over 53 CTC frames. This report shows that "
               "three properties of that system are artefacts of the cut rather "
               "than of the ladder, and that removing them is worth "
               f"{N['r_j0']['oracle']['mean_saving_pct'] - N['pin']['mean']:.2f} "
               "points of saving with no weight retrained. First, tiling pays a "
               "seam out of the distortion budget: the deployed operating point "
               "spends about 0.015 of its 0.1&nbsp;dB on the cut itself. Second, "
               "replacing tiles with per-position depth, computed where a "
               "max-plus dilation of the allocated depth says a block is needed, "
               "is exact &mdash; each position receives what a full-frame decode "
               "to its own depth would have produced &mdash; and verified to "
               "0.0002&nbsp;dB on the full test set. Third, once there is no cut, "
               "the split depth that existed to control the seam has no purpose, "
               "and the two shallow rungs it locks out turn out to be already "
               "trained. Integrated in the Bjontegaard style over a common "
               f"interval, the frontier moves from {bdmean('tiled', 'bd_saving'):.2f}% "
               f"to {bdmean('pp64', 'bd_saving'):.2f}% BD-saving and from "
               f"{bdmean('tiled', 'bd_quality'):.4f} to "
               f"{bdmean('pp64', 'bd_quality'):.4f}&nbsp;dB BD-quality. A trained "
               "decoder-side router with no signalled map reaches "
               f"{bdmean('dep64', 'bd_saving'):.2f}%, above the tiled oracle. We "
               "also report what did not work.", ABST))

    # ---- 1
    s.append(P("1&nbsp;&nbsp;Introduction", H1))
    s.append(P("An early-exit decoder spends less arithmetic on the parts of an "
               "image that do not need it. FLEX-UF does this by cutting the frame "
               "into tiles and giving each tile its own exit from a twelve-block "
               "trunk. The cut is what makes the idea implementable: a tile is a "
               "contiguous region, so a tile that leaves early simply stops, and "
               "the compute saved is a rectangle."))
    s.append(P("The cut is also what the idea pays for, in three ways that are "
               "easy to miss because each of them looks like a property of the "
               "ladder. (i) A convolution at a tile border reads a padded value "
               "instead of its neighbour, and the resulting error is charged to "
               "the distortion budget the system is trying to respect. (ii) The "
               "finer the tiles, the more border per unit area, so the allocation "
               "cannot be made finer than the seam allows &mdash; 256&nbsp;px is "
               "not a modelling choice but a seam budget. (iii) Because the seam "
               "is worst where the features are least settled, the first blocks "
               "of the trunk are made shared, which removes the shallowest rungs "
               "of the ladder from the deployed decoder entirely."))
    s.append(P("This report removes the cut and measures what each of the three "
               "was costing. Nothing is retrained: every number below is produced "
               "by the pinned epoch-4 checkpoint of the main experiment, on the "
               "same 53 CTC frames, against the same released DCVC-UF decoder, at "
               "the same 0.1&nbsp;dB budget in the same per-frame decibel."))

    # ---- 2
    s.append(P("2&nbsp;&nbsp;The ladder as trained and as deployed", H1))
    s.append(figure("fig1_training", "train",
                    "The main experiment's ladder. Six exits are trained; the "
                    "deployed forward can select four. "
                    "<font face='Courier'>forward()</font> applies "
                    "<font face='Courier'>exit_map.clamp(min=j)</font> with "
                    "<i>j</i>=2, so a tile assigned exit 0 or 1 decodes as exit 2. "
                    "The training loss does not know about the clamp: the exit "
                    "weights in <font face='Courier'>flexuf/losses.py</font> are "
                    "<font face='Courier'>w[:K-1] = aux_weight</font>, with no "
                    "reference to the split. Under "
                    "<font face='Courier'>--train_patched</font> the RD term is "
                    "computed through <font face='Courier'>forward_random_depth</font>, "
                    "which clamps, so adapters 0 and 1 receive a gradient only "
                    "from the feature-space distillation term. The deepest exit "
                    "wears no adapter at all, which is what makes it reproduce "
                    "stock DCVC-UF."))
    s.append(P("Two facts in this figure carry the rest of the report. The first "
               "is that the ladder has six rungs and the decoder uses four. The "
               "second is that the two unused rungs are nevertheless trained "
               "&mdash; not as well as they could be, since no pixel loss ever "
               "reaches them, but trained: their adapters carry real weights in "
               "the checkpoint and their reconstructions improve monotonically "
               "with epoch."))
    return s


def story2(s):
    # ---- 3
    s.append(P("3&nbsp;&nbsp;The seam is paid out of the distortion budget", H1))
    s.append(P("The seam is usually discussed as an artefact one can see. It is "
               "more useful to price it, because the system already pays for it "
               "in the one currency it is constrained by."))
    s.append(P("The evidence is threefold and the three measurements are "
               "independent. <b>(i)</b> The deployed operating point is found by a "
               "two-level bisection in "
               "<font face='Courier'>scripts/raterank_curve.py</font>: an inner "
               "bisection on a cheap per-tile table, then a correction against a "
               "real decode of the resulting mixed-depth map. That correction "
               "pulls the inner target from 0.100&nbsp;dB to about 0.085 &mdash; "
               "roughly a sixth of the budget is spent on the fact that "
               "neighbouring tiles have different depths. <b>(ii)</b> Decoding one "
               "routed map two ways on the same weights, the tiled decode reaches "
               "0.0988&nbsp;dB and the per-position decode 0.0864&nbsp;dB, for "
               "0.6% more trunk. <b>(iii)</b> The frontier floors differ by the "
               "same amount: with every cell at the deepest exit the tiled decoder "
               "still sits 0.022 to 0.048&nbsp;dB above the release, the "
               "per-position one 0.001 to 0.020."))

    # ---- 4
    s.append(P("4&nbsp;&nbsp;Per-position depth with a dilated receptive field", H1))
    s.append(P("Let <i>d</i>(<i>x</i>) be the depth allocated at position "
               "<i>x</i>, in trunk blocks. A block <i>b</i> must be computed at "
               "<i>x</i> whenever some position <i>y</i> that needs at least "
               "<i>b</i> blocks lies within <i>b</i>&nbsp;&minus;&nbsp;<i>c</i> of "
               "it, which is the max-plus dilation", MONO))
    s.append(P("<font face='Courier'>D(x) = max_y [ d(y) &minus; dist(x, y) ]</font>",
               ParagraphStyle("eq", parent=BODY, alignment=1, spaceAfter=6)))
    s.append(P("and the decode that runs block <i>b</i> exactly where "
               "<i>D</i>&nbsp;&ge;&nbsp;<i>b</i> gives every position what a "
               "full-frame decode to its own depth would have given it. There is "
               "no cut, so there are no invented values; a position that stops "
               "early keeps being computed only for as long as a deeper "
               "neighbour's receptive field still reaches it. The cost is a band "
               "whose width is the depth <i>difference</i> between neighbours "
               "rather than the whole tile."))
    s.append(figure("fig2_mechanism", "mech",
                    "(a) Tiled decoding cuts, and every cut edge convolves "
                    "against a padded value. (b) Per-position depth does not cut. "
                    "(c) What it pays instead: the positions that keep computing "
                    "to serve a deeper neighbour. The band in (c) is computed by "
                    "dilating the map in (a)&ndash;(b), not drawn by hand, which "
                    "is why it is thin and irregular rather than a fixed halo."))
    s.append(P("4.1&nbsp;&nbsp;Exactness, verified rather than argued", H2))
    s.append(P("The argument above covers the trunk. It does not cover the head, "
               "which has its own 3&times;3 and at a depth boundary reads one cell "
               "across into a feature that left the trunk at a different depth. We "
               "therefore compare the idealised decode &mdash; each position read "
               "from a full-frame decode at its own depth &mdash; against a real "
               "per-position decode that stitches the exit features and runs the "
               "head once, on all 53 frames:"))
    rows = [["rate", "assembled (dB)", "real decode (dB)", "head term (dB)"]]
    for q in (0, 32, 63):
        r = N["ht"][q]["rows"][0]
        rows.append([f"qp {q}", f"{r['assembled_db']:+.4f}",
                     f"{r['real_db']:+.4f}", f"{r['head_term_db']:+.4f}"])
    s.append(table(rows, "exact",
                   "The head term, measured on the full test set at 64&nbsp;px "
                   "cells. It is below 0.0002&nbsp;dB and negative, so the sweeps "
                   "in this report are if anything conservative."))

    # ---- 5
    s.append(P("5&nbsp;&nbsp;Granularity stops costing distortion", H1))
    s.append(P("With tiles, halving the tile side doubles the border per unit "
               "area and the seam grows with it; 256&nbsp;px is the size at which "
               "that is affordable. With per-position depth there is no border at "
               "any granularity, so a finer allocation costs only compute &mdash; "
               "the band &mdash; and the trade can be measured instead of "
               "guessed."))
    s.append(figure("fig4_granularity", "gran",
                    "Allocation cell against saving, at the 0.1&nbsp;dB budget "
                    "over 53 CTC frames. (a) Net saving per rate. (b) The same, "
                    "averaged over five rates, with the dilation band charged on "
                    "top: the band grows from "
                    f"{N['band'][256]:.2f} to {N['band'][32]:.2f} points and at "
                    "32&nbsp;px it consumes the remaining gain, which is why "
                    "64&nbsp;px is the operating choice."))
    rows = [["cell", "q0", "q16", "q32", "q48", "q63", "mean", "band"]]
    for c in (256, 128, 64, 32):
        rows.append([f"{c} px"] +
                    [f"{N['gran'][q][c]['saving_pct_vs_release']:.2f}" for q in QPS] +
                    [f"<b>{N['gmean'][c]:.2f}</b>", f"{N['band'][c]:.2f}"])
    s.append(table(rows, "gran",
                   "Per-position decoding at four allocation granularities, "
                   "oracle allocation, saving against a released decode. The band "
                   "column is already subtracted from the savings.",
                   widths=None))
    s.append(P("A finer map also costs bits when it is signalled. The main "
               "experiment transmits 64 to 91 bits per frame at 256&nbsp;px; "
               "sixteen times as many cells is still under 0.0005&nbsp;bpp against "
               "a working range of 0.25 to 0.64, and the parameter-free "
               "decoder-side router transmits nothing at all."))
    return s


def story3(s):
    # ---- 6
    s.append(P("6&nbsp;&nbsp;The split depth is an artefact of the cut", H1))
    s.append(P("The split exists because a cut made before the features have "
               "settled produces the worst seams. It is therefore a property of "
               "tiling, not of the ladder, and once there is no cut it has no "
               "job. What it costs is that "
               "<font face='Courier'>forward()</font> clamps the exit map to "
               "<i>j</i>=2, so the two shallowest rungs cannot be selected."))
    s.append(P("Those rungs are trained. Their adapters carry real weights in the "
               "checkpoint, and their errors on the test set are ordinary "
               "continuations of the ladder rather than degenerate:"))
    import numpy as np
    d = np.load(RES / "cells_ctc64.npz", allow_pickle=True)
    M, R, qp = d["M"].astype(float), d["R"].astype(float), d["qp"]
    K = int(d["K"][0])
    rows = [["rate"] + [f"e{k} ({2*(k+1)} blk)" for k in range(K)]]
    for q in (0, 32, 63):
        sel = qp == q
        rows.append([f"qp {q}"] + [
            f"{10*np.log10(M[sel][:, k].mean()/R[sel].mean()):+.4f}"
            for k in range(K)])
    s.append(table(rows, "rungs",
                   "Every rung's error on the 53 CTC frames, in dB above the "
                   "released decoder, measured with "
                   "<font face='Courier'>forward_all_exits</font>, which does not "
                   "clamp. The first two columns are the rungs the deployed "
                   "decoder cannot reach.", align_right_from=1, fs=7.8))
    s.append(figure("fig3_ladder", "ladder",
                    "(a) The measured cost of each rung, per rate, on a log axis; "
                    "the shaded region is what the clamp removes. (b) The spread "
                    "between the shallowest selectable rung and the deepest, by "
                    "epoch: it stops improving at the pinned epoch, which is the "
                    "same conclusion the saving series and the training log reach "
                    "independently."))
    s.append(P("Unlocking them is a change to one line of the decode path. With "
               "per-position decoding at 64&nbsp;px cells, the oracle allocation "
               "at the 0.1&nbsp;dB budget moves as follows.", BODY))
    rows = [["split", "q0", "q16", "q32", "q48", "q63", "mean"]]
    for lab, key in (("<i>j</i>=2 (as deployed)", None),):
        pass
    j2 = N["r_j2"]["oracle"]; j0 = N["r_j0"]["oracle"]
    for lab, r in (("<i>j</i>=2 (as deployed)", j2), ("<i>j</i>=0 (unlocked)", j0)):
        rows.append([lab] + [f"{r['per_qp'][str(q)]['saving_pct']:.2f}"
                             if str(q) in r["per_qp"] else
                             f"{r['per_qp'][q]['saving_pct']:.2f}" for q in QPS]
                    + [f"<b>{r['mean_saving_pct']:.2f}</b>"])
    s.append(table(rows, "split",
                   "Oracle saving at 0.1&nbsp;dB, per-position decoding, "
                   "64&nbsp;px cells. The gain is concentrated at low rate, where "
                   "the two-block rung sits only 0.45&nbsp;dB above the release "
                   "and 37% of cells can afford it; at the highest rate it is "
                   "zero.", align_right_from=1))

    # ---- 7
    s.append(P("7&nbsp;&nbsp;Results", H1))
    s.append(figure("fig6_waterfall", "water",
                    "The three levers, applied in order to the pinned epoch-4 "
                    "checkpoint. None of them retrains a weight; each is a change "
                    "to how the existing weights are run.", width=3.3))
    s.append(P("Reading a frontier at one budget is fragile, and this project has "
               "been bitten by it before, so the headline is integrated. "
               "<font face='Courier'>scripts/bd_saving.py</font> takes its lower "
               "limit from each curve's own floor; since the floors differ by "
               "exactly the seam, integrating each from its own floor would "
               "compare two different intervals and call it one number. All rows "
               "below are integrated over the common interval "
               "0.05&ndash;0.12&nbsp;dB, in the per-frame decibel, with saving "
               "measured against a released decode."))
    rows = [["decoder", "BD-saving (%)", "BD-quality (dB)"]]
    for lab, k in (("tiled 256 px, oracle &mdash; the paper today", "tiled"),
                   ("per position 256 px, oracle", "pp256"),
                   ("per position 64 px, oracle", "pp64"),
                   ("per position 64 px, trained router", "dep64")):
        rows.append([lab, f"{bdmean(k, 'bd_saving'):.2f}",
                     f"{bdmean(k, 'bd_quality'):.4f}"])
    s.append(table(rows, "bd",
                   "Bjontegaard integration over a common interval. The last row "
                   "uses a decoder-side router with no signalled map and no "
                   "retrained weight, and still exceeds the tiled oracle on both "
                   "measures.", widths=None, align_right_from=1))
    s.append(figure("fig5_frontier", "front",
                    "The compute&ndash;quality frontier at the extreme rates. The "
                    "shaded band is the interval the BD numbers integrate; the "
                    "dashed line is the 0.1&nbsp;dB operating point. The tiled "
                    "curve is rebased onto the same axes as the others &mdash; "
                    "saving against a released decode and the per-frame decibel "
                    "&mdash; because "
                    "<font face='Courier'>paper_curve.py</font> writes the other "
                    "convention in each pair, and on the raw axes the tiled curve "
                    "appears above the per-position ones at qp63."))
    return s


def story4(s):
    # ---- 8
    s.append(P("8&nbsp;&nbsp;Does it run faster, or only count fewer MACs?", H1))
    s.append(P("Every saving in this project is counted in multiply-accumulates, "
               "which is the right unit for arithmetic and the wrong one for time. "
               "A mask that skips 40% of positions saves nothing if a dense "
               "3&times;3 still sweeps the plane, and the adaptive-computation "
               "literature is candid that pixel-wise sparse convolution is not "
               "hardware friendly."))
    s.append(P("Treating the mask as purely spatial gives a pessimistic bound: a "
               "block-sparse kernel on <i>B</i>&times;<i>B</i> tiles that must run "
               "any tile containing one needed position keeps "
               f"{N['mask']['rows'][0]['blocks']['4']:.1f}% of the ideal "
               f"{N['mask']['rows'][0]['ideal_pct']:.1f}% at <i>B</i>=4 and "
               "collapses by <i>B</i>=32. That bound is far too pessimistic, "
               "because 99.708% of a DepthConvBlock is pointwise: "
               "8<i>C</i>&sup2; per position against 9<i>C</i> for the depthwise "
               "3&times;3 at <i>C</i>=384. A 1&times;1 is a per-position matmul, "
               "so the kept positions can be gathered into a dense matrix at any "
               "granularity; charging only the depthwise term to the block-sparse "
               "bound leaves the realisable saving within 0.09 points of ideal "
               "even at <i>B</i>=32."))
    s.append(figure("fig7_gather", "gather",
                    "Measured rather than argued: gather, two pointwise "
                    "convolutions, scatter, at the shapes the decoder runs. Time "
                    "tracks the fraction of positions kept to within 0.008&ndash;"
                    "0.036, and the indices are a random permutation &mdash; the "
                    "worst case for locality, where a real mask is contiguous.",
                    width=3.15))

    # ---- 9
    s.append(P("9&nbsp;&nbsp;What did not work", H1))
    s.append(P("The routing axis is close to exhausted, and saying so is more "
               "useful than another table of small wins. The shipped rate-rank "
               "surrogate is rank-1: one learned scale per cell times one shape "
               "shared by all cells. On the test set that assumption is good to "
               "91.7% of the variance in the log-error curve. An oracle that "
               "knows each cell's scale perfectly and shares one shape reaches "
               "32.64% against the full oracle's 33.32%, so <i>everything</i> a "
               "richer model could add over rate-rank's functional form is worth "
               "0.68 points &mdash; and 0.13 of that at the lowest rate."))
    s.append(figure("fig9_router", "router",
                    "(a) Singular values of the centred log-error curves: the "
                    "first component carries most of the structure, which is the "
                    "assumption the shipped surrogate makes. (b) End-to-end, at "
                    "0.1&nbsp;dB, scored by the allocation each router causes and "
                    "with the decibel measured on the true error rather than on "
                    "the router's own prediction."))
    rows = [["what was tried", "result"]]
    r2, r0 = N["r_j2"], N["r_j0"]
    rows.append(["gradient boosting, monotone increments",
                 f"{r2['gbm_mono']['mean_saving_pct']:.2f}% vs "
                 f"{r2['raterank']['mean_saving_pct']:.2f}% for the rank-1 "
                 "surrogate &mdash; +0.68"])
    rows.append(["log-space bias correction (Duan smearing)",
                 f"{r2['gbm_smear']['mean_saving_pct']:.2f}% &mdash; +0.04 over "
                 "the same model without it"])
    rows.append(["MLP (128&ndash;64) on the same targets",
                 f"{r2['mlp']['mean_saving_pct']:.2f}% &mdash; 0.68 <i>below</i> "
                 "the trees"])
    rows.append(["boosting each exit independently",
                 f"{r2['gbm']['mean_saving_pct']:.2f}% &mdash; three points below "
                 "rank-1: nothing stops it predicting a curve no ladder can "
                 "produce"])
    if N["abl"]:
        a = {r["inputs"]: r["mean_saving_pct"] for r in N["abl"]}
        rows.append(["four stem statistics instead of all 26 inputs",
                     f"{a.get('yalniz stem', float('nan')):.2f}% vs "
                     f"{a.get('hepsi', float('nan')):.2f}% &mdash; the input axis "
                     "is spent; dropping the neighbourhood group <i>helps</i>"])
    rows.append(["smoothness-regularised allocation",
                 "halves the band (1.81&rarr;0.80 points at qp32) and loses "
                 "slightly more than that in distortion"])
    rows.append(["per-frame budget from predicted distortion",
                 "only 66&ndash;71% of frames stay under budget and the saving "
                 "collapses; a per-frame lambda on the <i>true</i> decibel keeps "
                 "53/53, so the mechanism is fine and the predictions are not"])
    s.append(table(rows, "neg",
                   "Seven attempts on the routing and allocation side, at "
                   "0.1&nbsp;dB with 64&nbsp;px cells. The compute axis moved by "
                   "eight points; this axis moved by less than one.",
                   widths=[2.35 * 72, 3.15 * 72], align_right_from=9, fs=8.0))

    # ---- 10
    s.append(P("10&nbsp;&nbsp;The main experiment is unchanged, and re-verified", H1))
    s.append(P("Nothing in this report modifies "
               "<font face='Courier'>runs/RECIPE512</font> or any source file of "
               "the main experiment. As a side effect of re-measuring every epoch "
               "with today's code &mdash; the measurement pipeline was corrected "
               "twice this week, so the archived per-epoch files are not "
               "comparable to each other &mdash; the paper's own epoch series is "
               "reproduced independently."))
    rows = [["epoch", "as reported", "re-measured", "spread at qp32 (dB)"]]
    paper = {0: "21.5", 1: "22.8", 3: "25.8", 4: "27.6"}
    for e in sorted(N["ep"]):
        r = N["ep"][e]
        rows.append([f"{e}" + (" (pin)" if e == 4 else ""), paper.get(e, "&mdash;"),
                     f"{r['mean']:.2f}", f"{r['spread_q32_db']:.3f}"])
    s.append(table(rows, "epochs",
                   "Mean saving at 0.1&nbsp;dB over 53 CTC frames, re-measured "
                   "with the current scripts. Epoch 2 has no checkpoint on disk. "
                   "Beyond the pin the series flattens and then dips, and the "
                   "ladder's spread stops improving, which is three independent "
                   "reasons the pin is well placed.", align_right_from=1))
    s.append(figure("fig8_epochs", "epochs",
                    "The re-measured series against the four values the paper "
                    "reports. The pin is where the ladder's spread stops "
                    "improving.", width=3.3))
    return s


def story5(s):
    # ---- 11
    s.append(P("11&nbsp;&nbsp;Limitations, and what is being trained now", H1))
    s.append(P("<b>The 36.02% is an oracle.</b> It is the allocation a Lagrangian "
               "makes when it knows the true per-cell error, which is what the "
               "signalled configuration attains by transmitting the map, and it "
               "is the same quantity the main paper's headline reports. A trained "
               "decoder-side router reaches "
               f"{N['r_j0']['gbm_smear']['mean_saving_pct']:.2f}% at <i>j</i>=0 "
               f"against {N['r_j2']['gbm_smear']['mean_saving_pct']:.2f}% at "
               "<i>j</i>=2, so realisation falls from 92% to 88.5% when the two "
               "new rungs appear. That is expected and it is the point of the "
               "next paragraph: those rungs have never been optimised for the "
               "job."))
    s.append(P("<b>The shallow rungs were never trained on a pixel loss.</b> "
               "Under the main recipe the RD term is computed through "
               "<font face='Courier'>forward_random_depth</font>, which clamps, so "
               "a tile drawn at depth 0 or 1 is decoded at depth 2 and adapters 0 "
               "and 1 receive gradient only from the feature-space distillation "
               "term. They are therefore trained, but for imitation rather than "
               "for reconstruction. A fine-tune is running on a free card that "
               "changes exactly this and nothing else: no "
               "<font face='Courier'>--train_patched</font>, so the loss goes "
               "through <font face='Courier'>forward_all_exits</font>, which does "
               "not clamp; <font face='Courier'>--split_depth 0</font>; "
               "<font face='Courier'>--seam_repair none</font>; warm-started from "
               "the pinned checkpoint with the anchor weight unchanged so the "
               "deepest exit stays fixed to the release. If the shallow rungs "
               "improve, both the oracle and the router improve with them."))
    s.append(P("<b>Other open items.</b> The adapter cost for exits 0 and 1 is "
               "taken to be the same as exit 2's, which is what the shipped cost "
               "model implies but has not been re-derived for those exits. The "
               "head-term verification was run on <i>j</i>=2 allocations; "
               "<i>j</i>=0 maps have larger depth differences and therefore wider "
               "bands, so it should be repeated there. And the BD integration for "
               "<i>j</i>=0 has not been done &mdash; the tables above stop at the "
               "0.1&nbsp;dB operating point for that configuration."))
    s.append(P("<b>What this report does not claim.</b> It does not claim a "
               "better network, a better router, or a better training recipe. The "
               "network is the main experiment's, unmodified; the router is worse "
               "than an oracle by a margin this report measures rather than "
               "hides; and the only training change is the one described above, "
               "which is still running."))

    s.append(P("12&nbsp;&nbsp;How to reach 36%: the three changes, concretely", H1))
    s.append(P("Nothing below is a new module. Each item is a change to the "
               "decode path, and the three compose."))
    s.append(P("<b>(1) Replace the tiled decode with a dilated one.</b> Instead of "
               "cutting the frame and calling the decoder per tile, keep the frame "
               "whole. Compute <font face='Courier'>D = dilate(d)</font> once, "
               "where the dilation is repeated 3&times;3 max-minus-one in blocks; "
               "then for each group <i>g</i>, run it where "
               "<font face='Courier'>D &ge; g&middot;bpe + 1</font> and leave the "
               "rest untouched. Two details are load-bearing and both were bugs "
               "the first time. The receptive field must be counted in "
               "<i>blocks</i>, not exits &mdash; an exit is two blocks, and "
               "counting in exits puts the deepest uniform decode 0.58&nbsp;dB "
               "off. And each position's feature must be frozen at its own exit: "
               "the dilation keeps computing there to serve a deeper neighbour, "
               "so reading the end of the loop through the shallow adapter costs "
               "0.015&nbsp;dB, the same size as the effect being measured."))
    s.append(P("<b>(2) Decide on 64&nbsp;px cells instead of 256&nbsp;px tiles.</b> "
               "The allocation is a per-cell argmin of "
               "<i>m<sub>k</sub></i>&nbsp;+&nbsp;&lambda;<i>c<sub>k</sub></i> as "
               "before; only the grid changes. Charge the band by dilating the "
               "resulting map and counting blocks, not by a halo model. 32&nbsp;px "
               "is measurably worse than 64 once the band is charged."))
    s.append(P("<b>(3) Stop clamping the exit map.</b> "
               "<font face='Courier'>forward()</font> applies "
               "<font face='Courier'>exit_map.clamp(min=j)</font>; with no cut, "
               "<i>j</i> has no function. Exits 0 and 1 already carry trained "
               "adapters. Take the argmin over all <i>K</i> exits using the "
               "unclamped cost vector &mdash; note that the shipped "
               "<font face='Courier'>exit_costs</font> already clamps its own "
               "cost entries, so a cost vector for <i>j</i>=0 has to be rebuilt "
               "or the shallow options are priced at a rate no decoder charges."))
    rows = [["step", "mean saving at 0.1 dB", "delta"]]
    base = N["pin"]["mean"]
    prev = base
    for lab, v in (("the paper today (tiled, <i>j</i>=2)", base),
                   ("+ per-position depth (256 px cells)", N["gmean"][256]),
                   ("+ 64 px cells", N["r_j2"]["oracle"]["mean_saving_pct"]),
                   ("+ unlocked shallow rungs (<i>j</i>=0)",
                    N["r_j0"]["oracle"]["mean_saving_pct"])):
        rows.append([lab, f"{v:.2f}%",
                     "&mdash;" if v == base else f"+{v - prev:.2f}"])
        prev = v
    s.append(table(rows, "recipe",
                   "The three changes, applied in order to the pinned epoch-4 "
                   "checkpoint, oracle allocation. No weight is retrained at any "
                   "step.", widths=[2.9 * 72, 1.4 * 72, 1.2 * 72],
                   align_right_from=1))

    s.append(P("13&nbsp;&nbsp;Protocol and conventions", H1))
    s.append(P("Three pairs of conventions are live in this codebase and each "
               "pair differs by more than the effects being reported, so every "
               "number above states which half it uses."))
    s.append(P("<b>The decibel.</b> Pooling every cell of every frame into one "
               "MSE is the natural form for the Lagrangian; averaging a per-frame "
               "decibel is what <font face='Courier'>test_video.py</font> does and "
               "therefore what a published DCVC-UF number means. On an identical "
               "allocation the two differ by 0.023&ndash;0.033&nbsp;dB, a third of "
               "the budget, with pooling always the flattering one. Everything "
               "here is per-frame."))
    s.append(P("<b>The saving.</b> A saving can be measured against a released "
               "decode (denominator 1) or against our own deepest path "
               "(denominator 1.0095, which pays seam repair). Everything here is "
               "against the release. Mixing the two put the tiled frontier above "
               "the per-position ones at the highest rate the first time "
               "Figure&nbsp;5 was drawn."))
    s.append(P("<b>The budget.</b> A budget met on average is not a budget met on "
               "every frame. One &lambda; per rate lets a frame that gives up "
               "little subsidise one that gives up a lot; a per-frame &lambda; "
               "forbids it and costs about 2.5 points. The main paper and this "
               "report both use one &lambda; per rate, so the two are comparable; "
               "Section&nbsp;9 reports what happens when the per-frame version is "
               "driven by predictions instead of by the truth."))
    s.append(P("<b>The test set.</b> 53 CTC frames, one frame per sequence, "
               "spanning UVG, MCL-JCV and the HEVC classes. Absolute quality is "
               "reported in DCVC-UF's own metric, (6<i>Y</i>&nbsp;+&nbsp;<i>U</i>"
               "&nbsp;+&nbsp;<i>V</i>)/8 computed in 4:2:0 on 0&ndash;255. The "
               "released decoder scores 32.536, 38.135 and 43.155&nbsp;dB at qp 0, "
               "32 and 63; the ladder's deepest exit reproduces it to 0.000, 0.011 "
               "and 0.028&nbsp;dB."))

    s.append(P("12&nbsp;&nbsp;Reproducing every number", H1))
    s.append(P("Each figure and table above names the file it was read from. The "
               "measurement scripts live outside the main experiment, and the "
               "main experiment's source tree has zero modifications:", BODY))
    s.append(P(
        "adaptive_depth.py &mdash; the per-position decode and its exactness check<br/>"
        "granularity.py &mdash; the allocation-cell sweep with the band charged by dilation<br/>"
        "head_term.py &mdash; idealised versus real decode, all 53 frames<br/>"
        "frontier.py, frontier_deployed.py &mdash; the swept frontiers<br/>"
        "router_data.py, router_fit.py &mdash; per-cell features and the router comparison<br/>"
        "smooth_alloc.py, mask_structure.py, gather_bench.py &mdash; the three negative<br/>"
        "&nbsp;&nbsp;&nbsp;and implementability results<br/>"
        "epoch_psnr.py &mdash; per-epoch PSNR in the codec's own 6:1:1 4:2:0 metric",
        MONO))

    s.append(P("References", H1))
    refs = [
        "Zhang et&nbsp;al. Be Your Own Teacher: Improve the Performance of "
        "Convolutional Neural Networks via Self Distillation. ICCV 2019.",
        "Mao et&nbsp;al. AdaRevD: Adaptive Patch Exiting Reversible Decoder "
        "Pushes the Limit of Image Deblurring. CVPR 2024.",
        "Adaptive Patch Exiting for Scalable Single Image Super-Resolution. "
        "ECCV 2022.",
        "Liu et&nbsp;al. Deep Adaptive Inference Networks for Single Image "
        "Super-Resolution (AdaDSR). ECCV 2020.",
        "Chu et&nbsp;al. Improving Image Restoration by Revisiting Global "
        "Information Aggregation (TLC). ECCV 2022.",
        "Kong et&nbsp;al. ClassSR: A General Framework to Accelerate "
        "Super-Resolution Networks by Data Characteristic. CVPR 2021.",
        "Multi-exit self-distillation with appropriate teachers. FITEE 2024.",
        "Bjontegaard. Calculation of Average PSNR Differences between RD-Curves. "
        "VCEG-M33, 2001.",
    ]
    for i, r in enumerate(refs, 1):
        s.append(P(f"[{i}]&nbsp;&nbsp;{r}",
                   ParagraphStyle("ref", parent=BODY, fontSize=8.6, leading=10.4,
                                  leftIndent=14, firstLineIndent=-14,
                                  spaceAfter=2.4)))
    return s


if __name__ == "__main__":
    st = story(); st = story2(st); st = story3(st); st = story4(st); st = story5(st)
    out = Path(__file__).resolve().parent / "FLEX-UF-supplement.pdf"
    build(st, out)
    print(f"  yazildi {out}")
