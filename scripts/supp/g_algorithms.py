"""Algorithms, and how each step is implemented.

Section G of the supplement. The other sections say what was measured and what
it means; this one says how each move is carried out, in the files that carry
it out. Five procedures hold the system up: the encoder-side search for an
allocation, the decoder-side prediction of one, the hybrid that mixes them, the
coding of the map, and one training step. Each gets an algorithm box
written at the level of the calls the code makes and the shapes they return.
After them comes a table of the operations themselves, with the file and the
function for each, and then the places where the obvious implementation is
wrong.

What this section deliberately does NOT do is restate a procedure another
section already states. A.7 boxes the deployed decode and A.12 boxes the search
as a procedure; the boxes here name the function on every line and carry the
shape it returns, which is a different artefact for a different reader. B.6
prices the gap between the cost model and the meter; this section describes the
meter. C derives the allocation rule; this section describes the argmin that
runs it. F measures the router's inputs; this section describes the six lines
that turn its logits into an exit map.

Every number is read from results/ at build time by k.J. Nothing here opens a
checkpoint, allocates on a GPU, or writes anything.

Three numbers are typed rather than read, each with the source comment it was
typed from, because no file in results/ carries them; G.10 says so again.
"""

# ---------------------------------------------------------------------------
# Typed constants. Each names the file it came from. These are the only three
# numbers in the section that k.J cannot supply.
# ---------------------------------------------------------------------------

# flexuf/measure.py, the comment on the batch dimension: what the deepest exit
# measured when the meter counted output positions as height times width,
# against the figure in results/ceiling_measured.json.
METER_BUG = "0.43"

# tests/test_sorted_tiles.py, the tolerance table: exact on CUDA, and the CPU
# figure the oneDNN blocking forces.
SORTED_TOL = ("0.0", "1×10<super>-6</super>")

# scripts/signalled_curve.py, the docstring of map_bits: what merging the three
# indices that share the cheapest cost would take off the reported map, at the
# two ends of the rate range.
MERGE_SAVING = ("7", "0")


# ---------------------------------------------------------------------------
# Box helpers. Non-breaking spaces because reportlab collapses runs of ordinary
# ones, and an algorithm box whose indentation has been collapsed is unreadable.
# ---------------------------------------------------------------------------
IND = " " * 4
GAP = " " * 3


def _c(text):
    """A trailing comment inside a box."""
    return GAP + f"<i>{text}</i>"


def _box(k, lines, cap):
    """A pseudocode box: one column, ruled above and below, no header row."""
    return k.rows([[ln] for ln in lines], cap, header=False)


def _sci(x, d=1):
    """4.7e-05 -> 4.7×10<super>-5</super>, with a glyph reportlab can draw."""
    from math import floor, log10
    if x == 0:
        return "0"
    e = floor(log10(abs(x)))
    m = x / (10.0 ** e)
    return f"{m:.{d}f}×10<super>{e}</super>"


def _n(x):
    """Thousands separators, for a MAC count."""
    return f"{int(round(x)):,}"


def _n_equiv() -> int:
    """How many properties tests/test_equivalence.py actually holds.

    It said six and there are ten. A count typed into a sentence about tests
    is the same class of number as any other typed into a sentence about
    measurements, and it drifts the same way: four tests were added and the
    word did not move.
    """
    import re
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "tests/test_equivalence.py"
    try:
        return len(re.findall(r"^def test_", p.read_text(), re.M))
    except OSError:
        return 0


def content(k):
    sec = k.h1("Algorithms, and how each step is implemented")

    # ------------------------------------------------------------------ G.1
    k.h2("What this section covers")

    td = k.J("tile_definition.json")
    ac = k.J("adapter_cost.json")
    cm = k.J("ceiling_measured.json")
    ec = k.J("supp_encoder_cost_PAPER.json")
    sig = k.J("signalled_RECIPE512_ctc53.json")
    hyb = k.J("hybrid_RECIPE512_b01_e4head.json",
                "hybrid_RECIPE512_b01_fixed.json")

    cfg = td["config"]
    K = cfg["num_exits"]
    j = cfg["split_depth"]
    # b and the trunk length come off the ladder rather than being typed:
    # exit 0 skips (K-1-0)b of the N = Kb blocks.
    b = ac["exits"][0]["blocks_skipped"] // (K - 1)
    N = K * b
    P = td["feature_patch"]
    r1080 = [r for r in td["rows"] if r["name"] == "1920x1080"][0]

    k.par(
        "Sections A to F state what the system is, what it costs and what it "
        "delivers. This one states how it is done: the five procedures the "
        "work rests on, each as an algorithm box whose every line names the "
        "function that carries it out and the shape it returns, followed by "
        "the tensor operations underneath them and the failure each one is "
        "written to avoid. A reader who wants to reproduce the system reads "
        "A for the recipe and this section for the moves.")

    k.par(
        "The geometry is fixed throughout and is the pinned checkpoint's. The "
        f"ladder has K = {K} exits over the released decoder's {N} trunk "
        f"blocks, so b = {b} blocks per exit, and the split depth is j = {j}, "
        f"which leaves exits {j} to {K - 1} selectable and the first {j * b} "
        f"blocks full-frame. One tile is {td['rgb_patch']} px of RGB, "
        f"{td['feature_patch']} px of the feature grid the per-tile trunk runs "
        f"on and {td['latent_patch']} px of the latent. A 1920×1080 frame is "
        f"replicate-padded on its bottom and right to "
        f"{r1080['padded'][0]}×{r1080['padded'][1]}, which is a feature grid "
        f"of {r1080['feature_in_patchify'][1]}×"
        f"{r1080['feature_in_patchify'][0]} and "
        f"{r1080['tiles_measured']} tiles in {r1080['nh']} rows of "
        f"{r1080['nw']}. Write T for the tile count of a frame, c for the "
        "per-exit cost vector in units of one released decode, and M for the "
        "table of per-tile errors with one row per tile and one column per "
        "exit. Those three objects are what every procedure below "
        "manipulates.")

    rows = [["Procedure", "File", "Entry point"]]
    rows += [
        ["the deployed decode", "decoder.py", "forward"],
        ["the same, tiles ordered by depth", "decoder.py", "_suffix_sorted"],
        ["per-exit price of a tile", "cost.py", "exit_costs"],
        ["frame-level price of a map", "cost.py", "frame_relative_cost"],
        ["MACs a decode performed", "measure.py", "MacMeter"],
        ["saving, as the meter counts it", "measure.py",
         "measured_saving_pct"],
        ["per-tile error at every exit", "eval.py", "tiled_exit_mses"],
        ["error a mixed map delivers", "eval.py", "true_frame_mse"],
        ["the encoder-side search", "signalled_curve.py",
         "at_lam, bisect_to, true_db"],
        ["cost of the exit map, in bits", "signalled_curve.py", "map_bits"],
        ["the tilt and its bisection", "beta.py", "exit_map, bisect_beta"],
        ["per-tile logits", "head2.py", "StemRouterHeadV2.forward"],
        ["which tiles to override", "hybrid_curve.py", "hybrid_map, h2"],
        ["one training step", "train_flexuf_image.py", "train_one_epoch"],
        ["the training forwards", "model.py", "forward_random_depth"],
        ["the objective", "losses.py", "multi_exit_rd_loss"],
        ["adapters and the seam gate", "decoder.py",
         "build_adapter, GridSeamRepair"],
        ["tile-border padding", "padding.py", "wrap_tile_padding"],
    ]
    t_map = k.rows(
        rows,
        "Where each procedure lives. decoder.py and padding.py are under "
        "flexuf/backbone/ and head2.py under flexuf/router/; the two curve "
        "scripts are under scripts/; train_flexuf_image.py is at the "
        "repository root and the rest are flexuf/ itself. Take from it the "
        "division the rest of the section relies on: everything that "
        "decides an allocation is in the scripts and in beta.py, everything "
        "that executes one is under backbone/, and the two meet only "
        "through the two objects in the middle of the table, a table of "
        "per-tile errors and a vector of per-exit prices.")

    # ------------------------------------------------------------------ G.2
    k.h2("The encoder-side search")

    k.par(
        "The encoder holds the source, so it can evaluate the allocation rule "
        "exactly rather than predict it. Doing so means building the table M "
        "once per rate and then running an argmin over its columns for every "
        "candidate multiplier, which is why the expensive part is hoisted out "
        "of the search entirely: the per-tile errors, the released decoder's "
        "reference error and the bitrate are functions of the latent alone, "
        "and only the argmin and the aggregates move with the multiplier. "
        "Both loops of the bisection then cost tensor operations on a table "
        f"of {r1080['tiles_measured']} rows by {K} columns.")

    box_search = _box(k, [
        "<b>input</b> frames F, budget t in dB, cost vector c of length K",
        "&nbsp;1&nbsp; <b>for</b> each frame in F:" + _c("once per rate"),
        IND + "&nbsp;2&nbsp; xp = replicate-pad to whole tiles, bottom and "
        "right only",
        IND + "&nbsp;3&nbsp; y, q, aux = net._encode_to_latent(xp, qp)",
        IND + "&nbsp;4&nbsp; M = tiled_exit_mses(net.dec, y, q, xp, cfg)"
        + _c("[T, K]"),
        IND + "&nbsp;5&nbsp; R = reference_frame_mse(ref.dec, y, q, xp)"
        + _c("scalar"),
        "&nbsp;6&nbsp; floor = trueDb(0)" + _c("one decode per frame"),
        "&nbsp;7&nbsp; <b>if</b> floor &gt; t: record budget_reachable = "
        "false, stop",
        "&nbsp;8&nbsp; inner = t",
        "&nbsp;9&nbsp; <b>repeat</b> at most 6 times:",
        IND + "10&nbsp; lam = bisectTo(inner)" + _c("no decode"),
        IND + "11&nbsp; d = trueDb(lam)" + _c("one decode per frame"),
        IND + "12&nbsp; <b>if</b> |d − t| &lt; 5e-4: <b>leave</b>",
        IND + "13&nbsp; inner = min(1, max(1e-4, inner + (t − d)))",
        "14&nbsp; <b>for</b> each frame, at that lam:",
        IND + "15&nbsp; k = (M + lam · c).argmin(1)" + _c("[T], the map"),
        IND + "16&nbsp; bits = map_bits(k, K);&nbsp; bpp = bpp + bits / "
        "pixels",
        IND + "17&nbsp; saving = measured_saving_pct(net.dec, ref.dec, "
        "y, q, clamp(k, j))",
        "18&nbsp; report d, and the mean over frames of saving and bits",
        "<b>where</b>",
        IND + "k(lam) = (M + lam · c).argmin(1)",
        IND + "tableDb(lam) = 10 log<sub>10</sub>( mean<sub>t</sub> "
        "M[t, k(lam)<sub>t</sub>] / R )",
        IND + "trueDb(lam) = 10 log<sub>10</sub>( true_frame_mse(dec, y, q, "
        "xp, clamp(k(lam), j)) / R )",
        IND + "bisectTo(u): lo, hi = 0, 1; 60 times: mid = (lo + hi) / 2;",
        IND + IND + "<b>if</b> tableDb(mid) ≤ u <b>then</b> lo = mid "
        "<b>else</b> hi = mid;&nbsp; <b>return</b> lo",
    ], "The encoder-side search as scripts/signalled_curve.py runs it, one "
       "line per call. A.12 gives the same search in words and explains why "
       "there are two levels; what this adds is the shape at every step and "
       "the two places the map is clamped. Lines 4 and 5 are the whole "
       "expense and they are outside both loops. Line 15 is the only "
       "arithmetic the multiplier touches.")

    from math import log2

    lam01 = {r["qp"]: r for r in sig["rows"]
             if abs(r["budget_db"] - 0.1) < 1e-9}
    qlo, qhi = min(lam01), max(lam01)

    k.par(
        f"The inner bisection of Table {box_search} needs a bracket, and the "
        "bracket is the unit interval while the answers sit far down inside "
        f"it: at the 0.1 dB budget the multiplier runs from "
        f"{_sci(lam01[qlo]['lam'])} at q{qlo} to {_sci(lam01[qhi]['lam'])} at "
        f"q{qhi}. The first {log2(1 / lam01[qhi]['lam']):.0f} halvings at "
        f"the highest rate, and {log2(1 / lam01[qlo]['lam']):.0f} at the "
        "lowest, are spent bringing the candidate down to the right order "
        "of magnitude before the search resolves anything inside it; the "
        "forty or so that remain are what locate the multiplier. The test at the top of the bracket comes first and returns it "
        "unchanged when the budget is still not met there, so a multiplier of "
        "exactly 1 in the results file is a flag rather than a measurement, "
        "and D.8 reads it that way.")

    k.par(
        "Both levels rely on the same monotonicity. Raising the multiplier "
        "makes compute expensive relative to error, so tiles fall to "
        "shallower exits, the saving rises and the distortion rises with it; "
        "neither turns back, so a bisection is valid on either quantity and "
        "the code bisects on distortion because that is what the budget is "
        "written in. The outer loop exists because the two distortions are "
        "not the same quantity. The table measures a tile with every other "
        "tile at the same exit, a routed frame is mixed, and a tile's border "
        "sees whatever depth its neighbour chose, so the delivered figure "
        "sits a little away from the table's. Shifting the inner target by "
        "the observed difference and re-bisecting closes it in two or three "
        "passes at one decode per frame per pass, against one decode per "
        "bisection step if the search were run on the delivered figure "
        "directly.")

    tab_cost = sum(e["measured"] for e in cm["exits"])
    dup = 2 * cm["exits"][0]["measured"]
    k.par(
        "What the table costs is settled twice over. Building it is one tiled "
        f"decode per exit, and a shallow decode is cheaper than a deep one, "
        f"so the arithmetic is the sum of the per-exit costs: "
        f"{tab_cost:.2f} released decodes, from the hook counts in "
        f"results/ceiling_measured.json. Timed on a frame rather than added "
        f"up, it is {ec['x_deployed']:.2f} decodes. Of that, "
        f"{dup:.2f} decodes are two duplicate columns: the clamp makes exits "
        f"0, 1 and {j} the same decode, and tiled_exit_mses runs the loop "
        f"over all K exits so that an argmin over all K columns lands on a "
        f"choice the decoder can carry out. The alternative table, taken from "
        f"a single full-frame pass with a tap at each exit, costs "
        f"{ec['x_full_frame']:.2f} decodes and must not be used to report "
        "quality, because it runs the trunk full-frame and the tiling penalty "
        "then cancels against a full-frame reference. It can still be used to "
        f"rank: on that frame the two searches agree on "
        f"{100 * ec['approx']['agreement']:.1f}% of tiles and land "
        f"{abs(ec['exact']['saving'] - ec['approx']['saving']):.2f} points of "
        "saving apart.")
    k.note(
        "Sources: results/ceiling_measured.json for the per-exit hook counts "
        f"(pinned checkpoint, q{cm['qp']}, {cm['size']}, {cm['n_tiles']} "
        "tiles); results/supp_encoder_cost_PAPER.json for the timings "
        f"(pinned checkpoint, {ec['seq'].split('_')[0]} at q{ec['qp']}, "
        f"{ec['iters']} iterations on an {ec['device']}, one tiled decode at "
        f"{ec['ms_one_decode']:.1f} ms); results/"
        "signalled_RECIPE512_ctc53.json for the multipliers, pinned "
        f"checkpoint, {sig['n_sequences']} sequences. The main paper "
        "quotes the same ranking comparison from "
        "results/encoder_cost.json, which is the identical measurement on "
        "an earlier checkpoint and reads a little differently; the pinned "
        "file is the one used here.")

    k.par(
        "Two lines of the box decide what the reported numbers mean. Line 15 "
        "runs the argmin against the arithmetic cost model, because a "
        "bisection evaluates it for every tile at every candidate multiplier "
        "and a decode per candidate is not affordable. Line 17 counts what "
        "the chosen decode actually performed, with forward hooks, and that "
        "is the figure every table in this paper quotes; B.6 prices the "
        "difference between the two and shows it is a closed form rather than "
        "a tolerance. A mistake inside the model therefore costs a slightly "
        "worse allocation and cannot inflate a headline.")

    k.par(
        "The clamp appears twice and is absent once, which matters where the "
        f"map is coded in {sec}.5. Line 17 clamps before measuring and "
        "trueDb clamps before decoding, so both are priced and measured at "
        "the exit that runs. Line 15 does not clamp, and argmin returns the first "
        "minimal index, so a tile whose cheapest choice is the shallowest "
        "reachable exit is recorded as exit 0. The price is still right, "
        "because the cost model gives exits 0, 1 and 2 the same value; the "
        "picture is still right, because the decoder clamps; the only "
        "quantity that sees six symbols where four exist is the entropy of "
        "the map.")

    # ------------------------------------------------------------------ G.3
    k.h2("Routing at the decoder")

    k.par(
        "Configuration B adds no bits, so the decision has to be made from "
        "what the decoder already holds, and everything it reads is already "
        "in hand at the moment the decision is needed. The stem is the "
        f"feature after group {j - 1}, which the decode has to run whatever "
        "the map says; the decoded latent and the entropy model's scales "
        "fall out of the pass that produced them; the quality index is one "
        "number per frame. Only the head's own convolutions "
        "are new arithmetic, and F.2 prices them at \\RouterCostPct% of a "
        "decode.")

    box_route = _box(k, [
        "<b>input</b> latent y, quant step q, scales, quality index qp, tilt "
        "beta",
        "&nbsp;1&nbsp; stem = dec.upsample(y)",
        "&nbsp;2&nbsp; <b>for</b> g = 0 … j−1: stem = dec.groups[g](stem)"
        + _c("already needed"),
        f"&nbsp;3&nbsp; z = head2(stem, y, scales, qp, {P}, "
        f"{td['latent_patch']})" + _c("[T, K]"),
        "&nbsp;4&nbsp; z = z[:, j:]" + _c("drop the exits that cannot run"),
        "&nbsp;5&nbsp; lp = log_softmax(z, 1)" + _c("[T, K−j]"),
        "&nbsp;6&nbsp; <b>optionally</b> lp = lp / mean(|lp|)"
        + _c("per frame"),
        "&nbsp;7&nbsp; k = (lp − beta · c[j:]).argmax(1) + j"
        + _c("[T], the map"),
        "&nbsp;8&nbsp; saving = measured_saving_pct(…, k) − 100 · rshare",
        "<b>and the budget is met by bisecting the tilt</b>",
        IND + "bisectTo(u): lo, hi = −2e4, 2e4; 60 times: mid = (lo + hi) / "
        "2;",
        IND + IND + "<b>if</b> tableDb(mid) ≤ u <b>then</b> lo = mid "
        "<b>else</b> hi = mid" + _c("Table " + str(box_search) + "'s, "
        "with k from line 7"),
        IND + "then the same outer correction as Table " + str(box_search),
    ], "Routing at the decoder, as flexuf/beta.py and "
       "scripts/router_curve.py run it. Take from it lines 4 and 7, which are "
       "one operation split in two: the logits are sliced down to the exits "
       "that exist, and the column index is put back on the exit axis "
       "afterwards. Line 8 charges the decoder for the head it ran, which "
       "configuration A never runs.")

    b_q0 = [r for r in hyb["rows"] if r["qp"] == 0][0]
    k.par(
        f"Line 4 of Table {box_route} is a slice where the natural spelling "
        "is a mask, and the reason is a defect that reached the numbers. The "
        "head suppresses the exits below the split by assigning them a large "
        "negative constant, which is a suppression only while the head's own "
        "logits stay well above that constant. Nothing in a cross-entropy or "
        "a regret objective penalises a common offset, one drifted in, and "
        "the raw outputs settled at the scale of the mask itself, at which "
        "point the masked entries were the largest in every row and the "
        "argmax selected an exit that does not exist. The clamp in the "
        "decoder then turned that into the shallowest real exit, for reasons "
        "having nothing to do with the tile. Every decision site now reads "
        "the sliced logits, the head files were left alone because live "
        "training runs import them, and A.6 lists the four places the same "
        "bound has to be applied.")

    k.par(
        "The tilt needs a wider bracket than the multiplier does. A confident "
        "head has log-probability gaps of hundreds, so a tilt of a few tens "
        "moves nothing at all, and the bracket runs to plus and minus "
        "2×10<super>4</super>; at the 0.1 dB budget and q0 the bisection "
        f"settles at {b_q0['beta']:.0f}.")
    k.par(
        "The outer correction against a real decode is the same as in the "
        "search above, and it lives in the same two functions. The reason is "
        "the calibration experiment: it has to run the identical bisection on "
        "held-out images and then apply its answer unchanged to the test "
        "frames. A bisection copied into a second "
        "caller is a second bisection, and this project has twice paid for a "
        "selection rule that was fixed in one copy and not the other.")

    k.par(
        "Line 6 is available to a decoder and is worth stating for that "
        "reason. Dividing a frame's log-probabilities by their own mean "
        "magnitude is a statistic of the decoder's own output, so it adds no "
        "bits and reads nothing from the source; what it buys is that one "
        "global tilt means the same thing on a frame the head is confident "
        "about and on a frame it is not. Line 8 subtracts the head's own "
        "arithmetic from the saving, measured by the same hook count applied "
        "to the head alone rather than assumed.")

    # ------------------------------------------------------------------ G.4
    k.h2("Choosing what to override")

    k.par(
        "Configuration C signals the exit of a fraction of the tiles and "
        "lets the decoder predict the rest, so it needs a rule for which "
        "tiles are worth the bits. The rule is the one the objective itself "
        "supplies. At a fixed multiplier the frame's Lagrangian cost is a sum "
        "of independent one-tile terms, so replacing the router's choice on a "
        "set of tiles removes exactly the sum of those tiles' regrets, and "
        "the best set of a given size is the set of largest regrets. The "
        "selection is therefore a topk, and the recovered fraction as a "
        "function of the signalled fraction is the Lorenz curve of the regret "
        "distribution, which is the object C.7 works with.")

    _box(k, [
        "<b>input</b> M [T, K], lp [T, K−j], tilt beta, multiplier lam, "
        "fraction rho",
        "&nbsp;1&nbsp; kr = (lp − beta · c[j:]).argmax(1) + j"
        + _c("what the decoder would do"),
        "&nbsp;2&nbsp; <b>if</b> rho ≤ 0: <b>return</b> kr, 0"
        + _c("exactly configuration B"),
        "&nbsp;3&nbsp; L = M + lam · c" + _c("[T, K]"),
        "&nbsp;4&nbsp; ko = L.argmin(1)" + _c("what the encoder would do"),
        "&nbsp;5&nbsp; <b>if</b> rho ≥ 1: <b>return</b> ko, T"
        + _c("exactly configuration A"),
        "&nbsp;6&nbsp; d = L.gather(1, kr) − L.gather(1, ko)"
        + _c("regret, [T], never negative"),
        "&nbsp;7&nbsp; S = d.topk(round(rho · T)).indices",
        "&nbsp;8&nbsp; k = kr.clone();&nbsp; k[S] = ko[S]",
        "&nbsp;9&nbsp; <b>return</b> k, |S|",
        "<b>outside the rule</b>",
        IND + "beta is bisected once, to what configuration B needs for the "
        "budget",
        IND + "lam is re-bisected at every rho, over the signalled tiles",
        IND + "bits = T · H<sub>2</sub>(rho) + 3 · |S|",
        IND + "the router's own cost is charged unless rho = 1",
    ], "The override rule, as scripts/hybrid_curve.py runs it. Take from it "
       "that the two endpoints are reached exactly rather than approached: at "
       "rho = 0 nothing but the router's own argmax is evaluated, and at "
       "rho = 1 the router's output is not consulted at all and its compute "
       "is refunded, so the columns between them measure an interpolation "
       "between two systems rather than three unrelated ones.")

    r05 = [r for r in hyb["rows"] if r["qp"] == 0 and abs(r["rho"] - 0.05)
           < 1e-9][0]
    k.par(
        "One multiplier for both branches was tried first and does not work, "
        "for a reason the two bisected values make plain. The router's term "
        "is a log-probability and the encoder's is a squared error, so tying "
        "them together needs a conversion constant, and the head is confident "
        "enough that the constant is of order 10<super>-6</super>: a shared "
        "multiplier would have to run to 10<super>8</super> before the router "
        "responded at all, and at that multiplier the oracle branch ignores "
        "distortion entirely and sends every signalled tile to the cheapest "
        f"exit. Measured at the 0.1 dB budget and q0, the tilt lands at "
        f"{b_q0['beta']:.0f} and the multiplier at "
        f"{_sci(r05['lam'])}. Keeping them separate also matches what each "
        "one is: the router is a fixed component whose operating point does "
        "not depend on how many tiles are signalled, and the multiplier is "
        "the only knob left to absorb the quality the overrides buy back.")
    k.note("results/hybrid_RECIPE512_b01_fixed.json, pinned checkpoint, "
           f"{hyb['n_sequences']} sequences at one frame each, with the "
           f"router head runs/RECIPE512/routers2/v2_lam1.3e-5.pth, itself "
           "trained against runs/RECIPE512/ckpt_eval.pth.tar and not against "
           "the pinned checkpoint. F.2 carries that provenance in full.")

    k.par(
        "Two properties of the rule are asserted on synthetic tables in "
        "tests/test_hybrid_map.py rather than argued: that the overridden set "
        "is exactly the tiles of largest regret, with the smallest overridden "
        "regret at least the largest untouched one, and that the objective is "
        "monotone in the signalled fraction. The endpoint checks are there "
        "too, at the level of the returned map rather than of the summary "
        "statistics, so a rho of 0 that merely happened to score like "
        "configuration B would not pass.")

    # ------------------------------------------------------------------ G.5
    k.h2("What the map costs to send")

    hist_bits = 8 * K
    mb_lo = lam01[qlo]["map_bits"]
    mb_hi = lam01[qhi]["map_bits"]
    tiles = hyb["rows"][0]["tiles_per_frame"]

    k.par(
        "The exit map is the one thing configuration A adds to the "
        "bitstream, and it is charged before any saving is computed. What is "
        "charged is an entropy and a fixed side cost, in four lines.")

    _box(k, [
        "<b>input</b> exit map k of length T, symbol count K",
        "&nbsp;1&nbsp; p = bincount(k, minlength = K) / T"
        + _c("the frame's own histogram"),
        "&nbsp;2&nbsp; H = − sum over symbols of p log<sub>2</sub> p"
        + _c("bits per tile"),
        "&nbsp;3&nbsp; bits = H · T + 8K"
        + _c("payload, then the histogram"),
        "&nbsp;4&nbsp; bpp = bpp + bits / pixels"
        + _c("before any saving is computed"),
        "<b>and for configuration C, which sends a different object</b>",
        IND + "bits = T · H<sub>2</sub>(rho) + 3 · |S|"
        + _c("the mask, then one symbol per override"),
    ], "The map coder, in scripts/signalled_curve.py and "
       "scripts/hybrid_curve.py. Take from it that line 3 charges the "
       "histogram at a flat byte a symbol rather than coding it, and that no "
       "arithmetic coder is run at any point: the payload is the entropy, "
       "which is what an ideal coder would spend and a real one a little "
       "more of.")

    k.par(
        f"At the 0.1 dB budget that comes to {mb_lo:.0f} bits a frame at "
        f"q{qlo} and {mb_hi:.0f} at q{qhi}, of which {hist_bits} bits are the "
        f"histogram in both cases; the payload itself is {mb_lo - hist_bits:.0f} "
        f"and {mb_hi - hist_bits:.0f} bits. Against a mean of {tiles:.1f} "
        f"tiles a frame over the same test set, that is "
        f"{(mb_lo - hist_bits) / tiles:.2f} and "
        f"{(mb_hi - hist_bits) / tiles:.2f} bits a tile, where a fixed-length "
        f"code over the {K - j} exits that exist would spend 2. The map grows "
        "with the rate because the allocation spreads out: at low rate most "
        "tiles take the same exit and the histogram is concentrated, at high "
        "rate the mass is spread over the ladder and the entropy rises. As a "
        f"share of the bitstream it runs to at most "
        f"{_sci(max(r['bpp_added'] for r in lam01.values()))} bpp, and every "
        "row of the results file records both the bits and the bpp they "
        "became.")
    k.note("results/signalled_RECIPE512_ctc53.json, pinned checkpoint, "
           f"{sig['n_sequences']} sequences, rows at the 0.1 dB budget. The "
           "tile count is the tiles_per_frame field of "
           "results/hybrid_RECIPE512_b01_fixed.json, measured on the same "
           "frames.")

    k.par(
        "Two further things make the reported cost an upper bound, and they "
        "are worth naming, because understating our own overhead would be the "
        "same class of error as overstating our saving. Line 1 counts the "
        "unclamped argmin, so the cheapest allocation is spread over the "
        f"three indices that share one cost and counted as three symbols "
        f"where a codec would signal one; the comment in signalled_curve.py "
        f"records the effect as at most {MERGE_SAVING[0]} bits a frame at "
        f"q{qlo} and {MERGE_SAVING[1]} at q{qhi}. And the three raw bits a "
        f"configuration C override pays would code one of {K - j} exits in "
        "two.")

    k.par(
        "Configuration C sends a mask saying which tiles were overridden "
        "and a symbol for each override, so its bill is the binary entropy "
        "of the overridden fraction over the tiles, plus the overrides "
        f"themselves. At q{qlo} and a signalled fraction of "
        f"{100 * r05['rho']:.0f}% that is {r05['map_bits']:.0f} bits a frame "
        f"for {r05['overridden_per_frame']:.1f} overrides, against nothing at "
        "all at a signalled fraction of zero, which is what makes that column "
        "exactly configuration B. The binary entropy is checked against its "
        "closed form in the test file rather than trusted.")

    # ------------------------------------------------------------------ G.6
    k.h2("One training step")

    k.par(
        "The training step is where the pinned checkpoint's behaviour is "
        "decided, and the shape of it is not the shape the objective is "
        "usually written in. A step draws one exit per tile uniformly from "
        "the reachable ones, decodes the batch once at that mixed depth, and "
        "measures the reconstruction against the source. The per-exit "
        "auxiliary losses are not evaluated within a step at all.")

    box_train = _box(k, [
        "<b>input</b> crop x [B, 3, 512, 512], quality index qp [B], "
        "multipliers lam [B]",
        "&nbsp;1&nbsp; lr = schedule[epoch + 99]; new modules at 20×"
        + _c("per epoch, A.11"),
        "&nbsp;2&nbsp; y, q, aux = encode(x, qp)" + _c("analysis, pass 1"),
        f"&nbsp;3&nbsp; m ~ uniform over j … K−1, shape [B · T]"
        + _c("fresh every step"),
        "&nbsp;4&nbsp; xhat = dec(y, q, exit_map = m)"
        + _c("ONE mixed-depth decode"),
        "&nbsp;5&nbsp; L = mean( lam · mse(x, xhat) ) + bpp(aux)",
        "&nbsp;6&nbsp; y2, q2 = encode(x, qp);&nbsp; ref = "
        "frozen.dec.forward_full(y2, q2)" + _c("pass 2, no grad"),
        "&nbsp;7&nbsp; y3, q3 = encode(x, qp);&nbsp; deep = "
        "dec.forward_full(y3, q3)" + _c("pass 3"),
        "&nbsp;8&nbsp; L = L + w<sub>a</sub> · mean(lam) · mse(deep, ref)",
        "&nbsp;9&nbsp; y4 = encode(x, qp);&nbsp; f = dec.exit_features(y4)"
        + _c("pass 4"),
        "10&nbsp; L = L + w<sub>d</sub> · distill(f)",
        "11&nbsp; (L / accum).backward()",
        "12&nbsp; <b>if</b> the window closes: norm = clip_grad_norm_(0.1);",
        IND + "<b>if</b> norm is not finite: skip the step "
        "<b>else</b> optimizer.step()",
        "13&nbsp; every 200 steps: one full-frame forward_all_exits, for the "
        "per-exit log",
    ], "One training step of the pinned run, as train_flexuf_image.py "
       "executes it. A.10 and A.11 give the recipe and the value of every "
       "constant; what this adds is the order, the four analysis passes and the two "
       "lines that are diagnostics rather than objective. Line 3 draws from "
       "the reachable exits only, which is the same bound the decoder's clamp "
       "enforces.")

    k.par(
        f"Line 4 of Table {box_train} is the choice that separates this "
        "recipe from a plain multi-exit one. Decoding every exit separately "
        "trains a condition that never occurs, because a deployed frame is "
        "mixed: a tile at exit 2 sits next to one at exit 5 and the head has "
        "to stitch across that discontinuity. One decode at a random mixed "
        "depth trains the condition that always occurs, and it costs one "
        "decode a step instead of K. The relation to the weighted objective "
        "is exact enough to state. With the auxiliary weights all equal, "
        "which is what the pinned run sets, the random draw makes the step's "
        "distortion term an unbiased single-sample estimate of the equally "
        "weighted mean of the per-exit errors. The difference from evaluating "
        "that mean directly is what the neighbours are doing: the estimator "
        "measures each exit with its neighbours at random depths, and the "
        "direct form measures it with every neighbour at the same depth. That "
        "difference is the reason for the design, and the main paper's "
        "training subsection measures what closing it is worth.")

    k.par(
        "Lines 2, 6, 7 and 9 are four analysis passes over the same batch. "
        "The quantiser is a rounding with a straight-through gradient and the "
        "noise the rate estimate uses is drawn separately from it, so the "
        "four passes compute the same latent, and with the encoder frozen "
        "none of them can diverge from the others. The model carries a single "
        "entry point that produces every term from one pass and exists so "
        "that DistributedDataParallel sees each parameter inside one forward "
        "call; the runs here are one GPU each, the trainer calls the pieces "
        "directly, and the redundancy is what that leaves behind. It costs "
        "wall clock and nothing else, and no file in results/ measures how "
        "much.")

    k.par(
        "Line 12 is the released recipe's guard, applied to the parameters "
        "that are actually training rather than to all of them: the norm is "
        "clipped at 0.1 and a batch whose norm is not finite is dropped "
        "instead of being allowed to reach the weights. Gradients accumulate "
        "over a window and the loss is divided by the window length, so the "
        "accumulated gradient equals what a single large batch would produce; "
        "the pinned run uses a window of one. Line 13 exists because line 4 "
        "returns a single mixed-depth reconstruction and therefore no "
        "per-exit breakdown, and that breakdown is the health signal for the "
        "failure this literature reports most often, a ladder whose exits "
        "converge on each other until only two branches are alive. One extra "
        "full-frame pass every 200 steps recovers it for the log at a "
        "fraction of a percent of training time.")

    k.par(
        "The joint path exists in the same file and is not what the pinned "
        "checkpoint trained under. It replaces the random draw with the "
        "router's own sample, taken through a Gumbel-softmax with a hard "
        "argmax, and multiplies the chosen tile's output by the selected "
        "probability divided by its own detached copy. That factor is "
        "numerically 1, so the reconstruction is bit-identical to the "
        "deployed path, while the gradient of the rate-distortion loss "
        "reaches the logits that an argmax would have blocked. A.7 describes "
        "the gate as it appears in the decode; the launcher for the pinned "
        "run does not pass the flag that switches it on.")

    # ------------------------------------------------------------------ G.7
    k.h2("The operations underneath")

    C = ac["trunk_channels"]

    k.par(
        "Below the five procedures sit perhaps a dozen operations that have "
        "to be exactly right and are silent when they are not. Table " +
        str(k.peek_tbl()) + " writes each one out with the file and the "
        "function it lives in. Shapes are given for the pinned geometry: T "
        "tiles, C = " + f"{C}" + " channels, a feature tile of " +
        f"{P}×{P}" + " and an RGB tile of " +
        f"{td['rgb_patch']}×{td['rgb_patch']}" + ".")

    rows = [["The operation", "File and function"]]
    rows += [
        ["<b>tiles and their order</b>", ""],
        ["patchify: x.view(B, C, nh, P, nw, P).permute(0, 2, 4, 1, 3, 5)."
         "reshape(B·nh·nw, C, P, P), with P the feature tile. No arithmetic, "
         "and it asserts divisibility rather than padding, so a ragged last "
         "row stops the run",
         "decoder.py, patchify"],
        ["unpatchify: x.view(B, nh, nw, C, P, P).permute(0, 3, 1, 4, 2, 5)."
         "reshape(B, C, nh·P, nw·P), the exact inverse",
         "decoder.py, unpatchify"],
        ["with a halo, in patchify_with_halo: replicate-pad by h, then "
         "unfold(2, P+2h, P).unfold(3, P+2h, P), permute(0, 2, 3, 1, 4, 5), "
         "reshape, contiguous",
         "decoder.py"],
        ["per-tile error: e = ((xhat − x)²).mean(1), then the same "
         "permutation at the RGB tile size, then mean over the tile's pixels",
         "eval.py, per_tile_mse"],
        ["per-tile router features: the same permutation at the feature tile "
         "size, then mean and standard deviation over the tile",
         "head2.py, _pool"],
        ["per-tile bits: the same permutation at the latent tile size, then a "
         "sum over the tile",
         "hybrid_curve.py, main"],
        ["<b>the exit map</b>", ""],
        ["the clamp: exit_map.to(device).long().clamp(min = j, max = K−1), "
         "before the per-tile loop and after the shape assertion. A.6 lists "
         "the four places that have to repeat it",
         "decoder.py, forward"],
        ["sorted execution: order = argsort(m, descending, stable); "
         "bounds = (m_sorted[None, :] &gt; arange(j, K)[:, None]).sum(1)."
         "tolist(); each group is then a slice, and the canvas is restored "
         "with out[inv]",
         "decoder.py, _suffix_sorted"],
        ["<b>the modules the ladder adds</b>", ""],
        ["adapter choice: the FFN when the exit skips (K−1−k)·b ≥ 4 blocks, "
         "the 1×1 otherwise, and none at the deepest exit, whose feature goes "
         "to the head raw",
         "decoder.py, build_adapter"],
        ["adapter form: f + Wf for the 1×1; f + PW(act(PW(f))) for the FFN, "
         "C→4C then C→C because the gated activation folds 4:1",
         "decoder.py, the two adapter classes"],
        ["adapter initialisation: the LAST convolution's weight and bias are "
         "zeroed, and zeroed again by _zero_init_adapters after the "
         "released model's own initialisation pass has run over the whole "
         "network",
         "decoder.py; model.py"],
        ["seam repair: x + G[r mod P, c mod P] · PW(WSiLU(DW3×3(x))), with G "
         f"a {P}×{P} gate initialised to exp(−d/tau), d the distance in "
         "feature pixels to the nearest tile edge, and PW zeroed",
         "decoder.py, GridSeamRepair"],
        ["the gate laid on the canvas: gate.repeat(1, 1, ceil(H/P), "
         "ceil(W/P))[:, :, :H, :W]",
         "decoder.py, the gate in forward"],
        ["tile padding: the 3×3 depthwise convolutions of groups j and "
         "deeper have their padding mode reassigned, or are wrapped when the "
         "scheme is not one of PyTorch's, and are restored afterwards",
         "decoder.py, _set_tile_padding; "
         "padding.py, wrap_tile_padding"],
        ["<b>the two ways of counting</b>", ""],
        ["the model: c_k assembled from the measured shares with "
         "k_run = max(k, j) applied to both the block count and the adapter",
         "cost.py, exit_costs"],
        ["the meter: a forward hook on every Conv2d and Linear, MACs from the "
         "output shape, output positions taken as numel divided by channels",
         "measure.py, MacMeter"],
        ["the ratio: routed MACs on our decoder over released MACs on the "
         "reference, both on the same latent, so no calibration constant "
         "enters",
         "measure.py, measured_cost"],
    ]
    t_ops = k.rows(
        rows,
        "The operations, with the file and the function for each, under the "
        "directories Table " + f"{t_map}" + " gives. Take from "
        "it the six rows in the first two groups: the same row-major tile "
        "ordering is written out four times, at four different grid "
        "resolutions, and the exit map is indexed by all four. Only the first "
        "pair has an inverse and a test asserting round-trip exactness; the "
        "other two agree with it by construction and by inspection.")
    k.note("Shapes and module forms are read from the files named in the "
           "second column. The adapter selection and the two adapter costs "
           "are confirmed against results/adapter_cost.json, which measures "
           "the modules of the pinned checkpoint with hooks: the 1×1 costs " +
           _n(ac["adapters"]["conv1x1"]["mac_per_px"]) + " MAC per feature "
           "pixel and the FFN " + _n(ac["adapters"]["ffn"]["mac_per_px"]) +
           ", against a trunk block's " + _n(ac["block"]["mac_per_px"]) +
           ". B.3 prices them as shares of a decode.")

    # ------------------------------------------------------------------ G.8
    k.h2("Where a natural implementation is wrong")

    k.par(
        "Six operations, most of them in Table " + f"{t_ops}" + ", are written "
        "the way they are because the obvious alternative fails, and each "
        "failure is "
        "silent in the sense that matters: it produces a decode that runs, "
        "returns a plausible picture and reports a plausible number.")

    k.par(
        "<b>The four tile orderings.</b> A tile's exit, its measured error, "
        "its router features and its bit count are computed on four different "
        "grids, at the RGB, feature and latent resolutions, and every one of "
        "them is indexed by the same integer. A permutation that disagreed "
        "with the others would deliver a perfectly valid decode of the wrong "
        "allocation, and the symptom would be a router that predicts poorly "
        "or an oracle that underperforms its own bound, neither of which "
        "points at a reshape. The pair that has an inverse is pinned by a "
        "round-trip test with zero tolerance. The other two are pinned only "
        "by the permutation being written identically, which is why the table "
        "above writes it out three times rather than saying that they agree.")

    k.par(
        "<b>The quantisation step and its batch.</b> The decoder multiplies "
        "the trunk output by a per-quality-index scale of shape [B, C, 1, 1] "
        "before the head. Handing a batch of those to a decode of one image "
        "broadcasts a [1, C, H, W] tensor back up to batch B, without an "
        "error and without a warning, and the wrong-shaped answer surfaces "
        "somewhere unrelated. It happened twice, in two files, because "
        "nothing tied the call sites together, and forward now raises when "
        "the scale's batch is neither 1 nor the latent's. It is also why one "
        "of the two training forwards decodes image by image with the scale "
        "sliced per image, while the other can decode the whole batch at once "
        "with a single exit map of length B·T.")

    k.par(
        "<b>Installing a padding scheme.</b> Two of the four schemes are not "
        "PyTorch padding modes, so they are installed by wrapping the "
        "convolution: pre-pad by hand, then convolve with no padding. Three "
        "things about that wrapping are deliberate. The convolutions to be "
        "replaced are collected before any of them is replaced, because "
        "assigning a wrapper during a walk of the module tree inserts it into "
        "the tree being walked and the walk descends into it without end. A "
        "second wrapping is refused rather than allowed, because a wrapper "
        "whose inner convolution is another wrapper fails inside the "
        "convolution call with a message about a missing weight, far from the "
        "line that caused it. And the trained per-channel coefficient is one "
        "tensor held on the decoder and attached to each wrapper without "
        "being registered, since registering it would hand the optimiser one "
        "copy of the same parameter per wrapped convolution. The swap is "
        "applied only to the groups that run per tile and undone afterwards, "
        "which is what keeps the full-frame path bit-exact against the "
        "released decoder and the reference honest.")

    k.par(
        "<b>The phase of the seam gate.</b> The repair pass is told where the "
        "seams are instead of having to infer them, by a gate indexed by "
        "position within a tile. That indexing is a modulo, and a modulo is "
        "only correct if the tile lattice starts at the origin. It does, "
        f"because the frame is padded on its bottom and right only: "
        f"1920×1080 becomes {r1080['padded'][0]}×{r1080['padded'][1]} with "
        f"{r1080['pad_right']} columns and {r1080['pad_bottom']} rows added "
        "at the far edges. Centring the padding, which is the natural thing "
        "to do to an image, would put every seam out of phase with the gate "
        "that was trained on it. The gate is also the one module in the "
        "ladder that is tied to the tile size: it has one scalar per position "
        "in the tile, so a checkpoint trained at one tile size cannot be "
        "evaluated at another without turning it off, and the evaluation "
        "script that allows a tile-size override says so at the point of the "
        "override.")

    k.par(
        "<b>The meter's batch dimension.</b> A convolution's MAC count is its "
        "output positions times its output channels times its taps times its "
        "input channels over its groups, and output positions here are the "
        "output's element count divided by its channel count rather than its "
        "height times its width. The per-tile trunk runs T tiles as T batch "
        "elements, so height times width would divide that part of the decode "
        f"by the tile count; the comment in flexuf/measure.py records the "
        f"deepest exit measuring {METER_BUG} of a released decode under that "
        "reading, against the "
        f"{cm['exits'][K - 1]['measured']:.3f} in "
        "results/ceiling_measured.json. The hooks are registered when the "
        "context is entered and removed when it is left, so nothing leaks "
        "into the next measurement, and a module an early exit skipped costs "
        "nothing because its hook never fires. There is no bookkeeping to get "
        "wrong, which is the property the arithmetic model does not have.")

    diffs = [e["measured"] - e["modelled"] for e in cm["exits"]]
    k.par(
        "<b>The clamp inside the cost model.</b> The decoder clamps the map, "
        "so a tile assigned an exit shallower than the split still runs the "
        "group at the split and wears that exit's adapter, and the model has "
        "to bill both the same way. A.6 gives the failure when it did not. "
        "What is left after the clamp is a residual with a closed form, "
        "measured per exit on the pinned checkpoint: the model under-bills by "
        f"{diffs[0]:.4f} of a released decode at exits 0 to 3, "
        f"{diffs[K - 2]:.4f} at exit {K - 2} and {diffs[K - 1]:.4f} at the "
        "deepest.")
    k.par(
        "That residual is why the reported saving is the meter's, and why the "
        "model's part is confined to the inside of an argmin. B.6 takes it "
        "apart term by term.")
    k.note("results/ceiling_measured.json, pinned checkpoint, "
           f"q{cm['qp']}, {cm['size']}, {cm['n_tiles']} tiles. Its modelled "
           "column is flexuf/cost.py and its measured column is "
           "flexuf/measure.py, on the same latent. The reading attributed to "
           "the meter's earlier form is the comment in that file and is in no "
           "results file, which G.10 repeats.")

    # ------------------------------------------------------------------ G.9
    k.h2("The controls that hold it in place")

    k.par(
        "Every property above that could fail silently has a test that fails "
        "loudly, under tests/. They are asserted at zero tolerance wherever "
        "the arithmetic permits it, on the principle that a control which "
        "cannot fail is not a control, and the one tolerance that appears is "
        "a property of a library rather than of the algorithm.")

    k.bullets([
        "<b>test_reference_is_the_release.py</b> decodes with Microsoft's own "
        "class and forward function and requires our deepest exit to match it "
        "with a largest absolute difference of zero, so the claim rests on "
        "the two models producing the same pixels rather than on the key "
        "remap having been audited correctly.",
        f"<b>test_equivalence.py</b> holds {_n_equiv()} properties at zero "
        "tolerance: "
        "the warm-started ladder at its deepest exit is the released decoder; "
        "every untrained exit is that decoder truncated at its own depth; "
        "patchify and unpatchify round-trip losslessly; with j = K the tiled "
        "path reproduces the full decode, which exercises the halo, the loop, "
        "the canvas, the crop and the head at once; every seam-repair variant "
        "is the identity at initialisation and the grid gate is concentrated "
        "on the border; a mixed-depth map costs less than an all-deepest one; "
        "and one trunk pass tapped at every exit gives the same pixels as K "
        "separate passes, which is what makes the joint objective of "
        "Eq. (6)-(7) affordable and is the property the from-scratch run "
        "depends on.",
        "<b>test_cost_matches_reality.py</b> runs the decode, counts its MACs "
        "with hooks and requires the arithmetic model to agree, which is the "
        "control that was missing when the model and the decoder drifted "
        "apart three times.",
        "<b>test_sorted_tiles.py</b> requires sorted execution to give the "
        "same picture as the masked loop, on a mixed exit map and on random "
        "content, because a uniform map makes the permutation the identity "
        "and a constant image makes any permutation exact.",
        "<b>test_hybrid_map.py</b> requires the override endpoints to be "
        "exactly configurations B and A at the level of the returned map, the "
        "overridden set to be the tiles of largest regret, the objective to "
        "be monotone in the signalled fraction, and the mask cost to be the "
        "binary entropy.",
        "<b>test_router_mask.py</b> requires the masked exits to be minus "
        "infinity, never the largest logit, and never selected by the tilted "
        "argmax at any tilt.",
        "<b>test_router_inputs.py</b> requires a zeroed input group to be "
        "unable to move a logit at all, and every ablation variant to have "
        "one architecture and one parameter count, so that the variants "
        "differ in information and in nothing else.",
    ])

    k.par(
        "What they are chosen to catch is never a quality regression. Each "
        "pins a property whose violation would leave every downstream number "
        "internally consistent and wrong, which is the class of defect this "
        "project has actually shipped. Two of them are careful about their "
        "fixtures for the same reason: the equivalence tests initialise the "
        "adapters randomly where a zero would make the property trivial, and "
        "the sorted-execution test decodes content rather than zeros, where "
        "every tile is identical and any permutation is exact.")

    k.note("The sorted-execution comparison is exact on CUDA and carries a "
           f"tolerance of {SORTED_TOL[1]} on CPU, where oneDNN selects a "
           "different blocking at some batch sizes and a plain DepthConvBlock "
           "is already order-dependent at 13 tiles while being exact at 4, 9, "
           "16 and 40. That tolerance is the library's and is recorded in "
           "tests/test_sorted_tiles.py; no results file carries it.")

    # ----------------------------------------------------------------- G.10
    k.h2("What this section does not settle")

    k.bullets([
        "<b>Nothing here is a quality measurement.</b> The numbers in this "
        "section price procedures and describe shapes. No quality figure is "
        "established here: the savings and the decibels are measured under "
        "the protocol A.4 fixes and reported in D and E.",
        "<b>The search timings are one sequence.</b> The cost of building "
        f"the per-tile table is timed on {ec['seq'].split('_')[0]} at "
        f"q{ec['qp']}, {ec['iters']} iterations on an {ec['device']}, and it "
        "is a ratio to a decode on the same card rather than a distribution "
        "over the test set.",
        "<b>The step's time breakdown is not measured.</b> The four analysis "
        "passes are counted from the source. No file in results/ records what "
        "each term of the objective costs in wall clock, and A.10 records "
        "only the total for a step.",
        "<b>Two implemented paths are not the pinned ones.</b> The jointly "
        "trained router and canvas coupling are both implemented and both "
        "measured elsewhere in this document; the "
        "pinned checkpoint's configuration block records coupling off, and "
        "the launcher does not pass the joint-router flag. Where a number in "
        "this paper comes from either, it says so.",
        "<b>No test asserts that the four tile orderings agree.</b> Only the "
        "pair with an inverse is checked. The other two are checked by "
        "reading them.",
        "<b>The map cost is an entropy, not a coder.</b> No arithmetic coder "
        "is run over the exit map; the reported bits are the entropy of the "
        "per-frame histogram plus a flat byte a symbol for the histogram "
        "itself, which bounds what a real coder would spend from below on the "
        "payload and from above on the side information.",
        "<b>Three numbers here are typed from source comments.</b> The "
        "meter's earlier reading of the deepest exit, the CPU tolerance of "
        "the sorted-execution test, and the bits that merging the three "
        "equal-cost indices would save. Each is attributed where it is used, "
        "and no file in results/ carries any of them.",
    ])
