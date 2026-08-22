"""The router and the calibrated bit rule.

Three things: which of the head's inputs carry anything, what the head costs to
run, and how it fares against a routing rule that has no learned parameters.
The head itself, its inputs, its shapes, its decision rule and its objective are
specified in the main paper's allocation subsection and are not restated here.

Every number in this section is computed here from a file in results/, and the
file is named at the table or figure that uses it. Nothing reads a checkpoint,
opens a CUDA context or writes to results/. The results files read are

    router_ablation_e4.json             the six-variant input ablation
    router_ablation_<variant>.json      one training record per variant
    router_latency.json                 the head timed against its MAC share
    router_RECIPE512_b0*_e4head.json    the frozen-decoder head, two budgets
    router_RECIPE512_b0*_jointhead.json the jointly trained head, same budgets
    raterank_RECIPE512_b01/b03/b05.json the rule, the same budgets
    raterank_BEST_compare.json          the same comparison on a second run
    router_retrain_compare.json         the head retrained with the mask fixed
    static_RECIPE512_b01.json           the exit cost vector, pinned
    saturation_RECIPE512_ctc53.json     c_j and the architectural ceiling
    supp_encoder_cost_PAPER.json        what the encoder-side search costs
    signalled_RECIPE512_ctc53.json      deployed configuration A, for contrast

The two figures are drawn by scripts/supp_router_figs.py, which reads the same
files and touches nothing else.

paper/supp/f_router.tex carries the same section as LaTeX, with the same
numbers in the same order.
"""

#: The order the ablation is read in: the two heads that tie first, then the
#: single-group runs by descending agreement, then the no-per-tile control.
VARIANT_ORDER = ["stem", "all", "latent", "scales", "bits", "qp"]

def _joint_head_file(k):
    """The naming the jointly trained head's curves are stored under.

    These files were first written as router_RECIPE512_b0*_PAPER.json and then
    renamed to _jointhead, which says what they are rather than which
    checkpoint they are on. Both names are accepted so that this section builds
    on either side of that rename, and the one that exists is what the notes
    print.
    """
    for pat in ("router_RECIPE512_b0{b}_jointhead.json",
                "router_RECIPE512_b0{b}_PAPER.json"):
        if (k.RESULTS / pat.format(b=1)).exists():
            return pat
    raise FileNotFoundError(
        "f_router: no curve for the jointly trained head; expected "
        "results/router_RECIPE512_b01_jointhead.json")

def content(k):
    A = k.J("router_ablation_e4.json", "router_ablation.json")
    V = {x["label"]: x for x in A["variants"]}
    floor = V["stem"]["constant_best_agree"]
    # The per-variant records carry the training recipe and the wall-clock that
    # the merged file does not. Reading all six also puts them in the build's
    # provenance list, which is what the ablation table's note claims.
    REC = {l: k.J(f"router_ablation_{l}_e4.json", f"router_ablation_{l}.json")
           for l in VARIANT_ORDER}
    # Each variant falls back to the record from before the repin on its own,
    # so a sweep that is half re-measured would put six rows from two
    # checkpoints in one table and nothing would say so. The comparison is
    # between variants, so a mixture is not a small error: it is the whole
    # measurement.
    _eps = {(r.get("ckpt"), r.get("ckpt_epoch")) for r in REC.values()}
    _mixed = sorted(f"{l} on epoch {REC[l].get('ckpt_epoch')}"
                    for l in VARIANT_ORDER) if len(_eps) > 1 else []
    RL = k.J("router_latency.json")
    ST = k.J("static_RECIPE512_b01.json")
    SAT = k.J("saturation_RECIPE512_ctc53.json")
    EC = k.J("supp_encoder_cost_PAPER.json")
    rr = {b: k.J(f"raterank_RECIPE512_b0{b}.json") for b in (1, 3, 5)}
    # Two trained heads exist on the pinned checkpoint and they are different
    # objects, so both are read and both are reported. hj is the head the
    # checkpoint carries, trained jointly with the decoder; hf is the head
    # trained afterwards against the frozen decoder.
    HJ = _joint_head_file(k)
    hj = {b: k.J(HJ.format(b=b)) for b in (1, 3)}
    # The frozen-decoder head fitted to the pinned weights, which is the one
    # the paper reports, falling back to the head fitted on 19 August for a
    # build that runs before the refit.
    hf = {b: k.J(f"router_RECIPE512_b0{b}_e4head.json",
                 f"router_RECIPE512_b0{b}.json") for b in (1, 3)}
    QPS = [r["qp"] for r in rr[1]["rows"]]
    cost = {u["exit"]: 1.0 - u["saving"] / 100.0
            for u in ST["rows"][0]["uniform"]}
    block = (cost[3] - cost[2]) / 2.0
    st = REC["stem"]
    abl_hours = sum(REC[l]["wall_s"] for l in VARIANT_ORDER) / 3600.0

    def at(d, q):
        return [x for x in d["rows"] if x["qp"] == q][0]["saving_pct_vs_release"]

    def rule(b, q):
        return at(rr[b], q)

    k.h1("The router and the calibrated bit rule")

    k.par(
        "An exit map assigns one exit index to every tile of a frame, and this "
        "work produces one in three ways. Configuration A searches all K exits "
        "for every tile at the encoder, which holds the source, and signals the "
        "answer. Configuration B predicts the map at the decoder from data the "
        "decoder already holds, and signals nothing. The third way computes it "
        "at the decoder from one number the entropy coder has already produced, "
        "with nothing learned. The main paper states the rule by which the "
        "head turns logits into an exit and where its tilt comes from, and "
        "that is not repeated. What is here is the head itself, what each of "
        "its inputs is worth, what it costs to run, and how it fares against "
        "the third way.")

    k.par(
        "Write D(t,k) for the mean squared error of tile t decoded at exit k "
        "and c<sub>k</sub> for the cost of exit k in units of one released "
        "decode. At a price \\lambda on compute, the Lagrangian oracle gives "
        "each tile the exit that minimises")
    k.eq(r"\ell(t,k) \;=\; D(t,k) \,+\, \lambda\, c_k, \qquad "
         r"k^{*}(t) \;=\; \mathrm{arg\,min}_{\,k \geq j}\ \ell(t,k).")
    k.par(
        "The split depth j is the number of trunk groups every tile runs before "
        "any tile may leave, so exits below j do not exist and the minimisation "
        "starts there. Throughout this section, <i>agreement</i> means the "
        "fraction of tiles on which a predictor picks the same exit as this "
        "oracle at the same \\lambda. It is the quantity the head is trained "
        "on. Part of what F.4 reports is that it is not the quantity that "
        "decides which allocation is cheaper.")

    # ------------------------------------------------------------------ F.1
    k.h2("What each input is worth")
    k.par(
        "The head reads four per-tile signals and one per-frame signal. Since "
        "F.4 finds that a rule reading a single number beats it, it is fair to "
        "ask which of the five carries anything. The ablation trains the same "
        "head once per input group, with every other group zeroed after its "
        "projection rather than deleted. Architecture, parameter count, "
        "optimiser, seed, image order and quality draws are then identical "
        "across the six runs, and the only thing that differs is how much the "
        "head is allowed to know. All six carry "
        f"{st['router_params']:,} parameters, which is the deployed head's "
        f"{RL['params']:,} plus the projection for the bit count that the "
        "deployed head does not use.")
    k.par(
        "The quality index q stays live in every single-group run. It is one "
        "number per frame, identical for every tile in that frame, so it can "
        "say which operating point the decoder is at and cannot, even in "
        "principle, tell two tiles of one frame apart. Holding it live keeps "
        "the runs comparable on per-tile information alone. The run with q on "
        "its own is then the floor that choice implies, which is the agreement "
        "reachable with no per-tile information whatever.")

    _abl = ([["live inputs", "agreement", "± s.e.", "last 250", "over floor"]]
            + [[V[l]["inputs"], f"{V[l]['agree']:.4f}",
                f"{V[l]['stderr']:.4f}",
                f"{V[l]['heldout_agree_last250']:.4f}",
                f"{V[l]['agree'] - floor:+.4f}"] for l in VARIANT_ORDER])
    import json as _j3
    from pathlib import Path as _P3
    _j3.dump({"rows": _abl},
             open(_P3(__file__).resolve().parents[2] / "results/router_inputs_rows.json",
                  "w"), indent=2)

    k.rows(
        _abl,
        "<b>What each router input is worth.</b> Read the first column against "
        "the last: only the stem moves agreement far from what a single "
        "constant exit already achieves. Agreement with the oracle's exit "
        f"choice on {V['stem']['n_frames']:,} frames and "
        f"{V['stem']['n_tiles']:,} tiles the head never trained on, with one "
        "standard error taken across frames rather than across tiles, since "
        "tiles of one image are not independent draws. <i>last 250</i> is the "
        "running held-out agreement over the final 250 training steps, on 12 "
        "tiles a step, and confirms that the end-of-run measurement is not a "
        "fluctuation. <i>over floor</i> is the margin over the best single "
        f"constant exit, which scores {floor:.4f} on these tiles.")
    k.note(
        "results/router_ablation_e4.json, and the six per-variant training records "
        "results/router_ablation_stem.json and its five siblings. All on the "
        "pinned checkpoint, all at \\lambda = 1.3×10<super>-5</super>, all "
        f"{st['steps']:,} steps from seed {st['seed']}.")

    k.fig("supp_router_inputs.png",
          "<b>The input ablation, both halves of it.</b> <b>a</b>, agreement "
          "with the oracle per variant, with one standard error across frames; "
          "the dashed line is the best single constant exit. The two dark bars "
          "are the two heads that see the stem. <b>b</b>, where each variant "
          "sends its tiles, against the oracle's own distribution. Every "
          "variant reproduces the oracle's use of the shallowest exit to within "
          "a point, and what degrades as inputs are removed is the separation "
          "of the middle of the ladder: the bit count alone nearly empties exit "
          "3, and q alone uses two of the four exits available to it.",
          maxh=126)
    k.note("Drawn by scripts/supp_router_figs.py from "
           "results/router_ablation_e4.json, fields agree, stderr, pred_hist and "
           "oracle_hist.")

    d_sa = V["stem"]["agree"] - V["all"]["agree"]
    se_sa = (V["stem"]["stderr"] ** 2 + V["all"]["stderr"] ** 2) ** 0.5
    k.par(
        f"<b>The stem alone.</b> Reading the stem and q gives "
        f"{V['stem']['agree']:.4f}; reading the stem, the latent, the scales, "
        f"the bit count and q gives {V['all']['agree']:.4f}. The difference is "
        f"{d_sa:.4f} against a standard error of {se_sa:.4f} on the difference, "
        "so it is under a third of one standard error and its sign is not "
        "determined. Four of the five inputs add nothing this measurement can "
        "resolve. That result is about those four inputs in this head, and it "
        "does not show that the latent, the scales or the bit count are "
        "uninformative about a tile. What it shows is that whatever they carry "
        "is already carried by the stem, which is computed downstream of all of "
        "them.")
    k.par(
        "<b>The single-group ordering.</b> The stem at "
        f"{V['stem']['agree']:.4f} is "
        f"{V['stem']['agree'] - V['latent']['agree']:.4f} above the latent, "
        "against a standard error of "
        f"{(V['stem']['stderr']**2 + V['latent']['stderr']**2)**0.5:.4f} on "
        f"the difference; the latent is "
        f"{V['latent']['agree'] - V['scales']['agree']:.4f} above the scales; "
        "and the scales are "
        f"{V['scales']['agree'] - V['bits']['agree']:.4f} above the bit count, "
        "which is more than seven standard errors. The bit count is the "
        "weakest per-tile input the head has, and F.4 shows a rule that reads "
        "the bit count and nothing else beating the head at every rate. The "
        "input the head learns least from is the input that wins when it is "
        "used differently.")
    k.par(
        f"<b>The floor.</b> The quality index on its own reaches "
        f"{V['qp']['agree']:.4f} ± {V['qp']['stderr']:.4f} where the best "
        f"single constant exit scores {floor:.4f}. A head with no per-tile "
        "information cannot beat the best constant in expectation, and this one "
        f"does not. The floor belongs beside every other row: an agreement of "
        f"{V['bits']['agree']:.2f} cannot be read at all until it is known "
        f"that {floor:.2f} is free.")
    if _mixed:
        k.par(
            "<b>These six rows are not on one checkpoint.</b> " +
            ", ".join(_mixed) + ". The comparison between them is only a "
            "comparison of inputs when the weights are the same, so this "
            "table should not be read until the sweep has been repeated on "
            "one checkpoint throughout.")
    k.par(
        f"<b>The ablation's own cost.</b> The six trainings took "
        f"{abl_hours:.1f} hours of one NVIDIA RTX A6000 between them, from "
        f"{min(REC[l]['wall_s'] for l in VARIANT_ORDER) / 60:.0f} to "
        f"{max(REC[l]['wall_s'] for l in VARIANT_ORDER) / 60:.0f} minutes each "
        "(the ablation's own records, field wall_s). The decoder is frozen "
        "for all of them, so none of that is decoder training. It is the price "
        "of the question, and it is worth stating beside a negative answer.")

    # ------------------------------------------------------------------ F.2
    k.par(
        "One head covers every operating point, and the quality index is what "
        "makes that possible. It is an input rather than a selector: during "
        "training it is drawn uniformly at random for each batch, so one set "
        "of weights sees all 64 indices, and at test time the same weights "
        "run at every rate with the offline \u03b2 table of A.12 in front of "
        "them. No head per rate is proposed anywhere in this work, since that "
        "multiplies the stored parameters by the number of operating points a "
        "deployment offers, and F.5 is the measurement of what the single "
        "head gives up for that.")

    k.h2("What the head costs")
    k.rows(
        [["q", "decode (ms)", "stem (ms)", "router (ms)", "share (%)"]]
        + [[str(r["qp"]), f"{r['decode_ms']:.2f}", f"{r['stem_ms']:.2f}",
            f"{r['router_ms']:.3f}", f"{r['router_share_pct_time']:.3f}"]
           for r in RL["rows"]],
        "<b>The head against the decode it decides for.</b> The share column is "
        "the number to take away, and it is three times what the operation "
        "count predicts. Three timings on one latent, interleaved so that a "
        "co-tenant's load drift lands on all of them equally. <i>decode</i> is "
        "the tiled decoder at the deepest exit, which is what the head's share "
        "is a share of; <i>stem</i> is the first j groups, which the head does "
        "not pay for because the decoder runs them anyway before the split. The "
        "head takes \\RouterTimePct\\% of the decode where its operation count "
        "predicts \\RouterCostPct\\%, a factor of \\RouterTimeFactor.")
    k.note(
        "results/router_latency.json: 1920×1080 padded to 2048×1280, "
        f"{RL['iters']} iterations after warm-up, one {RL['device']}, the "
        f"{RL['params']:,} parameter head "
        "runs/RECIPE512/routers2/v2_lam1.3e-5.pth. Measured on "
        "runs/RECIPE512/ckpt_eval.pth.tar rather than on the pinned "
        "checkpoint; the quantity is a ratio of two timings of the same "
        "weights, so it does not depend on which epoch they came from.")
    k.par(
        "The extra \\RouterTimeExtra points are launch overhead. A head of "
        f"{RL['params']:,} parameters evaluated on 40 vectors is small enough "
        "that its wall-clock is dominated by the cost of starting its kernels, "
        "and an operation count does not contain that. Every configuration-B "
        "saving in this work charges the head at its operation share, which is "
        "the smaller of the two numbers and therefore the more flattering to "
        "us; the honest reading is that the head costs about half a percent of "
        "the decode in time. It stays small against what it decides about, "
        f"since one trunk block is {100 * block:.1f}% of the decode by the cost "
        "vector below.")
    k.par(
        "<b>Two heads, two shares.</b> The pinned checkpoint carries a head "
        "that was trained jointly with the decoder, and a second head was "
        "trained afterwards against that decoder once it was frozen. They are "
        "different objects and they are priced differently: the jointly trained "
        "one is charged "
        f"{hj[1]['router_compute_share_pct']:.3f}% of the decode and the "
        "frozen-decoder one \\RouterCostPct\\%, because the second reads the "
        "decoded latent and the entropy model's scales as well as the stem. "
        "Both shares are measured by hooks on the head's own convolutions and "
        "linear layers rather than assumed, in the same run that produced the "
        "saving. F.4 reports both heads.")
    k.note(
        f"results/{HJ.format(b=1)} and "
        "results/router_RECIPE512_b01_e4head.json, field "
        "router_compute_share_pct in each.")
    k.par(
        "<b>For scale, what the search costs instead.</b> Configuration A does "
        "not run a head; it runs the argmin itself, which needs the true error "
        "of every exit, and section G prices that at "
        f"{EC['x_deployed']:.2f} of a tiled decode per frame. A decoder-side "
        "head at half a percent of one decode, and a rule at nothing, are both "
        "several orders below it, which is the whole reason for wanting one.")

    # ------------------------------------------------------------------ F.3
    k.par(
        "<b>The head, exactly.</b> Three things enter. The first is the "
        "feature at the split point, 384 channels at one eighth of frame "
        "resolution, which is the last thing every tile shares. The second is "
        "the decoded latent concatenated with the entropy model's scales, the "
        "predicted Gaussian width used to code each latent position, which is "
        "already computed during the decode and is, position by position, an "
        "estimate of how hard that position was to code. The third is the "
        "quality index. Each of the first two is projected by a 1\u00d71 "
        "convolution, to 48 and 32 channels, then reduced to one vector per "
        "tile by taking the mean and the standard deviation over the tile. "
        "That gives 96 + 64 + 1 = 161 numbers per tile, which pass through a "
        "LayerNorm and a three-layer perceptron of width 256 with SiLU "
        "activations, ending in K logits. Almost all of the "
        f"{RL['params']:,} parameters sit in the 1\u00d71 on the stem, which "
        "is the only part that runs per pixel; the perceptron runs once per "
        "tile, forty times for a 1080p frame, so its width is nearly free.")

    k.fig("router_arch.png",
          "<b>The router head.</b> <b>a</b>, the three things it reads, all "
          "already held by the decoder at the moment the decision is needed. "
          "<b>b</b>, each projected by a 1\u00d71, then reduced to one vector "
          "per tile by its mean and standard deviation. <b>c</b>, a "
          "three-layer perceptron, one pass per tile. Every shape and "
          "parameter count in the figure is read off the module when the "
          "figure is drawn.")

    k.par(
        "<b>What it is trained against.</b> The head is fitted against a "
        "frozen decoder, so the exits it chooses between do not move while it "
        "learns. Its target is the Lagrangian oracle's choice and the loss is "
        "a cross-entropy to that target, with each tile weighted by its "
        "<i>regret</i>, meaning by how much the oracle's own bracket grows if "
        "the head's exit is used in place of the oracle's. A tile whose two "
        "best exits are nearly tied then counts for less than one where the "
        "wrong choice is expensive, which is the same asymmetry that makes "
        "agreement a poor score in F.4. A router of this shape can collapse "
        "onto a single exit during training, so it carries a balancing term: "
        "a per-exit bias added to the logits before the decision and updated "
        "by usage rather than by a gradient, which is what makes it "
        "loss-free [14]. It is biased toward the oracle's own exit "
        "distribution rather than toward uniform, because at a high "
        "multiplier the oracle genuinely does send every tile to one exit and "
        "forcing spread there would force mistakes.")

    k.h2("The calibrated bit rule")
    k.par(
        "The entropy model produces one number per tile before the trunk runs, "
        "at no cost, and then discards it: how many bits that tile's latents "
        "took. Per-block bit allocation is a standard quantity in learned "
        "compression, where it is something to choose, and block-level rate "
        "control sets it so that complex regions get more bits [40]. Here it is "
        "read in the other direction, after the fact and at the decoder, as a "
        "statement about how hard the region was. The results files call the "
        "rule <i>rate-rank</i>; the paper calls it the calibrated bit rule. "
        "What follows is all of it.")
    k.par(
        "<b>The statistic.</b> Sum the entropy coder's estimated bits "
        "r<sub>i</sub> over the latent positions i falling inside tile t, and "
        "divide by the mean over the N tiles of that frame. Normalising per "
        "frame rather than globally is deliberate: the absolute rate level is a "
        "property of the frame, and what decides a tile is how it compares with "
        "the rest of its own frame.")
    k.eq(r"b(t) \;=\; N\, \frac{\sum_{i \in t} r_i}{\sum_{i} r_i}.")
    k.par(
        "<b>The surrogate.</b> Assume every tile has the same shape of decay "
        "across the ladder, up to a scale that depends only on b(t).")
    k.eq(r"\log D(t,k) \;\approx\; \alpha \log b(t) \,+\, c \,+\, "
         r"\log \varphi_k.")
    k.par(
        "φ has one entry per reachable exit and says what that exit "
        "costs on an average tile; \\alpha is an exponent fitted rather than "
        "assumed, which is what lets the rule find the direction of the "
        "relation instead of asserting one. The fit is offline and "
        "leave-one-sequence-out, so no sequence contributes to the profile "
        "that routes it, and its whole output is six numbers per rate.")
    k.par(
        "<b>The decision.</b> Run the oracle's own Lagrangian on the surrogate "
        "table in place of the true one, and bisect \\lambda against the budget "
        "as configuration A does.")
    k.eq(r"\hat{k}(t) \;=\; \mathrm{arg\,min}_{\,k \geq j}\ "
         r"\left[\, e^{c}\, b(t)^{\alpha}\, \varphi_k \,+\, \lambda\, c_k "
         r"\,\right].")

    box = [
        ["<b>the decoder, once per frame</b>"],
        ["&nbsp;1&nbsp; <i>input</i> r, the entropy coder's estimated bits per "
         "latent position; the cost vector c; the price \\lambda; the "
         "calibration (\\alpha, c<sub>0</sub>, φ) for this quality "
         "index"],
        ["&nbsp;2&nbsp; <i>for</i> each tile t of the N tiles of the frame"],
        ["&nbsp;3&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; b ← N · (sum of r<sub>i</sub> inside t) / "
         "(sum of r<sub>i</sub> over the frame)"],
        ["&nbsp;4&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; <i>for</i> k = j … K−1"],
        ["&nbsp;5&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; L<sub>k</sub> ← "
         "e<super>c<sub>0</sub></super> · b<super>\\alpha</super> · "
         "φ<sub>k</sub> + \\lambda · c<sub>k</sub>"],
        ["&nbsp;6&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; exit(t) ← the k that minimises "
         "L<sub>k</sub>"],
        ["&nbsp;7&nbsp; <i>return</i> exit"],
        ["<b>offline, once per quality index, leave-one-sequence-out</b>"],
        ["&nbsp;8&nbsp; d(t) ← mean over k ≥ j of log D(t,k)"],
        ["&nbsp;9&nbsp; (\\alpha, c<sub>0</sub>) ← least squares of d(t) on "
         "log b(t)"],
        ["10&nbsp; log φ<sub>k</sub> ← mean over t of [ log D(t,k) − "
         "\\alpha log b(t) − c<sub>0</sub> ]"],
    ]
    k.rows(box,
           "<b>The calibrated bit rule, in full.</b> Lines 1 to 7 are what the "
           "decoder runs: one power and K products per tile, nothing learned, "
           "nothing signalled. Lines 8 to 10 are the offline calibration, whose "
           "entire output is the six numbers per rate printed in the next "
           "table. \\lambda is bisected in two levels, on the cheap per-tile "
           "table first and then against a real decode of the resulting map, "
           "because the table and the decode differ slightly at the tile seams. "
           "Sweeping \\lambda traces the lower convex hull of the achievable "
           "set in the sense of [26], so the point returned is the best one on "
           "that hull at the budget it consumes.",
           header=False)

    phi = [["q", "delivered dB", "\\alpha", "φ<sub>2</sub>", "φ<sub>3</sub>",
            "φ<sub>4</sub>", "φ<sub>5</sub>"]]
    for r in rr[1]["rows"]:
        phi.append([str(r["qp"]), f"{r['db_vs_uf']:.4f}", f"{r['alpha']:.2f}"]
                   + [f"{p:.4f}" for p in r["phi"]])
    sp0 = 100 * (rr[1]["rows"][0]["phi"][0] / rr[1]["rows"][0]["phi"][-1] - 1)
    sp63 = 100 * (rr[1]["rows"][-1]["phi"][0] / rr[1]["rows"][-1]["phi"][-1] - 1)
    k.rows(phi,
           "<b>The rule's entire state</b>, at the 0.1 dB budget. There is "
           "nothing else to store, and it is printed here so that the rule can "
           "be reproduced without refitting it. \\alpha is positive everywhere, "
           "so a tile whose latents cost more bits is modelled as harder at "
           "every exit and is sent deeper, which is what the correlations in "
           f"F.4 confirm. φ is nearly flat, spanning {sp0:.1f}% at q0 "
           f"and {sp63:.1f}% at q63 between its shallowest and deepest entry, "
           f"against costs that span {cost[2]:.2f} to {cost[5]:.2f}, so what "
           "moves a tile along the ladder is its own b(t) rather than the shape "
           "of the profile. <i>delivered dB</i> is what the two-level bisection "
           "actually landed on.")
    k.note("results/raterank_RECIPE512_b01.json, pinned checkpoint, \\NumSeq "
           "CTC sequences at one frame each.")

    k.par(
        "The costs c<sub>k</sub> on the pinned checkpoint are "
        + ", ".join(f"{cost[e]:.4f}" for e in sorted(cost))
        + " for exits 2 to 5, recovered from the frame-level saving of maps "
        "that send every tile to one fixed exit. The shallowest of them, "
        f"c_j = {SAT['cost_j']:.4f}, sets the architectural ceiling of "
        "\\CeilingModelled\\%. The deepest is above 1 because our ladder carries "
        "seam repair and an adapter that the released decoder does not.")
    k.note("results/static_RECIPE512_b01.json, field uniform, and "
           "results/saturation_RECIPE512_ctc53.json, fields cost_j and "
           "ceiling_pct.")

    # ------------------------------------------------------------------ F.4
    k.h2("The rule against the trained head")
    k.par(
        "Two trained heads exist on this checkpoint and both are reported, "
        "since averaging them would hide the only spread this work has. "
        "<i>joint</i> is the head the checkpoint carries, trained alongside the "
        "decoder and read straight out of it. <i>frozen</i> is the "
        "\\RouterParams head trained afterwards, against the decoder once it "
        "had stopped moving. Both were evaluated on the pinned checkpoint over "
        "\\NumSeq sequences at one frame each, by one script, at the same "
        "budgets, and both are charged for their own compute.")

    cmp_rows = [["q", "0.1 rule", "0.1 joint", "0.1 frozen",
                 "0.3 rule", "0.3 joint", "0.3 frozen"]]
    for q in QPS:
        cmp_rows.append(
            [str(q)]
            + [f"{rule(1, q):.2f}", f"{at(hj[1], q):.2f}", f"{at(hf[1], q):.2f}",
               f"{rule(3, q):.2f}", f"{at(hj[3], q):.2f}", f"{at(hf[3], q):.2f}"])
    # Whether the rule wins outright is a fact about the files, not a claim to
    # be typed: count it, and let the caption say the weaker thing if it must.
    n_cells = 2 * len(QPS)
    n_win = sum(1 for b in (1, 3) for q in QPS
                if rule(b, q) > at(hj[b], q) and rule(b, q) > at(hf[b], q))
    verdict = ("The rule is ahead of both heads in every cell of this table."
               if n_win == n_cells else
               f"The rule is ahead of both heads in {n_win} of the "
               f"{n_cells} cells.")
    k.rows(cmp_rows,
           f"<b>The calibrated bit rule against two trained heads</b>, as the "
           "percentage of the released decoder's operations saved, at two "
           f"budgets and five rates. {verdict} Every column is charged for what "
           "it costs: each head pays its own compute share and the rule pays "
           "nothing. The two heads are further apart from each other than "
           "either is from the rule at several rates, which F.5 takes up.")
    k.note(
        "Rule: results/raterank_RECIPE512_b01.json and b03. Joint head: "
        f"results/{HJ.format(b=1)} and its b03 sibling. Frozen head: "
        "results/router_RECIPE512_b01_e4head.json and b03. All on "
        "runs/RECIPE512/ckpt_PAPER.pth.tar.")

    k.fig("supp_router_frontier.png",
          "<b>The rule sits between the two heads and the oracle.</b> Saving at "
          "each rate, against the Lagrangian oracle measured inside the same "
          "run and the architectural ceiling (dotted). At 0.1 dB the rule is "
          "above both heads at every rate, and the gap between the two heads is "
          "wider than the gap between the rule and either of them. At 0.3 dB "
          "the three lowest rates saturate and the three curves separate only "
          "at q48 and q63.",
          maxh=130)
    k.note("Drawn by scripts/supp_router_figs.py from the six files named in "
           "the table above and results/saturation_RECIPE512_ctc53.json.")

    mj = [rule(1, q) - at(hj[1], q) for q in QPS]
    mf = [rule(1, q) - at(hf[1], q) for q in QPS]
    k.par(
        "<b>At the working budget.</b> Against the joint head the rule is ahead "
        f"by {min(mj):+.2f} points at its narrowest and {max(mj):+.2f} at its "
        f"widest, the latter at q{QPS[mj.index(max(mj))]}. Against the frozen "
        f"head it is ahead by {min(mf):+.2f} to {max(mf):+.2f} points. The rule "
        "reaches \\RateRankLow% at q0 and \\RateRankHigh% at q63, and it is "
        "ahead at all \\RateRankNWins measured rates whichever head it is "
        "measured against. A head trained on this decoder against this oracle "
        "returns nothing over a rule with no learned parameters, while "
        "carrying a compute share that the rule does not.")

    sat_j = SAT["ceiling_pct"] - at(hj[3], 0)
    sat_f = SAT["ceiling_pct"] - at(hf[3], 0)
    k.par(
        "<b>At 0.3 dB the low rates saturate and the arithmetic is exact.</b> "
        "At q0, q16 and q32 all three configurations send every tile to the "
        "shallowest exit, so the maps are identical and the only difference "
        "left is the price of deciding. The joint head falls "
        f"{sat_j:.3f} points below the \\CeilingModelled% ceiling and the frozen "
        f"head {sat_f:.3f}, which are their compute shares of "
        f"{hj[1]['router_compute_share_pct']:.3f}% and "
        f"{hf[1]['router_compute_share_pct']:.3f}% to three decimals. At q48 "
        "and q63 the budget still binds and the allocations separate: the rule "
        f"reaches {rule(3, 48):.1f}% and \\RateRankLoose% against the joint "
        f"head's {at(hj[3], 48):.1f}% and {at(hj[3], 63):.1f}% and the frozen "
        f"head's {at(hf[3], 48):.1f}% and {at(hf[3], 63):.1f}%. A looser budget "
        "gives an allocation more room to be wrong in as well as more room to "
        "be right, and the frozen head uses it to be wrong.")

    old_cj = 1.0 - rr[5]["rows"][0]["saving_pct_vs_release"] / 100.0
    k.par(
        "<b>Why 0.5 dB is not in that table.</b> At 0.5 dB every rate saturates "
        "for every configuration, so all three maps are identical and the only "
        "thing a comparison could measure is the cost vector each file was "
        "written with. The rule's 0.5 dB file is on "
        "runs/RECIPE512/ckpt_eval.pth.tar, where the shallowest exit was priced "
        f"at {old_cj:.4f} against the pinned {SAT['cost_j']:.4f}, a difference "
        f"worth {100 * (SAT['cost_j'] - old_cj):.2f} points of apparent saving. "
        "The file is results/raterank_RECIPE512_b05.json and it records that "
        "the rule matches the oracle map exactly at all five rates there, which "
        "is the only thing read from it.")

    diag = [["q", "rule agrees", "ρ depth", "ρ level", "ρ spread", "oracle"]]
    for r in rr[1]["rows"]:
        diag.append([str(r["qp"]), f"{r['agreement']:.3f}",
                     f"{-r['spearman_bits_vs_exit']:+.2f}",
                     f"{r['spearman_bits_vs_level']:+.2f}",
                     f"{r['spearman_bits_vs_spread']:+.2f}",
                     f"{r['oracle_saving_pct_vs_release']:.2f}"])
    k.rows(diag,
           "<b>Why the rule works, and it is not by agreeing</b>, at 0.1 dB. "
           "Compare the second column with the fifth: the rule picks the "
           "oracle's exit on a minority of tiles and still tracks the oracle's "
           "saving. The three Spearman correlations are between a tile's bit "
           "count and, respectively, the depth the oracle assigns it, its mean "
           "distortion across the ladder, and the spread between its shallowest "
           "and its deepest exit. <i>oracle</i> is the Lagrangian oracle "
           "measured inside the same run, which is the ceiling the rule is "
           "chasing.")
    k.note(
        "results/raterank_RECIPE512_b01.json. The file stores "
        "spearman_bits_vs_exit against the negated exit index, as "
        "scripts/raterank_curve.py computes it, so the correlation with depth "
        "quoted here is its negative. The oracle column is measured at one "
        "frame per sequence and is not charged for the exit map; deployed "
        "configuration A, which carries the map, reads "
        + ", ".join(f"{r['saving_pct_vs_release']:.1f}"
                    for r in k.J("signalled_RECIPE512_ctc53.json")["rows"]
                    if abs(r["budget_db"] - 0.1) < 1e-9)
        + "% over the same five rates at two frames per sequence "
          "(results/signalled_RECIPE512_ctc53.json).")

    k.par(
        "The rule agrees with the oracle on \\RateRankAgreeLo to "
        "\\RateRankAgreeHi of tiles, well under the frozen head's held-out "
        "\\RetrainAgreeOld, and saves more at every rate. Agreement counts a "
        "disagreement between two exits that are within a hair of each other "
        "exactly as heavily as one that costs most of the frame's error, and "
        "most disagreements are of the first kind. What the rule gets right is "
        "the ordering. A tile's bit count correlates with the spread across the "
        "ladder, which is how much that tile stands to gain from depth, at "
        "\\RateRankSpreadLo to \\RateRankSpreadHi at every rate, and with the "
        "depth the oracle actually assigns at "
        f"{min(-r['spearman_bits_vs_exit'] for r in rr[1]['rows']):+.2f} to "
        f"{max(-r['spearman_bits_vs_exit'] for r in rr[1]['rows']):+.2f}.")
    k.par(
        "What the rule cannot do is see past that ordering. A rank-1 model "
        "gives every tile the same relative profile over exits, so b(t) decides "
        "where on the ladder a tile falls and never the shape of its trade-off. "
        "That is the ceiling this baseline sits at, and it is the part a "
        "learned head would have to earn its parameters on. Neither of our two "
        "heads does.")

    # ------------------------------------------------------------------ F.5
    k.par(
        "<b>The two decoder-side signals are not complementary.</b> "
        "Normalising both surrogates to unit mean and blending them with one "
        "weight, from the calibrated bit rule at one end to the head's "
        "ordering at the other, the rule alone is best at every rate, and "
        "every positive weight is worse than none. Trusting the head "
        "completely costs \\BlendCostMin to \\BlendCostMax points, the most "
        "at \\BlendCostMaxQ. The head reads the entropy model's scales and "
        "the bit count is what those scales produce, so blending them mixes a "
        "signal with a learned approximation of itself.")
    k.par(
        "An earlier version of this table, measured before the checkpoint was "
        "pinned, showed a small gain at the two highest rates, and the paper "
        "carried it as evidence of weak complementarity. Re-measuring on the "
        "pinned weights removed it. The head in that measurement had been "
        "trained against whichever weights runs/RECIPE512/ckpt_eval.pth.tar "
        "held at the time, which is a moving pointer, so the first suspicion "
        "was that the head had simply been measured against the weights it "
        "was fitted to. It had not. Repeating the sweep on the epoch-1 pin "
        "with the corrected cost model gives w=0 at every rate as well "
        "(results/combined_RECIPE512_b01_e1.json), so the absence of "
        "complementarity survives a change of checkpoint and the gain was "
        "never a property of the two signals. What did change between the two "
        "measurements is the cost model: the earlier file predates the "
        "correction to the exit adapter's cost, and under the wrong cost an "
        "exit assignment can be scored better than it is. Every router number "
        "in this document is measured on the pinned checkpoint, after that "
        "correction, for this reason.")
    # Read from the two files rather than typed, so the sentence moves if
    # either sweep is repeated.
    def _w0(_f):
        _d = k.J(_f)
        if not _d:
            return {}
        return {r["qp"]: r["saving_pct_vs_release"] for r in _d["rows"]
                if abs(r["gamma"]) < 1e-9}
    _e0 = _w0("combined_RECIPE512_b01.json")
    _e1 = _w0("combined_RECIPE512_b01_e1.json")
    if _e0 and _e1:
        _lo, _hi = min(_e0), max(_e0)
        k.par(
            f"The same control makes a second point in passing. At w=0 the "
            f"epoch-1 ladder saves more than the epoch-0 one at every rate -- "
            f"{_e1[_lo]:.1f} against {_e0[_lo]:.1f} at q{_lo} and "
            f"{_e1[_hi]:.1f} against {_e0[_hi]:.1f} at q{_hi} -- which is the "
            f"epoch series of Section 5 seen through a different measurement. "
            f"The checkpoint the paper reports is the first one, and it is the "
            f"weakest one we have.")
    k.tbl("blend",
          "<b>Blending the two decoder-side signals</b> at 0.1 dB, both "
          "normalised to unit mean, with the weight running from the "
          "calibrated bit rule to the head's ordering. Bold is the best per "
          "rate. The two end columns are the two rules measured on their own "
          "through the same code path, which is the check that the blend is "
          "an interpolation and not a third system.")
    k.note("results/combined_RECIPE512_b01.json, on the pinned checkpoint, "
           "generated into paper/tables/blend.tex by "
           "scripts/make_paper_tables.py.")

    k.h2("How much a trained head varies")
    hh = [at(hj[1], q) - at(hf[1], q) for q in QPS]
    n_gap = sum(1 for i, q in enumerate(QPS)
                if abs(hh[i]) > min(rule(1, q) - at(hj[1], q),
                                    rule(1, q) - at(hf[1], q)))
    k.par(
        "No head in this work was trained twice at two seeds, so there is no "
        "seed variance to report and none was measured; "
        "results/router_ablation_e4.json records seed 0 for all six of its runs "
        "and one_checkpoint true. What can be reported is the spread between "
        "heads that were trained independently of each other, which is a looser "
        "quantity than seed variance and a larger one.")
    k.par(
        "<b>Two heads on one decoder.</b> The joint and frozen heads differ by "
        f"{hh[0]:+.2f} points at q0 and by {min(hh):+.2f} at q63, a range of "
        f"{max(hh) - min(hh):.2f} points across the five rates at 0.1 dB, and "
        "they cross: the joint head is the better of the two at q0 and the "
        f"worse at every other rate. At {n_gap} of the five rates the two heads "
        "differ from each other by more than the rule's margin over whichever "
        "of them is nearer, so a comparison against one trained head says less "
        "than it appears to. That is the reason both are printed in F.4 rather "
        "than the better of them.")

    BB = k.J("raterank_BEST_compare.json")
    best = [["q", "oracle", "head", "rule", "margin", "rule agrees"]]
    for q in [str(x) for x in QPS]:
        best.append([q, f"{BB['a_oracle'][q]:.2f}", f"{BB['router'][q]:.2f}",
                     f"{BB['raterank'][q]:.2f}",
                     f"{BB['raterank'][q] - BB['router'][q]:+.2f}",
                     f"{BB['agreement'][q]:.3f}"])
    k.rows(best,
           "<b>The same comparison on a second training run.</b> The margin "
           "column is the point: it is positive at four of the five rates and "
           "wider than on the pinned run, so the pinned numbers are the "
           "conservative ones. BEST is a separate recipe at a different epoch "
           "with its own head, trained the same way. The rule is ahead at "
           "\\BestRankWinsN of \\BestRankOfN rates, by up to \\BestRankBy "
           "points, and stays within \\BestRankToOracle points of the oracle "
           "everywhere. The one rate it concedes is the only rate on either run "
           "where a trained head finishes ahead, and it does so by under a "
           "point.")
    k.note("results/raterank_BEST_compare.json: the BEST run, \\NumSeq "
           "sequences, 0.1 dB, not the pinned checkpoint.")

    RT = k.J("router_retrain_compare.json")
    rt = [["q", "before", "after", "change", "β before", "β after"]]
    for q in [str(x) for x in QPS]:
        rt.append([q, f"{RT['v2'][q]:.2f}", f"{RT['v3'][q]:.2f}",
                   f"{RT['v3'][q] - RT['v2'][q]:+.2f}",
                   f"{RT['beta_v2'][q]:.1f}", f"{RT['beta_v3'][q]:.1f}"])
    k.rows(rt,
           "<b>Agreement rises and the saving does not follow.</b> The change "
           "column has both signs. The two heads are one recipe at one "
           "\\lambda, trained twice; held-out "
           "agreement went from \\RetrainAgreeOld to \\RetrainAgreeNew while "
           "the deployed saving moved by +\\RetrainGain points at "
           "q\\RetrainGainQp and &#8722;\\RetrainLoss at q\\RetrainLossQp. "
           "\\beta is the cost multiplier the bisection applies to move a head "
           "trained at one \\lambda onto this operating point, and the "
           "retrained head is ahead where that multiplier is small and behind "
           "where it is large. This is the same lesson as the rule's low "
           "agreement, from the other direction.")
    k.note(
        "results/router_retrain_compare.json. The file records no checkpoint, "
        f"and its before column reads {RT['v2']['0']:.2f}% at q0 where the "
        "pinned measurement in results/router_RECIPE512_b01_e4head.json "
        "reads "
        f"{at(hf[1], 0):.2f}%, so the two are not on one basis. Only the "
        "comparison of the two heads within the file is read here.")

    # ------------------------------------------------------------------ F.6
    k.par(
        "<b>Choosing the tilt without the test set.</b> Every \u03b2 in the "
        "main paper's A-against-B table was bisected against the budget on "
        "the test frames themselves, which is the one thing a deployed "
        "decoder cannot do. The table below repeats the measurement with the "
        "table a deployment would actually ship: \u03b2 bisected once per "
        "quality index on the \\HeldNCal held-out Open Images validation "
        "frames, then applied to the CTC frames without being touched again. "
        "Checkpoint, head, frames and budget are identical on both sides, so "
        "what separates the two columns is where \u03b2 came from.")
    k.tbl("beta_heldout",
          "<b>Choosing \u03b2 without the test set.</b> Left, \u03b2 "
          "bisected on the \\HeldNCal held-out validation frames and then "
          "applied to the test frames unchanged; right, \u03b2 bisected on "
          "the test frames themselves, which is what the main paper's "
          "signalled-against-predicted table reports. Savings are hook counts "
          "on the routed decode with the router's own compute charged against "
          "them. The last column re-bisects on the test set to the quality "
          "the held-out \u03b2 delivered, so the two allocations are "
          "differenced at one distortion rather than across two.")
    k.note("results/beta_calibration.json, generated into "
           "paper/tables/beta_heldout.tex by scripts/make_paper_tables.py. "
           "The calibration set is the 512 held-out Open Images validation "
           "frames the file records, disjoint from the training images and "
           "from the test sequences.")

    k.par(
        "The held-out \u03b2 misses the budget, and mostly on the high "
        "side: it delivers between \\HeldDbBest and \\HeldDbWorst dB where "
        "0.1 dB was asked for, overshooting at \\HeldNOver of the "
        "\\HeldNHeld rates it covers by \\HeldOverMin-\\HeldOverMax dB, "
        "which is \\HeldOverPctMin to \\HeldOverPctMax% of the budget, and "
        "undershooting at q\\HeldUnderMaxQp by \\HeldUnderMax. Distortion "
        "and saving move together, so the rows that overshoot report more "
        "saving, up to \\HeldGiveUpMinAbs points more at "
        "q\\HeldGiveUpMinQp, and the rate that undershoots reports "
        "\\HeldGiveUpMax less. The direction is not fixed by the frames "
        "alone: measured against a head fitted to other weights the same "
        "transfer undershot at every rate, so what a given \u03b2 delivers "
        "on video depends on the head as well as on the pictures. The last "
        "column shows what is not lost: at equal delivered quality the two "
        "allocations agree to within \\HeldTransferAbsMax points. What moves "
        "between the two sets is the decibel a given \u03b2 delivers, not "
        "the ordering it induces.")

    k.par(
        "The highest rate is where this used to fail outright. On the "
        "calibration frames the floor, the distortion tiling costs with every "
        "tile already at the deepest exit, is \\HeldFloorCalHigh dB there "
        "against \\HeldFloorTestHigh dB on the test frames. The difference "
        "changes sign along the ladder: the calibration floor "
        "sits below the test one at the \\HeldFloorNBelow lowest rates, by at "
        "most \\HeldFloorCalUnderMax dB, and above it at the "
        "\\HeldFloorNAbove highest, by \\HeldFloorCalOverMax dB at "
        "q\\HeldFloorCalOverMaxQp. On an "
        "earlier checkpoint that calibration floor sat above the budget: no "
        "allocation on 512 px photographs met 0.1 dB at the highest rate, the "
        "shipped table had a hole where that rate should be, and a decoder "
        "holding it fell back to the deepest allocation, which is our own "
        "full-depth path and saves nothing. Four epochs of training pulled "
        "the floor below the budget and the hole closed: \\HeldNNoBeta of "
        "the \\HeldNRates rates are missing from the table now. What remains "
        "is the difference between the two floors: a budget written as an "
        "absolute decibel sits a different distance above the floor on 512 px "
        "photographs than on 1080p video, and at q\\HeldFloorCalOverMaxQp "
        "that difference is a quarter of the budget on its own, which is why "
        "the transfer undershoots there. At the bottom of the ladder the two "
        "floors agree to within \\HeldFloorCalUnderMax dB and what is left "
        "is not the floor but what a given \u03b2 delivers above it. "
        "Calibrating on frames whose tiling penalty matches the ones the "
        "decoder will meet is the fix, and for a video decoder that means "
        "calibrating on video.")
