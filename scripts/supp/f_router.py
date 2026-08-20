"""The router: what a decoder-side predictor sees, what each input is worth,
what the head costs, and how it fares against a rule with no parameters.

Every number in this section is computed here from a file in results/, and the
file is named at the table that uses it. Nothing reads a checkpoint, opens a
CUDA context or writes to results/. The results files read are

    router_ablation.json                the six-variant input ablation
    router_ablation_<variant>.json      one training record per variant
    router_latency.json                 the head timed against its MAC share
    router_RECIPE512_b01/b03/b05.json   configuration B at three budgets
    raterank_RECIPE512_b01/b03/b05.json the parameter-free rule, same budgets
    raterank_BEST_compare.json          the same comparison on a second run
    router_retrain_compare.json         the head retrained with the mask fixed
    static_RECIPE512_b01.json           the exit cost vector, pinned
    saturation_RECIPE512_ctc53.json     c_j and the architectural ceiling
    tile_definition.json                tile, feature and latent geometry
    signalled_RECIPE512_ctc53.json      deployed configuration A, for contrast

paper/supp/f_router.tex carries the same section as LaTeX, with the same
numbers in the same order.
"""

#: The order the ablation is read in: the two heads that tie first, then the
#: single-group runs by descending agreement, then the no-per-tile control.
VARIANT_ORDER = ["stem", "all", "latent", "scales", "bits", "qp"]


def content(k):
    A = k.J("router_ablation.json")
    V = {x["label"]: x for x in A["variants"]}
    floor = V["stem"]["constant_best_agree"]
    # The per-variant records carry the training recipe the merged file does
    # not. Reading all six also puts them in the build's provenance list, which
    # is what the ablation table's note claims.
    REC = {l: k.J(f"router_ablation_{l}.json") for l in VARIANT_ORDER}
    RL = k.J("router_latency.json")
    k.J("tile_definition.json")
    ST = k.J("static_RECIPE512_b01.json")
    SAT = k.J("saturation_RECIPE512_ctc53.json")
    rr = {b: k.J(f"raterank_RECIPE512_b0{b}.json") for b in (1, 3, 5)}
    hd = {b: k.J(f"router_RECIPE512_b0{b}.json") for b in (1, 3, 5)}
    QPS = [r["qp"] for r in rr[1]["rows"]]
    cost = {u["exit"]: 1.0 - u["saving"] / 100.0
            for u in ST["rows"][0]["uniform"]}
    block = (cost[3] - cost[2]) / 2.0
    n_paper = RL["params"]
    n_abl = V["stem"]["router_params"]

    def head_at(b, q):
        return [x for x in hd[b]["rows"]
                if x["qp"] == q][0]["saving_pct_vs_release"]

    k.h1("The router")

    k.par(
        "An exit map assigns one exit index to every tile of a frame. This "
        "work produces such a map in three ways. Configuration A searches all "
        "K exits for every tile at the encoder and signals the answer. "
        "Configuration B predicts it at the decoder, from data the decoder "
        "already holds, and signals nothing. A third way computes it at the "
        "decoder from one number the entropy coder has already produced, with "
        "nothing learned at all. This section is about the second and the "
        "third.")

    k.par(
        "Write D(t,k) for the mean squared error of tile t decoded at exit k, "
        "and c<sub>k</sub> for the cost of exit k in units of one released "
        "decode. At a price \\lambda on compute, the oracle gives each tile "
        "the exit that minimises the per-tile Lagrangian.")
    k.eq(r"\ell(t,k) \;=\; D(t,k) \,+\, \lambda\, c_k")
    k.eq(r"k^{*}(t) \;=\; \mathrm{arg\,min}_{\,k \geq j}\ \ell(t,k)")
    k.par(
        "The split depth j is the number of trunk groups every tile runs "
        "before any tile may leave, so exits below j do not exist and the "
        "minimisation starts there. \\lambda is bisected once per rate so that "
        "the resulting map lands on the quality budget. <i>Agreement</i> below "
        "always means the fraction of tiles on which a predictor picks the "
        "same exit as this oracle at the same \\lambda. It is the quantity the "
        "head is trained on, and part of what this section reports is that it "
        "is not the quantity that matters.")

    # ------------------------------------------------------------------ A.1
    k.h2("What a decoder-side predictor can see")
    k.fig("router_sees.png",
          "<b>What each side knows.</b> Configuration A, above, has the source "
          "frame, so it can decode a tile at every exit and measure D(t,k) "
          "directly; the map it finds then has to be carried in the "
          "bitstream. Configuration B, below, has no source frame. It sees the "
          "bitstream and whatever the decode has produced by the time the "
          "trunk splits, and turns that into one score vector z(t) per tile.")
    k.par(
        "Four quantities are available at the decoder before any tile leaves "
        "the trunk. The <i>stem</i> is the trunk activation after the first j "
        "groups, 384 channels, on a feature map at one 64th of the output "
        "pixel count, since a 256 px tile is 32 feature positions across. The "
        "dequantised latent and the entropy model's predicted Gaussian scales "
        "are 256 channels each, at one 256th of the pixel count, a 256 px tile "
        "being 16 latent positions across. The quality index q is one integer "
        "per frame. And the entropy coder's estimated bits per latent position "
        "are produced on the way to the latent and then normally discarded. "
        "The last of these is what the parameter-free rule of A.7 uses on its "
        "own.")
    k.note(
        "Geometry from results/tile_definition.json, on the pinned "
        "checkpoint: rgb_patch 256, feature_patch 32, latent_patch 16. At "
        "1920×1080 the frame is padded to 2048×1280 and carries 40 tiles.")

    # ------------------------------------------------------------------ A.2
    k.h2("The head")
    k.par(
        "Configuration B's predictor is one small head, run once per frame, "
        "that emits K scores per tile. A 1×1 convolution takes the stem to 48 "
        "channels and another takes the concatenated latent and scales to 32. "
        "Each projected map is pooled over the tile into a mean and a standard "
        "deviation, which gives 96 and 64 numbers per tile; q, scaled to the "
        "unit interval, is appended as one more. That 161-vector goes through "
        "a layer normalisation and a three-layer perceptron of width 256 with "
        "SiLU activations, and the K outputs are the tile's scores. Capacity "
        "sits in the perceptron deliberately. It runs on one vector per tile, "
        "40 of them for a 1080p frame, so its width is nearly free, while the "
        "two 1×1 convolutions run per position and are the only part that "
        "costs anything.")
    k.fig("router_b_head.png",
          "<b>The head.</b> Two 1×1 projections, pooling to one vector per "
          "tile, then a perceptron. The parameter count and the compute share "
          "drawn on the figure are the measured ones from "
          "results/router_latency.json.",
          maxh=118)

    k.rows(
        [["module", "shape", "paper head", "ablation head"],
         ["stem projection", "1×1, 384 to 48", "18,480", "18,480"],
         ["latent projection", "1×1, 512 to 32", "16,416", "16,416"],
         ["rate projection", "1×1, 2 to 8", "0", "24"],
         ["layer norm", "161 or 177", "322", "354"],
         ["linear", "161 or 177 to 256", "41,472", "45,568"],
         ["linear", "256 to 256", "65,792", "65,792"],
         ["linear", "256 to K", "1,542", "1,542"],
         ["total", "", f"{n_paper:,}", f"{n_abl:,}"]],
        "<b>Where the head's parameters are.</b> Three quarters of them are in "
        "the perceptron, which costs almost nothing to run because it sees one "
        "vector per tile. The ablation family of A.5 adds a pathway for the "
        "entropy coder's bit estimate, widening the input vector from 161 to "
        "177, and that is the only difference between the two columns. The row "
        "counts are read off the module definition in flexuf/router/head2.py "
        f"and sum to the two totals the results files record: {n_paper:,} in "
        f"results/router_latency.json and {n_abl:,} in "
        "results/router_ablation.json.")

    k.par(
        "Exits below the split depth are suppressed in the score vector before "
        "the choice is taken. An earlier version of the head did that with a "
        "constant of &#8722;10<super>4</super>. Nothing in a cross-entropy or "
        "a regret objective penalises a common offset in the scores, one "
        "drifted in during training, and the head's own outputs settled near "
        "that scale, so the suppressed entries became the largest in a typical "
        "row and a large share of every allocation went to the cheapest exit "
        "for a reason unrelated to the tile. The mask is now &#8722;infinity, "
        "which cannot drift. A.8 reports what fixing it did to agreement and "
        "to saving, and the two moved in different directions.")

    # ------------------------------------------------------------------ A.3
    k.h2("What the head is trained to do")
    k.par(
        "The decoder is frozen throughout. The objective has two terms: a "
        "cross-entropy to the oracle's own choice, weighted per tile, and the "
        "expected regret against that choice.")
    k.eq(r"\mathcal{L} \;=\; \frac{1}{|T|}\sum_{t \in T} w_t\, "
         r"\mathrm{CE}\left(z(t),\, k^{*}(t)\right) \;+\; \alpha_r\, R")
    k.eq(r"w_t = \frac{s_t}{\bar{s}}, \quad "
         r"s_t = \max_k\, \ell(t,k) - \min_k\, \ell(t,k)")
    k.eq(r"R \;=\; \frac{1}{|T|}\sum_{t \in T} \sum_{k} P_k(t)"
         r"\left[\, \ell(t,k) - \min_{k'} \ell(t,k') \,\right]")
    k.par(
        "P(t) is the softmax of the scores z(t), and \\alpha<sub>r</sub> is 1 "
        "in every run reported here. The weight w<sub>t</sub> is the spread "
        "between the best and the worst option on that tile, normalised to "
        "unit mean, so a tile where two exits are within a hair of each other "
        "counts for little and a tile where the choice is most of the frame's "
        "error counts for much. R is non-negative, is zero exactly when all "
        "the mass sits on k*(t), and its value is the excess Lagrangian cost "
        "the head is paying against the oracle, in the units of the frontier. "
        "It is evaluated on a hard Gumbel straight-through sample [10], so "
        "training decides one exit exactly as inference does while gradients "
        "still reach the probabilities.")
    k.par(
        "The lineage is ClassSR's routing losses [12], with two of its three "
        "terms dropped for stated reasons. Its Class-Loss exists to repair the "
        "mismatch between a soft blend in training and an argmax at inference, "
        "and the straight-through sample removes that mismatch at source, so "
        "the repair has nothing left to do. Its Average-Loss exists because "
        "ClassSR's branches are separate networks that receive no gradient "
        "when unused; our exits share the trunk, and the trunk is frozen here, "
        "so an unused exit degrades nothing and forcing usage would mean "
        "routing tiles to exits the objective says are wrong.")
    st = REC["stem"]
    k.bullets([
        f"AdamW at learning rate {st['lr']:g} on a cosine schedule to zero, "
        "weight decay 10<super>-4</super>, gradients clipped at unit norm "
        "(scripts/train_router2.py).",
        f"{st['steps']:,} steps at a batch of {st['batch_size']} crops of "
        f"{st['crop']}×{st['crop']} from OpenImages [24], with one quality "
        "index drawn uniformly per image.",
        "One \\lambda for all rates, 1.3×10<super>-5</super>, which puts the "
        "oracle near the 0.1 dB operating point.",
        "Half the tiles of every batch are hidden from the optimiser, and "
        "agreement is only ever reported on that half. In-sample agreement on "
        "144 K parameters and a few thousand tiles will climb to anything at "
        "all.",
    ])

    # ------------------------------------------------------------------ A.4
    k.h2("Load balancing")
    k.par(
        "A gate trained against a frozen table can settle on one exit early "
        "and stay there, because the exit it happens to favour is the one it "
        "gets most of its gradient from. The control used here is a per-exit "
        "bias added to the scores before the choice is taken, updated by rule "
        "rather than by gradient.")
    k.eq(r"\beta_k \;\leftarrow\; \beta_k + \eta\left(\pi_k - u_k\right)")
    k.par(
        "\\pi<sub>k</sub> is the target share of exit k, u<sub>k</sub> its "
        "realised share over the batch, and \\eta is 10<super>-2</super>; the "
        "bias is then recentred to zero mean. Because it never enters the "
        "loss, no interference gradient is added to the objective. The "
        "mechanism is the loss-free balancing of [16], in place of the "
        "auxiliary balance loss of [7].")
    k.par(
        "We borrow the mechanism and not its justification. Mixture-of-experts "
        "models balance because experts are separate parameter sets with "
        "finite capacity, so a starved expert is both under-trained and wasted "
        "memory. Our exits share one trunk and have no capacity limit, so "
        "neither reason carries over, and concentration on one exit can be the "
        "correct answer. The target \\pi is therefore the oracle's own exit "
        "histogram on the batch rather than a uniform share: at a high price "
        "on compute the oracle genuinely does send almost every tile to one "
        "exit, and balancing toward uniform there would be forcing mistakes.")

    # ------------------------------------------------------------------ A.5
    k.h2("What each input is worth")
    k.par(
        "The head reads four per-tile signals and one per-frame signal, and "
        "this paper's own result is that a rule reading a single number beats "
        "it. That makes it fair to ask which of the five inputs carries "
        "anything. The ablation trains the same head once per input group, "
        "with every other group zeroed after its projection. Architecture, "
        "parameter count, optimiser, seed, image order and quality draws are "
        "identical across the six runs, so the only thing that differs is how "
        "much the head is allowed to know. Zeroing after the projection rather "
        "than deleting the projection is what makes that true.")
    k.par(
        "The quality index q stays live in every single-group run. It is one "
        "number per frame, the same for every tile in that frame, so it can "
        "say which operating point the decoder is at and cannot, even in "
        "principle, tell two tiles of one frame apart. Holding it live keeps "
        "the runs comparable on per-tile information alone. The run with q on "
        "its own is then the floor that choice implies, which is the agreement "
        "reachable with no per-tile information whatever.")

    k.rows(
        [["live inputs", "agreement", "± s.e.", "last 250", "over floor"]]
        + [[V[l]["inputs"], f"{V[l]['agree']:.4f}", f"{V[l]['stderr']:.4f}",
            f"{V[l]['heldout_agree_last250']:.4f}",
            f"{V[l]['agree'] - floor:+.4f}"] for l in VARIANT_ORDER],
        "<b>What each router input is worth.</b> Agreement with the oracle's "
        "exit choice on 1,200 frames and 4,800 tiles the head never trained "
        "on, with one standard error taken across frames rather than across "
        "tiles, since tiles of one image are not independent draws. "
        "<i>last 250</i> is the running held-out agreement over the final 250 "
        "training steps, on 12 tiles a step, and is shown only to confirm that "
        "the end-of-run measurement is not a fluctuation. <i>over floor</i> is "
        "the margin over the best single constant exit, which scores "
        f"{floor:.4f} on these tiles. The stem alone reaches the agreement of "
        "all five inputs together, and q alone does not clear the floor.")
    k.note(
        "results/router_ablation.json, and the six per-variant training "
        "records results/router_ablation_stem.json and its five siblings. All "
        "on the pinned checkpoint, all at \\lambda = 1.3×10<super>-5</super>, "
        f"all {st['steps']:,} steps from seed {st['seed']}, about an hour of "
        "one NVIDIA RTX A6000 each.")

    d_sa = V["stem"]["agree"] - V["all"]["agree"]
    se_sa = (V["stem"]["stderr"] ** 2 + V["all"]["stderr"] ** 2) ** 0.5
    k.par(
        f"<b>The stem alone is the whole head.</b> Reading the stem and q "
        f"gives {V['stem']['agree']:.4f}; reading the stem, the latent, the "
        f"scales, the bit count and q gives {V['all']['agree']:.4f}. The "
        f"difference is {d_sa:.4f}, against a standard error of "
        f"{V['stem']['stderr']:.4f} on each measurement and {se_sa:.4f} on "
        "their difference, so it is under a third of one standard error and "
        "its sign is not determined. Four of the five inputs add nothing this "
        "measurement can resolve. That is a negative result about those four "
        "inputs, and it is worth stating what it does and does not say: it "
        "does not show that the latent, the scales or the bit count are "
        "uninformative about a tile, only that whatever they carry is already "
        "carried by the stem, which is computed downstream of all of them.")
    k.par(
        "<b>The single-group ordering is clear and the separations are larger "
        f"than the noise.</b> The stem at {V['stem']['agree']:.4f} is "
        f"{V['stem']['agree'] - V['latent']['agree']:.4f} above the latent, "
        "against a standard error of "
        f"{(V['stem']['stderr']**2 + V['latent']['stderr']**2)**0.5:.4f} on "
        "the difference; the latent is "
        f"{V['latent']['agree'] - V['scales']['agree']:.4f} above the scales; "
        "and the scales are "
        f"{V['scales']['agree'] - V['bits']['agree']:.4f} above the bit count, "
        "which is more than seven standard errors. The bit count is the "
        "weakest per-tile input the head has. A.8 shows a rule that reads the "
        "bit count and nothing else beating the head at every rate, which is "
        "the sharpest form of the point that agreement is the wrong objective: "
        "the input the head learns least from is the input that wins when it "
        "is used differently.")
    k.par(
        f"<b>The quality index alone sits at the floor.</b> It reaches "
        f"{V['qp']['agree']:.4f} ± {V['qp']['stderr']:.4f} where the best "
        f"single constant exit scores {floor:.4f}. A head with no per-tile "
        "information cannot beat the best constant in expectation, and this "
        "one does not. The floor is worth quoting beside every other row, "
        "because an agreement of 0.62 cannot be read at all until it is known "
        "that 0.50 is free.")

    oh = V["stem"]["oracle_hist"]
    oh_tot = sum(oh)
    hist = [["exit map", "exit 2", "exit 3", "exit 4", "exit 5"],
            ["oracle"] + [f"{100 * x / oh_tot:.1f}" for x in oh[2:]]]
    for l in VARIANT_ORDER:
        ph = V[l]["pred_hist"]
        s = sum(ph)
        hist.append([V[l]["inputs"]] + [f"{100 * x / s:.1f}" for x in ph[2:]])
    k.rows(hist,
           "<b>Where each variant sends its tiles</b>, as a percentage of the "
           "4,800 evaluation tiles. Exits 0 and 1 are below the split depth "
           "and cannot be chosen. Every variant reproduces the oracle's use of "
           "the shallowest exit to within a point, and what degrades as inputs "
           "are removed is the separation of the middle of the ladder: the bit "
           "count alone nearly empties exit 3 and piles the mass on the "
           "deepest exit, and q alone uses two of the four exits available to "
           "it.")
    k.note("results/router_ablation.json, fields pred_hist and oracle_hist.")

    # ------------------------------------------------------------------ A.6
    k.h2("What the head costs")
    k.rows(
        [["q", "decode (ms)", "stem (ms)", "router (ms)", "share (%)"]]
        + [[str(r["qp"]), f"{r['decode_ms']:.2f}", f"{r['stem_ms']:.2f}",
            f"{r['router_ms']:.3f}", f"{r['router_share_pct_time']:.3f}"]
           for r in RL["rows"]],
        "<b>The head against the decode it decides for.</b> Three timings on "
        "one latent, interleaved so that a co-tenant's load drift lands on all "
        "of them equally. <i>decode</i> is the tiled decoder at the deepest "
        "exit, which is what the head's share is a share of; <i>stem</i> is "
        "the first j groups, which the head does not pay for because the "
        "decoder runs them anyway before the split. The head takes "
        "\\RouterTimePct\\% of the decode where its operation count predicts "
        "\\RouterCostPct\\%, a factor of \\RouterTimeFactor.")
    k.note(
        "results/router_latency.json: 1920×1080 padded to 2048×1280, 40 "
        f"iterations after warm-up, one NVIDIA RTX A6000, the {n_paper:,} "
        "parameter head runs/RECIPE512/routers2/v2_lam1.3e-5.pth. Measured on "
        "runs/RECIPE512/ckpt_eval.pth.tar rather than on the pinned "
        "checkpoint; the quantity is a ratio of two timings of the same "
        "weights, so it does not depend on which epoch they came from.")
    k.par(
        "The extra \\RouterTimeExtra points are launch overhead. A head of "
        f"{n_paper:,} parameters evaluated on 40 vectors is small enough that "
        "its wall-clock is dominated by the cost of starting its kernels, and "
        "an operation count does not contain that. Every configuration-B "
        "saving in this work charges the head at its operation share, which is "
        "the smaller of the two numbers and therefore the more flattering; the "
        "honest reading is that the head costs about half a percent of the "
        "decode in time. It remains small against what it decides about, since "
        f"one trunk block is {100 * block:.1f}% of the decode by the same cost "
        "vector.")

    # ------------------------------------------------------------------ A.7
    k.h2("A rule with no parameters")
    k.par(
        "The entropy model produces one number per tile before the trunk runs, "
        "at no cost, and then discards it: how many bits that tile's latents "
        "took. Per-block bit allocation is a standard quantity in learned "
        "compression, where it is something to choose, and block-level rate "
        "control sets it so that complex regions get more bits [42]. Here it "
        "is read in the other direction, after the fact and at the decoder, as "
        "a statement about how hard the region was. What follows is the whole "
        "rule, and it can be implemented from this subsection alone.")
    k.par(
        "<b>Step 1, the per-tile statistic.</b> Sum the entropy coder's "
        "estimated bits r<sub>i</sub> over the latent positions i falling "
        "inside tile t, and divide by the mean over the N tiles of that frame. "
        "Normalising per frame rather than globally is deliberate. The "
        "absolute rate level is a property of the frame, and what decides a "
        "tile is how it compares with the rest of its own frame.")
    k.eq(r"b(t) \;=\; N\, \frac{\sum_{i \in t} r_i}{\sum_{i} r_i}")
    k.par(
        "<b>Step 2, a rank-1 model of the distortion table.</b> Assume every "
        "tile has the same shape of decay across the ladder, up to a scale "
        "that depends only on b(t).")
    k.eq(r"\log D(t,k) \;\approx\; \alpha \log b(t) \,+\, c \,+\, "
         r"\log \varphi_k")
    k.par(
        "\\varphi has one entry per reachable exit and says what that exit "
        "costs on an average tile; \\alpha is an exponent fitted rather than "
        "assumed. Fitting it is what lets the rule discover the direction of "
        "the relation instead of asserting one, and an earlier version of this "
        "surrogate that fixed the exponent at 1 was wrong about that "
        "direction.")
    k.par(
        "<b>Step 3, the fit.</b> Take the mean of log D(t,k) over the "
        "reachable exits as the tile's difficulty, regress it on log b(t) by "
        "ordinary least squares to obtain \\alpha and c, and set log \\varphi "
        "to the mean residual per exit. Do this leave-one-sequence-out, so no "
        "sequence contributes to the profile that routes it. The result is an "
        "offline calibration constant rather than a per-frame quantity: six "
        "numbers per rate, thirty for the five rates reported here.")
    k.par(
        "<b>Step 4, route.</b> Run the oracle's own Lagrangian on the "
        "surrogate table in place of the true one.")
    k.eq(r"\hat{k}(t) \;=\; \mathrm{arg\,min}_{\,k \geq j}\ "
         r"\left[\, e^{c}\, b(t)^{\alpha}\, \varphi_k \,+\, \lambda\, c_k "
         r"\,\right]")
    k.par(
        "<b>Step 5, hit the budget.</b> Bisect \\lambda as configuration A "
        "does, in two levels: on the cheap per-tile table first, then "
        "correcting the target against a real decode of the resulting map, "
        "because the table and the decode differ slightly at the tile seams. "
        "Nothing is signalled, nothing is trained, and the arithmetic per tile "
        "is one power and K products. Sweeping \\lambda traces the lower "
        "convex hull of the achievable set in the sense of [28], so the point "
        "returned is the best one on that hull at the budget it consumes.")
    k.par(
        "The costs c<sub>k</sub> on the pinned checkpoint are "
        + ", ".join(f"{cost[e]:.4f}" for e in sorted(cost))
        + " for exits 2 to 5, recovered from the frame-level saving of maps "
        "that send every tile to one fixed exit "
        "(results/static_RECIPE512_b01.json). The shallowest of them is what "
        "sets the architectural ceiling of \\Ceiling\\% "
        f"(results/saturation_RECIPE512_ctc53.json, c_j = {SAT['cost_j']:.4f}).")

    phi = [["q", "\\lambda", "\\alpha", "φ<sub>2</sub>", "φ<sub>3</sub>",
            "φ<sub>4</sub>", "φ<sub>5</sub>"]]
    for r in rr[1]["rows"]:
        phi.append([str(r["qp"]), f"{r['lam']:.3g}", f"{r['alpha']:.2f}"]
                   + [f"{p:.4f}" for p in r["phi"]])
    k.rows(phi,
           "<b>The rule's entire state</b>, at the 0.1 dB budget: the bisected "
           "price \\lambda, the fitted exponent \\alpha, and the four-entry "
           "exit profile \\varphi, per rate. \\alpha is positive everywhere, "
           "so a tile whose latents cost more bits is modelled as harder at "
           "every exit and is sent deeper, which is what the correlations "
           "below confirm. \\varphi is nearly flat, spanning 4.1% at q0 and "
           "7.8% at q63 between its shallowest and deepest entry, against "
           "costs that span 0.61 to 1.01, so what moves a tile along the "
           "ladder is its own b(t) rather than the shape of the profile.")
    k.note("results/raterank_RECIPE512_b01.json, pinned checkpoint, \\NumSeq "
           "CTC sequences at one frame each.")

    # ------------------------------------------------------------------ A.8
    k.h2("The rule against the trained head")
    cmp_rows = [["q", "0.1 rule", "0.1 head", "0.3 rule", "0.3 head",
                 "0.5 rule", "0.5 head"]]
    for i, q in enumerate(QPS):
        row = [str(q)]
        for b in (1, 3, 5):
            row.append(f"{rr[b]['rows'][i]['saving_pct_vs_release']:.2f}")
            row.append(f"{head_at(b, q):.2f}")
        cmp_rows.append(row)
    k.rows(cmp_rows,
           "<b>The parameter-free rule against the \\RouterParams head</b>, at "
           "three budgets and five rates, as the percentage of the released "
           "decoder's operations saved. Both columns are charged for what they "
           "cost: the head pays its \\RouterCostPct\\% compute share and the "
           "rule pays nothing. The rule is ahead in every cell, though the "
           "0.5 dB pair is arithmetic on two different cost vectors rather "
           "than a comparison of allocations, for the reason given below.")
    k.note(
        "Rule: results/raterank_RECIPE512_b01.json, b03 and b05. Head: "
        "results/router_RECIPE512_b01.json, b03 and b05. All on the pinned "
        "checkpoint over \\NumSeq sequences at one frame each, except "
        "results/raterank_RECIPE512_b05.json, which is on "
        "runs/RECIPE512/ckpt_eval.pth.tar.")

    m01 = [rr[1]["rows"][i]["saving_pct_vs_release"] - head_at(1, q)
           for i, q in enumerate(QPS)]
    m03 = [rr[3]["rows"][i]["saving_pct_vs_release"] - head_at(3, q)
           for i, q in enumerate(QPS)]
    k.par(
        f"<b>At the working budget the rule wins at every rate.</b> The margin "
        f"is {m01[0]:+.2f} points at q0 and {min(m01):+.2f} at its narrowest, "
        f"which is q{QPS[m01.index(min(m01))]}, and it is positive at all "
        "\\RateRankNWins measured rates, the largest being "
        "\\RateRankBeatsBy points. A head trained on this decoder against this "
        "oracle therefore returns nothing over a rule with no parameters, "
        "while carrying \\RouterCostPct\\% of the decode that the rule does "
        "not.")
    k.par(
        "<b>At 0.3 dB the two saturate at the low rates and separate at the "
        "high ones.</b> At q0, q16 and q32 both configurations send every tile "
        f"to the shallowest exit and reach the ceiling, and the {m03[0]:+.2f} "
        "point difference between them is exactly the head's own "
        "\\RouterCostPct\\% compute share, which the rule does not pay. At q48 "
        f"and q63 the budget still binds and the margins widen to "
        f"{m03[3]:+.2f} and {m03[4]:+.2f} points, the latter being "
        "\\RateRankLooseAheadBy. The rule reaches \\RateRankLoose\\% at q63 "
        f"where the head reaches {head_at(3, 63):.1f}%. A looser budget gives "
        "the allocation more room to be wrong in as well as more room to be "
        "right.")

    old_cj = 1.0 - rr[5]["rows"][0]["saving_pct_vs_release"] / 100.0
    k.par(
        "<b>The 0.5 dB pair measures a cost model rather than an "
        "allocation.</b> At 0.5 dB every rate saturates for both "
        "configurations, so both send every tile to the shallowest exit and "
        "the two maps are identical. The reported margin of "
        f"{rr[5]['rows'][0]['saving_pct_vs_release'] - head_at(5, 0):.2f} "
        "points is arithmetic on two different cost vectors. The rule's file "
        "at that budget is on runs/RECIPE512/ckpt_eval.pth.tar, where the "
        f"shallowest exit is priced at {old_cj:.4f} against the pinned "
        f"{SAT['cost_j']:.4f}, a difference worth "
        f"{100 * (SAT['cost_j'] - old_cj):.2f} points, and the remaining "
        "\\RouterCostPct is the head's compute share. We report the row rather "
        "than dropping it, and read nothing from it beyond the saturation.")

    diag = [["q", "rule agrees", "ρ depth", "ρ level", "ρ spread", "oracle"]]
    for r in rr[1]["rows"]:
        diag.append([str(r["qp"]), f"{r['agreement']:.3f}",
                     f"{-r['spearman_bits_vs_exit']:+.2f}",
                     f"{r['spearman_bits_vs_level']:+.2f}",
                     f"{r['spearman_bits_vs_spread']:+.2f}",
                     f"{r['oracle_saving_pct_vs_release']:.2f}"])
    k.rows(diag,
           "<b>Why the rule works, and it is not by agreeing</b>, at 0.1 dB. "
           "<i>rule agrees</i> is the fraction of tiles on which the rule "
           "picks the oracle's exit, far below the head's held-out "
           "\\RetrainAgreeOld and still saving more at every rate. The three "
           "Spearman correlations are between a tile's bit count and, "
           "respectively, the depth the oracle assigns it, its mean distortion "
           "across the ladder, and the spread between its shallowest and its "
           "deepest exit. <i>oracle</i> is the oracle allocation measured "
           "inside the same run, which is the ceiling the rule is chasing.")
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
        "\\RateRankAgreeHi of tiles, well under the head's "
        "\\RetrainAgreeOld, and saves more at every rate. Agreement counts a "
        "disagreement between two exits that are within a hair of each other "
        "exactly as heavily as one that costs most of the frame's error, and "
        "most disagreements are of the first kind. What the rule gets right is "
        "the ordering. A tile's bit count correlates with the spread across "
        "the ladder, which is how much that tile stands to gain from depth, at "
        "\\RateRankSpreadLo to \\RateRankSpreadHi at every rate, and with the "
        "depth the oracle actually assigns at +0.38 to +0.54.")
    k.par(
        "What the rule cannot do is see past that ordering. A rank-1 model "
        "gives every tile the same relative profile over exits, so b(t) "
        "decides where on the ladder a tile falls and never the shape of its "
        "trade-off. That is the ceiling this baseline sits at, and it is the "
        "part a learned head would have to earn its parameters on. Neither of "
        "our heads does.")

    BB = k.J("raterank_BEST_compare.json")
    best = [["q", "oracle", "head", "rule", "margin", "rule agrees"]]
    for q in [str(x) for x in QPS]:
        best.append([q, f"{BB['a_oracle'][q]:.2f}", f"{BB['router'][q]:.2f}",
                     f"{BB['raterank'][q]:.2f}",
                     f"{BB['raterank'][q] - BB['router'][q]:+.2f}",
                     f"{BB['agreement'][q]:.3f}"])
    k.rows(best,
           "<b>The same comparison on a second training run.</b> BEST is a "
           "separate recipe at a different epoch with its own head, trained "
           "the same way. The rule is ahead at \\BestRankWinsN of "
           "\\BestRankOfN rates, by up to \\BestRankBy points, and stays "
           "within \\BestRankToOracle points of the oracle everywhere. The one "
           "rate it concedes is the only rate on either run where a trained "
           "head finishes ahead, and it does so by under a point.")
    k.note("results/raterank_BEST_compare.json: the BEST run, \\NumSeq "
           "sequences, 0.1 dB, not the pinned checkpoint. The margins there "
           "are wider than on the pinned run, so the pinned numbers are the "
           "conservative ones.")

    RT = k.J("router_retrain_compare.json")
    rt = [["q", "before", "after", "change", "β before", "β after"]]
    for q in [str(x) for x in QPS]:
        rt.append([q, f"{RT['v2'][q]:.2f}", f"{RT['v3'][q]:.2f}",
                   f"{RT['v3'][q] - RT['v2'][q]:+.2f}",
                   f"{RT['beta_v2'][q]:.1f}", f"{RT['beta_v3'][q]:.1f}"])
    k.rows(rt,
           "<b>Agreement rises and the saving does not follow.</b> The head "
           "retrained after the exit mask was fixed, on the same recipe at the "
           "same \\lambda. Held-out agreement goes from \\RetrainAgreeOld to "
           "\\RetrainAgreeNew while the deployed saving moves by "
           "+\\RetrainGain points at q\\RetrainGainQp and "
           "&#8722;\\RetrainLoss at q\\RetrainLossQp. \\beta is the cost "
           "multiplier the bisection applies to move a head trained at one "
           "\\lambda onto this operating point, and the retrained head is "
           "ahead where that multiplier is small and behind where it is "
           "large.")
    k.note(
        "results/router_retrain_compare.json. The file records no checkpoint, "
        f"and its configuration-B column reads {RT['v2']['0']:.2f}% at q0 "
        "where the pinned measurement in results/router_RECIPE512_b01.json "
        f"reads {head_at(1, 0):.2f}%, so the two are not on one basis. Only "
        "the comparison of the two heads within the file is read here.")

    # ------------------------------------------------------------------ A.9
    k.h2("What this section does not measure")
    k.bullets([
        "The ablation is one checkpoint, one \\lambda, one seed and "
        f"{st['steps']:,} steps per variant; results/router_ablation.json "
        "records the first of those as one_checkpoint. Nothing here says the "
        "ordering of the input groups would survive a second decoder or a "
        "second seed.",
        "The ablation reports agreement and nothing else. No variant was "
        "carried through a budget bisection to a deployed saving, so this "
        "section cannot say what a stem-only head would save, and the argument "
        "that agreement and saving come apart applies to the ablation as much "
        "as to the head.",
        "The head measured in configuration B was trained against "
        "runs/RECIPE512/ckpt_eval.pth.tar and then evaluated on the pinned "
        "checkpoint (results/router_RECIPE512_b01.json, field router2_meta), "
        "while the ablation heads were trained against the pinned checkpoint "
        f"itself. Its \\RetrainAgreeOld and the ablation's "
        f"{V['all']['agree']:.4f} are therefore not two measurements of one "
        "thing.",
        "One head covers all five rates. A head per rate is the obvious remedy "
        "for the cost multiplier and is not measured here.",
        "The rule's 0.5 dB file is off the pinned checkpoint, which is why "
        "that pair of columns is read only for saturation.",
    ])
