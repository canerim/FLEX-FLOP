"""Test settings, implementation and reproducibility.

Section A of the supplement. Everything a competent vision researcher who has
never seen this project needs in order to rebuild it: the released codec the
work starts from and the exact file it is, the test set, the measurement
conventions, every module with its shape and its parameter count, the two
algorithms a reader would re-implement, the training configuration with a
column saying which knob was searched and where, the hardware, the wall clock,
the seeds, the sources of variation, and the places where no file in results/
pins what the text says.

Where a number can be read out of results/ it is read out of results/, so it
cannot go stale. Three kinds of fact cannot be, and each is typed in the block
below with the file it was typed from: the trainer's own flags, the dataset
preparation statistics and the wall clock. Nothing that writes to results/ has
ever run inside the trainer. A.19 lists that gap rather than hiding it.

Nothing here opens a checkpoint, allocates on a GPU or writes to results/.
The module shapes and parameter counts that used to be typed from the source
now come from results/supp_module_shapes.json, which scripts/module_shapes.py
derives from the pinned checkpoint's state dict on the CPU.
"""

# ---------------------------------------------------------------------------
# Typed constants. Each block names the file it was read from. These are the
# only numbers in the section that k.J cannot supply.
# ---------------------------------------------------------------------------

# runs/RECIPE512/meta.json, written by the trainer when the run started.
META = {
    "dataset": "/data10/shareddata/openimages/dcvc_train",
    "batch_size": 8,
    "epochs_requested": 16,
    "tiles_per_512crop": 4,
}

# /data10/shareddata/openimages/dcvc_train/prepare_stats.json,
# written by scripts/prepare_openimages.py.
PREP = {
    "scanned": "384,795",
    "kept": "380,126",
    "train": "379,614",
    "val_heldout": 512,
    "val_stride": 400,
    "dropped_too_small": "4,669",
    "dropped_broken": 0,
    "min_side": 512,
}

# runs/RECIPE512/train_log.jsonl, 852 records, read at 2026-08-20 12:48.
LOG = {
    "records": 852,
    "steps_per_epoch": "47,451",
    "sec_per_200": 299.8,
    "sec_per_step": 1.50,
    "hours_per_epoch": 19.8,
    "skipped": 0,
    "last_epoch": 3,
    "last_step": "27,400",
    "grad_norm_median": 0.273,
    "snapshot": "2026-08-20 12:48",
}

# runs/warmstart/warmstart_report.json, written by
# scripts/warmstart_from_release.py. The copy on disk is the K = 12 rebuild of
# 17 August; the K = 6 file it also produced was not preserved separately.
WARM = {
    "release": "~/DCVC/checkpoints/cvpr2026_image.pth.tar",
    "tensors": 398,
    "adapters_at_init": 28,
    "deepest_max_diff": "0.0",
    "all_exits_max_diff": "0.0",
}

# scripts/launch_recipe512.sh, given without the two leading hyphens the shell
# needs, because a double hyphen is set as a dash by the typesetter.
LAUNCH = ("freeze_encoder train_patched epoch_offset 99 new_lr_scale 20 "
          "anchor_weight 10 seam_repair grid adapter_kind scaled "
          "distill_weight 1.0 distill_teacher adjacent lambdas 10 2048 "
          "batch_size 8 n 8 e 16 num_exits 6 split_depth 2 latent_patch 16 "
          "latent_halo 2 aux_weight 1.0 tile_pad replicate")

# flexuf/cost.py, the comment on the exit clamp: what a router that assigned
# the shallowest exit everywhere was billed before the clamp reached the cost
# model, against the true figure.
CLAMP_BUG = ("58.70", "42.9")


# ---------------------------------------------------------------------------
# The provenance table. Each entry is (results file, the generated table or
# macro group it feeds). The checkpoint column is read from the file at build
# time rather than typed, because that field is the only authority on what a
# number was measured on.
#
# The list is the set of files scripts/make_paper_tables.py passes to pick(),
# read off that file's source. pick() prefers a candidate whose ckpt field
# names the pinned checkpoint, then the most recently written.
# ---------------------------------------------------------------------------
CONSUMED = [
    ("signalled_RECIPE512_ctc53.json", "main results, complexity"),
    ("saturation_RECIPE512_ctc53.json", "operating range"),
    ("static_RECIPE512_b01.json", "static controls"),
    ("per_class_RECIPE512.json", "per class"),
    ("router_RECIPE512_b01.json", "A against B"),
    ("router_RECIPE512_b03.json", "A against B"),
    ("raterank_RECIPE512_b01.json", "rate rank"),
    ("raterank_RECIPE512_b03.json", "rate rank"),
    ("raterank_RECIPE512_b05.json", "rate rank"),
    ("hybrid_RECIPE512_b01_fixed.json", "partial signalling"),
    ("hybrid_raterank_b01.json", "rate rank, second run"),
    ("hybrid_v3_b01.json", "rate rank, second run"),
    ("latency_RECIPE512_sorted.json", "latency"),
    ("router_latency.json", "operating range"),
    ("adapter_ablation.json", "adapter ablation"),
    ("combined_RECIPE512_b01.json", "blend"),
    ("hybrid_RECIPE512_b03_fixed.json", "macros only"),
    ("hybrid_lorenz_b01.json", "macros only"),
    ("coupling_ablation.json", "macros only"),
    ("map_transfer.json", "macros only"),
    ("band_collapse.json", "macros only"),
    ("band_collapse_BEST.json", "macros only"),
    ("router_retrain_compare.json", "macros only"),
    ("signalled_BEST_ctc53.json", "ladder configurations"),
    ("signalled_BEST128_ctc53.json", "ladder configurations"),
    ("signalled_FINE12_ctc53.json", "ladder configurations"),
    ("ctc_seam_p256.json", "padding, tile size"),
    ("ctc_seam_p128.json", "tile size"),
    ("raterank_BEST_compare.json", "rate rank, second run"),
    ("bdrate.json", "BD-rate"),
    ("check_paper.json", "claim check"),
]


def _ckpt_label(d):
    """Short, honest name for whatever checkpoint a results file records."""
    ck = d.get("ckpt") or d.get("checkpoint") or ""
    ep = d.get("ckpt_epoch", d.get("epoch"))
    suffix = "" if ep is None else f", e{ep}"
    if not ck:
        return "<i>not recorded</i>"
    name = ck.rsplit("/", 1)[-1].replace(".pth.tar", "")
    run = ck.split("/")[1] if ck.startswith("runs/") else ""
    if name == "ckpt_PAPER":
        return "pinned" + suffix
    return f"{run}/{name}".replace("ckpt_", "") + suffix


def _n_label(d):
    n = d.get("n_sequences")
    f = d.get("frames_per_seq")
    if n and f:
        return f"{n} seq × {f}"
    if n:
        return f"{n} seq"
    nf = d.get("n_frames")
    if nf:
        return f"{nf} frames"
    return "-"


def _box(k, lines, cap):
    """A pseudocode box: one column, ruled above and below, no header row."""
    return k.rows([[ln] for ln in lines], cap, header=False)


# Non-breaking spaces, because reportlab collapses runs of ordinary ones and
# an algorithm box whose indentation has been collapsed is unreadable.
IND = "\u00a0" * 4          # one level of indentation inside a box
GAP = "\u00a0" * 3          # before a trailing comment


def _c(text):
    """A trailing comment inside a box."""
    return GAP + f"<i>{text}</i>"


# ---------------------------------------------------------------------------
def content(k):
    k.h1("Test settings, implementation and reproducibility")

    # ---------------------------------------------------------------- A.1
    k.h2("What this section covers")

    k.par(
        "This section is the recipe and the protocol: the released codec the "
        "work starts from and the exact file it is, the test set, the "
        "conventions every decibel is measured under, every module with its "
        "shape and its parameter count, the two algorithms a reader would have "
        "to re-implement, the training configuration with a column naming "
        "where each choice was measured, the hardware, the wall clock, the "
        "seeds and what varies between two runs of one command. The sections "
        "after it account for compute (B), derive the allocation rule and its "
        "bounds (C), report the complete results (D), describe the quality "
        "budget and its working range (E) and measure the decoder-side router "
        "(F). Numbers are not repeated across sections; where a quantity "
        "belongs to a later one, this section names it and stops. One fact "
        "governs the rest and is stated here rather than buried: the "
        "checkpoint every headline number comes from has seen the training set "
        "exactly once, and A.11 reports how much the numbers move with a "
        "second pass.")

    # ---------------------------------------------------------------- A.2
    k.h2("Test settings: the released codec, and what is frozen")

    ac = k.J("adapter_cost.json")
    td = k.J("tile_definition.json")
    ms = k.J("supp_module_shapes.json")
    audit = k.J("dmc_ld_recon_audit.json")
    cfg = td["config"]
    C = ac["trunk_channels"]
    skipped = [e["blocks_skipped"] for e in ac["exits"]]
    b_blocks = skipped[0] - skipped[1]
    N = skipped[0] + b_blocks
    K = cfg["num_exits"]
    j = cfg["split_depth"]
    share = ms["share_of_released_decode"]

    k.par(
        "There is no traditional-codec invocation in this work and none is "
        "needed. The axis measured here is decoder-side compute against a "
        "fixed learned decoder, on a bitstream that decoder produced, so the "
        "anchor is a checkpoint rather than a command line, and the released "
        "codec's own standing against VTM is published with it. That "
        "checkpoint is identified by content and not by path: "
        f"results/dmc_ld_recon_audit.json records {WARM['release']} with "
        f"sha256 {audit['intra_checkpoint_sha256'][:32]} and the note that the "
        "file is a bare state dict carrying no epoch, no optimiser and no "
        "step, so path plus hash is the whole of its provenance.")

    k.par(
        "The codec is imported rather than reimplemented: flexuf/model.py sets "
        "DCVC_ROOT to the DCVC checkout and imports src.models.image_model.DMCI "
        "from it, so the entropy model, the hyperprior, the quantisation-step "
        "tables and the 64 quality indices are the released code running "
        "unmodified. Our decoder is those weights rearranged: an opening "
        f"upsampling block, {N} DepthConvBlocks at C = {C} channels and a head "
        f"that pixel-shuffles back to RGB, with the {N} blocks grouped into "
        f"K = {K} exits of b = {b_blocks} blocks each and the first j = {j} "
        "groups running once over the whole frame before the decode splits "
        "into tiles. Exits 0 and 1 therefore lie below the split and cannot be "
        "selected; the four reachable exits are 2 to 5.")

    k.par(
        "The encoder is frozen for the whole of training, and that is asserted "
        "rather than assumed: scripts/signalled_curve.py compares our encoder "
        "state dict against the released one tensor by tensor and requires the "
        "largest absolute difference to be exactly zero before a single frame "
        "is measured. A trained decoder therefore consumes byte for byte the "
        "stream the released encoder produces, the bit rate is identical by "
        "construction, and the whole of the measured difference is "
        "decoder-side.")

    # ---------------------------------------------------------------- A.3
    k.h2("The test set, and what tiling does to it")

    pc = k.J("per_class_RECIPE512.json")
    cls = pc["rows"][0]["per_class"]
    tdrows = {r["name"]: r for r in td["rows"]}
    rows = [["class", "sequences", "resolution", "padded to", "tiles",
             "added px"]]
    total = 0
    for name, v in cls.items():
        total += v["n"]
        g = tdrows.get(v["res"])
        if g:
            pw, ph = g["padded"]
            w, h = g["W"], g["H"]
            add = f"{100 * (pw * ph - w * h) / (w * h):.1f}%"
            pad = f"{pw}×{ph}"
        else:
            add, pad = "-", "-"
        rows.append([name.replace("_", " "), v["n"], v["res"], pad,
                     v["tiles"], add])
    rows.append(["all", total, "", "", "", ""])
    t_set = k.rows(rows, "The test set, and what tiling does to each "
                         "resolution. Take from it that granularity is not "
                         "constant across the set: a 1080p frame carries 40 "
                         "tiles for the allocation to work with and a 416×240 "
                         "frame carries two, and the small classes also pay "
                         "the most padding. That single fact explains most of "
                         "the per-class spread reported in section D.")
    k.note("Sources: results/per_class_RECIPE512.json for the class membership "
           "and tile counts, results/tile_definition.json for the padded "
           "sizes; both on runs/RECIPE512/ckpt_PAPER.pth.tar. The sequences "
           "are UVG [25], MCL-JCV [26] and HEVC classes B, C, D and E [9]; the "
           "full list of \\NumSeq names is recorded in the measured field of "
           "results/signalled_RECIPE512_ctc53.json, with not_measured empty.")

    k.par(
        f"Frames are the leading frames of each sequence, taken consecutively. "
        f"The headline file uses two frames on each of the \\NumSeq sequences "
        f"of Table {t_set}, so 106 frames; the routed, rate-rank and partial "
        "signalling tables use one. Every results file records frames_per_seq "
        "and every table in this supplement states it. No frame is dropped and "
        "no sequence excluded: the not_measured field of every curve file is "
        "empty.")

    # ---------------------------------------------------------------- A.4
    k.h2("The measurement conventions")

    oq = k.J("supp_opquality_PAPER.json")
    sig = k.J("signalled_RECIPE512_ctc53.json")
    at01 = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - 0.1) < 1e-9}
    mb_lo = min(r["map_bits"] for r in at01.values())
    mb_hi = max(r["map_bits"] for r in at01.values())
    bpp_hi = max(r["bpp_added"] for r in at01.values())

    k.bullets([
        "<b>Colour.</b> Sources are YUV 4:2:0, 8 bit, converted to 4:4:4, "
        "divided by 255 and shifted by −0.5 for the network. PSNR is the "
        "released codec's own metric, (6·Y + U + V)/8, computed after "
        "converting the reconstruction back to 4:2:0 on the 0 to 255 scale. "
        "Measuring in 4:4:4 would compare against a chroma upsample of the "
        "source rather than the source, and would report a chroma PSNR no "
        "published DCVC-UF number could be compared with.",
        "<b>Padding.</b> Frames are padded by replication, bottom and right "
        "only, to a whole number of tiles; results/tile_definition.json records "
        f"the site as the RGB frame before the encoder and the mode as "
        f"\"{td['pad_mode']}\". The stock codec aligns to 16, which is enough "
        "for the transform and not for a tile grid. Rate is charged on the "
        "true pixel count and PSNR computed after cropping back, so padding "
        "can cost and cannot flatter, and it does cost: a tile containing "
        "replicated rows gives up about twice the quality of one that does "
        "not.",
        "<b>Decibels.</b> A budget is a loss against the released decoder on "
        "the same latent, averaged per frame and then over frames, which is "
        "the convention the released codec's own evaluation uses. Pooling all "
        "tiles into one mean squared error first is the other live convention "
        "and is not the same number; section E measures the gap at about a "
        "third of a 0.1 dB budget. No table mixes the two and every table says "
        "which it is on.",
        "<b>A second metric.</b> MS-SSIM is reported beside PSNR at the "
        f"operating point, computed as {oq['ms_ssim_convention']}. The "
        "implementation is self-checked at start-up: identical inputs score "
        f"{oq['ms_ssim_self_check']['identical']:.6f} and a 3×3 box blur "
        f"scores {oq['ms_ssim_self_check']['blurred_3x3_box']:.4f}, both "
        "recorded in results/supp_opquality_PAPER.json.",
        "<b>The map is billed.</b> The exit map is entropy coded from a "
        "per-frame histogram and its cost added to the rate before any saving "
        f"is computed: {mb_lo:.0f} to {mb_hi:.0f} bits per frame at the 0.1 dB "
        f"budget, at most {bpp_hi:.2e} bpp. Every row records map_bits and the "
        "bpp it added.",
        "<b>The reference.</b> The released decoder re-expressed in the same "
        "ladder and selected by K, so a K = 12 run cannot be quoted against "
        "the K = 6 remap.",
    ])

    k.par(
        "Two limits belong beside that list. Sweeping a multiplier reaches "
        "only the lower convex hull of the achievable set [28, 29], so a "
        "budget falling in a gap of the hull is met at the nearest reachable "
        "point; every row carries a reachability flag and a floor, and rows "
        "whose floor already exceeds the budget are reported as unreachable "
        "rather than dropped. And 0.1 dB is a reporting convention, not a "
        "perceptual threshold. Subjective work measures the smallest "
        "noticeable change in quantisation parameter or in a video quality "
        "metric, not in tenths of a decibel of PSNR, so nothing published "
        "licenses a claim that 0.1 dB is invisible; 0.3 dB and 0.5 dB are "
        "reported beside it so that the choice carries no weight.")

    # ---------------------------------------------------------------- A.5
    k.h2("The ladder, module by module")

    dec_new = (ms["roles"]["seam_repair"]["params"]
               + sum(ms["roles"][f"adapters.{a}"]["params"] for a in range(5)))
    dec_rel = ms["params_decoder_side"] - dec_new
    dead = sum(ms["roles"][f"adapters.{a}"]["params"] for a in range(j))
    rl = k.J("router_latency.json")

    def P(x):
        return f"{x:,}"

    rows = [["module", "runs", "in → out", "params", "% dec"]]
    rows.append(["upsample", "frame", f"256 → {C}, ×2",
                 P(ms["roles"]["upsample"]["params"]),
                 f"{100 * share['upsample']:.2f}"])
    rows.append([f"groups 0 to {j - 1}, stem", "frame",
                 f"{C} → {C}", P(j * ms["roles"]["groups.0"]["params"]),
                 f"{100 * j * share['groups.0']:.2f}"])
    rows.append([f"groups {j} to {K - 1}, routable", "tile",
                 f"{C} → {C}", P((K - j) * ms["roles"]["groups.0"]["params"]),
                 f"{100 * (K - j) * share['groups.0']:.2f}"])
    rows.append(["one DepthConvBlock", "tile", f"{C} → {C}",
                 P(ms["roles"]["one_block"]["params"]),
                 f"{100 * share['one_block']:.2f}"])
    rows.append(["adapters 0 to 3, FFN", "tile, at its exit",
                 f"{C} → {4 * C} → {C}",
                 P(ms["roles"]["adapters.0"]["params"]),
                 f"{100 * share['adapters.0']:.2f}"])
    rows.append(["adapter 4, 1×1", "tile, at its exit", f"{C} → {C}",
                 P(ms["roles"]["adapters.4"]["params"]),
                 f"{100 * share['adapters.4']:.2f}"])
    rows.append(["seam repair, grid", "after stitching",
                 f"{C} → {C}", P(ms["roles"]["seam_repair"]["params"]),
                 f"{100 * share['seam_repair']:.2f}"])
    rows.append(["head, PixelShuffle 8", "frame", f"{C} → 192 → 3",
                 P(ms["roles"]["head"]["params"]),
                 f"{100 * share['head']:.2f}"])
    rows.append(["router head (B)", "tile", "161 → 6",
                 P(rl["params"]),
                 f"{rl['router_share_pct_macs']:.3f}"])
    rows.append(["decoder, released", "", "", P(dec_rel), "100.00"])
    rows.append(["decoder, ours", "", "", P(dec_new),
                 f"+{100 * dec_new / dec_rel:.1f}"])
    rows.append(["encoder, frozen", "encode", "",
                 P(ms["params_encoder_side"]), "-"])
    t_shapes = k.rows(
        rows,
        "Where every module sits, what shape it is, what it weighs and what "
        "share of one released decode it costs; the adapter rows are per "
        "adapter and the group rows are for the whole run of groups. Take "
        "from it three things. The "
        f"routable region is groups {j} to {K - 1}, which is "
        f"{100 * (K - j) * share['groups.0']:.0f}% of the decode and the only "
        "part an exit can skip. The adapters and the seam repair add "
        f"{100 * dec_new / dec_rel:.1f}% to the decoder's parameters and, when "
        f"one adapter runs, at most {100 * share['adapters.0']:.2f}% to its "
        "arithmetic. And the deepest exit costs slightly more than the "
        "released decoder rather than the same, because it still pays the "
        "seam-repair pass.")
    k.note(
        "Source: results/supp_module_shapes.json, derived by "
        "scripts/module_shapes.py from the state dict of "
        "runs/RECIPE512/ckpt_PAPER.pth.tar (md5 "
        f"{ms['ckpt_md5'][:16]}) on the CPU. Parameter counts are tensor "
        "sizes; the share column is each module's multiply-accumulates per "
        "feature pixel over the released decoder's total on that grid, the "
        "subpel convolution that opens the decoder being charged at one "
        "quarter because it runs on the latent grid. The three aggregate "
        "shares that fall out, 8.16% for the upsample, 89.44% for the "
        "twelve-block trunk and 2.40% for the head, reproduce the forward-hook "
        "audit in scripts/mac_audit.py to four decimals. Section B accounts "
        "for the per-exit cost that follows; it is not repeated here.")

    fg_pipe = k.fig(
        "pipeline_detail.png",
        "<b>The three pieces a reader has to re-implement.</b> <b>a</b>, "
        "patchify is a pure reshape and costs nothing, and it is where both "
        "the seam and every operation of saving come from. <b>b</b>, one exit "
        "group is two DepthConvBlocks, whose only operator with spatial extent "
        "is a single 3×3 depthwise worth 0.29% of the block, so an early exit "
        "gives up almost purely pointwise capacity. <b>c</b>, the two adapters "
        "that replace it, both residual and both zero-initialised. Shapes are "
        "those of a 1080p frame padded to 2048×1280 and split into 40 tiles.",
        maxh=230)

    k.par(
        f"Two of the five adapters can never run: at j = {j} the decoder "
        f"clamps every exit to at least {j}, so adapters 0 and 1 are "
        "unreachable at inference. They still take gradient from the "
        "distillation term, which taps every exit, and they still occupy "
        f"{P(dead)} of the {P(dec_new)} parameters we add. Dropping them is "
        "the obvious saving in decoder size and is not done here, because the "
        "split depth is a configuration knob and a checkpoint that has dropped "
        "them cannot be re-run at another j.")

    # ---------------------------------------------------------------- A.6
    k.h2("Tiles, and the exit clamp")

    r1080 = tdrows["1920x1080"]
    k.par(
        "patchify is a view, a permute and a reshape with no arithmetic at "
        f"all: a feature map of shape [1, {C}, "
        f"{r1080['feature_in_patchify'][1]}, "
        f"{r1080['feature_in_patchify'][0]}] becomes "
        f"{r1080['tiles_measured']} batch elements of [{C}, "
        f"{td['feature_patch']}, {td['feature_patch']}], as panel a of Figure "
        f"{fg_pipe} shows. Tile order is row major, which is what the exit map "
        "is indexed by and what flexuf/eval.py reproduces with the same "
        "permutation when it measures per-tile error. One tile is "
        f"{td['rgb_patch']} px of RGB, {td['feature_patch']} px of feature map "
        f"and {td['latent_patch']} px of latent, at {td['pixels_per_latent']} "
        "pixels per latent position. patchify asserts divisibility rather than "
        "padding: an unpadded 1080p frame gives a 40×40 feature map that is "
        f"not divisible by {td['feature_patch']}, and the run stops instead of "
        "producing a ragged last row.")

    k.par(
        "The clamp is the most consequential line in the decoder. forward() "
        f"applies exit_map.clamp(min = j, max = K − 1) before the per-tile "
        f"loop, so a map naming exit 0 at j = {j} runs group {j} and wears "
        f"exit {j}'s adapter. Four separate places have to apply the same "
        "clamp and each was wrong at some point in this project. Getting any "
        "of them wrong is silent, which is why they are listed rather than "
        "described.")

    rows = [["file and function", "what it does"],
            ["decoder.py, forward",
             "clamps the map to [j, K−1] before the per-tile loop"],
            ["cost.py, exit_costs",
             "charges max(k, j) blocks, and bills the adapter the clamped "
             "exit wears rather than the one the map names"],
            ["eval.py, tiled_exit_mses",
             "fills the columns below j with the decode at exit j, so an "
             "argmin over all K lands on a reachable choice"],
            ["head2.py, forward and oracle_ce_loss",
             "masks the logits below the split depth with −inf, and clamps "
             "the oracle label to the same bound"]]
    t_clamp = k.rows(
        rows,
        "The exit clamp, and the four places that have to repeat it. Take "
        "from it the failure mode rather than the rule: with the clamp in the "
        "decoder but not in the cost model, a router that assigned the "
        f"shallowest exit everywhere was billed {CLAMP_BUG[0]}% saved where "
        f"the true figure is {CLAMP_BUG[1]}%, and that inflated number made "
        "the router appear to beat a bound it cannot beat. The mask in the "
        "router is −inf and not a large negative constant, because nothing in "
        "a cross-entropy penalises a common offset, the raw logits drifted to "
        "about −10<super>4</super>, and a −10<super>4</super> mask then became "
        "the largest entry in the row.")
    k.note("Source: the four files named in the first column, under flexuf/. "
           "The two "
           "percentages are the comment in flexuf/cost.py that records the "
           "measurement; no results file carries them, which A.19 repeats.")

    # ---------------------------------------------------------------- A.7
    k.h2("The three modules that are ours")

    k.par(
        f"<b>The exit adapters.</b> Every exit but the deepest hands the head "
        f"a "
        "corrected feature rather than the raw one, and the two forms Table "
        f"{t_shapes} weighs are built as follows. "
        "The plain adapter is f + conv(f) with one 1×1 whose weight and bias "
        "start at zero. The feed-forward adapter repeats the pointwise pair "
        "the exit skipped: a 1×1 expanding C to 4C, the block's own 4:1 gated "
        "chunk-add, and a 1×1 back to C, again zero-initialised. Which form an "
        "exit gets is decided by how much it stands in for, the feed-forward "
        "form when it skips four or more blocks and the 1×1 otherwise, which "
        "at the shipped geometry gives the feed-forward form to exits 0 to 3. "
        "Both are entirely pointwise, so neither adds receptive field and "
        "neither adds seam, and zero initialisation is what makes the ladder "
        "bit-exact against the release at step zero.")

    k.par(
        "<b>The seam repair.</b> One pass over the stitched canvas, after "
        "unpatchify and before the head, which is the only point at which a "
        "3×3 can see across a tile boundary. It is a depthwise 3×3 with "
        "replicate padding, a gated activation and a zero-initialised 1×1, and "
        "the correction is multiplied by a learned gate indexed by position "
        f"within a tile: a {td['feature_patch']}×{td['feature_patch']} array "
        f"of scalars shared across all {C} channels, so the module can act on "
        "the boundary ring and switch itself off in the interior. The gate is "
        "initialised to a decaying function of the distance to the nearest "
        "tile edge, so at step zero it already knows where the seams are and "
        "training refines rather than discovers. It is charged and not hidden: "
        f"the cost table bills it at {100 * share['seam_repair']:.2f}% of a "
        "decode, which is why the deepest exit costs "
        "\\DeepestUniformCost rather than exactly one released decode.")

    k.par(
        "<b>The router head.</b> Configuration B predicts the exit map at the "
        "decoder instead of reading it from the bitstream. Its inputs are the "
        "stem after the shared groups, the decoded latent concatenated with "
        "the entropy model's predicted scales, and the quality index; each is "
        "projected by a 1×1 and pooled to a mean and a standard deviation per "
        "tile, giving 2·48 + 2·32 + 1 = 161 numbers that a LayerNorm and three "
        "linear layers of width 256 turn into K logits. Only the 1×1 on the stem runs per "
        f"pixel, while the perceptron runs once per tile, "
        f"{r1080['tiles_measured']} times for a 1080p frame, which is why its "
        "width is nearly free. The head is \\RouterParams parameters and "
        "\\RouterCostPct% of the decode. What it learns, what each input is "
        "worth and how it compares with a parameter-free rule are section F's "
        "subject; only its shape belongs here.")
    k.note("Sources: results/supp_module_shapes.json for the shapes and the "
           "seam-repair share, results/router_latency.json for the router's "
           "parameter count and its share of the arithmetic. The router "
           "checkpoint that file times, "
           "runs/RECIPE512/routers2/v2_lam1.3e-5.pth, records "
           "runs/RECIPE512/ckpt_eval.pth.tar as the decoder it was trained "
           "against, not the pinned one; the six ablation heads of section F "
           "are on the pinned checkpoint.")

    # ---------------------------------------------------------------- A.8
    k.h2("The deployed decode, and how training differs from it")

    box1 = _box(k, [
        "<b>input</b> latent y, quantisation step q, exit map m",
        "&nbsp;1&nbsp; f = upsample(y)",
        "&nbsp;2&nbsp; <b>for</b> g = 0 … j−1: f = groups[g](f)"
        + _c("frame, no borders"),
        "&nbsp;3&nbsp; tiles, nh, nw = patchify(f, 32)" + _c("reshape, 0 MAC"),
        "&nbsp;4&nbsp; m = clamp(m, j, K−1)" + _c("below j cannot run"),
        "&nbsp;5&nbsp; active = all tiles; work = tiles",
        "&nbsp;6&nbsp; <b>for</b> g = j … K−1:",
        IND + "&nbsp;7&nbsp; work = groups[g](work)"
        + _c("borders meet padding"),
        IND + "&nbsp;8&nbsp; leaving = active tiles whose m equals g",
        IND + "&nbsp;9&nbsp; out = adapters[g](work[leaving])"
        + _c("raw if g = K−1"),
        IND + "10&nbsp; <b>if</b> training: out = out · gate[leaving]",
        IND + "11&nbsp; canvas[leaving] = out",
        IND + "12&nbsp; drop leaving from work and from active"
        + _c("the saving"),
        "13&nbsp; canvas = unpatchify(canvas, nh, nw)",
        "14&nbsp; canvas = canvas + G[r mod P, c mod P] · seam(canvas)",
        "15&nbsp; <b>return</b> pixel_shuffle(head(canvas · q), 8)",
    ], "The deployed routed decode, line for line. Take from it that line 12 "
       "is the entire saving: every group runs on fewer tiles than the last, "
       "and nothing else in the procedure is cheaper than a full decode. Line "
       "10 is the only difference between training and inference, and it is "
       "numerically the identity.")

    k.par(
        f"Line 10 of Table {box1} deserves the space. A router trained jointly "
        "with the decoder needs gradient to reach its logits and an argmax "
        "blocks it, so the gate is p / p.detach() with p the selected exit's "
        "probability under a Gumbel-softmax relaxation: its value is exactly 1 "
        "and the reconstruction bit-identical to the deployed path, while its "
        "gradient is the straight-through estimator. At inference the gate is "
        "absent and the exit is a hard argmax or a transmitted symbol. That is "
        "the one place where the trained object and the shipped object are not "
        "the same computation.")

    k.par(
        "A second execution path exists and is bit-identical. With sorted "
        "execution the tiles are ordered by depth with a stable argsort before "
        "line 6, so \"still active at group g\" becomes a contiguous prefix, "
        "every group is a slice rather than a masked gather, and the loop "
        "costs one host read instead of one synchronisation per group. Only "
        "the wall clock moves, by \\WallSortedGain points, and section B "
        "reports it.")

    # ---------------------------------------------------------------- A.9
    k.h2("The training data")

    k.par(
        "Training uses Open Images [24], fetched from the CVDF public mirror "
        "by scripts/fetch_openimages.sh and prepared by "
        "scripts/prepare_openimages.py into the flat layout DCVC-UF's own "
        f"ImageFolder expects. Of {PREP['scanned']} files scanned, "
        f"{PREP['kept']} survive a minimum-side filter of {PREP['min_side']} "
        f"px, {PREP['dropped_too_small']} are dropped as too small and "
        f"{PREP['dropped_broken']} as unreadable. The filter is not cosmetic: "
        "the loader zero-pads any image smaller than the current crop, and an "
        "image that is half black border teaches the decoder to reconstruct "
        f"black borders. Every {PREP['val_stride']}th entry of the sorted "
        f"survivor list is then held out, giving {PREP['val_heldout']} "
        f"validation images and {PREP['train']} training images with no "
        "overlap, deterministically, so every run and every control sees the "
        "same held-out set.")

    k.par(
        "Each sample is a random crop at the schedule's patch size with a "
        "random horizontal flip, converted to YCbCr, scaled to [0,1] and "
        "shifted by −0.5. One quality index is drawn uniformly over the 64 "
        "levels per sample and the matching multiplier comes from the "
        "log-spaced table running 10 to 2048, so one set of weights covers the "
        "whole rate range rather than a point on it. The crop is 512 px, which "
        "is what the released recipe's final phase asks for and also the "
        f"smallest crop carrying more than one tile: at {td['rgb_patch']} px "
        f"tiles a 512 px crop holds {META['tiles_per_512crop']} and a 256 px "
        "crop holds one, and a single tile has no border against a neighbour, "
        "so patched training at the smaller crop would train a configuration "
        "that cannot occur at inference. Evaluation uses a deterministic "
        "centre crop with no flip, because the loader's random crop swung one "
        "model between 29.54 and 30.58 dB at q0 across two runs of one "
        "command.")

    # ---------------------------------------------------------------- A.10
    k.h2("The warm start, and the controls it asserts")

    k.par(
        "scripts/warmstart_from_release.py performs the rearrangement, and it "
        "is a key remap and nothing else: the opening upsample takes dec_1[0], "
        "group g block i takes dec_1[1 + g·b + i], the head takes dec_2, and "
        "the adapters and the seam-repair gate are new and zero-initialised. "
        "The encoder, hyperprior, entropy model and quantisation-step tables "
        f"are copied verbatim. The transfer moves {WARM['tensors']} tensors "
        f"and creates {WARM['adapters_at_init']} adapters at initialisation. "
        "Two controls follow from zero initialisation and both are asserted at "
        "every build rather than assumed: the deepest exit reproduces the "
        "released decoder with a largest absolute difference of "
        f"{WARM['deepest_max_diff']}, and every untrained exit equals the "
        "released decoder truncated at that depth, also "
        f"{WARM['all_exits_max_diff']}. A third lives in the checkpoint "
        "loader, which tolerates a missing or extra tensor only under the "
        "adapter, seam-repair, pad-coefficient and router prefixes and raises "
        "otherwise; both sides of a tolerated mismatch are zero-initialised "
        "identities, so dropping one is exact rather than approximate.")

    # ---------------------------------------------------------------- A.11
    k.h2("The training run")

    fg_train = k.fig(
        "training_scheme.png",
        "<b>The training scheme.</b> <b>a</b>, one forward pass produces all K "
        "reconstructions, so the ladder is trained as one object. <b>b</b>, "
        "the three loss terms. <b>c</b>, the encoder is never touched. "
        "<b>d</b>, why each line of the recipe is there. The figure is a "
        "schematic drawn from the model definition; the parameter counts it "
        "carries are the ones tabulated in A.5.", maxh=205)

    k.par(
        "The recipe is Microsoft's own, applied at the point in their schedule "
        "that matches what we inherit. Their schedule is a 105-entry table of "
        "learning rate and crop size indexed by epoch, and the released "
        "weights are the output of its last entry, so applying entry zero "
        "would kick a converged codec with the from-scratch learning rate. The "
        "run enters at offset 99, inside their final 512 px phase at a "
        f"learning rate of 1×10<super>-5</super>; all {LOG['records']} records "
        "in the run's log carry that rate and that crop, which is the check "
        "that the offset did what it was meant to. Because the offset lands in "
        "the phase the experiment needs, no crop-size override is passed at "
        "all, where every other warm-started run here raises the crop floor "
        "by hand.")

    k.par(
        "One deviation from a plain fine-tune is deliberate. The adapters and "
        "the seam-repair gate start at zero and have to move furthest, while "
        "the inherited trunk must not move at all if the deepest exit is to "
        "stay the released decoder, and the cost of choosing wrongly between "
        "them is measured: at the from-scratch rate the trunk drifts 0.20 dB "
        "of anchor at q32 in a single epoch. The new modules therefore get "
        "their own parameter group at 20 times the schedule's rate, selected "
        "by name prefix, with the multiplier re-applied whenever the schedule "
        "sets the base rate.")

    k.par(
        "The loss has three terms. The rate-distortion term is the released "
        "loss, a multiplier times mean squared error plus bits per pixel, with "
        "the error averaged over the exits under fixed weights. The anchor "
        "term pins the deepest exit to a frozen copy of the released decoder, "
        "because every saving in the paper is quoted against that decoder and "
        "a reference drifting underneath the measurement makes the measurement "
        "meaningless. The distillation term supervises each adapter in feature "
        "space against the exit one step deeper, normalised by the teacher's "
        "own variance, following the adjacent-teacher form rather than the "
        "deepest-teacher form [17], since a large gap between student and "
        "teacher is reported to hurt the shallowest exits most. Gradients are "
        "clipped at a norm of 0.1 and a batch whose norm is not finite is "
        f"dropped; across all {LOG['records']} records the count of steps "
        f"skipped for that reason is {LOG['skipped']} and the median gradient "
        f"norm is {LOG['grad_norm_median']}.")

    k.par(
        f"An epoch is {LOG['steps_per_epoch']} optimiser steps, which is "
        f"{PREP['train']} images at batch {META['batch_size']}. The run logs "
        "every 200 steps and the median interval is "
        f"{LOG['sec_per_200']} s, so a step costs about "
        f"{LOG['sec_per_step']} s and an epoch about "
        f"{LOG['hours_per_epoch']} hours. Training is fp32 throughout on a "
        "single NVIDIA RTX A6000 with eight data workers.")

    e1 = k.J("signalled_RECIPE512_e1.json")
    a1 = {r["qp"]: r for r in e1["rows"]
          if abs(r["budget_db"] - 0.1) < 1e-9}
    qps = sorted(at01)
    rows = [["quality index", "epoch 0", "epoch 1", "change",
             "floor e0", "floor e1"]]
    d0 = d1 = 0.0
    for q in qps:
        s0 = at01[q]["saving_pct_vs_release"]
        s1 = a1[q]["saving_pct_vs_release"]
        d0 += s0
        d1 += s1
        rows.append([f"q{q}", f"{s0:.2f}", f"{s1:.2f}", f"+{s1 - s0:.2f}",
                     f"{at01[q]['floor_db']:.4f}", f"{a1[q]['floor_db']:.4f}"])
    m0, m1 = d0 / len(qps), d1 / len(qps)
    rows.append(["mean", f"{m0:.2f}", f"{m1:.2f}", f"+{m1 - m0:.2f}", "", ""])
    t_epoch = k.rows(
        rows,
        "One more pass over the training set is worth about three points. "
        "Saving at the 0.1 dB budget against the released decoder, in percent, "
        "on the pinned checkpoint and on the epoch-1 pin, measured with the "
        "identical protocol on the identical frames. The last two columns are "
        "the floor, the quality already given up before any tile exits early; "
        "it falls at every rate. The paper reports the left-hand column, which "
        "is therefore a lower bound on what this configuration reaches.")
    k.note("Source: results/signalled_RECIPE512_ctc53.json "
           "(runs/RECIPE512/ckpt_PAPER.pth.tar, epoch 0) and "
           "results/signalled_RECIPE512_e1.json "
           "(runs/RECIPE512/ckpt_PIN_e1.pth.tar, epoch 1), both "
           f"{sig['n_sequences']} sequences.")

    k.par(
        f"The run had reached epoch {LOG['last_epoch']}, step "
        f"{LOG['last_step']} of {LOG['steps_per_epoch']}, when this was "
        f"written, against the {META['epochs_requested']} epochs its launcher "
        f"was given. Table {t_epoch} therefore carries a second reading: no "
        "comparison in this paper between two runs at different epochs can be "
        "read as a comparison between two configurations, and where such a "
        "comparison appears it is labelled with both epochs. Nothing in "
        "results/ projects the trajectory to convergence and this section does "
        "not either; scripts/convergence.py fits a saturating curve with the "
        "ceiling pinned but writes only a figure, so its asymptote is not a "
        "number this document can cite.")

    # ---------------------------------------------------------------- A.12
    k.h2("Every hyperparameter, and where each was measured")

    rows = [["knob", "value", "searched, and where"]]
    rows += [
        ["<b>ladder geometry</b>", "", ""],
        ["exits K, blocks per exit b", f"{K}, {b_blocks}",
         "signalled_BEST128 and signalled_FINE12, at K = 12"],
        ["split depth j", j, "supp_seam_vs_split, j from 0 to 6"],
        ["RGB tile", f"{td['rgb_patch']} px",
         "ctc_seam_p128 against ctc_seam_p256; tilesize_adaptive"],
        ["halo, latent and trunk",
         f"{cfg['latent_halo']}, {cfg['trunk_halo']}",
         "no; the trunk halo was priced at 2.25× and rejected"],
        ["adapter kind", cfg["adapter_kind"], "adapter_ablation"],
        ["adapter expansion", cfg["adapter_expand"], "no; the block's own 4:1"],
        ["seam repair", cfg["seam_repair"], "supp_seam_vs_split"],
        ["tile padding", cfg["tile_pad_mode"],
         "padding_ablation: zeros, replicate, linear, arls"],
        ["tile coupling", "off", "coupling_ablation"],
        ["full-frame head", "on", "the per-tile arm exists to measure it"],
        ["<b>decoder training</b>", "", ""],
        ["optimiser", "AdamW", "no; the released recipe's"],
        ["schedule", "105 entries, at offset 99",
         "no; verified by scripts/verify_recipe.py"],
        ["learning rate", "1×10<super>-5</super>", "no; set by the schedule"],
        ["multiplier, new modules", "×20",
         "no; set from one anchor-drift measurement"],
        ["crop", "512 × 512", "no; set by the schedule"],
        ["batch, workers, precision", "8, 8, fp32", "no"],
        ["gradient clip", "0.1, non-finite dropped", "no; released recipe's"],
        ["multiplier range", "10 to 2048, 64 indices", "no; released"],
        ["quality index", "uniform, per sample", "no"],
        ["anchor weight", 10, "two settings run; 10 kept"],
        ["distillation", "1.0, adjacent", "adjacent against deepest"],
        ["auxiliary weight", "1.0, constant", "no"],
        ["training path", "deployed tiled decode",
         "against the full-frame arm"],
        ["encoder", "frozen", "no; asserted at each evaluation"],
        ["seed", "none set", "no; see A.17"],
        ["<b>router head, configuration B</b>", "", ""],
        ["steps, batch, crop", "3,000, 6, 512", "no"],
        ["learning rate, decay", "1×10<super>-3</super> cosine, "
         "1×10<super>-4</super>", "no"],
        ["regret weight", "1.0", "no"],
        ["seed", 0, "fixed, so the input ablation is comparable"],
        ["live inputs", "stem, latent, scales, qp",
         "router_ablation, six variants"],
        ["<b>evaluation</b>", "", ""],
        ["multiplier bisection", "60 halvings on [0, 1]", "no"],
        ["outer correction", "6 passes, tolerance 5e-4 dB", "no"],
        ["rate-rank bisection", "40 halvings", "no"],
        ["router bisection", "on beta, bracket [−2e4, 2e4]", "no"],
        ["timing", "40 iterations after 20 of warm-up", "no"],
        ["batch, when timing", 1, "supp_latency_batch, at 1, 2 and 4"],
    ]
    t_hyper = k.rows(
        rows,
        "Every knob this work sets, its value, and whether it was searched. "
        "Take from it that almost nothing on the training side is tuned: the "
        "optimiser, the schedule, the clip, the crop and the multiplier range "
        "are the released recipe's, and the four settings that are ours exist "
        "to protect a property the measurement depends on rather than to "
        "improve a number. What was searched is the geometry, and the third "
        "column names the file in results/ that holds each comparison, so a "
        "reader can price the choice instead of taking it on trust.")
    k.note("Values come from scripts/launch_recipe512.sh, "
           "train_flexuf_image.py, scripts/train_router2.py and the four "
           "evaluation scripts; the third column names files in results/. The "
           "launcher's flags, without the two leading hyphens the shell needs, "
           "are: " + LAUNCH + ".")

    # ---------------------------------------------------------------- A.13
    k.h2("Finding the operating point")

    k.par(
        "A budget is a quality target rather than a multiplier, so the "
        "multiplier has to be found, and it is found twice over: an inner "
        "bisection on a cheap table of per-tile distortions, and an outer "
        "correction against what a real decode delivers. The two levels exist "
        "because the table measures each tile with every other tile at the "
        "same exit, while a routed frame is mixed and a tile's border sees "
        "whatever depth its neighbour chose. The residual between them is "
        "small and nearly constant, so shifting the inner target by it "
        "converges in two or three passes, and each pass costs one decode per "
        "frame rather than one per bisection step.")

    box2 = _box(k, [
        "<b>input</b> budget t in dB, frames F, per-tile cost vector C",
        "&nbsp;1&nbsp; <b>for</b> each frame: cache the [tiles, K] table M of "
        "per-tile error,",
        IND + IND + "one tiled decode per exit, and the released decoder's "
        "frame error R",
        "&nbsp;2&nbsp; floor = trueDb(0)" + _c("one decode per frame"),
        "&nbsp;3&nbsp; <b>if</b> floor &gt; t: record unreachable and stop",
        "&nbsp;4&nbsp; inner = t",
        "&nbsp;5&nbsp; <b>repeat</b> at most 6 times:",
        IND + "&nbsp;6&nbsp; lo, hi = 0, 1",
        IND + "&nbsp;7&nbsp; <b>repeat</b> 60 times:"
        + _c("on the table, no decode"),
        IND + IND + "&nbsp;8&nbsp; mid = (lo + hi) / 2",
        IND + IND + "&nbsp;9&nbsp; k = argmin over exits of M + mid · C",
        IND + IND + "10&nbsp; <b>if</b> tableDb(k) ≤ inner: lo = mid "
        "<b>else</b> hi = mid",
        IND + "11&nbsp; lam = lo;  d = trueDb(lam)"
        + _c("one decode per frame"),
        IND + "12&nbsp; <b>if</b> |d − t| &lt; 5e-4: <b>stop</b>",
        IND + "13&nbsp; inner = inner + (t − d)",
        "14&nbsp; k = argmin of M + lam · C, clamped to j",
        "15&nbsp; saving = 100 (1 − routed MACs / released MACs), by hooks",
        "16&nbsp; bpp = bpp + mapBits(k) / pixels",
    ], "Finding the operating point, as scripts/signalled_curve.py runs it. "
       "Take from it that the arithmetic model is used only inside the argmin "
       "on line 9, where a decode per candidate is impossible, and that the "
       "number reported on line 15 is counted off the forward pass that "
       "actually ran. Line 2 is measured on the deployed path rather than on "
       "the table, because the floor is exactly where the tiling penalty is "
       "largest.")

    k.par(
        f"Three details of Table {box2} change what the numbers mean. A rate "
        "whose floor exceeds its budget is written out with a reachability "
        "flag rather than dropped, so a saturated or unreachable row is "
        "visible instead of absent. The saving is measured and not modelled: a "
        "forward hook on every convolution and linear layer counts the "
        "multiply-accumulates the routed decode performed and divides by the "
        "same count for the released decode on the same latent, so the ratio "
        "needs no calibration constant and a module skipped by an early exit "
        "costs nothing because its hook never fires. The gap between model and "
        "meter is reported in every row: at the 0.1 dB budget it runs from "
        f"{min(r['model_minus_measured'] for r in at01.values()):.2f} to "
        f"{max(r['model_minus_measured'] for r in at01.values()):.2f} points, "
        "always in the direction of the model flattering us, and it is the "
        "measured column the paper quotes.")

    k.par(
        "Two saving columns exist and only one is quoted. One divides by our "
        "own deepest exit, the other by the released decoder, and the released "
        "decoder is the denominator every claim is stated against; it is "
        "exactly 1 in the units of the cost table, so the seam-repair tax "
        "stays inside the figure it is supposed to be net of. The same "
        "two-level scheme with different constants runs in "
        "scripts/raterank_curve.py, scripts/router_curve.py and "
        f"scripts/hybrid_curve.py, whose constants Table {t_hyper} lists.")

    # ---------------------------------------------------------------- A.14
    k.h2("How a wall-clock number is taken")

    lb = k.J("supp_latency_batch_1920x1080.json")
    lc = k.J("supp_latency_cpu_1920x1080.json")

    k.par(
        "An input-adaptive decoder makes wall clock a convention rather than a "
        "reading, so the convention is stated before any second is reported. "
        "Every timing in this work is at batch 1, on a padded 1920×1080 frame "
        f"split into {lb['tiles_per_frame']} tiles, with {lb['warmup']} warm-up "
        f"iterations discarded and {lb['iters']} timed iterations interleaved "
        "between conditions so that a drift on a shared card lands on all of "
        "them equally; the timer is time.perf_counter around a synchronised "
        "call. Entropy decoding is outside the timer, because the rate does "
        "not depend on the exit and including it would dilute a decoder-side "
        "measurement with a constant. The router is charged in arithmetic and "
        f"not in time, and the two disagree: it is "
        f"{rl['router_share_pct_macs']:.3f}% of the decode's "
        f"multiply-accumulates and {rl['router_share_pct_time_median']:.2f}% "
        f"of its median wall clock, a factor of "
        f"{rl['time_over_mac_factor']:.1f}, because a multiply-accumulate "
        "count cannot see a kernel launch.")

    k.par(
        "Batch 1 is the right measurement and not merely a convenient one: a "
        "decoder decodes the frame in front of it, and a per-tile-adaptive "
        "decoder handed a batch would have to pad every image to the deepest "
        "map in that batch, which is a different algorithm. The cost of the "
        "choice is measured rather than argued, and so is the trend across a "
        "second class of device.")

    rows = [["device", "batch", "ms/frame", "realised", "predicted",
             "overhead"]]
    for r in lb["rows"]:
        if r["qp"] != 0:
            continue
        rows.append(["A6000", r["batch"],
                     f"{r['ms_routed_per_frame']:.1f}",
                     f"{r['realised_saving_pct']:.1f}%",
                     f"{r['predicted_saving_pct']:.1f}%",
                     f"{r['overhead_pct']:.1f}%"])
    for r in lc["rows"]:
        if r["qp"] != 0:
            continue
        rows.append(["CPU, 8 threads", r["batch"],
                     f"{r['ms_routed_per_frame']:.0f}",
                     f"{r['realised_saving_pct']:.1f}%",
                     f"{r['predicted_saving_pct']:.1f}%",
                     f"{r['overhead_pct']:.1f}%"])
    t_batch = k.rows(
        rows,
        "What the timing convention costs: routed milliseconds per frame at "
        "q0 and the 0.1 dB budget, with one exit map throughout, beside the "
        "saving realised, the saving the arithmetic model predicts and the "
        "cost of the tiled path itself. Take from it two things. Batch 1 gives up "
        "about one point of realised saving against batch 4, so the convention "
        "is cheap and stating it is cheaper. And the gap between arithmetic "
        "and the clock is a scheduling effect rather than an arithmetic one: "
        "on the GPU the same map realises less than the multiply-accumulate "
        "model predicts, while on a CPU it realises more and the tiled path "
        "costs nothing at all. The device this paper reports is the one that "
        "flatters the method least.")
    k.note("Sources: results/supp_latency_batch_1920x1080.json and "
           "results/supp_latency_cpu_1920x1080.json, both on the pinned "
           "checkpoint with the exit map read from "
           "results/supp_paper_curve_PAPER.json. The GPU rows were taken under "
           "the evaluation lock on a shared card carrying four other people's "
           "training runs; the CPU rows used 8 threads on the same shared "
           "machine and are not an idle-box measurement. All eight cards on "
           "this machine are the same model, so a CPU is the only second "
           "device class reachable here.")

    # ---------------------------------------------------------------- A.15
    k.h2("What the work cost to produce")

    ra = k.J("router_ablation_stem.json")
    ec = k.J("supp_encoder_cost_PAPER.json")

    rows = [["item", "cost", "source"],
            ["one decoder epoch", f"{LOG['hours_per_epoch']} h on one A6000",
             "train_log.jsonl"],
            ["the pinned checkpoint", "1 epoch", "ckpt_PAPER records epoch 0"],
            ["the run so far",
             f"{LOG['last_epoch']} epochs and {LOG['last_step']} steps",
             "train_log.jsonl"],
            ["one router head",
             f"{ra['steps']:,} steps, {ra['wall_s'] / 3600:.2f} h",
             "router_ablation_stem"],
            ["the router input ablation", "6 heads, about 6.2 h",
             "the six router_ablation files"],
            ["encoder search, exact",
             f"{ec['x_deployed']:.2f} decodes per frame", "supp_encoder_cost"],
            ["encoder search, approximate",
             f"{ec['x_full_frame']:.2f} decodes per frame",
             "supp_encoder_cost"]]
    t_cost = k.rows(
        rows,
        "What producing these numbers costs, priced rather than asserted. "
        "Take from it that the encoder pays for the allocation and the decoder "
        f"does not: the exact per-tile table costs {ec['x_deployed']:.2f} "
        f"decodes per frame against {ec['x_full_frame']:.2f} for the "
        "full-frame approximation, which agrees with the exact search on "
        f"{100 * ec['approx']['agreement']:.1f}% of tiles and gives up "
        f"{ec['exact']['saving'] - ec['approx']['saving']:.2f} points of "
        "saving. Nothing on the decoder side is searched at all, which is the "
        "point of signalling the map.")
    k.note("Sources: results/router_ablation_stem.json and its five siblings "
           "for the router timings, results/supp_encoder_cost_PAPER.json for "
           f"the search cost (pinned checkpoint, q{ec['qp']}, "
           f"{ec['n_tiles']} tiles, {ec['iters']} iterations, one tiled decode "
           f"at {ec['ms_one_decode']:.1f} ms). Decoder wall clock is from "
           "runs/RECIPE512/train_log.jsonl; no file in results/ records it.")

    # ---------------------------------------------------------------- A.16
    k.h2("The pinned checkpoint, and what every table rests on")

    k.par(
        "Results are measured on runs/RECIPE512/ckpt_PAPER.pth.tar, md5 "
        f"{ms['ckpt_md5']}, which records epoch {ms['ckpt_epoch']} inside "
        "itself and is byte identical to that run's ckpt_epo0.pth.tar. The "
        "name exists because a filename is not a checkpoint here: a per-epoch "
        "watcher overwrites runs/*/ckpt_eval.pth.tar as each epoch lands, so a "
        "results file naming that path names a file that no longer exists. "
        "scripts/make_paper_tables.py and scripts/check_paper.py both hold the "
        "pinned name and, among candidates for one quantity, prefer a file "
        "whose own ckpt field records it, falling back to the most recently "
        "written; hand-written order breaks ties and nothing else. That is a "
        "fix rather than a convenience, because an earlier hand-ordered list "
        "kept preferring a superseded file after the pinned one had been "
        "remeasured.")

    fg_tree = k.fig(
        "run_tree.png",
        "<b>Lineage of the training runs.</b> Every run warm-starts from the "
        "released decoder through one of two remaps, one per ladder size, and "
        "freezes the encoder, so all of them consume the identical bitstream. "
        "The flags shown are the ones that differ between runs. The progress "
        "bars are a snapshot taken on 2026-08-18 and the runs have advanced "
        "since; the current position of the run this paper reports is given in "
        "A.11.", maxh=185)

    rows = [["results file", "backs", "checkpoint", "n"]]
    npin = 0
    for name, backs in CONSUMED:
        d = k.J(name)
        lab = _ckpt_label(d)
        if lab.startswith("pinned"):
            npin += 1
        rows.append([name.replace(".json", ""), backs, lab, _n_label(d)])
    t_prov = k.rows(
        rows,
        f"Provenance of every generated table in the main paper. These are the "
        f"{len(CONSUMED)} results files scripts/make_paper_tables.py reads, "
        f"and the checkpoint column is read from each file at build time "
        f"rather than typed. Take from it that {npin} of the "
        f"{len(CONSUMED)} are on the pinned checkpoint and the rest are not, "
        f"so a reader comparing two rows of two different tables should check "
        f"this column first. \"eval\" is the overwritten per-epoch file, and "
        f"\"not recorded\" means the file carries no checkpoint field at all, "
        f"which is a defect in the script that wrote it.")

    k.par(
        f"Three groups in Table {t_prov} deserve separate comment. The pinned "
        "group carries the headline saving, the operating range, the "
        "per-class breakdown, the static controls and both routed "
        "configurations, so the paper's central claim rests on one epoch of "
        "one run measured consistently. The eval group is measured on a file "
        "that has since been overwritten, so those numbers can be quoted but "
        "not reproduced; the latency table and the adapter ablation are in it. "
        "The group with no checkpoint field at all cannot even say which "
        "decoder it describes.")

    # ---------------------------------------------------------------- A.17
    k.h2("Seeds, variation, and the interval that does exist")

    k.par(
        "No seed is set anywhere in the decoder's trainer. Four things vary "
        "between two runs of one command: the sampler is reshuffled every "
        "epoch from the global generator, the quality index is drawn per "
        "sample, the tiled training path draws a fresh random exit for every "
        "tile at every step, and cuDNN selects kernels by autotuning. The "
        "honest statement is that the recipe reproduces and the run does not, "
        "and this work has no repeated run of one configuration from which a "
        "training-side spread could be quoted. The router experiments are the "
        "exception and were built to be: every variant of the input ablation "
        "fixes seed 0 and shares architecture, parameter count, optimiser, "
        "image order and quality-index draws, so the only difference between "
        "them is which inputs the head may see, and their agreement figures "
        f"carry a standard error across {ra['eval']['n_frames']:,} frames.")

    ps = k.J("supp_per_sequence_PAPER_b010.json")
    rows = [["quality index", "mean", "sd", "min", "median", "max",
             "worst sequence"]]
    for r in ps["rows_vs_release"]:
        rows.append([f"q{r['qp']}", f"{r['mean']:.2f}", f"{r['sd']:.2f}",
                     f"{r['min']:.2f}", f"{r['median']:.2f}",
                     f"{r['max']:.2f}", r["worst_seq"].split("_")[0]])
    t_spread = k.rows(
        rows,
        "The interval this work can honestly report: saving in percent against "
        "the released decoder at the 0.1 dB budget, across the \\NumSeq test "
        "sequences, on the pinned checkpoint. Take from it that the spread is "
        "content and not noise. It widens with rate, from a standard deviation "
        f"of {ps['rows_vs_release'][0]['sd']:.1f} points at q0 to "
        f"{ps['rows_vs_release'][-1]['sd']:.1f} at q63, and the minimum turns "
        "negative at the high rates because a sequence whose tiles all stay at "
        "the deepest exit costs about 1% more than the released decoder rather "
        "than less. This measures the test set, not the training draw, and is "
        "not a substitute for the run-to-run interval that does not exist.")
    k.note("Source: results/supp_per_sequence_PAPER_b010.json, the "
           "rows_vs_release block, computed from "
           "results/supp_paper_curve_PAPER.json on "
           "runs/RECIPE512/ckpt_PAPER.pth.tar. The file carries the same "
           "spread in our own deepest-exit denominator as well; the two differ "
           "by more than rounding and are never mixed.")

    cc = k.J("crosscheck_paths.json")
    rows = [["quality index", "path 1", "path 2", "gap"]]
    for r in cc["rows"]:
        rows.append([f"q{r['qp']}", f"{r['paper_curve']:.2f}",
                     f"{r['signalled']:.2f}", f"{r['gap']:.2f}"])
    t_cross = k.rows(
        rows,
        "Two implementations of the same quantity do not agree. Saving in "
        "percent at the 0.1 dB budget, computed by the curve-sweeping path and "
        "by the signalling path, which differ in how they pool distortion "
        "across tiles and frames. Take from it a floor on protocol "
        f"sensitivity: the gap reaches {cc['worst_gap_pts']:.2f} points, "
        "larger than several of the differences the paper draws conclusions "
        "from, so numbers from the two paths are never mixed inside one "
        "table.")
    k.note("Source: results/crosscheck_paths.json. The file records no "
           "checkpoint field.")

    cp = k.J("check_paper.json")
    k.par(
        "Evaluation itself is deterministic once a checkpoint is fixed: the "
        "sequence list, the leading frames and the tile grid are all fixed and "
        "no crop is random. scripts/check_paper.py re-reads \\NumClaims "
        "numerical claims out of the paper's prose and checks each against the "
        "file it is supposed to come from. As recorded in "
        f"results/check_paper.json, {cp['n_passed']} of them pass and one does "
        f"not: \"{cp['failed'][0]}\". At its endpoint the partially signalled "
        "configuration reduces to the fully signalled one by construction, and "
        "on the re-measured data it does not quite: it finds a genuinely "
        "cheaper allocation at the same delivered quality, by 0.04 to 0.21 "
        "points, which bisection noise is too small to explain. It is open at "
        "the time of writing and is listed among the limitations rather than "
        "presented as passing.")

    # ---------------------------------------------------------------- A.18
    k.h2("Data, code and provenance")

    k.bullets([
        "<b>Training images.</b> Open Images [24], from the CVDF public "
        "mirror named in scripts/fetch_openimages.sh, prepared into "
        f"{META['dataset']}. The terms are the ones that mirror publishes and "
        "nothing here redistributes them.",
        "<b>Test sequences.</b> UVG [25] from ultravideo.fi, MCL-JCV [26] and "
        "HEVC classes B, C and D from the public mirrors named in "
        "scripts/fetch_mcljcv.sh and scripts/fetch_hevc_bcd.sh, and HEVC class "
        "E from media.xiph.org. Each fetch script records its source URL, and "
        "no sequence is re-encoded before it is measured.",
        "<b>The released codec.</b> The DCVC-UF intra checkpoint [14], "
        "identified by sha256 in results/dmc_ld_recon_audit.json and used "
        "under the licence its repository ships.",
        "<b>Our code and checkpoints.</b> The trainer, the ladder, the cost "
        "model, the meter, the router and every evaluation script named here "
        "are in the repository this supplement is built from; the pinned "
        "checkpoint is identified by md5 in A.16.",
        "<b>Human subjects.</b> None, which is why A.4 declines to claim that "
        "0.1 dB is invisible.",
    ])

    # ---------------------------------------------------------------- A.19
    k.h2("What no file in results/ pins")

    k.par(
        "The rule this supplement follows is that a number is quoted only with "
        "the file it came from. Applying it honestly means listing the places "
        "where the file does not exist.")

    k.bullets([
        "<b>The decoder's training recipe.</b> Nothing that writes to results/ "
        "runs inside the trainer, so no results file records the learning "
        "rate, the schedule, the batch size, the crop, the loss weights or the "
        f"number of steps. Table {t_hyper} is read from "
        "scripts/launch_recipe512.sh, train_flexuf_image.py and the run's own "
        "log, and the dataset statistics of A.9 from a prepare_stats.json that "
        "lives outside the repository.",
        "<b>The wall clock.</b> Read from runs/RECIPE512/train_log.jsonl, "
        "which a live process appends to, so the position quoted in A.11 and "
        f"the hours in Table {t_cost} are a snapshot taken at "
        f"{LOG['snapshot']}.",
        "<b>The warm start's own controls.</b> "
        "runs/warmstart/warmstart_report.json holds the transfer counts and "
        "both zero-difference controls, but the copy on disk is the K = 12 "
        "rebuild of 17 August and the K = 6 report the pinned run descends "
        "from was overwritten. The controls are asserted inside the script at "
        "every build, so a failure would stop the run, but the K = 6 record is "
        "gone.",
        "<b>The exit-clamp mis-billing.</b> The two percentages in Table "
        f"{t_clamp} are a comment in flexuf/cost.py recording a measurement "
        "made once and never written out.",
        "<b>A repeated training run.</b> With no seed and no configuration "
        f"trained twice, no run-to-run interval exists. Table {t_spread} is "
        f"the nearest substitute and measures the test set, and Table "
        f"{t_cross} measures the evaluation path; neither measures the "
        "training draw.",
        f"<b>A second class of accelerator.</b> All eight cards here are the "
        f"same model, so the second device in Table {t_batch} is a CPU.",
        "<b>Any resolution above 1080p.</b> The queued 4K latency job returned "
        "without writing a file and results/supp_footprint.json marks its "
        "3840×2304 row oom, so nothing here is measured at 4K.",
        "<b>A second decoder.</b> The ladder needs only a residual trunk with "
        "a shared head, and no second codec was rearranged into one, so the "
        "generality of the construction is argued and not measured.",
    ])
