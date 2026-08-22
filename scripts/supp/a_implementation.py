"""Test settings, implementation and reproducibility.

Section A of the supplement. Everything a competent vision researcher who has
never seen this project needs in order to rebuild it: the released codec the
work starts from and the exact file it is, the test set, the measurement
conventions, every module with its shape and its parameter count, the two
algorithms a reader would re-implement, the training configuration with a
column saying which knob was searched and where, the hardware, the wall clock,
the seeds and the sources of variation.

Where a number can be read out of results/ it is read out of results/, so it
cannot go stale. Three kinds of fact cannot be, and each is typed in the block
below with the file it was typed from: the trainer's own flags, the dataset
preparation statistics and the wall clock. Nothing that writes to results/ runs
inside the trainer, which is why those three are typed; section H lists that gap
with the rest of the open items.

Nothing here opens a checkpoint, allocates on a GPU or writes to results/.
Module shapes and parameter counts come from results/supp_module_shapes.json,
which scripts/module_shapes.py derives from the pinned checkpoint's state dict
on the CPU.
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
    ("router_RECIPE512_b01_e4head.json", "A against B"),
    ("router_RECIPE512_b03_e4head.json", "A against B"),
    ("raterank_RECIPE512_b01.json", "rate rank"),
    ("raterank_RECIPE512_b03.json", "rate rank"),
    ("raterank_RECIPE512_b05.json", "rate rank"),
    ("hybrid_RECIPE512_b01_fixed.json", "partial signalling"),
    ("hybrid_raterank_b01.json", "rate rank, second run"),
    ("hybrid_v3_b01_pin.json", "the retrained head, on the pin"),
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
        return "<i>none</i>"
    name = ck.rsplit("/", 1)[-1].replace(".pth.tar", "")
    run = ck.split("/")[1] if ck.startswith("runs/") else ""
    if name == "ckpt_PAPER":
        return "pinned" + suffix
    return (f"{run}/{name}".replace("ckpt_", "").replace("RECIPE512", "R512")
            + suffix)


def _n_label(d):
    n = d.get("n_sequences")
    f = d.get("frames_per_seq")
    if n and f:
        return f"{n}×{f}"
    if n:
        return f"{n}"
    nf = d.get("n_frames")
    if nf:
        return f"{nf} fr"
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
        "shape and its parameter count, the training configuration with a "
        "column naming where each choice was measured, the hardware, the "
        "seeds and what varies between two runs of one command. Numbers are "
        "not repeated across sections; where a quantity belongs to a later "
        "one, this section names it and stops. One fact governs the rest and "
        "is stated here rather than buried: the checkpoint every headline "
        "number comes from has seen the training set exactly once, and A.10 "
        "reports how much the numbers move with a second pass.")

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

    k.par(
        "There is no traditional-codec invocation in this work and none is "
        "needed. The axis measured here is decoder-side compute against a "
        "fixed learned decoder, on a bitstream that decoder produced, so the "
        "anchor is a checkpoint rather than a command line, and the released "
        "codec's own standing against VTM is published with it. That "
        "checkpoint is identified by content and not by path, because the "
        "released file is a bare state dict carrying no epoch, no optimiser "
        "and no step, so a path plus a hash is the whole of its provenance.")
    k.note(
        f"The anchor is {WARM['release']}, sha256 "
        f"{audit['intra_checkpoint_sha256']}, recorded with that reading in "
        "results/dmc_ld_recon_audit.json.")

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
           "are UVG [23], MCL-JCV [24] and HEVC classes B, C, D and E [7]; the "
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
    k.par(
        "Border padding is an <i>estimator</i> of the unseen neighbour, and "
        "the seam is its error, so the four rules below are four estimators "
        "compared on the same frames with early exit switched off, leaving "
        "tiling as the only difference from a full-frame decode.")
    k.tbl("padding",
          "<b>Border estimators</b>, dB below the released decoder on the "
          "same latent, 256 px tiles, full CTC, with every tile at full "
          "depth. Lower is better. The ordering is not the obvious one. "
          "Replication, a zero-order hold, beats linear extrapolation by a "
          "wide margin, because extrapolating the local gradient past a "
          "boundary amplifies whatever noise sits on that boundary and "
          "assuming local constancy does not: \\SeamLinearHigh dB against "
          "\\SeamReplHigh dB at q63. The fitted per-channel AR(1) rule wins "
          "on quality, reaching \\SeamArlsHigh dB, and loses on cost at "
          "+10.7% of decode wall-clock, which against a 0.1 dB budget does "
          "not close; section H carries that verdict in full.")
    k.note("results/ctc_seam_p256.json, generated into "
           "paper/tables/padding.tex by scripts/make_paper_tables.py. The "
           "shipped configuration is replicate padding.")

    k.par(
        "Tile size is the other half of what tiling costs, and it is free in "
        "arithmetic: a tiled decode's multiply-accumulate count does not "
        "depend on the tile side at all, because every tile runs the same "
        "graph and the tiles partition the same feature map. What the side "
        "buys is granularity for the allocation, and what it costs is seam. "
        "The two sides we have measured are below.")
    k.tbl("tilesize",
          "<b>Tile size, at q63.</b> dB below the released decoder with every "
          "tile at full depth, so this is tiling penalty and nothing else. "
          "Doubling the side roughly halves the penalty, which is what a "
          "perimeter-driven error predicts, and it costs no computation. "
          "Against that, 256 px gives a 1080p frame 40 tiles to allocate "
          "across and 128 px gives it 160; section H reports what happened "
          "when we tried to spend that granularity.")
    k.note("results/ctc_seam_p256.json and results/ctc_seam_p128.json, "
           "generated into paper/tables/tilesize.tex by "
           "scripts/make_paper_tables.py. The two tile sizes come from "
           "separate training runs, which is the confound section H states.")

    k.h2("The measurement conventions")

    oq = k.J("supp_opquality_PAPER.json")
    sig = k.J("signalled_RECIPE512_ctc53.json")
    at01 = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - 0.1) < 1e-9}
    mb_lo = min(r["map_bits"] for r in at01.values())
    mb_hi = max(r["map_bits"] for r in at01.values())
    bpp_hi = max(r["bpp_added"] for r in at01.values())
    tp = oq["tail_by_padding"]["0"]
    pad_some, n_some = tp["some_padding"]["mean_db"], tp["some_padding"]["n"]
    pad_none, n_none = tp["no_padding"]["mean_db"], tp["no_padding"]["n"]

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
        "<b>Decibels.</b> The main paper's protocol box fixes the "
        "convention; what it does not say is that pooling all tiles into one "
        "mean squared error first, which is the other live convention, moves "
        "the answer by about a third of a 0.1 dB budget. Section E measures "
        "that gap. No table here mixes the two and every table says which it "
        "is on.",
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
    k.note(
        "The padding figure and the second metric are both from "
        "results/supp_opquality_PAPER.json on the pinned checkpoint, "
        f"{oq['n_sequences']} sequences at the 0.1 dB operating point: mean "
        f"per-tile penalty {pad_some:.3f} dB over the {n_some:,} tiles "
        f"containing replicated rows against {pad_none:.3f} dB over the "
        f"{n_none:,} that do not, at q0. The map-bit range is from "
        "results/signalled_RECIPE512_ctc53.json, field map_bits.")

    k.par(
        "Two limits belong beside that list. Sweeping a multiplier reaches "
        "only the lower convex hull of the achievable set [26, 27], so a "
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

    rows = [["module", "runs", "in → out", "params", "share"]]
    rows.append(["upsample", "frame", f"256 → {C}, ×2",
                 P(ms["roles"]["upsample"]["params"]),
                 f"{100 * ms['roles']['upsample']['params'] / dec_rel:.1f}"])
    rows.append([f"groups 0 to {j - 1}, stem", "frame",
                 f"{C} → {C}", P(j * ms["roles"]["groups.0"]["params"]),
                 f"{100 * j * ms['roles']['groups.0']['params'] / dec_rel:.1f}"])
    rows.append([f"groups {j} to {K - 1}, routable", "tile",
                 f"{C} → {C}", P((K - j) * ms["roles"]["groups.0"]["params"]),
                 f"{100 * (K - j) * ms['roles']['groups.0']['params'] / dec_rel:.1f}"])
    rows.append(["one DepthConvBlock", "tile", f"{C} → {C}",
                 P(ms["roles"]["one_block"]["params"]),
                 f"{100 * ms['roles']['one_block']['params'] / dec_rel:.1f}"])
    rows.append(["adapters 0 to 3, FFN", "tile, at its exit",
                 f"{C} → {4 * C} → {C}",
                 P(ms["roles"]["adapters.0"]["params"]),
                 f"{100 * ms['roles']['adapters.0']['params'] / dec_rel:.1f}"])
    rows.append(["adapter 4, 1×1", "tile, at its exit", f"{C} → {C}",
                 P(ms["roles"]["adapters.4"]["params"]),
                 f"{100 * ms['roles']['adapters.4']['params'] / dec_rel:.1f}"])
    rows.append(["seam repair, grid", "after stitching",
                 f"{C} → {C}", P(ms["roles"]["seam_repair"]["params"]),
                 f"{100 * ms['roles']['seam_repair']['params'] / dec_rel:.1f}"])
    rows.append(["head, PixelShuffle 8", "frame", f"{C} → 192 → 3",
                 P(ms["roles"]["head"]["params"]),
                 f"{100 * ms['roles']['head']['params'] / dec_rel:.1f}"])
    rows.append(["router head (B)", "tile", "161 → 6",
                 P(rl["params"]),
                 f"{100 * rl['params'] / dec_rel:.1f}"])
    rows.append(["decoder, released", "", "", P(dec_rel), "100.0"])
    rows.append(["decoder, ours", "", "", P(dec_new),
                 f"+{100 * dec_new / dec_rel:.1f}"])
    rows.append(["encoder, frozen", "encode", "",
                 P(ms["params_encoder_side"]),
                 f"{100 * ms['params_encoder_side'] / dec_rel:.0f}"])
    t_shapes = k.rows(
        rows,
        "Where every module sits, what shape it is and what it weighs, with "
        "the last column as a percentage of the released decoder's own "
        "parameters; adapter rows are per adapter and group rows cover the "
        "whole run of groups. Take from it three things. The ladder adds "
        f"{100 * dec_new / dec_rel:.1f}% to the decoder's parameters, almost "
        f"all of it in the four feed-forward adapters. Of that, {P(dead)} "
        f"parameters can never run, because at j = {j} the decoder clamps "
        "every exit to at least 2 and adapters 0 and 1 are unreachable; they "
        "still take gradient from the distillation term, which taps every "
        "exit, and dropping them would forfeit the ability to re-run the same "
        "checkpoint at another split depth. And the frozen encoder is larger "
        "than the whole decoder, which is why an encoder-side search is the "
        "expensive half of this system and a decoder-side one is not.")
    k.note(
        "Source: results/supp_module_shapes.json, derived by "
        "scripts/module_shapes.py from the state dict of "
        "runs/RECIPE512/ckpt_PAPER.pth.tar (md5 "
        f"{ms['ckpt_md5'][:16]}) on the CPU. Parameter counts are tensor "
        "sizes read straight from the state dict. The same file also carries "
        "each module's multiply-accumulates per feature pixel, and the three "
        "aggregate shares that fall out of them, 8.16% for the upsample, "
        "89.44% for the twelve-block trunk and 2.40% for the head, reproduce "
        "the forward-hook audit of results/mac_audit.json to four decimals. "
        "Section B prices the decode module by module and derives the per-exit "
        "cost; none of that is repeated here.")

    k.par(
        f"One initialisation detail behind the seam-repair row of Table "
        f"{t_shapes} is load-bearing and is not visible in the arithmetic. "
        "The pass multiplies its correction by a "
        f"learned gate indexed by position within a tile, a "
        f"{td['feature_patch']}×{td['feature_patch']} array of scalars shared "
        f"across all {C} channels, and that gate starts as a decaying function "
        "of the distance to the nearest tile edge. At step zero the module "
        "therefore already knows where the seams are and training refines "
        "rather than discovers, while its output convolution is still zero, so "
        "the whole module is exactly the identity and the warm-start controls "
        "of A.9 still hold.")

    fg_pipe = k.fig(
        "pipeline_detail.png",
        "<b>The three pieces a reader has to re-implement.</b> <b>a</b>, "
        "patchify is a pure reshape and costs nothing, and it is where both "
        "the seam and every operation of saving come from. <b>b</b>, one exit "
        "group is two DepthConvBlocks, whose only operator with spatial extent "
        "is a single 3×3 depthwise worth 0.33% of the block (9C of the block's "
        "7C² + 9C, results/mac_audit.json), so an early exit "
        "gives up almost purely pointwise capacity. <b>c</b>, the two adapters "
        "that replace it, both residual and both zero-initialised. Shapes are "
        "those of a 1080p frame padded to 2048×1280 and split into 40 tiles.",
        maxh=230)

    # ---------------------------------------------------------------- A.6
    k.fig("adapters.png",
          "<b>Inside an exit adapter.</b> <b>a</b>, one DepthConvBlock and the "
          "two adapters, drawn to scale: bar length is a share of the block, "
          "bar height the channel width written. Neither adapter contains the "
          "block's only 3\u00d73 (the hairline rule), so neither adds "
          "receptive field or seam penalty, which is the design constraint "
          "the seam measurement imposes. <b>b</b>, blocks skipped per exit, "
          "coloured by adapter; the rule switches to the FFN at four skipped "
          "blocks. Lengths are counted with hooks off the modules themselves.")

    k.par(
        "What the adapters are worth is measured rather than argued, and the "
        "measurement is available because the identity is exactly what they "
        "were initialised to. Setting each adapter back to that identity "
        "leaves the ladder otherwise untouched, so the difference is what "
        "training put into them.")
    k.tbl("adapters_ablation",
          "<b>What the adapters are worth.</b> dB below the released decoder "
          "with every tile at that exit, with the trained adapters and with "
          "each set back to the identity it was initialised to. The deepest "
          "exit has no adapter by construction and is the control, and it "
          "moves by exactly zero. Without the adapters the shallowest exit "
          "costs \\AdapterNoneHigh dB at q63, forty-four times the working "
          "budget, and the adapter buys \\AdapterGainHigh dB of that back.")
    k.note("results/adapter_ablation.json, on runs/RECIPE512/ckpt_PAPER.pth.tar, "
           "generated into "
           "paper/tables/adapters_ablation.tex by "
           "scripts/make_paper_tables.py.")

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
        f"exit {j}'s adapter. Four separate places have to apply that same "
        "clamp, and getting any of them wrong is silent rather than loud, "
        "which is why they are listed rather than described.")

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
        "The exit clamp, and the four places that have to repeat it. A clamp "
        "in the decoder without the matching clamp in the cost model bills a "
        "shallow map for compute it never ran, which is a saving no meter "
        "agrees with. The mask in the router is −inf and not a large negative "
        "constant, because nothing in a cross-entropy penalises a common "
        "offset added to every logit, so a finite mask can stop being the "
        "smallest entry in its row.")
    k.note("Source: the four files named in the first column, under flexuf/.")

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
        "Training uses Open Images [22], fetched from the CVDF public mirror "
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
        "crop holds one. A single tile has no border against a neighbour, so "
        "patched training at the smaller crop would train a configuration "
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
        f"The recipe, laid out in Figure {fg_train}, is Microsoft's own, "
        "applied at the point in their schedule "
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
        "deepest-teacher form [15], since a large gap between student and "
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
    _fl0 = [at01[q]["floor_db"] for q in qps]
    _fl1 = [a1[q]["floor_db"] for q in qps]

    k.par(
        f"The pinned checkpoint is the first epoch boundary of a run whose "
        f"launcher was given {META['epochs_requested']} epochs. Section H "
        f"tabulates what the later checkpoints of this run and of its sibling "
        f"measure at the same budget on the same frames; the one figure that "
        f"belongs here is the floor, the quality given up before any tile "
        f"exits early, which falls at every rate between the two epoch pins: "
        f"it spans {min(_fl0):.4f} to {max(_fl0):.4f} dB at epoch 0 and "
        f"{min(_fl1):.4f} to {max(_fl1):.4f} at epoch 1. No comparison in "
        "this paper "
        "between two runs at different epochs can be read as a comparison "
        "between two configurations, and where such a comparison appears it "
        "is labelled with both epochs. Nothing here extrapolates to a "
        "converged run.")
    k.note("Floors from results/signalled_RECIPE512_ctc53.json "
           "(runs/RECIPE512/ckpt_PAPER.pth.tar, epoch 0) and "
           "results/signalled_RECIPE512_e1.json "
           "(runs/RECIPE512/ckpt_PIN_e1.pth.tar, epoch 1), both "
           f"{sig['n_sequences']} sequences at the 0.1 dB budget.")

    # ---------------------------------------------------------------- A.12
    k.eq(r"\mathcal{L} = \mathcal{L}_{\mathrm{RD}} + w_{a}\,"
         r"\mathcal{L}_{\mathrm{anchor}} + w_{d}\,"
         r"\mathcal{L}_{\mathrm{distill}}.")

    k.par(
        "The distillation term is written in feature space and not in pixels "
        "because the head is a fixed map from feature to RGB, so matching the "
        "deeper feature is the stronger constraint, with " + str(C) + " dense "
        "channels of target instead of three. Each exit imitates its "
        "neighbour, exit k following exit k+1, rather than the deepest exit.")

    k.par(
        "One choice in the recipe matters more than the weights on those "
        "three terms. The adapters are trained <i>through the tiled decode "
        "path they are deployed in</i>, which is what the train_patched flag "
        "of the launcher above selects. Training them full frame and tiling "
        "only at inference loses 0.14 to 0.24 dB; training through the "
        "deployed path gains 0.51 to 0.90 dB, so the sign of the effect "
        "flips. A ladder that has never seen a seam does not know it has to "
        "compensate for one.")
    k.note("Those two ranges are read from the run log recorded in "
           "DECISIONS.md for the warm-start and train_patched comparisons; no "
           "file in results/ holds them.")

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
        ["seed", "none set", "no; see A.15"],
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
        ["timing", "40 iterations after 20 of warm-up",
         "protocol in section B"],
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

    k.par(
        "Section G writes that search out line by line, with the shape each "
        "step returns. Three details of it change what the numbers mean. A "
        "rate whose floor exceeds its budget is written out with a "
        "reachability "
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

    # ---------------------------------------------------------------- A.13
    k.h2("What the work cost to produce")

    ra = k.J("router_ablation_stem_e4.json", "router_ablation_stem.json")
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
            ["encoder search, per frame",
             f"{ec['x_deployed']:.2f} decodes", "supp_encoder_cost"],
            ["decoder search, per frame", "none", "the map is signalled"]]
    t_cost = k.rows(
        rows,
        "What producing these numbers costs, priced rather than asserted. "
        "Take from it that the decoder is cheap to build and cheap to run, and "
        "that the whole of the allocation cost falls on the encoder, which "
        "already has the source and already runs an analysis pass. Sections B "
        "and F price the encoder-side search and the router heads against what "
        "they buy; this table only says what the work took.")
    k.note("Sources: results/router_ablation_stem.json for one head's "
           "wall_s field, results/supp_encoder_cost_PAPER.json for the search "
           f"cost (pinned checkpoint, q{ec['qp']}, {ec['n_tiles']} tiles, "
           f"{ec['iters']} iterations, one tiled decode at "
           f"{ec['ms_one_decode']:.1f} ms). Decoder wall clock is from "
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
        "written; hand-written order breaks ties and nothing else.")

    k.par(
        "Every run named in this document warm-starts from the released "
        "decoder through one of two key remaps, one per ladder size, and "
        "freezes the encoder, so all of them consume the identical bitstream "
        "and differ only in the ladder flags of A.11. They are compared only "
        "at labelled epochs, for the reason section H measures.")

    rows = [["results file", "backs", "checkpoint", "seq×fr"]]
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
        f"this column first. R512 abbreviates the run RECIPE512, \"eval\" "
        f"is a per-epoch file the watcher overwrites in place, and \"none\" "
        f"means the file records no checkpoint field.")

    k.par(
        f"The pinned group of Table {t_prov} carries the headline saving, the "
        "operating range, the per-class breakdown, the static controls and "
        "both routed configurations, so the paper's central claim rests on "
        "one epoch of one run measured consistently. Section H states what "
        "the rest of the column costs in reproducibility.")

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
    _failed = cp.get("failed") or []
    _tail = (
        f" One does not: \"{_failed[0]}\", which is listed among the "
        "limitations of section H rather than presented as passing."
        if _failed else " All of them pass.")
    k.par(
        "Evaluation itself is deterministic once a checkpoint is fixed: the "
        "sequence list, the leading frames and the tile grid are all fixed and "
        "no crop is random. scripts/check_paper.py re-reads \\NumClaims "
        "numerical claims out of the paper's prose and checks each against the "
        "file it is supposed to come from. As recorded in "
        f"results/check_paper.json, {cp['n_passed']} of them pass." + _tail)

    # ---------------------------------------------------------------- A.18
    k.h2("Data, code and provenance")

    k.bullets([
        "<b>Training images.</b> Open Images [22], from the CVDF public "
        "mirror named in scripts/fetch_openimages.sh. The terms are the ones "
        "that mirror publishes and nothing here redistributes them.",
        "<b>Test sequences.</b> UVG [23] from ultravideo.fi, MCL-JCV [24] and "
        "HEVC classes B, C and D from the public mirrors named in "
        "scripts/fetch_mcljcv.sh and scripts/fetch_hevc_bcd.sh, and HEVC class "
        "E from media.xiph.org. Each fetch script records its source URL, and "
        "no sequence is re-encoded before it is measured.",
        "<b>The released codec.</b> The DCVC-UF intra checkpoint [12], "
        "identified by sha256 in results/dmc_ld_recon_audit.json and used "
        "under the licence its repository ships.",
        "<b>Our code and checkpoints.</b> The trainer, the ladder, the cost "
        "model, the meter, the router and every evaluation script named here "
        "are in the repository this supplement is built from; the pinned "
        "checkpoint is identified by md5 in A.14.",
        "<b>Human subjects.</b> None, which is why A.4 declines to claim that "
        "0.1 dB is invisible.",
    ])
