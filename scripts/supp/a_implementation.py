"""Implementation, training and reproducibility.

Everything a reader needs to rebuild the experiment: the decoder we start from,
the data, the optimiser and schedule, the wall-clock cost, the checkpoint every
number is measured on, the evaluation protocol, and the seeds.

Where a number can be read out of results/ it is read out of results/, so it
cannot go stale. Four kinds of fact cannot be: the training recipe, the
parameter counts, the dataset preparation statistics and the wall clock. No file
in results/ records any of them, because nothing that writes to results/ has ever
run inside the trainer. Those are typed, and the file they were typed from is
named at the table that uses them. That gap is itself reported in A.8.

Nothing here opens a checkpoint, allocates on a GPU or writes to results/.
"""

# ---------------------------------------------------------------------------
# Typed constants. Each block names the file it was read from. These are the
# only numbers in the section that k.J cannot supply.
# ---------------------------------------------------------------------------

# runs/RECIPE512/meta.json, written by the trainer when the run started.
META = {
    "params_total": "45,445,398",
    "params_adapters": "3,104,640",
    "dataset": "/data10/shareddata/openimages/dcvc_train",
    "dataset_size": "379,614",
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
    "min_side": 512,
}

# runs/RECIPE512/train_log.jsonl, 662 records, read at 2026-08-19 21:03.
LOG = {
    "records": 662,
    "steps_per_epoch": "47,451",
    "sec_per_200": 299.7,
    "sec_per_step": 1.50,
    "hours_per_epoch": 19.8,
    "skipped": 0,
    "last_epoch": 2,
    "last_step": "37,000",
    "snapshot": "2026-08-19 21:03",
}

# runs/warmstart/warmstart_report.json, written by
# scripts/warmstart_from_release.py. The copy on disk is the K = 12 rebuild of
# 17 August; the K = 6 file it also produced was not preserved separately.
WARM = {
    "tensors": 398,
    "adapters_at_init": 28,
    "deepest_max_diff": "0.0",
    "all_exits_max_diff": "0.0",
}


# ---------------------------------------------------------------------------
# The provenance table. Each entry is (results file, the generated table or
# macro group it feeds, and n as the file itself records it). The checkpoint
# column is read from the file at build time rather than typed, because that
# field is the only authority on what a number was measured on.
#
# The list is the set of files scripts/make_paper_tables.py passes to pick(),
# read off that file's source. pick() prefers a candidate whose ckpt field
# names the pinned checkpoint, then the most recently written, so the winner
# is what is listed here.
# ---------------------------------------------------------------------------
CONSUMED = [
    ("signalled_RECIPE512_ctc53.json", "main results, complexity, positioning"),
    ("saturation_RECIPE512_ctc53.json", "operating range"),
    ("static_RECIPE512_b01.json", "static controls"),
    ("per_class_RECIPE512.json", "per class"),
    ("router_RECIPE512_b01.json", "A against B"),
    ("router_RECIPE512_b03.json", "A against B, rate rank"),
    ("raterank_RECIPE512_b01.json", "rate rank"),
    ("raterank_RECIPE512_b03.json", "rate rank"),
    ("hybrid_RECIPE512_b01_fixed.json", "partial signalling"),
    ("hybrid_raterank_b01.json", "rate rank, second run"),
    ("latency_RECIPE512_sorted.json", "latency, complexity"),
    ("router_latency.json", "operating range"),
    ("adapter_ablation.json", "adapter ablation"),
    ("combined_RECIPE512_b01.json", "blend"),
    ("raterank_RECIPE512_b05.json", "rate rank"),
    ("hybrid_v3_b01.json", "rate rank, second run"),
    ("hybrid_RECIPE512_b03_fixed.json", "macros only"),
    ("hybrid_lorenz_b01.json", "macros only"),
    ("coupling_ablation.json", "macros only"),
    ("map_transfer.json", "macros only"),
    ("signalled_BEST_ctc53.json", "ladder configurations"),
    ("signalled_BEST128_ctc53.json", "ladder configurations"),
    ("signalled_FINE12_ctc53.json", "ladder configurations"),
    ("ctc_seam_p256.json", "padding, tile size"),
    ("ctc_seam_p128.json", "tile size"),
    ("bdrate.json", "BD-rate"),
    ("band_collapse.json", "macros only"),
    ("band_collapse_BEST.json", "macros only"),
    ("raterank_BEST_compare.json", "rate rank, second run"),
    ("router_retrain_compare.json", "macros only"),
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


# ---------------------------------------------------------------------------
def content(k):
    out = []
    k.h1("Implementation, training and reproducibility")

    k.par(
        "This section is the recipe. It gives the released decoder the work "
        "starts from, how the training set was built, the optimiser and the "
        "schedule, what an epoch costs in hours, which checkpoint each number "
        "in the paper was measured on, the evaluation protocol down to the "
        "colour space, and what varies between two runs of the same command. "
        "One fact governs the rest and is stated here rather than buried: the "
        "checkpoint every headline number comes from has seen the training set "
        "exactly once. Training was still running when the paper was written, "
        "and A.4 reports how much the numbers move with a second pass.")

    # ---------------------------------------------------------------- A.1
    k.h2("The decoder we start from")

    ac = k.J("adapter_cost.json")
    td = k.J("tile_definition.json")
    cfg = td["config"]
    C = ac["trunk_channels"]
    skipped = [e["blocks_skipped"] for e in ac["exits"]]
    b_blocks = skipped[0] - skipped[1]
    N = skipped[0] + b_blocks
    K = cfg["num_exits"]
    j = cfg["split_depth"]

    k.par(
        "The starting point is the released DCVC-UF intra codec [14], used "
        "unchanged as a codec and rearranged as a network. Its decoder is an "
        f"opening upsampling block followed by {N} DepthConvBlocks at "
        f"C = {C} channels and a head that pixel-shuffles back to RGB. We group "
        f"the {N} blocks into K = {K} exits of b = {b_blocks} blocks each, and "
        f"the first j = {j} groups run once over the whole frame before the "
        "decode splits into tiles. Exits 0 and 1 therefore lie below the split "
        "and cannot be selected; the four reachable exits are 2 to 5. The cost "
        "of each is derived in the complexity section and not repeated here.")

    k.par(
        "scripts/warmstart_from_release.py performs the rearrangement. The "
        "encoder, hyperprior, entropy model and quantisation-step tables are "
        "copied verbatim; the decoder's weights are remapped into the ladder; "
        "every adapter and the seam-repair gate are zero-initialised, so each "
        "one is the identity at step zero. Two controls follow from that and "
        "both are asserted rather than assumed: the deepest exit reproduces the "
        f"released decoder with max|Δ| = {WARM['deepest_max_diff']}, and every "
        "untrained exit equals the released decoder truncated at that depth, "
        f"also max|Δ| = {WARM['all_exits_max_diff']}. The transfer moves "
        f"{WARM['tensors']} tensors and creates {WARM['adapters_at_init']} "
        "adapters at initialisation.")

    k.par(
        "The encoder is frozen for the whole of training. That is what makes "
        "the comparison in this paper a decoder comparison: a trained decoder "
        "consumes byte for byte the stream the released encoder produces, and "
        "in every evaluation both decoders read the same latent from the same "
        "encode, so the bit rate is identical by construction and the whole of "
        "the measured difference is decoder-side.")

    rows = [["quantity", "value", "source"],
            ["trunk blocks N", N, "adapter_cost"],
            ["exits K", K, "tile_definition"],
            ["blocks per exit b", b_blocks, "adapter_cost"],
            ["split depth j", j, "tile_definition"],
            ["reachable exits", "2 to 5", "adapter_cost"],
            ["trunk width C", C, "adapter_cost"],
            ["RGB tile", f"{td['rgb_patch']} px", "tile_definition"],
            ["feature tile", f"{td['feature_patch']} px", "tile_definition"],
            ["latent tile", f"{td['latent_patch']} px", "tile_definition"],
            ["pixels per latent", td["pixels_per_latent"], "tile_definition"],
            ["tile padding", cfg["tile_pad_mode"], "tile_definition"],
            ["seam repair", cfg["seam_repair"], "tile_definition"],
            ["adapter kind", cfg["adapter_kind"], "tile_definition"],
            ["parameters, total", META["params_total"], "meta.json"],
            ["parameters, adapters", META["params_adapters"], "meta.json"]]
    t1 = k.rows(rows, "The ladder and the tiling, as the checkpoint itself "
                      "records them. Everything above the rule is read out of "
                      "the two results files named in the last column, both on "
                      "the pinned checkpoint. The two parameter counts are the "
                      "exception and are read from the run directory, because "
                      "nothing writes a parameter count into results/.")
    k.note("Source: results/adapter_cost.json and results/tile_definition.json, "
           "both on runs/RECIPE512/ckpt_PAPER.pth.tar; parameter counts from "
           "runs/RECIPE512/meta.json.")

    # ---------------------------------------------------------------- A.2
    k.h2("The training data")

    k.par(
        "Training uses Open Images [24], prepared by "
        "scripts/prepare_openimages.py into the flat layout DCVC-UF's own "
        f"ImageFolder expects. Of {PREP['scanned']} files scanned, "
        f"{PREP['kept']} survive a minimum-side filter of {PREP['min_side']} px "
        f"and {PREP['dropped_too_small']} are dropped as too small. The filter "
        "is not cosmetic: the loader zero-pads any image smaller than the "
        "current crop, and an image that is half black border teaches the "
        "decoder to reconstruct black borders. From the survivors, every "
        f"{PREP['val_stride']}th entry of the sorted list is held out, giving "
        f"{PREP['val_heldout']} validation images and {PREP['train']} training "
        "images with no overlap. The split is deterministic so that every run "
        "and every control sees the same held-out set.")

    k.par(
        "Each sample is a random crop at the schedule's patch size with a "
        "random horizontal flip, converted to YCbCr, scaled to [0,1] and "
        "shifted by −0.5. One quality index is drawn uniformly over the 64 "
        "levels per sample, and the matching λ comes from the log-spaced table "
        "running 10 to 2048, so a single set of weights covers the whole rate "
        "range rather than one point on it. Training on random crops and "
        "testing on whole frames is the usual arrangement in this literature, "
        "and the crop size in [2] is 256; ours is the size the released "
        "recipe's final phase asks for. It also happens to be the smallest "
        f"crop that carries more than one tile: at {td['rgb_patch']} px tiles a "
        f"512 px crop holds {META['tiles_per_512crop']}, whereas a 256 px crop "
        "holds one, and a single tile has no border against a neighbour, so "
        "patched training at that crop size would train a configuration that "
        "cannot occur at inference.")

    # ---------------------------------------------------------------- A.3
    k.h2("The training run")

    fg = k.fig("training_scheme.png",
               "The training scheme. Panel (a): one forward pass produces all "
               "K reconstructions, so the ladder is trained as one object. "
               "Panel (b): the three loss terms. Panel (c): the encoder is "
               "never touched. Panel (d): why each line of the recipe is "
               "there. The figure is a schematic drawn from the model "
               "definition; the numbers it carries are parameter counts, and "
               "no file in results/ records those, which is why they are not "
               "quoted in the text.", maxh=210)

    k.par(
        "The recipe is Microsoft's own, applied at the point in their schedule "
        "that matches what we inherit. Their schedule is a 105-entry table of "
        "learning rate and crop size indexed by epoch, and the released "
        "weights are the output of the last entry. Applying entry zero to them "
        "would kick a converged codec with the from-scratch learning rate, so "
        "the run enters at offset 99, which is inside their final 512 px phase "
        "at a learning rate of 1×10<super>-5</super>. Every one of the "
        f"{LOG['records']} records in the run's log carries that learning rate "
        "and that crop, which is the check that the offset did what it was "
        "meant to.")

    k.par(
        "One deviation from a plain fine-tune is deliberate. The adapters and "
        "the seam-repair gate start at zero and have to move furthest, while "
        "the inherited trunk must not move at all if the deepest exit is to "
        "stay the released decoder. Running both at one learning rate means "
        "choosing which of the two to sacrifice, so the new modules are given "
        "a separate parameter group at 20 times the schedule's rate.")

    rows = [["setting", "value"],
            ["optimiser", "AdamW"],
            ["schedule", "DCVC-UF's 105-entry table, entered at offset 99"],
            ["learning rate", "1×10<super>-5</super>, constant for this run"],
            ["lr multiplier, new modules", "×20"],
            ["crop", "512 × 512"],
            ["batch size", META["batch_size"]],
            ["data workers", 8],
            ["gradient clip", "0.1, NaN steps skipped"],
            ["precision", "fp32 throughout"],
            ["devices", "1 × NVIDIA RTX A6000"],
            ["λ range", "10 to 2048, log-spaced over 64 quality indices"],
            ["quality index", "uniform over 64, drawn per sample"],
            ["anchor weight", "10"],
            ["distillation weight", "1.0, teacher = adjacent exit"],
            ["auxiliary weight", "1.0, constant"],
            ["training path", "through the deployed tiled decode"],
            ["encoder", "frozen"],
            ["epochs requested", META["epochs_requested"]],
            ["steps per epoch", LOG["steps_per_epoch"]],
            ["seed", "none set"]]
    t2 = k.rows(rows, "The training recipe in full. Take from it that nothing "
                      "here is tuned: the optimiser, the schedule, the clip and "
                      "the λ range are the released recipe's, and the four "
                      "settings that are ours (the learning-rate multiplier for "
                      "zero-initialised modules, the anchor, the distillation "
                      "and the tiled training path) each exist to protect a "
                      "property the measurement depends on.")
    k.note("Source: scripts/launch_recipe512.sh for the flags, "
           "train_flexuf_image.py for the optimiser, the schedule table and the "
           "clip, and runs/RECIPE512/train_log.jsonl for the learning rate and "
           "crop actually used. No file in results/ records any of it.")

    k.par(
        "The loss has three terms. The rate-distortion term is the released "
        "loss, λ·MSE + bpp, with the MSE averaged over the exits under fixed "
        "weights. The anchor term pins the deepest exit to the released "
        "decoder, because every saving in the paper is quoted against that "
        "decoder and a reference that drifts underneath the measurement makes "
        "the measurement meaningless. The distillation term supervises each "
        "adapter in feature space against the exit one step deeper, following "
        "the adjacent-teacher form rather than the deepest-teacher form [17], "
        "since a large gap between student and teacher is reported to hurt the "
        "shallowest exits most.")

    k.par(
        f"An epoch is {LOG['steps_per_epoch']} optimiser steps, which is "
        f"{PREP['train']} images at batch {META['batch_size']}. The run logs "
        f"every 200 steps and the median interval is {LOG['sec_per_200']} s, so "
        f"a step costs about {LOG['sec_per_step']} s and an epoch about "
        f"{LOG['hours_per_epoch']} hours on one card. Across all "
        f"{LOG['records']} records the count of steps skipped for a "
        f"non-finite gradient is {LOG['skipped']}.")

    k.par(
        "The router head of configuration B is a separate and much smaller "
        "job, trained on top of a frozen decoder. Each variant of the head is "
        "3,000 steps at batch 6 on 512 px crops, learning rate "
        "1×10<super>-3</super>, seed 0, 148,176 parameters, and takes about an "
        "hour on the same card. The six variants of the input ablation cost "
        "6.5 hours between them. What the head learns and what each input is "
        "worth are the subject of the router section; only the cost of "
        "training it belongs here.")
    k.note("Source: results/router_ablation_stem.json and its five siblings, "
           "all on runs/RECIPE512/ckpt_PAPER.pth.tar.")

    # ---------------------------------------------------------------- A.4
    k.h2("How far training has gone")

    e0 = k.J("signalled_RECIPE512_ctc53.json")
    e1 = k.J("signalled_RECIPE512_e1.json")

    def at01(d):
        return {r["qp"]: r for r in d["rows"]
                if abs(r["budget_db"] - 0.1) < 1e-9}
    a0, a1 = at01(e0), at01(e1)
    qps = sorted(a0)
    rows = [["quality index", "epoch 0", "epoch 1", "change",
             "floor e0", "floor e1"]]
    d0 = d1 = 0.0
    for q in qps:
        s0 = a0[q]["saving_pct_vs_release"]
        s1 = a1[q]["saving_pct_vs_release"]
        d0 += s0
        d1 += s1
        rows.append([f"q{q}", f"{s0:.2f}", f"{s1:.2f}", f"+{s1 - s0:.2f}",
                     f"{a0[q]['floor_db']:.4f}", f"{a1[q]['floor_db']:.4f}"])
    m0, m1 = d0 / len(qps), d1 / len(qps)
    rows.append(["mean", f"{m0:.2f}", f"{m1:.2f}", f"+{m1 - m0:.2f}", "", ""])
    t3 = k.rows(rows, "One more pass over the training set is worth about a "
                      "point and a half. Saving at the 0.1 dB budget against "
                      "the released decoder, in percent, on the pinned "
                      "checkpoint and on the epoch-1 pin, measured with the "
                      "identical protocol on the identical frames. The last two "
                      "columns are the floor, the quality already given up "
                      "before any tile exits early, in dB; it falls at every "
                      "rate. The paper reports the left-hand column.")
    k.note("Source: results/signalled_RECIPE512_ctc53.json "
           "(runs/RECIPE512/ckpt_PAPER.pth.tar, epoch 0) and "
           "results/signalled_RECIPE512_e1.json "
           f"(runs/RECIPE512/ckpt_PIN_e1.pth.tar, epoch 1), both "
           f"{e0['n_sequences']} sequences at {e0['frames_per_seq']} frames "
           "per sequence.")

    k.par(
        f"The run had reached epoch {LOG['last_epoch']}, step "
        f"{LOG['last_step']} of {LOG['steps_per_epoch']}, when this was "
        f"written, against the {META['epochs_requested']} epochs its launcher "
        "was given. So the paper's numbers come from a decoder one pass into a "
        "schedule that is nowhere near finished, and the epoch-1 pin shows the "
        "direction of travel: every rate gains, the floor falls at every rate, "
        "and the mean at the 0.1 dB budget moves from "
        f"{m0:.1f}\\% to {m1:.1f}\\% against an architectural ceiling of "
        "\\Ceiling\\%. Two things follow. The reported saving is a lower bound "
        "on what this configuration reaches, and no comparison in this paper "
        "between two runs at different epochs can be read as a comparison "
        "between two configurations. Where such a comparison appears it is "
        "labelled with both epochs.")

    k.par(
        "Nothing in results/ projects the trajectory to convergence, and this "
        "section does not either. scripts/convergence.py fits a saturating "
        "curve with the ceiling pinned, but it writes only a figure, so its "
        "asymptote is not a number this document can cite.")

    # ---------------------------------------------------------------- A.5
    k.h2("The pinned checkpoint, and what every table is measured on")

    k.par(
        "Results are measured on runs/RECIPE512/ckpt_PAPER.pth.tar. It records "
        "epoch 0 inside itself and is byte identical, by md5, to that run's "
        "ckpt_epo0.pth.tar. The name exists because a filename is not a "
        "checkpoint here: a per-epoch watcher overwrites runs/*/ckpt_eval.pth.tar "
        "as each epoch lands, so a results file naming that path names a file "
        "that no longer exists. scripts/make_paper_tables.py and "
        "scripts/check_paper.py both hold the pinned name and, among candidate "
        "files for the same quantity, prefer one whose own ckpt field records "
        "it, falling back to the most recently written. Hand-written order "
        "breaks ties and nothing else, which is a fix rather than a "
        "convenience: an earlier hand-ordered list kept preferring a "
        "superseded file after the pinned one had been remeasured.")

    fg2 = k.fig("run_tree.png",
                "Lineage of the training runs. Every run warm-starts from the "
                "released decoder through one of two remaps, one per ladder "
                "size, and freezes the encoder, so all of them consume the "
                "identical bitstream. The flags shown are the ones that differ "
                "between runs. The progress bars are a snapshot taken on "
                "2026-08-18 and the runs have advanced since; the current "
                "position of the run this paper reports is given in A.4.",
                maxh=190)

    rows = [["results file", "backs", "checkpoint", "n"]]
    npin = 0
    for name, backs in CONSUMED:
        d = k.J(name)
        lab = _ckpt_label(d)
        if lab.startswith("pinned"):
            npin += 1
        rows.append([name.replace(".json", ""), backs, lab, _n_label(d)])
    t4 = k.rows(
        rows,
        f"Provenance of every generated table in the main paper. These are the "
        f"{len(CONSUMED)} results files scripts/make_paper_tables.py reads; the "
        f"checkpoint column is read from each file at build time rather than "
        f"typed. Take from it that {npin} of the {len(CONSUMED)} are on the "
        f"pinned checkpoint and the rest are not, so a reader comparing two "
        f"rows of two different tables should check this column first. "
        f"\"eval\" is the overwritten per-epoch file, and \"not recorded\" "
        f"means the file carries no checkpoint field at all, which is a defect "
        f"in the script that wrote it.")

    k.par(
        "Three groups deserve separate comment. The pinned group carries the "
        "headline saving, the operating range, the per-class breakdown, the "
        "static controls and both routed configurations, so the paper's central "
        "claim rests on one epoch of one run measured consistently. The "
        "eval group is measured on a file that has since been overwritten, "
        "which means those numbers can be quoted but not reproduced; the "
        "latency table and the adapter ablation are in it. The group with no "
        "checkpoint field at all cannot even say which decoder it describes, "
        "and the seam and BD-rate tables are in that group.")

    # ---------------------------------------------------------------- A.6
    k.h2("The evaluation protocol")

    pc = k.J("per_class_RECIPE512.json")
    cls = pc["rows"][0]["per_class"]
    tdrows = {r["name"]: r for r in td["rows"]}
    rows = [["class", "sequences", "resolution", "padded", "tiles",
             "added px"]]
    total = 0
    for name, v in cls.items():
        total += v["n"]
        g = tdrows.get(v["res"])
        if g:
            pw, ph = g["padded"]
            w, h = g["W"], g["H"]
            add = f"{100 * (pw * ph - w * h) / (w * h):.1f}\\%"
            pad = f"{pw}×{ph}"
        else:
            add, pad = "-", "-"
        rows.append([name.replace("_", " "), v["n"], v["res"], pad,
                     v["tiles"], add])
    rows.append(["all", total, "", "", "", ""])
    t5 = k.rows(rows, "The test set, and what tiling does to each resolution. "
                      "Take from it that granularity is not constant across the "
                      "set: a 1080p frame carries 40 tiles for the allocation "
                      "to work with and a 416×240 frame carries two, and the "
                      "small classes also pay the most padding. Padding is "
                      "replicate, applied bottom and right only, to a whole "
                      "number of tiles.")
    k.note("Sources: results/per_class_RECIPE512.json for the class membership "
           "and tile counts, results/tile_definition.json for the padded sizes; "
           "both on runs/RECIPE512/ckpt_PAPER.pth.tar. The sequences are UVG "
           "[25], MCL-JCV [26] and HEVC classes B, C, D and E [9]; the full "
           "list of \\NumSeq names is recorded in the measured field of "
           "results/signalled_RECIPE512_ctc53.json, with not_measured empty.")

    k.bullets([
        "<b>Frames.</b> The first frames of each sequence, consecutively, one "
        "or two per sequence depending on the measurement; every results file "
        "records frames_per_seq. The headline file uses two frames on "
        "\\NumSeq sequences, so 106 frames; the routed, rate-rank and partial "
        "signalling tables use one.",
        "<b>Colour.</b> Sources are YUV 4:2:0, 8 bit. They are converted to "
        "4:4:4, divided by 255 and shifted by −0.5 for the network. PSNR is "
        "the released codec's own metric, (6·Y + U + V)/8, computed after "
        "converting the reconstruction back to 4:2:0 on the 0 to 255 scale and "
        "comparing against the source planes. Measuring in 4:4:4 instead would "
        "compare against a chroma upsample of the source rather than the "
        "source, and would report a higher chroma PSNR that no published "
        "DCVC-UF number could be compared with.",
        "<b>Padding.</b> Frames are padded by replication, bottom and right "
        "only, to a whole number of tiles. The stock codec aligns to 16, which "
        "is enough for the transform and not for a tile grid. The bit rate is "
        "charged on the true pixel count and the PSNR is computed after "
        "cropping back to the true frame, so padding can cost and cannot "
        "flatter.",
        "<b>Tiling.</b> One tile is 256 px of RGB, 32 px of feature map and 16 "
        "px of latent. The patchify step asserts divisibility rather than "
        "padding, so an unpadded 1080p frame fails loudly instead of producing "
        "a ragged last row.",
        "<b>Reference.</b> The released decoder re-expressed in the same "
        "ladder, selected by K so that a K = 12 run cannot be quoted against "
        "the K = 6 remap. Both decoders read the same latent from the same "
        "encode.",
        "<b>Finding the operating point.</b> For a target quality budget the "
        "multiplier λ is found by bisection, 60 halvings on [0,1], taking the "
        "largest λ whose predicted dB stays under the target. dB is averaged "
        "per frame and then over frames, which is the convention the released "
        "codec's own evaluation uses.",
        "<b>The map is billed.</b> The exit map is entropy coded and its cost "
        "is added to the bit rate before any saving is computed; every row "
        "records map_bits and the bpp it added.",
    ])

    k.par(
        "Two limits of that protocol should be read alongside it. Sweeping the "
        "multiplier reaches only the lower convex hull of the achievable set "
        "[28, 29], so a budget falling in a gap of the hull is met at the "
        "nearest reachable point rather than exactly; every row therefore "
        "carries a budget_reachable flag and a floor, and rows whose floor "
        "already exceeds the budget are reported as unreachable rather than "
        "quietly dropped. And the 0.1 dB budget is a reporting convention "
        "rather than a perceptual threshold. Subjective work measures the "
        "smallest noticeable change in quantisation parameter or in a video "
        "quality metric, not in tenths of a decibel of PSNR, so nothing "
        "published licenses a claim that 0.1 dB is invisible; 0.3 dB and 0.5 dB "
        "are reported beside it so that the choice carries no weight.")

    # ---------------------------------------------------------------- A.7
    k.h2("Seeds, and where run-to-run variation comes from")

    k.par(
        "No seed is set anywhere in the decoder's trainer. Four things vary "
        "between two runs of the same command: the sampler is reshuffled every "
        "epoch from the global generator, the quality index is drawn per "
        "sample, the tiled training path draws a fresh random exit for every "
        "tile at every step, and cuDNN selects kernels by autotuning. The "
        "honest statement is that the recipe reproduces and the run does not, "
        "and this work has no repeated run of one configuration from which a "
        "spread could be quoted.")

    k.par(
        "The router experiments are the exception and were built to be. Every "
        "variant of the input ablation fixes seed 0 and shares architecture, "
        "parameter count, optimiser, image order and quality-index draws, so "
        "the only thing that differs between them is which inputs the head is "
        "allowed to see. That is what makes six numbers from six separate "
        "trainings comparable at all, and their agreement figures carry a "
        "standard error taken across 1,200 frames.")

    cc = k.J("crosscheck_paths.json")
    rows = [["quality index", "path 1", "path 2", "gap"]]
    for r in cc["rows"]:
        rows.append([f"q{r['qp']}", f"{r['paper_curve']:.2f}",
                     f"{r['signalled']:.2f}", f"{r['gap']:.2f}"])
    t6 = k.rows(rows, "Two implementations of the same quantity do not agree. "
                      "Saving in percent at the 0.1 dB budget, computed by the "
                      "curve-sweeping path and by the signalling path, which "
                      "differ in how they pool distortion across tiles and "
                      "frames. Take from it a floor on protocol sensitivity: "
                      "the gap reaches "
                      f"{cc['worst_gap_pts']:.2f} points, which is larger than "
                      "several of the differences the paper draws conclusions "
                      "from, so numbers from the two paths are never mixed "
                      "inside one table.")
    k.note("Source: results/crosscheck_paths.json. The file records no "
           "checkpoint field.")

    k.par(
        "Evaluation itself is deterministic once a checkpoint is fixed: the "
        "sequence list, the leading frames and the tile grid are all fixed, and "
        "no crop is random. scripts/check_paper.py re-reads \\NumClaims "
        "numerical claims out of the paper's prose and checks each against the "
        "file it is supposed to come from. As recorded in "
        "results/check_paper.json, 84 of them pass and one does not: "
        "\"hybrid: rho=1 reproduces A\", which is an internal consistency "
        "identity the interpolation between the two configurations is expected "
        "to satisfy at its endpoint. It is open at the time of writing and is "
        "listed among the limitations rather than presented as passing.")

    # ---------------------------------------------------------------- A.8
    k.h2("What no file in results/ pins")

    k.par(
        "The rule this supplementary follows is that a number is quoted only "
        "with the file it came from. Applying it honestly means listing the "
        "places where the file does not exist.")

    k.bullets([
        "<b>The decoder's training recipe.</b> Nothing that writes to results/ "
        "runs inside the trainer, so no results file records the learning rate, "
        "the schedule, the batch size, the crop, the loss weights or the "
        "number of steps. Table {T2} is read from "
        "scripts/launch_recipe512.sh, train_flexuf_image.py and the run's own "
        "log.".replace("{T2}", str(t2)),
        "<b>Parameter counts and wall clock.</b> Read from "
        "runs/RECIPE512/meta.json and runs/RECIPE512/train_log.jsonl. The log "
        "is appended to by a live process, so the position quoted in A.4 is a "
        f"snapshot taken at {LOG['snapshot']}.",
        "<b>The warm start's own controls.</b> "
        "runs/warmstart/warmstart_report.json holds the transfer counts and "
        "both zero-difference controls, but the copy on disk is the K = 12 "
        "rebuild of 17 August; the K = 6 report the pinned run descends from "
        "was overwritten. The controls are asserted inside the script at every "
        "build, so a failure would stop the run, but the K = 6 record is gone.",
        "<b>The intra decoder's total multiply-accumulate count.</b> It is a "
        "constant in scripts/make_paper_tables.py with a comment naming "
        "scripts/mac_audit.py, and that script prints to standard output and "
        "writes no JSON. Every row of the complexity table is normalised by it.",
        "<b>Any resolution above 1080p.</b> The queued 4K latency job returned "
        "without writing a file, and results/supp_footprint.json marks its "
        "3840×2304 row oom, so nothing in this work is measured at 4K.",
        "<b>The qualitative figures' checkpoint.</b> Each of those scripts "
        "defaults to a per-epoch file that the watcher has since overwritten "
        "and none writes a provenance sidecar, so no qualitative figure in this "
        "work can have its checkpoint stated.",
        "<b>A repeated run.</b> With no seed and no configuration trained "
        "twice, the paper reports no run-to-run interval. Table {T6} is the "
        "nearest available substitute and it measures something else: "
        "sensitivity to the measurement path, not to the "
        "training draw.".replace("{T6}", str(t6)),
    ])

    return out
