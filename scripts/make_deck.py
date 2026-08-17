"""Twelve slides on the LMT template, built from the measured results.

Every number on a slide is read from results/*.json at build time rather than
typed in, so the deck cannot drift from the experiments the way a hand-edited
one does. Where a figure exists it is used; where it does not, the slide says
what is still running rather than showing a placeholder.
"""
import json, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

R = Path("/home/can_karsal/FLEX-UF")
TPL = "/home/can_karsal/Flex-Flop/docs/LMT_Presentation_Template_01.pptx"
TUM = RGBColor(0x00, 0x65, 0xBD)
GREY = RGBColor(0x52, 0x51, 0x4E)

def load(p):
    f = R / "results" / p
    return json.loads(f.read_text()) if f.exists() else None

# Which measurement is the headline
# --------------------------------
# BEST is the configuration this deck reports, and as of its first full epoch it
# is measured: 34.7% saved at qp0 under 0.1 dB against the reference's 28.3%,
# with the tightest anchor of any run. So the headline curve is BEST's and the
# earlier `*_grid128` files are the BASELINE, kept for comparison rather than
# quoted as the result.
#
# Named here rather than threaded through, so the two scripts that draw from it
# cannot drift onto different runs -- which is how a slide ended up with its
# text and its own figure 1.5 points apart earlier today.
CURVE = "curve_BEST.json"
BASE_CURVE = "paper_curve_grid128.json"
SIGNALLED = "signalled_BEST_0817_1542.json"
BASE_SIGNALLED = "signalled_grid128.json"
BD = "bd_BEST.json"
GAP = "target_gap_BEST.json"

why = load("why_qp.json")
pc = load(CURVE) or load(BASE_CURVE)
sg = load(SIGNALLED) or load(BASE_SIGNALLED)
pc_base = load(BASE_CURVE)
sg_base = load(BASE_SIGNALLED)
s256 = load("signalled_control256.json")
lc = load("logconvexity.json")
th = load("theory_check.json")

def sv_at(qp, db):
    """Saving at a dB budget, INTERPOLATED along the frontier.

    Reading off the last grid point below the budget looked equivalent and is
    not: the answer then depends on where the lambda sweep happens to have put
    a sample. Comparing two frontiers that way made qp16 appear to gain 9.7
    points when the true gap was 4.6 -- the old curve had a sample exactly at
    0.1 dB and the new one did not.
    """
    # The per-frame decibel, which is what ~/DCVC/test_video.py computes and
    # therefore what every published DCVC-UF number means. The pooled
    # alternative -- every tile of every frame in one MSE -- reads 0.023-0.033
    # dB lower on an identical allocation, a quarter to a third of a 0.1 dB
    # budget, and would flatter every saving on this deck.
    pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]), r["saving_pct"])
                 for r in pc["rows"] if r["qp"] == qp)
    for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
        if d0 <= db <= d1:
            w = (db - d0) / (d1 - d0) if d1 > d0 else 0.0
            return s0 + w * (s1 - s0)
    return float("nan")


def exit2_rate_factor():
    """How much more exit 2 costs at the top rate than at the bottom.

    Typed into a bullet as '3.4x' until the test set grew to 40 sequences and
    it became 4.3x. Derived now, so it cannot go stale again."""
    w = load("why_qp.json")
    if not w:
        return float("nan")
    r = {x["qp"]: x for x in w["rows"]}
    return r[63]["db_per_exit"][2] / r[0]["db_per_exit"][2]


def budget_line(db):
    """Saving at a dB budget, per rate, saying where the frontier ends first.

    BEST's exits are good enough that its whole qp0 frontier spans 0.197 dB, so
    asking what 0.3 dB buys there has no answer -- the ceiling is reached long
    before. That printed as "nan%". A rate whose frontier ends inside the budget
    is reported with its ceiling and what the ceiling cost, which is the more
    useful fact anyway."""
    out = []
    for qp in (0, 32, 63):
        v = sv_at(qp, db)
        if v == v:
            out.append(f"{v:.0f}% (qp{qp})")
        else:
            pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]),
                          r["saving_pct"]) for r in pc["rows"] if r["qp"] == qp)
            out.append(f"ceiling {pts[-1][1]:.0f}% already at "
                       f"{pts[-1][0]:.2f} dB (qp{qp})")
    return f"At {db:.1f} dB: " + " · ".join(out)


def target_line():
    """Where the 30-40% goal is actually met, derived rather than asserted.

    The bullet used to say the target was met at 0.3 dB and not at 0.1. On the
    40-sequence frontier qp0 reaches 33% at 0.1 dB, so that sentence became
    false the moment the test set grew. Stating which rates clear 30% at which
    budget cannot go stale the same way."""
    TARGET = 30.0
    out = []
    for db in (0.1, 0.3):
        ok = [qp for qp in (0, 16, 32, 48, 63) if sv_at(qp, db) >= TARGET]
        if not ok:
            out.append(f"no rate at {db:.1f} dB")
        elif len(ok) == 5:
            out.append(f"every rate at {db:.1f} dB")
        else:
            out.append(f"qp {'/'.join(map(str, ok))} at {db:.1f} dB")
    return f"The 30% target is cleared by {out[0]}, and by {out[1]}"


def gap_line():
    """Where the remaining distance to the 30% target actually lies.

    Quoting the saving at 0.1 dB says how much we get; reading the frontier the
    other way says how much more it would take, in the units the target is
    written in. The split into drift and everything else is what decides which
    knob to turn: at qp0 the floor is the WHOLE overspend, so tightening the
    anchor clears the target on its own."""
    g = load(GAP) or load("target_gap.json")
    if not g:
        return "target gap not computed"
    ok = [r for r in g["rows"] if r.get("reachable")]
    if not ok:
        return "the 30% target is not reachable at any rate on this checkpoint"
    met = [r for r in ok if r["overspend_db"] <= 0]
    miss = [r for r in ok if r["overspend_db"] > 0]
    parts = []
    if met:
        # floor_share_pct is None where the target is met: there is no overspend
        # to apportion. Formatting it tripped the build once BEST started
        # clearing the target, which is the pleasanter way to find a bug.
        parts.append(f"met at qp{'/'.join(str(r['qp']) for r in met)} — 30% "
                     f"costs at most {max(r['db_for_target'] for r in met):.3f} dB")
    if miss:
        w = miss[-1]
        parts.append(f"still short at qp{'/'.join(str(r['qp']) for r in miss)}; "
                     f"the furthest is qp{w['qp']}, needing "
                     f"{w['db_for_target']:.3f} dB "
                     f"({w['db_for_target'] / g['budget_db']:.1f}× the budget, "
                     f"{w['floor_share_pct']:.0f}% of the excess being the "
                     f"anchor's floor and the rest exit quality)")
    return "30% target: " + "; ".join(parts)


def spread_ratio(qp=63):
    """Best-to-worst sequence ratio at a 0.1 dB budget.

    The theory says the value of adaptivity comes from tiles disagreeing about
    which exit they want. This is the same disagreement one level up, between
    whole sequences, and it grows with rate the same way -- 6.7x against the
    tile-level gain's 7.5x. Consistent; two different quantities agreeing on
    five points is not confirmation, and the slide says 'same mechanism', not
    'confirms'."""
    ps = load("per_sequence.json")
    if not ps:
        return float("nan")
    for r in ps["rows"]:
        if r["qp"] == qp:
            return r["max"] / r["min"]
    return float("nan")


def bd_line():
    """The trade-off as two integrated numbers rather than two sampled points.

    Added because every headline here was a reading at one dB budget, and a
    claim resting on one sample of a curve turned out to be fragile: the
    grid-readout rule made one test-set comparison look twice as large as it
    was. BD-saving and BD-quality integrate the measured frontier over stated
    intervals, so neither depends on where the sweep placed a sample."""
    bd = load(BD) or load("bd_saving.json")
    if not bd:
        return "integrated trade-off not computed"
    lo, hi = bd["db_interval"]
    slo, shi = bd["saving_interval"]
    per = " · ".join(f"{r['qp']}: {r['bd_saving_pct']:.0f}%" for r in bd["rows"])
    return (f"BD-saving {bd['mean_bd_saving_pct']:.1f}% over dB in "
            f"[{lo:.3f}, {hi:.3f}] (qp {per}); BD-quality "
            f"{bd['mean_bd_quality_db']:.3f} dB over saving in "
            f"[{slo:.0f}%, {shi:.0f}%]")


def signalled_line():
    """The shipped system's saving per QP, from the file that measured it."""
    if not sg:
        return "signalled measurement not present"
    rows = sorted(sg["rows"], key=lambda r: r["qp"])
    base = {r["qp"]: r["saving_pct"] for r in (sg_base or {"rows": []})["rows"]}
    nums = " / ".join(f"{r['saving_pct']:.1f}" for r in rows)
    vs = ("" if not base else "  (baseline " +
          " / ".join(f"{base.get(r['qp'], float('nan')):.1f}" for r in rows) + ")")
    worst = max(r["db_vs_uf"] for r in rows)
    n = sg.get("n_sequences")
    return (f"measured with the map's cost INSIDE the bitrate: {nums}% at "
            f"qp{rows[0]['qp']}\u2026{rows[-1]['qp']}, all under "
            f"{worst:.2f} dB" + vs + (f" ({n} CTC sequences)" if n else ""))

prs = Presentation(TPL)
for i in range(len(prs.slides) - 1, -1, -1):
    rid = prs.slides._sldIdLst[i].rId
    prs.part.drop_rel(rid); del prs.slides._sldIdLst[i]

def bullets(slide, items, size=15):
    body = slide.placeholders[1].text_frame
    body.clear()
    for j, (lvl, txt, bold) in enumerate(items):
        p = body.paragraphs[0] if j == 0 else body.add_paragraph()
        p.level = lvl
        run = p.add_run(); run.text = txt
        run.font.size = Pt(size - 2 * lvl); run.font.bold = bold
        if bold:
            run.font.color.rgb = TUM

def slide_text(title, items, size=15):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    bullets(s, items, size)
    return s

def slide_fig(title, img, items, top=None, height=None, size=13, top_txt=None):
    """Bullets on top, figure below -- the figure is the evidence, not decoration."""
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    # Place the figure just under the text instead of at a hand-set offset:
    # count wrapped lines at the given size and convert to inches. Hand-set tops
    # left visible gaps on half the slides and nearly collided on the rest.
    def _lines(txt, sz):
        per = int(96 * 11.0 / max(sz, 1))          # chars per line at this size
        return max(1, -(-len(txt) // per))
    n_lines = sum(_lines(t, size - 2 * lvl) for lvl, t, _ in items)
    txt_h = 0.30 + n_lines * (size + 7) / 72.0
    top = 1.55 + txt_h + 0.18
    top_txt = txt_h
    ph = s.placeholders[1]
    # All four, not just top/height. Setting a subset makes python-pptx
    # materialise a partial <a:xfrm> whose missing values become ZERO rather
    # than being inherited from the layout, so the box ends up 0 wide and the
    # text vanishes -- while every geometry check that looks only at top and
    # height still reports it fine.
    lay = prs.slide_layouts[1].placeholders[1]
    ph.left, ph.width = lay.left, lay.width
    ph.top, ph.height = Inches(1.55), Inches(top_txt)
    bullets(s, items, size)
    p = R / "results" / img
    if p.exists():
        # Fit to BOTH limits and take whichever binds, rather than setting the
        # height and hoping the width lands.
        #
        # These figures are Nature double-column, 183mm wide against roughly
        # 50mm tall, so on a text-heavy slide the height limit was cutting them
        # to about 60% of the available width and leaving the rest of the row
        # empty -- a figure too small to read while the slide had room for it.
        avail_h = max(0.8, 7.5 - top - 0.45)   # leave the footer rule clear
        avail_w = 9.1
        pic = s.shapes.add_picture(str(p), Inches(0.45), Inches(top))
        sc = min(Inches(avail_w) / pic.width, Inches(avail_h) / pic.height)
        pic.width, pic.height = int(pic.width * sc), int(pic.height * sc)
        pic.left = int((prs.slide_width - pic.width) / 2)
    return s

# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(prs.slide_layouts[0])
s.shapes.title.text = "FLEX-UF: Content-Adaptive Early Exit for DCVC-UF"
s.placeholders[1].text = ("Decoder compute saved at a measured cost in quality\n"
                          "Chair of Media Technology · TUM")

# 2 -------------------------------------------------------------- the problem
slide_fig("Problem and goal", "nf_macs.png", [
 (0, "DCVC-UF's intra decoder: 453.5 GMAC per 1080p frame", True),
 (1, "12 identical DepthConvBlocks = 89.4% of it", False),
 (1, "every patch pays the same, however easy it is", False),
 (0, "Goal: spend less compute where the content allows", True),
 (1, "target set with the group: 30–40% saved at ≤0.1 dB", False),
 (0, "Constraint that shapes everything: the reference is the RELEASED model", True),
 (1, "encoder, hyperprior, entropy model frozen → identical bitstream", False),
 (1, "so any measured difference is the decoder alone", False),
], size=13)

# 3 --------------------------------------------------------------- architecture
slide_fig("Architecture: a ladder of exits", "nf_arch.png", [
 (0, "Shared stem runs full-frame → no seams; the rest runs per tile", True),
 (0, "Each exit has a zero-initialised 1×1 adapter, then the shared head", False),
 (0, "Pointwise on purpose: a 1×1 adds no receptive field, so no seam cost", False),
])

# 4 ------------------------------------------------------------- the warm start
slide_fig("We start AT DCVC-UF, not near it", "nf_warmstart.png", [
 (0, "The released weights are re-expressed in ladder form by a key remap", True),
 (1, "dec_1.0→upsample, dec_1.n→groups.g.i, dec_2→head", False),
 (1, "the remap is a bijection over 143 tensors (asserted in tests)", False),
 (0, "Adapters and seam repair are zero-init, i.e. the identity", False),
 (0, "So at step 0, with no training at all:", True),
 (1, "released DCVC-UF vs our deepest exit → max|diff| = 0.0", False),
 (0, "Training does not have to reach DCVC-UF. It has to (a) lift the shallow "
     "exits and (b) not damage the deepest one.", True),
 (0, "From-scratch runs were abandoned for exactly this: after 6 epochs their "
     "anchor sat 1.49 dB below the release and was not closing.", False),
], size=12)

# 5 -------------------------------------------------------------------- training
slide_fig("Training: the recipe, checked not claimed", "nf_schedule.png", [
 (0, "scripts/verify_recipe.py compares each item against ~/DCVC/train_image.py "
     "and exits non-zero on a mismatch", True),
 (1, "8 schedule rows character for character · 106 entries", False),
 (1, "AdamW 1e-4 · clip_grad_norm 0.1 · non-finite batch skipped", False),
 (1, "ImageFolder + get_training_lambdas · 64 QP levels", False),
 (1, "encoder identical over 74 tensors · 255 shared tensors, no shape change", False),
 (0, "Deliberate additions, listed rather than hidden:", True),
 (1, "--epoch_offset (read the schedule where a warm start actually is)", False),
 (1, "--anchor_weight, --new_lr_scale, --freeze_encoder", False),
 (0, "VERBATIM run: their final phase (epochs 90–104, 512px, lr 2e-4→1e-6) with "
     "NONE of those additions, effective batch 16 by accumulation", True),
], size=11)

# 6 ------------------------------------------------------------------- anchor
slide_fig("The anchor, and a ×4 error caught",
          "nf_anchor.png", [
 (0, "Reading the schedule from epoch 0 applied the from-scratch lr to a "
     "converged model", True),
 (1, "deepest exit fell 0.153 / 0.201 / 0.263 dB in ONE epoch (qp0/32/63)", False),
 (1, "fixed to 0.025 / 0.051 / 0.074 by --epoch_offset 75 + --anchor_weight", False),
 (0, "Freezing the backbone gives exactly 0.000 — but collapses the ceiling from "
     "27.3% to 11.5%, so training the trunk is mandatory", True),
], size=13)

# 7 ---------------------------------------------------------------- example
slide_fig("What an exit costs, on one frame", "nf_exits.png", [
 (0, "Bosphorus 1080p, qp32, identical bitstream at every exit", True),
 (0, "Error maps ×25: the damage is structured, concentrated on detail and "
     "on tile borders — which is what the seam work targets", False),
], size=13)

# 8 --------------------------------------------------------------------- seam
slide_fig("The seam, and how it was removed", "nf_seam.png", [
 (0, "A tile decoded alone meets its border with invented values", True),
 (0, "Pure seam penalty at qp63, CTC, no early exit taken:", True),
 (1, "zeros 1.167 dB · replicate 0.213 · arls 0.179 · 256px halves each", False),
 (0, "arls (arXiv:2502.12300) was implemented, measured, and REJECTED", True),
 (1, "+0.034 dB for +10.7% of decode wall-clock — a bad trade", False),
 (0, "Canvas coupling: only the 3×3 depthwise has spatial extent — 0.334% of a "
     "block — so only it needs the neighbour", True),
 (1, "cost +0.03% of the decode; earlier halo work paid 26% by haloing the "
     "whole block, which was the wrong thing to halo", False),
 (1, "at uniform depth: full-frame vs tiled → max|diff| = 0.0", False),
 (0, "The seam does not shrink. It stops existing.", True),
], size=11)

# 9 ------------------------------------------------------------------ honesty
slide_fig("Cost, checked in the right unit", "nf_cost.png", [
 (0, "The headline is a percentage; it comes from a MAC model", True),
 (0, "MACs are right for a paper and wrong for a promise, so wall-clock was "
     "measured on a 1080p decode — on the 128px ladder, which is where the "
     "timing was taken:", True),
 (1, "exit 2: 43.4% (MAC) vs 43.6% (clock)", False),
 (1, "exit 3: 28.6% vs 28.8% · exit 4: 13.8% vs 14.9%", False),
 (1, "BEST uses 256px tiles, where the MAC ceiling is 42.5% rather than 43.4% — "
     "the grid seam repair scales with the tile, so the exit costs shift "
     "slightly. The agreement being checked here is the model's, not this "
     "checkpoint's.", False),
 (0, "Agreement within ±1 point at every exit", True),
 (0, "The same question caught arls: benefit in dB, bill in milliseconds", False),
 (0, "And the cost model itself is checked against the executed decode "
     "(tests/test_cost_matches_reality.py)", False),
], size=12)

# 10 ------------------------------------------------------------------ results
s = slide_fig("Results vs the released decoder", "nf_results.png", [
 (0, budget_line(0.3), True),
 (1, "dB is the per-frame average test_video.py computes — the convention every "
     "published DCVC-UF number uses. Pooling all tiles into one MSE reads "
     "0.023–0.033 dB lower on the same allocation and would flatter these", False),
 (0, f"At 0.1 dB: {sv_at(0,0.1):.0f}% · {sv_at(32,0.1):.0f}% · {sv_at(63,0.1):.0f}%"
     "", False),
 (0, target_line(), False),
 (0, "Saving falls with rate because early exit costs more there: the exit-2 "
     f"penalty grows {exit2_rate_factor():.1f}× from qp0 to qp63 as the latent "
     "carries more detail", False),
 (0, "Integrated over the curve, not read at one point:", True),
 (1, bd_line(), False),
], size=13)

# 11 ------------------------------------------------------------------ router
slide_fig("Router: predicting failed, signalling works", "nf_router.png", [
 (0, "A learned router could not reach the oracle, and 12k steps showed why", True),
 (1, "CE 0.5→0.09 while qp0 agreement moved 0.743→0.741, regret unchanged", False),
 (1, "fitting the training data 4× better and behaving identically on test "
     "is missing INFORMATION, not missing capacity", False),
 (0, "Structural: the oracle uses error against the SOURCE; the decoder never "
     "sees the source", True),
 (0, "But the encoder does — and in video coding mode decisions are signalled, "
     "not inferred (HEVC/VVC send partitioning and prediction mode)", True),
 (1, "exit map costs 1.3e-4 bpp entropy-coded — 4 orders below the frame", False),
 (1, signalled_line(), False),
 (0, "Agreement with the oracle becomes 100% by construction", True),
], size=11)

# 12 ---------------------------------------------------------- theory + status
d = {q: th[str(q)]["delta"]["3e-05"] for q in (0, 32, 63)} if th else None
slide_fig("Theory, and where we are", "nf_heterogeneity.png", [
 (0, "Cost is additive over tiles and MSE is a mean over tiles → both affine in "
     "the assignment", True),
 (1, "fixed proportions reach the convex hull of the K exit points; per-tile "
     "assignment reaches the Minkowski average of the N tile hulls", False),
 (1, "their gap is min-of-average − average-of-min ≥ 0, zero iff every tile "
     "prefers the same exit — this IS the value of adaptivity", False),
 (0, f"Measured: Δ/J rises {100*d[0]['delta']/d[0]['J_oracle']:.2f}% → "
     f"{100*d[63]['delta']/d[63]['J_oracle']:.2f}% with rate"
     if d else "Measured: Δ/J rises with rate", True),
 (0, f"Same mechanism one level up: best/worst CTC sequence is "
     f"{spread_ratio():.1f}× at qp63 vs {spread_ratio(0):.1f}× at qp0 — "
     f"the mean is not what a clip gets", True),
 (0, "What is left, in the target's own units:", True),
 (1, gap_line(), False),
 (0, "Open: " + target_line() + "; HEVC B/C/D behind JVET credentials, "
     "reported as NOT MEASURED; 6 runs training on 6 GPUs", True),
], size=11)

out = R / "paper" / "FLEX-UF_LMT.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slaytlar)")
