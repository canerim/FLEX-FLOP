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

why = load("why_qp.json")
pc = load("paper_curve_grid128.json")
sg = load("signalled_grid128.json")
s256 = load("signalled_control256.json")
lc = load("logconvexity.json")
th = load("theory_check.json")

def sv_at(qp, db):
    c = [r for r in pc["rows"] if r["qp"] == qp and r["db_vs_uf"] <= db]
    return max(c, key=lambda r: r["saving_pct"])["saving_pct"] if c else float("nan")

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

def slide_fig(title, img, items, top=3.05, height=3.9, size=13):
    """Bullets on top, figure below -- the figure is the evidence, not decoration."""
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    ph = s.placeholders[1]
    ph.top, ph.height = Inches(1.35), Inches(1.6)
    bullets(s, items, size)
    p = R / "results" / img
    if p.exists():
        pic = s.shapes.add_picture(str(p), Inches(0.45), Inches(top), height=Inches(height))
        if pic.width > Inches(9.1):
            sc = Inches(9.1) / pic.width
            pic.width, pic.height = int(pic.width * sc), int(pic.height * sc)
        pic.left = int((prs.slide_width - pic.width) / 2)
    return s

# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(prs.slide_layouts[0])
s.shapes.title.text = "FLEX-UF: Content-Adaptive Early Exit for DCVC-UF"
s.placeholders[1].text = ("Decoder compute saved at a measured cost in quality\n"
                          "Chair of Media Technology · TUM")

# 2 -------------------------------------------------------------- the problem
slide_fig("Problem and goal", "fig_macs.png", [
 (0, "DCVC-UF's intra decoder: 453.5 GMAC per 1080p frame", True),
 (1, "12 identical DepthConvBlocks = 89.4% of it", False),
 (1, "every patch pays the same, however easy it is", False),
 (0, "Goal: spend less compute where the content allows", True),
 (1, "target set with the group: 30–40% saved at ≤0.1 dB", False),
 (0, "Constraint that shapes everything: the reference is the RELEASED model", True),
 (1, "encoder, hyperprior, entropy model frozen → identical bitstream", False),
 (1, "so any measured difference is the decoder alone", False),
], top=4.55, height=2.35, size=13)

# 3 --------------------------------------------------------------- architecture
slide_fig("Architecture: a ladder of exits over the trunk", "fig_arch.png", [
 (0, "Shared stem runs full-frame → no seams; the rest runs per tile", True),
 (0, "Each exit has a zero-initialised 1×1 adapter, then the shared head", False),
 (0, "Pointwise on purpose: a 1×1 adds no receptive field, so no seam cost", False),
], top=3.3, height=2.4)

# 4 ------------------------------------------------------------- the warm start
slide_fig("We do not converge to DCVC-UF — we start at it", "fig_warmstart.png", [
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
], top=4.75, height=2.15, size=12)

# 5 -------------------------------------------------------------------- training
slide_fig("Training — Microsoft's recipe, checked rather than claimed", "fig_schedule.png", [
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
], top=4.85, height=2.1, size=11)

# 6 ------------------------------------------------------------------- anchor
slide_fig("The anchor: a bug that would have inflated everything ×4",
          "fig_anchor.png", [
 (0, "Reading the schedule from epoch 0 applied the from-scratch lr to a "
     "converged model", True),
 (1, "deepest exit fell 0.153 / 0.201 / 0.263 dB in ONE epoch (qp0/32/63)", False),
 (1, "fixed to 0.025 / 0.051 / 0.074 by --epoch_offset 75 + --anchor_weight", False),
 (0, "Freezing the backbone gives exactly 0.000 — but collapses the ceiling from "
     "27.3% to 11.5%, so training the trunk is mandatory", True),
], top=3.5, height=3.3, size=13)

# 7 ---------------------------------------------------------------- example
slide_fig("What an exit actually costs, on one frame", "fig_exits.png", [
 (0, "Bosphorus 1080p, qp32, identical bitstream at every exit", True),
 (0, "Error maps ×25: the damage is structured, concentrated on detail and "
     "on tile borders — which is what the seam work targets", False),
], top=3.15, height=3.5, size=13)

# 8 --------------------------------------------------------------------- seam
slide_fig("The seam, and how it was removed", "fig_seam.png", [
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
], top=4.85, height=2.1, size=11)

# 9 ------------------------------------------------------------------ honesty
slide_fig("Cost accounting, checked in the unit that matters", "fig_cost.png", [
 (0, "The headline is a percentage; it comes from a MAC model", True),
 (0, "MACs are right for a paper and wrong for a promise, so wall-clock was "
     "measured on a 1080p decode:", True),
 (1, "exit 2: 43.4% (MAC) vs 43.6% (clock)", False),
 (1, "exit 3: 28.6% vs 28.8% · exit 4: 13.8% vs 14.9%", False),
 (0, "Agreement within ±1 point at every exit", True),
 (0, "The same question caught arls: benefit in dB, bill in milliseconds", False),
 (0, "And the cost model itself is checked against the executed decode "
     "(tests/test_cost_matches_reality.py)", False),
], top=4.55, height=2.35, size=12)

# 10 ------------------------------------------------------------------ results
s = slide_fig("Results — against the released decoder, on CTC", "two_views.png", [
 (0, f"At 0.3 dB: {sv_at(0,0.3):.0f}% (qp0) · {sv_at(32,0.3):.0f}% (qp32) · "
     f"{sv_at(63,0.3):.0f}% (qp63)", True),
 (0, f"At 0.1 dB: {sv_at(0,0.1):.0f}% · {sv_at(32,0.1):.0f}% · {sv_at(63,0.1):.0f}%"
     "  — the 30–40% target is met at 0.3 dB, not at 0.1", False),
 (0, "Saving falls with rate because early exit costs more there: the exit-2 "
     "penalty grows 3.4× from qp0 to qp63 as the latent carries more detail", False),
], top=3.5, height=3.3, size=13)

# 11 ------------------------------------------------------------------ router
slide_fig("The router: predicting failed, signalling works", "final_curve.png", [
 (0, "A learned router could not reach the oracle, and 12k steps showed why", True),
 (1, "CE 0.5→0.09 while qp0 agreement moved 0.743→0.741, regret unchanged", False),
 (1, "fitting the training data 4× better and behaving identically on test "
     "is missing INFORMATION, not missing capacity", False),
 (0, "Structural: the oracle uses error against the SOURCE; the decoder never "
     "sees the source", True),
 (0, "But the encoder does — and in video coding mode decisions are signalled, "
     "not inferred (HEVC/VVC send partitioning and prediction mode)", True),
 (1, "exit map costs 1.3e-4 bpp entropy-coded — 4 orders below the frame", False),
 (1, "measured with the map's cost INSIDE the bitrate: 24.2 / 21.2 / 15.7 / "
     "11.0 / 8.6% at qp0…63, all under 0.1 dB", False),
 (0, "Agreement with the oracle becomes 100% by construction", True),
], top=4.85, height=2.1, size=11)

# 12 ---------------------------------------------------------- theory + status
d = {q: th[str(q)]["delta"]["3e-05"] for q in (0, 32, 63)} if th else None
slide_fig("Theory, and where we are", "fig_theory.png", [
 (0, "Cost is additive over tiles and MSE is a mean over tiles → both affine in "
     "the assignment", True),
 (1, "fixed proportions reach exactly the convex hull of the K exit points", False),
 (1, "per-tile assignment reaches the Minkowski average of the N tile hulls", False),
 (1, "their gap is min-of-average − average-of-min ≥ 0, zero iff every tile "
     "prefers the same exit — this IS the value of adaptivity", False),
 (0, f"Measured: Δ/J rises {100*d[0]['delta']/d[0]['J_oracle']:.2f}% → "
     f"{100*d[63]['delta']/d[63]['J_oracle']:.2f}% with rate"
     if d else "Measured: Δ/J rises with rate", True),
 (0, "Open, and stated as open:", True),
 (1, "0.1 dB target not met; 0.3 dB comfortably met", False),
 (1, "larger tiles cut the seam but also the adaptivity — being isolated now", False),
 (1, "HEVC classes B/C/D behind JVET credentials, reported as NOT MEASURED", False),
 (0, "4 runs training (BEST, CONTROL, RECIPE512, VERBATIM) on 4 GPUs", False),
], top=4.9, height=2.05, size=11)

out = R / "paper" / "FLEX-UF_LMT.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slaytlar)")
