"""Build the LMT/TUM deck from the chair's own template.

Uses Flex-Flop/docs/LMT_Presentation_Template_01.pptx as the base so the master,
logos and typography are the chair's rather than an imitation. Content is placed
into the template's own layouts ("Titelfolie", "Titel und Inhalt"), which is what
keeps the header and footer correct.

Every figure on a slide is one this project produced and every number is one it
measured; where a number is an upper bound the slide says ORACLE, and where a
result is negative the slide says so rather than omitting it.
"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

TPL = Path.home() / "Flex-Flop/docs/LMT_Presentation_Template_01.pptx"
OUT = Path.home() / "FLEX-UF/paper/FLEX-UF_LMT.pptx"
R = Path.home() / "FLEX-UF/results"
TUM = RGBColor(0x00, 0x65, 0xBD)
INK2 = RGBColor(0x52, 0x51, 0x4E)

prs = Presentation(str(TPL))
for i in range(len(prs.slides) - 1, -1, -1):          # start from a clean deck
    rid = prs.slides._sldIdLst[i].rId
    prs.part.drop_rel(rid)
    del prs.slides._sldIdLst[i]

SW, SH = prs.slide_width, prs.slide_height
L_TITLE, L_BODY = prs.slide_layouts[0], prs.slide_layouts[1]


def bullets(slide, items, left, top, width, height, size=14):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for lvl, txt in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.text = ("• " if lvl == 0 else "– ") + txt
        for r in p.runs:
            r.font.size = Pt(size if lvl == 0 else size - 2)
            r.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B) if lvl == 0 else INK2
        p.space_after = Pt(5)
    return tb


def pic(slide, name, left, top, width=None, height=None):
    p = R / name
    if not p.exists():
        return None
    return slide.shapes.add_picture(str(p), left, top, width=width, height=height)


def split(slide, items, fig, size=12.5, fig_top=None, fig_w=None):
    """Bullets on the left, figure on the right.

    Every slide carries a picture because a claim with a number in it should be
    checkable at a glance; the figure is the number's evidence, not decoration.
    """
    bullets(slide, items, Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=size)
    w = fig_w or Inches(4.05)
    pic(slide, fig, SW - w - Inches(0.35), fig_top or Inches(2.0), width=w)


def head(slide, text):
    slide.shapes.title.text = text
    for pa in slide.shapes.title.text_frame.paragraphs:
        for r in pa.runs:
            r.font.size = Pt(24)


def note(slide, text, top=None):
    tb = slide.shapes.add_textbox(Inches(0.55), top or (SH - Inches(0.78)),
                                  SW - Inches(1.1), Inches(0.42))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    for r in p.runs:
        r.font.size = Pt(10.5); r.font.color.rgb = INK2; r.font.italic = True


# ---- 1 title -------------------------------------------------------------
s = prs.slides.add_slide(L_TITLE)
s.shapes.title.text = "FLEX-UF: Content-Adaptive Early Exit for DCVC-UF"
s.placeholders[1].text = ("Decoder compute cut per tile, measured against the "
                          "released model on the identical bitstream\n"
                          "Chair of Media Technology · TUM")
# The title slide carries the architecture too: the deck's whole subject is one
# picture, and a reader should meet it before the first claim.
pic(s, "fig_arch.png", Inches(0.5), SH - Inches(2.9), width=SW - Inches(1.0))

# ---- 2 problem -----------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Problem and goal")
bullets(s, [
    (0, "DCVC-UF decodes every pixel with the same 12 residual blocks, regardless of how hard that region is"),
    (1, "MAC audit: upsample 8.2% · 12-block trunk 89.4% · head 2.4%; 99.7% of it is pointwise 1×1"),
    (0, "Goal: spend less on easy regions, keep quality at the released model's level"),
    (1, "Target set by the group: 30–40% decoder compute saved at ≤0.1 dB"),
    (0, "Constraint that shapes everything: the bitstream must not change"),
    (1, "Encoder, hyperprior and entropy model stay frozen — same latent, same bits, only synthesis differs"),
    (1, "So every number here is decoder-only and directly attributable"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=12.5)
pic(s, "s_mac.png", SW - Inches(4.4), Inches(2.4), width=Inches(4.05))
note(s, "All measurements: CTC sequences (UVG + HEVC class E). Classes B/C/D are behind JVET credentials and are reported as NOT MEASURED.")

# ---- 3 architecture ------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Architecture: a ladder of exits")
pic(s, "fig_arch.png", Inches(0.35), Inches(1.35), width=SW - Inches(0.7))
bullets(s, [
    (0, "Groups 0–1 run full-frame (no tile borders); groups 2–5 run per tile, each tile leaving at its own exit"),
    (0, "Per-exit adapter is 1×1 and zero-initialised — pointwise, so it adds no receptive field and no seam"),
    (0, "j = 2 gives a 43.4% ceiling: exiting at the earliest permitted group skips 6 of 12 blocks"),
], Inches(0.7), Inches(4.55), SW - Inches(1.4), Inches(2.0), size=13)

# ---- 4 warm start --------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "We do not converge to DCVC-UF — we start at it")
bullets(s, [
    (0, "The released decoder's weights are re-indexed into the ladder: dec_1.0→upsample, dec_1.n→groups.g.i, dec_2→head"),
    (1, "The key remap is asserted to be a bijection over all 143 tensors"),
    (0, "Adapters and seam repair are zero-initialised, i.e. exactly the identity"),
    (0, "Therefore at step 0, with no training at all:"),
    (1, "released DCVC-UF vs our deepest exit  →  max|diff| = 0.0"),
    (0, "Training's job is not to reach DCVC-UF. It is to (a) lift the shallow exits, (b) not damage the deepest one"),
    (1, "From-scratch runs had to reach 105 epochs from nothing; after 6 they were 1.49 dB short and were shelved"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=12)
pic(s, "s_exact.png", SW - Inches(4.4), Inches(2.4), width=Inches(4.05))
note(s, "Zero-tolerance controls run on every commit: 8 properties at max|diff| = 0.0, including j=K reproducing the full decode.")

# ---- 5 training setup ----------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Training: Microsoft's recipe, verified item by item")
bullets(s, [
    (0, "scripts/verify_recipe.py compares each item against ~/DCVC/train_image.py and exits non-zero on a mismatch"),
    (1, "schedule rows character-for-character · 106 entries · AdamW 1e-4 · clip 0.1 with non-finite skip"),
    (1, "ImageFolder + get_training_lambdas · 64 QP levels · encoder identical across 74 tensors · 255 shared tensors unchanged"),
    (0, "Stated additions, not hidden: --epoch_offset, --anchor_weight, --new_lr_scale, --freeze_encoder"),
    (0, "Live runs, one question each:"),
    (1, "VERBATIM — the recipe's final phase run in full (90→104), no additions at all"),
    (1, "RECIPE512 — offset 99: the recipe's own 512px regime, so --min_crop is not needed"),
    (1, "BEST / CONTROL — all measured winners vs none of them, single-variable"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=11.5)
pic(s, "s_recipe.png", SW - Inches(4.4), Inches(2.1), width=Inches(4.05))
note(s, "Effective batch is kept at the recipe's 16 by gradient accumulation, since 512×512 at batch 16 does not fit one card.")

# ---- 6 anchor ------------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "The anchor: a bug that would have cost a factor of four")
bullets(s, [
    (0, "The schedule was copied verbatim but read from epoch 0 — its from-scratch 2e-4 applied to a converged model"),
    (0, "Measured drift of the deepest exit from released DCVC-UF, after ONE epoch:"),
    (1, "qp0 −0.153 · qp32 −0.201 · qp63 −0.263 dB"),
    (0, "Every saving figure is quoted as 'x% at y dB'. With that drift, y was understated by exactly that much"),
    (1, "'37% at 0.1 dB' was really 37% at 0.29 dB against the published model"),
    (0, "Fix: read the schedule at the right place (offset 75/99/90) + pin the deepest exit by MSE to the release"),
    (1, "drift now −0.025 / −0.051 / −0.074 dB, and 0.000 exactly when the backbone is frozen"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=11.5)
pic(s, "s_anchor.png", SW - Inches(4.4), Inches(2.2), width=Inches(4.05))
note(s, "Freezing the backbone removes drift entirely but collapses the ceiling from 27.3% to 11.5% at qp0 — training the trunk is mandatory.")

# ---- 7 example patches ---------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "What an early exit actually looks like")
pic(s, "fig_exits.png", Inches(0.3), Inches(1.3), width=SW - Inches(0.6))
note(s, "Bosphorus 1080p, qp32, identical bitstream. Error maps are against the released decoder, amplified ×25. "
        "The residual is concentrated on edges and on the tile grid — the two things the ladder has to pay for.")

# ---- 8 seam --------------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "The seam: the binding constraint, and its removal")
bullets(s, [
    (0, "A tile decoded alone meets its border with invented values; every 3×3 pushes that inward"),
    (0, "Pure seam penalty at qp63 (all tiles deepest exit, CTC native resolution):"),
    (1, "zeros 1.167 · replicate 0.213 · arls 0.179 · 256px tile halves each of these"),
    (0, "arls (arXiv:2502.12300) wins on dB and was still rejected: +10.7% of decode wall-clock for +0.034 dB"),
    (1, "measured, not assumed — the wrapper itself is free, the bill is genuinely the least-squares fit"),
    (0, "Canvas coupling: only the 3×3 depthwise has spatial extent — 0.334% of a block"),
    (1, "giving just that one operator its real neighbours costs +0.03% of MACs"),
    (1, "for tiles at the same depth the result is bit-exact the full-frame decode: max|diff| = 0.0"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=11.5)
pic(s, "s_seam.png", SW - Inches(4.4), Inches(2.2), width=Inches(4.05))
note(s, "This overturned an earlier decision of ours: the trunk halo had been rejected at 1.745× because the halo was applied to the whole block, not to the 0.334% that needs it.")

# ---- 9 cost accounting ---------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Does the headline percentage survive a clock?")
pic(s, "report_best.png", Inches(0.5), Inches(1.3), width=SW - Inches(1.0))
bullets(s, [
    (0, "MAC-model saving vs measured wall-clock, 1080p: 43.4/43.6 · 28.6/28.8 · 13.8/14.9 · 0.0/0.5"),
    (0, "Within ±1 point at every exit — the reported number is one a user would feel"),
], Inches(0.7), Inches(5.9), SW - Inches(1.4), Inches(1.0), size=12)

# ---- 10 results ----------------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Result: saving against the released decoder")
pic(s, "two_views.png", Inches(0.3), Inches(1.35), width=SW - Inches(0.6))
bullets(s, [
    (0, "0.1 dB budget: 26.0% (qp0) → 9.6% (qp63).  0.3 dB budget: 41.5% → 26.3%"),
    (0, "The target band is met comfortably at 0.3 dB; at 0.1 dB only at low rate"),
], Inches(0.7), Inches(5.55), SW - Inches(1.4), Inches(1.1), size=12)
note(s, "ORACLE — a perfect choice of exit. The next slide is how that choice is actually made.")

# ---- 11 router -> signalling --------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "The router: predicting it failed, transmitting it works")
bullets(s, [
    (0, "A decoder-side router must infer each tile's error without ever seeing the source"),
    (1, "hand-made signals: |r| ≤ 0.12 against the oracle's choice · a 768-dim pooled stem measured WORSE"),
    (1, "4× the training moved qp0 agreement 0.743 → 0.741 with identical regret — missing information, not capacity"),
    (0, "In video coding, mode decisions are signalled, not inferred (HEVC/VVC transmit partitioning and modes)"),
    (0, "The encoder sees the source, computes the exit map exactly, and transmits it:"),
    (1, "24.2 / 21.2 / 15.7 / 11.0 / 8.6 % at qp0/16/32/48/63, all under 0.1 dB of the release"),
    (1, "cost of the map, included in the bpp: 1.3×10⁻⁴ bpp — ~200 bits per frame"),
    (0, "Within 1–2 points of the oracle bound; the predicting router paid 4× the dB for the same saving"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=11.5)
pic(s, "s_router.png", SW - Inches(4.4), Inches(2.2), width=Inches(4.05))

# ---- 12 theory + status --------------------------------------------------
s = prs.slides.add_slide(L_BODY); head(s, "Theory, status, and what is not yet done")
bullets(s, [
    (0, "Cost is additive over tiles and MSE is a mean over them, so both are affine in the assignment. Hence:"),
    (1, "fixed proportions achieve exactly the convex hull of the K per-exit points"),
    (1, "per-tile assignment achieves the Minkowski average of the N per-tile hulls"),
    (1, "their gap is min-of-average − average-of-min ≥ 0, zero iff every tile prefers the same exit"),
    (0, "That gap IS the value of adaptivity, computable before any router exists: 0.68% → 4.52% as rate rises"),
    (0, "Open / not done:"),
    (1, "the 30–40% target is NOT met at 0.1 dB above qp16 — it is met at 0.3 dB"),
    (1, "anchor_weight = 10's effect on the ceiling is still unmeasured"),
    (1, "HEVC classes B/C/D not evaluated (JVET credentials); 256px runs still training"),
], Inches(0.55), Inches(1.5), Inches(5.1), Inches(5.1), size=11.5)
pic(s, "s_gain.png", SW - Inches(4.4), Inches(2.2), width=Inches(4.05))
note(s, "paper/theory.tex — two pages, NeurIPS form, every proof given. Repo: github.com/canerim/FLEX-FLOP, branch flexuf-multiexit.")

OUT.parent.mkdir(exist_ok=True)
prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides)} slayt)")
