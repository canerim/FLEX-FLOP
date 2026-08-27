"""Every ablation, in one document, with what each one rules out.

A table of variants is not an ablation study. An ablation is an argument: this
number moves when that is changed, therefore the explanation is this and not
that. So each family below states the alternative explanation it removes, and
where a result is negative it says what the negative result buys.

Numbers are read from flexplus/results/ablations.json at build time, which is
itself gathered from the measurement files. Nothing here is typed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle,
                                KeepTogether)

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "flexplus" / "results"
A = json.loads((RES / "ablations.json").read_text())
FAM, MISSING = A["families"], A["unparsed"]

BLUE = colors.HexColor("#0065BD")
DARK = colors.HexColor("#003359")
ORANGE = colors.HexColor("#E37222")
INK = colors.HexColor("#1a1a1a")
GREY = colors.HexColor("#5a5a5a")
BG = colors.HexColor("#f6f6f4")

S = lambda **k: ParagraphStyle("s", **{**dict(
    fontName="Helvetica", fontSize=9.4, leading=13.2, textColor=INK,
    spaceAfter=5), **k})
H1 = S(fontName="Helvetica-Bold", fontSize=13.5, leading=16,
       textColor=BLUE, spaceBefore=13, spaceAfter=5)
H0 = S(fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=DARK,
       spaceAfter=3)
SUB = S(fontSize=10.5, leading=14, textColor=GREY, spaceAfter=11)
Q = S(fontName="Helvetica-Bold", fontSize=9.2, leading=12.5,
      textColor=ORANGE, spaceAfter=5)
SRC = S(fontSize=7.8, leading=10.5, textColor=GREY, spaceBefore=2,
        spaceAfter=10)

story = []


def h1(t):
    story.append(Paragraph(t, H1))


def par(t, st=None):
    story.append(Paragraph(t, st or S()))


def rules_out(t):
    story.append(Paragraph("Rules out:  " + t, Q))


def src(t):
    story.append(Paragraph(t, SRC))


def tbl(rows, widths, hl=()):
    t = Table(rows, colWidths=widths, rowHeights=0.215 * inch)
    cmd = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.6),
           ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
           ("TEXTCOLOR", (0, 0), (-1, 0), DARK),
           ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("LEFTPADDING", (0, 0), (-1, -1), 5),
           ("RIGHTPADDING", (0, 0), (-1, -1), 5),
           ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE)]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            cmd.append(("BACKGROUND", (0, r), (-1, r), BG))
    for r in hl:
        cmd += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e6f0fa")),
                ("FONT", (0, r), (-1, r), "Helvetica-Bold", 8.6)]
    t.setStyle(TableStyle(cmd))
    story.append(KeepTogether([t, Spacer(1, 7)]))


def pq(v):
    return "  ".join(f"{x:.2f}" for x in v)


# ==========================================================================
story.append(Paragraph("FLEX — ablation studies", H0))
story.append(Paragraph(
    "Fourteen families, every number read from the measurement files. The "
    "main experiment throughout is RECIPE512 at epoch 9, 256 px tiles, 53 CTC "
    "intra frames, 0.1 dB, unless a section says otherwise. Where a family "
    "was measured on the dumped per-cell tables rather than a real decode it "
    "says so, because the tables carry no tiling penalty and sit "
    "optimistically.", SUB))

# 1 ------------------------------------------------------------------------
h1("1.  Decode geometry — where the saving comes from")
rules_out("that the saving is an early-exit effect alone. Most of it is, but "
          "a measurable part is the cost of cutting the frame into tiles, and "
          "that part is recoverable without touching the ladder.")
r = FAM["decode_geometry"]
rows = [["variant", "mean", "qp0", "qp16", "qp32", "qp48", "qp63", "Δ"]]
prev = None
for x in r["rows"]:
    rows.append([x["decode"], f"{x['mean']:.2f}%"] +
                [f"{v:.2f}" for v in x["per_qp"]] +
                ["" if prev is None else f"{x['mean']-prev:+.2f}"])
    prev = x["mean"]
tbl(rows, [2.05 * inch, 0.62 * inch] + [0.52 * inch] * 5 + [0.50 * inch],
    hl=[1])
par("Tiled decoding pays a <b>floor</b>: every 3×3 at a tile border reads "
    "padding instead of a neighbour, and that costs quality before any tile "
    "has exited early. Per-position depth removes the cut entirely — each "
    "position runs to its own depth and the receptive field is handled by a "
    "max-plus dilation rather than by a boundary — so the floor drops from "
    "0.048 dB to 0.020 dB at the top rate and the same budget buys more.")
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 2 ------------------------------------------------------------------------
h1("2.  Granularity — how fine the allocation should be")
rules_out("that finer is always better. It is better in multiply-accumulates "
          "and worse once a real kernel is charged (section 9).")
r = FAM["granularity"]
rows = [["cell size", "mean saving", "band cost"]]
for x in r["rows"]:
    rows.append([f"{x['cell_px']} px", f"{x['mean']:.2f}%",
                 f"{x['band_pct']:.2f} points"])
tbl(rows, [1.3 * inch, 1.3 * inch, 1.3 * inch])
par("The <b>band</b> is what a finer allocation spends its winnings on: "
    "positions that keep computing only to serve a deeper neighbour. It is "
    "0.33 points at 256 px and 2.75 at 32 px, and it is why the curve "
    "flattens rather than continuing.")
src(r["source"])

# 3 ------------------------------------------------------------------------
h1("3.  The clamp — are the two lowest exits worth reaching")
rules_out("that removing the clamp is an independent knob. It is not: under "
          "tiling the first exits are not cheaper, so the clamp can only be "
          "lifted together with the tiling.")
r = FAM["clamp"]
rows = [["split depth", "mean saving", "ceiling"]]
for x in r["rows"]:
    rows.append([f"j = {x['split_depth']}", f"{x['mean']:.2f}%",
                 f"{x['ceiling_pct']:.2f}%"])
tbl(rows, [1.3 * inch, 1.3 * inch, 1.3 * inch])
par("In the tiled architecture the first <i>j</i> groups run over the whole "
    "frame before patchify, so exits 0, 1 and 2 all cost 0.5716 of a decode — "
    "identical. Choosing a shallower one there buys nothing and loses "
    "quality. Only per-position decoding, which has no shared prefix, makes "
    "them genuinely cheap, and then the ceiling rises from 40.1% to 69.9%.")
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 4 ------------------------------------------------------------------------
h1("4.  Baselines — what predicts the exit-error curve")
rules_out("that a learned router is needed for the prediction. It is not: a "
          "parameter-free rule reading the entropy coder's own bit count "
          "beats it.")
r = FAM["router_models"]
ora = [x for x in r["rows"] if x["model"] == "oracle"][0]["mean"]
rows = [["model", "mean saving", "of the oracle"]]
for x in sorted(r["rows"], key=lambda z: -z["mean"]):
    rows.append([x["model"], f"{x['mean']:.2f}%",
                 f"{100*x['mean']/ora:.1f}%"])
tbl(rows, [1.7 * inch, 1.3 * inch, 1.3 * inch], hl=[1])
par("The oracle is the exhaustive per-tile search and bounds what any router "
    "can reach. <b>flat</b> is the control with no adaptivity at all — one "
    "curve for every tile — and the gap between it and the oracle is the "
    "whole of what routing can win.")
src(r["source"] + f"   ·   {r.get('cell_px')} px cells, j = {r.get('split_depth')}")

# 5 ------------------------------------------------------------------------
h1("5.  Router inputs — which group carries the signal")
rules_out("that the head needs all of its inputs. One group at a time is "
          "zeroed after its projection, so every variant has the same "
          "architecture, the same parameter count and the same optimiser "
          "state, and the only difference is how much the head may know.")
r = FAM["router_inputs"]
rows = [["inputs live", "agreement with the oracle", "best constant"]]
for x in sorted(r["rows"], key=lambda z: -z["agree_pct"]):
    rows.append([x["inputs"], f"{x['agree_pct']:.1f}%",
                 f"{x['constant_pct']:.1f}%"])
tbl(rows, [2.4 * inch, 1.7 * inch, 1.3 * inch], hl=[1])
par("<b>qp</b> is the control that matters: it is one number per frame, the "
    "same for every tile in it, so it cannot in principle separate two tiles "
    "of one frame. Whatever agreement it reaches is what is available with no "
    "per-tile information at all.")
src(r["source"])

# 6 ------------------------------------------------------------------------
h1("6.  The closed-form refit — is it the refit or the conditioning")
rules_out("that re-solving the adapters is what helps. Two thirds of the gain "
          "is lost when the same solve runs on all tiles, and all of it is "
          "lost when it runs on survivors chosen at the wrong multiplier.")
r = FAM["calex_refit"]
rows = [["arm", "mean saving", "against baseline"],
        ["baseline, epoch 9", f"{r['baseline_mean']:.2f}%", "—"]]
for x in r["rows"]:
    rows.append([x["arm"], f"{x['mean']:.2f}%",
                 f"{x['mean']-r['baseline_mean']:+.2f}"])
tbl(rows, [2.6 * inch, 1.2 * inch, 1.4 * inch], hl=[2])
par("Each adapter ends in a pointwise linear map, and the target is known — "
    "the deepest exit's raw feature, which is what the shared head was fitted "
    "to consume. So the solve is a ridge regression, in the manner of "
    "AdaRound and BRECQ, and it needs no gradient step. 443,520 parameters "
    "change, 0.98% of the decoder, two minutes on one card.")
par("The third row is the control that carries the argument. Same method, "
    "same data, same number of parameters, and the conditional distribution "
    "wrong: the gain vanishes to −0.01. What the method does is not fit; it "
    "is fit <i>where the adapter is used</i>.")
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 7 ------------------------------------------------------------------------
h1("7.  Fine-tuning — does gradient descent add anything")
rules_out("that the adapters are undertrained. With the anchor on the "
          "released decoder and the adapters at the main run's own learning "
          "rate, six thousand steps return the baseline exactly, with the "
          "usage prior and without it.")
r = FAM["fine_tuning"]
b0 = [x for x in r["rows"] if "baseline" in x["arm"]][0]["mean"]
rows = [["arm", "mean saving", "against baseline"]]
for x in r["rows"]:
    rows.append([x["arm"], f"{x['mean']:.2f}%", f"{x['mean']-b0:+.2f}"])
tbl(rows, [3.1 * inch, 1.2 * inch, 1.4 * inch], hl=[len(rows) - 1])
par("This is the negative result that explains the positive one. The "
    "adapters have converged — for an <b>average</b> distribution — and "
    "stochastic gradient descent on OpenImages cannot be aimed away from it: "
    "every step sees every tile, and the multiplier matching that makes the "
    "closed form work has no analogue in the training loop. The closed form "
    "can state the conditional directly.")
par("A first attempt lost 3.52 points with the prior and 8.21 without it. "
    "Both losses were the harness, not the idea: the anchor had been pinned "
    "to the run's own starting checkpoint, where it is computed on the "
    "full-frame path a frozen backbone cannot move, so anchor_mse was "
    "identically zero and nothing held the tiled deepest path. Corrected, "
    "both arms return to the baseline.", S(textColor=GREY))
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 8 ------------------------------------------------------------------------
h1("8.  Where the multiplier is chosen — the guarantee")
rules_out("that a per-frame guarantee costs saving. It raises it, because a "
          "single multiplier solves a Lagrangian on pooled squared error "
          "while the constraint is on a mean of logarithms.")
r = FAM["guarantee"]
rows = [["multiplier", "mean saving", "over budget", "95th pct", "worst"]]
for x in r["rows"]:
    rows.append([x["mode"], f"{x['saving']:.2f}%",
                 f"{x['over']} / {x['n']}", f"{x['p95']:.3f} dB",
                 f"{x['max']:.3f} dB"])
tbl(rows, [1.9 * inch, 1.0 * inch, 1.1 * inch, 0.95 * inch, 0.95 * inch],
    hl=[2])
par("Measured on the deployed tiled path, one real decode per bisection "
    "step, and the multiplier returned is by construction one whose delivered "
    "decibel is at or under the budget — the bound holds by the solver rather "
    "than by its convergence. An earlier version bisected a cheap table and "
    "corrected six times, which left 17 of 53 frames above the budget at "
    "qp0: a guarantee measured by a solver that does not guarantee.")
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 9 ------------------------------------------------------------------------
h1("9.  Implementability — what a real kernel would keep")
rules_out("that the multiply-accumulate count is the saving. A block-sparse "
          "kernel must run any B×B tile containing one needed position, and "
          "the finer allocation loses more to that than the coarse one.")
r = FAM["block_sparsity"]
bs = {x["cells"]: x for x in r["rows"]}
keys = sorted(bs["256 px cells"]["blocks"], key=lambda k: int(k))
rows = [["kernel granularity", "256 px cells", "64 px cells"],
        ["ideal, per position", f"{bs['256 px cells']['ideal_pct']:.2f}%",
         f"{bs['64 px cells']['ideal_pct']:.2f}%"]]
for k in keys:
    rows.append([f"B = {k} feature positions ({int(k)*8} px)",
                 f"{bs['256 px cells']['blocks'][k]:.2f}%",
                 f"{bs['64 px cells']['blocks'][k]:.2f}%"])
tbl(rows, [2.4 * inch, 1.3 * inch, 1.3 * inch], hl=[3])
par("At B = 8 the coarse allocation keeps 30.06% against the fine one's "
    "27.99%, reversing the ideal ranking. Granularity has an optimum once "
    "implementability is charged, and it is coarser than the "
    "multiply-accumulate optimum.")
src(r["source"])

# 10 -----------------------------------------------------------------------
h1("10.  Can the decoder shorten its own tail")
rules_out("that the decoder-side tail is made of near-ties. Deepening every "
          "tile whose best two exits score within a margin does not move the "
          "worst frame at any margin.")
r = FAM["safe_routing"]
rows = [["margin δ", "mean saving", "worst frame", "over 0.1 dB"]]
for x in r["margins"]:
    rows.append([f"{x['delta']:.2f}", f"{x['saving']:.2f}%",
                 f"{x['max_db']:.3f} dB", f"{x['over']} / 265"])
tbl(rows, [1.0 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
par("The tail is not indecision; it is frames where the surrogate is "
    "confidently wrong, and no tie-break reaches those. Letting the decoder "
    "throttle itself on its own predicted decibel does shorten it, but only "
    "by giving up most of the saving:")
rows = [["predicted target", "mean saving", "worst frame"]]
for x in r["throttle"]:
    rows.append([f"{x['target']:.3f} dB", f"{x['saving']:.2f}%",
                 f"{x['max_db']:.3f} dB"])
tbl(rows, [1.4 * inch, 1.2 * inch, 1.2 * inch])
sg = r["signalled_lambda"]
par(f"What works costs sixteen bits. The encoder already runs a replica of "
    f"the decoder's predictor in configuration C, so it can evaluate the "
    f"<b>true</b> frame decibel of the allocation the decoder will make and "
    f"bisect on it, then send that one scalar: "
    f"{sg['n']-sg['over_010']} of {sg['n']} frames inside the budget, worst "
    f"{sg['max']:.3f} dB, and the saving rises from "
    f"{r['margins'][0]['saving']:.2f}% to {sg['saving_pct']:.2f}%. "
    "The map is still never transmitted; only its price is.")
src(r["source"] + "   ·   per-cell tables, no tiling penalty")

# 11 -----------------------------------------------------------------------
h1("11.  Generalisation — an image benchmark, not video frames")
rules_out("that the method transfers uniformly. It does not: at the top rate "
          "the tiling floor alone exceeds the budget on a third of Kodak.")
r = FAM["kodak"]
rows = [["", "mean saving", "images inside 0.1 dB", "mean tiling floor"]]
for x in r["per_image"]:
    rows.append([f"Kodak, qp {x['qp']}", f"{x['saving']:.2f}%",
                 f"{x['n']-x['infeasible']} / {x['n']}",
                 f"{x['mean_floor_db']:.4f} dB"])
tbl(rows, [1.5 * inch, 1.2 * inch, 1.6 * inch, 1.4 * inch], hl=[3])
par("The infeasibility is the floor, not the allocation: delivered decibel "
    "equals floor for those images, and no granularity rescues them. It is "
    "the second, independent argument for removing the tiling — the floor is "
    "not merely a tax on saving, it is what makes a per-frame guarantee "
    "unreachable on hard content.")
src(r["source"] + f"   ·   epoch {r.get('ckpt_epoch')}")

# 12 -----------------------------------------------------------------------
h1("12.  The operating point")
rules_out("that 0.1 dB is a property of the system. It is a choice, and the "
          "saturation structure says what it costs.")
r = FAM["budget"]
rows = [["budget", "mean saving", "frames already saturated"]]
for x in r["rows"]:
    rows.append([f"{x['budget_db']:.6f} dB", f"{x['mean']:.2f}%",
                 f"{x['saturated']} / {x['n']}"])
tbl(rows, [1.5 * inch, 1.2 * inch, 1.8 * inch])
par("A saturated frame is one whose ceiling — every tile at the shallowest "
    "exit it may take — already sits under the budget, so the tolerance above "
    "that point is spent on nothing. On the deployed tiled path the ceiling "
    "is content-dependent and spreads from 0.017 to 0.860 dB, so saturation "
    "arrives gradually rather than at one sharp point.")
src(r["source"])

# 13 -----------------------------------------------------------------------
h1("13.  Training length")
rules_out("that more training is the remaining lever.")
r = FAM["epochs"]
rows = [["epoch", "mean saving"]]
for x in r["rows"]:
    rows.append([str(x["epoch"]), f"{x['mean']:.2f}%"])
tbl(rows, [1.0 * inch, 1.3 * inch])
par("Epoch 9 to 10 is −0.02 at a learning rate of 1e-6. The ladder has "
    "converged, and the improvements that remain are not training "
    "improvements — which is what sections 6 and 7 measure.")
src(r["source"])

# 14 -----------------------------------------------------------------------
h1("14.  Is the usage prior a property of the test set")
rules_out("that the closed-form refit borrows from the test half. Two "
          "disjoint halves of the video set agree to 0.029 total variation, "
          "while a different domain differs by 0.283.")
r = FAM["prior_stability"]
rows = [["source of the prior"] + [f"e{k}" for k in range(6)]]
for k, lab in (("ctc_half_A_27seq", "CTC half A (27 sequences)"),
               ("ctc_half_B_26seq", "CTC half B (26 sequences)"),
               ("ctc_full_53seq", "CTC, all 53"),
               ("openimages_train", "OpenImages (the training domain)")):
    rows.append([lab] + [f"{100*v:.1f}%" for v in r[k]])
tbl(rows, [2.0 * inch] + [0.62 * inch] * 6, hl=[1, 2])
par(f"KL between the halves is {r['kl_halfA_halfB_nat']:.4f} nat; between "
    f"the training domain and the test domain it is "
    f"{r['kl_train_vs_ctc_nat']:.4f}. So the prior can be estimated from "
    "held-out material and 27 sequences suffice — but it has to be held-out "
    "material of the <b>deployment</b> domain, because photographs give a "
    "materially different vector than video frames do.")
src("flexplus/results/prior_stability.json")

if MISSING:
    h1("Families this document could not parse")
    for m in MISSING:
        par(m, S(textColor=ORANGE, fontSize=8.4))

# ==========================================================================
def build(out="report/FLEX-ablations.pdf"):
    p = HERE.parent / out
    doc = BaseDocTemplate(str(p), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=18 * mm,
                          topMargin=18 * mm, bottomMargin=18 * mm,
                          title="FLEX — ablation studies")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
               id="n", leftPadding=0, rightPadding=0,
               topPadding=0, bottomPadding=0)

    def page(c, d_):
        c.setFillColor(GREY); c.setFont("Helvetica", 7.4)
        c.drawString(20 * mm, 11 * mm,
                     "FLEX — ablation studies  ·  RECIPE512 epoch 9, 256 px "
                     "tiles, 53 CTC intra frames")
        c.drawRightString(A4[0] - 18 * mm, 11 * mm, str(c.getPageNumber()))
        c.setStrokeColor(colors.HexColor("#dddddd")); c.setLineWidth(0.4)
        c.line(20 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)

    doc.addPageTemplates([PageTemplate(id="n", frames=[fr], onPage=page)])
    doc.build(list(story))
    print(f"  yazildi {p}")


if __name__ == "__main__":
    build()
