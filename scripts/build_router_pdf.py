"""A standalone explainer: how the exit map is chosen, on both sides.

Same reportlab machinery as build_pdf.py, single column, figure-led. The point
of a separate document is that the router is the part of the system a reader has
to understand before any number means anything, and in the paper it is
necessarily compressed into two subsections.
"""
import json, re, sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Image, Paragraph, Spacer,
                                Table, TableStyle, KeepTogether)

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from build_pdf import sub, tex_table, MACROS      # noqa: E402

FIGS = R / "paper" / "figures"


def S(n, **kw):
    b = dict(fontName="Times-Roman", fontSize=9.4, leading=12.4,
             alignment=TA_JUSTIFY, spaceAfter=6)
    b.update(kw)
    return ParagraphStyle(n, **b)


BODY = S("b")
H1 = S("h1", fontName="Times-Bold", fontSize=13, leading=15, spaceBefore=14,
       spaceAfter=5, alignment=0)
H2 = S("h2", fontName="Times-Bold", fontSize=10.6, leading=12.6,
       spaceBefore=10, spaceAfter=3, alignment=0)
CAP = S("cap", fontSize=8.2, leading=10.0)
TITLE = S("t", fontName="Times-Bold", fontSize=18, leading=21,
          alignment=TA_CENTER, spaceAfter=4)
SUB = S("s", fontSize=10.5, leading=13, alignment=TA_CENTER, spaceAfter=16,
        textColor=colors.HexColor("#555555"))
EQ = S("eq", fontName="Times-Italic", fontSize=11, leading=14,
       alignment=TA_CENTER, spaceBefore=6, spaceAfter=8)


def figure(name, cap, width=6.6 * inch):
    p = FIGS / name
    if not p.exists():
        p = R / "docs" / "figures" / name
    if not p.exists():
        return [Paragraph(f"<i>[{name} missing]</i>", CAP)]
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    im = Image(str(p), width=width, height=width * h / w)
    im.hAlign = "CENTER"
    return [Spacer(1, 4), im, Spacer(1, 3), Paragraph(sub(cap), CAP),
            Spacer(1, 8)]


def build(out="paper/FLEX-UF-router.pdf"):
    doc = SimpleDocTemplate(str(R / out), pagesize=letter,
                            leftMargin=0.95 * inch, rightMargin=0.95 * inch,
                            topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                            title="Choosing the exit map",
                            author="FLEX-UF")
    F = []
    A = F.append

    def par(t):
        A(Paragraph(sub(t), BODY))

    def h1(t):
        A(Paragraph(t, H1))

    def h2(t):
        A(Paragraph(t, H2))

    def eq(t):
        A(Paragraph(t, EQ))

    def fig(n, c, w=6.6 * inch):
        for x in figure(n, c, w):
            A(x)

    A(Paragraph("Choosing the exit map", TITLE))
    A(Paragraph("How the encoder decides, how the decoder decides, and what "
                "separates them", SUB))

    # ------------------------------------------------------------ the problem
    h1("1. The decision")
    par(r"The decoder is cut into a ladder: a tile can leave the trunk at any of "
        r"K exits, and the shallower it leaves the less it costs and the worse "
        r"it looks. A 1080p frame at 256 px tiles is 40 tiles and four reachable "
        r"exits, so a frame is one choice out of 4<super>40</super> — about "
        r"10<super>24</super>. Something "
        r"has to make that choice, once per frame, and the whole method is worth "
        r"exactly as much as that something is good.")
    par(r"Write D(t,k) for the distortion tile t incurs if it leaves at exit k, "
        r"and c_k for what exit k costs in units of one released decode. Then "
        r"the allocation that minimises total distortion under a compute budget "
        r"is the solution of a Lagrangian, and it decouples per tile:")
    eq("k*(t) &nbsp;=&nbsp; arg min<sub>k</sub> [ D(t,k) + λ · c<sub>k</sub> ]")
    par(r"One λ for the whole frame; raise it and every tile becomes more "
        r"willing to leave early. The budget is met by bisecting λ. This is the "
        r"same construction a codec uses to hit a rate target, with compute in "
        r"place of bits.")
    par(r"So the decision is not the hard part. <b>Getting D(t,k) is.</b> And "
        r"there the two sides of the codec are not symmetric at all.")

    fig("router_sees.png",
        r"<b>Figure 1.</b> The asymmetry the whole design turns on. D(t,k) is "
        r"an error against the source. The encoder holds the source, so it can "
        r"decode all K exits and measure D directly. The decoder never sees the "
        r"source: for it, D is not hard to estimate, it is <i>absent from the "
        r"input</i>, and no architecture recovers it. The two configurations "
        r"below are the two honest responses to that.")

    # ------------------------------------------------------------ A
    h1("2. Configuration A — the encoder searches")
    par(r"The encoder has x. It runs the decoder K times over the frame, one "
        r"per exit, and reads off D(t,k) exactly. No estimation, no model, no "
        r"training: the table is measured. Then it bisects λ until the frame "
        r"lands on the quality budget and transmits the resulting map.")

    fig("router_a_search.png",
        r"<b>Figure 2.</b> The search, on one real frame (Bosphorus, q32, 40 "
        r"tiles). <b>a</b>, the measured table D(t,k), tiles sorted by "
        r"difficulty. The dominant variation is down the rows, not across them: "
        r"tiles differ from each other far more than exits differ within a "
        r"tile, which is why allocating per tile is worth anything at all. "
        r"<b>b</b>, the cost ladder — four reachable rungs, roughly 0.14 apart. "
        r"<b>c</b>, the argmin at every λ in the sweep. Tiles migrate from the "
        r"deepest exit to the shallowest as λ grows, and they migrate in order "
        r"of difficulty: the easy tiles at the top leave first, the hard ones at "
        r"the bottom hold on longest. Above λ ≈ 5×10⁻⁵ everything has left and "
        r"the ladder is saturated.")

    fig("router_frontier.png",
        r"<b>Figure 3.</b> What the sweep traces. <b>a</b>, distortion against "
        r"λ; the dashed line is the 0.1 dB budget, and bisection finds where "
        r"the curve crosses it. <b>b</b>, the same allocations as a frontier of "
        r"compute against quality, with the budget point marked. The frontier "
        r"is concave and ends flat: past saturation, spending more quality buys "
        r"no more compute.", 5.0 * inch)

    h2("What it costs")
    par(r"<b>Bits.</b> Four symbols over 40 tiles is 80 bits raw, and the "
        r"distribution is far from uniform, so an arithmetic coder with a "
        r"per-frame histogram spends about \MapBits bits — \MapOverheadLow% of "
        r"a typical 1080p bitrate. Every bpp we report includes it.")
    par(r"<b>Encoder compute.</b> About 1.21 decodes per frame: one shared trunk "
        r"pass plus the adapter and head at each exit. Encoders are already "
        r"asymmetric with decoders by a large factor, and this is a search a "
        r"real encoder would fold into its existing mode decision.")
    par(r"<b>Decoder compute.</b> None. It is told.")
    par(r"<b>Bitstream syntax.</b> One new field, so both ends must agree. This "
        r"is the conventional design rather than the exotic one: HEVC and VVC "
        r"signal block partitioning, prediction mode and the transform tree "
        r"rather than having the decoder infer them.")

    # ------------------------------------------------------------ B
    h1("3. Configuration B — the decoder predicts")
    par(r"The other honest response is to send nothing and let the decoder "
        r"guess. The file is then byte-identical to a stock stream, no syntax "
        r"changes, and a decoder vendor can deploy it alone. What the decoder "
        r"has to work with is everything it has already decoded before the trunk "
        r"runs: the full-frame stem feature, the latent ŷ, the entropy model's "
        r"scales, and the quality index q.")

    fig("router_b_head.png",
        r"<b>Figure 4.</b> The predictor. Two 1×1 projections reduce the stem "
        r"and the latent side, both are pooled to one vector per tile, the "
        r"quality index is appended, and a three-layer MLP emits one logit per "
        r"exit. Everything is per tile and pointwise, so the head adds no "
        r"receptive field and no seam.", 5.6 * inch)

    par(r"The decision mirrors A's, with the log-probability standing in for "
        r"negative distortion and β for λ:")
    eq("k̂(t) &nbsp;=&nbsp; arg max<sub>k</sub> [ log softmax(z<sub>t</sub>)"
       "<sub>k</sub> − β · c<sub>k</sub> ]")
    par(r"β is bisected exactly as λ is, so the budget is met the same way. The "
        r"head is 144,024 parameters and 0.163% of one decode, and that 0.163% "
        r"is charged against every saving configuration B reports.")

    h2("How it is trained, and how it fails")
    par(r"Against a <b>frozen</b> decoder, with a cost-sensitive cross-entropy "
        r"to the oracle's choice: each tile weighted by the regret of choosing "
        r"wrongly, so capacity goes where the decision matters and the ties are "
        r"left alone. Plus loss-free load balancing — a per-exit bias nudged by "
        r"usage rather than by a gradient — biased toward the <i>oracle's</i> "
        r"exit distribution rather than toward uniform. Uniform is what a "
        r"mixture-of-experts wants because its experts are interchangeable; "
        r"ours are not, and at a high λ the oracle genuinely does send every "
        r"tile to one exit.")
    par(r"Trained jointly with a decoder that is still moving, it can collapse. "
        r"We have three routers and three outcomes, all measured: one jointly "
        r"trained router collapsed to a rate-dependent constant with zero "
        r"agreement with the oracle; a second jointly trained router did not "
        r"collapse; the frozen-decoder recipe reaches 0.85–0.92 held-out "
        r"agreement. So joint training is not automatically fatal, but it is "
        r"not reliable either, and the frozen recipe is what we report.")

    # ------------------------------------------------------------ compare
    h1("4. What separates them")
    A(KeepTogether([tex_table("static", 6.4 * inch), Spacer(1, 3),
                    Paragraph(sub(
                        r"<b>Table 1.</b> Allocations at matched compute, "
                        r"0.1 dB. The two shuffled rows are the useful controls: "
                        r"both take the oracle's own exit histogram — the same "
                        r"mix of depths, hence the same average cost — and "
                        r"differ only in which tile gets which depth."), CAP),
                    Spacer(1, 8)]))
    par(r"Given the oracle's histogram but assigned at random, quality "
        r"collapses. What the allocation buys is not that some tiles run "
        r"shallower; it is knowing <i>which</i> ones can.")
    par(r"The rate-ranked row is the one that should temper any enthusiasm for "
        r"a learned router. It keeps that histogram and orders it by a signal "
        r"the decoder already holds and did not have to learn: the number of "
        r"bits the entropy model spent on each tile. A tile that cost more bits "
        r"carries more detail and should run deeper. No parameters, no training, "
        r"and the number is available before the trunk starts. At q63 a random "
        r"assignment costs \RandomDb dB where the oracle costs \OracleDb dB at "
        r"identical compute; rate-ranking costs \RateRankDb dB — "
        r"<b>\RateRankRecovers% of the oracle's advantage over chance</b>, at "
        r"0.68–0.79 agreement with the oracle's map. Any learned router has to "
        r"beat that before it has earned its 0.163%.")

    fig("router_ab.png",
        r"<b>Figure 5.</b> A against B across rate and budget. The gap is real "
        r"when the budget is tight and vanishes when it is loose: once the "
        r"budget saturates the ladder, both configurations send every tile to "
        r"the same rung and the only difference left is the predictor's own "
        r"compute.")

    h1("5. Which one to use")
    A(KeepTogether([Table(
        [[Paragraph(f"<b>{a}</b>", CAP), Paragraph(b, CAP), Paragraph(c, CAP)]
         for a, b, c in [
            ("", "A — signalled", "B — predicted"),
            ("Optimality", "exact, it is the oracle", "0.55–0.86 agreement"),
            ("Bits added", f"≈{MACROS.get('MapBits','94')}/frame "
                           f"({MACROS.get('MapOverheadLow','0.011')}%)", "zero"),
            ("Decoder compute", "none", "0.163% of a decode"),
            ("Encoder compute", "≈1.21 decodes", "none"),
            ("File", "coded payload identical, plus a field",
             "byte-identical"),
            ("Syntax", "needs a new field", "unchanged"),
            ("Deployable by", "both ends together", "a decoder vendor alone"),
            ("Failure mode", "none observed",
             "collapses if trained on a moving decoder"),
         ]],
        colWidths=[1.5 * inch, 2.5 * inch, 2.4 * inch])])),
    F[-1] = KeepTogether(F[-1][0]) if isinstance(F[-1], tuple) else F[-1]
    par("")
    par(r"They answer different questions. A is what the system can do when the "
        r"allocation is right, and it is the number to quote as the method's "
        r"capability. B is what survives when nothing may change in the "
        r"bitstream, and it is the number to quote as the method's "
        r"deployability. Reporting only one of them would be reporting half the "
        r"result.")

    doc.build(F)
    print(f"  -> {out}")


if __name__ == "__main__":
    build()
