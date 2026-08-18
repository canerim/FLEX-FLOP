"""Render the paper to PDF without a LaTeX installation.

There is no TeX on this machine and installing a distribution on a shared box to
produce one document is not a reasonable trade. `paper/main.tex` remains the
submission source -- it is what goes to Overleaf and to the CVPR kit -- and this
produces a reviewable PDF from the same generated tables and figures, so what is
on screen is what the measurements say.

Two-column, letter, 10pt, in the shape a CVPR reader expects. Content lives in
`PAPER` below as (kind, payload) pairs; tables and macros are read from
`paper/tables/`, so nothing numeric is typed here either.
"""
import json, re, sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Image,
                                Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, FrameBreak, NextPageTemplate)

R = Path(__file__).resolve().parents[1]
PAPER, FIGS, TABLES = R / "paper", R / "paper" / "figures", R / "paper" / "tables"

# ---------------------------------------------------------------- macros
MACROS = {}
mt = TABLES / "macros.tex"
if mt.exists():
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", mt.read_text()):
        MACROS[m.group(1)] = m.group(2)


def sub(t):
    """Expand \Macro and the handful of TeX-isms the text uses."""
    for k in sorted(MACROS, key=len, reverse=True):
        t = t.replace("\\" + k, MACROS[k])
    t = (t.replace(r"\dB", " dB").replace(r"\%", "%").replace(r"\,", " ")
          .replace(r"\times", "×").replace(r"\emph{", "<i>")
          .replace(r"\textbf{", "<b>").replace(r"\approx", "≈")
          .replace("---", "\u2014").replace("--", "\u2013")
          .replace(r"\lambda", "λ").replace(r"\beta", "β")
          .replace("``", "\u201c").replace("''", "\u201d"))
    # close the braces opened by emph/textbf, in order
    out, depth = [], []
    i = 0
    while i < len(t):
        if t[i] == "<" and t[i:i + 3] in ("<i>", "<b>"):
            depth.append("</i>" if t[i:i + 3] == "<i>" else "</b>")
            out.append(t[i:i + 3]); i += 3
        elif t[i] == "}" and depth:
            out.append(depth.pop()); i += 1
        else:
            out.append(t[i]); i += 1
    return "".join(out)


# ---------------------------------------------------------------- styles
def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=8.6, leading=10.4,
                alignment=TA_JUSTIFY, spaceAfter=4)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = S("body")
ABST = S("abst", fontSize=8.4, leading=10.0)
H1 = S("h1", fontName="Times-Bold", fontSize=11.5, leading=13,
       spaceBefore=9, spaceAfter=4, alignment=0)
H2 = S("h2", fontName="Times-Bold", fontSize=9.6, leading=11,
       spaceBefore=7, spaceAfter=3, alignment=0)
CAP = S("cap", fontSize=7.4, leading=8.8)
TITLE = S("title", fontName="Times-Bold", fontSize=17, leading=20,
          alignment=TA_CENTER, spaceAfter=6)
AUTH = S("auth", fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=12)


# ------------------------------------------------------- LaTeX table -> flowable
def tex_table(name, width):
    p = TABLES / f"{name}.tex"
    if not p.exists():
        return Paragraph(f"<i>[{name} not generated]</i>", CAP)
    rows, txt = [], p.read_text()
    for line in txt.splitlines():
        line = line.strip()
        if (not line or line.startswith("\\begin") or line.startswith("\\end")
                or line.startswith("\\toprule") or line.startswith("\\midrule")
                or line.startswith("\\bottomrule") or line.startswith("\\cmidrule")):
            continue
        line = line.rstrip("\\").rstrip()
        # \multicolumn{n}{a}{text} -> text; the spanning is a LaTeX layout
        # device and reportlab does its own, so keeping the wrapper would print
        # "multicolumn2cSaved" into the header.
        line = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
        cells = [sub(c.strip()) for c in line.split("&")]
        cells = [re.sub(r"\$([^$]*)\$", r"\1", c) for c in cells]
        cells = [c.replace(r"\_", "_").replace("{", "").replace("}", "")
                 .replace(r"^\ast", "*").replace(r"^\dagger", "†")
                 .replace("\\", "") for c in cells]
        fs = 7.2 if len(cells) <= 7 else 6.0
        rows.append([Paragraph(c, S("cell", fontSize=fs, leading=fs * 1.16,
                                    alignment=0 if i == 0 else 2))
                     for i, c in enumerate(cells)])
    if not rows:
        return Spacer(1, 1)
    ncol = max(len(r) for r in rows)
    rows = [r + [Paragraph("", CAP)] * (ncol - len(r)) for r in rows]
    first = 0.34 if ncol <= 4 else (0.28 if ncol <= 7 else 0.20)
    cw = [width * first] + [width * (1 - first) / (ncol - 1)] * (ncol - 1)
    t = Table(rows, colWidths=cw, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


_FIGN = [0]
_TABN = [0]


def _autonum(cap, counter, word):
    """Replace a hand-written 'Figure N.' with the running count.

    The numbers were typed into the captions and drifted the moment a figure was
    inserted in the middle -- which happened twice tonight.
    """
    counter[0] += 1
    return re.sub(rf"{word}\s+\d+\.", f"{word} {counter[0]}.", cap, count=1)


def fig(name, width, cap, maxh=None):
    cap = _autonum(cap, _FIGN, "Figure")
    """Figure + caption, scaled to `width` and, if given, capped at `maxh`.

    The cap matters for the teaser: the pipeline diagram is 7.2 x 4.3 inches, so
    at full text width it is taller than any banner frame worth having, and
    reportlab draws an oversized flowable over whatever is beneath it instead of
    refusing.
    """
    p = FIGS / name
    if not p.exists():
        return [Paragraph(f"<i>[{name} missing]</i>", CAP)]
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    iw, ih = width, width * h / w
    if maxh and ih > maxh:
        iw, ih = maxh * w / h, maxh
    im = Image(str(p), width=iw, height=ih)
    im.hAlign = "CENTER"
    return [im, Spacer(1, 3), Paragraph(sub(cap), CAP), Spacer(1, 6)]


# ---------------------------------------------------------------- the content
def content(colw, fullw):
    """(flowables for the two-column body, flowables for the full-width banner)."""
    F = []
    A = F.append

    def par(t):
        A(Paragraph(sub(t), BODY))

    def h1(t):
        A(Paragraph(t, H1))

    def h2(t):
        A(Paragraph(t, H2))

    def tbl(name, cap):
        A(KeepTogether([tex_table(name, colw), Spacer(1, 3),
                        Paragraph(sub(_autonum(cap, _TABN, "Table")), CAP),
                        Spacer(1, 6)]))

    def figure(name, cap):
        for f in fig(name, colw, cap):
            A(f)

    # ---- abstract -------------------------------------------------------
    A(Paragraph("<b>Abstract</b>", S("ah", fontName="Times-Bold", fontSize=9.4,
                                     alignment=TA_CENTER, spaceAfter=4)))
    A(Paragraph(sub(
        r"A learned image decoder spends the same computation on every frame "
        r"regardless of what the frame contains. We add an early-exit ladder to "
        r"the intra decoder of DCVC-UF, cut each frame into tiles, and let every "
        r"tile leave the trunk at whichever of K exits is deep enough for that "
        r"tile. The encoder is never touched and the coded payload is unchanged, "
        r"so the method deploys against existing bitstreams. On the full "
        r"\NumSeq-sequence HEVC/UVG/MCL-JCV common test set a 0.1 dB quality "
        r"budget buys \MainLowRate% of the decoder's multiply-accumulates at the "
        r"lowest rate and \MainHighRate% at the highest, at a BD-Rate cost of "
        r"0.88%. Two findings shape the design. First, tiling is not free: a 3×3 "
        r"convolution at a tile border meets padding instead of a neighbour, and "
        r"on this decoder that costs \SeamZerosHigh dB at q63 under the stock "
        r"rule — several times the entire budget, spent before a single tile "
        r"has exited early. The two remedies that work cost nothing, a "
        r"higher-order border estimator is <i>worse</i> than a zeroth-order one, "
        r"and a learned repair module cannot pay for itself. Second, a quality "
        r"budget only buys compute inside a bounded window: below a <i>floor</i> "
        r"set by tiling no allocation is feasible, and above a <i>saturation</i> "
        r"point every tile already sits on the cheapest rung. We measure both "
        r"ends at every rate. We also show that the obvious implementation of the "
        r"ladder is <i>slower</i> than the dense decoder, and that a "
        r"bit-identical reordering of the per-tile loop recovers \WallSorted% of "
        r"wall-clock against a \WallPredicted% arithmetic prediction."), ABST))
    A(Spacer(1, 6))

    # ---- 1 introduction --------------------------------------------------
    h1("1. Introduction")
    par(r"Learned codecs are compared on rate and distortion, with complexity "
        r"reported as a single number: so many GMAC per frame, so many "
        r"milliseconds. That number is a constant. A learned decoder runs the "
        r"same graph on a page of text and on a cloudless sky, and the sky is "
        r"not harder.")
    par(r"Classification networks abandoned this a decade ago. Early-exit "
        r"architectures attach classifiers at intermediate depths and stop once "
        r"the prediction is confident [3, 15, 20]. Super-resolution adopted the "
        r"idea spatially: ClassSR [12] routes patches to networks of different "
        r"capacity, APE [1] exits patches at different depths. Learned "
        r"compression went another way — slimmable autoencoders [22, 23] "
        r"give one model several complexity levels, but the level is chosen "
        r"<i>per stream</i>, not per region, and switching it changes the "
        r"bitstream.")
    par(r"We ask the spatial-adaptivity question inside a learned decoder under "
        r"a constraint that makes the answer deployable: <b>the encoder is "
        r"frozen and the coded payload is unchanged.</b> That rules out "
        r"retraining the analysis transform, changing the entropy model or "
        r"altering the latent, and leaves exactly one place to spend adaptivity "
        r"— the synthesis transform.")
    par(r"FLEX-UF is an exit ladder over the twelve residual blocks of the "
        r"DCVC-UF intra decoder. The first j blocks run over the whole frame; "
        r"the rest run per tile on a shrinking set, and that shrinkage is the "
        r"saving. Each exit reaches the shared head through a small pointwise "
        r"adapter, zero-initialised so that at step zero the deepest exit "
        r"reproduces the released decoder bit-exactly.")
    par(r"Making it work depended less on the ladder than on two things that "
        r"have little to do with early exit as usually studied.")
    h2("Tiling has a price, and it is large.")
    par(r"Spatial adaptivity needs tiles, and the moment a frame becomes tiles "
        r"every 3×3 convolution at a border reads invented values. The error is "
        r"\SeamZerosHigh dB at high rate under the stock padding rule — "
        r"several times the budget, paid before any tile has saved an "
        r"operation. Section 4 treats padding as an <i>estimator</i> of the "
        r"unseen neighbour and measures four; the ordering is not the obvious "
        r"one.")
    h2("A quality budget has a working range.")
    par(r"Because tiling costs something even when nothing exits early there is "
        r"a <i>floor</i>: budgets below it admit no allocation. Because the "
        r"ladder has a shallowest rung there is a <i>saturation</i> point above "
        r"which more quality buys nothing. The working 0.1 dB budget uses "
        r"\BandUseLow% of that window at the lowest rate and \BandUseHigh% at "
        r"the highest.")
    h2("Contributions.")
    par(r"(i) A tile-adaptive early-exit ladder for a learned image decoder "
        r"that leaves the encoder and coded payload untouched, saving "
        r"\MainLowRate% to \MainHighRate% of decoder MACs for 0.1 dB on the "
        r"full CTC set. (ii) The floor/saturation characterisation of a quality "
        r"budget, with both ends measured, and the observation that the "
        r"architectural ceiling is reachable at low rate. (iii) A quantitative "
        r"account of the tiling penalty, including two negative results. "
        r"(iv) Two allocation regimes — an encoder-side search signalling "
        r"a ~\MapBits-bit map, and a decoder-side predictor signalling nothing "
        r"— with bits and compute charged on both sides. (v) A wall-clock "
        r"result: the obvious implementation is slower than the dense decoder, "
        r"and a bit-identical reordering recovers \WallSorted%.")

    # ---- 2 related -------------------------------------------------------
    h1("2. Related work")
    h2("Early exit.")
    par(r"BranchyNet [20] and MSDNet [3] established intermediate classifiers "
        r"with a confidence rule; SDN [15] framed it as mitigating "
        r"overthinking. Joint training of all exits follows Scardapane et al. "
        r"[18]; distillation between exits follows Phuong and Lampert [17], "
        r"with the caveat from [19] that too large a student-teacher gap hurts "
        r"the shallowest exits — which is why ours is between "
        r"<i>adjacent</i> exits. All of this exits on a confidence signal "
        r"computed from the network's own output. A decoder has no such signal: "
        r"the quantity that would decide is the error against a source it "
        r"cannot see.")
    h2("Spatially adaptive inference.")
    par(r"ClassSR [12] sorts patches into easy/medium/hard; APE [1] exits "
        r"patches at different depths; Glance-and-Focus [8] spends resolution "
        r"adaptively. These share our per-region premise and share the tiling "
        r"problem, though the cost of tile borders is rarely quantified. The "
        r"closest prior work on the border itself is the per-channel AR(1) "
        r"padding of Kaseva et al. [11], which we implement and evaluate.")
    h2("Complexity control in learned compression.")
    par(r"SlimCAE [22] and slimmable video codecs [23] expose several widths of "
        r"one model; the choice is per stream and changes the encoder, so the "
        r"bitstream is not interchangeable. DCVC-FM [13] and DCVC-UF [14] "
        r"reduce cost architecturally, for every frame equally. Our axis is "
        r"orthogonal: model fixed, bitstream fixed, only <i>how much of the "
        r"decoder runs where</i> varies.")
    h2("Signalling versus prediction.")
    par(r"That a decoder is <i>told</i> a mode decision rather than inferring "
        r"it is the norm in standardised video coding: HEVC [9] and VVC [21] "
        r"transmit partitioning, prediction mode and transform tree. We "
        r"evaluate both and treat the signalled variant as the conventional "
        r"design. The nearest neighbour in compression is spatial competition "
        r"[27], which selects among several codecs per region and signals a mode "
        r"map: the structure of the side information is ours, the axis is not "
        r"— they select which network at fixed complexity for a rate gain, "
        r"we select how much of one network at fixed rate for compute.")
    h2("Allocating a budget over units.")
    par(r"The construction we use is not new and should not be presented as "
        r"such. Shoham and Gersho [28] showed that for a finite set of per-unit "
        r"operating points a Lagrangian sweep decouples the allocation across "
        r"units and traces exactly the lower convex hull of the achievable set; "
        r"Ortega and Ramchandran [29] made it standard practice in image and "
        r"video coding. What we add is the structure this particular operating "
        r"set has — a floor below which no allocation is feasible, a "
        r"saturation point above which none improves, and a measurement of what "
        r"the convex-hull restriction costs. The same relaxation has resurfaced "
        r"for test-time compute in language models [30], with per-instance "
        r"decoupling and a binary search on the multiplier: the identical "
        r"structure in a domain with no rate axis.")
    h2("Tile boundaries.")
    par(r"Every method that processes an image in independently-computed tiles "
        r"meets the same artefact, and the remedies in the literature are "
        r"overlap and averaging, local padding from neighbouring patches [31], "
        r"training with overlaps [32], and fitted extrapolation [11]. Local "
        r"padding is closest to the exact remedy of Section 4.4, and the "
        r"difference is accounting: it pads every convolutional layer and does "
        r"not report the cost, while we pad only the 0.29% of each block that "
        r"has any spatial extent — which is what makes the exact fix cost "
        r"0.032% of the decode rather than a multiplier on all of it.")
    h2("Where the time goes.")
    par(r"DCVC-RT [33] argues that operational rather than computational "
        r"complexity is the speed bottleneck for neural codecs, evidenced by "
        r"channel reductions that yield linear rather than quadratic speedups. "
        r"Section 5.6 is a sharp instance of that claim inside one loop: a 20% "
        r"reduction in operations produced a 10.9% <i>slow-down</i> until the "
        r"loop was reordered, with the arithmetic untouched.")

    # ---- 3 method --------------------------------------------------------
    h1("3. Method")
    h2("3.1 Where the computation is")
    par(r"The DCVC-UF intra decoder is one upsampling block, twelve "
        r"DepthConvBlocks and a head, costing 453.5 GMAC per 1080p frame. The "
        r"twelve blocks are 89.4% of it, which is why the ladder is built "
        r"across them. Inside one block at C=384, the only operator with any "
        r"spatial extent is a 3×3 depthwise costing 9C against the block's "
        r"8C²+9C — <b>0.29%.</b> That number recurs throughout: it is why "
        r"tile borders damage the picture, why our adapters are pointwise on "
        r"purpose, and why the exact remedy of Section 4.4 is affordable.")
    h2("3.2 The exit ladder")
    par(r"With K exits over the twelve blocks and split depth j, groups 0..j−1 "
        r"run full frame for every tile and groups j..K−1 run per tile. A tile "
        r"assigned exit k runs groups j..k and leaves. Writing c_k for the cost "
        r"of exit k in units of one released decode, a frame with exit map k "
        r"costs the mean of c over its tiles.")
    par(r"Two consequences are easy to state wrongly. Exits shallower than j "
        r"are indistinguishable, so the usable ladder has K−j distinct costs, "
        r"and the <i>architectural ceiling</i> is S_max = 100(1−c_j)%, attained "
        r"when every tile takes exit j. For the shipped setting (K=6, j=2, 256 "
        r"px tiles) S_max = \Ceiling%. Section 5.4 shows this bound is "
        r"<i>reached</i> at low rate, which makes it an operating point rather "
        r"than an asymptote.")
    figure("adapters.png",
           r"<b>Figure 2. Inside an exit adapter.</b> Both kinds are entirely "
           r"pointwise, so no adapter adds receptive field and none contributes "
           r"any tile-boundary penalty. Capacity is matched to the number of "
           r"blocks the exit skips, and charged for: an exit-2 tile saves six "
           r"blocks minus 0.25, not six.")
    h2("3.3 Exit adapters")
    par(r"An early exit hands the shared head a feature the head was not fitted "
        r"to; the adapter is the correction. We use a residual 1×1, Ad(f) = f + "
        r"Wf with W zero-initialised, for exits that skip little, and the "
        r"pointwise expand/activate/contract pair of the block's own FFN for "
        r"exits that skip four blocks or more. Both are <i>pointwise by "
        r"design</i>: a 3×3 inside an adapter would add seam damage precisely "
        r"at the tiles that took an early exit, the ones least able to afford "
        r"it. Zero-initialising the last layer makes every adapter the identity "
        r"at step zero, so the deepest exit is bit-exactly the released decoder "
        r"before training begins.")
    h2("3.4 Allocation")
    par(r"Given per-tile distortions D(t,k) and costs c_k, the allocation "
        r"minimising distortion at a compute budget is the Lagrangian "
        r"<b>k*(t) = argmin_k [ D(t,k) + λ·c_k ]</b>, with λ bisected so the "
        r"frame lands on the budget. Both quantities are available <i>to the "
        r"encoder</i>, which holds the source. This is configuration <b>A</b>: "
        r"the encoder searches and transmits the map, which entropy-codes to "
        r"~\MapBits bits per 1080p frame, \MapOverheadLow% of a typical "
        r"bitrate. It costs the encoder about one extra decode and the decoder "
        r"nothing.")
    par(r"The decoder cannot evaluate it: D(t,k) is the error against a source "
        r"it never sees. This is not a hard estimation problem, it is a missing "
        r"variable, and no architecture removes it. Configuration <b>B</b> "
        r"therefore predicts: a 144K-parameter head reads the stem feature, the "
        r"decoded latent, the entropy model's scales and the quality index, and "
        r"decides <b>k(t) = argmax_k [ log softmax(z_t)_k − β·c_k ]</b>, with β "
        r"bisected as λ is. Nothing is added to the file. The head is trained "
        r"against a <i>frozen</i> decoder with a cost-sensitive cross-entropy to "
        r"the oracle's choice, each tile weighted by the regret of choosing "
        r"wrongly, plus loss-free load balancing [16] biased toward the "
        r"oracle's own exit distribution rather than toward uniform.")
    h2("3.5 Training")
    par(r"All exits are decoded every step and the objective is L = L_RD + "
        r"w_a·L_anchor + w_d·L_distill. L_RD is the released rate-distortion "
        r"loss with MSE averaged over exits. L_anchor pins the deepest exit to "
        r"the released decoder, so the reference every saving is quoted against "
        r"cannot drift underneath the measurement. L_distill supervises "
        r"adapters in <i>feature</i> space, exit k imitating exit k+1 — "
        r"feature space because the head is a fixed map from feature to RGB and "
        r"matching the deeper feature is the stronger constraint, 384 dense "
        r"channels of target instead of 3; adjacent rather than deepest for the "
        r"reason in [19].")
    par(r"Critically, the adapters are trained <i>through the tiled decode path "
        r"they are deployed in</i>. Training them full-frame and tiling only at "
        r"inference loses 0.14–0.24 dB; training through the deployed path "
        r"gains 0.51–0.90 dB. The sign of the effect flips.")

    # ---- 4 the seam ------------------------------------------------------
    h1("4. The price of tiles")
    figure("seam_problem.png",
           r"<b>Figure 3. The artefact, before anything is done about it.</b> "
           r"One frame decoded twice from the <i>same</i> bitstream with the "
           r"same weights, every tile at full depth — no early exit "
           r"anywhere. The only difference is that the right-hand decode was "
           r"tiled. The error map is the tile lattice and nothing else.")
    par(r"A 3×3 depthwise computes a weighted sum over its neighbours. Decoded "
        r"full frame those neighbours exist; decoded per tile they do not, and "
        r"the kernel is handed whatever the padding rule invents. Each "
        r"convolution extends the contaminated region by one ring, so with b "
        r"per-tile blocks on a tile of side F the fraction of the tile within "
        r"reach of an invented value is 1 − ((F−2b)/F)², which is 0.750 at the "
        r"shipped F=32, b=8: not a thin border but a structured error across "
        r"most of the tile.")
    par(r"That fraction says which pixels are affected, not how badly, and it "
        r"is not what governs the penalty. Sweeping the split depth sweeps b "
        r"from 12 to 0, with b=0 as an exact zero-seam control, and the measured "
        r"seam follows a power law: <b>seam ∝ b<super>α</super></b> with α = "
        r"2.38, 2.22 and 1.93 at q0, q32 and q63, r = 0.98–0.995 in log-log. "
        r"Fitted with one free scale the area fraction is wrong by 160–264% "
        r"where the power law is wrong by 12–22%. The exponent near two has a "
        r"reading: the count of contaminated pixels grows like perimeter times "
        r"depth, and the error accumulated in each grows with how many "
        r"convolutions reached it. The area law saturates once every pixel is "
        r"touched; the penalty does not.")
    figure("contamination.png",
           r"<b>Figure 4. The seam against per-tile depth.</b> Sweeping the "
           r"split depth sweeps b; b=0 is an exact control and measures 0.0000 "
           r"dB at all three rates. <b>a</b>, the measurement with the fitted "
           r"power law. <b>b</b>, both models against q63 with one free scale "
           r"each. <b>c</b>, mean relative error. The area fraction saturates "
           r"once every pixel is contaminated; the penalty does not.")
    h2("4.1 Padding is an estimator")
    par(r"The useful way to see border padding is as an <i>estimator</i> of the "
        r"unseen neighbour, whose error is the seam. Table 1 measures four, "
        r"with early exit switched off so tiling is the only difference from a "
        r"full-frame decode.")
    tbl("padding",
        r"<b>Table 1. Border estimators</b>, dB below the released decoder on "
        r"the same latent, 256 px tiles, full CTC. Lower is better.")
    par(r"Two results deserve emphasis. <b>A higher-order estimator is worse.</b> "
        r"Extrapolating the local gradient past a boundary amplifies whatever "
        r"noise sits on it; assuming local constancy does not. Guessing harder "
        r"is not guessing better, and the gap is large: \SeamLinearHigh dB "
        r"against \SeamReplHigh dB at q63. <b>The best estimator loses on "
        r"cost.</b> The per-channel AR(1) fit of [11] reaches \SeamArlsHigh dB, "
        r"about 0.02 dB better than replication, for 10.7% of decode "
        r"wall-clock. Against a 0.1 dB budget and a ~24% saving that trade does "
        r"not close, and we drop it.")
    h2("4.2 What actually removes the seam")
    figure("seam_vs_qp.png",
           r"<b>Figure 4. The tiling penalty across the whole rate range</b>, "
           r"every tile at full depth. The two steps that matter cost nothing, "
           r"and the largest single factor is training the ladder with the seam "
           r"present. Right: the floor is charged <i>inside</i> the quality "
           r"budget and consumes a growing share of it.")
    par(r"Tile size is free — a tiled decode's MAC count does not depend on "
        r"it at all — and the damage scales with the tile perimeter (Table "
        r"2). What it costs instead is routing granularity: 40 tiles per 1080p "
        r"frame at 256 px against 160 at 128 px.")
    tbl("tilesize",
        r"<b>Table 2. Tile size</b>, dB at q63. Doubling the side roughly "
        r"halves the penalty, as the contamination law predicts, and costs no "
        r"computation.")
    par(r"The largest factor of all is easiest to overlook. Replicate padding "
        r"leaves \SeamReplHigh dB at q63 on the untrained ladder; after "
        r"training the same configuration leaves \FloorHigh dB. More than two "
        r"thirds of what the estimator could not fix is absorbed by weights "
        r"learning to live with it. No module in this paper removes as much.")
    h2("4.3 A learned repair module, and why we reject it")
    figure("seam_module.png",
           r"<b>Figure 5. Grid seam repair.</b> A correction gated by position "
           r"within a tile, applied once to the stitched frame. The trained "
           r"gate learns the boundary ring (0.76) but never switches off inside "
           r"(0.145 over 94% of pixels), and the measured effect follows: it "
           r"helps on the ring and hurts everywhere else.")
    par(r"Since the tile lattice is known exactly at training and inference, a "
        r"repair can be <i>told</i> where to look rather than having to infer "
        r"it: Rep(f) = f + G[i mod P, j mod P]·PW(WSiLU(DW3×3(f))), with G a "
        r"P×P gate shared over channels, initialised at exp(−d/τ). It costs "
        r"0.95% of the decode.")
    par(r"It does not earn that. Splitting per-pixel error by distance from the "
        r"nearest tile boundary, the module gains 0.27% in the 0–4 px band "
        r"(6.2% of pixels) and loses 0.04–0.05% everywhere else. Even with a "
        r"<i>perfect</i> gate — zero correction in the interior, the "
        r"boundary gain unchanged — the ceiling on what it could earn is "
        r"≈0.0008 dB for 0.95% of the decode. We rejected AR(1) padding at "
        r"0.0019 dB per point of decode; this is worse by an order of "
        r"magnitude. Tightening the gate cannot rescue it, because there is "
        r"almost nothing left to win.")
    h2("4.4 Removing the cause, and why it does not help")
    par(r"Because only 0.29% of each block has spatial extent, the exact fix is "
        r"affordable: give the 3×3 its real neighbour from a shared canvas, for "
        r"+0.032% of the decode. And it is exact — at uniform depth a coupled "
        r"tiled decode is bit-identical to a full-frame one, once the comparison "
        r"is not confounded by the repair module, which runs on the stitched "
        r"canvas and has no full-frame counterpart. Measured: 1.13e-2 with the "
        r"repair on, <b>exactly 0</b> with it off, against 6.06e-2 for replicate "
        r"padding.")
    par(r"It also removes most of the floor. Switched on at inference the floor "
        r"falls from 0.036 to 0.003 dB at q0 and from 0.056 to 0.030 at q63, "
        r"i.e. 91% and 47% of the tiling penalty.")
    par(r"<b>And it destroys the allocation.</b> At the same 0.1 dB budget the "
        r"saving falls from 25.9% to 4.2% at q32 and from 19.3% to 0.4% at q63.")
    par(r"The reason is the condition in the exactness statement. Coupling is "
        r"exact when every tile is at the <i>same</i> depth, and routing is the "
        r"deliberate violation of that condition. With replicate padding a tile "
        r"is entirely independent of its neighbours, so the allocation may give "
        r"adjacent tiles any depths it likes; coupling makes a tile depend on "
        r"its neighbours, and under routing those neighbours ran a different "
        r"number of blocks. A shallow tile reading a deep neighbour's activation "
        r"is a configuration the weights have never seen, and that mismatch "
        r"costs far more than the seam it removed. We report it because the "
        r"natural reading of an exactness result — adopt the exact fix — is "
        r"the wrong one here, and the tension is a property of the combination "
        r"that any spatially adaptive decoder inherits.")

    # ---- 5 experiments ---------------------------------------------------
    h1("5. Experiments")
    h2("Setup.")
    par(r"The decoder is fine-tuned on OpenImages 512×512 crops with the "
        r"encoder frozen (max|Δ| = 0 asserted every run), one λ drawn per "
        r"sample from a log-spaced range covering all 64 quality indices, so a "
        r"single set of weights serves the whole rate range. Evaluation is on "
        r"the \NumSeq sequences of the common test set — UVG, MCL-JCV and "
        r"HEVC classes B, C, D and E — one intra frame each.")
    h2("Measurement protocol.")
    par(r"Two details change the numbers enough to state. First, the per-tile "
        r"distortion table must be built on the <i>deployed</i> decode path "
        r"— one tiled decode per exit — and not by tapping the exits of "
        r"a single full-frame forward pass. The latter is the natural "
        r"implementation and correct for training, but with a full-frame "
        r"reference on the other side of the ratio the tiling penalty cancels "
        r"and the reported quality belongs to a decoder nobody ships. On our "
        r"model the difference is +0.035 dB at the deepest exit and +0.008 dB "
        r"at the shallowest; it grows with how many blocks ran per tile, so it "
        r"cannot be corrected after the fact. Second, after the allocation is "
        r"chosen we decode that <i>mixed</i> map once and report the distortion "
        r"it actually produces.")
    h2("Reporting conventions.")
    par(r"Every dB is measured against the <i>released</i> decoder's "
        r"full-frame decode of the <i>same</i> latent, and our side is the "
        r"deployed tiled decode, so the tiling penalty is inside every number. "
        r"Savings are fractions of the released decoder's cost; our own deepest "
        r"exit costs 1.0095 of it, and using that as the denominator would "
        r"flatter every result by 0.6–0.8 points.")
    h2("5.1 Main result")
    figure("qualitative.png",
           r"<b>Figure 6. What the saving looks like.</b> The same bitstream "
           r"decoded by the released decoder and by ours at the 0.1 dB operating "
           r"point, 30.5% fewer multiply-accumulates. The crop is the tile that "
           r"gave up the most quality, chosen automatically, so this is the "
           r"method's worst case on this frame rather than a flattering one.")
    tbl("main_results",
        r"<b>Table 3. Decoder MACs saved</b> (%) against the released decoder, "
        r"per quality index and budget, configuration A. The 0.3 and 0.5 dB "
        r"rows sit on the architectural ceiling almost everywhere: past that "
        r"point a looser budget buys nothing.")
    par(r"At 0.1 dB the method saves \MainLowRate% at the lowest rate and "
        r"\MainHighRate% at the highest, averaging \MainMean%, at a BD-Rate "
        r"cost of 0.88% — that is, the compute is worth about the same as a "
        r"0.88% increase in bitrate. The fall with rate is not an artefact of "
        r"the ladder: high-rate reconstructions carry detail the shallow exits "
        r"cannot reproduce, and the floor rises with rate.")
    tbl("complexity",
        r"<b>Table 4. Decoder complexity</b> at 1080p. Our deepest exit costs "
        r"slightly more than the release because it still pays the seam-repair "
        r"module; that 1.0095, not 1.0, is what every saving here is "
        r"<i>not</i> divided by.")
    h2("5.2 Is per-tile adaptivity necessary?")
    figure("exit_map.png",
           r"<b>Figure 6. Where the decoder spends.</b> (a) the assignment "
           r"overlaid on the frame, (b) the exit index per tile, (c) the "
           r"quality each tile gives up against its <i>own</i> full-depth "
           r"reference. Water and sky leave at the shallowest rung; the boat "
           r"and shoreline run deep.")
    tbl("static",
        r"<b>Table 5. Adaptivity against three controls</b> at 0.1 dB. "
        r"† marks uniform depths exceeding the budget. Random and rate-ranked "
        r"draw the oracle's own exit histogram — same average cost, same mix "
        r"of depths — and differ only in which tile gets which depth.")
    par(r"The uniform rows answer the obvious alternative, and the answer "
        r"sharpens with rate. At the lowest rate a static decoder reaches exit "
        r"3 within the budget and saves 27.0%, against the oracle's "
        r"\MainLowRate%. At q48 and q63 the only uniform depth that fits is "
        r"the deepest, which <i>costs</i> 0.95% rather than saving anything "
        r"— so there the choice is not between 17% and less, it is between "
        r"17% and nothing. Averaged over rates the best static allocation saves "
        r"10.2% where the adaptive one saves \MainMean%.")
    par(r"The two shuffled rows separate effects that are easy to conflate. "
        r"Given the oracle's histogram but assigned at random, quality "
        r"collapses: what the allocation buys is not that some tiles run "
        r"shallower, it is knowing <i>which</i> ones can. The rate-ranked row "
        r"keeps that histogram and orders it by a signal the decoder already "
        r"holds — the bits the entropy model spent on each tile. It is the "
        r"cheapest conceivable router, with no parameters and no training, and "
        r"it is the natural analogue of the confidence rules early-exit "
        r"classifiers use.")
    par(r"It recovers most of the gap. At q63, random costs 0.141 dB and the "
        r"oracle 0.100 dB at identical compute; rate-ranking costs 0.110 dB, "
        r"i.e. 75% of the oracle's advantage over chance, at 0.68–0.79 "
        r"agreement with the oracle's map. A learned router therefore has to "
        r"beat a free baseline already three quarters of the way there — a "
        r"comparison we would not have made without this control, and one we "
        r"suggest any adaptive-inference paper should report.")
    h2("5.3 Resolution, and the granularity of a tile")
    figure("perclass.png",
           r"<b>Figure 7. Saving by test class</b> at one global operating "
           r"point. λ is bisected once so the whole set lands on the budget, "
           r"exactly as it would be in deployment; each class is then reported "
           r"at that λ. The ordering follows tile count, not content.")
    tbl("perclass",
        r"<b>Table 6. Saving by class</b> (%) at a 0.1 dB budget set globally "
        r"over all 53 sequences. ``Tiles'' is how many 256 px tiles a frame of "
        r"that resolution contains.")
    par(r"Completing the test set exposed a dependence the 1080p-only subset "
        r"had hidden. At 1080p, where a frame is 40 tiles, the method saves "
        r"\BigResLow% (MCL-JCV), 33.1% (UVG) and 33.8% (HEVC B) at the lowest "
        r"rate — three classes of very different content within three "
        r"points of each other. At 832×480 it is 8 tiles and \MidResLow%; at "
        r"416×240 it is 2 tiles and \SmallResLow%. At q63 the two "
        r"low-resolution classes fall to \SmallResHigh% against "
        r"\BigResHigh% for 1080p. The method is a function of tile count far "
        r"more than of content.")
    par(r"The cause is granularity: with two tiles per frame there is almost no "
        r"allocation to make and the Lagrangian degenerates toward a uniform "
        r"choice. The fix is not more compute but a tile size chosen relative "
        r"to the frame — the tiling penalty scales with the tile perimeter "
        r"and the MAC count does not depend on tile size at all, so the trade "
        r"is between seam and granularity and should be resolved per "
        r"resolution. We report the single 256 px setting used throughout and "
        r"note that a resolution-adaptive tile size is the obvious extension.")
    h2("5.4 The working range of a quality budget")
    figure("saturation_RECIPE512.png",
           r"<b>Figure 8. Three regions, and only the middle one is a design "
           r"choice.</b> Below the floor no allocation meets the budget; above "
           r"saturation every tile is already on the cheapest rung. The working "
           r"0.1 dB budget uses \BandUseLow% of the window at the lowest rate "
           r"and \BandUseHigh% at the highest.")
    tbl("operating",
        r"<b>Table 7. Floor and saturation</b>, dB below the released decoder. "
        r"The floor is what tiling costs with no early exit at all; saturation "
        r"is where every tile reaches exit j and the ceiling is attained.")
    par(r"Two numbers bound what any budget can do. The <b>floor</b> is the "
        r"distortion of a tiled decode with every tile at full depth — pure "
        r"tiling penalty, \FloorLow dB at q0 rising to \FloorHigh dB at q63. "
        r"A budget below it admits no allocation. The <b>saturation</b> point "
        r"is the distortion when every tile takes exit j; any budget at or "
        r"above it attains the ceiling and a larger one attains nothing more.")
    par(r"A consequence worth stating plainly: at q0 the ceiling is reached at "
        r"\SatLow dB. The architectural bound is not an asymptote, it is an "
        r"operating point, and beyond it the limiting factor is the "
        r"<i>ladder</i>, not the budget — an argument for a finer ladder "
        r"rather than a looser budget. It also identifies a narrow regime: "
        r"budgets in [\SatLow, \SatWindowHi) dB saturate the lowest rate and "
        r"nothing else, a window \SatWindowMb millibels wide.")
    h2("5.5 Signalled versus predicted allocation")
    figure("router_ab.png",
           r"<b>Figure 9. Who decides.</b> A signals the oracle map at "
           r"~\MapBits bits/frame; B predicts it from decoder-side data only "
           r"and signals nothing. The gap collapses to the router's own compute "
           r"once the budget is loose enough that both saturate.")
    par(r"Configuration A is exact by construction and costs bits; B is "
        r"approximate and costs none. The comparison is not a wash in either "
        r"direction: the prediction gap is real at a tight budget and vanishes "
        r"at a loose one, because once the budget saturates both configurations "
        r"send every tile to the same rung and the only remaining difference is "
        r"the predictor's own 0.163% of decode.")
    h2("5.6 Complexity and wall-clock")
    tbl("latency",
        r"<b>Table 8. Wall-clock</b>, 1080p, median of 40 interleaved "
        r"iterations, at the 0.1 dB operating point. ``MACs'' is what the "
        r"arithmetic predicts; ``measured'' is the sorted per-tile loop.")
    par(r"A saving in multiply-accumulates is not a saving in time, and here "
        r"the gap was initially total. Tiling costs \TilingOverhead% before "
        r"anything exits early, and the obvious implementation of a shrinking "
        r"active set — a boolean mask, a gather of survivors and a scatter "
        r"of the finished at every group boundary — forces a device-to-host "
        r"synchronisation per boundary. Profiling attributes 32% of the "
        r"per-tile loop to that bookkeeping. Measured end to end, the routed "
        r"decoder was <i>slower</i> than the dense one at high rate despite "
        r"doing 20% fewer operations.")
    par(r"Sorting the tiles once by exit depth removes it. Descending, ``still "
        r"active at group g'' becomes a contiguous prefix: each group is a "
        r"slice rather than a gather, the boundaries come from one cumulative "
        r"count rather than a mask per group, and the finished tiles return "
        r"through the inverse permutation in a single scatter. The arithmetic "
        r"is unchanged and the output is bit-identical on GPU. The saving at q0 "
        r"goes from \WallMasked% to \WallSorted% against a \WallPredicted% "
        r"arithmetic prediction.")
    par(r"We report this because the negative result is the more useful half: a "
        r"paper reporting only MACs would have claimed a speedup that the same "
        r"code, run as written, did not deliver.")
    h2("5.7 The right ladder depends on the budget")
    tbl("runs",
        r"<b>Table 9. Ladder configurations</b>, mean saving (%) over the five "
        r"rates, same test set and protocol. * one rate is infeasible at that "
        r"budget — the ladder's floor exceeds it — so the mean is over "
        r"the remaining four.")
    par(r"Section 5.4 argued that once a budget saturates a ladder the only way "
        r"to spend more is a rung that does not exist. Table 9 measures it. A "
        r"finer ladder (K=12, j=4) has a ceiling of 50.3% against 41.9%, and "
        r"the orderings cross between 0.1 and 0.3 dB. At 0.1 dB the coarse "
        r"ladder wins, and the fine one cannot even reach the budget at the "
        r"highest rate: splitting later puts twice as many blocks in the "
        r"per-tile section, which raises the floor. At 0.5 dB the fine ladder "
        r"wins by 6.7 points, \FineHalfDb% against \CoarseHalfDb%, because "
        r"the coarse one has been pinned at its ceiling since 0.3 dB.")
    par(r"So the ladder is not a hyperparameter to be tuned once. It is a "
        r"function of the operating point, and the floor-saturation window is "
        r"what tells you which side of the crossover you are on.")

    # ---- 6 limitations ---------------------------------------------------
    h1("6. Limitations")
    par(r"<b>The seam is reduced, and the exact remedy is not usable as it "
        r"stands.</b> Canvas coupling removes 47–91% of the floor and is "
        r"bit-exact at uniform depth, yet collapses the routed saving because "
        r"routing puts neighbouring tiles at different depths. Whether a decoder "
        r"trained with coupling recovers both at once is open, and it is the "
        r"experiment we would run next. Until it exists the floor is a cost this "
        r"method pays.")
    par(r"<b>Intra frames only.</b> This is the image path of a video codec. "
        r"Extending the ladder to inter frames raises a question this paper "
        r"does not answer: an exit map propagates through the reference chain, "
        r"so a shallow tile in one frame is a worse reference for the next.")
    par(r"<b>Training is not converged.</b> The measured saving is still "
        r"increasing at every checkpoint measured more than once, so these "
        r"numbers are a lower bound.")
    par(r"<b>A single decoder.</b> All results are on DCVC-UF's intra decoder. "
        r"Nothing in the method is specific to it — the ladder needs only a "
        r"residual trunk with a shared head — but that is an argument, not "
        r"a measurement.")

    # ---- 7 conclusion ----------------------------------------------------
    h1("7. Conclusion")
    par(r"Learned decoders spend a constant amount of computation on a "
        r"non-constant world. An early-exit ladder over the tiles of a frame "
        r"recovers a substantial fraction of it — \MainLowRate% to "
        r"\MainHighRate% of decoder MACs for a 0.1 dB budget — without "
        r"touching the encoder or the coded payload.")
    par(r"The two lessons we would carry to any spatially adaptive decoder are "
        r"the ones that were not about early exit. Tiling is expensive, its "
        r"cost is dominated by a single 3×3 that is 0.29% of the arithmetic, "
        r"and the remedies that work are geometric rather than learned. And a "
        r"quality budget is only a control variable inside a measurable window: "
        r"below the floor it admits nothing, above saturation it buys nothing, "
        r"and reporting a saving without stating where in that window it sits "
        r"leaves the most useful part of the result out.")
    return F


# ------------------------------------------------------------ references
REFS = [
 "S. Wang et al. Adaptive patch exiting for scalable single image super-resolution. ECCV, 2022.",
 "J. Ballé et al. Variational image compression with a scale hyperprior. ICLR, 2018.",
 "G. Huang et al. Multi-scale dense networks for resource efficient image classification. ICLR, 2018.",
 "G. Bjøntegaard. Calculation of average PSNR differences between RD curves. VCEG-M33, 2001.",
 "B. Bross et al. Overview of the Versatile Video Coding standard. IEEE TCSVT, 2021.",
 "Z. Cheng et al. Learned image compression with discretized Gaussian mixture likelihoods. CVPR, 2020.",
 "W. Fedus et al. Switch Transformers. JMLR, 2022.",
 "G. Huang et al. Glance and Focus networks for dynamic visual recognition. IEEE TPAMI, 2023.",
 "G. J. Sullivan et al. Overview of the High Efficiency Video Coding standard. IEEE TCSVT, 2012.",
 "E. Jang et al. Categorical reparameterization with Gumbel-softmax. ICLR, 2017.",
 "T. Kaseva et al. Per-channel autoregressive linear prediction padding in tiled CNN processing of 2D spatial data. arXiv:2502.12300, 2025.",
 "X. Kong et al. ClassSR: a general framework to accelerate super-resolution networks by data characteristic. CVPR, 2021.",
 "J. Li, B. Li, Y. Lu. Neural video compression with feature modulation. CVPR, 2024.",
 "J. Li, B. Li, Y. Lu. Uniformly accelerated neural video codec with feature refinement. ACM MM, 2024.",
 "Y. Kaya et al. Shallow-deep networks: understanding and mitigating network overthinking. ICML, 2019.",
 "L. Wang et al. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv:2408.15664, 2024.",
 "M. Phuong, C. H. Lampert. Distillation-based training for multi-exit architectures. ICCV, 2019.",
 "S. Scardapane et al. Why should we add early exits to neural networks? Cognitive Computation, 2020.",
 "W. Sun et al. Multi-exit self-distillation with appropriate teachers. FITEE, 2024.",
 "S. Teerapittayanon et al. BranchyNet: fast inference via early exiting from deep neural networks. ICPR, 2016.",
 "B. Bross et al. Versatile Video Coding. IEEE TCSVT, 2021.",
 "F. Yang et al. Slimmable compressive autoencoders for practical neural image compression. CVPR, 2021.",
 "M. A. Yılmaz et al. Slimmable video codec. CVPRW, 2022.",
 "A. Kuznetsova et al. The Open Images Dataset V4. IJCV, 2020.",
 "A. Mercat et al. UVG dataset: 50/120fps 4K sequences for video codec analysis. ACM MMSys, 2020.",
 "H. Wang et al. MCL-JCV: a JND-based H.264/AVC video quality assessment dataset. ICIP, 2016.",
]


def build(out="paper/FLEX-UF.pdf"):
    PW, PH = letter
    M, GAP = 0.62 * inch, 0.28 * inch
    colw = (PW - 2 * M - GAP) / 2
    # Title block plus the full-width teaser, CVPR style. Everything after
    # page 1 is two columns, which needs an explicit NextPageTemplate -- without
    # it reportlab keeps using the first template and every page gets a
    # full-width band across the top.
    top_banner = 5.05 * inch

    doc = BaseDocTemplate(str(R / out), pagesize=letter,
                          leftMargin=M, rightMargin=M,
                          topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                          title="Where to Stop: Tile-Adaptive Early Exit in a "
                                "Learned Image Decoder",
                          author="FLEX-UF")
    H = PH - 1.4 * inch
    # page 1: full-width banner (title + teaser), then two columns
    f_ban = Frame(M, PH - 0.7 * inch - top_banner, PW - 2 * M, top_banner,
                  id="ban", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    h1c = H - top_banner
    f_l1 = Frame(M, 0.7 * inch, colw, h1c, id="l1", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    f_r1 = Frame(M + colw + GAP, 0.7 * inch, colw, h1c, id="r1", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    f_l = Frame(M, 0.7 * inch, colw, H, id="l", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
    f_r = Frame(M + colw + GAP, 0.7 * inch, colw, H, id="r", leftPadding=0,
                rightPadding=0, topPadding=0, bottomPadding=0)

    def num(canv, d):
        canv.saveState()
        canv.setFont("Times-Roman", 8)
        canv.setFillColor(colors.HexColor("#666666"))
        canv.drawCentredString(PW / 2, 0.42 * inch, str(canv.getPageNumber()))
        canv.restoreState()

    doc.addPageTemplates([
        PageTemplate(id="first", frames=[f_ban, f_l1, f_r1], onPage=num),
        PageTemplate(id="rest", frames=[f_l, f_r], onPage=num)])

    banner_cap = ("<b>Figure 1. FLEX-UF end to end.</b> Grey is frozen and "
                  "never touched; blue is inherited from DCVC-UF and "
                  "fine-tuned; orange and green are new. The first j block "
                  "groups run over the whole frame, the rest run per tile on a "
                  "shrinking active set, and the exit map that drives the "
                  "shrinkage comes either from an encoder-side search (A) or "
                  "from a decoder-side predictor (B).")
    story = [
        Paragraph("Where to Stop:<br/>Tile-Adaptive Early Exit in a Learned "
                  "Image Decoder", TITLE),
        Paragraph("Anonymous CVPR submission &nbsp;&nbsp;·&nbsp;&nbsp; Paper ID "
                  "****", AUTH),
    ]
    story += fig("sys_pipeline.png", PW - 2 * M, banner_cap,
                 maxh=2.95 * inch)
    story += [NextPageTemplate("rest"), FrameBreak()]
    story += content(colw, PW - 2 * M)
    story.append(Paragraph("References", H1))
    for i, r in enumerate(REFS, 1):
        story.append(Paragraph(f"[{i}] {r}",
                               S("ref", fontSize=7.2, leading=8.4, spaceAfter=2)))
    doc.build(story)
    print(f"  -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "paper/FLEX-UF.pdf")
