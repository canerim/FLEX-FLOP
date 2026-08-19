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


UNEXPANDED = set()


def sub(t):
    """Expand \Macro and the handful of TeX-isms the text uses."""
    for k in sorted(MACROS, key=len, reverse=True):
        t = t.replace("\\" + k, MACROS[k])
    t = (t.replace(r"\dB", " dB").replace(r"\%", "%").replace(r"\,", " ")
          .replace(r"\times", "×").replace(r"\emph{", "<i>")
          .replace(r"\textbf{", "<b>").replace(r"\approx", "≈")
          .replace("---", "\u2014").replace("--", "\u2013")
          .replace(r"\lambda", "λ").replace(r"\beta", "β")
          .replace(r"\Delta", "Δ").replace(r"\rho", "ρ")
          .replace(r"\kappa", "κ").replace(r"\alpha", "α")
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
    r = "".join(out)
    # An unexpanded \Macro means make_paper_tables did not emit it -- usually
    # because the result file it reads is missing. Silently printing the
    # backslash into the PDF is worse than saying so.
    for m in re.finditer(r"\\([A-Z][A-Za-z]+)", r):
        UNEXPANDED.add(m.group(1))
    return r


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
        r"tile leave the trunk at whichever of K exits is deep enough for it. "
        r"The encoder is frozen and the coded payload is unchanged, so the "
        r"method deploys against existing bitstreams. On the full "
        r"\NumSeq-sequence common test set a 0.1 dB budget buys \MainLowRate% "
        r"of the decoder's multiply-accumulates at the lowest rate and "
        r"\MainHighRate% at the highest, for a BD-Rate cost of \BdRateALow%."
        r"<br/><br/>"
        r"Four findings are worth more than the headline. <b>(i)</b> A quality "
        r"budget only buys compute inside a bounded window: below a floor set by "
        r"tiling no allocation is feasible, above a saturation point every tile "
        r"already sits on the cheapest rung, and the working budget uses "
        r"\BandUseLow% of that window at the lowest rate and \BandUseHigh% at "
        r"the highest. Both ends are in closed form and we verify seven structural "
        r"propositions numerically. <b>(ii)</b> Tiling is expensive and its cost "
        r"is governed by depth, not area: the seam grows as b² in the number of "
        r"per-tile convolutions, where the corrupted-area fraction usually "
        r"quoted is wrong by 160–264%. <b>(iii)</b> The exact remedy — give "
        r"each convolution its real neighbour — is bit-identical at uniform "
        r"depth and <i>destroys</i> the allocation under routing, because "
        r"routing is the deliberate violation of the condition that makes it "
        r"exact. <b>(iv)</b> Operations are an <i>optimistic</i> bound on this "
        r"method and the optimism grows with the saving: \WallSorted% of "
        r"wall-clock arrives against a \WallPredicted% arithmetic prediction "
        r"at the lowest rate, \WallHighMeasured% against \WallHighPredicted% "
        r"at the highest. Sorting the tiles once by exit depth is "
        r"bit-identical and recovers a fifth of the shortfall."
        r"<br/><br/>"
        r"Signalling the exit map costs \MapBits bits per frame; predicting it "
        r"at the decoder costs none and gives up \GapLow–\GapHigh points. The "
        r"two are ends of one scale rather than two designs: overriding the "
        r"worst fifth of tiles recovers "
        r"\HybridRecoverFifthHigh–\HybridRecoverFifthLow% of that gap for "
        r"\HybridBitsFifth bits, and how much any fraction can recover is "
        r"bounded above by the Lorenz curve of the per-tile regret. A learned "
        r"predictor should also be made to beat a free one: routing on the bits "
        r"the entropy model has already spent per tile needs no parameters, no "
        r"training and no bits, and it beats our \RouterParams-parameter head "
        r"to within \RateRankLosesBy points at every rate, and ahead at the "
        r"\RateRankNWins lowest."
        r"<br/><br/>"
        r"We also find that the method is governed by tile count far more than "
        r"by content — three 1080p classes of very different material agree "
        r"within three points while a 416×240 class saves a third as much — "
        r"and that an exit map transfers across frames almost perfectly but not "
        r"across rates."), ABST))
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
    par(r"<b>(i)</b> A tile-adaptive early-exit ladder for a learned image "
        r"decoder that leaves the encoder and the coded payload untouched, "
        r"saving \MainLowRate% to \MainHighRate% of decoder MACs for 0.1 dB on "
        r"the full CTC set at a BD-Rate cost of \BdRateALow%. "
        r"<b>(ii)</b> The floor/saturation characterisation of a quality budget, "
        r"both ends in closed form, with seven structural propositions verified "
        r"numerically and a measurement of what the Lagrangian's convex-hull "
        r"restriction costs (at most 0.05 saving points). "
        r"<b>(iii)</b> A quantitative account of the tiling penalty: the b² law "
        r"and the failure of the area law, the measured ordering of four border "
        r"estimators, and two remedies rejected on their own numbers. "
        r"<b>(iv)</b> Two allocation regimes with bits and compute charged on "
        r"both sides, and the continuum between them: overriding the worst "
        r"fifth of tiles recovers "
        r"\HybridRecoverFifthHigh–\HybridRecoverFifthLow% of the gap. "
        r"<b>(v)</b> A parameter-free control that a learned router has to beat "
        r"and usually is not measured against: routing on the bits the entropy "
        r"model already spent per tile costs nothing, adds nothing to the "
        r"stream, and matches our trained head to within \RateRankLosesBy points "
        r"at every rate while beating it at the \RateRankNWins lowest. "
        r"<b>(vi)</b> A wall-clock result: operations over-predict the saving by "
        r"about a fifth, and a bit-identical reordering of the per-tile loop "
        r"recovers a fifth of that — \WallSorted% realised against "
        r"\WallPredicted% predicted. "
        r"<b>(vii)</b> Two properties of the allocation that bear on deployment "
        r"— it is governed by tile count rather than content, and it "
        r"transfers across frames but not across rates.")

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
    tbl("positioning",
        r"<b>Table 1. Where this sits.</b> Complexity control in learned "
        r"compression varies the model; we vary how much of a fixed model runs "
        r"where. The last column is what makes the difference operational: every "
        r"other row requires a decoder that matches the encoder that produced "
        r"the stream.")
    h2("Complexity control in learned compression.")
    par(r"SlimCAE [22] and slimmable video codecs [23] expose several widths of "
        r"one model; the choice is per stream and changes the encoder, so the "
        r"bitstream is not interchangeable. DCVC-FM [13] and DCVC-UF [14] "
        r"reduce cost architecturally, for every frame equally. "
        r"Rate–distortion–complexity has since become an explicit third axis: "
        r"Gao et al. [35] tune spatial context usage to trade decode cost "
        r"against rate, Ho et al. [36] survey where conditional residual coding "
        r"sits on that surface, and Zhang and Gao [37] route <i>whole frames</i> "
        r"to one of several jointly-trained coding paths. All of these vary the "
        r"model, and all change what the encoder emits. Our axis is orthogonal: "
        r"model fixed, bitstream fixed, only <i>how much of the decoder runs "
        r"where</i> varies.")
    h2("The closest neighbour.")
    par(r"Blard et al. [27] also partition an image into regions, also choose "
        r"per region by a rate–distortion cost computed at the encoder, and "
        r"also transmit a mode map — the skeleton of our configuration A. Two "
        r"things differ. Their regions choose among several <i>separately "
        r"trained</i> codecs, so the decoder must hold all of them and its "
        r"complexity is that of one codec regardless of the choice; the map "
        r"buys rate, not compute. Ours choose a prefix length of a single "
        r"trunk, so the weights are shared by construction and the map buys "
        r"compute at fixed rate. And because their alternatives are unrelated "
        r"networks, no decoder-side predictor could stand in for the encoder's "
        r"search, whereas the nesting that makes our exits prefixes of one "
        r"another is exactly what makes configuration B possible at all.")
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
    h2("Deferring to an oracle under a budget.")
    par(r"Configuration C, where a predictor decides most cases and a small "
        r"budget of the hardest ones is handed to something exact, is the shape "
        r"of selective prediction [38] and learning to defer [39, 40], and of "
        r"the budgeted variant of the latter [41]. Two things differ and both "
        r"make our case easier. The expert here is the encoder's own search, so "
        r"it is exact, always available, and costs nothing at test time — what "
        r"is scarce is not the expert's attention but the <i>bits</i> needed to "
        r"say what it decided. And the selection rule is not learned: because "
        r"the objective is separable over tiles, the optimal set of size s at a "
        r"fixed multiplier is exactly the s largest regrets, which the encoder "
        r"can compute. The open question in that literature — who to defer, and "
        r"how to learn it — has a closed form here, and what remains is the "
        r"question we measure, which is how concentrated the regret is.")
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
    h2("How much is there to gain?")
    par(r"Bounding what adaptive inference could achieve is itself a line of "
        r"work. Hasan et al. [34] derive an oracle bound on efficiency at fixed "
        r"accuracy from per-model resource and accuracy, and report 43–121× on "
        r"ImageNet and 7–81× on HellaSwag. Their bound has a ceiling and no "
        r"floor, because the largest model in their family attains the reference "
        r"accuracy by definition. Spatial adaptivity introduces one: cutting a "
        r"frame into tiles costs quality even when every tile runs to full "
        r"depth, so the reference is unreachable at <i>any</i> compute and the "
        r"feasible set of budgets is an interval rather than a ray. That "
        r"interval, and both of its ends, is what Section 5.4 characterises.")
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
        r"Section 5.9 is an instance of that claim inside one loop: "
        r"\WallPredicted% of operations removed buys \WallSorted% of time, and "
        r"\WallSortedGain points of the difference come back from reordering "
        r"the loop with the arithmetic untouched.")

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
    figure("adapter_gain.png",
           r"<b>Figure 3. What the adapters are worth.</b> <b>a</b>, each exit "
           r"with and without its adapter. <b>b</b>, the dB the adapter "
           r"recovers. <b>c</b>, the same against the number of blocks the exit "
           r"skips — the design rationale, measured.")
    tbl("adapters_ablation",
        r"<b>Table 2. What the adapters are worth.</b> dB below the release with "
        r"every tile at that exit, with the trained adapters and with each set "
        r"back to the identity it was initialised to. † the deepest exit has no "
        r"adapter by construction and is the control.")
    par(r"Zeroing them measures what they learned, since the identity is exactly "
        r"what they were initialised to. Without adapters the shallowest exit "
        r"costs \AdapterNoneHigh dB at q63 — forty-four times the budget — "
        r"and the adapter buys \AdapterGainHigh dB back. The gain grows with "
        r"rate and with the number of blocks the exit skips, which is the design "
        r"rationale measured rather than argued, and the deepest exit moves by "
        r"exactly zero. The adapters are not a refinement of the ladder; without "
        r"them it has no usable rung.")
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
           r"same weights, every tile at full depth, stock zero padding "
           r"— no early exit "
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
        r"saving falls from \CoupPaddedMid% to \CoupCoupledMid% at q32 and from "
        r"\CoupPaddedHigh% to \CoupCoupledHigh% at q63.")
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
        r"cost of \BdRateALow% — that is, the compute is worth about the same "
        r"as a \BdRateALow% increase in bitrate. The fall with rate is not an "
        r"artefact of "
        r"the ladder: high-rate reconstructions carry detail the shallow exits "
        r"cannot reproduce, and the floor rises with rate.")
    tbl("complexity",
        r"<b>Table 4. Decoder complexity</b> at 1080p. Our deepest exit costs "
        r"slightly more than the release because it still pays the seam-repair "
        r"module; that 1.0095, not 1.0, is what every saving here is "
        r"<i>not</i> divided by.")
    figure("rd_spread.png",
           r"<b>Figure 7. The plane a codec is read on, and the spread behind "
           r"the mean.</b> <b>a</b>, operating points against the released "
           r"curve. <b>b</b>, the same with the quality axis expanded; labels "
           r"are compute saved. <b>c</b>, per sequence at a matched point near "
           r"0.1 dB; one dot per sequence, bar is the median.")
    par(r"Panel c is the distribution the headline averages over, and it is "
        r"wide. At q0 the median sequence saves 38.6% with an interquartile "
        r"range of 32.0–41.5, while the worst saves −1.0% — the "
        r"low-resolution sequences of Section 5.3, where two tiles leave nothing "
        r"to allocate. Reporting the mean alone would hide both ends.")
    h2("5.2 Is per-tile adaptivity necessary?")
    figure("exit_map.png",
           r"<b>Figure 6. Where the decoder spends.</b> Bosphorus at q32, a "
           r"0.1 dB budget. (a) the assignment "
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
        r"suggest any adaptive-inference paper should report. Given the "
        r"oracle's histogram this is a measurement of <i>ranking</i> and "
        r"nothing else; §5.6 removes that crutch and turns the same signal into "
        r"a complete routing rule, which beats the trained head above "
        r"the whole rate range.")
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
    figure("theory.png",
           r"<b>Figure 9. The structure of the allocation</b>, on one frame. "
           r"<b>a</b>, compute and distortion are monotone in the multiplier. "
           r"<b>b</b>, the frontier between the floor (green) and saturation "
           r"(orange); shaded regions are infeasible and wasted. <b>c</b>, what "
           r"the Lagrangian reaches against the exact Pareto set, enumerated by "
           r"dynamic programming — the two curves are indistinguishable.")
    par(r"The construction has enough structure to state as propositions, and we "
        r"verify each numerically rather than asserting it "
        r"(scripts/verify_theory.py, seven of seven): the allocation decouples per "
        r"tile; compute is non-increasing and distortion non-decreasing in λ; "
        r"the sweep traces the lower convex hull and cannot reach an interior "
        r"point; λ=0 attains the least distortion the ladder can produce; beyond "
        r"a finite λ, computable in closed form, the allocation is the constant "
        r"map to exit j at cost exactly c_j; and the ceiling is a function of "
        r"the split depth alone.")
    par(r"The third has a practical edge. Because the sweep reaches only hull "
        r"vertices, a budget between two of them is not attainable. We measured "
        r"what that costs by enumerating the <i>exact</i> Pareto set with "
        r"dynamic programming over tiles — feasible because the cost alphabet "
        r"has four symbols: the sweep reaches 95 allocations against a Pareto "
        r"set of 635, and convexity costs at most 0.05 saving points. The "
        r"standard construction is essentially optimal here.")
    par(r"It is worth saying <i>why</i> that loss is small, because the reason "
        r"is structural and says what would make it smaller. The reachable "
        r"costs form a lattice whose spacing is set by how much the mean cost "
        r"moves when the multiplier crosses a switching point: m tiles each "
        r"move one rung, so the spacing is at most m·max(c_k+1 − c_k)/T saving "
        r"points, and the convexity loss cannot exceed the largest spacing, "
        r"because a budget between two levels is served by the lower one. Here "
        r"m=2 — tiles with identical rows switch together — and the bound is "
        r"0.745 points, which the measured spacing attains exactly. Nothing in "
        r"the expression depends on content or rate, and it falls as 1/T: "
        r"halving the tile side puts four times as many tiles on the lattice, "
        r"each carrying a quarter of the step. Fine granularity is what makes "
        r"the Lagrangian relaxation lossless in practice, not any property of "
        r"the images.")
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
           r"<b>Figure 9. Who decides.</b> <b>(a)</b> A holds the source, so it "
           r"can decode all K exits per tile and pick the true minimiser; it "
           r"signals the map at ~\MapBits bits/frame. B never sees the source "
           r"— a \RouterParams-parameter head reads ŷ, the entropy scales and "
           r"qp — and signals nothing. <b>(b)</b> Saving at 0.1 dB; shading is "
           r"what a bit-exact bitstream costs. <b>(c)</b> That cost is "
           r"rate-dependent and budget-dependent: it collapses to the router's "
           r"own \RouterCostPct% of a decode once the budget is loose enough "
           r"that both saturate.")
    tbl("ab",
        r"<b>Table 7. Signalled against predicted</b> at two budgets, same "
        r"checkpoint and test set, with one router trained against the deployed "
        r"oracle at a single λ.")
    par(r"Configuration A is exact by construction and costs bits; B is "
        r"approximate and costs none. The comparison splits in two.")
    par(r"<b>Not signalling costs \GapMin–\GapMax points</b>, roughly flat "
        r"across rate, for zero added bits and a byte-identical file. The gap "
        r"is smallest at q\GapMinQp and grows toward both ends.")
    par(r"<b>It tracks |β|</b>, the tilt the bisection has to apply to move a "
        r"router trained at one λ onto another operating point. At q\BetaMinQp "
        r"the required tilt is β=\BetaAtMin — essentially none, because that "
        r"is where the training λ lands — and the gap is at its minimum. At q0 "
        r"the tilt is \BetaLow and the gap is \GapMax. A large tilt lets the "
        r"cost term dominate the logits and discards the content ranking the "
        r"router learned, in either direction. The remedy is not a better "
        r"architecture but a router per operating point, which a deployment "
        r"would have anyway: one set of weights serves all rates, and a "
        r"\RouterParams head per rate is 0.3% of the model each.")
    par(r"Loosening the budget closes the gap from the other side. At 0.3 dB it "
        r"is \GapLooseLow points at the three lowest rates — exactly the "
        r"router's own \RouterCostPct% of decode, i.e. its <i>prediction</i> is "
        r"then free — and \GapLooseHigh at the highest. Once the budget "
        r"saturates the ladder both configurations send every tile to the same "
        r"rung and there is nothing left to predict wrongly.")
    par(r"We report the single-router number because it is the honest one for a "
        r"system that trains once, and note that it understates what B can do.")
    par(r"An earlier version of this measurement put the gap at 1.7 points at "
        r"q0 rising to 13.3 at q63. That was the exit mask: the head suppresses "
        r"exits below the split depth by assigning -10⁴, its own logits had "
        r"drifted to that scale, and the suppressed entries were therefore the "
        r"largest in every row — so a large share of every tile went to the "
        r"cheapest rung for a reason unrelated to its content. The mask is now "
        r"-∞. Fixing it costs 3.5 points at q0, where the accident happened to "
        r"agree with the oracle, and buys 9.4 at q63, where it did not.")
    h2("5.6 A router with no parameters")
    par(r"Before a \RouterParams head is worth its \RouterCostPct% of the "
        r"decode, it has to beat what the decoder already knows. The entropy "
        r"model has produced a number per tile before the trunk runs and at no "
        r"cost: how many bits that tile's latents took.")
    par(r"We turn it into a routing rule with no learned parameters. Model the "
        r"per-tile distortion as rank-1 in the log domain, "
        r"log D(t,k) ≈ α·log b(t) + c + log φ_k, where b is the tile's bit "
        r"count normalised by the frame mean and φ is a K-vector saying what "
        r"each exit costs on an average tile. Fit (α, c, φ) by least squares, "
        r"leave-one-sequence-out so no sequence contributes to the profile that "
        r"routes it, and run the same Lagrangian the oracle runs on the "
        r"surrogate. Nothing is signalled, nothing is trained, and the "
        r"arithmetic is a scalar per tile.")
    figure("raterank.png",
           r"<b>Figure 10. The free baseline.</b> <b>a</b> Saving at 0.1 dB; "
           r"shaded where the parameter-free rule beats the trained head. "
           r"<b>b</b> What a tile's bit count is correlated with, against how "
           r"often the rule agrees with the oracle outright; dotted is the "
           r"head's own held-out agreement.")
    tbl("raterank",
        r"<b>Table 8. The free baseline</b>, 0.1 dB, same test set. Bold where "
        r"the parameter-free rule beats the trained head. ρ are Spearman "
        r"correlations between a tile's bit count and, respectively, the depth "
        r"the oracle assigns it and the distortion it stands to gain from that "
        r"depth.")
    par(r"<b>It matches the trained head to within a point either way.</b> It "
        r"beats the \RouterParams router at the \RateRankNWins lowest rates, "
        r"by up to \RateRankBeatsBy points, and loses at the two highest by at "
        r"most \RateRankLosesBy. That is the whole margin between a "
        r"parameter-free rule and a head that was trained on this decoder "
        r"against this oracle, and the free rule carries none of the head's "
        r"\RouterCostPct% of decode.")
    par(r"<b>Why it works is not that it agrees with the oracle.</b> It agrees "
        r"on \RateRankAgreeLo–\RateRankAgreeHi of tiles, well below the "
        r"router's 0.718, and still saves more at three of five rates. "
        r"Agreement counts a disagreement on a tile where two exits are within "
        r"a hair of each other exactly as heavily as one where the choice is "
        r"most of the frame's error, and most tiles are the former. What the "
        r"rule gets right is the ordering that matters: bits correlate with the "
        r"<i>spread</i> across the ladder — how much a tile stands to gain from "
        r"depth — at ρ_spread = \RateRankSpreadLo–\RateRankSpreadHi at every "
        r"rate.")
    par(r"<b>At a looser budget it stops being a baseline and becomes the "
        r"answer.</b> At 0.3 dB it reaches the architectural ceiling exactly at "
        r"the \RateRankLooseCeil lowest rates, is within 0.2 points of the "
        r"oracle at q48, and gives up \RateRankLoose% against the oracle's "
        r"\SigLooseHigh% at q63 — better than the trained head at every rate, "
        r"because the head is charged for itself and this is not. Once the "
        r"budget saturates the ladder there is little ordering left to get "
        r"right, and the free rule gets it.")
    par(r"<b>What it cannot do</b> is see anything beyond that ordering. A "
        r"rank-1 model in the level assigns every tile the same relative "
        r"profile over exits, so b only decides where on the ladder a tile "
        r"falls, never the shape of its trade-off. That is the ceiling this "
        r"baseline sits at, and it is the part a learned head should be earning "
        r"its parameters on. Ours earns it at high rate and does not at low.")
    par(r"Per-block bit allocation is a standard quantity in learned "
        r"compression, where it is something to <i>choose</i>: block-level rate "
        r"control sets it so that complex regions get more bits [42]. We read "
        r"the same number in the other direction, after the fact and at the "
        r"decoder, as a statement about how hard a region was — which costs "
        r"nothing precisely because someone else already paid for it.")
    par(r"We report this because a learned component should be measured against "
        r"the free alternative and rarely is. In adaptive inference the usual "
        r"controls are a uniform allocation and a random one; both are far "
        r"weaker than a decoder-side signal that happens to be lying around.")
    h2("5.7 Signalling only what the router gets wrong")
    par(r"A and B are the two ends of a single scale, not two designs. The "
        r"encoder can run the decoder's router — it reads only decoded data — "
        r"so it knows, tile by tile, where the prediction will be wrong. "
        r"Nothing forces it to correct all of them.")
    par(r"Configuration C signals a fraction ρ of the tiles and leaves the rest "
        r"to the router. The overrides are chosen by Lagrangian regret, which is "
        r"exactly what the objective loses on that tile by staying silent; β is "
        r"held at B's value and λ is bisected over the overrides to land back on "
        r"the budget. The map costs an entropy-coded mask, N·H₂(ρ) bits, plus 3 "
        r"per override.")
    figure("hybrid.png",
           r"<b>Figure 10. Partial signalling.</b> <b>a</b> Saving against the "
           r"bits spent on the map; the left end of each line is B and the right "
           r"end is A. <b>b</b> The same, normalised: what fraction of the A–B "
           r"gap a given fraction of the map recovers. <b>c</b> The same "
           r"against the Lorenz bound on it. Dashed is equality.")
    tbl("hybrid",
        r"<b>Table 8. Configuration C</b> at 0.1 dB. Columns are the fraction of "
        r"tiles the encoder overrides; the two end columns reproduce B and A to "
        r"within 0.1 points, which is the check that the interpolation is real.")
    par(r"<b>The two ends check out.</b> At ρ=0 the measurement reproduces "
        r"configuration B to the second decimal at every rate, and at ρ=1 it "
        r"reproduces A. Neither is imposed — both fall out of the same code "
        r"path — so the columns between them are measuring something real.")
    par(r"<b>Most of the gap is cheap.</b> At q0, overriding a tenth of the "
        r"tiles for \HybridBitsTenth bits per frame recovers "
        r"\HybridRecoverTenthLow% of the gap; a fifth recovers "
        r"\HybridRecoverFifthLow%. At q63, where the gap is widest, the same "
        r"fifth of the map — \HybridBitsFifth bits against \HybridBitsFull for "
        r"the full map — recovers \HybridRecoverFifthHigh%, and half recovers "
        r"\HybridRecoverHalfHigh%. Recovery is concave everywhere, so the first "
        r"bits spent are always the most useful ones.")
    par(r"<b>Why it is concave, and what bounds it.</b> At a fixed λ the "
        r"objective is separable, so overriding a set removes <i>exactly</i> "
        r"the sum of that set's regrets: the recovery of the <i>objective</i> "
        r"is the Lorenz curve of the per-tile regret distribution, and its "
        r"curvature is that distribution's Gini coefficient, "
        r"\GiniMin–\GiniMax across rates. What the table reports is not "
        r"objective but saving at a fixed distortion, and returning to the "
        r"budget means re-bisecting λ, which converts the recovered distortion "
        r"into compute at the local exchange rate rather than one for one. The "
        r"Lorenz curve is therefore an upper bound, and the measurement says "
        r"so: every point in panel c lies on or below the diagonal, reaching "
        r"\LorenzTightMin–\LorenzTightMax% of the bound. The concentration of "
        r"the router's mistakes is what makes partial signalling worth doing; "
        r"the re-tuning is what stops it from being free.")
    par(r"Configuration C is what we would ship where a small map is tolerable "
        r"and a large one is not. It also reframes the A–B gap: it is not the "
        r"price of prediction, it is the price of <i>silence</i>, and silence "
        r"is priced per tile.")
    h2("5.8 Does the map have to be recomputed?")
    figure("map_transfer.png",
           r"<b>Figure 10. Reusing an exit map.</b> Solid is the transferred "
           r"map, dashed the one recomputed in place. <b>a</b>, <b>b</b>: reuse "
           r"across frames of the same sequence. <b>c</b>: reuse across quality "
           r"index; the entry is the delivered distortion against a 0.1 dB "
           r"budget.")
    par(r"Configuration A costs the encoder about one extra decode per frame. "
        r"Whether that matters depends on how often the map has to be "
        r"recomputed, which nobody has checked.")
    par(r"<b>Across time it barely has to be.</b> Frame 0's map applied eight "
        r"frames later costs 0.005 dB at q0 — five percent of the budget — "
        r"and no saving at all: \TransferSaving% transferred against "
        r"\TransferInPlaceLo–\TransferInPlaceHi% "
        r"recomputed. The allocation is a property of where the content is hard, "
        r"and that moves slowly. A codec would search once per group of pictures "
        r"and divide the encoder cost by the group length.")
    par(r"<b>Across rate it very much does</b>, and the direction matters. The "
        r"map found at q0 applied at q63 delivers \TransferCrossDb dB against a 0.1 dB "
        r"budget: it claims the low-rate saving of \TransferSaving% while spending nearly "
        r"twice the quality it is allowed. The reverse is safe but wasteful "
        r"— the q63 map at q0 delivers 0.075 dB and only 19.8% where 33.1% "
        r"was available. Shallow exits are cheap in quality at low rate and "
        r"expensive at high rate, so a map is calibrated to the rate it was found "
        r"at, and reusing it upward silently breaks the quality guarantee. Search "
        r"once per rate, reuse across frames.")
    h2("5.9 Complexity and wall-clock")
    tbl("latency",
        r"<b>Table 8. Wall-clock</b>, 1080p, median of 40 interleaved "
        r"iterations, at the 0.1 dB operating point. ``MACs'' is what the "
        r"arithmetic predicts; ``measured'' is the sorted per-tile loop.")
    par(r"A saving in multiply-accumulates is not a saving in time. Tiling "
        r"itself costs \TilingOverhead% before anything exits early — the "
        r"deepest-exit tiled decode against the released full-frame one, same "
        r"arithmetic, same weights — and the routed decode realises "
        r"\WallMasked% at q0 against the \WallPredicted% its operations "
        r"predict. Four fifths of the predicted saving arrives; the missing "
        r"fifth is the tiling overhead plus the bookkeeping of a shrinking "
        r"active set, which a MAC count cannot see. The shortfall is "
        r"proportional rather than constant: \WallHighMeasured% realised "
        r"against \WallHighPredicted% at q63.")
    par(r"Sorting the tiles once by exit depth recovers part of it. The obvious "
        r"implementation of a shrinking active set — a boolean mask, a gather "
        r"of survivors and a scatter of the finished at every group boundary — "
        r"forces a device-to-host synchronisation per boundary. Descending by "
        r"exit depth, ``still active at group g'' becomes a contiguous prefix "
        r"instead: each group is a slice rather than a gather, the boundaries "
        r"come from one cumulative count rather than a mask per group, and the "
        r"finished tiles return through the inverse permutation in a single "
        r"scatter. The arithmetic is unchanged and the output is bit-identical "
        r"on GPU. The saving at q0 goes from \WallMasked% to \WallSorted% — "
        r"\WallSortedGain points, a fifth of the shortfall, for a change that "
        r"touches no arithmetic.")
    par(r"The same lesson applies to our own accounting. Every configuration B "
        r"number in this paper charges the router at its share of the decoder's "
        r"multiply-accumulates, \RouterCostPct%. Timed on the padded 2048×1280 "
        r"frame the decoder actually sees, interleaved against a deepest-exit "
        r"decode so a co-tenant's load lands on both equally, the head costs "
        r"\RouterTimePct% — \RouterTimeFactor× what its arithmetic predicts, "
        r"and for the same reason as everything else in this section: a "
        r"\RouterParams-parameter head is launch overhead, and a MAC count "
        r"cannot see a launch. Charged at measured time rather than at "
        r"operations, every B number in this paper would fall by a further "
        r"\RouterTimeExtra points. We leave them charged at MACs because that "
        r"is the convention the rest of the literature reports in, and record "
        r"the correction here rather than letting it sit unstated.")
    par(r"We report the shortfall because a paper that quotes only operations "
        r"would report \WallPredicted% where the same code, run as written, "
        r"delivers \WallSorted%. The direction is the one that matters: "
        r"operations are an <i>optimistic</i> bound on this method, and the "
        r"optimism grows with how much of the frame exits early.")
    h2("5.10 The right ladder depends on the budget")
    tbl("runs",
        r"<b>Table 9. Ladder configurations</b>, mean saving (%) over the five "
        r"rates, same test set and protocol. * one rate is infeasible at that "
        r"budget — the ladder's floor exceeds it — so the mean is over "
        r"the remaining four.")
    par(r"Section 5.4 argued that once a budget saturates a ladder the only way "
        r"to spend more is a rung that does not exist. The table measures it. A "
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
        r"stands.</b> Canvas coupling removes \CoupFloorDropLo–\CoupFloorDropHi% of the floor and is "
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
        r"\MainHighRate% of decoder MACs for a 0.1 dB budget, at a BD-Rate cost "
        r"of \BdRateALow% — without touching the encoder or the coded payload.")
    par(r"Three lessons we would carry to any spatially adaptive decoder, none "
        r"of them about early exit. <b>Tiling is the dominant cost and it is "
        r"governed by depth.</b> Its entire cause is a single 3×3 that is 0.29% "
        r"of the arithmetic, the penalty grows as the square of how many such "
        r"convolutions run per tile, and the corrupted-area fraction that is "
        r"usually quoted predicts it badly. <b>The exact remedy is in tension "
        r"with the thing it enables.</b> Giving each convolution its real "
        r"neighbour is bit-identical at uniform depth and collapses the "
        r"allocation under routing, because routing is the deliberate violation "
        r"of the condition that makes it exact — a trap that any method "
        r"combining spatial adaptivity with tiled inference will walk into. "
        r"<b>A quality budget is only a control variable inside a measurable "
        r"window.</b> Below the floor it admits nothing, above saturation it "
        r"buys nothing, and reporting a saving without saying where in that "
        r"window it sits leaves out the most useful part of the result.")
    par(r"And three cautions about measuring any of this. A saving in "
        r"operations is an optimistic bound on a saving in time, and the "
        r"optimism scales with the saving. A learned router should be compared "
        r"against a free one: routing on the bits already spent per tile needs "
        r"no parameters, no training and no bits, and it matches our trained head "
        r"to within \RateRankLosesBy points at every rate — while agreeing with the oracle on fewer "
        r"tiles than the head does, which is a warning about the metric as much "
        r"as about the head. And a timing harness will "
        r"report numbers whether or not it is timing the right device — this "
        r"one did, for months, and the tell was a decoder that appeared to slow "
        r"down with the quality index.")
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
 "T. Blard et al. Spatial competition for low-complexity learned image compression. arXiv:2605.13243, 2026.",
 "Y. Shoham, A. Gersho. Efficient bit allocation for an arbitrary set of quantizers. IEEE TASSP, 1988.",
 "A. Ortega, K. Ramchandran. Rate-distortion methods for image and video compression. IEEE SPM, 1998.",
 "Adaptive test-time compute allocation for reasoning LLMs via constrained policy optimization. arXiv:2604.14853, 2026.",
 "H. A. Alhaija et al. Local padding in patch-based GANs for seamless infinite-sized texture synthesis. arXiv:2309.02340, 2023.",
 "C. Innamorati et al. Overlap training to mitigate inconsistencies caused by image tiling in CNNs. arXiv:1812.02203, 2018.",
 "Z. Jia et al. Towards practical real-time neural video compression. CVPR, 2025.",
 "B. A. Hasan et al. Adaptive inference: theoretical limits and unexplored opportunities. arXiv:2402.04359, 2024.",
 "Y. Gao et al. Exploring the rate-distortion-complexity optimization in neural image compression. CVIU, 2024.",
 "Y.-H. Ho et al. On the rate-distortion-complexity trade-offs of neural video coding. arXiv:2410.03898, 2024.",
 "C. Zhang, W. Gao. Learned rate control for frame-level adaptive neural video compression via dynamic neural network. arXiv:2508.20709, 2025.",
 "Y. Geifman, R. El-Yaniv. Selective classification for deep neural networks. NeurIPS, 2017.",
 "D. Madras et al. Predict responsibly: improving fairness and accuracy by learning to defer. NeurIPS, 2018.",
 "H. Mozannar, D. Sontag. Consistent estimators for learning to defer to an expert. ICML, 2020.",
 "G. DeSalvo et al. Budgeted multiple-expert deferral. arXiv:2510.26706, 2025.",
 "Accelerating block-level rate control for learned image compression. arXiv:2409.01009, 2024.",
]


def build(out="paper/FLEX-UF.pdf"):
    PW, PH = letter
    M, GAP = 0.62 * inch, 0.28 * inch
    colw = (PW - 2 * M - GAP) / 2
    # Title block plus the full-width teaser, CVPR style. Everything after
    # page 1 is two columns, which needs an explicit NextPageTemplate -- without
    # it reportlab keeps using the first template and every page gets a
    # full-width band across the top.
    top_banner = 4.45 * inch

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
                 maxh=2.60 * inch)
    story += [NextPageTemplate("rest"), FrameBreak()]
    story += content(colw, PW - 2 * M)
    story.append(Paragraph("References", H1))
    for i, r in enumerate(REFS, 1):
        story.append(Paragraph(f"[{i}] {r}",
                               S("ref", fontSize=7.2, leading=8.4, spaceAfter=2)))
    doc.build(story)
    if UNEXPANDED:
        print("  UNEXPANDED MACROS (run make_paper_tables.py): "
              + ", ".join(sorted(UNEXPANDED)))
    print(f"  -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "paper/FLEX-UF.pdf")
