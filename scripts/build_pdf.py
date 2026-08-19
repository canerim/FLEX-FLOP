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
                                KeepTogether, FrameBreak, NextPageTemplate,
                                PageBreak)

R = Path(__file__).resolve().parents[1]
# Body frame height, the yardstick for "can this flowable share a column".
FRAME_H = letter[1] - 1.4 * inch
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



# --------------------------------------------------------------- equations
_EQN = [0]
_EQ_DIR = R / "docs" / "figures" / "_eq"


def _render_math(tex: str, fontsize: float = None, dpi: int = 600) -> tuple:
    """LaTeX math -> a tight transparent PNG, and its size in points.

    reportlab has no math engine, so display equations were being written as
    bold body text ("k*(t) = argmin_k [ D(t,k) + lambda c_k ]"), which is not
    what an equation looks like in a CVPR paper. matplotlib's mathtext renders
    the same source main.tex uses.

    Two things have to match the page or the equation reads as a pasted-in
    picture, which is what it was. `stix` is the Times-metric math font; the
    default `dejavusans` set a sans-serif equation in a Times paper. And the
    size is the body size -- mathtext points are typographic points and the PNG
    is placed at its natural size (w/dpi*72), so a fontsize of 11 against 8.6pt
    body rendered every equation 28% oversized.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = BODY.fontSize
    _EQ_DIR.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-zA-Z0-9]+", "_", tex)[:60] or "eq"
    out = _EQ_DIR / f"{key}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="black")
    # The pad is what separates glyph from crop, and at 600 dpi 0.02in is 1.4pt
    # of slack on every side that the centring afterwards has to guess at.
    fig.savefig(out, dpi=dpi, transparent=True, bbox_inches="tight",
                pad_inches=0.005)
    plt.close(fig)
    from PIL import Image as PILImage
    w, h = PILImage.open(out).size
    return str(out), w / dpi * 72.0, h / dpi * 72.0


# ---------------------------------------------------------------- styles
def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=8.6, leading=10.4,
                alignment=TA_JUSTIFY, spaceAfter=4)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = S("body")
ABST = S("abst", fontSize=8.4, leading=10.0)
# Headings are deliberately NOT keepWithNext. It is the right rule in a
# single-column flow, but here the thing after a heading is usually a figure or
# a table block of 100-160pt, so gluing them moves both: it took the short
# columns from 2 to 7 (worst 55.4pt -> 105.1pt), split Table 7 from its caption
# and still left a heading stranded. Three headings end a column; that is the
# cheaper defect and it predates this pass.
H1 = S("h1", fontName="Times-Bold", fontSize=11.5, leading=13,
       spaceBefore=9, spaceAfter=4, alignment=0)
H2 = S("h2", fontName="Times-Bold", fontSize=9.6, leading=11,
       spaceBefore=7, spaceAfter=3, alignment=0)
CAP = S("cap", fontSize=7.4, leading=8.8, spaceAfter=6)
# \abovedisplayskip/\belowdisplayskip: one body size of air on each side.
DISPLAY_SKIP = BODY.fontSize
TITLE = S("title", fontName="Times-Bold", fontSize=17, leading=20,
          alignment=TA_CENTER, spaceAfter=6)
AUTH = S("auth", fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=12)


# ------------------------------------------------------- LaTeX table -> flowable
CELL_PAD = 3.0   # per side; reportlab's default 6 spent 48pt of a 251pt column


def _col_widths(text_rows, ncol, width, fs):
    """Column widths taken from the text that has to fit in them.

    The old rule handed the first column a flat 34% (28% or 20% for wider
    tables) and split the rest evenly, which is blind to what is in them. In
    `static` that gave the q column 73pt of text space for 7pt of digits and
    left Allocation 43pt for a 66pt label, so all 35 rows wrapped to two lines
    and the table stood 599.6pt tall in a 691.2pt frame -- tall enough that it
    could never share a column with anything, which is where page 6's blank
    half came from. Measured, the whole table needs 208.6pt of the 251.3
    available and nothing has to wrap at all.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth
    nat = [1.0] * ncol
    for r in text_rows:
        for i, c in enumerate(r[:ncol]):
            f = "Times-Bold" if "<b>" in c else "Times-Roman"
            nat[i] = max(nat[i], stringWidth(re.sub(r"<[^>]+>", "", c), f, fs))
    avail = width - 2 * CELL_PAD * ncol
    tot = sum(nat)
    if tot <= avail:
        # Everything fits unwrapped; spend the slack in proportion so the table
        # still spans the column instead of huddling on the left.
        nat = [n + (avail - tot) * n / tot for n in nat]
    else:
        # Genuinely too wide. Shrink the roomiest columns towards a floor and
        # let the longest text wrap, rather than scaling the narrow numeric
        # columns into one character per line.
        floor = min(avail / ncol, 2.6 * fs)
        for _ in range(12):
            free = [i for i, x in enumerate(nat) if x > floor]
            if not free or abs(sum(nat) - avail) < 0.01:
                break
            fixed = sum(x for i, x in enumerate(nat) if i not in free)
            k = (avail - fixed) / sum(nat[i] for i in free)
            for i in free:
                nat[i] = max(floor, nat[i] * k)
    return [n + 2 * CELL_PAD for n in nat]


def tex_table(name, width):
    p = TABLES / f"{name}.tex"
    if not p.exists():
        return Paragraph(f"<i>[{name} not generated]</i>", CAP)
    text_rows, txt = [], p.read_text()
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
        # \cite{key} has no meaning here -- there is no bibtex pass -- and the
        # brace-stripping below turned "SlimCAE~\cite{slimcae}" into
        # "SlimCAE~citeslimcae" in the positioning table.
        line = re.sub(r"~?\\cite\{[^}]*\}", "", line)
        # \mathrm{sat} and friends: keep the text, drop the wrapper. Without
        # this the brace-stripping below prints "D_mathrmsat".
        line = re.sub(r"\\(?:mathrm|mathbf|text|textrm)\{([^}]*)\}",
                      r"\1", line)
        cells = [sub(c.strip()) for c in line.split("&")]
        cells = [re.sub(r"\$([^$]*)\$", r"\1", c) for c in cells]
        cells = [c.replace(r"\_", "_").replace("{", "").replace("}", "")
                 .replace(r"^\ast", "*").replace(r"^\dagger", "†")
                 .replace("\\", "") for c in cells]
        text_rows.append(cells)
    if not text_rows:
        return Spacer(1, 1)
    # One size for the whole table. This used to be recomputed per line, so a
    # short trailing row silently set the font for every row above it.
    ncol = max(len(r) for r in text_rows)
    fs = 7.2 if ncol <= 7 else 6.0
    text_rows = [r + [""] * (ncol - len(r)) for r in text_rows]
    cw = _col_widths(text_rows, ncol, width, fs)
    rows = [[Paragraph(c, S("cell", fontSize=fs, leading=fs * 1.16,
                            alignment=0 if i == 0 else 2))
             for i, c in enumerate(r)] for r in text_rows]
    t = Table(rows, colWidths=cw, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD),
        ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD),
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
    im.spaceBefore = 6
    return [im, Spacer(1, 3), Paragraph(sub(cap), CAP)]


# ---------------------------------------------------------------- the content
def content(colw, fullw):
    """(flowables for the two-column body, flowables for the full-width banner)."""
    F = []
    A = F.append

    def figure_wide(name, cap, height=4.45 * inch):
        """A figure that spans both columns.

        reportlab has no float mechanism, so this switches to a page template
        whose top frame is full width, emits the figure there, and switches
        back. It forces a page break -- which is what LaTeX would do with a
        figure* anyway, and the alternative is a three-panel schematic rendered
        at 3.2 inches wide, which is what it looked like before.
        """
        A(NextPageTemplate("wide"))
        A(PageBreak())
        for f in fig(name, fullw, cap, maxh=height):
            A(f)
        A(FrameBreak())
        A(NextPageTemplate("rest"))

    def par(t):
        A(Paragraph(sub(t), BODY))

    def h1(t):
        A(Paragraph(t, H1))

    def h2(t):
        A(Paragraph(t, H2))

    def tbl(name, cap):
        t = tex_table(name, colw)
        t.spaceBefore = 6
        block = [t, Spacer(1, 3),
                 Paragraph(sub(_autonum(cap, _TABN, "Table")), CAP)]
        # KeepTogether is right for a table that can still fit in a column
        # somebody has already started writing in. Past about half the frame it
        # cannot, so reportlab pushes it whole to the next frame and strands
        # everything above it: `static` is 415.9pt of a 691.2pt frame and left
        # 254.2pt of page 6's left column blank. Over that size let it break
        # across the column the way a longtable does -- repeatRows=1 on the
        # table reprints the header on the far side of the break.
        if t.wrap(colw, FRAME_H)[1] > 0.5 * FRAME_H:
            for f in block:
                A(f)
        else:
            A(KeepTogether(block))

    def eq(tex, tag=True):
        """A display equation centred on the column, with a right-aligned number.

        Three cells, not two. The old layout put the equation in a cell of
        width colw-0.30in and the number in the remainder, so "centred" meant
        centred on the column minus the number -- every equation sat 10.8pt
        left of where the eye expects it. An empty cell of the tag's width now
        balances the tag, which puts the middle cell's centre on the column's.
        """
        _EQN[0] += 1
        tagw = 0.34 * inch
        avail = colw - 2 * tagw
        path, w, h = _render_math(tex)
        if w > avail:
            # Rep(f) renders 308pt wide. In a 230pt cell reportlab centred it
            # and drew the overhang straight over the margin and the gutter --
            # measured ink from x=7.9pt to x=311.8pt in a 44.6..295.9 column.
            h *= avail / w
            w = avail
        im = Image(path, width=w, height=h)
        num = (Paragraph(f"({_EQN[0]})", S("eqnum", fontSize=BODY.fontSize,
                                           leading=BODY.fontSize, alignment=2))
               if tag else "")
        t = Table([["", im, num]], colWidths=[tagw, avail, tagw])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("ALIGN", (2, 0), (2, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        # LaTeX's \abovedisplayskip is the body size; platypus collapses this
        # against the preceding paragraph's spaceAfter (max, not sum), so
        # setting both ends equal is what actually gets equal gaps. The old 4pt
        # cell padding lost to the paragraph's own 4pt above and won below,
        # which is why the equations sat closer to the text under them.
        t.spaceBefore = t.spaceAfter = DISPLAY_SKIP
        A(t)

    def figure(name, cap):
        # As four loose flowables the image and its caption were free to land in
        # different columns, and three of them did: seam_module, budget_band and
        # hybrid each printed their picture at the foot of the left column and
        # its caption at the head of the right. A column figure is one object.
        A(KeepTogether(fig(name, colw, cap)))

    # ---- abstract -------------------------------------------------------
    A(Paragraph("<b>Abstract</b>", S("ah", fontName="Times-Bold", fontSize=9.4,
                                     alignment=TA_CENTER, spaceAfter=4)))
    A(Paragraph(sub(
        r"A learned image decoder spends the same computation on every frame, "
        r"whatever the frame contains. We add an early-exit ladder to the "
        r"intra decoder of DCVC-UF, cut each frame into tiles, and let every "
        r"tile leave the trunk at whichever of K exits is deep enough for it. "
        r"The encoder stays frozen and the coded payload is unchanged, so the "
        r"method can be deployed against bitstreams that already exist. On the "
        r"full \NumSeq-sequence common test set a 0.1 dB budget buys "
        r"\MainLowRate% of the decoder's multiply-accumulates at the lowest "
        r"rate and \MainHighRate% at the highest, for a BD-Rate cost of "
        r"\BdRateALow%."
        r"<br/><br/>"
        r"Four further findings matter more to us than that headline number. "
        r"<b>(i)</b> A quality budget buys compute only inside a bounded "
        r"window. Below a <i>floor</i> set by tiling, no allocation is "
        r"feasible at all; above a <i>saturation</i> point, every tile already "
        r"sits on the cheapest rung. The budget we work to occupies "
        r"\BandUseLow% of that window at the lowest rate and \BandUseHigh% at "
        r"the highest. Both ends are in closed form, and we verify seven "
        r"structural propositions numerically rather than asserting them. The "
        r"window also accounts for most of the rate dependence: five rates "
        r"that lie \BandRawSpread points apart at a matched decibel collapse "
        r"to \BandSpreadMean once the budget is measured in units of each "
        r"rate's own band. <b>(ii)</b> Tiling is expensive, and what it costs "
        r"is set by depth rather than by area. The seam error follows a power "
        r"law in the number of per-tile convolutions with a fitted exponent "
        r"near two, and the corrupted-area fraction that is usually quoted is "
        r"wrong by 160–264%. <b>(iii)</b> The exact remedy is to give each "
        r"convolution its real neighbour. At uniform depth it is "
        r"bit-identical; under routing it <i>destroys</i> the allocation, "
        r"since routing is the deliberate violation of the condition that "
        r"makes the remedy exact. <b>(iv)</b> Operations are an "
        r"<i>optimistic</i> bound on what this method saves, and the optimism "
        r"grows with the saving. At the lowest rate, \WallSorted% of "
        r"wall-clock arrives against a \WallPredicted% arithmetic prediction; "
        r"at the highest, \WallHighMeasured% against \WallHighPredicted%. "
        r"Sorting the tiles once by exit depth is bit-identical and recovers a "
        r"fifth of the shortfall."
        r"<br/><br/>"
        r"Signalling the exit map costs \MapBits bits per frame; predicting it "
        r"at the decoder costs none and gives up \GapMin–\GapMax points. We "
        r"treat the two as ends of one scale. Overriding the worst fifth of "
        r"tiles recovers \HybridRecoverFifthLow–\HybridRecoverFifthHigh% of "
        r"that gap for \HybridBitsFifth bits, and at \HybridBeatsAN of "
        r"\HybridBeatsAOf rates half the map <i>beats</i> the whole of it, "
        r"because the hybrid has two multipliers where the signalled "
        r"configuration has one. A learned predictor should also be made to "
        r"beat a free one. Routing on the bits the entropy model has already "
        r"spent per tile is parameter-free, needs no training and adds nothing "
        r"to the stream, and it beats our \RouterParams-parameter head at "
        r"every one of the \RateRankNWins rates we measure, by up to "
        r"\RateRankBeatsBy points."
        r"<br/><br/>"
        r"We find that tile count governs the method much more closely than "
        r"content does. Three 1080p classes of very different material agree "
        r"within three points, while a 416×240 class saves a third as much. An "
        r"exit map transfers across frames almost perfectly, but not across "
        r"rates."), ABST))
    A(Spacer(1, 6))

    # ---- 1 introduction --------------------------------------------------
    h1("1. Introduction")
    par(r"Learned codecs are compared on rate and distortion, with complexity "
        r"given as a single number, so many GMAC per frame or so many "
        r"milliseconds. That number is a constant. A learned decoder runs the "
        r"same graph on a page of text as on a cloudless sky, although the sky "
        r"is much the easier picture.")
    par(r"Classification networks abandoned this a decade ago. Early-exit "
        r"architectures attach classifiers at intermediate depths and stop as "
        r"soon as the prediction is confident [3, 15, 20], and the literature "
        r"around them is by now mature. Super-resolution adopted the same idea "
        r"spatially. ClassSR [12] routes image patches to networks of "
        r"different capacity by difficulty, and APE [1] exits patches at "
        r"different depths of one network. Learned compression has taken a "
        r"different route. Slimmable autoencoders [22, 23] give one model "
        r"several complexity levels, but the level is chosen <i>per stream</i> "
        r"rather than per region, and switching it changes the bitstream.")
    par(r"We ask the spatial-adaptivity question inside a learned decoder, "
        r"under a constraint that makes the answer deployable: <b>the encoder "
        r"is frozen and the coded payload is unchanged.</b> Whatever we do has "
        r"to consume the bitstream the released encoder already produces. That "
        r"rules out re-training the analysis transform, changing the entropy "
        r"model, or altering the latent. It leaves one place to spend "
        r"adaptivity, the synthesis transform.")
    par(r"Our method, FLEX-UF, is an exit ladder over the twelve residual "
        r"blocks of the DCVC-UF intra decoder. The first j blocks run over the "
        r"whole frame. The rest run per tile, over a set of tiles that shrinks "
        r"with depth, and the shrinkage is where the saving comes from. Each "
        r"exit hands its feature to the shared head through a small pointwise "
        r"adapter, zero-initialised so that at step zero the deepest exit "
        r"reproduces the released decoder bit-exactly.")
    par(r"Getting this to work depended less on the ladder than on two "
        r"problems that have little to do with early exit as it is usually "
        r"studied.")
    h2("Tiling has a large price.")
    par(r"Spatial adaptivity needs tiles, and once a frame has been cut into "
        r"tiles, every 3×3 convolution at a tile border reads invented values. "
        r"Under the stock padding rule the resulting error on this decoder is "
        r"\SeamZerosHigh dB at high rate. That is several times the entire "
        r"budget the method works to, and it is paid before any tile has saved "
        r"a single operation. In Section 4 we treat padding as an "
        r"<i>estimator</i> of the unseen neighbour and measure four of them. "
        r"The ordering we get is not the obvious one.")
    h2("A quality budget has a working range.")
    par(r"Tiling costs something even when nothing exits early, so there is a "
        r"<i>floor</i>, and budgets below it admit no allocation at all. The "
        r"ladder also has a shallowest rung, so there is a <i>saturation</i> "
        r"point above which every tile already sits on that rung and more "
        r"quality buys nothing. Between the two the budget genuinely trades "
        r"quality for compute. We measure both ends at every rate in Section "
        r"5.4, where the working budget uses \BandUseLow% of that window at "
        r"low rate and \BandUseHigh% at high rate.")
    h2("Contributions.")
    par(r"<b>(i)</b> A tile-adaptive early-exit ladder for a learned image "
        r"decoder. It leaves the encoder and the coded payload untouched, and "
        r"it saves \MainLowRate% to \MainHighRate% of decoder MACs for 0.1 dB "
        r"on the full CTC set, at a BD-Rate cost of \BdRateALow%. "
        r"<b>(ii)</b> The floor/saturation characterisation of a quality "
        r"budget, with both ends measured and both in closed form, seven "
        r"structural propositions verified numerically, and a measurement of "
        r"what the Lagrangian's convex-hull restriction costs (at most 0.05 "
        r"saving points). The architectural ceiling turns out to be reachable "
        r"at low rate, so beyond \SatLow dB at q0 the limit is the ladder "
        r"rather than the budget. "
        r"<b>(iii)</b> A quantitative account of the tiling penalty. We "
        r"measure what it costs and what removes it for free, we establish the "
        r"b² law it follows in per-tile depth and the failure of the area "
        r"law that the contaminated fraction would predict, and we measure the "
        r"ordering of four border estimators. A higher-order estimator turns "
        r"out to be worse than a zeroth-order one, and a learned repair module "
        r"is not worth its compute even with a perfect gate. "
        r"<b>(iv)</b> Two allocation regimes, an encoder-side search that "
        r"signals a ~\MapBits-bit map per frame and a decoder-side predictor "
        r"that signals nothing, with the bits and the compute charged on both "
        r"sides. We also measure the continuum between them. Overriding the "
        r"worst fifth of tiles recovers "
        r"\HybridRecoverFifthLow–\HybridRecoverFifthHigh% of the gap, and at "
        r"\HybridBeatsAN of \HybridBeatsAOf rates half the map beats all of "
        r"it, because the hybrid has two multipliers where the signalled "
        r"configuration has one. "
        r"<b>(v)</b> A parameter-free control that a learned router has to "
        r"beat, and that such routers are usually not measured against. "
        r"Routing on the bits the entropy model already spent per tile costs "
        r"nothing and adds nothing to the stream. It beats our trained head at "
        r"all \RateRankNWins measured rates, by up to \RateRankBeatsBy points, "
        r"while agreeing with the oracle on fewer tiles "
        r"than the head does. "
        r"<b>(vi)</b> A wall-clock result. Operations over-predict the saving "
        r"by about a fifth, and a bit-identical reordering of the per-tile "
        r"loop recovers a fifth of that. We realise \WallSorted% against a "
        r"\WallPredicted% prediction. "
        r"<b>(vii)</b> Two properties of the allocation that bear on "
        r"deployment: it is governed by tile count rather than by content, and "
        r"it transfers across frames but not across rates.")

    # ---- 2 related -------------------------------------------------------
    h1("2. Related work")
    h2("Early exit.")
    par(r"BranchyNet [20] and MSDNet [3] established the pattern of "
        r"intermediate classifiers with a confidence rule; SDN [15] framed it "
        r"as mitigating overthinking. We train all exits jointly, after "
        r"Scardapane et al. [18], and distil between exits as in Phuong and "
        r"Lampert [17]. The caveat in [19] is that too large a student-teacher "
        r"gap hurts the shallowest exits, so our distillation runs between "
        r"<i>adjacent</i> exits. All of this work exits on a confidence signal "
        r"computed from the network's own output. A decoder has no such "
        r"signal. There is no class posterior, and the quantity that would "
        r"decide is the error against a source the decoder cannot see "
        r"(Section 3.4).")
    h2("Spatially adaptive inference.")
    par(r"ClassSR [12] sorts super-resolution patches into easy/medium/hard "
        r"and runs a different network on each; APE [1] exits patches at "
        r"different depths; Glance-and-Focus [8] spends resolution adaptively. "
        r"We share the per-region premise with all three, and we inherit the "
        r"same tiling problem, though the cost of tile borders is rarely "
        r"quantified in that line of work. To our knowledge the estimator view "
        r"of border padding and its measured ordering (Section 4) is new. The "
        r"closest prior work is the per-channel AR(1) padding of Kaseva et al. "
        r"[11], which we implement and evaluate.")
    tbl("positioning",
        r"<b>Table 1. Where this sits.</b> Complexity control in learned "
        r"compression varies the model; we vary how much of a fixed model runs "
        r"where. The last column is what makes the difference operational. "
        r"Every other row requires a decoder that matches the encoder that "
        r"produced the stream.")
    h2("Complexity control in learned compression.")
    par(r"SlimCAE [22] and slimmable video codecs [23] expose several widths "
        r"of one model. The choice is per stream and it changes the encoder, "
        r"so the bitstream is not interchangeable. DCVC-FM [13] and DCVC-UF "
        r"[14] reduce cost architecturally, for every frame equally. "
        r"Rate–distortion–complexity has since become an explicit third axis. "
        r"Gao et al. [35] tune spatial context usage to trade decode cost "
        r"against rate, Ho et al. [36] survey where conditional residual "
        r"coding sits on that surface, and Zhang and Gao [37] route <i>whole "
        r"frames</i> to one of several jointly-trained coding paths, spending "
        r"less computation on cheaper frames. Every one of these changes the "
        r"model, and with it what the encoder emits. Our axis is orthogonal. "
        r"The model and the bitstream are both fixed, and only <i>how much of "
        r"the decoder runs where</i> varies.")
    h2("The closest neighbour.")
    par(r"Blard et al. [27] also partition an image into regions, also choose "
        r"per region by a rate–distortion cost computed at the encoder, and "
        r"also transmit a mode map, which is the skeleton of our configuration "
        r"A. The differences are in what the regions choose among and in what "
        r"the map buys. Their regions choose among several <i>separately "
        r"trained</i> codecs, so the decoder must hold all of them and its "
        r"complexity is that of one codec regardless of the choice; the map "
        r"buys rate, not compute. Ours choose a prefix length of a single "
        r"trunk, so the weights are shared by construction and the map buys "
        r"compute at fixed rate. Because their alternatives are unrelated "
        r"networks, no decoder-side predictor could stand in for the encoder's "
        r"search. The nesting that makes our exits prefixes of one another is "
        r"exactly what makes configuration B possible at all.")
    h2("Signalling versus prediction.")
    par(r"Being <i>told</i> a mode decision rather than inferring it is the "
        r"norm in standardised video coding. HEVC [9] and VVC [21] transmit "
        r"partitioning, prediction mode and transform tree. We evaluate both "
        r"and treat the signalled variant as the conventional design.")
    h2("Deferring to an oracle under a budget.")
    par(r"Configuration C, where a predictor decides most cases and a small "
        r"budget of the hardest ones is handed to something exact, is the "
        r"shape of selective prediction [38] and learning to defer [39, 40], "
        r"and of the budgeted variant of the latter [41]. Two things differ, "
        r"and both make our case easier. The expert here is the encoder's own "
        r"search, so it is exact and always available at no test-time cost. "
        r"What is scarce is the <i>bits</i> needed to say what it decided. The "
        r"selection rule is not learned either. Because the objective is "
        r"separable over tiles, the optimal set of size s at a fixed "
        r"multiplier is exactly the s largest regrets, which the encoder can "
        r"compute. Who to defer and how to learn it is the open question in "
        r"that literature, and it has a closed form here. What remains is the "
        r"question we measure, which is how concentrated the regret is.")
    h2("Allocating a budget over units.")
    par(r"The construction we use is not new and we do not present it as such. "
        r"Shoham and Gersho [28] showed that for a finite set of per-unit "
        r"operating points, a Lagrangian sweep decouples the allocation across "
        r"units and traces exactly the lower convex hull of the achievable "
        r"set; Ortega and Ramchandran [29] made it standard practice in image "
        r"and video coding. What we add is the structure this particular "
        r"operating set has: a floor below which no allocation is feasible, a "
        r"saturation point above which none improves, and a measurement of "
        r"what the convex-hull restriction costs (Section 5.4). The same "
        r"relaxation has resurfaced for test-time compute in language models "
        r"[30], with per-instance decoupling and a binary search on the "
        r"multiplier. That is the identical structure in a domain with no rate "
        r"axis, and we read it as evidence that the floor/saturation "
        r"characterisation is worth stating generally.")
    h2("How much is there to gain?")
    par(r"Bounding what adaptive inference could achieve is itself a line of "
        r"work. Hasan et al. [34] derive an oracle bound on efficiency at "
        r"fixed accuracy, given per-model resource and accuracy, and report "
        r"43–121× on ImageNet and 7–81× on HellaSwag. Their bound has a "
        r"ceiling and no floor, because the largest model in their family "
        r"reaches the reference accuracy by definition. Spatial adaptivity "
        r"introduces a floor. Cutting a frame into tiles costs quality even "
        r"when every tile runs to full depth, so the reference is unreachable "
        r"at <i>any</i> compute and the feasible set of budgets is an interval "
        r"rather than a ray. Section 5.4 characterises that interval and both "
        r"of its ends.")
    h2("Tile boundaries.")
    par(r"Every method that processes an image in independently-computed tiles "
        r"meets the same artefact. The remedies in the literature are overlap "
        r"and averaging, local padding from neighbouring patches [31], "
        r"training with overlaps [32], and fitted extrapolation [11]. Local "
        r"padding is the closest to the exact remedy we describe in Section "
        r"4.4, and the difference is accounting. It pads every convolutional "
        r"layer and does not report the cost. We pad only the 0.29% of each "
        r"block that has any spatial extent, which is what makes the exact fix "
        r"cost 0.032% of the decode rather than a multiplier on all of it. To "
        r"our knowledge the estimator view of the padding rule, the measured "
        r"ordering of four estimators, and the dependence of the penalty on "
        r"per-tile depth (Section 4) have not been reported.")
    h2("Where the time goes.")
    par(r"DCVC-RT [33] argues that operational rather than computational "
        r"complexity is the speed bottleneck for neural codecs, and cites "
        r"channel reductions that yield linear rather than quadratic speedups. "
        r"Section 5.9 is an instance of that claim inside one loop. Removing "
        r"\WallPredicted% of the operations buys \WallSorted% of the time, and "
        r"\WallSortedGain points of the difference come back from reordering "
        r"the loop with the arithmetic untouched.")

    # ---- 3 method --------------------------------------------------------
    h1("3. Method")
    h2("3.1 Where the computation is")
    par(r"The DCVC-UF intra decoder is one upsampling block, twelve "
        r"DepthConvBlocks and a head, costing 453.5 GMAC per 1080p frame. The "
        r"twelve blocks are 89.4% of that, which is why we build the ladder "
        r"across them. Inside one block at C=384 channels, the only operator "
        r"with any spatial extent is a 3×3 depthwise convolution, and it costs "
        r"9C against the block's 8C²+9C. That is <b>0.29%.</b> We come back to "
        r"the fraction several times below. Tile borders damage the picture "
        r"through it, and it is what makes the exact remedy of Section 4.4 "
        r"affordable at all. It is also why we kept the adapters pointwise.")
    h2("3.2 The exit ladder")
    par(r"With K exits over the twelve blocks and split depth j, groups "
        r"0..j−1 run full frame for every tile and groups j..K−1 run per tile. "
        r"A tile assigned exit k runs groups j..k and then leaves. Writing c_k "
        r"for the cost of exit k in units of one released decode, the cost of "
        r"a frame with exit map k is the mean of c over its tiles.")
    par(r"Two consequences follow, and both are easy to state wrongly. Exits "
        r"shallower than j are indistinguishable from each other, because the "
        r"first j groups run for every tile regardless, and the usable ladder "
        r"therefore has K−j distinct costs. The second consequence is the "
        r"<i>architectural ceiling</i> S_max = 100(1−c_j)%, reached when every "
        r"tile takes exit j. For our shipped setting (K=6, j=2, 256 px tiles) "
        r"S_max = \Ceiling%. We show in Section 5.4 that this bound is "
        r"<i>reached</i> in practice at low rate, so it behaves as an "
        r"operating point and not as an asymptote.")
    figure("adapters.png",
           r"<b>Figure 2. Inside an exit adapter.</b> <b>a</b>, one "
           r"DepthConvBlock and the two adapters drawn to scale and aligned at "
           r"their right-hand ends. Bar length is a convolution's share of the "
           r"block and bar height is the channel width it writes, so each "
           r"adapter is visibly the tail of the block it stands in for: the "
           r"FFN pair at 0.71 of a block and the single pointwise at 0.14. The "
           r"hairline rule inside the block is its only 3×3, and neither "
           r"adapter contains one, so no adapter adds receptive field and none "
           r"contributes any tile-boundary penalty. <b>b</b>, the blocks each "
           r"exit skips, coloured by the adapter it wears. Capacity is matched "
           r"to the gap, the rule switching to the FFN at four skipped blocks "
           r"(dotted), and the adapter's own cost is charged against the "
           r"saving, so an exit-2 tile saves six blocks less 0.71. The faded "
           r"exits lie below the split depth, where no tile can leave. Every "
           r"length is counted with hooks off the modules themselves by "
           r"scripts/adapter_cost.py.")
    h2("3.3 Exit adapters")
    par(r"An early exit hands the shared head a feature the head was not "
        r"fitted to, and the adapter is the correction. For exits that skip "
        r"little we use a residual 1×1, Ad(f) = f + Wf with W zero-initialised. "
        r"For exits that skip four blocks or more we use the pointwise "
        r"expand/activate/contract pair of the block's own FFN. Both are "
        r"<i>pointwise by design</i>. A 3×3 inside an adapter would add seam "
        r"damage at the tiles that took an early exit, and those are the tiles "
        r"least able to afford it. Zero-initialising the last layer makes "
        r"every adapter exactly the identity at step zero, so the deepest exit "
        r"is bit-exactly the released decoder before training begins and every "
        r"shallow exit starts from ``the decoder as it is''.")
    figure("adapter_gain.png",
           r"<b>Figure 3. What the adapters are worth.</b> <b>a</b>, each exit "
           r"with and without its adapter. <b>b</b>, the dB the adapter "
           r"recovers. <b>c</b>, the same against the number of blocks the "
           r"exit skips, which is the design rationale measured.")
    tbl("adapters_ablation",
        r"<b>Table 2. What the adapters are worth.</b> dB below the release with "
        r"every tile at that exit, with the trained adapters and with each set "
        r"back to the identity it was initialised to. † the deepest exit has no "
        r"adapter by construction and is the control.")
    par(r"Zeroing the adapters measures what they learned, since the identity "
        r"is exactly what they were initialised to. Without them the "
        r"shallowest exit costs \AdapterNoneHigh dB at q63, forty-four times "
        r"the budget, and the adapter buys \AdapterGainHigh dB of that back. "
        r"We find the gain grows with rate and with the number of blocks the "
        r"exit skips, so the design rationale here is a measurement rather "
        r"than an argument, and the deepest exit moves by exactly zero. We "
        r"would not describe the adapters as a refinement of the ladder, "
        r"because without them it has no usable rung.")
    h2("3.4 Allocation")
    par(r"Given per-tile distortions D(t,k) and costs c_k, the allocation that "
        r"minimises distortion at a compute budget is the Lagrangian")
    eq(r"k^{*}(t) = \mathrm{arg\,min}_{k}\ \left[\, D(t,k) + \lambda\, c_{k} \,\right]")
    par(r"with λ found by bisection so the frame lands on the quality budget. "
        r"Both quantities are available <i>to the encoder</i>, which holds the "
        r"source. That gives configuration <b>A</b>. The encoder searches and "
        r"transmits the map, which entropy-codes to ~\MapBits bits per 1080p "
        r"frame, or \MapOverheadLow% of a typical bitrate. It costs the "
        r"encoder about one extra decode and the decoder nothing.")
    par(r"The decoder cannot evaluate it, because D(t,k) is the error against "
        r"a source it never sees. What is missing is a variable, so no "
        r"architecture recovers it and no amount of estimation capacity "
        r"substitutes for it. Configuration <b>B</b> therefore <i>predicts</i>. "
        r"A 144K-parameter head reads the full-frame stem feature, the decoded "
        r"latent y-hat, the entropy model's scales and the quality index, "
        r"emits logits z_t, and decides")
    eq(r"\hat{k}_{t} = \mathrm{arg\,max}_{k}\ \left[\, \log\,\mathrm{softmax}(z_{t})_{k} - \beta\, c_{k} \,\right]")
    par(r"with β bisected exactly as λ is. Nothing is added to the file. We "
        r"train the head against a <i>frozen</i> decoder with a cost-sensitive "
        r"cross-entropy to the oracle's choice, each tile weighted by the "
        r"regret of choosing wrongly. Load balancing is loss-free [16], and we "
        r"bias it toward the oracle's own exit distribution instead of toward "
        r"uniform. At a high λ the oracle genuinely does send every tile to "
        r"one exit, and forcing spread there would force mistakes.")
    h2("3.5 Training")
    par(r"All exits are decoded every step and the objective is")
    eq(r"\mathcal{L} = \mathcal{L}_{\mathrm{RD}} + w_{a}\,"
       r"\mathcal{L}_{\mathrm{anchor}} + w_{d}\,\mathcal{L}_{\mathrm{distill}}")
    par(r"where L_RD is the released rate-distortion loss averaged over exits. "
        r"The anchor term L_anchor pins the deepest exit to the released "
        r"decoder, so the reference every saving is quoted against cannot "
        r"drift away underneath the measurement. The distillation term "
        r"L_distill supervises the adapters in feature space. We work in "
        r"feature space and not in pixels because the head is a fixed map from "
        r"feature to RGB, so matching the deeper feature is the stronger "
        r"constraint, with 384 dense channels of target instead of 3. Each "
        r"exit imitates its neighbour, exit k following exit k+1, and not the "
        r"deepest exit, for the reason given in [19].")
    par(r"We train the adapters <i>through the tiled decode path they are "
        r"deployed in</i>. Training them full frame and tiling only at "
        r"inference loses 0.14–0.24 dB; training through the deployed path "
        r"gains 0.51–0.90 dB, so the sign of the effect flips.")

    # ---- 4 the seam ------------------------------------------------------
    h1("4. The price of tiles")
    figure("seam_problem.png",
           r"<b>Figure 3. The artefact, before anything is done about it.</b> "
           r"One frame decoded twice from the <i>same</i> bitstream with the "
           r"same weights, every tile at full depth, stock zero padding, and "
           r"no early exit anywhere. The only difference is that the "
           r"right-hand decode was tiled. The error map is the tile lattice "
           r"and nothing else.")
    par(r"A 3×3 depthwise at feature position (i,j) computes a weighted sum "
        r"over its neighbours. Decoded full frame, those neighbours exist. "
        r"Decoded per tile they do not, and the kernel is handed whatever the "
        r"padding rule invents. Each convolution extends the contaminated "
        r"region by one ring, so with b per-tile blocks on a tile of side F "
        r"the fraction of the tile within reach of an invented value is")
    eq(r"1 - \left(\frac{F-2b}{F}\right)^{2}", tag=False)
    par(r"At our shipped F=32, b=8 that comes to 0.750. Three quarters of the "
        r"tile is affected, so what we are looking at is a structured error "
        r"over most of the tile and not a thin border at its edge.")
    par(r"That fraction tells us which pixels are affected and not how badly, "
        r"and it turns out to be the wrong predictor of the penalty. We swept "
        r"the split depth, which sweeps b from 12 to 0 with b=0 as an exact "
        r"zero-seam control, and measured")
    eq(r"\mathrm{seam} \;\propto\; b^{\alpha}, \qquad \alpha \approx 2")
    par(r"with α = 2.38 at q0, 2.22 at q32 and 1.93 at q63, and r = 0.98–0.995 "
        r"in log-log. Fitted with one free scale, the area fraction is wrong "
        r"by 160–264% where the power law is wrong by 12–22%. Fixing the "
        r"exponent at exactly two costs a factor of two, 31–38%, so the square "
        r"is the right shape without being quite the right law. The exponent "
        r"has a reading. The count of contaminated pixels grows like perimeter "
        r"times depth, and the error accumulated in each of them grows with "
        r"how many convolutions reached it. Once every pixel has been touched "
        r"the area law saturates and the penalty does not, which is where the "
        r"area law fails as a predictor.")
    figure("contamination.png",
           r"<b>Figure 4. The seam against per-tile depth.</b> Sweeping the "
           r"split depth sweeps b; b=0 is an exact control and measures 0.0000 "
           r"dB at all three rates. <b>a</b>, the measurement with the fitted "
           r"power law. <b>b</b>, both models against q63 with one free scale "
           r"each. <b>c</b>, mean relative error. The area fraction saturates "
           r"once every pixel is contaminated; the penalty does not.")
    h2("4.1 Padding is an estimator")
    par(r"Border padding is an <i>estimator</i> of the unseen neighbour, and "
        r"the seam is its error. The table below measures four of them, with "
        r"early exit switched off so that tiling is the only difference from a "
        r"full-frame decode.")
    tbl("padding",
        r"<b>Table 1. Border estimators</b>, dB below the released decoder on "
        r"the same latent, 256 px tiles, full CTC. Lower is better. The "
        r"zeroth-order hold beats the first-order extrapolation by a wide "
        r"margin, and the fitted AR(1) wins on quality while costing +10.7% of "
        r"decode wall-clock.")
    par(r"<b>A higher-order estimator is worse.</b> Extrapolating the local "
        r"gradient past a boundary amplifies whatever noise sits on that "
        r"boundary, and assuming local constancy does not. The gap is wide, "
        r"\SeamLinearHigh dB against \SeamReplHigh dB at q63. <b>The best "
        r"estimator loses on cost.</b> The per-channel AR(1) fit of [11] "
        r"reaches \SeamArlsHigh dB, about 0.02 dB better than replication, and "
        r"it costs 10.7% of decode wall-clock. Against a 0.1 dB budget and a "
        r"~24% saving that trade does not close, so we drop it.")
    h2("4.2 What actually removes the seam")
    figure("seam_vs_qp.png",
           r"<b>Figure 4. The tiling penalty across the whole rate range</b>, "
           r"every tile at full depth. The two steps that matter cost nothing, "
           r"and the largest single factor is training the ladder with the seam "
           r"present. Panel (b): the floor is charged <i>inside</i> the quality "
           r"budget and consumes a growing share of it.")
    par(r"Tile size is free in arithmetic. A tiled decode's MAC count does not "
        r"depend on it at all, and the damage scales with the tile perimeter, "
        r"as the table below shows. What tile size costs is routing "
        r"granularity, 40 tiles per 1080p frame at 256 px against 160 at "
        r"128 px.")
    tbl("tilesize",
        r"<b>Table 2. Tile size</b>, dB at q63. Doubling the side roughly "
        r"halves the penalty, as the contamination law predicts, and costs no "
        r"computation.")
    par(r"The largest factor of all is the easiest one to miss. On the "
        r"untrained ladder, replicate padding leaves \SeamReplHigh dB at q63; "
        r"after training, the same configuration leaves \FloorHigh dB. So more "
        r"than two thirds of what the estimator could not fix is absorbed by "
        r"weights learning to live with it. That single change removes more of "
        r"the seam than any module in this paper does.")
    h2("4.3 A learned repair module, and why we reject it")
    figure("seam_module.png",
           r"<b>Figure 5. Grid seam repair.</b> A correction gated by position "
           r"within a tile, applied once to the stitched frame, for 0.95% of "
           r"the decode. <b>a</b>, the gate against distance from the tile "
           r"boundary, beside the initialisation it started from; shading is "
           r"the spread over cells at the same distance. The trained gate keeps "
           r"the ring, up to 0.76 at a tile corner, and then flattens onto a "
           r"floor of 0.26 instead of switching off. <b>b</b>, the change in "
           r"error measured with the repair on, by distance from the nearest "
           r"tile boundary; bar width is the share of pixels, so bar area is "
           r"what a band contributes to the frame. It wins 0.27% on the 6.2% "
           r"of pixels within 4 px of a boundary and loses 0.04 to 0.05% on "
           r"the 94% beyond it.")
    par(r"The tile lattice is known exactly at training and at inference, so a "
        r"repair can be <i>told</i> where to look instead of having to infer "
        r"it")
    eq(r"\mathrm{Rep}(f) = f + G[\,i\ \mathrm{mod}\ P,\ j\ \mathrm{mod}\ P\,]"
       r"\cdot \mathrm{PW}\left(\mathrm{WSiLU}(\mathrm{DW}_{3\times3}(f))\right)")
    par(r"with G a P×P gate shared over channels and initialised at exp(−d/τ). "
        r"It costs 0.95% of the decode.")
    par(r"It does not earn that. Splitting the per-pixel error by distance "
        r"from the nearest tile boundary, we find the module gains 0.27% in "
        r"the 0–4 px band, which is 6.2% of pixels, and loses 0.04–0.05% "
        r"everywhere else. Give it a <i>perfect</i> gate, with zero correction "
        r"in the interior and the boundary gain unchanged, and the ceiling on "
        r"what it could earn is ≈0.0008 dB, for 0.95% of the decode. We "
        r"rejected AR(1) padding at 0.0019 dB per point of decode, and this is "
        r"worse by an order of magnitude. Tightening the gate cannot rescue "
        r"it, because there is almost nothing left to win.")
    h2("4.4 Removing the cause, and why it does not help")
    par(r"Only 0.29% of each block has spatial extent, so the exact fix is "
        r"affordable. Give the 3×3 its real neighbour from a shared canvas and "
        r"the cost is +0.032% of the decode, thirty times less than the repair "
        r"module. The fix is also exact. At uniform depth a coupled tiled "
        r"decode is bit-identical to a full-frame one, once the comparison is "
        r"not confounded by the repair module, which runs on the stitched "
        r"canvas and has no full-frame counterpart. Measured: 1.13e-2 with the "
        r"repair on, <b>exactly 0</b> with it off, against 6.06e-2 for "
        r"replicate padding.")
    par(r"Coupling also removes most of the floor. Switched on at inference, "
        r"it drops the floor from 0.036 to 0.003 dB at q0 and from 0.056 to "
        r"0.030 at q63, which is \CoupFloorDropLow% and \CoupFloorDropHigh% of "
        r"the tiling penalty.")
    par(r"<b>And it destroys the allocation.</b> At the same 0.1 dB budget the "
        r"saving falls from \CoupPaddedMid% to \CoupCoupledMid% at q32 and from "
        r"\CoupPaddedHigh% to \CoupCoupledHigh% at q63.")
    par(r"The reason is structural, and it is written into the condition on "
        r"the exactness claim. Coupling is exact when every tile is at the "
        r"<i>same</i> depth, and routing deliberately violates that condition. "
        r"With replicate padding a tile is entirely independent of its "
        r"neighbours, so the allocation is free to give adjacent tiles any "
        r"depths it likes. Coupling makes a tile depend on its neighbours' "
        r"features, and under routing those neighbours ran a different number "
        r"of blocks. A shallow tile reading a deep neighbour's activation is a "
        r"configuration the weights have never seen, and that mismatch costs "
        r"more than the seam it removed.")
    par(r"The natural reading of the exactness result is that the exact fix "
        r"should simply be adopted, and that reading is wrong. We report the "
        r"negative result for that reason, and because the tension between "
        r"coupling and routing belongs to the combination itself, so any "
        r"spatially-adaptive decoder will inherit it. One caveat keeps the "
        r"finding from being final, the same one that applies to the repair "
        r"module. This model was trained with padding, and a run trained "
        r"<i>with</i> coupling could reverse the result. The burden of proof "
        r"lies with that run.")

    # ---- 5 experiments ---------------------------------------------------
    h1("5. Experiments")
    h2("Setup.")
    par(r"We fine-tune the decoder on 512×512 crops from OpenImages with the "
        r"encoder frozen (max|Δ| = 0 is asserted every run). One λ is drawn "
        r"per sample from a log-spaced range covering all 64 quality indices, "
        r"so a single set of weights covers the whole rate range. We evaluate "
        r"on the \NumSeq sequences of the common test set (UVG, MCL-JCV and "
        r"HEVC classes B, C, D and E), one intra frame each.")
    h2("Measurement protocol.")
    par(r"Two details of the protocol move the numbers enough that we state "
        r"them here. The per-tile distortion table has to be built on the "
        r"<i>deployed</i> decode path, that is, one tiled decode per exit, "
        r"rather than by tapping the exits of a single full-frame forward "
        r"pass. Tapping is the natural implementation and it is correct for "
        r"training. But with a full-frame reference on the other side of the "
        r"ratio, the tiling penalty cancels, and the quality reported is that "
        r"of a decoder nobody ships. On our model the gap is +0.035 dB at the "
        r"deepest exit and +0.008 dB at the shallowest. It is not a constant "
        r"offset; it grows with how many blocks ran per tile, so we cannot "
        r"correct for it after the fact. The second detail is that once the "
        r"allocation is chosen, we decode that <i>mixed</i> map once and "
        r"report the distortion it actually produces.")
    h2("Reporting conventions.")
    par(r"Every dB is measured against the <i>released</i> decoder's "
        r"full-frame decode of the <i>same</i> latent, and our side is the "
        r"deployed tiled decode, so the tiling penalty sits inside every "
        r"number. Savings are fractions of the released decoder's cost. Our "
        r"own deepest exit costs 1.0095 of it, and using that as the "
        r"denominator would flatter every result by 0.6–0.8 points.")
    h2("5.1 Main result")
    figure("qualitative.png",
           r"<b>Figure 6. What the saving looks like.</b> The same bitstream "
           r"decoded by the released decoder and by ours at the 0.1 dB operating "
           r"point, 30.5% fewer multiply-accumulates. The crop is the tile that "
           r"gave up the most quality, chosen automatically, so it shows the "
           r"method's worst case on this frame rather than a flattering one.")
    tbl("main_results",
        r"<b>Table 3. Decoder MACs saved</b> (%) against the released decoder, "
        r"per quality index and quality budget, configuration A. The 0.3 and "
        r"0.5 dB rows sit on the architectural ceiling almost everywhere; past "
        r"that point a looser budget buys nothing.")
    par(r"The table carries the headline numbers. At 0.1 dB the method saves "
        r"\MainLowRate% at the lowest rate and \MainHighRate% at the highest, "
        r"and \MainMean% on average, at a BD-Rate cost of \BdRateALow%; that "
        r"is, the compute is worth about the same as a \BdRateALow% increase "
        r"in bitrate. We do not read the fall with rate as an artefact of the "
        r"ladder. High-rate reconstructions carry detail the shallow exits "
        r"cannot reproduce, and the floor rises with rate as well; both "
        r"effects push the same way.")
    tbl("complexity",
        r"<b>Table 4. Decoder complexity</b> at 1080p. Our deepest exit costs "
        r"slightly more than the release because it still pays the seam-repair "
        r"module; that 1.0095, not 1.0, is what every saving in this paper is "
        r"<i>not</i> divided by. Wall-clock is the sorted loop of Section 5.9.")
    figure("rd_spread.png",
           r"<b>Figure 7. The plane a codec is read on, and the spread behind "
           r"the mean.</b> <b>a</b>, operating points against the released "
           r"curve. <b>b</b>, the same with the quality axis expanded; labels "
           r"are compute saved. <b>c</b>, per sequence at a matched point near "
           r"0.1 dB; one dot per sequence, bar is the median.")
    par(r"Panel c shows the distribution that the headline averages over, and "
        r"it is wide. At q0 the median sequence saves 38.6%, with an "
        r"interquartile range of 32.0–41.5, while the worst saves −1.0%. The "
        r"worst cases are the low-resolution sequences of Section 5.3, where "
        r"two tiles leave nothing to allocate. Reporting the mean alone would "
        r"hide both ends.")
    h2("5.2 Is per-tile adaptivity necessary?")
    figure("exit_map.png",
           r"<b>Figure 6. Where the decoder spends.</b> Bosphorus at q32, a "
           r"0.1 dB budget. (a) the assignment overlaid on the frame, (b) the "
           r"exit index per tile, (c) the quality each tile gives up, against "
           r"its <i>own</i> full-depth reference. Water and sky leave at the "
           r"shallowest rung; the boat and the shoreline run deep.")
    tbl("static",
        r"<b>Table 5. Adaptivity against two controls</b> at the 0.1 dB budget. "
        r"† marks uniform depths whose distortion exceeds the budget, so they "
        r"are not admissible allocations at all. ``Random'' draws each frame's "
        r"map from the oracle's own exit histogram and shuffles it across "
        r"tiles, which preserves the average cost and the mix of depths while "
        r"discarding the content dependence.")
    par(r"The obvious alternative to routing tiles is to run every tile at the "
        r"same shallower exit and accept the loss. The table puts that option, "
        r"together with two stronger controls, at matched compute.")
    par(r"The uniform rows answer the question directly, and the answer "
        r"sharpens with rate. At the lowest rate a static decoder can reach "
        r"exit 3 inside the budget and save 27.0%, against the oracle's "
        r"\MainLowRate%. At q48 and q63 the only uniform depth that fits is "
        r"the deepest one, which <i>costs</i> 0.95% instead of saving "
        r"anything. At those two rates, then, the comparison is between 17% "
        r"and nothing at all, not between 17% and something smaller. Averaged "
        r"over rates, the best static allocation saves \BestStaticMean% where "
        r"the adaptive one saves \MainMean%.")
    par(r"The two shuffled rows separate effects that are easy to conflate. We "
        r"keep the oracle's own exit histogram, so the mix of depths and hence "
        r"the average cost are unchanged, and assign it to tiles at random. "
        r"Quality collapses. What the allocation buys is knowing <i>which</i> "
        r"tiles can afford to run shallower; running some fraction of them "
        r"shallower is worth nothing by itself. The rate-ranked row keeps the "
        r"same histogram and orders it by a signal the decoder already holds, "
        r"namely the bits the entropy model spent on each tile. This is the "
        r"cheapest router we can think of. It has no parameters and no "
        r"training, and the number is available before the trunk starts. It is "
        r"also the natural analogue of the confidence rules used by early-exit "
        r"classifiers [3, 20], which likewise read a signal off the network's "
        r"own output.")
    par(r"It recovers most of the gap. At q63, random costs \RandomDb dB and "
        r"the oracle 0.100 dB at identical compute, while rate-ranking costs "
        r"\RateRankDb dB. That is \RateRankRecovers% of the oracle's advantage "
        r"over chance, at 0.68–0.79 agreement with the oracle's map. A learned "
        r"router therefore has to beat a free baseline that is already three "
        r"quarters of the way there. We would not have run that comparison "
        r"without this control, and we suggest that any adaptive-inference "
        r"paper report it. Given the oracle's histogram, what we measure here "
        r"is <i>ranking</i> and nothing else. Section 5.6 removes that crutch "
        r"and turns the same signal into a complete routing rule, which "
        r"matches the trained head across the whole rate range.")
    h2("5.3 Resolution, and the granularity of a tile")
    figure("perclass.png",
           r"<b>Figure 7. Saving by test class</b> at one global operating "
           r"point. λ is bisected once so the whole set lands on the budget, "
           r"exactly as it would be in deployment; each class is then reported "
           r"at that λ. The ordering follows tile count, not content.")
    tbl("perclass",
        r"<b>Table 6. Saving by test class</b> (%), at a 0.1 dB budget set "
        r"globally over all \NumSeq sequences rather than per class. "
        r"``Tiles'' is how many 256 px tiles a frame of that resolution "
        r"contains. The saving tracks tile count much more closely than it "
        r"tracks content.")
    par(r"Completing the test set exposed a dependence that the 1080p-only "
        r"subset had hidden. The table groups the saving by class. At 1080p, "
        r"where a frame is 40 tiles, we save \BigResLow% (MCL-JCV), 33.1% "
        r"(UVG) and 33.8% (HEVC B) at the lowest rate, three classes of very "
        r"different content within three points of each other. At 832×480 a "
        r"frame is 8 tiles and the saving is \MidResLow%; at 416×240 it is 2 "
        r"tiles and \SmallResLow%. At high rate the small resolutions collapse "
        r"to \SmallResHigh% against \BigResHigh% for 1080p.")
    par(r"The natural explanation is granularity. With two tiles per frame "
        r"there is almost no allocation left to make, and the Lagrangian "
        r"degenerates towards a uniform choice. That suggests a fix. Choose "
        r"the tile size relative to the frame, since the MAC count does not "
        r"depend on tile size at all and only the seam does.")
    par(r"<b>We tested that fix and it mostly does not work.</b> We compare "
        r"the 256 px configuration against a 128 px one at q0, which "
        r"multiplies the tile count by 3.4. MCL-JCV (1080p) goes from 36.3 to "
        r"34.3, HEVC B (1080p) from 33.8 to 32.1, HEVC C (832×480) from 20.9 "
        r"to 17.6, and HEVC D (416×240) from 11.3 to <b>13.7</b>.")
    par(r"Smaller tiles help only at the smallest resolution, where the count "
        r"goes from two to eight. Everywhere else they cost between 1.7 and "
        r"3.3 points, including at 832×480, where the count rises from 8 to 28 "
        r"and the saving still falls. Beyond a modest number of tiles, the "
        r"extra seam outweighs the extra granularity. We report the comparison "
        r"as indicative, since the two configurations come from different "
        r"training runs and tile size is therefore confounded with training. "
        r"But the direction is consistent, and it is enough for us to say that "
        r"a resolution-adaptive tile size is not the easy win the granularity "
        r"argument suggests.")
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
           r"dynamic programming; the two curves are indistinguishable.")
    par(r"The construction has enough structure to state as propositions, and "
        r"we check each one numerically instead of asserting it "
        r"(scripts/verify_theory.py, seven of seven). The allocation decouples "
        r"per tile; compute is non-increasing and distortion non-decreasing in "
        r"λ; the sweep traces the lower convex hull and therefore cannot reach "
        r"an interior point of the achievable set; λ=0 reaches the least "
        r"distortion the ladder can produce; beyond a finite λ, computable in "
        r"closed form, the allocation is the constant map to exit j at cost "
        r"exactly c_j; and the ceiling is a function of the split depth alone.")
    par(r"The third of those has a practical edge. The sweep reaches only hull "
        r"vertices, so a budget that falls between two of them cannot be met "
        r"exactly. We measured what that costs by enumerating the <i>exact</i> "
        r"Pareto set with dynamic programming over tiles, which is feasible "
        r"because the cost alphabet has four symbols. The sweep reaches 95 "
        r"allocations against a Pareto set of 635, and the loss from convexity "
        r"is at most 0.05 saving points. For this problem the standard "
        r"construction is essentially optimal.")
    par(r"It matters <i>why</i> that loss is small, since the reason is "
        r"structural and points at what would make it smaller. The reachable "
        r"costs form a lattice, and its spacing is set by how much the mean "
        r"cost moves when the multiplier crosses a switching point. If m tiles "
        r"each move one rung there, the spacing is at most "
        r"m·max(c_k+1 − c_k)/T saving points. The convexity loss cannot exceed "
        r"the largest spacing, because a budget between two levels is served "
        r"by the lower one. Here m=2, since tiles with identical rows switch "
        r"together, and the bound is 0.745 points; the measured spacing sits "
        r"exactly on it. Nothing in the expression depends on content or rate, "
        r"and the bound falls as 1/T: halving the tile side puts four times as "
        r"many tiles on the lattice, each carrying a quarter of the step. Fine "
        r"granularity is what makes the Lagrangian relaxation lossless in "
        r"practice, and no property of the images is doing that work.")
    par(r"Two numbers bound what any budget can do. The <b>floor</b> is the "
        r"distortion of a tiled decode with every tile at full depth, which is "
        r"pure tiling penalty: \FloorLow dB at q0, rising to \FloorHigh dB at "
        r"q63. A budget below it admits no allocation. The <b>saturation</b> "
        r"point is the distortion when every tile takes exit j; any budget at "
        r"or above it reaches the ceiling, and a larger one reaches nothing "
        r"more.")
    par(r"One consequence of that is worth stating plainly. At q0 the ceiling "
        r"is reached at \SatLow dB, so the architectural bound behaves as an "
        r"operating point rather than as an asymptote. Past it, the limiting "
        r"factor is the <i>ladder</i> and not the budget, which is an argument "
        r"for a finer ladder before a looser budget. The same two numbers pick "
        r"out a narrow regime: budgets in [\SatLow, \SatWindowHi) dB saturate "
        r"the lowest rate and nothing else, a window \SatWindowMb millibels "
        r"wide.")
    figure("budget_band.png",
           r"<b>Figure 9. The band is the rate dependence.</b> <b>a</b> Saving "
           r"against the budget, with each rate's floor and saturation point "
           r"ticked. <b>b</b> The same with the budget axis rescaled onto each "
           r"rate's own band. Five curves become one; dashed is the one-parameter "
           r"power law fitted to all of them.")
    par(r"<b>And the window is almost all of the rate dependence.</b> At a "
        r"matched decibel the five rates are \BandRawSpread points apart. "
        r"Rescale the budget axis onto each rate's own band, with the floor at "
        r"0 and saturation at 1, and they collapse onto a single curve, "
        r"\BandSpreadMean points apart on average and \BandSpreadMax at worst. "
        r"The answer to ``how much does a 0.1 dB budget buy at this rate'' is "
        r"therefore, to within a couple of points, ``where does 0.1 dB sit in "
        r"this rate's band''. The floor and the saturation point are both in "
        r"closed form and both cheap to measure. What they leave over is small "
        r"enough that a deployment could calibrate the two ends and read the "
        r"rest off one curve. That curve is a one-parameter power law, "
        r"saving ≈ C·u^\BandExp, with C the architectural ceiling and u the "
        r"position in the band. We fit it in log space over all five rates and "
        r"get R² = \BandR2, worst residual \BandFitErr points.")
    par(r"<b>It is a property of the ladder, not of this checkpoint.</b> We "
        r"repeated the measurement on a different training run, BEST, a "
        r"separate recipe taken at a different epoch, and the same thing "
        r"happens. The rates are \BandRawSpreadB points apart at a matched "
        r"decibel and \BandSpreadMeanB after rescaling, and the collapsed "
        r"curve is again a power law, R² = \BandRTwoB. The <i>exponent</i> "
        r"does not carry over: \BandExpB there against \BandExp here, so it "
        r"describes a trained decoder and not the architecture. What "
        r"replicates is the collapse. Within a checkpoint, the rate dependence "
        r"of the trade-off is the rate dependence of the band.")
    figure("tradeoff.png",
           r"<b>Figure 9. The trade-off, whole.</b> <b>a</b> What a budget buys, "
           r"per rate; the ceiling is \Ceiling% and q0 reaches it at "
           r"\SatLow dB. <b>b</b> The same relation inverted, so it reads as "
           r"what a saving target costs. The three budgets reported elsewhere "
           r"are three points on this curve.")
    h2("5.5 Signalled versus predicted allocation")
    figure("router_ab.png",
           r"<b>Figure 9. Who decides.</b> <b>(a)</b> A holds the source, so it "
           r"can decode all K exits per tile and pick the true minimiser; it "
           r"signals the map at ~\MapBits bits/frame. B never sees the source "
           r"and signals nothing; its \RouterParams-parameter head reads "
           r"y-hat, the entropy scales and qp. <b>(b)</b> Saving at 0.1 dB; "
           r"shading is what a bit-exact bitstream costs. <b>(c)</b> That cost "
           r"is smallest at q\GapMinQp, where the tilt needed to move the "
           r"router off its training multiplier is essentially zero, and grows "
           r"in both directions. At 0.3 dB it collapses to the router's own "
           r"\RouterCostPct% wherever the budget saturates the ladder, and "
           r"widens where it does not.")
    tbl("ab",
        r"<b>Table 7. Signalled against predicted</b> at two budgets, same "
        r"checkpoint and test set, with one router trained against the deployed "
        r"oracle at a single λ.")
    par(r"Configuration A is exact by construction and costs bits; "
        r"configuration B is approximate and costs none. The table above "
        r"compares them.")
    par(r"<b>Not signalling costs \GapMin–\GapMax points</b>, roughly flat "
        r"across rate, for zero added bits and a byte-identical file. The gap "
        r"is smallest at q\GapMinQp and widens toward both ends of the rate "
        r"range.")
    par(r"<b>It tracks |β|</b>, the tilt the bisection has to apply to move a "
        r"router trained at one λ onto another operating point. At q\BetaMinQp "
        r"the required tilt is β=\BetaAtMin, essentially none, since that is "
        r"where the training λ lands, and there the gap is at its minimum. At "
        r"q0 the tilt is \BetaLow and the gap is \GapMax. A large tilt in "
        r"either direction lets the cost term dominate the logits, which "
        r"discards the content ranking the router learned. The remedy is a "
        r"router per operating point, and a deployment would have one anyway: "
        r"a single set of weights covers all rates, and a \RouterParams head "
        r"per rate is 0.3% of the model each.")
    par(r"Loosening the budget closes the gap only where the budget saturates "
        r"the ladder. At 0.3 dB the gap is \GapLooseLow points at the three "
        r"lowest rates. That is exactly the router's own \RouterCostPct% of "
        r"decode, so its <i>prediction</i> is free there; both configurations "
        r"send every tile to the cheapest rung, and nothing is left to predict "
        r"wrongly. At the two highest rates 0.3 dB does not saturate, and the "
        r"gap <i>widens</i> to \GapLooseHigh points. A looser budget gives the "
        r"allocation more room to be wrong in as well as more room to be "
        r"right.")
    par(r"For a system that trains once, the single-router number is the "
        r"honest one, so that is what we report. It understates what B can do.")
    par(r"<b>Retraining with a working mask raises agreement and does not "
        r"simply raise the saving.</b> The head above was trained against a "
        r"suppression term sitting above its own logits (below). We retrained "
        r"it with the mask fixed, same recipe and same λ. Held-out agreement "
        r"rises from \RetrainAgreeOld to \RetrainAgreeNew, and the deployed "
        r"saving moves by +\RetrainGain points at q\RetrainGainQp and "
        r"−\RetrainLoss at q\RetrainLossQp. We find the retrained head ahead "
        r"where the required tilt is small and behind where it is large, which "
        r"is the same axis the gap runs along. That is the second time in this "
        r"paper that agreement with the oracle has moved opposite to the "
        r"quantity that matters. We keep the original head for every other "
        r"measurement so the comparisons stay on one system, and we read the "
        r"retrain as evidence that the mask cost the head real capacity, and "
        r"that agreement is not the objective.")
    par(r"An earlier version of this measurement put the gap at 1.7 points at "
        r"q0, rising to 13.3 at q63. The exit mask was at fault. It suppresses "
        r"exits below the split depth by assigning -1e4, the head's own logits "
        r"had drifted to that scale, and the suppressed entries were therefore "
        r"the largest in every row. A large share of every tile went to the "
        r"cheapest rung for a reason unrelated to its content. The mask is now "
        r"negative infinity. Fixing it costs 3.5 points at q0, where the "
        r"accident happened to agree with the oracle, and buys 9.4 at q63, "
        r"where it did not.")
    h2("5.6 A router with no parameters")
    par(r"Before a \RouterParams head is worth its \RouterCostPct% of the "
        r"decode, it has to beat what the decoder already knows. The entropy "
        r"model has produced one number per tile before the trunk runs, and at "
        r"no cost: how many bits that tile's latents took.")
    par(r"We turn it into a routing rule with no learned parameters. We model "
        r"the per-tile distortion as rank-1 in the log domain")
    eq(r"\log D(t,k) \;\approx\; \alpha \log b(t) + c + \log \varphi_{k}")
    par(r"where b is the tile's bit count normalised by the frame mean and φ "
        r"is a K-vector saying what each exit costs on an average tile. We fit "
        r"(α, c, φ) by least squares, leave-one-sequence-out, so that no "
        r"sequence contributes to the profile that routes it. The same "
        r"Lagrangian the oracle runs is then run on the surrogate. Nothing is "
        r"signalled and nothing is trained; the arithmetic is one scalar per "
        r"tile.")
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
    par(r"<b>It beats the trained head at every rate.</b> The free rule is "
        r"ahead of the \RouterParams router at all \RateRankNWins measured "
        r"rates, by margins that run from half a point at q48 to "
        r"\RateRankBeatsBy points at q0. A head trained on this decoder "
        r"against this oracle therefore returns nothing over a rule with no "
        r"parameters at all, and it carries \RouterCostPct% of the decode "
        r"that the free rule does not.")
    par(r"<b>Why it works is not that it agrees with the oracle.</b> It agrees "
        r"on \RateRankAgreeLo–\RateRankAgreeHi of tiles, well below the "
        r"router's 0.718, and still saves more at every rate. "
        r"Agreement weighs a disagreement on a tile where two exits are within "
        r"a hair of each other exactly as heavily as one where the choice is "
        r"most of the frame's error, and most tiles are the former. The "
        r"ordering is what the rule gets right. Bits correlate with the "
        r"<i>spread</i> across the ladder, that is, with how much a tile "
        r"stands to gain from depth, at "
        r"ρ_spread = \RateRankSpreadLo–\RateRankSpreadHi at every rate.")
    par(r"<b>At a looser budget it stops being a baseline and becomes the "
        r"answer.</b> At 0.3 dB the free rule matches the oracle exactly at "
        r"the three lowest rates, where both reach the same architectural "
        r"ceiling, comes within 0.2 points of it at q48, and gives up "
        r"\RateRankLoose% against the oracle's \SigLooseHigh% at q63. "
        r"That is ahead of the "
        r"trained head at every rate, by up to \RateRankLooseAheadBy points. "
        r"The head is charged for itself and the free rule is not. At a loose "
        r"budget the head's ordering also has less left to contribute, because "
        r"once the budget saturates the ladder there is little ordering left "
        r"to get right, and the free rule gets it. At 0.5 dB it reaches the "
        r"ceiling at \RateRankHalfDbExact rate and its assignment is "
        r"<i>identical</i> to the oracle's on every tile.")
    par(r"<b>What it cannot do</b> is see anything beyond that ordering. A "
        r"rank-1 model in the level assigns every tile the same relative "
        r"profile over exits, so b only decides where on the ladder a tile "
        r"falls, never the shape of its trade-off. That is the ceiling this "
        r"baseline sits at, and it is the part a learned head should be "
        r"earning its parameters on. Ours does not earn them at any rate we "
        r"measured.")
    tbl("raterank_best",
        r"<b>Table 9. The same comparison on a second training run</b> (BEST), "
        r"0.1 dB, same test set. Bold where the parameter-free rule beats that "
        r"run's own trained head.")
    par(r"<b>It replicates on a second training run.</b> "
        r"BEST is a separate recipe at a different epoch, with its own router "
        r"trained the same way. There the free rule beats the trained head at "
        r"\BestRankWinsN of \BestRankOfN rates, by up to \BestRankBy points, "
        r"and comes within \BestRankToOracle points of the <i>oracle</i> "
        r"everywhere. The margins there are wider than here, and the one rate "
        r"it concedes is the only rate on either checkpoint where the head "
        r"finishes ahead, by under a point. We do not read that as showing a "
        r"learned head cannot be worth its parameters, only that neither of "
        r"our training runs produced one that is. The free rule stays close "
        r"to the oracle on both.")
    par(r"Per-block bit allocation is a standard quantity in learned "
        r"compression, where it is something to <i>choose</i>; block-level "
        r"rate control sets it so that complex regions get more bits [42]. We "
        r"read the same number in the other direction, after the fact and at "
        r"the decoder, as a statement about how hard a region was. It costs "
        r"nothing because someone else has already paid for it.")
    tbl("blend",
        r"<b>Table 9. Blending the two decoder-side signals</b> at 0.1 dB, both "
        r"normalised to unit mean, weight w from the bit rule to the head's "
        r"ordering. Bold is the best per rate.")
    par(r"<b>Are the two signals complementary?</b> Barely. We normalise both "
        r"surrogates to unit mean and blend them with one weight w, where w=0 "
        r"is the bit rule and w=1 the head's ordering. The best blend is w=0 "
        r"at the three lowest rates and a small head weight at the two "
        r"highest, worth +2.1 points at q48 and +0.9 at q63. The head reads "
        r"the entropy model's scales, so it already has most of what the bit "
        r"count carries; what it adds is confined to q48 and q63.")
    par(r"A learned component should be measured against the free alternative, "
        r"and it rarely is. In adaptive inference the usual controls are a "
        r"uniform allocation and a random one. Both are much weaker than a "
        r"decoder-side signal that is already lying around.")
    h2("5.7 Signalling only what the router gets wrong")
    par(r"A and B are the two ends of one scale. The encoder can run the "
        r"decoder's router, which reads only decoded data, so the encoder "
        r"knows tile by tile where the prediction will be wrong. Nothing "
        r"obliges it to correct every one of them.")
    par(r"Configuration C signals a fraction ρ of the tiles and leaves the "
        r"rest to the router. We choose the overrides by Lagrangian regret, "
        r"which is exactly what the objective loses on that tile by staying "
        r"silent. We hold β at B's value and bisect λ over the overrides to "
        r"land back on the budget. The map costs an entropy-coded mask, "
        r"N·H(ρ) bits with H the binary entropy, plus 3 per override.")
    figure("hybrid.png",
           r"<b>Figure 10. Partial signalling.</b> <b>a</b> Saving against the "
           r"bits spent on the map; the left end of each line is B and the right "
           r"end is A. <b>b</b> The same, normalised: what fraction of the A–B "
           r"gap a given fraction of the map recovers; dashed is proportional "
           r"recovery, and above 100% a partial map is beating a complete one. "
           r"<b>c</b> The same against the fixed-λ Lorenz prediction; dashed is "
           r"equality, and points above it are allocations a single multiplier "
           r"cannot reach.")
    tbl("hybrid",
        r"<b>Table 8. Configuration C</b> at 0.1 dB. Columns are the fraction of "
        r"tiles the encoder overrides; the two end columns reproduce B and A to "
        r"within 0.1 points, which is the check that the interpolation is real.")
    par(r"<b>The two ends check out.</b> At ρ=0 the measurement reproduces "
        r"configuration B to the second decimal at every rate, and at ρ=1 it "
        r"reproduces A. Neither end is imposed; both fall out of the same code "
        r"path run against independently measured files, so the columns "
        r"between them are measuring something real.")
    par(r"<b>Most of the gap is cheap.</b> Overriding a tenth of the tiles for "
        r"\HybridBitsTenth bits per frame recovers "
        r"\HybridRecoverTenthLow–\HybridRecoverTenthHigh% of the gap, and a "
        r"fifth recovers \HybridRecoverFifthLow–\HybridRecoverFifthHigh%. "
        r"Recovery is concave everywhere, so the first bits spent are the most "
        r"useful ones.")
    par(r"<b>And a partial map can beat a complete one.</b> At \HybridBeatsAN "
        r"of \HybridBeatsAOf rates, signalling half the tiles for "
        r"\HybridBitsHalf bits saves <i>more</i> than signalling all of them, "
        r"by up to \HybridBeatsABy points, at the same distortion. Both land "
        r"within the bisection's 5e-4 dB tolerance of the budget, and the "
        r"local exchange rate is about 48 saving points per decibel, so the "
        r"tolerance is worth 0.02 points against a margin ten to twenty times "
        r"that. The margin is therefore real, and it does not come from a "
        r"better predictor. The hybrid has <i>two</i> multipliers where A has "
        r"one: the oracle's λ governs the overridden tiles and the router's "
        r"fixed β the rest. A two-multiplier allocation reaches points on the "
        r"distortion–compute plane that a single Lagrangian sweep cannot, for "
        r"the same reason the sweep misses interior Pareto points at all. "
        r"Configuration A is optimal among allocations reachable by one "
        r"multiplier, and that is a smaller set than the word suggests.")
    par(r"<b>Why the recovery is concave, and what bounds it.</b> At a fixed λ "
        r"the objective is separable, so overriding a set removes <i>exactly</i> "
        r"the sum of that set's regrets. The recovery of the <i>objective</i> "
        r"is then the Lorenz curve of the per-tile regret distribution, and "
        r"its curvature is that distribution's Gini coefficient, "
        r"\GiniMin–\GiniMax across rates. What the table reports is saving at "
        r"fixed distortion, not the objective, and returning to the budget "
        r"means re-bisecting λ. That re-bisection is also what lets the "
        r"two-multiplier allocations above escape the bound.")
    par(r"<b>At the loose budget it matters where the gap is.</b> At 0.3 dB "
        r"the three lowest rates are saturated, and there is nothing for a "
        r"partial map to buy. At \HybridLooseQps, where the gap is "
        r"\GapLooseHigh points, half the map recovers "
        r"\HybridLooseRecLo–\HybridLooseRecHi% of it. The allocation follows "
        r"the same rule here as everywhere else, which is to spend the bits "
        r"where the ladder still has somewhere to go.")
    par(r"<b>A better predictor concentrates its regret, which is what the "
        r"Gini was for.</b> We rebuild C on the retrained head of Section 5.5. "
        r"The Gini of the per-tile regret rises at \GiniRetrainUpN of "
        r"\GiniRetrainOfN rates, to \GiniRetrainLo–\GiniRetrainHi. Its "
        r"mistakes are rarer and larger. The same table shows the consequence. "
        r"C converges on A faster from a better base and has less left to "
        r"recover, and the half-beats-all effect disappears at q0, because a "
        r"base close to A leaves little for the extra multiplier to exploit. "
        r"The Lorenz reading is therefore worth having as a diagnostic and not "
        r"as a decoration; it reports what <i>kind</i> of predictor one has as "
        r"well as how good it is.")
    par(r"<b>Which predictor should C be built on?</b> Either will do, and the "
        r"answer follows the same split as Section 5.6. Running the identical "
        r"override rule over the parameter-free surrogate instead of the head "
        r"is worth up to \CPredBitsAhead points at the lowest rate and costs "
        r"up to \CPredHeadAhead at the highest. Both converge on A at ρ=1 to "
        r"the second decimal at every rate, which is the third independent "
        r"check that the two code paths agree. The best single operating point "
        r"we measured is the free predictor with half the map signalled, at "
        r"\CBestSave% at q\CBestQp. That is \CBestOverA points <i>above</i> "
        r"full signalling, with no learned component anywhere in the decoder.")
    par(r"Configuration C is what we would ship where a small map is tolerable "
        r"and a large one is not. It also reframes the A–B gap. What the gap "
        r"measures is the price of <i>silence</i>, charged tile by tile, "
        r"rather than the price of prediction.")
    h2("5.8 Does the map have to be recomputed?")
    figure("map_transfer.png",
           r"<b>Figure 10. Reusing an exit map.</b> Solid is the transferred "
           r"map, dashed the one recomputed in place. <b>a</b>, <b>b</b>: reuse "
           r"across frames of the same sequence. <b>c</b>: reuse across quality "
           r"index; the entry is the delivered distortion against a 0.1 dB "
           r"budget.")
    par(r"Configuration A costs the encoder about one extra decode per frame. "
        r"How much that matters depends on how often the map has to be "
        r"recomputed, and we have not seen that measured anywhere.")
    par(r"<b>Across frames.</b> Reuse is close to free. We take the map found "
        r"on frame 0 and apply it eight frames later; at q0 the delivered "
        r"distortion rises by \TransferDbCost dB, five percent of the budget, "
        r"and the saving does not move at all: \TransferSaving% transferred "
        r"against \TransferInPlaceLo–\TransferInPlaceHi% recomputed. The "
        r"allocation is a property of where the content is hard, and that "
        r"moves slowly. A codec would search once per group of pictures "
        r"instead of once per frame, and divide the encoder cost by the group "
        r"length.")
    par(r"<b>Across rate.</b> Here the map does have to be recomputed, and the "
        r"direction of the reuse decides how badly. Found at q0 and applied at "
        r"q63, it delivers \TransferCrossDb dB against a 0.1 dB budget, "
        r"claiming the low-rate saving of \TransferSaving% while spending "
        r"nearly twice the quality it is allowed. The reverse is safe but "
        r"wasteful: the q63 map applied at q0 delivers 0.075 dB and only 19.8% "
        r"where 33.1% was available. Shallow exits are cheap in quality at low "
        r"rate and expensive at high rate, so a map is calibrated to the rate "
        r"it was found at, and reusing one upward breaks the quality guarantee "
        r"without anything in the decode reporting that it has. An encoder "
        r"should search once per rate and reuse that search across frames.")
    h2("5.9 Complexity and wall-clock")
    tbl("latency",
        r"<b>Table 8. Wall-clock</b>, 1080p, median of 40 interleaved "
        r"iterations on one A100-class GPU, at the 0.1 dB operating point. "
        r"``MACs'' is what the arithmetic predicts; ``measured'' is the sorted "
        r"per-tile loop.")
    par(r"A saving in multiply-accumulates is not a saving in time. Tiling "
        r"costs \TilingOverhead% before anything exits early, measured as the "
        r"deepest-exit tiled decode against the released full-frame one at the "
        r"same arithmetic and the same weights. On top of that, the routed "
        r"decode realises \WallMasked% at q0 where its operations predict "
        r"\WallPredicted%. Four fifths of the predicted saving arrives. The "
        r"missing fifth is the tiling overhead together with the bookkeeping "
        r"of a shrinking active set, and a MAC count sees neither. The "
        r"shortfall scales with the saving instead of sitting at a fixed "
        r"offset: at q63 we measure \WallHighMeasured% realised against "
        r"\WallHighPredicted% predicted.")
    par(r"Sorting the tiles once by exit depth recovers part of it. The "
        r"obvious implementation of a shrinking active set keeps a boolean "
        r"mask, gathers the survivors and scatters the finished at every group "
        r"boundary, and each of those boundaries forces a device-to-host "
        r"synchronisation. Order the tiles by descending exit depth and "
        r"``still active at group g'' becomes a contiguous prefix: each group "
        r"is then a slice instead of a gather, and one cumulative count "
        r"supplies every boundary, so no per-group mask is built at all. The "
        r"finished tiles come back through the inverse permutation in a single "
        r"scatter. The arithmetic is unchanged and the output is bit-identical "
        r"on GPU. At q0 the saving goes from \WallMasked% to \WallSorted%, a "
        r"gain of \WallSortedGain points, or a fifth of the shortfall, for a "
        r"change that touches no arithmetic.")
    par(r"The same gap appears in our own accounting. Every configuration B "
        r"number in this paper charges the router at its share of the "
        r"decoder's multiply-accumulates, \RouterCostPct%. We also timed the "
        r"head directly, on the padded 2048×1280 frame the decoder actually "
        r"sees, interleaved against a deepest-exit decode so that a "
        r"co-tenant's load falls on both equally. It costs \RouterTimePct%, "
        r"which is \RouterTimeFactor× what its arithmetic predicts, and for "
        r"the same reason as everything else in this section: a "
        r"\RouterParams-parameter head is launch overhead, and a MAC count "
        r"cannot see a launch. Charged at measured time instead of at "
        r"operations, every B number in this paper would fall by a further "
        r"\RouterTimeExtra points. We leave them charged at MACs because that "
        r"is the convention the rest of the literature reports in, and we "
        r"record the correction here so that it does not sit unstated.")
    par(r"A paper that quotes only operations would report \WallPredicted% "
        r"where the same code, run as written, delivers \WallSorted%. That is "
        r"why we give the shortfall as well. The error runs one way: "
        r"operations are an <i>optimistic</i> bound on this method, and the "
        r"optimism grows with how much of the frame exits early.")
    h2("5.10 The right ladder depends on the budget")
    tbl("runs",
        r"<b>Table 9. Ladder configurations</b>, mean saving (%) over the five "
        r"rates, same test set and protocol. ``Ceiling'' is the architectural "
        r"ceiling. * one rate is infeasible at that budget, because the "
        r"ladder's floor exceeds it, so the mean is over the remaining four.")
    par(r"Section 5.4 argued that once a budget saturates a ladder, the only "
        r"way to spend more is a rung that does not exist. The table measures "
        r"that. A finer ladder (K=12, j=4) has a ceiling of 50.3% against "
        r"\Ceiling%, and the two orderings cross somewhere between 0.1 and "
        r"0.3 dB. At 0.1 dB the coarse ladder wins, and the fine one cannot "
        r"even reach the budget at the highest rate. Splitting later (j=4 "
        r"against j=2) puts twice as many blocks in the per-tile section, "
        r"which by the contamination law raises the floor. At 0.5 dB the fine "
        r"ladder wins by 6.7 points, \FineHalfDb% against \CoarseHalfDb%, "
        r"because the coarse one has been pinned at its ceiling since 0.3 dB "
        r"and has nothing left to spend.")
    par(r"So the ladder is a choice made at the operating point, not a "
        r"hyperparameter tuned once and fixed. The floor-saturation window of "
        r"Section 5.4 tells you which side of the crossover a given budget "
        r"sits on.")

    # ---- 6 limitations ---------------------------------------------------
    h1("6. Limitations")
    par(r"<b>The seam is reduced, and the exact remedy is not usable as it "
        r"stands.</b> Canvas coupling removes "
        r"\CoupFloorDropLo–\CoupFloorDropHi% of the floor and is bit-exact at "
        r"uniform depth (Section 4.4). It also collapses the routed saving, "
        r"because routing puts neighbouring tiles at different depths. We do "
        r"not know whether a decoder trained with coupling can recover both at "
        r"once, and that is the experiment we would run next. Until someone "
        r"runs it, the floor is a cost this method pays.")
    par(r"<b>Intra frames only.</b> This is the image path of a video codec. "
        r"Extending the ladder to inter frames raises a question we do not "
        r"answer here, because an exit map propagates through the reference "
        r"chain and a shallow tile in one frame is a worse reference for the "
        r"next one.")
    par(r"<b>Training is not converged.</b> At every checkpoint we have "
        r"measured more than once, the saving was still going up. We therefore "
        r"read the numbers reported here as a lower bound on what a converged "
        r"run would give.")
    par(r"<b>A single decoder.</b> All results are on DCVC-UF's intra decoder. "
        r"Nothing in the method looks specific to it, since the ladder needs "
        r"only a residual trunk with a shared head, but we have not measured a "
        r"second decoder to check.")

    # ---- 7 conclusion ----------------------------------------------------
    h1("7. Conclusion")
    par(r"Learned decoders spend a constant amount of computation on a "
        r"non-constant world. An early-exit ladder over the tiles of a frame "
        r"recovers a large fraction of it. At a 0.1 dB budget we save "
        r"\MainLowRate% to \MainHighRate% of decoder MACs for a BD-Rate cost "
        r"of \BdRateALow%, with no change to the encoder and none to the coded "
        r"payload.")
    par(r"Three lessons we would carry to any spatially adaptive decoder, and "
        r"none of them is about early exit.")
    par(r"<b>Tiling is the dominant cost and it is governed by depth.</b> All "
        r"of that cost traces back to a single 3×3 that is 0.29% of the "
        r"arithmetic, and the penalty grows as the square of how many such "
        r"convolutions run per tile. The corrupted-area fraction usually "
        r"quoted alongside it predicts that penalty badly.")
    par(r"<b>The exact remedy is in tension with the thing it enables.</b> "
        r"Giving each convolution its real neighbour is bit-identical at "
        r"uniform depth, and under routing it collapses the allocation, "
        r"because routing is the deliberate violation of the condition that "
        r"makes it exact. We expect any method that combines spatial "
        r"adaptivity with tiled inference to walk into this.")
    par(r"<b>A quality budget is only a control variable inside a measurable "
        r"window.</b> Below the floor the budget admits nothing at all, and "
        r"above saturation more of it buys nothing. A saving quoted without "
        r"saying where in that window it sits has left out the part of the "
        r"result a reader most needs.")
    par(r"We would attach three cautions to the measurements themselves. A "
        r"saving in operations is an optimistic bound on a saving in time, and "
        r"the optimism scales with the saving. A learned router should be "
        r"compared against a free one. Routing on the bits already spent per "
        r"tile needs no parameters and no training, it adds nothing to the "
        r"stream, and it beats our trained head at every rate we measured, "
        r"while agreeing with the oracle on fewer tiles "
        r"than the head does. We read that as a warning about the metric quite "
        r"as much as about the head. Finally, a timing harness will report "
        r"numbers whether or not it is timing the right device. Ours timed the "
        r"wrong one for months, and what gave it away was a decoder that "
        r"appeared to slow down with the quality index.")
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

    wide_band = 4.95 * inch
    f_wide = Frame(M, PH - 0.7 * inch - wide_band, PW - 2 * M, wide_band,
                   id="wide", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
    hw = H - wide_band
    f_lw = Frame(M, 0.7 * inch, colw, hw, id="lw", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    f_rw = Frame(M + colw + GAP, 0.7 * inch, colw, hw, id="rw", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[f_ban, f_l1, f_r1], onPage=num),
        PageTemplate(id="rest", frames=[f_l, f_r], onPage=num),
        PageTemplate(id="wide", frames=[f_wide, f_lw, f_rw], onPage=num)])

    banner_cap = ("<b>Figure 1. FLEX-UF end to end.</b> Grey is frozen and "
                  "never touched; blue is inherited from DCVC-UF and "
                  "fine-tuned; orange and green are new. The first j block "
                  "groups run over the whole frame and the rest run per tile "
                  "on a shrinking active set. The exit map that drives the "
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
