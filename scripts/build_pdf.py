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
          .replace(r"\times", "×")
          .replace(r"\rightarrow", "\u2192").replace(r"\to", "\u2192").replace(r"\emph{", "<i>")
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
        # Map the key to the number this document gives it, rather than
        # dropping the citation. A comparison table of other people's numbers
        # is only usable if the reader can see whose numbers they are, and the
        # keys are stable while the numbers move as REFS grows.
        def _cite(m):
            keys = [k.strip() for k in m.group(1).split(",")]
            nums = [str(CITE[k]) for k in keys if k in CITE]
            return (" [" + ", ".join(nums) + "]") if nums else ""
        line = re.sub(r"~?\\cite\{([^}]*)\}", _cite, line)
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


def simple_table(rows, width, first_col=0.17):
    """A small table whose content is prose, not measurement.

    Everything in tables/ is generated from results/ and must stay that way.
    The notation table in Section 3.1 carries no measurement, so it lives with
    the text it explains, in this file and in main.tex alike.
    """
    text_rows = [[sub(c) for c in r] for r in rows]
    fs = 7.2
    cw = [first_col * width, (1 - first_col) * width]
    cells = [[Paragraph(c, S("cell", fontSize=fs, leading=fs * 1.16,
                             alignment=0))
              for c in r] for r in text_rows]
    t = Table(cells, colWidths=cw, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), CELL_PAD),
        ("RIGHTPADDING", (0, 0), (-1, -1), CELL_PAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    t.spaceBefore = t.spaceAfter = 5
    return t


_FIGN = [0]
_TABN = [0]


def _autonum(cap, counter, word):
    """Replace a hand-written 'Figure N.' with the running count.

    The numbers were typed into the captions and drifted the moment a figure was
    inserted in the middle -- which happened twice tonight.
    """
    counter[0] += 1
    # The placeholder may be a digit from an earlier build or a literal "N"
    # typed by whoever wrote the caption. Matching only digits meant a caption
    # written as "Figure N." kept the N, incremented the counter, and printed
    # "Figure N." in the submitted PDF.
    return re.sub(rf"{word}\s+(?:\d+|[A-Z])\.", f"{word} {counter[0]}.",
                  cap, count=1)


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
WIDE_BAND = 2.62 * inch


def content(colw, fullw):
    """(flowables for the two-column body, flowables for the full-width banner)."""
    F = []
    A = F.append

    def figure_wide(name, cap, height=WIDE_BAND - 0.62 * inch):
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

    def rows_tbl(rows):
        # main.tex sets this as a tabular inside a center environment, which
        # LaTeX will not break across a column. Keep it whole here too, or the
        # notation table loses its header on the far side of the break.
        A(KeepTogether(simple_table(rows, colw)))

    def tbl(name, cap, width=None):
        # width lets a table that cannot be read at column width run the full
        # page. The literature comparison is six columns and eighteen rows, and
        # at 251pt every cell wraps.
        t = tex_table(name, width or colw)
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
        if t.wrap(width or colw, FRAME_H)[1] > 0.5 * FRAME_H:
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
        # Capped so a figure and its caption stay small enough to finish the
        # column they start in. Uncapped, several three-panel figures came to
        # about a third of a column with their captions, and any one of them
        # arriving with less than that left would jump to the next column and
        # strand the text behind it. Page 3 lost 35% of its left column that
        # way.
        A(KeepTogether(fig(name, colw, cap, maxh=1.25 * inch)))

    # ---- abstract -------------------------------------------------------
    A(Paragraph("<b>Abstract</b>", S("ah", fontName="Times-Bold", fontSize=9.4,
                                     alignment=TA_CENTER, spaceAfter=4)))
    A(Paragraph(sub(
        r"A learned image decoder reconstructs a picture by running a fixed "
        r"network over it. On the intra decoder we study, that network costs "
        r"453 GMAC for a 1080p frame and spends the same arithmetic everywhere "
        r"in it: a flat sky and a face are decoded at the same price. Making "
        r"decoders cheaper is now its own line of work, and almost all of it "
        r"changes the model, which changes the bitstream, so a file encoded "
        r"yesterday cannot benefit from a decoder redesigned today. What is "
        r"left to vary is not what the decoder is, but how much of it runs "
        r"where."
        r"<br/><br/>"
        r"Here we show that decoding computation can be allocated by content "
        r"on a decoder that is not allowed to change. A frame is cut into "
        r"tiles after the shared part of the decoder has run, and each tile "
        r"leaves the remaining trunk at its own depth, with the encoder, the "
        r"entropy model and the coded latent untouched. On \NumSeq CTC intra "
        r"frames a 0.1 dB constraint removes \MainLowRate to \MainHighRate% "
        r"of decoder multiply-accumulates across the rate range, \MeanAtOne% "
        r"on average, for a BD-Rate cost of \BdRateALow%."
        r"<br/><br/>"
        r"Two findings are not specific to this decoder. Adaptivity is usable "
        r"only inside a window that tiling itself creates, between a floor "
        r"below which no allocation meets the budget and a saturation point "
        r"above which none improves; rescaling the budget between those limits "
        r"collapses five rate curves \BandRawSpread points apart onto one "
        r"within \BandSpreadMean. And what is needed to allocate is already "
        r"in the file: routing on the bits the entropy model has spent on a "
        r"tile beats our trained \RouterParams-parameter router at every "
        r"rate, by up to \RateRankBeatsBy points, while agreeing with the "
        r"exhaustive search on fewer tiles. A compressed representation "
        r"records not only what to reconstruct, but how much computation the "
        r"reconstruction is worth."), ABST))
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
    par(r"Our method, FLEX-UF, attaches exits at intervals through the twelve "
        r"residual blocks of the DCVC-UF intra decoder; we call that ordered "
        r"set of exits the <i>ladder</i>. The first j blocks run over the "
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
        r"The tile-boundary artefact that follows is what we call the "
        r"<i>seam</i>, and under the default padding rule it costs "
        r"\SeamZerosHigh dB on this decoder at high rate. That is several times the entire "
        r"budget the method works to, and it is paid before any tile has saved "
        r"a single operation. In Section 4 we treat padding as an "
        r"<i>estimator</i> of the unseen neighbour and measure four of them. "
        r"The ordering we get is not the obvious one.")
    h2("A distortion budget only works inside a band.")
    par(r"Tiling costs something even when nothing exits early, so there is a "
        r"<i>floor</i>, and budgets below it admit no allocation at all. The "
        r"ladder also has a shallowest usable exit, so there is a "
        r"<i>saturation</i> point above which every tile already takes that "
        r"exit and more quality buys nothing. Between the two, in what we call "
        r"the <i>band</i>, the budget genuinely trades quality for compute. We "
        r"measure both ends at every rate in Section 5.4, where the 0.1 dB "
        r"budget uses \BandUseLow% of the band at low rate and \BandUseHigh% "
        r"at high rate.")
    par(r"The paper makes three claims and everything else in it is support "
        r"for one of them: that decoding computation can be allocated by "
        r"content on a decoder nobody is allowed to change; that a distortion "
        r"budget is a usable control only inside a window that tiling itself "
        r"creates; and that the file already says where the computation should "
        r"go.")
    h2("Contributions.")
    par(r"<b>(i)</b> An early-exit ladder for a learned image decoder that "
        r"picks a depth per tile, which is spatial adaptivity at the "
        r"granularity of a tile. The encoder, the entropy model and the coded "
        r"latent are untouched in both modes we report. <i>Signalled</i> "
        r"FLEX-UF leaves the latent unchanged and adds \MapBits bits per 1080p "
        r"frame of routing metadata, or \MapOverheadLow% of a typical "
        r"bitrate, and saves \MainLowRate% to \MainHighRate% of decoder MACs "
        r"for 0.1 dB on the common test set at a BD-Rate cost of "
        r"\BdRateALow%. <i>Bitstream-identical</i> FLEX-UF adds nothing at "
        r"all, decodes a byte-identical file, and reaches \RouterLowRate% to "
        r"\RouterHighRate%.")
    par(r"<b>(ii)</b> The band a distortion budget works in: a floor below "
        r"which no allocation is feasible, a saturation point above which none "
        r"improves, both in closed form, and seven propositions verified "
        r"numerically rather than asserted. Measuring the budget in units of "
        r"that band accounts for most of the rate dependence, on two "
        r"independently trained checkpoints.")
    par(r"<b>(iii)</b> The bits a tile has already cost predict how deep it "
        r"has to decode, better than a trained router does. Routing on the "
        r"entropy model's own output beats our \RouterParams head at every "
        r"rate, by up to \RateRankBeatsBy points, with nothing added to the "
        r"file and nothing learned. It does so while picking the search's exit "
        r"on FEWER tiles than the head does, which says that accuracy over "
        r"exit labels is the wrong objective and that ordering is what a "
        r"router has to get right. A zero-learned-parameter control that adaptive-inference work is "
        r"rarely measured against. Routing on the bits the entropy model has "
        r"already spent per tile costs nothing, adds nothing to the stream, and "
        r"beats our trained head at every rate, while agreeing with the oracle "
        r"on fewer tiles than the head does.")
    h1("2. Related work")
    par(r"Three lines of work meet in this paper and none of them quite "
        r"contains it. Early exit made depth a per-input decision and left it "
        r"there; spatially adaptive inference made it a per-region decision "
        r"but chose between separate networks; and learned compression has "
        r"spent a decade making decoders cheaper by making them different, "
        r"which is exactly what a deployed bitstream will not tolerate. What "
        r"follows sets out what each gives us and what none of them gives.")
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
        r"(Section 3.5).")
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
        r"<b>Table 1. Where this sits.</b> What each method varies, at what "
        r"granularity, and what that costs the stream. A single new-bitstream "
        r"column cannot represent our own two modes, so the last three ask "
        r"what actually changes: whether the coded latent is the same, what "
        r"side data travels with it, and whether the decoder alone can do it.")
    tbl("literature_img",
        r"<b>Table 2. What a learned image decoder costs.</b> Eleven codecs as "
        r"published, measured on one machine by Li et al. [43]: BD-Rate against "
        r"VTM-22.0 on Kodak, arithmetic per pixel and parameters. The spread in "
        r"cost is an order of magnitude, and every entry pays its cost on every "
        r"pixel of every image.")
    tbl("literature_vid",
        r"<b>Table 3. The family this work modifies</b>, from DCVC-UF [14]: "
        r"MACs per 1080p frame, parameters, and BD-Rate against VTM-17.0 low "
        r"delay. Comparable within this table and not against Table 2, which "
        r"uses a different anchor and a different test set. The last column is "
        r"the one this paper is about: whether the decoder spends more "
        r"computation where the picture needs it. Our row is the intra decoder "
        r"of the last DCVC-UF entry, at the 0.1 dB budget.")
    figure("field.png",
           r"<b>Figure 2. Every published decoder is one number.</b> "
           r"Arithmetic per pixel for the eleven image codecs of Table 2, and "
           r"for the intra decoder this work modifies. Costs are comparable "
           r"here even though the BD-Rate anchors are not, which is why this "
           r"axis and not a rate-distortion one. The bar at the top is the "
           r"same decoder as the rest of the paper: a 0.1 dB budget removes "
           r"the pale section, and how much it removes depends on the "
           r"picture.")
    figure("tiles_unequal.png",
           r"<b>Figure 3. Why depth should not be uniform.</b> One 1080p "
           r"frame, 40 tiles. <b>a</b>, what each tile loses if it stops at "
           r"the shallowest exit the ladder allows: a few lose nothing "
           r"measurable and one loses 0.71 dB. A single depth for the frame "
           r"has to be set by the worst of them. <b>b</b>, the bits the "
           r"entropy model has already spent on a tile against the exit the "
           r"Lagrangian oracle sends it to, at the 0.1 dB budget. The signal "
           r"the decoder needs is already in the file.")
    par(r"Together the two tables say where the field spends its decoder "
        r"budget. Complexity has moved a long way in both directions: TCM [46] "
        r"and WeConvene [50] buy rate with an order of magnitude more "
        r"arithmetic per pixel than CHARM [51] or STF [45], and DCVC-UF [14] "
        r"moves the other way, cutting a 1080p frame from DCVC-FM's 2642 GMAC "
        r"to 170. Two things are constant down both tables. The cost is a "
        r"property of the model, chosen once and paid on every image; and it is "
        r"uniform over the frame, so a flat sky and a face are decoded at the "
        r"same price. Those are the two the ladder in this paper changes, and "
        r"it changes them without touching the model that was shipped.")
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
        r"also transmit a mode map. The differences are in what the regions "
        r"choose among and in what "
        r"the map buys. Their regions choose among several <i>separately "
        r"trained</i> codecs, so the decoder must hold all of them and its "
        r"complexity is that of one codec regardless of the choice; the map "
        r"buys rate, not compute. Ours choose a prefix length of a single "
        r"trunk, so the weights are shared by construction and the map buys "
        r"compute at fixed rate. Because their alternatives are unrelated "
        r"networks, no decoder-side predictor could stand in for the encoder's "
        r"search. Our exits are prefixes of one another, and that nesting is "
        r"what lets a decoder-side predictor stand in at all (Section 3.1).")
    h2("Signalling versus prediction.")
    par(r"Being <i>told</i> a mode decision rather than inferring it is the "
        r"norm in standardised video coding. HEVC [9] and VVC [21] transmit "
        r"partitioning, prediction mode and transform tree. We evaluate both, "
        r"and treat the signalled variant as the conventional design "
        r"(Section 3.1).")
    h2("Deferring to an oracle under a budget.")
    par(r"Letting a predictor decide most cases and handing a small budget of "
        r"the hardest ones to something exact is the shape of selective "
        r"prediction [38] and learning to defer [39, 40], and of the budgeted "
        r"variant of the latter [41]. We build on that shape in Section 5.7. "
        r"Two things differ, and both make our case easier. The expert here is the encoder's own "
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
    par(r"The construction is short. A frame is cut into tiles once the "
        r"shared part of the decoder has run, each tile is allowed to leave "
        r"the remaining trunk at its own depth, and a small adapter reconciles "
        r"an early departure with a head that was fitted to the full one. "
        r"Everything difficult is in the three questions that follow from it: "
        r"what cutting costs, what a budget can buy, and who decides. This "
        r"section states the construction and the decision rule; Section 4 "
        r"prices the cut and Section 5 measures the rest.")
    h2("3.1 Setting and notation")
    par(r"We set out here the decoder we work on, the notation the rest of the "
        r"paper uses, and the three configurations we compare.")
    figure("ladder.png",
           r"<b>Figure 4. The ladder, priced.</b> What a tile costs at each "
           r"exit, as a percentage of one released decode, counted with hooks "
           r"on the executed pass and beside the arithmetic model the argmin "
           r"uses. Exits below the split depth are not distinct, because the "
           r"first j groups run for every tile whatever it does. The deepest "
           r"exit costs slightly more than the release, which is the "
           r"full-frame deblocking pass. Reported savings in this paper are "
           r"the hook count.")
    figure("allocation.png",
           r"<b>Figure 5. Where the tiles actually go</b>, at the 0.1 dB "
           r"budget, pooled over the whole test set. At the lowest rate two "
           r"thirds of tiles take the cheapest available exit; at the highest "
           r"only a fifth do and a quarter need the deepest. The allocation is "
           r"not degenerate at either end, which is what makes the choice "
           r"worth making.")
    h2("The decoder.")
    par(r"We work on the intra decoder of DCVC-UF [14]. The analysis side "
        r"stays frozen throughout: the patch embedding, the chunk encoder, the "
        r"entropy model and the coded payload are never touched (Figure 1). "
        r"What we replace is the synthesis trunk that turns the decoded latent "
        r"into a picture. We call the unmodified public decoder the "
        r"<i>released decoder</i>, and one full-frame run of it is the unit of "
        r"both cost and quality below.")
    h2("Tiles and exits.")
    par(r"A frame is cut into square tiles, 256 px on a side unless we say "
        r"otherwise, and each tile is carried through part of the trunk on its "
        r"own. Tiles are indexed by t and there are N of them, N=40 at 1080p. "
        r"The trunk itself is a stack of twelve blocks; we cut it into K "
        r"groups and attach an exit to each, so exits are indexed by k and "
        r"exit K−1 is the whole trunk. The first j groups run once over the "
        r"whole frame and the remaining K−j run per tile, so j is the "
        r"<i>split depth</i>. A tile that leaves at exit k runs groups j to k "
        r"and skips everything after it. Writing c_k for what a tile costs at "
        r"exit k, a frame that assigns exit k_t to tile t costs the mean of "
        r"c over its tiles. Tiles are cut on the trunk's feature grid, which "
        r"is coarser than the pixel grid, but we quote tile sizes in pixels.")
    h2("Distortion and the budget.")
    par(r"D(t,k) is the error tile t incurs if it leaves at exit k. It is "
        r"measured on the path we deploy at inference, a tiled decode, against "
        r"the released decoder's full-frame decode of the same latent, so what "
        r"tiling costs is inside it; we call that path the <i>deployed path</i> "
        r"and the table built on it the <i>deployed table</i>. Tapping the "
        r"exits of one full-frame forward pass gives a cheaper <i>full-frame "
        r"table</i> that is not the same quantity, and Section 5 "
        r"measures the difference. Every result is quoted at a <i>distortion "
        r"budget</i>, the decibels a decoded frame is allowed to fall below "
        r"the released decoder on that same latent, and the headline budget is "
        r"0.1 dB. To allocate is to choose an exit for every tile so that the "
        r"frame lands on its budget as cheaply as possible. A budget only buys "
        r"something inside its band, because tiling costs quality before any "
        r"tile exits early and the shallowest usable exit bounds what can be "
        r"saved; Section 5.4 measures both ends. Rate is set by a quality "
        r"index running from q0, the lowest rate, to q63, the highest, and we "
        r"report five of them: q0, q16, q32, q48 and q63.")
    rows_tbl([
        ["symbol", "meaning"],
        ["t, N", r"tile index and tiles per frame; N=40 at 1080p"],
        ["k, K", r"exit index and number of exits, k in 0..K−1; K=6"],
        ["j", r"split depth: groups 0..j−1 run full frame, j..K−1 per tile; "
              r"j=2"],
        ["c_k", r"cost of a tile at exit k, as a fraction of one released "
                r"full-frame decode"],
        ["D(t,k)", r"error of tile t at exit k, tiled decode against the "
                   r"released full-frame decode of the same latent"],
        ["λ", r"multiplier trading c_k against D(t,k), bisected to the budget"],
        ["β", r"the same trade applied to the router's logits"],
        ["q", r"quality index, q0 the lowest rate and q63 the highest"],
    ])
    h2("Three configurations.")
    par(r"All three choose the same object, one exit per tile, on the same "
        r"weights and the same coded payload. Configurations <b>A</b> and "
        r"<b>B</b> differ only in where that choice is made, and <b>C</b> sits "
        r"between them.")
    par(r"<b>A</b> decides at the encoder. The encoder holds the source, so it "
        r"knows D(t,k), picks each tile's exit by the Lagrangian of Section "
        r"3.5, and transmits the resulting <i>exit map</i>, one exit index per "
        r"tile, which is the analogue of the mode map a standard codec "
        r"signals. It entropy-codes to ~\MapBits bits per 1080p frame, or "
        r"\MapOverheadLow% of a typical bitrate. It costs the encoder about one extra decode per frame and "
        r"the decoder nothing.")
    par(r"<b>B</b> decides at the decoder. A \RouterParams-parameter head "
        r"reads decoded data only and predicts the same map (Section 3.5). "
        r"Nothing is transmitted and the file stays byte-identical to the "
        r"released one. It costs the decoder \RouterCostPct% of its own "
        r"multiply-accumulates, which is charged inside every B number we "
        r"report. Because that head reads nothing the encoder cannot also "
        r"read, the encoder can run it too, and so knows tile by tile where it "
        r"will be wrong.")
    par(r"Two of these are deployment modes and we name them so that no claim "
        r"in this paper is ambiguous about what is sent. <i>Signalled</i> "
        r"FLEX-UF is configuration A: the coded latent is unchanged and a small "
        r"exit map travels beside it. <i>Bitstream-identical</i> FLEX-UF is "
        r"configuration B: nothing is added, and the decoder reads a file that "
        r"is byte for byte the one the released encoder produced. Every saving "
        r"quoted in this paper is labelled with the mode it was measured in.")
    par(r"<b>C</b> interpolates between the two. The encoder signals a "
        r"fraction ρ of the tiles, the ones where leaving B alone is most "
        r"costly, and B decides the rest, so ρ=0 is B and ρ=1 is A. Any "
        r"decoder-side predictor can take B's place there, and Section 5.7, "
        r"which measures C, tries a second one. C transmits a mask rather than "
        r"a full map, costs the encoder A's search plus one run of the "
        r"predictor, and costs the decoder whatever that predictor costs.")
    h2("3.2 Where the computation is")
    par(r"The DCVC-UF intra decoder is one upsampling block, twelve "
        r"DepthConvBlocks and a head, costing 453.5 GMAC per 1080p frame. The "
        r"twelve blocks are 89.4% of that, which is why we build the ladder "
        r"across them. Inside one block at C=384 channels, the only operator "
        r"with any spatial extent is a 3×3 depthwise convolution, and it costs "
        r"9C against the block's 8C²+9C. That is <b>0.29%.</b> We come back to "
        r"the fraction several times below. Tile borders damage the picture "
        r"through it, and it is what makes the exact remedy of Section 4.4 "
        r"affordable at all. It is also why we kept the adapters pointwise.")
    h2("3.3 The exit ladder")
    par(r"Two consequences follow from the ladder, and both are easy to state "
        r"wrongly. Exits "
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
           r"DepthConvBlock and the two adapters, drawn to scale: bar "
           r"length is a share of the block, bar height the channel width "
           r"written. Neither adapter contains the block's only 3×3 (the "
           r"hairline rule), so neither adds receptive field or seam "
           r"penalty. <b>b</b>, blocks skipped per exit, coloured by "
           r"adapter; the rule switches to the FFN at four skipped blocks. "
           r"Lengths counted with hooks off the modules themselves.")
    figure_wide("patchify.png",
                r"<b>Figure 4. What patchify does.</b> Captured from one real "
                r"decode. <b>a</b>, the frame, padded to a whole number of "
                r"tiles, with the grid the decoder will impose. <b>b</b>, the "
                r"feature map after the shared stem, at one eighth of frame "
                r"resolution, so a 256 pixel tile of picture is 32×32 of "
                r"features; the tiling happens here and not in the pixel "
                r"domain. <b>c</b>, four of the 40 tiles as the trunk now sees "
                r"them, each its own batch element. <b>d</b>, the operation "
                r"itself, and its inverse. It costs no arithmetic and the "
                r"round trip is exact.")
    h2("3.4 Exit adapters")
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
        r"<b>Table 2. What the adapters are worth.</b> dB below the released "
        r"decoder with "
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
        r"because without them it has no usable exit.")
    h2("3.5 Allocation")
    par(r"The ladder and its costs are fixed by this point, and one decision is "
        r"left. Every tile has to be given an exit, drawn from the usable ones, "
        r"k ≥ j, and we want the set of choices that keeps the frame inside its "
        r"distortion budget for the least compute. There are K−j choices per tile "
        r"and N tiles, so searching assignments one by one is out of the "
        r"question. What makes the problem tractable is that the tiles do not "
        r"interact: each contributes its own error D(t,k) and its own cost "
        r"c_k, and the only thing they share is the single budget the frame "
        r"has to meet.")
    par(r"Problems of that shape are solved with a Lagrangian. Put a price λ on "
        r"compute, add λ c_k to the error of exit k, and the budget constraint "
        r"drops out; what is left is N separate one-tile problems, each a "
        r"minimum over K−j numbers. Shoham and Gersho [28] showed that sweeping "
        r"the price this way traces the lower convex hull of what the units can "
        r"achieve together, and Ortega and Ramchandran [29] made the "
        r"construction standard in image and video coding.")
    par(r"What λ does is set the exchange rate between compute and error. At "
        r"λ=0 compute is free and every tile takes whichever exit has the "
        r"smallest error. As λ rises, compute becomes expensive relative to "
        r"error, and a tile drops to a shallower exit as soon as what that exit "
        r"saves is worth more than the error it adds. Compute falls and error "
        r"grows as λ grows, neither of them turning back, so one value of λ "
        r"puts the frame exactly on its budget and we find that value by "
        r"bisection. The rule each tile follows is")
    eq(r"k^{*}(t) = \mathrm{arg\,min}_{k}\ \left[\, D(t,k) + \lambda\, c_{k} \,\right]")
    par(r"with the minimum taken over k ≥ j. We call k*(t) the choice of the "
        r"<i>Lagrangian oracle</i>: it is made with the true error of every "
        r"exit in hand, which no decoder has, but it is optimal only within the "
        r"family of allocations one multiplier can reach. That qualifier is not "
        r"cosmetic. Section 5.7 exhibits allocations that are cheaper at the "
        r"same distortion than anything this rule produces, so <i>oracle</i> "
        r"here means best under a single multiplier and not the global "
        r"constrained optimum. Where we mean the latter we say so. Both "
        r"quantities are available <i>to the encoder</i>, which holds the "
        r"source, so configuration A can run this search exactly.")
    par(r"What the search costs the <i>encoder</i> depends on how the table of "
        r"D(t,k) is built, and the two ways of building it are further apart "
        r"than they look. Take one tiled decode at 1080p as the unit. The "
        r"deployed table of Section 3.1 needs one tiled decode per exit, which "
        r"comes to 4.6× the unit rather than K times it, because a shallow "
        r"exit is cheaper to run than a full one. The full-frame table taps "
        r"every exit off a single full-frame pass and comes to 1.4×. "
        r"Section 5 shows that the full-frame table must not be used to "
        r"<i>report</i> quality, but it can still be used to <i>rank</i>. "
        r"Running the argmin on it, and the deployed path only to check the "
        r"budget, we find the two searches agree on 85% of tiles and land "
        r"within 0.4 points of saving of each other (19.0% against 18.6%, both "
        r"under budget). The exact search therefore costs 4.6 decodes and a "
        r"practical encoder need not pay it; ranking on the full-frame table "
        r"is the one extra decode per frame we quote for A.")
    par(r"The decoder cannot run the argmin. D(t,k) is the error against the "
        r"source, and the source is the one thing a decoder never receives. The "
        r"gap is one of information: the errors the argmin compares depend on a "
        r"picture that is not in the bitstream, so no decoder-side model "
        r"recovers them, however large it is. What a "
        r"decoder can do is estimate which exit the Lagrangian oracle would have picked, "
        r"and configuration B therefore <i>predicts</i>. Its head reads what "
        r"the decoder already holds, the feature at the split point, the "
        r"decoded latent y-hat, the entropy model's scales and the quality "
        r"index. For each tile t it emits a vector of logits z_t, one entry per "
        r"exit, and the tile takes")
    eq(r"\hat{k}_{t} = \mathrm{arg\,max}_{k}\ \left[\, \log\,\mathrm{softmax}(z_{t})_{k} - \beta\, c_{k} \,\right]")
    par(r"The first term is the head's score for exit k, on a log scale. The "
        r"second is the price on compute that λ carries in the Lagrangian: "
        r"raising β takes more away from the deep exits than from the shallow "
        r"ones, so more tiles are handed to cheap exits and the frame again "
        r"gets cheaper and worse. We bisect β against the budget just as we "
        r"bisect λ.")
    par(r"<b>Where β comes from at deployment.</b> A real decoder cannot run "
        r"that bisection. It would need to know the delivered distortion, and "
        r"the distortion is measured against a source the decoder never "
        r"receives, which is the same missing variable that stopped it "
        r"running the Lagrangian in the first place. Saying otherwise would "
        r"smuggle the source back in at test time.")
    par(r"So β is not chosen at test time. It is calibrated offline, once, on "
        r"a held-out set, and shipped as a table indexed by the quality index "
        r"and the budget. The decoder reads the quality index out of the "
        r"bitstream, looks β up, and runs the argmax above; nothing in the "
        r"loop needs the source. The table is 64 quality indices by however "
        r"many budgets a deployment offers, which is a few hundred floats.")
    par(r"We calibrate it on the \HeldNCal held-out Open Images validation "
        r"frames, which are disjoint from the training images and from the "
        r"test sequences: β is bisected to the budget on those, once per "
        r"quality index, and the resulting table is then applied unchanged to "
        r"the test set. Section 5.5 reports both that number and the one "
        r"obtained by bisecting on the test set itself, because the "
        r"difference between them is the only honest measure of whether the "
        r"table transfers.")
    par(r"<b>Why a logarithm.</b> The oracle's rule adds a distortion to a "
        r"price. The head does not produce a distortion; it produces a "
        r"distribution over exits, and the quantity that plays the same "
        r"additive role is the surprisal −log p, which is what the head is "
        r"trained to make small on the oracle's label. Writing the rule with "
        r"log softmax puts the two terms in one currency: what exit k costs in "
        r"belief against what it costs in compute, with β the exchange rate "
        r"between them. It also makes the comparison invariant to a constant "
        r"added to every logit, which the softmax removes, so βc<sub>k</sub> "
        r"is the only thing tilting it. Using the probability itself would set "
        r"a number bounded in [0,1] against a cost measured in decodes, and "
        r"would compress every distinction between confident exits into a "
        r"range near one, where β would have to move by orders of magnitude to "
        r"change any decision.")
    par(r"The bracket for the bisection is [−2×10<super>4</super>, "
        r"2×10<super>4</super>], which is wider than it looks like it needs to "
        r"be. A trained head is confident: its log-probability gaps reach the "
        r"hundreds, and β has to be able to outweigh them before the "
        r"allocation moves at all. A narrower bracket we tried first, [−50, "
        r"50], never reached the all-deepest allocation and so never bracketed "
        r"the budget from below.")
    par(r"<b>The head, exactly.</b> Three things enter. The first is the "
        r"feature at the split point, 384 channels at one eighth of frame "
        r"resolution, which is the last thing every tile shares. The second is "
        r"the decoded latent ŷ concatenated with the entropy model's scales σ, "
        r"the predicted Gaussian width used to code each latent position: σ is "
        r"already computed during the decode and is, position by position, an "
        r"estimate of how hard that position was to code. The third is the "
        r"quality index. Each of the first two is projected by a 1×1 "
        r"convolution, to 48 and 32 channels, and then reduced to one vector "
        r"per tile by taking the mean and the standard deviation over the "
        r"tile. That gives 96 + 64 + 1 = 161 numbers per tile, which pass "
        r"through LayerNorm and a three-layer perceptron of width 256 with "
        r"SiLU activations, ending in K logits. The whole head is 144,030 "
        r"parameters and 0.162% of the decode it is deciding about, almost all "
        r"of it in the 1×1 on the stem, which is the only part that runs per "
        r"pixel. The perceptron runs once per tile, forty times for a 1080p "
        r"frame, so its width is nearly free.")
    figure("router_arch.png",
           r"<b>Figure N. The router head.</b> <b>a</b>, the three things "
           r"it reads, all already held by the decoder. <b>b</b>, each "
           r"projected by a 1×1, then reduced to one vector per tile by its "
           r"mean and standard deviation. <b>c</b>, a three-layer "
           r"perceptron, one pass per tile. Every shape and parameter count "
           r"is read off the module when the figure is drawn.")
    par(r"Two details matter for reproducing it. Exits below the split depth "
        r"do not exist, and their logits are masked to −∞ rather than to a "
        r"large negative constant; nothing in the objective penalises a "
        r"constant offset, the raw logits drifted to around −10<super>4</super> "
        r"during training, and a −10<super>4</super> mask then became the "
        r"<i>largest</i> entry in the row. The balancing term is a per-exit "
        r"bias added to the logits before the decision and updated by usage "
        r"rather than by a gradient, which is what makes it loss-free.")
    par(r"<b>One head for every quality index.</b> The index is an input, not "
        r"a selector. During training it is drawn uniformly at random for each "
        r"batch, so one set of weights covers all 64 operating points, and at "
        r"test time the same head runs at every rate. It enters as a single "
        r"number per frame, which means it can tell the head which operating "
        r"point it is at and cannot, even in principle, distinguish two tiles "
        r"of the same frame. That is why it stays live in every variant of the "
        r"input ablation: the variants then differ in per-tile information and "
        r"in nothing else, and the variant with only the quality index is the "
        r"control that measures what agreement is reachable with no per-tile "
        r"information at all.")
    par(r"The head is trained against a <i>frozen</i> decoder, so the exits it "
        r"chooses between do not move while it learns. Its target is the "
        r"oracle's choice k*(t) and the loss is a cross-entropy to that target, "
        r"with each tile weighted by its <i>regret</i>, meaning by how much the "
        r"bracket in the Lagrangian grows if the head's exit is used in place "
        r"of the oracle's. A tile whose two best exits are nearly tied then "
        r"counts for less than one where the wrong choice is expensive. A "
        r"router like this can collapse onto a single exit during training, so "
        r"it carries a balancing term; ours is loss-free [16] and is biased "
        r"toward the oracle's own exit distribution instead of toward uniform. "
        r"At a high λ the oracle genuinely does send every tile to one exit, "
        r"and forcing spread there would force mistakes. Section 5.5 measures "
        r"what B gives up against A.")
    figure("mechanism.png",
           r"<b>Figure N. How a multiplier becomes a map.</b> Five tiles "
           r"of one real frame. <b>a</b>, what each loses at each exit, "
           r"against the deepest. <b>b</b>, what each exit costs. "
           r"<b>c</b>, the two added with the multiplier that meets a "
           r"0.1 dB budget, each curve scaled to its own minimum; the "
           r"star is where the argmin lands and that is the tile's exit. "
           r"<b>d</b>, the map the same rule produces for all 40 tiles. "
           r"One number, λ, decides the whole frame.")
    h2("3.6 Training")
    par(r"All exits are decoded every step and the objective is")
    eq(r"\mathcal{L} = \mathcal{L}_{\mathrm{RD}} + w_{a}\,"
       r"\mathcal{L}_{\mathrm{anchor}} + w_{d}\,\mathcal{L}_{\mathrm{distill}}")
    par(r"where L_RD is the released rate-distortion loss averaged over exits, "
        r"weighted per exit by fixed α_k and scaled by λ_rd, the codec's own "
        r"rate-distortion weight, which is a different multiplier from the λ "
        r"of the Lagrangian above. The anchor term L_anchor pins the deepest "
        r"exit's reconstruction to the released decoder's, so the reference "
        r"every saving is quoted against cannot drift away underneath the "
        r"measurement. The distillation term L_distill supervises the adapters "
        r"in feature space, matching the feature at exit k to a "
        r"stop-gradiented copy of the feature at exit k+1. We work in "
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
           r"<b>Figure 3. The artefact, before anything is done about "
           r"it.</b> Bosphorus at 1080p, quality index 63, zero padding at "
           r"the tile border. <b>a</b>, the frame decoded in one piece. "
           r"<b>b</b>, the same bitstream and weights decoded in 256 pixel "
           r"tiles, every tile at full depth: the difference between the "
           r"two, amplified 30 times. <b>c</b>, the boxed region of each.")
    par(r"Cutting a frame into tiles is what makes per-tile depth possible "
        r"and it is not free, so before any saving is claimed the cut has to "
        r"be priced. This section does that, and the price is larger and "
        r"stranger than the literature on tiled inference suggests.")
    par(r"A 3×3 depthwise at feature position (x,y) computes a weighted sum "
        r"over its neighbours. Decoded full frame, those neighbours exist. "
        r"Decoded per tile they do not, and the kernel is handed whatever the "
        r"padding rule invents. Each convolution extends the affected "
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
        r"has a reading. The count of affected pixels grows like perimeter "
        r"times depth, and the error accumulated in each of them grows with "
        r"how many convolutions reached it. Once every pixel has been touched "
        r"the area law saturates and the penalty does not, which is where the "
        r"area law fails as a predictor.")
    figure("contamination.png",
           r"<b>Figure 4. The seam against per-tile depth.</b> <b>a</b>, "
           r"the measurement with the fitted power law. <b>b</b>, both "
           r"models against q63, one free scale each. <b>c</b>, mean "
           r"relative error. The area fraction saturates once the border "
           r"reaches every pixel; the penalty does not.")
    h2("4.1 Padding is an estimator")
    par(r"Border padding is an <i>estimator</i> of the unseen neighbour, and "
        r"the seam is its error. The table below measures four of them, with "
        r"early exit switched off so that tiling is the only difference from a "
        r"full-frame decode.")
    tbl("padding",
        r"<b>Table 1. Border estimators</b>, dB below the released decoder on "
        r"the same latent, 256 px tiles, full CTC. Lower is better. "
        r"Replication, a zero-order hold, beats linear extrapolation by a wide "
        r"margin, and the fitted AR(1) wins on quality while costing +10.7% of "
        r"decode wall-clock.")
    par(r"A higher-order estimator is worse. Extrapolating the local "
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
           r"present. Panel (b): the floor is charged <i>inside</i> the "
           r"distortion budget and consumes a growing share of it.")
    par(r"If padding is the estimator, the obvious next question is what a "
        r"better one is worth, and the answer is: less than changing the "
        r"geometry. Tile size is free in arithmetic. A tiled decode's MAC count does not "
        r"depend on it at all, and the damage scales with the tile perimeter, "
        r"as the table below shows. What tile size costs is routing "
        r"granularity, 40 tiles per 1080p frame at 256 px against 160 at "
        r"128 px.")
    tbl("tilesize",
        r"<b>Table 2. Tile size</b>, dB at q63. Doubling the side roughly "
        r"halves the penalty, as the power law above predicts, and costs no "
        r"computation.")
    par(r"The largest factor of all is the easiest one to miss. On the "
        r"untrained ladder, replicate padding leaves \SeamReplHigh dB at q63; "
        r"after training, the same setting leaves \FloorHigh dB. So more "
        r"than two thirds of what the estimator could not fix is absorbed by "
        r"weights learning to live with it. That single change removes more of "
        r"the seam than any module in this paper does.")
    h2("4.3 A learned deblocking filter, and why it is not sufficient")
    figure("seam_gate.png",
           r"<b>Figure N. The deblocking gate, and what it does.</b> "
           r"<b>a</b>, the trained gate G, one scalar per position within a "
           r"tile, shared across all 384 channels. <b>b</b>, the same gate "
           r"tiled over the canvas, which is how it is applied. <b>c</b>, the "
           r"correction it actually adds on a real frame: the tile lattice and "
           r"nothing else. <b>d</b>, a cut through the middle of a tile. The "
           r"gate reaches 0.76 at a corner and then flattens at 0.17 rather "
           r"than switching off, so the module is not a pure boundary filter.")
    par(r"What a standard codec does about a partition boundary is deblock it, "
        r"with a filter applied after reconstruction [9, 21]. The tile lattice "
        r"is known "
        r"exactly at training and at inference, so ours can be <i>told</i> "
        r"where to look instead of having to infer it")
    eq(r"\mathrm{Rep}(f) = f + G[\,x\ \mathrm{mod}\ P,\ y\ \mathrm{mod}\ P\,]"
       r"\cdot \mathrm{PW}\left(\mathrm{WSiLU}(\mathrm{DW}_{3\times3}(f))\right)")
    par(r"with PW a pointwise convolution, DW a depthwise one, WSiLU the "
        r"weighted SiLU of the DCVC line [33], G a P×P gate shared over "
        r"channels, P the tile pitch in feature samples, and G initialised at "
        r"exp(−d/τ) in the distance d to the nearest tile boundary with τ a "
        r"fixed decay length. It costs 0.95% of the decode.")
    par(r"It does not earn that. Splitting the per-pixel error by distance "
        r"from the nearest tile boundary, we find the filter gains 0.27% in "
        r"the 0–4 px ring, which is 6.2% of pixels, and loses 0.04–0.05% "
        r"everywhere else. Give it a <i>perfect</i> gate, with zero correction "
        r"in the interior and the boundary gain unchanged, and the ceiling on "
        r"what it could earn is ≈0.0008 dB, for 0.95% of the decode. We "
        r"rejected AR(1) padding at 0.0019 dB per point of decode, and this is "
        r"worse by an order of magnitude. Tightening the gate cannot rescue "
        r"it, because there is almost nothing left to win.")
    h2("4.4 Removing the cause, and why it does not help")
    par(r"Only 0.29% of each block has spatial extent, so the exact fix is "
        r"affordable. Give the 3×3 its real neighbours across the tile border, "
        r"a <i>halo exchange</i> narrowed to the one operator that needs it; "
        r"local padding [31] does the same at every layer. The cost is "
        r"+0.032% of the decode, thirty times less than the deblocking "
        r"filter. The fix is also exact. At uniform depth a tiled decode with "
        r"the exchange is bit-identical to a full-frame one, once the "
        r"comparison is not confounded by the filter, which runs on the "
        r"stitched frame and has no full-frame counterpart. Measured: 1.13e-2 "
        r"with the filter on, <b>exactly 0</b> with it off, against 6.06e-2 "
        r"for replicate padding.")
    par(r"The exchange also removes most of the floor. Switched on at inference, "
        r"it drops the floor from 0.036 to 0.003 dB at q0 and from 0.056 to "
        r"0.030 at q63, which is \CoupFloorDropLow% and \CoupFloorDropHigh% of "
        r"the tiling penalty.")
    par(r"And it destroys the allocation. At the same 0.1 dB budget the "
        r"saving falls from \CoupPaddedMid% to \CoupCoupledMid% at q32 and from "
        r"\CoupPaddedHigh% to \CoupCoupledHigh% at q63.")
    par(r"The reason is structural, and it is written into the condition on "
        r"the exactness claim. The exchange is exact when every tile is at the "
        r"<i>same</i> depth, and routing deliberately violates that condition. "
        r"With replicate padding a tile is entirely independent of its "
        r"neighbours, so the allocation is free to give adjacent tiles any "
        r"depths it likes. The exchange makes a tile depend on its neighbours' "
        r"features, and under routing those neighbours ran a different number "
        r"of blocks. A shallow tile reading a deep neighbour's activation is an "
        r"arrangement the weights have never seen, and that mismatch costs "
        r"more than the seam it removed.")
    par(r"The natural reading of the exactness result is that the exact fix "
        r"should simply be adopted, and that reading is wrong. We report the "
        r"negative result for that reason, and because the tension between "
        r"the halo exchange and routing belongs to the combination itself, so any "
        r"spatially-adaptive decoder will inherit it. One caveat keeps the "
        r"finding from being final, the same one that applies to the "
        r"deblocking filter. This model was trained with padding, and a run "
        r"trained <i>with</i> the exchange could reverse the result. The burden of proof "
        r"lies with that run.")

    # ---- 5 experiments ---------------------------------------------------
    h1("5. Experiments")
    par(r"The measurements answer the questions the method leaves open, in "
        r"the order it leaves them. What does adaptivity buy over the "
        r"alternatives that need no ladder at all; where does a budget stop "
        r"working; what does it cost to decide on the decoder's side instead "
        r"of the encoder's; and what does any of it come to in seconds and "
        r"joules rather than arithmetic.")
    h2("Setup.")
    par(r"We fine-tune the decoder on 512×512 crops from OpenImages with the "
        r"encoder frozen (max|Δ| = 0 is asserted every run). One λ_rd is drawn "
        r"per sample from a log-spaced range covering all 64 quality indices, "
        r"so a single set of weights covers the whole rate range. We evaluate "
        r"on one intra frame from each of the \NumSeq sequences of the common "
        r"test set (CTC: UVG, MCL-JCV "
        r"and HEVC classes B, C, D and E), one intra frame each.")
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
    par(r"<b>The protocol every number in this paper obeys.</b> Two "
        r"implementations of the same quantity have differed here by as "
        r"much as 3.29 saving points, and the choice between per-frame and "
        r"pooled distortion moves an integrated saving by 7 to 10, so the "
        r"convention is not a detail and we fix it once.")
    proto = [["reference", "the released DCVC-UF decoder, run full-frame on the "
                       "same latent"],
         ["distortion", "MSE pooled over the frame, then converted to dB "
                        "against that reference, then averaged over frames"],
         ["budget", "applied per frame, not to the set mean"],
         ["λ and β", "one scalar per (quality index, budget), not per frame "
                     "and not per tile"],
         ["the map", "re-decoded: every reported dB is a real decode of the "
                     "mixed-depth map, never the uniform-exit table"],
         ["saving", "hook-counted MACs of that decode over the released "
                    "decode's, on the same latent"]]
    rows_tbl([["", ""]] + proto)
    A(Paragraph(sub(
        r"<b>Table N. The measurement protocol.</b> Every number in this "
        r"paper is measured this way. The supplement varies each line in turn "
        r"and reports what the alternative costs."), CAP))
    h2("5.1 Main result")
    tbl("main_results",
        r"<b>Table 3. Decoder MACs saved</b> (%) against the released decoder, "
        r"per quality index and distortion budget, configuration A. The 0.3 and "
        r"0.5 dB rows sit on the architectural ceiling almost everywhere; past "
        r"that point a looser budget buys nothing.")
    par(r"A tenth of a decibel is an engineering convention. It is "
        r"an engineering convention, chosen because it is small against the "
        r"spacing of the rate points and because codec work has long used "
        r"differences of this size as a working tolerance. It is not evidence "
        r"that the difference is invisible, and we make no perceptual claim "
        r"for it. The 0.2, 0.3 and 0.5 dB rows are there so a reader who "
        r"disagrees with the choice can read off another one. We say this "
        r"before the numbers because every number in this section is "
        r"conditioned on it.")
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
        r"slightly more than the released decoder because it still pays the "
        r"deblocking filter; that 1.0095, not 1.0, is what every saving in this paper is "
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
           r"shallowest exit; the boat and the shoreline run deep.")
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
        r"exit 3 inside the budget and save \UniformThreeLow%, against the "
        r"Lagrangian oracle's \MainLowRate%. At q48 and q63 the only uniform "
        r"depth that fits is the deepest one, which <i>costs</i> "
        r"\DeepestUniformCost% instead of saving anything. At those two "
        r"rates, then, the comparison is between \OracleAtFortyEight% and "
        r"\OracleAtSixtyThree% against nothing at all, not against something "
        r"smaller. Averaged "
        r"over rates, the best static allocation saves \BestStaticMean% where "
        r"the adaptive one saves \MainMean%.")
    par(r"The two shuffled rows separate effects that are easy to conflate. We "
        r"keep the oracle's own exit histogram, so the mix of depths and hence "
        r"the average cost are unchanged, and assign it to tiles at random. "
        r"Quality falls sharply. What the allocation buys is knowing <i>which</i> "
        r"tiles can afford to run shallower; running some fraction of them "
        r"shallower is worth nothing by itself. The oracle-histogram bit ranking keeps the "
        r"same histogram and orders it by a signal the decoder already holds, "
        r"namely the bits the entropy model spent on each tile. This is the "
        r"cheapest router we can think of. It has no parameters and no "
        r"training, and the number is available before the trunk starts. It is "
        r"also the natural analogue of the confidence rules used by early-exit "
        r"classifiers [3, 20], which likewise read a signal off the network's "
        r"own output.")
    par(r"It recovers most of the gap. At q63, random costs \RandomDb dB and "
        r"the oracle 0.100 dB at identical compute, while the oracle-histogram bit ranking costs "
        r"\RateRankDb dB. That is \RateRankRecovers% of the oracle's advantage "
        r"over chance, at 0.68–0.79 agreement with the oracle's map. A learned "
        r"router therefore has to beat a calibrated bit rule that is already three "
        r"quarters of the way there. We would not have run that comparison "
        r"without this control, and we suggest that any adaptive-inference "
        r"paper report it. Given the Lagrangian oracle's histogram, what we measure here "
        r"is <i>ranking</i> and nothing else. Section 5.6 removes that crutch "
        r"and turns the same signal into a complete routing rule, which "
        r"matches the trained head across the whole rate range.")
    h2("5.3 Resolution, and the granularity of a tile")
    par(r"<b>This comparison is indicative and confounded.</b> The 128 and 256 "
        r"pixel ladders are different training runs, so anything that "
        r"separates them separates two runs as well as two tile sizes. The "
        r"size of that confound is measurable and it is not small: two runs of "
        r"the SAME recipe, at a matched epoch on the same frames, differ by "
        r"4.8 saving points, which is as large as the tile-size effect "
        r"reported below. We report the comparison because the resolution "
        r"dependence it exposes is real and was hidden by a 1080p-only test "
        r"set, and we do not read the tile size as its cause.")
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
        r"tiles and \SmallResLow%. At high rate the small resolutions fall "
        r"to \SmallResHigh% against \BigResHigh% for 1080p.")
    par(r"The natural explanation is granularity. With two tiles per frame "
        r"there is almost no allocation left to make, and the Lagrangian "
        r"degenerates towards a uniform choice. That suggests a fix. Choose "
        r"the tile size relative to the frame, since the MAC count does not "
        r"depend on tile size at all and only the seam does.")
    par(r"We tested that fix and it mostly does not work. We compare "
        r"the 256 px tiling against a 128 px one at q0, which "
        r"multiplies the tile count by 3.4. MCL-JCV (1080p) goes from 36.3 to "
        r"34.3, HEVC B (1080p) from 33.8 to 32.1, HEVC C (832×480) from 20.9 "
        r"to 17.6, and HEVC D (416×240) from 11.3 to <b>13.7</b>.")
    par(r"Smaller tiles help only at the smallest resolution, where the count "
        r"goes from two to eight. Everywhere else they cost between 1.7 and "
        r"3.3 points, including at 832×480, where the count rises from 8 to 28 "
        r"and the saving still falls. Beyond a modest number of tiles, the "
        r"extra seam outweighs the extra granularity. We report the comparison "
        r"as indicative, since the two tile sizes come from different "
        r"training runs and tile size is therefore confounded with training. "
        r"But the direction is consistent, and it is enough for us to say that "
        r"a resolution-adaptive tile size is not the easy win the granularity "
        r"argument suggests.")
    h2("5.4 The band a distortion budget works in")
    figure("saturation_RECIPE512.png",
           r"<b>Figure 8. Three regions, and only the middle one is a design "
           r"choice.</b> Below the floor no allocation meets the budget; above "
           r"saturation every tile already takes the cheapest exit. The "
           r"0.1 dB budget uses \BandUseLow% of the band at the lowest rate "
           r"and \BandUseHigh% at the highest.")
    tbl("operating",
        r"<b>Table 7. Floor and saturation</b>, dB below the released decoder. "
        r"The floor is what tiling costs with no early exit at all; saturation "
        r"is where every tile reaches exit j and the ceiling is attained.")
    figure("theory.png",
           r"<b>Figure 9. The structure of the allocation</b>, on one "
           r"frame. <b>a</b>, compute and distortion are monotone in the "
           r"multiplier. <b>b</b>, the frontier between floor and "
           r"saturation; shaded regions are infeasible or wasted. <b>c</b>, "
           r"what the Lagrangian reaches against the exact Pareto set, "
           r"enumerated by dynamic programming.")
    par(r"The argument that follows assumes tiles are independent. It "
        r"treats the frame's distortion as a sum over tiles, while the "
        r"deblocking pass runs on the stitched frame and therefore sees the "
        r"whole exit map at once, so the tiles are not strictly independent. "
        r"We measured the size of that coupling over 60 allocations and the "
        r"largest mismatch between the separable prediction and a real decode "
        r"is 5.5×10<super>-5</super> dB, four orders of magnitude below the "
        r"budget. We proceed as if the objective were separable and treat that "
        r"as measured rather than assumed.")
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
        r"because there are only four distinct exit costs. The sweep reaches 95 "
        r"allocations against a Pareto set of 635, and the loss from convexity "
        r"is at most 0.05 saving points. For this problem the standard "
        r"construction is essentially optimal.")
    par(r"It matters <i>why</i> that loss is small, since the reason is "
        r"structural and points at what would make it smaller. The reachable "
        r"costs form a lattice, and its spacing is set by how much the mean "
        r"cost moves when the multiplier crosses a breakpoint. If m tiles "
        r"each move to the next exit there, the spacing is at most "
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
        r"the lowest rate and nothing else, a range \SatWindowMb thousandths "
        r"of a decibel wide.")
    figure("budget_band.png",
           r"<b>Figure 9. The band is the rate dependence.</b> <b>a</b> Saving "
           r"against the budget, with each rate's floor and saturation point "
           r"ticked. <b>b</b> The same with the budget axis rescaled onto each "
           r"rate's own band. Five curves become one; dashed is the one-parameter "
           r"power law fitted to all of them.")
    par(r"And the band is almost all of the rate dependence. At a "
        r"matched decibel the five rates are \BandRawSpread points apart. "
        r"Rescale the budget axis onto each rate's own band, with the floor at "
        r"0 and saturation at 1, and they collapse onto a single master curve, "
        r"\BandSpreadMean points apart on average and \BandSpreadMax at worst. "
        r"The answer to ``how much does a 0.1 dB budget buy at this rate'' is "
        r"therefore, to within a couple of points, ``where does 0.1 dB sit in "
        r"this rate's band''. The floor and the saturation point are both in "
        r"closed form and both cheap to measure. What they leave over is small "
        r"enough that a deployment could calibrate the two ends and read the "
        r"rest off one curve. That curve is a one-parameter power law, "
        r"saving ≈ C·u^\BandExp, with C the architectural ceiling and u the "
        r"position in the band. We fit it in log space over all five rates and "
        r"get R² = \BandRTwo, worst residual \BandFitErr points. The claim we make "
        r"is the <i>rescaling</i>: measuring the budget in units of each "
        r"rate's own band is what collapses the curves. The exponent is an "
        r"empirical fit to that collapse and not a law. Inverse fits to the "
        r"same frontier disagree in it by up to 40%, so we report it as a "
        r"description of these curves and read nothing into its value.")
    par(r"The collapse appears robust across two checkpoints. We "
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
    figure("window.png",
           r"<b>Figure N. The window a budget works in.</b> <b>a</b>, saving "
           r"against the distortion actually delivered, per rate. Every curve "
           r"begins at a floor, below which no allocation meets the budget, "
           r"and flattens at a saturation point, above which a looser budget "
           r"buys nothing. <b>b</b>, those two limits against rate. The shaded "
           r"band is the only region in which a budget is a design choice "
           r"rather than a formality.")
    h2("5.5 Signalled versus predicted allocation")
    figure("router_ab.png",
           r"<b>Figure 9. Who decides.</b> <b>a</b>, the two decision paths "
           r"side by side: the encoder's search over all K exits per tile "
           r"against the head's forward pass over decoded data. <b>b</b>, "
           r"saving at 0.1 dB; shading is what a bit-exact bitstream costs. "
           r"<b>c</b>, that cost is smallest at q\GapMinQp\ and grows in "
           r"both directions.")
    tbl("ab",
        r"<b>Table 7. Signalled against predicted</b> at two budgets, same "
        r"checkpoint and test set, with one router trained against the oracle "
        r"on the deployed table at a single λ.")
    par(r"The table above compares the two, and what it prices is exactness "
        r"against bits.")
    par(r"<b>Not signalling costs \GapMin–\GapMax points</b>, roughly flat "
        r"across rate, for zero added bits and a byte-identical file. The gap "
        r"is smallest at q\GapMinQp and widens toward both ends of the rate "
        r"range.")
    par(r"It tracks |β|, the cost multiplier the bisection has to apply "
        r"to move a router trained at one λ onto another operating point. At "
        r"q\BetaMinQp the required β is \BetaAtMin, essentially none, since "
        r"that is where the training λ lands, and there the gap is at its "
        r"minimum. At q0 it is \BetaLow and the gap is \GapMax. A large β in "
        r"either direction lets the cost term dominate the logits, which "
        r"discards the content ranking the router learned. What we ship is "
        r"one head for every rate, with β read from an offline table indexed "
        r"by quality index and budget (Section 3.5). A head per operating "
        r"point would close this gap and we do not propose it: it multiplies "
        r"the stored model by the number of rates, and the rule of Section 5.6 "
        r"closes more of the gap for nothing.")
    par(r"Loosening the budget closes the gap only where the budget saturates "
        r"the ladder. At 0.3 dB the gap is \GapLooseLow points at the three "
        r"lowest rates. That is exactly the router's own \RouterCostPct% of "
        r"decode, so its <i>prediction</i> is free there; both configurations "
        r"send every tile to the cheapest exit, and nothing is left to predict "
        r"wrongly. At the two highest rates 0.3 dB does not saturate, and the "
        r"gap <i>widens</i> to \GapLooseHigh points. A looser budget gives the "
        r"allocation more room to be wrong in as well as more room to be "
        r"right.")
    par(r"For a system that trains once, the single-router number is the "
        r"honest one, so that is what we report. It understates what B can do.")
    tbl("beta_heldout",
        r"<b>Table N. Choosing β without the test set.</b> Left, β bisected on "
        r"the \HeldNCal held-out validation frames and then applied to the "
        r"test frames unchanged; right, β bisected on the test frames "
        r"themselves, which is what the signalled-against-predicted table "
        r"reports. Savings are hook "
        r"counts on the routed decode with the router's own compute charged "
        r"against them. The last column re-bisects on the test set to the "
        r"quality the held-out β delivered, so the two allocations are "
        r"differenced at one distortion rather than across two.")
    par(r"<b>Where the β above came from.</b> Every β in the signalled-against-"
        r"predicted table was bisected against the budget on the test frames "
        r"themselves, which is the one thing Section 3.4 says a decoder cannot "
        r"do. The table immediately above repeats the measurement with the "
        r"table a deployment would ship: β bisected once per quality index on "
        r"the \HeldNCal held-out "
        r"Open Images validation frames, then applied to the CTC frames "
        r"without being touched again. Checkpoint, head, frames and budget are "
        r"identical on both sides, so what separates the two columns is where "
        r"β came from.")
    par(r"At q0 the two agree, \HeldBetaLow against \BetaLow and "
        r"\HeldSavingLow% against \BLow%, which is the same measurement to a "
        r"hundredth of a point. At the rates in between, the held-out β misses "
        r"the budget on the high side: it delivers \HeldDbWorst dB at "
        r"q\HeldDbWorstQp where 0.1 dB was asked for, and \HeldNOver of the "
        r"\HeldNHeld rates it covers overshoot. Distortion and saving move "
        r"together, so those rows also report more saving, and reading one "
        r"column straight against the other would credit the router with "
        r"compute it bought using quality the budget did not allow. The last "
        r"column takes that back out, re-bisecting on the test set to the "
        r"quality the held-out β actually delivered: at equal delivered "
        r"quality the two allocations agree to within \HeldTransferAbsMax "
        r"points. What moves between the two sets is the decibel a given β "
        r"delivers.")
    par(r"The highest rate fails harder than that. On the calibration frames "
        r"the floor, the distortion tiling costs with every tile already at "
        r"the deepest exit, is \HeldNoBetaFloor dB at q\HeldNoBetaQp. That "
        r"is above the budget, so no allocation on that set meets 0.1 dB and "
        r"the bisection has nothing to return. The shipped table has a hole "
        r"where the highest rate should be, and a decoder holding it falls "
        r"back to the deepest allocation, which is our own full-depth path and "
        r"costs slightly more than the release, so it saves nothing. The test "
        r"frames, whose floor at that rate is \HeldFloorTestHigh dB, would "
        r"have supported \HeldNoBetaForgone points. What fails there is the "
        r"calibration set rather than the router: a budget written as an "
        r"absolute decibel sits a different distance above the floor on 512px "
        r"photographs than on 1080p video, and at the highest rate it sits "
        r"below it.")
    par(r"We keep the test-bisected figures as B's headline, since the "
        r"comparison against A is built on them, and the held-out table is the "
        r"correction a reader should apply. Calibrating on frames whose tiling "
        r"penalty matches the ones the decoder will meet is the fix, and for a "
        r"video decoder that means calibrating on video.")
    par(r"<b>Retraining with the mask fixed raises agreement and does not "
        r"simply raise the saving.</b> The head above was trained against a "
        r"suppression term sitting above its own logits (below). We retrained "
        r"it with the mask fixed, same recipe and same λ. Held-out agreement "
        r"rises from \RetrainAgreeOld to \RetrainAgreeNew, and the deployed "
        r"saving moves by +\RetrainGain points at q\RetrainGainQp and "
        r"−\RetrainLoss at q\RetrainLossQp. We find the retrained head ahead "
        r"where the required β is small and behind where it is large, which "
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
        r"cheapest exit for a reason unrelated to its content. The mask is now "
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
           r"<b>Figure 10. The calibrated bit rule.</b> <b>a</b> Saving at 0.1 dB; "
           r"shaded where the calibrated bit rule beats the trained head. "
           r"<b>b</b> What a tile's bit count is correlated with, against how "
           r"often the rule agrees with the oracle outright; dotted is the "
           r"head's own held-out agreement.")
    tbl("raterank",
        r"<b>Table 8. The calibrated bit rule</b>, 0.1 dB, same test set; it is "
        r"the ``rate-rank'' column. Bold where the rule beats the trained head. "
        r"ρ are Spearman "
        r"correlations between a tile's bit count and, respectively, the depth "
        r"the oracle assigns it and the distortion it stands to gain from that "
        r"depth.")
    par(r"It beats the trained head at every rate. The zero-learned-parameter "
        r"rule is ahead of the \RouterParams router at all \RateRankNWins measured "
        r"rates, by margins that run from half a point at q48 to "
        r"\RateRankBeatsBy points at q0. A head trained on this decoder "
        r"against this oracle therefore returns nothing over a rule with no "
        r"parameters at all, and it carries \RouterCostPct% of the decode "
        r"that the calibrated bit rule does not.")
    par(r"Why it works is not that it agrees with the oracle. It agrees "
        r"on \RateRankAgreeLo–\RateRankAgreeHi of tiles, well below the "
        r"router's 0.718, and still saves more at every rate. "
        r"Agreement weighs a disagreement on a tile where two exits are within "
        r"a hair of each other exactly as heavily as one where the choice is "
        r"most of the frame's error, and most tiles are the former. The "
        r"ordering is what the rule gets right. Bits correlate with the "
        r"<i>spread</i> across the ladder, that is, with how much a tile "
        r"stands to gain from depth, at "
        r"ρ_spread = \RateRankSpreadLo–\RateRankSpreadHi at every rate.")
    par(r"At a looser budget it stops being a baseline and becomes the "
        r"answer. At 0.3 dB the calibrated bit rule matches the oracle exactly at "
        r"the three lowest rates, where both reach the same architectural "
        r"ceiling, comes within 0.2 points of it at q48, and gives up "
        r"\RateRankLoose% against the oracle's \SigLooseHigh% at q63. "
        r"That is ahead of the "
        r"trained head at every rate, by up to \RateRankLooseAheadBy points. "
        r"The head is charged for itself and the rule is not. At a loose "
        r"budget the head's ordering also has less left to contribute, because "
        r"once the budget saturates the ladder there is little ordering left "
        r"to get right, and the rule gets it. At 0.5 dB it reaches the "
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
        r"0.1 dB, same test set. Bold where the calibrated bit rule beats that "
        r"run's own trained head.")
    par(r"It replicates on a second training run. "
        r"BEST is a separate recipe at a different epoch, with its own router "
        r"trained the same way. There the calibrated bit rule beats the trained head at "
        r"\BestRankWinsN of \BestRankOfN rates, by up to \BestRankBy points, "
        r"and comes within \BestRankToOracle points of the <i>oracle</i> "
        r"everywhere. The margins there are wider than here, and the one rate "
        r"it concedes is the only rate on either checkpoint where the head "
        r"finishes ahead, by under a point. We do not read that as showing a "
        r"learned head cannot be worth its parameters, only that neither of "
        r"our training runs produced one that is. The calibrated bit rule stays close "
        r"to the oracle on both.")
    par(r"Per-block bit allocation is a standard quantity in learned "
        r"compression, where it is something to <i>choose</i>; block-level "
        r"rate control sets it so that complex regions get more bits [42]. We "
        r"read the same number in the other direction, after the fact and at "
        r"the decoder, as a statement about how hard a region was. It costs "
        r"nothing because someone else has already paid for it.")
    tbl("blend",
        r"<b>Table 9. Blending the two decoder-side signals</b> at 0.1 dB, both "
        r"normalised to unit mean, weight w from the calibrated bit rule to the "
        r"head's ordering. Bold is the best per rate.")
    par(r"<b>Are the two signals complementary?</b> Barely. We normalise both "
        r"surrogates to unit mean and blend them with one weight w, where w=0 "
        r"is the calibrated bit rule and w=1 the head's ordering. The best blend is w=0 "
        r"at the three lowest rates and a small head weight at the two "
        r"highest, worth +2.1 points at q48 and +0.9 at q63. The head reads "
        r"the entropy model's scales, so it already has most of what the bit "
        r"count carries; what it adds is confined to q48 and q63.")
    par(r"A learned component should be measured against the free alternative, "
        r"and it rarely is. In adaptive inference the usual controls are a "
        r"uniform allocation and a random one. Both are much weaker than a "
        r"decoder-side signal that is already lying around.")
    figure("deciders.png",
           r"<b>Figure N. Three ways to choose an exit.</b> <b>a</b>, saving "
           r"at the 0.1 dB budget. The encoder search sees the source and "
           r"sends 89 bits a frame; the other two send nothing. The bit rule "
           r"beats the trained head at every rate, by the margin printed above "
           r"it, with no learned parameters against the head's 144,030. "
           r"<b>b</b>, bars are the tiles on which the rule picks the exit the "
           r"search would have, and the line is the fraction of the search's "
           r"saving it nonetheless captures. Picking a different exit on most "
           r"tiles costs it far less than picking a different exit sounds like "
           r"it should, which is the case against training a router on exit "
           r"labels.")
    h2("5.7 Signalling only what the router gets wrong")
    par(r"The encoder knows tile by tile where the router will be wrong, "
        r"because it can run that router itself (Section 3.1), and nothing "
        r"obliges it to correct every one of them. That is the room "
        r"configuration C works in.")
    par(r"We choose the ρN overridden tiles by Lagrangian regret, "
        r"Δ(t) = L(t,k-hat) − L(t,k*) with L(t,k) = D(t,k) + λc_k the "
        r"objective of the Lagrangian in Section 3.5, so Δ(t) is exactly what "
        r"that objective loses on tile t by staying silent. We hold β at B's "
        r"value and bisect λ over the overrides to land back on the budget. "
        r"The map costs an entropy-coded mask, N·H(ρ) bits with H the binary "
        r"entropy, plus 3 per override.")
    figure("hybrid.png",
           r"<b>Figure 10. Partial signalling.</b> <b>a</b>, saving against "
           r"the bits spent on the map; each line runs from B on the left "
           r"to A on the right. <b>b</b>, the same normalised by the A-to-B "
           r"gap; dashed is proportional recovery. <b>c</b>, against the "
           r"fixed-λ Lorenz bound; points above the dashed line are "
           r"allocations one multiplier cannot reach.")
    tbl("hybrid",
        r"<b>Table 8. Configuration C</b> at 0.1 dB. Columns are the fraction of "
        r"tiles the encoder overrides; the two end columns reproduce B and A to "
        r"within 0.1 points, which is the check that the interpolation is real.")
    par(r"The two ends check out. At ρ=0 the measurement reproduces "
        r"configuration B to the second decimal at every rate, and at ρ=1 it "
        r"reproduces A. Neither end is imposed; both fall out of the same code "
        r"path run against independently measured files, so the columns "
        r"between them are measuring something real.")
    par(r"Most of the gap is cheap. Overriding a tenth of the tiles for "
        r"\HybridBitsTenth bits per frame recovers "
        r"\HybridRecoverTenthLow–\HybridRecoverTenthHigh% of the gap, and a "
        r"fifth recovers \HybridRecoverFifthLow–\HybridRecoverFifthHigh%. "
        r"Recovery is concave everywhere, so the first bits spent are the most "
        r"useful ones.")
    par(r"And a partial map can beat a complete one. At \HybridBeatsAN "
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
    par(r"Why the recovery is concave, and what bounds it. At a fixed λ "
        r"the objective is separable, so overriding a set removes <i>exactly</i> "
        r"the sum of that set's regrets. The recovery of the <i>objective</i> "
        r"is then the Lorenz curve of the per-tile regret distribution, and "
        r"its curvature is that distribution's Gini coefficient, "
        r"\GiniMin–\GiniMax across rates. What the table reports is saving at "
        r"fixed distortion, not the objective, and returning to the budget "
        r"means re-bisecting λ. That re-bisection is also what lets the "
        r"two-multiplier allocations above escape the bound.")
    par(r"At the loose budget it matters where the gap is. At 0.3 dB "
        r"the three lowest rates are saturated, and there is nothing for a "
        r"partial map to buy. At \HybridLooseQps, where the gap is "
        r"\GapLooseHigh points, half the map recovers "
        r"\HybridLooseRecLo–\HybridLooseRecHi% of it. The allocation follows "
        r"the same rule here as everywhere else, which is to spend the bits "
        r"where the ladder still has somewhere to go.")
    par(r"A better predictor concentrates its regret, which is what the "
        r"Gini was for. We rebuild C on the retrained head of Section 5.5. "
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
        r"override rule over the calibrated bit rule instead of the head "
        r"is worth up to \CPredBitsAhead points at the lowest rate and costs "
        r"up to \CPredHeadAhead at the highest. Both converge on A at ρ=1 to "
        r"the second decimal at every rate, which is the third independent "
        r"check that the two code paths agree. The best single operating point "
        r"we measured is the calibrated bit rule with half the map signalled, at "
        r"\CBestSave% at q\CBestQp. That is \CBestOverA points <i>above</i> "
        r"full signalling, with no learned component anywhere in the decoder.")
    par(r"Configuration C is what we would ship where a small map is tolerable "
        r"and a large one is not. It also reframes the A–B gap. What the gap "
        r"measures is the price of <i>silence</i>, charged tile by tile, "
        r"rather than the price of prediction.")
    figure("concentration.png",
           r"<b>Figure N. The loss is concentrated.</b> <b>a</b>, the share of "
           r"the total regret carried by the worst fraction of tiles, per "
           r"rate; the dotted line is what an even spread would look like. "
           r"<b>b</b>, the same as a Gini coefficient. At every rate a small "
           r"minority of tiles carries most of what staying silent costs, "
           r"which is why signalling a fraction of the map recovers most of "
           r"the gap.")
    h2("5.8 Does the map have to be recomputed?")
    figure("map_transfer.png",
           r"<b>Figure 10. Reusing an exit map.</b> Solid is the transferred "
           r"map, dashed the one recomputed in place. <b>a</b>, <b>b</b>: reuse "
           r"across frames of the same sequence. <b>c</b>: reuse across quality "
           r"index; the entry is the delivered distortion against a 0.1 dB "
           r"budget.")
    par(r"How much A's extra decode at the encoder matters depends on how "
        r"often the map has to be recomputed, and we have not seen that "
        r"measured anywhere.")
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
        r"without anything in the decode reporting that it has. In our reuse "
        r"probe, then, an encoder can search once per rate and reuse that "
        r"search across the frames we tested. How far that carries beyond the "
        r"offsets and sequences probed here we have not measured.")
    h2("5.9 Complexity and wall-clock")
    tbl("latency",
        r"<b>Table 8. Wall-clock</b>, 1080p, median of 40 interleaved "
        r"iterations on one NVIDIA RTX A6000, 300 W board limit, PyTorch "
        r"2.6.0 and CUDA 12.4, single precision, at the 0.1 dB operating "
        r"point. "
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
    par(r"<b>Seconds and joules.</b> Multiply-accumulates are the right unit "
        r"to optimise, because they do not depend on the machine, but they are "
        r"not the unit a deployment cares about. We measured both on the "
        r"padded 1080p frame, sampling board power from the driver at 100 ms "
        r"and running each condition for a fixed wall time rather than a fixed "
        r"iteration count, because the sensor updates too slowly for ten "
        r"iterations of a 100 ms decode to mean anything. Idle power was "
        r"measured before and after each condition so that a neighbouring "
        r"process waking up would appear as drift rather than as a saving.")
    par(r"Energy follows time, not arithmetic. At q0, q32 and q63 the routed "
        r"decode removes 33.0, 22.9 and 17.8% of the arithmetic; it removes "
        r"29.1, 19.1 and 14.1% of the wall clock and 29.6, 19.3 and 14.2% of "
        r"the joules per frame. Board power barely moves, 298 W against 296 W "
        r"at q0, because the later groups run on a shrinking set of tiles and a "
        r"partly idle GPU still draws most of its static power. What early exit "
        r"buys is a shorter decode, not a cooler one. Frame rate at 1080p goes "
        r"from 9.0 to 11.2, and peak memory rises by 13.3% at every resolution "
        r"we could measure: tiling turns one feature map into a batch of small "
        r"ones, so the peak is set by the widest point of the ladder rather "
        r"than by its deepest.")
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
    par(r"The same gap appears in our own accounting. The router is charged "
        r"at its share of the decoder's multiply-accumulates, "
        r"\RouterCostPct% (Section 3.1). We also timed the "
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
        r"<b>Table 9. Ladder settings</b>, mean saving (%) over the five "
        r"rates, on each run's own latest checkpoint. Ceilings are 100(1-c_j) "
        r"for that ladder, corrected by the offset the hook count shows. "
        r"* some rate did not reach the budget, so the mean is over the rest. "
        r"† the saving is from the arithmetic model rather than counted off "
        r"the decode, and is therefore 0.4 to 0.8 points optimistic; only "
        r"RECIPE512 has been re-measured with hooks at every budget.")
    par(r"Section 5.4 argued that once a budget saturates a ladder, the only "
        r"way to spend more is an exit that does not exist. The table measures "
        r"that. A finer ladder (K=12, j=4) has a higher ceiling than the "
        r"shipped one's \Ceiling%, and the two orderings cross somewhere "
        r"between 0.1 and "
        r"0.3 dB. At 0.1 dB the coarse ladder wins, and the fine one cannot "
        r"even reach the budget at the highest rate. Splitting later (j=4 "
        r"against j=2) puts twice as many blocks in the per-tile section, "
        r"which by the power law of Section 4 raises the floor. At 0.5 dB the fine "
        r"ladder wins by 6.7 points, \FineHalfDb% against \CoarseHalfDb%, "
        r"because the coarse one has been pinned at its ceiling since 0.3 dB "
        r"and has nothing left to spend.")
    par(r"So the ladder is a choice made at the operating point, not a "
        r"hyperparameter tuned once and fixed. The band of Section 5.4 tells "
        r"you which side of the crossover a given budget sits on.")

    # ---- 6 limitations ---------------------------------------------------
    h1("6. Against the released decoder")
    par(r"Everything above is measured against the released DCVC-UF intra "
        r"decoder, but always as a percentage. This section states it once in "
        r"the units a codec paper reports, so the comparison can be read "
        r"without arithmetic.")
    tbl("released",
        r"<b>Table 10. FLEX-UF against the decoder it modifies.</b> The "
        r"released DCVC-UF intra decoder and the same decoder with the exit "
        r"ladder, at the three budgets. Arithmetic is counted with hooks on "
        r"the executed decode. ``dB below release'' is what the routed decode "
        r"actually delivers, which is under the budget at 0.1 dB and well "
        r"under it at 0.3 and 0.5, because by then the ladder has saturated "
        r"and the budget cannot be spent. BD-Rate is the rate a codec would "
        r"have to add to buy that quality back.")
    par(r"The released decoder spends \RelGmac GMAC on a padded 1080p frame, "
        r"which is \RelKMacPx thousand multiply-accumulates for every pixel it "
        r"produces, and it spends the same on every pixel whatever the picture "
        r"is doing there. At a 0.1 dB budget the ladder takes that to "
        r"\GmacAtOne GMAC, \KmacAtOne kMAC per pixel, and delivers "
        r"\DbAtOne dB below the release for a BD-Rate cost of \BdRateALow%.")
    par(r"The 0.3 and 0.5 dB rows are the same measurement with the "
        r"constraint loosened, and they say something the percentages hide. "
        r"Between them the saving moves by less than a point, from "
        r"\MeanAtThree% to \MeanAtFive%, while the budget grows by two "
        r"thirds. The ladder has run out of exits: at 0.3 dB most tiles are "
        r"already on the cheapest rung they are allowed and the extra "
        r"allowance buys nothing. The delivered distortion says the same, "
        r"\DbAtThree dB against a 0.3 dB allowance. A looser budget is not "
        r"the way to get more out of this decoder; a finer ladder is, and "
        r"Section 5.10 measures one.")
    par(r"In wall clock on one RTX A6000 the 0.1 dB row decodes a padded 1080p "
        r"frame in 89.6 ms against the release's 110.8, which is 9.0 frames "
        r"per second becoming 11.2, and it does so for 26.6 J against 33.0. "
        r"The arithmetic saving is 22.9% at that rate and the energy saving is "
        r"19.3%, and the gap between those two numbers is the scheduling cost "
        r"of a decode that runs its last groups on a handful of tiles.")
    h1("7. Limitations")
    par(r"<b>The seam is reduced, and the exact remedy is not usable as it "
        r"stands.</b> The halo exchange removes "
        r"\CoupFloorDropLo–\CoupFloorDropHi% of the floor and is bit-exact at "
        r"uniform depth (Section 4.4). It also destroys the routed saving, "
        r"because routing puts neighbouring tiles at different depths. We do "
        r"not know whether a decoder trained with the exchange can recover both at "
        r"once, and that is the experiment we would run next. Until someone "
        r"runs it, the floor is a cost this method pays.")
    par(r"<b>Intra frames only.</b> This is the image path of a video codec. "
        r"Extending the ladder to inter frames raises a question we do not "
        r"answer here, because an exit map propagates through the reference "
        r"chain and a shallow tile in one frame is a worse reference for the "
        r"next one.")
    par(r"<b>The checkpoint is early in its schedule.</b> Every number in "
        r"this paper is measured on one pinned checkpoint, "
        r"runs/RECIPE512/ckpt_PAPER.pth.tar, taken at the end of the first "
        r"pass over the training set. It is pinned because a moving checkpoint "
        r"was overwritten mid-measurement once, and a paper whose tables come "
        r"from three different epochs is worse than one whose tables are "
        r"early. Later checkpoints of the same run improved the measured "
        r"trade-off: on the same 53 frames at the same budget, epoch 1 reads "
        r"25.3% and epoch 2 reads 24.6% where the pinned checkpoint reads "
        r"\MeanAtOne%. Two points do not establish a trend and we do not read "
        r"them as a bound on what a converged run would give.")
    par(r"<b>A single decoder.</b> All results are on DCVC-UF's intra decoder. "
        r"Nothing in the method looks specific to it, since the ladder needs "
        r"only a residual trunk with a shared head, but we have not measured a "
        r"second decoder to check.")

    # ---- 7 conclusion ----------------------------------------------------
    h1("8. Conclusion")
    par(r"The intra decoder of DCVC-UF can be run at a depth chosen per tile "
        r"from what that tile contains. At a 0.1 dB budget this removes "
        r"\MainLowRate% to \MainHighRate% of its multiply-accumulates for a "
        r"BD-Rate cost of \BdRateALow%, with no change to the encoder and none "
        r"to the coded payload.")
    par(r"Three lessons we would expect to carry to decoders of this class, "
        r"a residual trunk behind a shared head, and "
        r"none of them is about early exit.")
    par(r"Tiling is the dominant cost and it is governed by depth. All "
        r"of that cost traces back to a single 3×3 that is 0.29% of the "
        r"arithmetic, and the penalty grows as the square of how many such "
        r"convolutions run per tile. The affected-area fraction usually "
        r"quoted alongside it predicts that penalty badly.")
    par(r"The exact remedy is in tension with the thing it enables. "
        r"Giving each convolution its real neighbour is bit-identical at "
        r"uniform depth, and under routing it destroys the allocation, "
        r"because routing is the deliberate violation of the condition that "
        r"makes it exact. We expect any method that combines spatial "
        r"adaptivity with tiled inference to walk into this.")
    par(r"A distortion budget is only a control variable inside a "
        r"measurable band. Below the floor the budget admits nothing at "
        r"all, and above saturation more of it buys nothing. A saving quoted "
        r"without saying where in that band it sits has left out the part of the "
        r"result a reader most needs.")
    par(r"We would attach three cautions to the measurements themselves. A "
        r"saving in operations is an optimistic bound on a saving in time, and "
        r"the optimism scales with the saving. A learned router should be "
        r"compared against a free one. Routing on the bits already spent per "
        r"tile needs no parameters and no training, it adds nothing to the "
        r"stream, and it beats our trained head at every rate we measured, "
        r"while agreeing with the Lagrangian oracle on fewer tiles "
        r"than the head does. We read that as a warning about the metric quite "
        r"as much as about the head. Finally, a timing harness will report "
        r"numbers whether or not it is timing the right device. Ours timed the "
        r"wrong one for months, and what gave it away was a decoder that "
        r"appeared to slow down with the quality index.")
    return F


# ------------------------------------------------------------ references
# Bibliography keys used inside generated tables, mapped to this document's own
# reference numbers. paper/main.tex resolves these through bibtex; there is no
# bibtex here, so the map is explicit and lives beside REFS. A key that is not
# here prints no citation rather than a wrong one.
CITE = {
    "arls": 11, "slimcae": 22, "slimvc": 23, "evc": 22,
    "spatialcompetition": 27, "dcvcrt": 33, "dcvcfm": 13, "dcvcuf": 14,
    "hpcm": 43, "elic": 44, "stf": 45, "tcm": 46, "mlicpp": 47, "flic": 48,
    "mambavc": 49, "weconvene": 50, "charm": 51, "dcvcdc": 52,
}

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
 "M. A. Yilmaz et al. Slimmable video codec. CVPRW, 2022.",
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
 "M. Dong, M. Lu, and Z. Ma. Accelerating block-level rate control for learned image compression. arXiv:2409.01009, 2024.",
    "Y. Li et al. Learned image compression with hierarchical progressive context modeling. ICCV, 2025.",
    "D. He, Z. Yang, W. Peng, R. Ma, H. Qin, Y. Wang. ELIC: efficient learned image compression with unevenly grouped space-channel contextual adaptive coding. CVPR, 2022.",
    "R. Zou, C. Song, Z. Zhang. The devil is in the details: window-based attention for image compression. CVPR, 2022.",
    "J. Liu, H. Sun, J. Katto. Learned image compression with mixed transformer-CNN architectures. CVPR, 2023.",
    "W. Jiang, R. Wang. MLIC++: linear complexity multi-reference entropy modeling for learned image compression. ICML Neural Compression Workshop, 2023.",
    "H. Li, S. Li, W. Dai, C. Li, J. Zou, H. Xiong. Frequency-aware transformer for learned image compression. ICLR, 2024.",
    "S. Qin et al. MambaVC: learned visual compression with selective state spaces. arXiv:2405.15413, 2024.",
    "H. Fu, J. Liang, Z. Fang, J. Han, F. Liang, G. Zhang. WeConvene: learned image compression with wavelet-domain convolution and entropy model. ECCV, 2024.",
    "D. Minnen, S. Singh. Channel-wise autoregressive entropy models for learned image compression. ICIP, 2020.",
    "J. Li, B. Li, Y. Lu. Neural video compression with diverse contexts. CVPR, 2023.",
]


def build(out="paper/FLEX-UF.pdf"):
    PW, PH = letter
    M, GAP = 0.62 * inch, 0.28 * inch
    colw = (PW - 2 * M - GAP) / 2
    # Title block plus the full-width teaser, CVPR style. Everything after
    # page 1 is two columns, which needs an explicit NextPageTemplate -- without
    # it reportlab keeps using the first template and every page gets a
    # full-width band across the top.
    # Title block plus the baseline figure and its caption. Sized to what it
    # holds: a band taller than its content leaves both columns of page 1 short,
    # and a wide figure placed in the flow instead would force a page break and
    # empty a column outright.
    top_banner = 4.05 * inch

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

    # Sized to the one wide figure the paper carries, plus its caption. A band
    # taller than its content leaves the two columns beneath it short, which is
    # the defect this template was added to avoid.
    wide_band = WIDE_BAND
    f_wide = Frame(M, PH - 0.7 * inch - wide_band, PW - 2 * M, wide_band,
                   id="wide", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
    hw = H - wide_band
    f_lw = Frame(M, 0.7 * inch, colw, hw, id="lw", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    f_rw = Frame(M + colw + GAP, 0.7 * inch, colw, hw, id="rw", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)

    # The supplement's title block gets its own band. Reusing the wide-figure
    # template put a 2.62 inch band above a 1.1 inch title and left the rest of
    # the strip empty, a field of white between "Supplementary Material" and
    # section A.
    supp_band = 1.16 * inch
    f_supp = Frame(M, PH - 0.7 * inch - supp_band, PW - 2 * M, supp_band,
                   id="suppband", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
    hs = H - supp_band
    f_ls = Frame(M, 0.7 * inch, colw, hs, id="ls", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    f_rs = Frame(M + colw + GAP, 0.7 * inch, colw, hs, id="rs", leftPadding=0,
                 rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[f_ban, f_l1, f_r1], onPage=num),
        PageTemplate(id="rest", frames=[f_l, f_r], onPage=num),
        PageTemplate(id="wide", frames=[f_wide, f_lw, f_rw], onPage=num),
        PageTemplate(id="supp", frames=[f_supp, f_ls, f_rs], onPage=num)])

    banner_cap = ("<b>Figure 1. The decoder we modify.</b> Figure 3 of "
                  "DCVC-UF [14], reproduced. Everything up to the "
                  "reconstruction stays frozen in this work: the patch "
                  "embedding, the chunk encoder, the entropy model and the "
                  "coded payload. FLEX-UF replaces the frame-specific decoders "
                  "on the right with a ladder of exits taken per tile.")
    story = [
        Paragraph("Where to Stop:<br/>Tile-Adaptive Early Exit in a Learned "
                  "Image Decoder", TITLE),
        Paragraph("Anonymous CVPR submission &nbsp;&nbsp;·&nbsp;&nbsp; Paper ID "
                  "****", AUTH),
    ]
    story += fig("dcvcuf_framework.png", PW - 2 * M, banner_cap,
                 maxh=2.45 * inch)
    story += [NextPageTemplate("rest"), FrameBreak()]
    story += content(colw, PW - 2 * M)
    story.append(Paragraph("References", H1))
    for i, r in enumerate(REFS, 1):
        story.append(Paragraph(f"[{i}] {r}",
                               S("ref", fontSize=7.2, leading=8.4, spaceAfter=2)))

    # The supplement is its own document again. It was folded in here when the
    # target was one CVPR PDF; the author now wants it separate, and
    # scripts/build_supp_pdf.py has always been able to build it alone.
    doc.build(story)
    if UNEXPANDED:
        print("  UNEXPANDED MACROS (run make_paper_tables.py): "
              + ", ".join(sorted(UNEXPANDED)))
    print(f"  -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "paper/FLEX-UF.pdf")
