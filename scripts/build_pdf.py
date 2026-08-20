"""Render the paper to PDF without a LaTeX installation.

There is no TeX on this machine and installing a distribution on a shared box to
produce one document is not a reasonable trade. `paper/main.tex` remains the
submission source (it is what goes to Overleaf and to the CVPR kit) and this
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
    # An unexpanded \Macro means make_paper_tables did not emit it, usually
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
    size is the body size: mathtext points are typographic points and the PNG
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
    and the table stood 599.6pt tall in a 691.2pt frame, tall enough that it
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
        # \cite{key} has no meaning here (there is no bibtex pass) and the
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
    inserted in the middle, which happened twice tonight.
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
        back. It forces a page break, which is what LaTeX would do with a
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

    def rows_tbl(rows, first_col=0.17):
        # main.tex sets this as a tabular inside a center environment, which
        # LaTeX will not break across a column. Keep it whole here too, or the
        # notation table loses its header on the far side of the break.
        A(KeepTogether(simple_table(rows, colw, first_col)))

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
        # across the column the way a longtable does; repeatRows=1 on the
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
        centred on the column minus the number, and every equation sat 10.8pt
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
        r"Learned image decoders spend the same computation on every region of "
        r"a frame, whatever that region contains. We show that decoding compute "
        r"can instead be allocated spatially, by content, with the encoder, the "
        r"entropy model and the transmitted representation left untouched. We "
        r"add a multi-exit ladder to the intra decoder of DCVC-UF, so that "
        r"tiles of one frame leave a shared reconstruction trunk at different "
        r"depths. On \NumSeq CTC intra frames, one from each sequence, under a "
        r"0.1 dB quality constraint "
        r"this removes \MainLowRate% to \MainHighRate% of decoder "
        r"multiply-accumulates across the rate range, \MeanAtOne% on average, "
        r"at a BD-Rate cost of \BdRateALow%. At 0.2 dB the mean saving is "
        r"\MeanAtTwo% and at 0.3 dB it is \MeanAtThree%, within 0.9 points of "
        r"the ladder's \Ceiling% architectural ceiling."
        r"<br/><br/>"
        r"Adaptive decoding works only inside a finite window. Tiling sets a "
        r"rate-dependent quality floor below which no allocation is feasible, "
        r"and a saturation point above which every tile already sits on the "
        r"cheapest path. Rescaling the budget between those two limits "
        r"collapses five rate curves that differ by \BandRawSpread points onto "
        r"one within \BandSpreadMean. Finally, the bits the entropy model has "
        r"already spent on a tile predict how deep that tile must decode "
        r"better than a trained \RouterParams-parameter router does, by up to "
        r"\RateRankBeatsBy points, with no parameters and nothing added to the "
        r"file. It wins while agreeing with the oracle's choice on fewer tiles "
        r"than the head does, which suggests that exit-label accuracy is the "
        r"wrong objective to train a router on and that the ordering is what a "
        r"router has to get right. On this decoder, the bits already spent on "
        r"a region say how much computation reconstructing it needs."),
        ABST))
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
        r"around them is by now mature [18, 54, 55]. Super-resolution adopted "
        r"the same idea "
        r"spatially: ClassSR [12] routes image patches to networks of "
        r"different capacity by difficulty, and APE [1] exits patches at "
        r"different depths of one network. Learned compression has taken a "
        r"different route. Slimmable autoencoders [22, 23] give one model "
        r"several complexity levels, but the level is chosen <i>per stream</i> "
        r"rather than per region, and switching it changes the bitstream.")
    par(r"We ask the spatial-adaptivity question inside a learned decoder, "
        r"under a constraint that makes the answer deployable: <b>the encoder "
        r"is frozen and the coded payload is unchanged.</b> That rules out "
        r"re-training the analysis transform, changing the entropy model, or "
        r"altering the latent, and leaves one place to spend adaptivity, the "
        r"synthesis transform.")
    par(r"Our method, FLEX-UF, attaches exits at intervals through the twelve "
        r"residual blocks of the DCVC-UF intra decoder; we call that ordered "
        r"set of exits the <i>ladder</i>. The first j blocks run over the "
        r"whole frame. The rest run per tile, over a set of tiles that shrinks "
        r"with depth, and the shrinkage is where the saving comes from. Each "
        r"exit hands its feature to the shared head through a small pointwise "
        r"adapter, zero-initialised so that at step zero the deepest exit "
        r"reproduces the released decoder bit-exactly. Getting this to work "
        r"depended less on the ladder than on two problems that have little to "
        r"do with early exit as it is usually studied.")
    h2("Tiling has a large price.")
    par(r"Spatial adaptivity needs tiles, and once a frame has been cut into "
        r"tiles, every 3×3 convolution at a tile border reads invented values. "
        r"The tile-boundary artefact that follows is what we call the "
        r"<i>seam</i>, and under the default padding rule it costs "
        r"\SeamZerosHigh dB on this decoder at high rate. That is several times the entire "
        r"budget the method works to, and it is paid before any tile has saved "
        r"a single operation. Section 4 measures what the penalty depends on, "
        r"which is not the affected area, and what removes it.")
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
    h2("The free baseline wins, and what that says about routers.")
    par(r"The decoder is already holding a number that says how hard each tile "
        r"was: the bits the entropy model spent on that tile's latents, "
        r"available before the trunk starts and at no cost. A rule calibrated "
        r"on that number, with no learned parameters in it, saves more decoder "
        r"arithmetic than our \RouterParams-parameter head at every one of the "
        r"\RateRankNWins rates we measure, by up to \RateRankBeatsBy points, "
        r"and it charges the decoder nothing where the head charges "
        r"\RouterCostPct% of the decode (Section 6.2). The part we did not "
        r"expect is that it wins while agreeing with the Lagrangian oracle's "
        r"chosen exit on <i>fewer</i> tiles than the head does. Agreement over "
        r"exit labels weighs a tile whose two best exits are within a hair of "
        r"each other exactly as heavily as one that carries most of the "
        r"frame's error, and most tiles are the first kind. What a router has "
        r"to get right is the order in which tiles want depth, and label "
        r"accuracy is a poor stand-in for it.")
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
        r"which no allocation is feasible and a saturation point above which "
        r"none improves, both in closed form. Measuring the budget in units of "
        r"that band accounts for most of the rate dependence, on two "
        r"independently trained checkpoints.")
    par(r"<b>(iii)</b> A zero-learned-parameter control that adaptive-inference work is "
        r"rarely measured against, and which we report as a result rather than "
        r"as a baseline. Routing on the bits the entropy model has "
        r"already spent per tile costs nothing, adds nothing to the stream, and "
        r"beats our trained head at every rate, while agreeing with the oracle "
        r"on fewer tiles than the head does. The quantity that decides a "
        r"router is the ranking it induces over tiles, and exit-label accuracy "
        r"tracks it poorly.")

    # ---- 2 related work --------------------------------------------------
    h1("2. Related work")
    h2("Early exit.")
    par(r"BranchyNet [20] and MSDNet [3] established the pattern of "
        r"intermediate classifiers with a confidence rule; SDN [15] framed it "
        r"as mitigating overthinking, and ZTW [55] recycles earlier "
        r"predictions. We train all exits jointly, after "
        r"Scardapane et al. [18], and distil between <i>adjacent</i> exits as "
        r"in Phuong and Lampert [17], since too large a student-teacher gap "
        r"hurts the shallowest exits [19]. All of this work exits on a "
        r"confidence signal computed from the network's own output. A decoder "
        r"has no such signal: there is no class posterior, and the quantity "
        r"that would decide is the error against a source the decoder cannot "
        r"see (Section 3.3).")
    h2("Spatially adaptive inference.")
    par(r"ClassSR [12] sorts super-resolution patches into easy/medium/hard "
        r"and runs a different network on each; APE [1] exits patches at "
        r"different depths; Glance-and-Focus [8] spends resolution adaptively. "
        r"We share the per-region premise with all three, and we inherit the "
        r"same tiling problem, though the cost of tile borders is rarely "
        r"quantified in that line of work. The closest prior work on the "
        r"border itself is the per-channel AR(1) padding of Kaseva et al. "
        r"[11], which we implement and evaluate.")
    tbl("positioning",
        r"<b>Table 1. Where this sits.</b> What each method varies, at what "
        r"granularity, and what that costs the stream. A single new-bitstream "
        r"column cannot represent our own two modes, so the last three ask "
        r"what actually changes: whether the coded latent is new, what "
        r"side data travels with it, and whether a decoder alone can do it.")
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
    par(r"Table 1 places this work against the methods above; Tables 2 and 3 "
        r"say where the field spends its decoder "
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
        r"of one model, per stream and by changing the encoder. DCVC-FM [13] "
        r"and DCVC-UF [14] reduce cost architecturally, for every frame "
        r"equally. Rate–distortion–complexity has since become an explicit "
        r"third axis [35, 36], and Zhang and Gao [37] route <i>whole "
        r"frames</i> to one of several jointly-trained coding paths. Every one "
        r"of these changes the model, and with it what the encoder emits. Our "
        r"axis is orthogonal: the model and the bitstream are both fixed, and "
        r"only <i>how much of the decoder runs where</i> varies.")
    h2("The closest neighbour.")
    par(r"Blard et al. [27] also partition an image into regions, also choose "
        r"per region by a rate–distortion cost computed at the encoder, and "
        r"also transmit a mode map. Their regions choose among several "
        r"<i>separately trained</i> codecs, so the decoder must hold all of "
        r"them and its complexity is that of one codec regardless of the "
        r"choice; the map buys rate, not compute. Ours choose a prefix length "
        r"of a single trunk, so the weights are shared by construction and the "
        r"map buys compute at fixed rate. Because their alternatives are "
        r"unrelated networks, no decoder-side predictor could stand in for the "
        r"encoder's search. Our exits are prefixes of one another, and that "
        r"nesting is what lets a decoder-side predictor stand in at all "
        r"(Section 3.1).")
    h2("Signalling versus prediction.")
    par(r"Being <i>told</i> a mode decision rather than inferring it is the "
        r"norm in standardised video coding. HEVC [9] and VVC [21] transmit "
        r"partitioning, prediction mode and transform tree. We evaluate both, "
        r"and treat the signalled variant as the conventional design "
        r"(Section 3.1).")
    h2("Allocating a budget over units.")
    par(r"The construction we use is not new and we do not present it as such. "
        r"Shoham and Gersho [28] showed that for a finite set of per-unit "
        r"operating points, a Lagrangian sweep decouples the allocation across "
        r"units and traces exactly the lower convex hull of the achievable "
        r"set; Ortega and Ramchandran [29] made it standard practice in image "
        r"and video coding. What we add is the structure this particular "
        r"operating set has: a floor below which no allocation is feasible and "
        r"a saturation point above which none improves (Section 5.4). The same "
        r"relaxation has resurfaced for test-time compute in language models "
        r"[30], with per-instance decoupling and a binary search on the "
        r"multiplier. That is the identical structure in a domain with no rate "
        r"axis, and it suggests that the floor and saturation "
        r"characterisation is worth stating beyond this decoder.")
    h2("How much is there to gain?")
    par(r"Bounding what adaptive inference could achieve is itself a line of "
        r"work. Hasan et al. [34] derive an oracle bound on efficiency at "
        r"fixed accuracy and report 43–121× on ImageNet. Their bound has a "
        r"ceiling and no floor, because the largest model in their family "
        r"reaches the reference accuracy by definition. Spatial adaptivity "
        r"introduces a floor: cutting a frame into tiles costs quality even "
        r"when every tile runs to full depth, so the reference is unreachable "
        r"at <i>any</i> compute and the feasible set of budgets is an interval "
        r"rather than a ray. Section 5.4 characterises that interval and both "
        r"of its ends.")
    h2("Tile boundaries.")
    par(r"Every method that processes an image in independently-computed tiles "
        r"meets the same artefact. The remedies in the literature are overlap "
        r"and averaging, local padding from neighbouring patches [31], "
        r"training with overlaps [32], and fitted extrapolation [11]. Local "
        r"padding is the closest to the exact remedy we measure in "
        r"Section 4.2, and the difference is accounting: it pads every "
        r"convolutional layer and does not report the cost. To our knowledge "
        r"the estimator view of the padding rule and the dependence of the "
        r"penalty on per-tile depth (Section 4) have not been reported.")
    h2("Where the time goes.")
    par(r"DCVC-RT [33] argues that operational rather than computational "
        r"complexity is the speed bottleneck for neural codecs. "
        r"Section 6.3 is an instance of that claim inside one loop: removing "
        r"\WallPredicted% of the operations buys \WallSorted% of the time, and "
        r"\WallSortedGain points of the difference come back from reordering "
        r"the loop with the arithmetic untouched.")

    # ---- 3 method --------------------------------------------------------
    h1("3. A tile-adaptive decoder on a frozen latent")
    h2("3.1 Setting and notation")
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
        r"own. Tiles are indexed by t and there are N of them, N=40 at 1080p, "
        r"and what they lose at a given depth is not the same from one to the "
        r"next (Figure 2). "
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
        r"frame lands on its budget as cheaply as possible. Rate is set by a "
        r"quality index running from q0, the lowest rate, to q63, the highest, "
        r"and we report five of them: q0, q16, q32, q48 and q63.")
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
    figure("tiles_unequal.png",
           r"<b>Figure 2. Why depth should not be uniform.</b> One 1080p "
           r"frame, 40 tiles. <b>a</b>, what each tile loses if it stops at "
           r"the shallowest exit the ladder allows: several lose nothing "
           r"measurable and the worst loses several times the budget, so a "
           r"single depth for the frame has to be set by the worst of them. "
           r"<b>b</b>, the bits the entropy model has already spent on a tile "
           r"against the exit the Lagrangian oracle sends it to, at the "
           r"0.1 dB budget. The signal a decoder-side router needs is already "
           r"in the file.")
    h2("Two deployment modes.")
    par(r"Both choose the same object, one exit per tile, on the same weights "
        r"and the same coded payload, and they differ only in where that "
        r"choice is made. <i>Signalled</i> FLEX-UF, configuration <b>A</b>, "
        r"decides at the encoder. The encoder holds the source, so it knows "
        r"D(t,k), picks each tile's exit by the Lagrangian of Section 3.3, and "
        r"transmits the resulting <i>exit map</i>, one exit index per tile, "
        r"which is the analogue of the mode map a standard codec signals. It "
        r"entropy-codes to ~\MapBits bits per 1080p frame, or \MapOverheadLow% "
        r"of a typical bitrate; the coded latent is unchanged. It costs the "
        r"encoder about one extra decode per frame and the decoder nothing. "
        r"<i>Bitstream-identical</i> FLEX-UF, configuration <b>B</b>, decides "
        r"at the decoder. A \RouterParams-parameter head reads decoded data "
        r"only and predicts the same map (Section 3.3). Nothing is transmitted "
        r"and the file stays byte for byte the one the released encoder "
        r"produced. It costs the decoder \RouterCostPct% of its own "
        r"multiply-accumulates, which is charged inside every B number we "
        r"report. Every saving quoted in this paper is labelled with the mode "
        r"it was measured in. Because the head reads nothing the encoder "
        r"cannot also read, the encoder can run it too and so knows tile by "
        r"tile where it will be wrong; supplement G measures the interpolation "
        r"between the two modes that this allows.")
    h2("3.2 The ladder, its cost and its ceiling")
    par(r"The DCVC-UF intra decoder is one upsampling block, twelve "
        r"DepthConvBlocks and a head, costing \IntraGmac GMAC per 1080p frame. The "
        r"twelve blocks are 89.4% of that, which is why we build the ladder "
        r"across them. Inside one block at C=384 channels, the only operator "
        r"with any spatial extent is a 3×3 depthwise convolution, and it costs "
        r"9C against the block's 8C²+9C. That is <b>0.29%,</b> and it is the "
        r"fraction the rest of this paper turns on. Tile borders damage the "
        r"picture through it, it is what makes the exact remedy of Section 4.2 "
        r"affordable at all, and it is why the exit adapters are pointwise: a "
        r"3×3 inside an adapter would add seam damage at exactly the tiles "
        r"that took an early exit. Supplement A gives every adapter its shape "
        r"and measures what each is worth.")
    par(r"Two consequences follow from the ladder, and both are easy to state "
        r"wrongly. Exits shallower than j are indistinguishable from each "
        r"other, because the first j groups run for every tile regardless, and "
        r"the usable ladder therefore has K−j distinct costs. The second "
        r"consequence is the <i>architectural ceiling</i>")
    eq(r"S_{\mathrm{max}} = 100\,(1 - c_{j})\ \%")
    par(r"reached when every tile takes exit j. For our shipped setting (K=6, "
        r"j=2, 256 px tiles) S_max = \Ceiling%. We show in Section 5.4 that "
        r"this bound is <i>reached</i> in practice at low rate, so it behaves "
        r"as an operating point and not as an asymptote.")
    h2("3.3 Allocation: encoder search, decoder prediction")
    par(r"Every tile has to be given an exit, drawn from the usable ones, "
        r"k ≥ j, and we want the set of choices that keeps the frame inside its "
        r"distortion budget for the least compute. There are K−j choices per tile "
        r"and N tiles, so searching assignments one by one is out of the "
        r"question. What makes the problem tractable is that the tiles do not "
        r"interact: each contributes its own error D(t,k) and its own cost "
        r"c_k, and the only thing they share is the single budget the frame "
        r"has to meet. Problems of that shape are solved with a Lagrangian. "
        r"Put a price λ on compute, add λ c_k to the error of exit k, and the "
        r"budget constraint drops out; what is left is N separate one-tile "
        r"problems, each a minimum over K−j numbers. Compute falls and error "
        r"grows as λ grows, neither of them turning back, so one value of λ "
        r"puts the frame exactly on its budget and we find that value by "
        r"bisection. The rule each tile follows is")
    eq(r"k^{*}(t) = \mathrm{arg\,min}_{k}\ \left[\, D(t,k) + \lambda\, c_{k} \,\right]")
    par(r"with the minimum taken over k ≥ j. We call k*(t) the choice of the "
        r"<i>Lagrangian oracle</i>: it is made with the true error of every "
        r"exit in hand, which no decoder has, but it is optimal only within the "
        r"family of allocations one multiplier can reach. That qualifier is not "
        r"cosmetic. Supplement G exhibits allocations that are cheaper at the "
        r"same distortion than anything this rule produces, so <i>oracle</i> "
        r"here means best under a single multiplier and not the global "
        r"constrained optimum. Where we mean the latter we say so. Both "
        r"quantities are available <i>to the encoder</i>, which holds the "
        r"source, so configuration A can run this search exactly; supplement G "
        r"prices the two ways of building the table it searches.")
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
    par(r"The first term is the head's score for exit k, on a log scale, which "
        r"is the currency the surprisal −log p puts it in. The "
        r"second is the price on compute that λ carries in the Lagrangian: "
        r"raising β takes more away from the deep exits than from the shallow "
        r"ones, so more tiles are handed to cheap exits and the frame again "
        r"gets cheaper and worse. Supplement F gives the head's architecture, "
        r"its training objective and what each of its inputs is worth.")
    h2("Where β comes from at deployment.")
    par(r"A real decoder cannot bisect β against the budget. It would need to "
        r"know the delivered distortion, and the distortion is measured "
        r"against a source the decoder never receives, which is the same "
        r"missing variable that stopped it running the Lagrangian in the first "
        r"place. So β is calibrated offline, once, on a held-out set, and "
        r"shipped as a table indexed by the quality index and the budget: the "
        r"decoder reads the quality index out of the bitstream, looks β up, "
        r"and runs the argmax above with nothing in the loop needing the "
        r"source. We calibrate on \HeldNCal held-out Open Images validation "
        r"frames, disjoint from the training images and from the test "
        r"sequences, bisecting once per quality index and applying the table "
        r"to the test set unchanged. Section 6.1 reports both that number and "
        r"the one obtained by bisecting on the test set itself, and "
        r"supplement F.5 gives the measurement of whether the table transfers, "
        r"because the difference between the two is the only honest reading of "
        r"that.")

    # ---- 4 the seam ------------------------------------------------------
    h1("4. Tiling sets a floor")
    h2("4.1 The seam, and how it scales with depth")
    figure("seam_problem.png",
           r"<b>Figure 3. The artefact, before anything is done about "
           r"it.</b> Bosphorus at 1080p, quality index 63, zero padding at "
           r"the tile border. <b>a</b>, the frame decoded in one piece. "
           r"<b>b</b>, the same bitstream and weights decoded in 256 pixel "
           r"tiles, every tile at full depth: the difference between the "
           r"two, amplified 30 times. <b>c</b>, the boxed region of each.")
    par(r"A 3×3 depthwise at feature position (x,y) computes a weighted sum "
        r"over its neighbours. Decoded full frame, those neighbours exist. "
        r"Decoded per tile they do not, and the kernel is handed whatever the "
        r"padding rule invents. Each convolution extends the affected "
        r"region by one ring, so with b per-tile blocks on a tile of side F "
        r"the fraction of the tile within reach of an invented value is "
        r"1 − ((F−2b)/F)². At our shipped F=32, b=8 that comes to 0.750. "
        r"Three quarters of the tile is affected, so what we are looking at "
        r"is a structured error over most of the tile and not a thin border "
        r"at its edge.")
    par(r"That fraction tells us which pixels are affected and not how badly, "
        r"and it turns out to be the wrong predictor of the penalty. We swept "
        r"the split depth, which sweeps b from 12 to 0 with b=0 as an exact "
        r"zero-seam control, and measured")
    eq(r"\mathrm{seam} \;\propto\; b^{\alpha}, \qquad \alpha \approx 2")
    par(r"with α = 2.38 at q0 falling to 1.93 at q63. Fitted with one free "
        r"scale, the area fraction is wrong "
        r"by 160–264% where the power law is wrong by 12–22%. The exponent "
        r"has a reading. The count of affected pixels grows like perimeter "
        r"times depth, and the error accumulated in each of them grows with "
        r"how many convolutions reached it. Once every pixel has been touched "
        r"the area law saturates and the penalty does not, which is where the "
        r"area law fails as a predictor.")
    par(r"Border padding is an <i>estimator</i> of the unseen neighbour, and "
        r"the seam is its error. Supplement A measures four of them with early "
        r"exit switched off, and the ordering is not the obvious one: a "
        r"higher-order estimator is worse, because extrapolating the local "
        r"gradient past a boundary amplifies whatever noise sits on that "
        r"boundary, and the best of the four, the per-channel AR(1) fit of "
        r"[11], wins by about 0.02 dB for 10.7% of decode wall-clock, which "
        r"against a 0.1 dB budget does not close.")
    h2("4.2 What removes it, and what does not")
    par(r"The largest factor of all is the easiest one to miss. On the "
        r"untrained ladder, replicate padding leaves \SeamReplHigh dB at q63; "
        r"after training, the same setting leaves \FloorHigh dB. More "
        r"than two thirds of what the estimator could not fix is absorbed by "
        r"weights learning to live with it, and that single change removes "
        r"more of the seam than any module in this paper does. Tile size is "
        r"the other free variable: a tiled decode's MAC count does not depend "
        r"on it at all, the damage scales with the tile perimeter as the power "
        r"law above predicts, and what tile size costs is routing granularity. "
        r"Supplement A measures both, and supplement H reports what happened "
        r"when we made the tile size track the frame resolution.")
    par(r"Two modules that ought to have removed the rest did not. The first "
        r"is a learned deblocking filter, gated by position within a tile and "
        r"applied once to the stitched frame for 0.95% of the decode; "
        r"splitting the per-pixel error by distance from the nearest tile "
        r"boundary shows it winning on the ring nearest a boundary and losing "
        r"everywhere else, and even a perfect gate could not earn its cost. "
        r"That measurement rejects the filter's premise rather than the "
        r"filter, which is in the weights every number here was measured on "
        r"and is why our deepest exit costs 1.0095 of a released decode; its "
        r"cost is charged against every saving we report rather than netted "
        r"out. The second is the exact fix. Give the 3×3 its real neighbours "
        r"across the tile border, a <i>halo exchange</i> narrowed to the one "
        r"operator that needs it, and a tiled decode at uniform depth becomes "
        r"bit-identical to a full-frame one for +0.032% of the decode. "
        r"Switched on at inference it removes \CoupFloorDropLo–\CoupFloorDropHi% "
        r"of the floor, and it destroys the routed allocation: at the same "
        r"0.1 dB budget the saving falls from \CoupPaddedHigh% to "
        r"\CoupCoupledHigh% at q63. The exchange is exact only when every tile "
        r"is at the <i>same</i> depth, and routing is the deliberate violation "
        r"of that condition, so a shallow tile ends up reading a deep "
        r"neighbour's activation, an arrangement the weights have never seen. "
        r"Supplement H carries both measurements in full.")

    # ---- 5 experiments ---------------------------------------------------
    h1("5. The band, and what a budget buys")
    h2("5.1 Protocol")
    par(r"We fine-tune the decoder on 512×512 crops from OpenImages with the "
        r"encoder frozen (max|Δ| = 0 is asserted every run). One λ_rd is drawn "
        r"per sample from a log-spaced range covering all 64 quality indices, "
        r"so a single set of weights covers the whole rate range. We evaluate "
        r"on \NumSeq intra frames, one from each sequence of the common "
        r"test set (CTC: UVG, MCL-JCV and HEVC classes B, C, D and E).")
    par(r"One detail of the protocol moves the numbers enough to state on its "
        r"own. The per-tile table D(t,k) has to be built on the "
        r"<i>deployed</i> decode path, one tiled decode per exit, "
        r"rather than by tapping the exits of a single full-frame forward "
        r"pass. Tapping is the natural implementation and it is correct for "
        r"training, but with a full-frame reference on the other side of the "
        r"ratio the tiling penalty cancels and the quality reported is that "
        r"of a decoder nobody ships. On our model the gap is +0.035 dB at the "
        r"deepest exit and +0.008 dB at the shallowest, and it grows with how "
        r"many blocks ran per tile, so we cannot correct for it after the "
        r"fact. The remaining conventions are fixed below; each of them is a "
        r"choice this project has at some point made both ways in different "
        r"files, and the two decibel conventions differ by a substantial "
        r"fraction of the working budget. Every number in this paper is in the "
        r"convention stated, and the alternatives are measured in supplement E "
        r"as a sensitivity study.")
    rows_tbl([
        ["what is fixed", "what every number in this paper uses"],
        ["reference", r"the released decoder's full-frame decode of the same "
                      r"latent; our side is the deployed tiled decode, so the "
                      r"tiling penalty sits inside every dB"],
        ["distortion pooling", r"one mean squared error per frame, converted "
                               r"to dB, then averaged over the \NumSeq "
                               r"frames; this is what the released codec's own "
                               r"evaluation reports"],
        ["scope of λ and β", r"one value per quality index and budget, never "
                             r"per frame and never per sequence"],
        ["the budget", r"applied globally: the multiplier is bisected once so "
                       r"that the whole test set lands on the budget, which is "
                       r"what a deployment would do"],
        ["where β comes from", r"offline, on the held-out calibration frames, "
                               r"for a deployable decoder; bisected on the "
                               r"test frames for the A-against-B comparison of "
                               r"Section 6.1, which supplement F.5 then "
                               r"corrects"],
        ["the mixed map", r"re-decoded once, and the distortion that decode "
                          r"delivers is what we report; the per-exit table is "
                          r"used inside the search only"],
        ["saving", r"counted with hooks on the executed pass, as a fraction of "
                   r"one released full-frame decode"],
    ], first_col=0.26)
    par(r"The last row is worth one more sentence, because the denominator is "
        r"the easiest of these to take for granted. Our own deepest exit costs "
        r"1.0095 of a released decode, and dividing by that instead would "
        r"flatter every result in the paper by 0.6–0.8 points.")
    h2("5.2 What a 0.1 dB budget buys")
    tbl("main_results",
        r"<b>Table 4. Decoder MACs saved</b> (%) against the released decoder, "
        r"per quality index and distortion budget, signalled FLEX-UF. The 0.3 "
        r"and 0.5 dB rows sit on the architectural ceiling almost everywhere; "
        r"past that point a looser budget buys nothing.")
    par(r"One word on the budget before the numbers. A tenth of a decibel is "
        r"an engineering convention, chosen because it is small against the "
        r"spacing of the rate points and because codec work has long used "
        r"differences of this size as a working tolerance. It is not evidence "
        r"that the difference is invisible, and we make no perceptual claim "
        r"for it. The 0.2, 0.3 and 0.5 dB rows are there so a reader who "
        r"disagrees with the choice can read off another one.")
    par(r"The table carries the headline numbers. At 0.1 dB the method saves "
        r"\MainLowRate% at the lowest rate and \MainHighRate% at the highest, "
        r"and \MainMean% on average. We do not read the fall with rate as an "
        r"artefact of the ladder. High-rate reconstructions carry detail the "
        r"shallow exits cannot reproduce, and the floor rises with rate as "
        r"well; both effects push the same way. The distribution behind those "
        r"means is wide, and supplement D reports it per sequence: at q0 the "
        r"median sequence saves well above the mean while the worst saves "
        r"nothing at all, and the worst cases are the two-tile sequences, "
        r"where there is almost no allocation left to make. Saving tracks tile "
        r"count more closely than it tracks content, which supplement D "
        r"measures class by class.")
    h2("5.3 Adaptivity against its controls")
    figure("exit_map.png",
           r"<b>Figure 4. Where the decoder spends.</b> Bosphorus at q32, a "
           r"0.1 dB budget. (a) the assignment overlaid on the frame, (b) the "
           r"exit index per tile, (c) the quality each tile gives up, against "
           r"its <i>own</i> full-depth reference. Water and sky leave at the "
           r"shallowest exit; the boat and the shoreline run deep.")
    tbl("static_main",
        r"<b>Table 5. Adaptivity against three controls</b> at the 0.1 dB "
        r"budget. Each cell is the delivered dB over the MACs saved (%). "
        r"† marks a uniform depth whose distortion exceeds the budget, so it "
        r"is not an admissible allocation at all. ``Random'' draws each "
        r"frame's map from the oracle's own exit histogram and shuffles it "
        r"across tiles; ``bit ranking'' keeps that histogram and orders it by "
        r"the bits the entropy model spent per tile. Both are matched to the "
        r"oracle on compute rather than on quality, which is why they carry no "
        r"dagger.")
    par(r"The obvious alternative to routing tiles is to run every tile at the "
        r"same shallower exit and accept the loss. The table puts that option, "
        r"together with two stronger controls, at matched compute.")
    par(r"The uniform rows answer the question directly, and the answer "
        r"sharpens with rate. At the lowest rate a static decoder can reach "
        r"exit 3 inside the budget and save \UniformThreeLow%, against the "
        r"Lagrangian oracle's \MainLowRate%. At q48 and q63 the only uniform "
        r"depth that fits is the deepest one, which <i>costs</i> "
        r"\DeepestUniformCost% instead of saving anything, so at those two "
        r"rates the comparison is against nothing at all rather than against "
        r"something smaller. Averaged "
        r"over rates, the best static allocation saves \BestStaticMean% where "
        r"the adaptive one saves \MainMean%.")
    par(r"The two shuffled rows separate effects that are easy to conflate. "
        r"Keeping the oracle's own exit histogram leaves the mix of depths and "
        r"hence the average cost unchanged; assigning it to tiles at random "
        r"makes quality fall sharply. What the allocation buys is knowing "
        r"<i>which</i> tiles can afford to run shallower, and running some "
        r"fraction of them shallower is worth nothing by itself. The "
        r"oracle-histogram bit ranking keeps the same histogram and orders it "
        r"by a signal the decoder already holds, the bits the entropy model "
        r"spent on each tile, which is the cheapest router we can think of and "
        r"the natural analogue of the confidence rules early-exit classifiers "
        r"use [3, 20]. It recovers most of the gap: at q63, random costs "
        r"\RandomDb dB and the Lagrangian oracle \OracleDb dB at identical "
        r"compute, while the bit ranking costs \RateRankDb dB, which is "
        r"\RateRankRecovers% of the oracle's advantage over chance. A learned "
        r"router therefore has to beat a free signal that is already three "
        r"quarters of the way there, and we would report that control in "
        r"adaptive-inference work of this kind. Given the oracle's histogram, "
        r"what this measures is <i>ranking</i> and nothing else; Section 6.2 "
        r"removes the crutch and turns the same signal into a complete routing "
        r"rule.")
    h2("5.4 Floor, saturation, and the band")
    tbl("operating",
        r"<b>Table 6. Floor and saturation</b>, dB below the released decoder. "
        r"The floor is what tiling costs with no early exit at all; saturation "
        r"is where every tile reaches exit j and the ceiling is attained.")
    par(r"One assumption is worth naming before any of this. The argument "
        r"treats the frame's distortion as a sum over tiles, while the "
        r"deblocking pass runs on the stitched frame and therefore sees the "
        r"whole exit map at once, so the tiles are not strictly independent. "
        r"We measured the size of that coupling over 60 allocations and the "
        r"largest mismatch between the separable prediction and a real decode "
        r"is 5.5×10<super>-5</super> dB, four orders of magnitude below the "
        r"budget. We proceed as if the objective were separable and treat that "
        r"as measured rather than assumed. Supplement C states the structure "
        r"of the allocation as propositions, each checked numerically rather "
        r"than asserted, and prices what the convex-hull restriction costs "
        r"against the exact Pareto set.")
    par(r"Two numbers bound what any budget can do. The <b>floor</b> is the "
        r"distortion of a tiled decode with every tile at full depth, which is "
        r"pure tiling penalty: \FloorLow dB at q0, rising to \FloorHigh dB at "
        r"q63. A budget below it admits no allocation. The <b>saturation</b> "
        r"point is the distortion when every tile takes exit j; any budget at "
        r"or above it reaches the ceiling, and a larger one reaches nothing "
        r"more. At q0 the ceiling is reached at \SatLow dB, so the "
        r"architectural bound behaves as an operating point rather than as an "
        r"asymptote, and past it the limiting factor is the <i>ladder</i> and "
        r"not the budget. That is an argument for a finer ladder before a "
        r"looser budget, and supplement E measures the crossover: a ladder "
        r"with twice as many exits loses at 0.1 dB, because splitting later "
        r"puts twice as many blocks in the per-tile section and the power law "
        r"of Section 4.1 raises the floor, and wins by several points at "
        r"0.5 dB, where the shipped ladder has been pinned at its ceiling "
        r"since 0.3 dB.")
    figure("budget_band.png",
           r"<b>Figure 5. The band is the rate dependence.</b> <b>a</b> Saving "
           r"against the budget, with each rate's floor and saturation point "
           r"ticked. <b>b</b> The same with the budget axis rescaled onto each "
           r"rate's own band. Five curves become one; dashed is the one-parameter "
           r"power law fitted to all of them.")
    par(r"The band accounts for almost all of the rate dependence. At a "
        r"matched decibel the five rates are \BandRawSpread points apart. "
        r"Rescale the budget axis onto each rate's own band, with the floor at "
        r"0 and saturation at 1, and they collapse onto a single master curve, "
        r"\BandSpreadMean points apart on average and \BandSpreadMax at worst. "
        r"The answer to how much a 0.1 dB budget buys at a given rate is "
        r"therefore, to within a couple of points, where 0.1 dB sits in that "
        r"rate's band. The floor and the saturation point are both in "
        r"closed form and both cheap to measure, and what they leave over is "
        r"small enough that a deployment could calibrate the two ends and read "
        r"the rest off one curve. That curve is a one-parameter power law, "
        r"saving ≈ C·u^\BandExp, with C the architectural ceiling and u the "
        r"position in the band, fitted in log space over all five rates at "
        r"R² = \BandRTwo. The claim we make "
        r"is the <i>rescaling</i>: measuring the budget in units of each "
        r"rate's own band is what collapses the curves. The exponent is an "
        r"empirical fit to that collapse and not a law, and inverse fits to the "
        r"same frontier disagree in it by up to 40%, so we read nothing into "
        r"its value.")
    par(r"The collapse appears robust across two checkpoints. We repeated the "
        r"measurement on BEST, a separate recipe taken at a different epoch, "
        r"and the same thing happens, with the rates \BandRawSpreadB points "
        r"apart at a matched decibel and \BandSpreadMeanB after rescaling. The "
        r"<i>exponent</i> does not carry over, \BandExpB there against "
        r"\BandExp here, so it describes a trained decoder and not the "
        r"architecture. What replicates is the collapse: within a checkpoint, "
        r"the rate dependence of the trade-off is the rate dependence of the "
        r"band.")

    # ---- 6 routing -------------------------------------------------------
    h1("6. Routing on bits, and real time")
    h2("6.1 Signalled against predicted")
    figure("router_ab.png",
           r"<b>Figure 6. Who decides.</b> <b>a</b>, the two decision paths "
           r"side by side: the encoder's search over all K exits per tile "
           r"against the head's forward pass over decoded data. <b>b</b>, "
           r"saving at 0.1 dB; shading is what a bit-exact bitstream costs. "
           r"<b>c</b>, that cost is smallest at q\GapMinQp, where the "
           r"router needs almost no cost multiplier to reach the budget from "
           r"where it was trained, and grows in both directions. At 0.3 dB it "
           r"falls to the router's own \RouterCostPct% wherever the budget "
           r"saturates the ladder, and widens where it does not.")
    tbl("ab",
        r"<b>Table 7. Signalled against predicted</b> at two budgets, same "
        r"checkpoint and test set, with one router trained against the oracle "
        r"on the deployed table at a single λ.")
    par(r"The table and figure above price exactness against bits. Not "
        r"signalling costs \GapMin–\GapMax points, roughly flat "
        r"across rate, for zero added bits and a byte-identical file. The gap "
        r"is smallest at q\GapMinQp and widens toward both ends of the rate "
        r"range, tracking |β|, the cost multiplier the bisection has to apply "
        r"to move a router trained at one λ onto another operating point. A "
        r"large β in either direction lets the cost term dominate the logits, "
        r"which discards the content ranking the router learned. What we ship "
        r"against that is one head for every quality index, with the offline β "
        r"table of Section 3.3 in front of it: the head's ordering is what a "
        r"deployment stores, and the multiplier that moves it onto an "
        r"operating point is a looked-up scalar rather than a second set of "
        r"weights. We do not propose a head per rate anywhere in this paper, "
        r"since that would multiply the stored parameters by the number of "
        r"operating points a deployment offers.")
    par(r"Loosening the budget closes the gap only where the budget saturates "
        r"the ladder. At 0.3 dB the gap is \GapLooseLow points at the three "
        r"lowest rates, which is exactly the router's own \RouterCostPct% of "
        r"decode, so its <i>prediction</i> is free there: both configurations "
        r"send every tile to the cheapest exit and nothing is left to predict "
        r"wrongly. At the two highest rates 0.3 dB does not saturate, and the "
        r"gap <i>widens</i> to \GapLooseHigh points. A looser budget gives the "
        r"allocation more room to be wrong in as well as more room to be "
        r"right. Every β in the table was bisected on the test frames, which "
        r"is the one thing Section 3.3 says a decoder cannot do; supplement "
        r"F.5 repeats the measurement with the table a deployment would ship "
        r"and reports where it overshoots.")
    h2("6.2 The calibrated bit rule")
    par(r"Before a \RouterParams head is worth its \RouterCostPct% of the "
        r"decode, it has to beat what the decoder already knows. The entropy "
        r"model has produced one number per tile before the trunk runs, and at "
        r"no cost: how many bits that tile's latents took. We turn it into a "
        r"routing rule with no learned parameters by modelling the per-tile "
        r"distortion as rank-1 in the log domain, "
        r"log D(t,k) ≈ α log b(t) + c + log φ_k, "
        r"where b is the tile's bit count normalised by the frame mean and φ "
        r"is a K-vector saying what each exit costs on an average tile. We fit "
        r"(α, c, φ) by least squares, leave-one-sequence-out, so that no "
        r"sequence contributes to the profile that routes it, and then run the "
        r"same Lagrangian the oracle runs on the surrogate. Nothing is "
        r"signalled and nothing is trained; the arithmetic is one scalar per "
        r"tile.")
    figure("raterank.png",
           r"<b>Figure 7. The calibrated bit rule.</b> <b>a</b> Saving at 0.1 dB; "
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
    par(r"It beats the trained head at every rate. The calibrated bit rule "
        r"is ahead of the \RouterParams router at all \RateRankNWins measured "
        r"rates, by margins that run from half a point at q48 to "
        r"\RateRankBeatsBy points at q0. A head trained on this decoder "
        r"against this oracle therefore returns nothing over a rule with no "
        r"parameters at all, and it carries \RouterCostPct% of the decode "
        r"that the rule does not. It replicates on BEST: there the rule beats "
        r"that run's own head at \BestRankWinsN of \BestRankOfN rates, by up "
        r"to \BestRankBy points, and comes within \BestRankToOracle points of "
        r"the <i>oracle</i> everywhere. We do not read that as showing a "
        r"learned head cannot be worth its parameters, only that neither of "
        r"our training runs produced one that is.")
    par(r"Why it works is not that it agrees with the oracle. It agrees "
        r"on \RateRankAgreeLo–\RateRankAgreeHi of tiles, well below the "
        r"router's \RetrainAgreeOld, and still saves more at every rate. "
        r"Agreement weighs a disagreement on a tile where two exits are within "
        r"a hair of each other exactly as heavily as one where the choice is "
        r"most of the frame's error, and most tiles are the former. The "
        r"ordering is what the rule gets right: bits correlate with the "
        r"<i>spread</i> across the ladder, that is, with how much a tile "
        r"stands to gain from depth, at "
        r"ρ_spread = \RateRankSpreadLo–\RateRankSpreadHi at every rate. At "
        r"0.3 dB the rule matches the oracle exactly at the three lowest rates "
        r"and reaches \RateRankLoose% at q63 against the oracle's "
        r"\SigLooseHigh%, which is ahead of the trained head everywhere. What it cannot do is see anything beyond that "
        r"ordering. A rank-1 model in the level assigns every tile the same "
        r"relative profile over exits, so b only decides where on the ladder a "
        r"tile falls, never the shape of its trade-off. That is the ceiling "
        r"this baseline sits at, and it is the part a learned head should be "
        r"earning its parameters on. Ours does not earn them at any rate we "
        r"measured. Per-block bit allocation is a standard quantity in learned "
        r"compression, where it is something to <i>choose</i> [42]; we read "
        r"the same number in the other direction, after the fact and at the "
        r"decoder, as a statement about how hard a region was.")
    h2("6.3 What the arithmetic does not see")
    tbl("latency",
        r"<b>Table 9. Wall-clock</b>, 1080p, median of 40 interleaved "
        r"iterations on one NVIDIA RTX A6000 with a 300 W board limit, PyTorch "
        r"2.6.0 and CUDA 12.4, single precision, one frame per iteration after "
        r"20 warm-up iterations, at the 0.1 dB operating point. "
        r"``MACs'' is what the arithmetic predicts; ``measured'' is the sorted "
        r"per-tile loop.")
    par(r"A saving in multiply-accumulates is not a saving in time. Tiling "
        r"costs \TilingOverhead% before anything exits early, measured as the "
        r"deepest-exit tiled decode against the released full-frame one at the "
        r"same arithmetic and the same weights. On top of that, the routed "
        r"decode realises \WallMasked% at q0 where its operations predict "
        r"\WallPredicted%. Four fifths of the predicted saving arrives; the "
        r"missing fifth is the tiling overhead together with the bookkeeping "
        r"of a shrinking active set, and a MAC count sees neither. Sorting the "
        r"tiles once by descending exit depth turns ``still active at group "
        r"g'' into a contiguous prefix, so each group is a slice rather than a "
        r"gather and no per-group mask is built at all. The arithmetic is "
        r"unchanged and the output is bit-identical on GPU, and at q0 the "
        r"saving goes from \WallMasked% to \WallSorted%, a gain of "
        r"\WallSortedGain points. Supplement B reports the same measurement in "
        r"joules, where energy follows time rather than arithmetic.")
    par(r"The same gap appears in our own accounting. The router is charged "
        r"at its share of the decoder's multiply-accumulates, "
        r"\RouterCostPct%. Timed directly, on the padded frame the decoder "
        r"actually sees and interleaved against a deepest-exit decode, it "
        r"costs \RouterTimePct%, which is \RouterTimeFactor× what its "
        r"arithmetic predicts: a \RouterParams-parameter head is launch "
        r"overhead, and a MAC count cannot see a launch. Charged at measured "
        r"time instead of at operations, every B number in this paper would "
        r"fall by a further \RouterTimeExtra points. We leave them charged at "
        r"MACs because that is the convention the rest of the literature "
        r"reports in, and we record the correction here so that it does not "
        r"sit unstated. A paper that quotes only operations would report "
        r"\WallPredicted% where the same code delivers \WallSorted%. The error "
        r"runs one way: operations are an <i>optimistic</i> bound on this "
        r"method, and the optimism grows with how much of the frame exits "
        r"early.")

    # ---- 7 limitations ---------------------------------------------------
    h1("7. Limitations")
    par(r"<b>The seam is reduced, and the exact remedy is not usable as it "
        r"stands.</b> The halo exchange removes "
        r"\CoupFloorDropLo–\CoupFloorDropHi% of the floor and is bit-exact at "
        r"uniform depth (Section 4.2). On a decoder trained with "
        r"independent-tile padding, introducing the exchange only at inference "
        r"also destroys the routed saving, because routing puts neighbouring "
        r"tiles at different depths and the exchange then carries a shallow "
        r"tile's feature into a deep neighbour. We state it that way "
        r"deliberately: the finding is about this decoder and this training, "
        r"and not about halo exchange in general. We do "
        r"not know whether a decoder trained with the exchange can recover both at "
        r"once, and that is the experiment we would run next.")
    par(r"<b>The shipped decoder pays for a filter it does not earn.</b> The "
        r"deblocking pass of Section 4.2 is inside the weights every number in "
        r"this paper was measured on. It costs 0.95% of the decode, it is the "
        r"reason our deepest exit costs more than the released decoder, and "
        r"the measurement in supplement H says it buys back almost nothing. We "
        r"charge it against the savings rather than remove it, because "
        r"removing it changes the weights and would invalidate every "
        r"measurement here. A ladder retrained without it starts from a "
        r"cheaper deepest exit and a slightly higher ceiling.")
    par(r"<b>Intra frames only.</b> This is the image path of a video codec. "
        r"Extending the ladder to inter frames raises a question we do not "
        r"answer here, because an exit map propagates through the reference "
        r"chain and a shallow tile in one frame is a worse reference for the "
        r"next one.")
    par(r"<b>The checkpoint is an early one.</b> Every number in this paper "
        r"is measured on one checkpoint, ckpt_PAPER, an early snapshot of the "
        r"RECIPE512 run. It is pinned because it is the checkpoint every "
        r"measurement in the paper has been made on, and a result table "
        r"assembled half from one checkpoint and half from another is the "
        r"failure this project has spent the most time undoing. Later "
        r"checkpoints of the same run improved the measured trade-off on the "
        r"same \NumSeq frames at the same budget, so the numbers here are not "
        r"the best the run produced. We do not present them as a bound on what "
        r"a converged run would give, since the two later checkpoints we "
        r"measured do not agree on a direction and neither of them is "
        r"converged either. Moving the paper to a later checkpoint also has a "
        r"cost that this one does not pay: the anchor that pins the deepest "
        r"exit to the released decoder drifts as training goes on, and a later "
        r"checkpoint would have to report that drift as its own row.")
    par(r"<b>A single decoder.</b> All results are on DCVC-UF's intra decoder. "
        r"Nothing in the method looks specific to it, since the ladder needs "
        r"only a residual trunk with a shared head, but we have not measured a "
        r"second decoder to check.")

    # ---- 8 conclusion ----------------------------------------------------
    h1("8. Conclusion")
    par(r"The intra decoder of DCVC-UF can be run at a depth chosen per tile "
        r"from what that tile contains. At a 0.1 dB budget signalled FLEX-UF "
        r"removes \MainLowRate% to \MainHighRate% of its "
        r"multiply-accumulates for a BD-Rate cost of \BdRateALow%, with no "
        r"change to the encoder, none to the coded latent, and \MapBits bits "
        r"per frame of routing metadata beside it; bitstream-identical FLEX-UF "
        r"adds nothing at all and reaches \RouterLowRate% to "
        r"\RouterHighRate%.")
    par(r"Three lessons we would carry to the next spatially adaptive decoder "
        r"we build, and none of them is about early exit. We have measured one "
        r"decoder, so we offer them as expectations for this architecture "
        r"class rather than as results about adaptive decoding in general.")
    par(r"<b>Tiling is the dominant cost and it is governed by depth.</b> All "
        r"of that cost traces back to a single 3×3 that is 0.29% of the "
        r"arithmetic, and the penalty grows as the square of how many such "
        r"convolutions run per tile. The affected-area fraction usually "
        r"quoted alongside it predicts that penalty badly.")
    par(r"<b>The exact remedy is in tension with the thing it enables.</b> "
        r"Giving each convolution its real neighbour is bit-identical at "
        r"uniform depth, and under routing it destroys the allocation, "
        r"because routing is the deliberate violation of the condition that "
        r"makes it exact. We expect a method that combines spatial "
        r"adaptivity with tiled inference to walk into this.")
    par(r"<b>A distortion budget is only a control variable inside a "
        r"measurable band.</b> Below the floor the budget admits nothing at "
        r"all, and above saturation more of it buys nothing. A saving quoted "
        r"without saying where in that band it sits has left out the part of the "
        r"result a reader most needs.")
    par(r"We would attach two cautions to the measurements themselves. A "
        r"saving in operations is an optimistic bound on a saving in time, and "
        r"the optimism scales with the saving. And we would compare a learned "
        r"router against a free one: routing on the bits already spent per "
        r"tile needs no parameters and no training, it adds nothing to the "
        r"stream, and it beats our trained head at every rate we measured, "
        r"while agreeing with the Lagrangian oracle on fewer tiles "
        r"than the head does. We read that as a warning about the metric quite "
        r"as much as about the head.")
    return F


# ------------------------------------------------------------ references
# Bibliography keys used inside generated tables, mapped to this document's own
# reference numbers. paper/main.tex resolves these through bibtex; there is no
# bibtex here, so the map is explicit and lives beside REFS. A key that is not
# here prints no citation rather than a wrong one.
CITE = {
    "arls": 11, "slimcae": 22, "slimvc": 23, "evc": 53,
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
 "G.-H. Wang, J. Li, B. Li, Y. Lu. EVC: towards real-time neural image compression with mask decay. ICLR, 2023.",
 "Y. Matsubara, M. Levorato, F. Restuccia. Split computing and early exiting for deep learning applications: survey and research challenges. ACM Computing Surveys, 2022.",
 "M. Wolczyk, B. Baran, A. Devoto et al. Zero time waste: recycling predictions in early exit neural networks. NeurIPS, 2021.",
]


def build(out="paper/FLEX-UF.pdf"):
    PW, PH = letter
    M, GAP = 0.62 * inch, 0.28 * inch
    colw = (PW - 2 * M - GAP) / 2
    # Title block plus the full-width teaser, CVPR style. Everything after
    # page 1 is two columns, which needs an explicit NextPageTemplate; without
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

    # The supplement is part of this document, not a second one. CVPR submits a
    # single PDF and the DCVC-UF paper this work builds on puts its
    # supplementary material after the references in the same file, which is
    # also what keeps the figure, table and equation numbers continuous.
    #
    # Imported inside build() rather than at module scope because
    # build_supp_pdf imports this module. When this file is run as a script it
    # is __main__, so that import loads a second copy of it under the name
    # build_pdf, with its own counters at zero; the supplement would then
    # restart at Figure 1 and the same table would be Table 10 in
    # FLEX-UF-supp.pdf and Table 1 here. Hand whichever copy the supplement
    # writes through the counts this document has reached. It is a no-op when
    # the two names are the same module.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_supp_pdf as SUPP
    SUPP.B._FIGN[0], SUPP.B._TABN[0], SUPP.B._EQN[0] = (_FIGN[0], _TABN[0],
                                                        _EQN[0])
    kit = SUPP.Kit(colw, PW - 2 * M)
    built = SUPP._sections(kit)
    if kit.flow:
        story += [NextPageTemplate("supp"), PageBreak(),
                  Paragraph("Where to Stop:<br/>Tile-Adaptive Early Exit in a "
                            "Learned Image Decoder", TITLE),
                  Paragraph("Supplementary Material", SUPP.SUBTITLE),
                  NextPageTemplate("rest"), FrameBreak()]
        story += kit.flow
        print(f"  supplement: {', '.join(built)}")
        if kit.bad_glyphs:
            print(f"  NO GLYPH IN Times-Roman: {sorted(kit.bad_glyphs)}")

    doc.build(story)
    if UNEXPANDED:
        print("  UNEXPANDED MACROS (run make_paper_tables.py): "
              + ", ".join(sorted(UNEXPANDED)))
    print(f"  -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "paper/FLEX-UF.pdf")
