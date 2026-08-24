"""The supplementary report, built the way this project builds papers.

There is no LaTeX on this machine -- scripts/build_pdf.py assembles the main
paper with reportlab -- so this does the same, in the NeurIPS single-column
geometry (5.5 in text, 10 pt). Numbers are read from the result files rather
than typed, for the same reason every table in the main paper is: a figure and
a sentence that disagree is the failure mode this project has already paid for.
"""
import json, sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Image,
                                Paragraph, Spacer, Table, TableStyle,
                                KeepTogether)

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
RES = HERE.parent / "flexplus" / "results"
UF = Path.home() / "FLEX-UF"

INK = colors.HexColor("#111111")
INK2 = colors.HexColor("#555555")
RULE = colors.HexColor("#c9c9c9")
BAND = colors.HexColor("#f4f6f8")

W, H = letter
LM = RM = (W - 5.5 * inch) / 2
TM, BM = 1.0 * inch, 1.0 * inch


def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=9.4, leading=11.6,
                textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5.2)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = S("body")
ABST = S("abst", fontSize=9.0, leading=11.0, spaceAfter=6)
H1 = S("h1", fontName="Times-Bold", fontSize=11.6, leading=13.4,
       spaceBefore=11, spaceAfter=4.5, alignment=0)
H2 = S("h2", fontName="Times-Bold", fontSize=9.9, leading=11.8,
       spaceBefore=7, spaceAfter=3, alignment=0)
CAP = S("cap", fontSize=8.2, leading=9.8, spaceBefore=3, spaceAfter=9,
        textColor=INK2, alignment=TA_JUSTIFY)
TITLE = S("title", fontName="Times-Bold", fontSize=15.5, leading=18,
          alignment=TA_CENTER, spaceAfter=3)
AUTH = S("auth", fontSize=9.6, leading=11.6, alignment=TA_CENTER, spaceAfter=13,
         textColor=INK2)
MONO = S("mono", fontName="Courier", fontSize=8.2, leading=10.0,
         spaceBefore=3, spaceAfter=6)


def J(name):
    return json.loads((RES / name).read_text())


_FIGN = {}
_TABN = {}


def fign(key):
    if key not in _FIGN:
        _FIGN[key] = len(_FIGN) + 1
    return _FIGN[key]


def tabn(key):
    if key not in _TABN:
        _TABN[key] = len(_TABN) + 1
    return _TABN[key]


def P(txt, st=BODY):
    return Paragraph(txt, st)


def figure(name, key, caption, width=5.5):
    img = Image(str(FIG / f"{name}.png"))
    iw, ih = img.imageWidth, img.imageHeight
    img.drawWidth = width * inch
    img.drawHeight = width * inch * ih / iw
    cap = P(f"<b>Sekil {fign(key)}.</b> {caption}", CAP)
    return KeepTogether([img, cap])


def table(rows, key, caption, widths=None, align_right_from=1, fs=8.4):
    cs = S("cell", fontSize=fs, leading=fs * 1.22, alignment=0, spaceAfter=0)
    cr = S("cellr", fontSize=fs, leading=fs * 1.22, alignment=2, spaceAfter=0)
    ch = S("cellh", fontName="Times-Bold", fontSize=fs, leading=fs * 1.22,
           alignment=0, spaceAfter=0)
    chr_ = S("cellhr", fontName="Times-Bold", fontSize=fs, leading=fs * 1.22,
             alignment=2, spaceAfter=0)
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, cell in enumerate(row):
            right = c >= align_right_from
            st = (chr_ if right else ch) if r == 0 else (cr if right else cs)
            out.append(Paragraph(str(cell), st))
        data.append(out)
    if widths is None:
        n = len(rows[0])
        widths = [1.75 * inch] + [(5.5 * inch - 1.75 * inch) / (n - 1)] * (n - 1)
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    cap = P(f"<b>Tablo {tabn(key)}.</b> {caption}", CAP)
    return KeepTogether([cap, t, Spacer(1, 9)])


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.6)
    canvas.setFillColor(INK2)
    canvas.drawCentredString(W / 2, BM - 24, str(doc.page))
    canvas.restoreState()


def build(story, out):
    doc = BaseDocTemplate(str(out), pagesize=letter, leftMargin=LM,
                          rightMargin=RM, topMargin=TM, bottomMargin=BM,
                          title="FLEX-UF ek deney raporu")
    frame = Frame(LM, BM, W - LM - RM, H - TM - BM, id="col",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_page)])
    doc.build(story)
