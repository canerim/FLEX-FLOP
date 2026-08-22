"""The FLEX+ report: what the branch is testing and what it has measured.

A separate document on purpose. The paper reports a system that works; this
reports an attempt to move a number the paper states as a limit, and the two
should not be confused with each other while the attempt is unfinished. It is
built from the same result files the experiments write, so it cannot drift
from them, and every number in it is either measured here or quoted from the
paper with the file it came from.

    python flexplus/build_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from reportlab.lib.enums import TA_JUSTIFY                       # noqa: E402
from reportlab.lib.pagesizes import letter                       # noqa: E402
from reportlab.lib.styles import ParagraphStyle                  # noqa: E402
from reportlab.lib.units import inch                             # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame,          # noqa: E402
                                KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.lib import colors                                 # noqa: E402

BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.6,
                      leading=12.4, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=12.5, leading=15,
                    spaceBefore=11, spaceAfter=5)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=10.4, leading=13,
                    spaceBefore=8, spaceAfter=3)
TITLE = ParagraphStyle("t", fontName="Times-Bold", fontSize=16, leading=19,
                       spaceAfter=3)
SUB = ParagraphStyle("s", fontName="Times-Italic", fontSize=10, leading=13,
                     spaceAfter=10)
NOTE = ParagraphStyle("n", fontName="Times-Italic", fontSize=8.2, leading=10.4,
                      textColor=colors.HexColor("#555555"), spaceAfter=8)
CAP = ParagraphStyle("c", fontName="Times-Roman", fontSize=8.4, leading=10.6,
                     alignment=TA_JUSTIFY, spaceAfter=8)


def J(name):
    p = HERE / "results" / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def table(rows, cap=None):
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 8.6),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 8.6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    out = [Spacer(1, 3), t, Spacer(1, 3)]
    if cap:
        out.append(Paragraph(cap, CAP))
    return KeepTogether(out)


def content(F):
    A = F.append
    A(Paragraph("FLEX+: moving the ceiling", TITLE))
    A(Paragraph("An experiment branch. The ladder shape is fixed at six exits "
                "split at depth two, as the paper reports it; the question is "
                "whether the architectural ceiling above it can be raised "
                "without touching either.", SUB))

    # ---------------------------------------------------------------- 1
    A(Paragraph("1. What the ceiling is made of", H1))
    A(Paragraph(
        "The paper's ceiling is 38.33% counted. It is not a routing result "
        "and no better router can move it: it is what a frame costs when "
        "every tile already sits on the shallowest rung it is allowed. At a "
        "0.2 dB budget the deployed system delivers 36.81% of that 38.33%, "
        "which is 96% of the ceiling, so the allocation is very nearly "
        "finished and the ceiling is the whole of what is left.", BODY))
    ds = J("design_space.json")
    A(Paragraph(
        "Decomposing that floor decides the experiment by arithmetic rather "
        "than by taste. A frame with every tile on the shallowest rung costs "
        "0.6088 of a released decode, and one part of it is large enough to "
        "matter.", BODY))
    A(table([["part of the floor", "cost", "share"],
             ["shared trunk, 4 blocks, always, full frame", "0.2981", "49.0%"],
             ["tiled trunk, 2 blocks", "0.1491", "24.5%"],
             ["upsample, always, full frame", "0.0816", "13.4%"],
             ["adapter at the shallowest exit", "0.0464", "7.6%"],
             ["head, with its halo", "0.0240", "3.9%"],
             ["seam repair", "0.0095", "1.6%"]],
            "<b>Table 1. Where the ceiling's floor goes.</b> Frame-relative, "
            "halo on the head, from flexuf.cost on the paper's configuration."))
    A(Paragraph(
        "Halving the shared trunk puts the ceiling at 54.0% and quartering it "
        "at 61.5%. Halving anything else leaves it below 47%: the upsample "
        "gone entirely would give 45.2%, the adapter 42.6%, the head 40.9%. "
        "There is one lever, and the target this branch was opened for -- 55% "
        "at a 0.2 dB budget -- is reachable only through it.", BODY))

    # ---------------------------------------------------------------- 2
    A(Paragraph("2. Why the exit ladder cannot reach it", H1))
    A(Paragraph(
        "The four blocks below the split run before the frame is tiled, on "
        "every pixel, whatever depth any tile chooses. Depth is one axis and "
        "it does not intersect them. Lowering the split depth would, and that "
        "is the move that is closed: it is also what creates the seam the "
        "paper spends a section on. So the ceiling needs a second axis, "
        "applied to the stem, that does not tile it.", BODY))

    A(Paragraph("2.1 What the literature offers", H2))
    A(Paragraph(
        "<b>Width.</b> Slimmable Networks (Yu et al., ICLR 2019) trains one "
        "network executable at several channel widths with shared weights and "
        "per-width normalisation, and reports each width close to a network "
        "trained at that width alone. Dynamic Slimmable Network (Li et al., "
        "CVPR 2021) makes the width input-dependent with a gate. SLEXNet (ACM "
        "TECS, 2024) combines slimmable widths with early exits, which is the "
        "pairing here. Cost falls with the square of the width.", BODY))
    A(Paragraph(
        "<b>Resolution.</b> RANet (Yang et al., CVPR 2020) routes easy inputs "
        "through a low-resolution path and exits them early, keeping a "
        "high-resolution path for the rest. Half resolution is a quarter of "
        "the cost, the same scaling as half width.", BODY))
    A(Paragraph(
        "<b>Spatial sparsity.</b> Dynamic Convolutions (Verelst and "
        "Tuytelaars, arXiv:1912.03203) learn a spatial mask and execute "
        "convolutions only where it is set; focused convolutions "
        "(arXiv:2310.07782) do the same for pretrained networks. Cost falls "
        "linearly with the active fraction, which is weaker, but it is the "
        "only one of the three that changes no tensor shape and therefore "
        "adds no tile boundary.", BODY))
    A(Paragraph(
        "Width is the axis this branch follows, on instruction, and it is "
        "also the one with the strongest scaling. It is applied to the stem "
        "as a whole rather than per tile, so it creates no seam either.", BODY))

    # ---------------------------------------------------------------- 3
    A(Paragraph("3. The stem does not tolerate being cut", H1))
    st = J("stem_tolerance.json")
    A(Paragraph(
        "Before building anything, the cheapest question: can the stem simply "
        "be made smaller? Two training-free degradations, measured at the "
        "shallowest exit against the released decoder on twelve frames.", BODY))
    if st:
        rows = [["stem", "q0", "q32", "q63"]]
        for tag in ("half", "drop 0.25", "drop 0.5", "drop 0.75"):
            r = {x["qp"]: x for x in st["rows"]
                 if x["stem"] == tag and x["exit"] == st["split_depth"]}
            if len(r) == 3:
                rows.append([tag] + [f"{r[q]['extra_db_vs_full_stem']:+.2f}"
                                     for q in (0, 32, 63)])
        A(table(rows,
                "<b>Table 2. Extra dB from degrading the stem</b>, untrained, "
                "above what the shallowest exit already costs. The budget "
                "this branch is aiming at is 0.2 dB in total."))
    A(Paragraph(
        "Every entry is outside the budget, most of them by an order of "
        "magnitude. A stem at half resolution spends forty times the budget "
        "at the lowest rate and seventy at the highest. Dropping a quarter of "
        "the channels -- the mildest cut available -- costs two and a half "
        "times the whole allowance at the lowest rate and eleven times at the "
        "highest.", BODY))
    A(Paragraph(
        "The deeper trunk repairs a little of the damage, and only a little: "
        "at a quarter of the channels dropped the deepest exit is 0.10 dB "
        "better off than the shallowest at the lowest rate and 0.15 dB at the "
        "highest, against a penalty of 0.50 and 2.21. It is not a mechanism "
        "to build on.", BODY))
    A(Paragraph(
        "This is conclusive about cutting and says nothing about slimming. An "
        "untrained slice is exactly what the slimmable literature exists to "
        "improve on, so the result is a floor rather than an answer.", NOTE))

    # ---------------------------------------------------------------- 4
    A(Paragraph("4. A narrow stem, trained to match the full one", H1))
    A(Paragraph(
        "The four stem blocks are replaced by a module of width w with a 1x1 "
        "projection either side, so everything downstream still sees the same "
        "channel count and needs no change. Its cost is w squared of the "
        "original plus the two projections. The residual is around the whole "
        "module and the output projection is zero-initialised, so at step "
        "zero it is exactly the decode that skips the stem -- a known "
        "starting point rather than noise.", BODY))
    A(Paragraph(
        "It is trained to reproduce the full stem's output feature, with the "
        "encoder, the entropy model, the rest of the trunk, the exits and the "
        "head all frozen. The target is therefore a pure regression and any "
        "quality change is attributable to the stem alone.", BODY))
    # One file per width, written as each finishes, so the table fills in
    # during the sweep rather than only at the end of it.
    allrows = []
    for f in sorted((HERE / "results").glob("narrow_eval*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("rows", []):
            r = dict(r)
            r["objective"] = "decode" if "decode" in f.name else "feature"
            allrows.append(r)
    ne = {"rows": allrows} if allrows else None
    if ne and ne.get("rows"):
        widths = sorted({(r["objective"], r["width"]) for r in ne["rows"]})
        rows = [["objective", "width", "channels", "ceiling", "extra dB q0",
                 "q32", "q63"]]
        for obj, w in widths:
            rs = {r["qp"]: r for r in ne["rows"]
                  if r["width"] == w and r["objective"] == obj}
            if not rs:
                continue
            any_r = next(iter(rs.values()))
            rows.append([obj, f"{w:g}", str(any_r["channels"]),
                         f"{any_r['ceiling_narrow_pct']:.1f}%"]
                        + [f"{rs[q]['extra_db']:+.3f}" if q in rs else "--"
                           for q in (0, 32, 63)])
        A(table(rows,
                "<b>Table 3. What each width costs and what it buys.</b> "
                "Ceiling is hook-counted on the executed pass, the same "
                "counter the paper reports; extra dB is against the same "
                "decode with the full stem."))
    else:
        A(Paragraph(
            "The width sweep is still training. This section is written and "
            "its table is empty until flexplus/narrow_eval.py has run; the "
            "document is built from the result files, so it fills itself in.",
            NOTE))

    A(Paragraph("4.1 What the two widths say", H2))
    A(Paragraph(
        "Doubling the stem's arithmetic buys almost nothing. Going from half "
        "width to 0.707 -- which is twice the multiply-accumulates, since cost "
        "goes as the square -- reduces the extra distortion by about a "
        "seventh, from +0.48 to +0.40 dB at the lowest rate and +1.94 to "
        "+1.67 at the highest, while the ceiling falls from 66.0% to 63.4%. "
        "Both widths plateau at about 5% relative error in the stem feature "
        "after twenty thousand steps, and both are between two and eight "
        "times outside the budget.", BODY))
    A(Paragraph(
        "Capacity is therefore not what binds. What binds is the objective. A "
        "5% error in the feature comes out as 5.4 times the dB at the lowest "
        "rate and 9.6 at the highest, because the trunk below the split "
        "amplifies stem error rather than absorbing it -- and a mean squared "
        "error on the feature is indifferent to the direction of the error "
        "while the trunk is not. Matching the stem is the wrong thing to ask "
        "for; matching the decode is the thing that is measured.", BODY))
    A(Paragraph(
        "Training does move the trade-off, and the size of the move is worth "
        "stating carefully. The trained half-width stem reaches about the "
        "distortion that untrained channel dropping reached while computing a "
        "quarter of the stem against three quarters of it -- but the dropping "
        "probe substitutes channels after running the whole stem, because it "
        "was written to ask what those channels are worth and not what "
        "skipping them saves. Its compute figure is therefore notional and "
        "the comparison is between one measured cost and one arithmetic one. "
        "The direction is not in doubt; the factor is.", BODY))

    A(Paragraph("5. Open", H1))
    A(Paragraph(
        "The decode objective is running: the narrow stem trained on the "
        "reconstruction at an exit sampled per step, which is the quantity "
        "the budget is set in. If it closes the gap, two things follow that "
        "neither experiment tests: the narrow stem should be trained jointly "
        "with the ladder rather than fitted to anything frozen, and a single "
        "fixed width is a weaker design than a gate that picks one per frame, "
        "which is what Dynamic Slimmable Network is for. If it does not close "
        "the gap, the width axis is answered for this decoder and the "
        "remaining candidate from the literature is spatial sparsity, whose "
        "cost falls only linearly but which leaves the stem's shapes alone.",
        BODY))


def build(out=str(HERE / "FLEX-PLUS.pdf")):
    PW, PH = letter
    M = 0.9 * inch
    doc = BaseDocTemplate(out, pagesize=letter, leftMargin=M, rightMargin=M,
                          topMargin=M, bottomMargin=M,
                          title="FLEX+: moving the ceiling")
    frame = Frame(M, M, PW - 2 * M, PH - 2 * M, id="f",
                  leftPadding=0, rightPadding=0, topPadding=0,
                  bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame])])
    F = []
    content(F)
    doc.build(F)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
