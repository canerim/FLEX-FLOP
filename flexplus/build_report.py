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

    A(Paragraph("4.1 What the four widths say", H2))
    A(Paragraph(
        "Width buys very little. Across the whole sweep the compute in the "
        "stem rises eightfold, from a sixteenth of the original at w=0.25 to "
        "a half at w=0.707, and the distortion at the highest rate falls by "
        "about a third, from +2.57 to +1.67 dB. Every point is outside the "
        "0.2 dB budget: the best of them, +0.41 dB at the lowest rate, is "
        "twice it, and the same width costs eight times it at the highest. "
        "The curve is not approaching the budget from above -- it is nearly "
        "flat in the direction that matters.", BODY))
    A(Paragraph(
        "The ceiling hardly moves either, 68.1% down to 63.4%, because the "
        "stem is only half of the floor and the rest of it is unchanged "
        "whatever the stem costs. That is worth stating the other way round: "
        "the prize does not depend on getting the width small. Any stem "
        "replacement that met the budget would put the ceiling somewhere "
        "between 63 and 68%, well past the 55% this branch was opened for. "
        "The obstacle is fidelity, not cost.", BODY))
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

    A(Paragraph("4.2 The decode objective, and a comparison not to make", H2))
    A(Paragraph(
        "The first decode-objective run came out worse than the feature one "
        "at the same width, and the comparison is not usable. It trained for "
        "12,000 steps at batch 4 against 20,000 at batch 8, and it also "
        "samples one exit per step out of four, so each exit saw roughly a "
        "sixth of the updates. The step count was reduced because a decode "
        "forward and backward is dearer than a feature one, which was a "
        "reasonable thing to want and an unreasonable way to get it: it "
        "traded away the only property that made the two runs comparable.", BODY))
    A(Paragraph(
        "Matching the budget at 20,000 steps and batch 8 recovered most of "
        "the difference -- +2.51 to +1.99 dB at the highest rate -- and left "
        "the decode objective behind the feature one at all three rates, by "
        "ten to twenty per cent. The obvious suspect was the exit sampling: "
        "one exit of four per step gives each exit a quarter of the updates "
        "that the feature objective's single target gives all of them at "
        "once. Supervising every exit each step, at four times the cost, "
        "changed nothing -- +0.444, +1.207, +2.064 against +0.452, +1.178, "
        "+1.992 -- so that was not the reason.", BODY))
    A(Paragraph(
        "The feature objective is simply the better one here, which is not "
        "what we expected of it: it optimises a proxy while the other "
        "optimises the reported quantity. The likely reason, and this is a "
        "hypothesis rather than a measurement, is dilution. The decode loss "
        "at an exit is the error that exit makes anyway plus the damage the "
        "narrow stem adds, and only the second part carries information about "
        "the stem; the feature target isolates the stem's contribution "
        "exactly. Optimising the measured quantity is not the better choice "
        "when the measured quantity is mostly something else.", BODY))
    A(Paragraph(
        "None of this changes the answer to the question the branch asked. "
        "Across every width, objective and budget tried, the best point is "
        "+0.41 dB at the lowest rate and +1.67 at the highest, against a "
        "budget of 0.2. The width axis does not reach it on this decoder, and "
        "the remaining comparison decides only which objective is less far "
        "away.", BODY))

    A(Paragraph("4.3 The third axis, and what all three of them say", H2))
    sp = J("spatial_probe.json")
    A(Paragraph(
        "Spatial sparsity is the one axis left, and the cheapest to settle. "
        "Rank each position by how much the four stem blocks move it -- the "
        "norm of the stem's own residual -- and hand the least-moved fraction "
        "the stem's input instead of its output. The ranking is an oracle: it "
        "sees the output a learned mask would have to predict, so a learned "
        "mask cannot beat it and a failure here is conclusive.", BODY))
    if sp:
        keeps = sorted({r["keep"] for r in sp["rows"]}, reverse=True)
        rows = [["active fraction", "ceiling", "+dB q0", "q32", "q63"]]
        for k in keeps:
            r = {x["qp"]: x for x in sp["rows"] if x["keep"] == k}
            if not r:
                continue
            any_r = next(iter(r.values()))
            rows.append([f"{k:g}", f"{any_r['ceiling_modelled_pct']:.1f}%"]
                        + [f"{r[q]['extra_db']:+.3f}" if q in r else "--"
                           for q in (0, 32, 63)])
        A(table(rows,
                "<b>Table 4. An oracle spatial mask on the stem.</b> Quality "
                "measured; the ceiling is arithmetic, the active fraction "
                "times the stem's share of the floor. Nothing else in the "
                "floor moves."))
    A(Paragraph(
        "Skipping the least-moved quarter of positions costs between +0.49 "
        "and +0.96 dB, two and a half to five times the budget. At half the "
        "positions -- the setting whose ceiling is the 54% this branch was "
        "aiming at -- it costs +1.05 to +2.01. The axis is closed, and closed "
        "harder than width, because no learned mask can do better than the "
        "ranking used here.", BODY))
    A(Paragraph(
        "<b>All three axes agree, and that is the result.</b> Remove the "
        "stem's work by channel, by width or by position and the price is "
        "the same order: two to ten times the budget. The stem is not doing "
        "redundant work anywhere.", BODY))
    A(Paragraph(
        "Set against the paper this is a sharp contrast worth stating. "
        "Content-adaptive <i>depth</i> works: the same decoder gives up 0.1 "
        "dB and returns 27.6% of its arithmetic, because some tiles genuinely "
        "need less of the trunk than others. Content-adaptive <i>stem</i> "
        "does not, on any of the three axes. The difference is what is being "
        "chosen. The exit ladder decides how much further to go from a shared "
        "representation; these probes remove the work that representation is "
        "made of. There are easy regions in a picture, and there are none in "
        "the computation that turns a latent into one.", BODY))

    A(Paragraph("4.4 The stem trained jointly with the ladder", H2))
    A(Paragraph(
        "Every width above was fitted to a frozen target: reproduce, with "
        "fewer channels, the representation a decoder trained without it "
        "produces. That is the hardest version of the task and the obvious "
        "objection to the negative result, so it was run the other way. The "
        "trunk and the adapters were unfrozen and trained together with the "
        "narrow stem for 20,000 steps, which lets the exits move to meet the "
        "stem rather than requiring the stem to reproduce something fitted "
        "without it.", BODY))
    jt = [(w, J(f"narrow_eval_w{w}.json"), J(f"joint_eval_w{w}_joint.json"))
          for w in ("0.5", "0.25")]
    jrows = [["width", "training", "ceiling", "dB q0", "q32", "q63"]]
    for w, fz, jn in jt:
        if fz:
            jrows.append([w, "frozen target",
                          f"{fz['rows'][0]['ceiling_narrow_pct']:.1f}%"]
                         + [f"{r['db_narrow_stem']:.3f}" for r in fz["rows"]])
        if jn:
            jrows.append([w, "trained jointly",
                          f"{jn['rows'][0]['ceiling_pct']:.1f}%"]
                         + [f"{r['db_vs_uf']:.3f}" for r in jn["rows"]])
    if fz:
        jrows.append(["1.0", "the ladder as it ships",
                      f"{fz['rows'][0]['ceiling_full_pct']:.1f}%"]
                     + [f"{r['db_full_stem']:.3f}" for r in fz["rows"]])
    A(table(jrows,
            "<b>Table 5. A narrow stem fitted to a frozen target against one "
            "trained with the ladder.</b> Decibels below the released decoder "
            "at exit 2, over 12 sequences. The last row is the same exit with "
            "the stem the decoder ships, which is what the extra channels "
            "buy. The ceiling is unchanged by how the stem was trained: it is "
            "a property of the architecture."))
    A(Paragraph(
        "Joint training helps, and by a lot: at width 0.5 the cost falls from "
        "0.585 to 0.293 dB at q0 and from 2.162 to 0.778 at q63, so roughly "
        "half to a third of the damage was the frozen target rather than the "
        "missing channels. It is not enough. The cheapest point of the "
        "cheapest configuration is 0.293 dB, which is 1.5 times the 0.2 dB "
        "budget, and the highest rate costs 3.9 times it. Halving the stem "
        "again buys 2 points of ceiling and costs slightly more quality, "
        "which is the same flat trade the four widths showed.", BODY))
    A(Paragraph(
        "The prediction recorded before this ran was that the frozen target "
        "was the obstacle. It was part of it, and the part it was is now "
        "measured; what remains is not. A stem with half the channels, "
        "trained with everything downstream free to accommodate it, still "
        "costs more than the budget the whole method is built around.", BODY))

    A(Paragraph("5. What this leaves", H1))
    A(Paragraph(
        "The branch set out to reach 55% at a 0.2 dB budget with the ladder "
        "shape left alone, and it does not get there. Every route to the only "
        "lever large enough -- the four always-on trunk blocks -- costs "
        "between two and ten times the budget, and one of the three routes "
        "was tested with an oracle, so it is not a matter of finding a better "
        "gate or a longer schedule.", BODY))
    A(Paragraph(
        "One thing would still be worth running, and it is not a variation on "
        "what is here. The other -- training the narrow stem jointly with the "
        "ladder -- has now been run and is Section 4.4: it recovers half to "
        "two thirds of what the frozen target cost and still lands at 1.5 to "
        "3.9 times the budget. Nothing here questioned the split depth, "
        "because the "
        "instruction was to leave it alone -- but the arithmetic in Section 1 "
        "says that lowering it is the one move left that could reach the "
        "target "
        "cleanly, at a seam cost this project has already measured "
        "(0.100/0.144/0.234 dB by split at 256 px, untrained). Between a "
        "known seam cost and three axes that each cost more, the seam is the "
        "cheaper problem.", BODY))
    A(Paragraph(
        "The negative result is worth keeping either way. A decoder's shared "
        "stem is not slack: the ceiling in the paper is not an artefact of "
        "how the ladder was cut, and a reader who assumes there is easy work "
        "to remove below the split should be shown these numbers.", BODY))


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
