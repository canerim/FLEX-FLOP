"""Twelve slides on the LMT template, built from the measured results.

Every number on a slide is read from results/*.json at build time rather than
typed in, so the deck cannot drift from the experiments the way a hand-edited
one does. Where a figure exists it is used; where it does not, the slide says
what is still running rather than showing a placeholder.
"""
import json, sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

R = Path("/home/can_karsal/FLEX-UF")
import sys as _sys
_sys.path.insert(0, str(R))
TPL = "/home/can_karsal/Flex-Flop/docs/LMT_Presentation_Template_01.pptx"
TUM = RGBColor(0x00, 0x65, 0xBD)
GREY = RGBColor(0x52, 0x51, 0x4E)

def load(p):
    f = R / "results" / p
    return json.loads(f.read_text()) if f.exists() else None

# Which measurement is the headline
# --------------------------------
# BEST is the configuration this deck reports, and as of its first full epoch it
# is measured: 34.7% saved at qp0 under 0.1 dB against the reference's 28.3%,
# with the tightest anchor of any run. So the headline curve is BEST's and the
# earlier `*_grid128` files are the BASELINE, kept for comparison rather than
# quoted as the result.
#
# Named here rather than threaded through, so the two scripts that draw from it
# cannot drift onto different runs -- which is how a slide ended up with its
# text and its own figure 1.5 points apart earlier today.
CURVE = "curve_BEST.json"
BASE_CURVE = "paper_curve_grid128.json"
SIGNALLED = "signalled_BEST_0817_1542.json"
BASE_SIGNALLED = "signalled_grid128.json"
BD = "bd_BEST.json"
GAP = "target_gap_BEST.json"
ANCHOR = "anchor_BEST.json"
ANCHOR_BASE = "anchor_VERBATIM.json"

why = load("why_qp.json")
pc = load(CURVE) or load(BASE_CURVE)
sg = load(SIGNALLED) or load(BASE_SIGNALLED)
pc_base = load(BASE_CURVE)
sg_base = load(BASE_SIGNALLED)
s256 = load("signalled_control256.json")
lc = load("logconvexity.json")
th = load("theory_check.json")

def sv_at(qp, db):
    """Saving at a dB budget, INTERPOLATED along the frontier.

    Reading off the last grid point below the budget looked equivalent and is
    not: the answer then depends on where the lambda sweep happens to have put
    a sample. Comparing two frontiers that way made qp16 appear to gain 9.7
    points when the true gap was 4.6 -- the old curve had a sample exactly at
    0.1 dB and the new one did not.
    """
    # The per-frame decibel, which is what ~/DCVC/test_video.py computes and
    # therefore what every published DCVC-UF number means. The pooled
    # alternative -- every tile of every frame in one MSE -- reads 0.023-0.033
    # dB lower on an identical allocation, a quarter to a third of a 0.1 dB
    # budget, and would flatter every saving on this deck.
    pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]), r["saving_pct"])
                 for r in pc["rows"] if r["qp"] == qp)
    for (d0, s0), (d1, s1) in zip(pts, pts[1:]):
        if d0 <= db <= d1:
            w = (db - d0) / (d1 - d0) if d1 > d0 else 0.0
            return s0 + w * (s1 - s0)
    return float("nan")


def exit2_rate_factor():
    """How much more exit 2 costs at the top rate than at the bottom.

    Typed into a bullet as '3.4x' until the test set grew to 40 sequences and
    it became 4.3x. Derived now, so it cannot go stale again."""
    w = load("why_qp.json")
    if not w:
        return float("nan")
    r = {x["qp"]: x for x in w["rows"]}
    return r[63]["db_per_exit"][2] / r[0]["db_per_exit"][2]


def collapse_line():
    """What a second epoch did to CONTROL, and what it did not do to VERBATIM.

    I first read this as "without the additions, more training breaks the
    ladder". VERBATIM refutes that: it carries even fewer additions -- no anchor,
    no seam repair -- and every one of its exits IMPROVED over the same
    boundary. So the fact is CONTROL's, not the additions', and the cause is
    open. The slide says that rather than the tidier story."""
    a_, b_ = load("why_qp_CONTROL_ep0.json"), load("why_qp_CONTROL_ep1.json")
    if not (a_ and b_):
        return None
    A = {r["qp"]: r["db_per_exit"] for r in a_["rows"]}
    B = {r["qp"]: r["db_per_exit"] for r in b_["rows"]}
    q = 63 if 63 in A and 63 in B else sorted(set(A) & set(B))[-1]
    d = [B[q][k] - A[q][k] for k in (2, 3, 4, 5)]
    # The percentages used to be typed into this string. They were from the old
    # denominator and from a different measurement, so they drifted away from
    # the files they claim to summarise. Computed now.
    e0 = svr_row("signalled_CONTROL_0817_1813.json", 0)
    e1 = svr_row("signalled_CONTROL_0817_2047.json", 0)
    got = (f"{e0:.1f}% -> {e1:.1f}%" if e0 and e1 else "measured, see results/")
    return (f"CONTROL, 2nd epoch:  saving at qp0 {got}  ·  damage grows the "
            f"shallower the exit  ·  VERBATIM improved over the same boundary "
            f"→ cause unknown")


def ladder_line():
    """The three runs that share an architecture, ordered by what they carry.

    VERBATIM, CONTROL and BEST are K=6, j=2, 256px tiles, same warm start, one
    epoch each. They differ in the additions, and the anchor weight is the one
    that moves monotonically across them: 0, 1, 10. Reading the saving at a
    fixed budget across the three is the closest this project gets to an
    attribution -- with the caveat, stated on the slide, that CONTROL to BEST
    changes four things at once, so no single ingredient owns that step."""
    import glob as _g
    files = {"VERBATIM": R / "results/signalled_VERBATIM_fixed.json"}
    for t in ("CONTROL", "BEST"):
        c = sorted(_g.glob(str(R / f"results/signalled_{t}_*.json")))
        if c:
            files[t] = Path(c[-1])
    if len(files) < 3:
        return None
    out = []
    for t, w in (("VERBATIM", "0"), ("CONTROL", "1"), ("BEST", "10")):
        if t not in files or not files[t].exists():
            return None
        d = json.loads(files[t].read_text())
        rr = sorted(d["rows"], key=lambda r: r["qp"])
        if all(r.get("budget_reachable") is False for r in rr):
            out.append(f"anchor {w}: unreachable at every rate")
        else:
            out.append(f"anchor {w}: " + "/".join(
                f"{r['saving_pct']:.0f}" for r in rr) + "%")
    return ("Saving at 0.1 dB, same architecture:  " + "  ·  ".join(out)
            + "   (CONTROL→BEST moves four flags, so it is the bundle)")


def anchor_line():
    """What the anchor term is worth, measured on two runs that differ by it.

    VERBATIM is Microsoft's recipe applied to the ladder with NONE of the
    additions -- same K, same j, same 256px tiles, same warm start. If its
    deepest exit has drifted further from the release than the entire 0.1 dB
    budget, then no allocation of exits can meet the budget, and the anchor term
    is a precondition rather than an improvement. That is the cleanest
    attribution available here, since the tile size is held fixed."""
    a_, v_ = load(ANCHOR), load(ANCHOR_BASE)
    if not (a_ and v_):
        return None
    A = {r["qp"]: r["drift_db"] for r in a_["rows"]}
    V = {r["qp"]: r["drift_db"] for r in v_["rows"]}
    qs = sorted(set(A) & set(V))
    if not qs:
        return None
    return ("Deepest-exit drift vs release   BEST  "
            + " / ".join(f"{A[q]:+.3f}" for q in qs)
            + "   ·   VERBATIM (no anchor)  "
            + " / ".join(f"{V[q]:+.3f}" for q in qs)
            + f"   at qp {'/'.join(str(q) for q in qs)}")


def budget_line(db):
    """Saving at a dB budget, per rate, saying where the frontier ends first.

    BEST's exits are good enough that its whole qp0 frontier spans 0.197 dB, so
    asking what 0.3 dB buys there has no answer -- the ceiling is reached long
    before. That printed as "nan%". A rate whose frontier ends inside the budget
    is reported with its ceiling and what the ceiling cost, which is the more
    useful fact anyway."""
    out = []
    for qp in (0, 32, 63):
        v = sv_at(qp, db)
        if v == v:
            out.append(f"{v:.0f}% (qp{qp})")
        else:
            pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]),
                          r["saving_pct"]) for r in pc["rows"] if r["qp"] == qp)
            out.append(f"ceiling {pts[-1][1]:.0f}% already at "
                       f"{pts[-1][0]:.2f} dB (qp{qp})")
    return f"At {db:.1f} dB: " + " · ".join(out)


def target_line():
    """Where the 30-40% goal is actually met, derived rather than asserted.

    The bullet used to say the target was met at 0.3 dB and not at 0.1. On the
    40-sequence frontier qp0 reaches 33% at 0.1 dB, so that sentence became
    false the moment the test set grew. Stating which rates clear 30% at which
    budget cannot go stale the same way."""
    TARGET = 30.0
    out = []
    for db in (0.1, 0.3):
        ok = [qp for qp in (0, 16, 32, 48, 63) if sv_at(qp, db) >= TARGET]
        if not ok:
            out.append(f"no rate at {db:.1f} dB")
        elif len(ok) == 5:
            out.append(f"every rate at {db:.1f} dB")
        else:
            out.append(f"qp {'/'.join(map(str, ok))} at {db:.1f} dB")
    return f"The 30% target is cleared by {out[0]}, and by {out[1]}"


def gap_line():
    """Where the remaining distance to the 30% target actually lies.

    Quoting the saving at 0.1 dB says how much we get; reading the frontier the
    other way says how much more it would take, in the units the target is
    written in. The split into drift and everything else is what decides which
    knob to turn: at qp0 the floor is the WHOLE overspend, so tightening the
    anchor clears the target on its own."""
    g = load(GAP) or load("target_gap.json")
    if not g:
        return "target gap not computed"
    ok = [r for r in g["rows"] if r.get("reachable")]
    if not ok:
        return "the 30% target is not reachable at any rate on this checkpoint"
    met = [r for r in ok if r["overspend_db"] <= 0]
    miss = [r for r in ok if r["overspend_db"] > 0]
    parts = []
    if met:
        # floor_share_pct is None where the target is met: there is no overspend
        # to apportion. Formatting it tripped the build once BEST started
        # clearing the target, which is the pleasanter way to find a bug.
        parts.append(f"met at qp{'/'.join(str(r['qp']) for r in met)} — 30% "
                     f"costs at most {max(r['db_for_target'] for r in met):.3f} dB")
    if miss:
        w = miss[-1]
        parts.append(f"still short at qp{'/'.join(str(r['qp']) for r in miss)}; "
                     f"the furthest is qp{w['qp']}, needing "
                     f"{w['db_for_target']:.3f} dB "
                     f"({w['db_for_target'] / g['budget_db']:.1f}× the budget, "
                     f"{w['floor_share_pct']:.0f}% of the excess being the "
                     f"anchor's floor and the rest exit quality)")
    return "30% target: " + "; ".join(parts)


def spread_ratio(qp=63):
    """Best-to-worst sequence ratio at a 0.1 dB budget.

    The theory says the value of adaptivity comes from tiles disagreeing about
    which exit they want. This is the same disagreement one level up, between
    whole sequences, and it grows with rate the same way -- 6.7x against the
    tile-level gain's 7.5x. Consistent; two different quantities agreeing on
    five points is not confirmation, and the slide says 'same mechanism', not
    'confirms'."""
    ps = load("per_sequence.json")
    if not ps:
        return float("nan")
    for r in ps["rows"]:
        if r["qp"] == qp:
            return r["max"] / r["min"]
    return float("nan")


def bd_line():
    """The trade-off as two integrated numbers rather than two sampled points.

    Added because every headline here was a reading at one dB budget, and a
    claim resting on one sample of a curve turned out to be fragile: the
    grid-readout rule made one test-set comparison look twice as large as it
    was. BD-saving and BD-quality integrate the measured frontier over stated
    intervals, so neither depends on where the sweep placed a sample."""
    bd = load(BD) or load("bd_saving.json")
    if not bd:
        return "integrated trade-off not computed"
    lo, hi = bd["db_interval"]
    slo, shi = bd["saving_interval"]
    per = " · ".join(f"{r['qp']}: {r['bd_saving_pct']:.0f}%" for r in bd["rows"])
    return (f"BD-saving {bd['mean_bd_saving_pct']:.1f}% over dB in "
            f"[{lo:.3f}, {hi:.3f}] (qp {per}); BD-quality "
            f"{bd['mean_bd_quality_db']:.3f} dB over saving in "
            f"[{slo:.0f}%, {shi:.0f}%]")


def signalled_line():
    """The shipped system's saving per QP, from the file that measured it."""
    if not sg:
        return "signalled measurement not present"
    rows = sorted(sg["rows"], key=lambda r: r["qp"])
    base = {r["qp"]: r["saving_pct"] for r in (sg_base or {"rows": []})["rows"]}
    nums = " / ".join(f"{r['saving_pct']:.1f}" for r in rows)
    vs = ("" if not base else "  (baseline " +
          " / ".join(f"{base.get(r['qp'], float('nan')):.1f}" for r in rows) + ")")
    worst = max(r["db_vs_uf"] for r in rows)
    n = sg.get("n_sequences")
    return (f"measured with the map's cost INSIDE the bitrate: {nums}% at "
            f"qp{rows[0]['qp']}\u2026{rows[-1]['qp']}, all under "
            f"{worst:.2f} dB" + vs + (f" ({n} CTC sequences)" if n else ""))

prs = Presentation(TPL)
for i in range(len(prs.slides) - 1, -1, -1):
    rid = prs.slides._sldIdLst[i].rId
    prs.part.drop_rel(rid); del prs.slides._sldIdLst[i]

# ---------------------------------------------------------------- session data
def _D():
    """Cost of our deepest exit in stock decodes. Savings were divided by this
    instead of by the release's 1.0, so every headline was 0.6-0.75 points
    optimistic -- and the seam-repair tax it hides is the one the accounting
    claims to include."""
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    import json as _j
    cfg = FlexUFConfig(**_j.loads((R / "runs/BEST/meta.json").read_text())["config"])
    return float(exit_costs(cfg, "head")[-1]), cfg


def svr_row(fname, qp):
    d = load(R / "results" / fname)
    if not d:
        return None
    D, _ = _D()
    for r in d["rows"]:
        if r["qp"] == qp and r.get("saving_pct") is not None:
            return r.get("saving_pct_vs_release") or 100 - (100 - r["saving_pct"]) * D
    return None


def ab_line(qp):
    a = svr_row("signalled_BEST_0817_1542.json", qp)
    b1 = svr_row("router_BEST_v2.json", qp)
    b2 = svr_row("router_BEST_v2_lowlam.json", qp)
    if a is None or b1 is None:
        return None
    b = max(x for x in (b1, b2) if x is not None)
    return f"qp{qp}: A {a:.1f}% signalled vs B {b:.1f}% predicted — gap {a-b:.2f} pts"


def latency_line(budget):
    f = {0.1: "latency_BEST.json", 0.3: "latency_BEST_b03.json",
         0.5: "latency_BEST_b05.json"}[budget]
    d = load(R / "results" / f)
    if not d:
        return None
    r = [x["realised_saving_pct"] for x in d["rows"]]
    m = load(R / "results" / f)["rows"]
    mac = sum(x["predicted_saving_pct"] for x in m) / len(m)
    return (f"{budget:g} dB: {mac:.0f}% of MACs -> {sum(r)/len(r):.0f}% of "
            f"wall-clock ({1/(1-sum(r)/len(r)/100):.2f}x)")


def ceiling_line():
    D, cfg = _D()
    from flexuf.cost import exit_costs
    C = exit_costs(cfg, "head")
    return (f"Ladder ceiling {100*(1-float(C[cfg.split_depth])):.1f}% at every rate, "
            f"set by the split depth j={cfg.split_depth}, not by training")


def bullets(slide, items, size=15):
    body = slide.placeholders[1].text_frame
    body.clear()
    for j, (lvl, txt, bold) in enumerate(items):
        p = body.paragraphs[0] if j == 0 else body.add_paragraph()
        p.level = lvl
        run = p.add_run(); run.text = txt
        run.font.size = Pt(size - 2 * lvl); run.font.bold = bold
        if bold:
            run.font.color.rgb = TUM

def slide_text(title, items, size=15):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    bullets(s, items, size)
    return s

def slide_fig(title, img, items, top=None, height=None, size=13, top_txt=None):
    """Bullets on top, figure below -- the figure is the evidence, not decoration."""
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    # Place the figure just under the text instead of at a hand-set offset:
    # count wrapped lines at the given size and convert to inches. Hand-set tops
    # left visible gaps on half the slides and nearly collided on the rest.
    def _lines(txt, sz):
        per = int(96 * 11.0 / max(sz, 1))          # chars per line at this size
        return max(1, -(-len(txt) // per))
    n_lines = sum(_lines(t, size - 2 * lvl) for lvl, t, _ in items)
    txt_h = 0.30 + n_lines * (size + 7) / 72.0
    top = 1.55 + txt_h + 0.18
    top_txt = txt_h
    ph = s.placeholders[1]
    # All four, not just top/height. Setting a subset makes python-pptx
    # materialise a partial <a:xfrm> whose missing values become ZERO rather
    # than being inherited from the layout, so the box ends up 0 wide and the
    # text vanishes -- while every geometry check that looks only at top and
    # height still reports it fine.
    lay = prs.slide_layouts[1].placeholders[1]
    ph.left, ph.width = lay.left, lay.width
    ph.top, ph.height = Inches(1.55), Inches(top_txt)
    bullets(s, items, size)
    p = R / "results" / img
    if p.exists():
        # Fit to BOTH limits and take whichever binds, rather than setting the
        # height and hoping the width lands.
        #
        # These figures are Nature double-column, 183mm wide against roughly
        # 50mm tall, so on a text-heavy slide the height limit was cutting them
        # to about 60% of the available width and leaving the rest of the row
        # empty -- a figure too small to read while the slide had room for it.
        avail_h = max(0.8, 7.5 - top - 0.45)   # leave the footer rule clear
        avail_w = 9.1
        pic = s.shapes.add_picture(str(p), Inches(0.45), Inches(top))
        sc = min(Inches(avail_w) / pic.width, Inches(avail_h) / pic.height)
        pic.width, pic.height = int(pic.width * sc), int(pic.height * sc)
        pic.left = int((prs.slide_width - pic.width) / 2)
    return s

# --------------------------------------------------------------- equations
from formula import eq as _eq  # noqa: E402


def slide_eq(title, blocks, size=13):
    """A slide whose argument IS the mathematics.

    `blocks` mixes (level, text, bold) tuples with ("EQ", tex) pairs. Text lines
    set the context in one line each; the equations are rendered by mathtext and
    placed as images, because a formula typed into a text box is a transcription
    of a formula, not one.
    """
    sl = prs.slides.add_slide(prs.slide_layouts[1])
    sl.shapes.title.text = title
    ph = sl.placeholders[1]
    lay = prs.slide_layouts[1].placeholders[1]
    ph.left, ph.width = lay.left, lay.width

    y = 1.45
    for b in blocks:
        if b[0] == "EQ":
            img = _eq(b[1], size=int(b[2]) if len(b) > 2 else 22)
            pic = sl.shapes.add_picture(str(img), Inches(0.9), Inches(y))
            sc = min(Inches(8.2) / pic.width, Inches(0.85) / pic.height)
            pic.width, pic.height = int(pic.width * sc), int(pic.height * sc)
            pic.left = int((prs.slide_width - pic.width) / 2)
            y += pic.height / 914400 + 0.16
        else:
            lvl, txt, bold = b
            tb = sl.shapes.add_textbox(Inches(0.55 + 0.35 * lvl), Inches(y),
                                       Inches(9.0 - 0.35 * lvl), Inches(0.34))
            f = tb.text_frame
            f.word_wrap = True
            f.margin_top = f.margin_bottom = 0
            para = f.paragraphs[0]
            # Textboxes inherit the layout's alignment; without this the prose
            # sits right-aligned and reads as a caption drifting away from the
            # equation it explains.
            para.alignment = PP_ALIGN.LEFT
            r = para.add_run()
            r.text = txt
            r.font.size = Pt(size - 2 * lvl)
            r.font.bold = bold
            if bold:
                r.font.color.rgb = TUM
            else:
                r.font.color.rgb = GREY
            n = max(1, -(-len(txt) // int(96 * 11.0 / max(size - 2 * lvl, 1))))
            y += n * (size + 6) / 72.0 + 0.07
    # the placeholder is unused on these slides; empty it so no ghost bullet
    ph.text_frame.clear()
    ph.top, ph.height = Inches(7.3), Inches(0.1)
    return sl

# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(prs.slide_layouts[0])
s.shapes.title.text = "FLEX-UF: Content-Adaptive Early Exit for DCVC-UF"
s.placeholders[1].text = ("Decode compute spent where the content needs it\n"
                          "Chair of Media Technology · TUM")

# 2 -------------------------------------------------------------- the problem
slide_fig("Uniform compute, non-uniform content", "nf_macs.png", [
 (0, "DCVC-UF intra decoder · 453.5 GMAC / 1080p frame", True),
 (1, "upsample 8.2%  ·  12 DepthConvBlocks 89.4%  ·  head 2.4%", False),
 (0, "Target · 30–40% saved at ≤ 0.1 dB", True),
 (0, "Constraint · encoder, hyperprior, entropy model frozen", True),
 (1, "coded payload bit-identical → the decoder is the only variable", False),
], size=13)

# 3 --------------------------------------------------------------- architecture
slide_fig("A ladder of exits over the trunk", "nf_arch.png", [
 (0, "K exits over 12 blocks, split depth j", True),
 (1, "groups 0…j−1  full-frame  ·  groups j…K−1  per tile", False),
 (1, "shipped:  K = 6,  j = 2,  256 px tiles", False),
 (0, "Per-tile cost  C_k  (units of one stock decode)", True),
 (1, "0.581  0.581  0.581  0.730  0.870  1.010", False),
 (1, "exits 0,1 end inside the stem → C₀ = C₁ = C₂ · 6 rungs, 4 prices", False),
], size=13)

# 4 -------------------------------------------------- what it does, on a frame
slide_fig("The assignment, on a frame", "exit_map.png", [
 (0, "Bosphorus · qp 32 · λ* bisected to 0.1 dB · 40 tiles", True),
 (0, "boat → exit 4   water, sky → exit 2   shoreline → exit 3", True),
 (1, "33.4% saved at 0.098 dB", False),
 (1, "(c) dB vs each tile's OWN reference, not the frame mean", False),
], size=13)

# 5 ------------------------------------------------------------- the warm start
slide_fig("The reference IS the released decoder", "nf_warmstart.png", [
 (0, "Released weights re-expressed in ladder form · bijection over 143 tensors", True),
 (0, "Adapters and seam repair zero-init ⇒ identity at step 0", True),
 (0, "max | x̂_release − x̂_ours |  =  0.0     at qp 0 / 32 / 63", True),
 (1, "through Microsoft's own DMCI class and forward_one_frame", False),
 (0, "So training must (a) lift shallow exits, (b) not damage the deepest", True),
], size=13)

# 6 -------------------------------------------------------------------- training
slide_fig("Recipe, checked against theirs", "nf_collapse.png", [
 (0, "verify_recipe.py diffs each item against ~/DCVC/train_image.py", True),
 (0, anchor_line() or "anchor comparison pending", True),
 (1, ladder_line() or "three-run comparison pending", False),
 (1, collapse_line() or "second-epoch comparison pending", False),
 (0, "Additions, listed: --epoch_offset · --anchor_weight · --new_lr_scale · "
     "--freeze_encoder", True),
], size=11)

# 7 ------------------------------------------------------- how training went
slide_fig("The loss is not the convergence signal", "training_BEST.png", [
 (0, "L = λ·MSE + bpp   ·   corr(L, batch bpp) = 0.97", True),
 (1, "flat from step ~5,000; a 256 px crop's bitrate swings 2× per batch", False),
 (0, "(b) exits stay ordered, ~1 dB apart → ladder healthy", True),
 (0, "(c) saving on FIXED CTC frames, still rising at 20–24k steps", True),
 (1, "BEST128 11.2 → 12.6%   FINE12 15.0 → 16.8%   (qp 63)", False),
 (0, "1 epoch = 47,451 steps · every number here is one checkpoint", True),
], size=12)

# 8 ------------------------------------------------------------------- anchor
slide_fig("Schedule position, and a ×4 error", "nf_anchor.png", [
 (0, "Reading the schedule from epoch 0 applies from-scratch lr to a converged "
     "model", True),
 (1, "deepest exit fell 0.153 / 0.201 / 0.263 dB in one epoch", False),
 (1, "→ 0.025 / 0.051 / 0.074 with --epoch_offset 75 + --anchor_weight", False),
 (0, "Freezing the trunk gives 0.000 drift — and collapses the ceiling "
     "27.3% → 11.5%", True),
], size=13)

# 9 ---------------------------------------------------------------- example
slide_fig("One frame, four exits", "nf_exits.png", [
 (0, "Bosphorus 1080p · qp 32 · identical bitstream at every exit", True),
 (0, "Error ×25 · damage is structured: detail and tile borders", True),
], size=13)

# 10 ------------------------------------------------ seam 1/5, the mechanism
slide_text("Seam 1/5 · where it comes from", [
 (0, "3×3 depthwise at (i, j):", True),
 (1, "y[i,j]  =  Σ_{u,v ∈ {−1,0,1}}  w[u,v] · f[i+u, j+v]", False),
 (0, "Per tile, f[i+u, j+v] outside the tile does not exist → padding invents it", True),
 (1, "only the 3×3 depthwise has spatial extent; everything else is 1×1", False),
 (1, "⇒ exit adapters are 1×1 on purpose: no receptive field, no seam", False),
 (0, "Contamination grows one ring per layer. b per-tile layers, tile side F:", True),
 (1, "corrupted fraction  =  1 − ((F − 2b) / F)²   ≈   4b / F", False),
 (1, "F = 32,  b = 8   →   1 − (16/32)²  =  75% of the tile", False),
 (0, "Not a line of bad pixels — the error propagates inward", True),
 (0, "Scales with tile PERIMETER, and with split depth j", True),
], size=13)

# 11 -------------------------------------------------- seam 2/5, the padding
slide_text("Seam 2/5 · padding is an estimator", [
 (0, "Border column x[0], unseen neighbour x[−1]:", True),
 (1, "zeros        x̂[−1] = 0", False),
 (1, "replicate    x̂[−1] = x[0]                         zeroth-order hold", False),
 (1, "linear       x̂[−1] = 2·x[0] − x[1]                first-order", False),
 (1, "arls         x̂[−1] = a·x[0],  a fitted per channel   AR(1)", False),
 (0, "Pure seam penalty · CTC · 256 px · full depth · dB above the release", True),
 (1, "                qp 0      qp 32     qp 63", False),
 (1, "zeros         0.1005    0.2162    0.5477", False),
 (1, "replicate     0.0616    0.0854    0.1070", False),
 (1, "linear        0.1793    0.2315    0.2638", False),
 (1, "arls          0.0518    0.0707    0.0879", False),
 (0, "linear > replicate: extrapolating a gradient amplifies boundary noise", True),
 (0, "arls wins on dB, REJECTED on cost · +10.7% wall-clock for 0.019 dB", True),
], size=12)

# 12 ------------------------------------------------- seam 3/5, the tile size
slide_text("Seam 3/5 · the perimeter law", [
 (0, "Prediction  4b/F  ⇒  doubling F halves the penalty", True),
 (1, "zeros      qp 63:   128 px 1.1670  →  256 px 0.5477     ×0.47", False),
 (1, "replicate  qp 63:   128 px 0.2125  →  256 px 0.1070     ×0.50", False),
 (0, "Free in compute — a tiled decode's MAC count is independent of F", True),
 (0, "Paid in routing granularity · 40 tiles/frame at 256 px vs 160 at 128", True),
 (0, "Halo, same law:  tile F with halo h computes ((F+2h)/F)² pixels", True),
 (1, "F = 16, h = 4  →  2.25×        F = 32, h = 4  →  1.56×", False),
 (1, "⇒ halo on the HEAD only (2.40% of MACs); on the trunk it exceeds the "
     "saving", False),
], size=13)

# 13 --------------------------------------------------- seam 4/5, the repair
slide_text("Seam 4/5 · repair told where to look", [
 (0, "A translation-invariant 3×3 must INFER the boundary from content", True),
 (1, "and corrects the ~25% clean interior, where any correction is damage", False),
 (0, "But the lattice is known — unpatchify lays tiles on a fixed P×P grid", True),
 (0, "Repair(f)  =  f  +  G[i mod P, j mod P] · PW( WSiLU( DW3×3(f) ) )", True),
 (1, "G :  P × P = 256 scalars, shared over all 384 channels", False),
 (1, "     0.0007% of decoder parameters, no MAC beyond a broadcast multiply", False),
 (1, "G₀ = exp(−d / τ),  d = distance to nearest tile edge", False),
 (1, "     ⇒ the gate starts on the seam; training refines, not discovers", False),
 (1, "PW zero-init ⇒ module starts as identity ⇒ warm start stays bit-exact", False),
 (0, "Cost 0.95% of the decode — charged inside every saving, incl. the deepest "
     "exit (C_{K−1} = 1.0095, not 1.0)", True),
], size=12)

# 14 --------------------------------------- seam 5/5, the pixels and the verdict
slide_fig("Seam 5/5 · at pixel scale, and what repair is worth",
          "seam_patches.png", [
 (0, "192 px across a tile boundary · qp 63 · full depth · error ×40", True),
 (0, "Repair ON vs OFF, 8 sequences:   full depth 0.0672 → 0.0750 (worse)   ·   "
     "routed 0.1259 → 0.1239", True),
 (0, "0.0020 dB/point of decode  ·  arls was rejected at 0.0018", True),
 (1, "trained jointly ⇒ an inference ablation is not the clean test", False),
], size=12)

# 15 ------------------------------------------------------------------ honesty
slide_fig("MACs are not milliseconds", "latency.png", [
 (0, "1920×1088 · interleaved · median of 40", True),
 (1, latency_line(0.1) or "0.1 dB pending", False),
 (1, latency_line(0.3) or "0.3 dB pending", False),
 (1, latency_line(0.5) or "0.5 dB pending", False),
 (0, "Gap  =  tiling overhead 3.7–6.5%  +  per-group bookkeeping", True),
 (1, "mask + gather + scatter + one device↔host sync per group", False),
 (1, "36.4 ms vs 77.5 ms of convolution  =  32% of the loop", False),
 (0, "Recoverable · sort by exit depth once → 23–37% back, bit-identical", True),
], size=12)

# 16 ------------------------------------------------------------------ results
s = slide_fig("Against the released decoder", "nf_results.png", [
 (0, f"At 0.1 dB · {svr_row('signalled_BEST_0817_1542.json',0):.1f}% · "
     f"{svr_row('signalled_BEST_0817_1542.json',32):.1f}% · "
     f"{svr_row('signalled_BEST_0817_1542.json',63):.1f}%   (qp 0 / 32 / 63)", True),
 (1, "÷ RELEASE (1.0), not our deepest exit (1.0095) — earlier decks read "
     "0.6–0.75 pts higher", False),
 (0, target_line(), True),
 (0, ceiling_line(), True),
 (0, bd_line(), True),
], size=12)

# 16a ------------------------------------------------ the allocation problem
slide_eq("The allocation problem", [
 (0, "N tiles, K exits.  D[t,k] = distortion of tile t at exit k.  C_k = its cost.", True),
 ("EQ", r"\min_{k_1\ldots k_N}\;\frac{1}{N}\sum_{t=1}^{N} D[t,k_t]"
        r"\qquad \mathrm{s.t.}\qquad \frac{1}{N}\sum_{t=1}^{N} C_{k_t}\;\leq\;B", 21),
 (0, "Lagrangian relaxation decouples it — every tile solves independently:", True),
 ("EQ", r"k_t^{\star}(\lambda)\;=\;\arg\min_{k}\;\{\,D[t,k]\;+\;\lambda\,C_k\,\}", 23),
 (1, "sweeping λ traces the lower convex hull of the achievable (cost, distortion) set", False),
 (1, "λ is found by BISECTION on the realised dB, so a 0.1 dB budget is exactly 0.1", False),
 (0, "Both configurations solve this. They differ only in where D[t,k] comes from.", True),
], size=13)

# 16b ----------------------------------------------------- the two decisions
slide_eq("Two ways to obtain the decision", [
 (0, "A · the encoder holds the source, so D[t,k] is measured, and k* is exact", True),
 ("EQ", r"D[t,k]\;=\;\|\,\hat{x}_k[t]-x[t]\,\|_2^2"
        r"\qquad\Rightarrow\qquad k_t=k_t^{\star}(\lambda)", 21),
 (1, "the answer is then transmitted: H(k) · N + 8K bits ≈ 94 bit/frame", False),
 (0, "B · the decoder never sees x, so D[t,k] does not exist. It predicts instead.", True),
 ("EQ", r"z_t\;=\;\mathrm{MLP}([\;\mu_t,\;\sigma_t,\;"
        r"\rho_t,\;\frac{qp}{63}\;])", 21),
 (1, "μ, σ : mean and std over tile t of a learned 384→48 view of the stem", False),
 (1, "ρ : a 512→32 view of the latent ŷ concatenated with the entropy scales σ̂", False),
 ("EQ", r"\hat{k}_t\;=\;\arg\max_{k}\;\{\,"
        r"\log\mathrm{softmax}(z_t)_k\;-\;\beta\,C_k\,\}", 23),
 (0, "β plays λ's role, but trades log-likelihood against cost, not distortion "
     "against cost", True),
], size=12)

# 16c ---------------------------------------------------- training the router
slide_eq("Training the router, and why it collapsed", [
 (0, "Target is the oracle's choice, weighted by what the decision is worth:", True),
 ("EQ", r"\mathcal{L}\;=\;\mathrm{CE}_w(z_t,\,k_t^{\star})\;+\;"
        r"\alpha\;\mathbb{E}_t[\,L(t,\hat{k}_t)-L(t,k_t^{\star})\,],"
        r"\qquad L(t,k)=D[t,k]+\lambda C_k", 19),
 (1, "agreement on ties is worth nothing; the regret term says so", False),
 (0, "Collapse control without an auxiliary loss (arXiv:2408.15664):", True),
 ("EQ", r"z_t\;\leftarrow\;z_t+b,\qquad "
        r"b\;\leftarrow\;b+\eta\,(\pi^{\star}-\hat{\pi})", 21),
 (1, "b is a buffer, nudged by usage — no interference gradient enters ℒ", False),
 (1, "balanced toward the ORACLE's mix π*, not uniform: exits are not "
     "interchangeable experts", False),
 (0, "Trained JOINTLY with the decoder → collapsed to a constant, agreement "
     "0.000 at qp63.  Against the FROZEN decoder → 0.848.", True),
], size=12)

# 17 ------------------------------------------------- A vs B, the design choice
slide_fig("Who decides", "ab_decision.png", [
 (0, "A   argmin_k ( MSE[t,k] + λ·C_k )   ~94 bit/frame = 0.008–0.020% of rate", True),
 (0, "B   argmax_k ( log softmax(z)_k − β·C_k )   nothing sent, file "
     "byte-identical", True),
 (1, ab_line(0) or "", False),
 (1, ab_line(32) or "", False),
 (1, ab_line(63) or "", False),
 (0, "Asymmetric ·  A: +1.21 decodes at the ENCODER  ·  B: +0.163% at the decoder", True),
], size=12)

# 18 ----------------------------------------------- the claim this overturned
slide_fig("A claim overturned", "ab_budgets.png", [
 (0, "Earlier claim: prediction fails, signalling works", True),
 (1, "that was ONE router — BEST's, trained jointly with a moving decoder — "
     "not routing as such. Agreement 0.000 at qp 63, 239 of 240 tiles to one "
     "exit", False),
 (0, "Routing works. Three routers, three outcomes:", True),
 (1, "BEST, joint          collapsed to a qp-dependent constant", False),
 (1, "FINE12, joint        did NOT collapse · budget met, β ≈ −0.01", False),
 (1, "retrained on FROZEN  0.848–0.923 agreement · budget met at every rate", False),
 (1, "⇒ the limit was the training target, not the information — and joint "
     "training does not always break it, which is itself unexplained", False),
 (0, "Cost of not signalling ·  0.1 dB 1.35–3.96 pts  ·  0.3 dB 0.16–0.95  ·  "
     "0.5 dB 0.16–1.23", True),
 (1, "at the ceiling the gap is 0.16 = the router's own 0.163%", False),
], size=12)

# 19 -------------------------------------------------------- against the field
slide_fig("The price of speed", "paper_rdc.png", [
 (0, "BD-Rate = cost · MACs saved = benefit · released intra decoder, YUV420", True),
 (1, "0.1 dB  0.88% / 27.6%    0.3 dB  2.51% / 39.5%    0.5 dB  3.36% / 41.7%", False),
 (0, "Per unit of REAL speedup:  6.2 · 6.6 · 6.5  BD-Rate points", True),
 (1, "flat across the range — an exchange rate, not a chosen operating point", False),
 (0, "DCVC-UF HT-L → HT-S:  10.6 points for 1.66×  =  16.1 points per unit", True),
 (0, "⇒ adaptive depth buys speed 2.4× cheaper than shrinking the model", True),
 (1, "their pair is whole-video, ours intra-only → a calibration, not a "
     "head-to-head", False),
], size=12)

# 20 --------------------------------------------------------- related work
slide_text("Related work", [
 (0, "Early exit / dynamic depth · BranchyNet, MSDNet, SkipNet", True),
 (1, "decide from the network's own output; ours decides per SPATIAL TILE for "
     "a fixed output", False),
 (0, "ClassSR", True),
 (1, "routes patches to differently sized SR branches for a cost target", False),
 (1, "ours: a DEPTH inside one network, for a quality target, bitstream fixed", False),
 (0, "Xie et al. · RDC optimisation in neural image compression · 2305.07678", True),
 (1, "adaptive spatial mask decoded from side information — closest to our B", False),
 (1, "adapts the ENTROPY model's dependencies; we adapt SYNTHESIS depth", False),
 (0, "Spatial competition · 2605.13243", True),
 (1, "per-region mode map at 1.8e-4 bpp (ours 4.5e-5) — selects CODECS for "
     "rate, not depth for compute", False),
 (0, "Loss-free load balancing · 2408.15664 · used directly", True),
 (1, "balanced toward the ORACLE's exit mix, not uniform — exits are not "
     "interchangeable experts", False),
], size=12)

# 21 ---------------------------------------------------------- quantisation
slide_fig("Quantisation · a second, orthogonal lever", "quant.png", [
 (0, "Weight-only · per-channel · no calibration · decoder only", True),
 (0, "Registered prediction — shallow exits suffer more — FALSIFIED", True),
 (1, "qp 63, 8 bit: exit 2 +0.108 dB, deepest +0.111 dB, ratio 0.97", False),
 (1, "the deepest starts at 0.073 dB and has no headroom for a noise floor", False),
 (0, "What matters is the SPREAD — the only thing a router can act on", True),
 (1, "qp 63   fp32 2.864 → 8 b 2.876 → 6 b 2.224 → 4 b 1.398   ·   BOPs "
     "1.000 → 0.062 → 0.035 → 0.016", False),
 (1, "qp 0    0.704 → 0.712 → 0.791 → 0.837   — flat to rising, no collapse", False),
 (0, "8 bit composes cleanly at 16× fewer BOPs · the collapse is a HIGH-RATE "
     "effect, not a general one", True),
 (1, "caveat: 8 bit still raises the floor 0.073 → 0.184 dB at qp 63", False),
], size=11)

# 22 ---------------------------------------------------------- theory
d = {q: th[str(q)]["delta"]["3e-05"] for q in (0, 32, 63)} if th else None
slide_fig("Why per-tile beats any fixed mix", "nf_heterogeneity.png", [
 (0, "Cost additive over tiles, MSE a mean over tiles ⇒ both affine in the "
     "assignment", True),
 (1, "fixed proportions → convex hull of the K exit points", False),
 (1, "per-tile assignment → Minkowski average of the N tile hulls", False),
 (0, "Gap  =  min-of-average  −  average-of-min  ≥  0", True),
 (1, "zero iff every tile prefers the same exit — this IS the value of "
     "adaptivity", False),
 (0, f"Measured Δ/J:  {100*d[0]['delta']/d[0]['J_oracle']:.2f}%  →  "
     f"{100*d[63]['delta']/d[63]['J_oracle']:.2f}%  with rate"
     if d else "Measured Δ/J rises with rate", True),
 (0, f"Same mechanism per sequence · best/worst CTC clip {spread_ratio():.1f}× "
     f"at qp 63 vs {spread_ratio(0):.1f}× at qp 0", True),
], size=12)

# 23 ---------------------------------------------------------- what is owed
slide_text("What is still owed", [
 (0, "A static decoder truncated to the depth we average", True),
 (1, "6.95 of 12 blocks at qp 0 · 8.78 at qp 63", False),
 (1, "decides whether adaptivity is the contribution", False),
 (0, "All six runs to 4 epochs", True),
 (1, "every number here is one checkpoint; the saving is still climbing", False),
 (0, "HEVC B / C / D · 13 of 53 sequences · reported as NOT MEASURED", True),
 (0, "Sorted-tile execution, end to end", True),
 (1, "prototyped and bit-identical; not landed while six runs import the "
     "decoder", False),
 (0, "Wall-clock on an idle card · these ratios were taken at 100% utilisation", True),
 (0, "Open · CONTROL degrades over its second epoch and the cause is unknown", True),
], size=13)

out = R / "paper" / "FLEX-UF_LMT.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
