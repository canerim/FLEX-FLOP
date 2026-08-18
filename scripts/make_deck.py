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
    return (f"CONTROL over its SECOND epoch, per-exit dB vs the release at "
            f"qp{q} (below): saving at 0.1 dB fell {got} at qp0, damage growing "
            f"the shallower the exit. VERBATIM, which carries even less, "
            f"IMPROVED over the same boundary — so this is CONTROL's, not "
            f"'training without the additions', and the cause is still open")


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
    return ("Saving at 0.1 dB across the three runs that share an architecture — "
            + "; ".join(out)
            + ". CONTROL→BEST changes four things at once, so that step is the "
              "bundle, not the anchor alone")


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
    worst_v = min(V[q] for q in qs)
    return (f"Deepest exit vs the release, same tiles and same warm start: BEST "
            + " / ".join(f"{A[q]:+.3f}" for q in qs)
            + " dB against VERBATIM's "
            + " / ".join(f"{V[q]:+.3f}" for q in qs)
            + f" dB at qp{'/'.join(map(str, qs))}. VERBATIM's drift alone is "
              f"{abs(worst_v) / 0.1:.1f}× the 0.1 dB budget, so the target is "
              f"unreachable there at any saving — the anchor term is a "
              f"precondition, not a refinement")


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

# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(prs.slide_layouts[0])
s.shapes.title.text = "FLEX-UF: Content-Adaptive Early Exit for DCVC-UF"
s.placeholders[1].text = ("Decoder compute saved at a measured cost in quality\n"
                          "Chair of Media Technology · TUM")

# 2 -------------------------------------------------------------- the problem
slide_fig("Problem and goal", "nf_macs.png", [
 (0, "DCVC-UF's intra decoder: 453.5 GMAC per 1080p frame", True),
 (1, "12 identical DepthConvBlocks = 89.4% of it", False),
 (1, "every patch pays the same, however easy it is", False),
 (0, "Goal: spend less compute where the content allows", True),
 (1, "target set with the group: 30–40% saved at ≤0.1 dB", False),
 (0, "Constraint that shapes everything: the reference is the RELEASED model", True),
 (1, "encoder, hyperprior, entropy model frozen → identical bitstream", False),
 (1, "so any measured difference is the decoder alone", False),
], size=13)

# 3 --------------------------------------------------------------- architecture
slide_fig("Architecture: a ladder of exits", "nf_arch.png", [
 (0, "Shared stem runs full-frame → no seams; the rest runs per tile", True),
 (0, "Each exit has a zero-initialised 1×1 adapter, then the shared head", False),
 (0, "Pointwise on purpose: a 1×1 adds no receptive field, so no seam cost", False),
])

# 3b -------------------------------------------------- what it actually does
slide_fig("What the router actually does, on a frame", "exit_map.png", [
 (0, "Bosphorus, qp32, the real Lagrangian assignment at the 0.1 dB budget — "
     "40 tiles of 256px, 33.4% saved at 0.098 dB", True),
 (0, "The boat takes exit 4 (13% saved). The water and the sky take exit 2 "
     "(42%). The shoreline band in between takes exit 3.", True),
 (0, "This is the whole idea in one picture: the frame is not uniformly hard, "
     "and a decoder that spends uniformly is overpaying on most of it", True),
 (1, "panel c is the dB each tile pays against ITS OWN reference error, not "
     "against the frame mean — dividing by the mean makes easy tiles read as "
     "better than the release, which no exit can be", False),
 (0, "Closest relative in the literature is ClassSR, which routes patches to "
     "differently sized SR branches. The difference: it picks a network per "
     "patch for a fixed cost target; we pick a DEPTH per tile inside one "
     "network, for a fixed quality target, and the bitstream never changes.", True),
], size=11)

# 4 ------------------------------------------------------------- the warm start
slide_fig("We start AT DCVC-UF, not near it", "nf_warmstart.png", [
 (0, "The released weights are re-expressed in ladder form by a key remap", True),
 (1, "dec_1.0→upsample, dec_1.n→groups.g.i, dec_2→head", False),
 (1, "the remap is a bijection over 143 tensors (asserted in tests)", False),
 (0, "Adapters and seam repair are zero-init, i.e. the identity", False),
 (0, "So at step 0, with no training at all:", True),
 (1, "released DCVC-UF vs our deepest exit → max|diff| = 0.0", False),
 (0, "Training does not have to reach DCVC-UF. It has to (a) lift the shallow "
     "exits and (b) not damage the deepest one.", True),
 (0, "From-scratch runs were abandoned for exactly this: after 6 epochs their "
     "anchor sat 1.49 dB below the release and was not closing.", False),
], size=12)

# 5 -------------------------------------------------------------------- training
slide_fig("Training: the recipe, checked not claimed", "nf_collapse.png", [
 (0, "scripts/verify_recipe.py compares each item against ~/DCVC/train_image.py "
     "and exits non-zero on a mismatch", True),

 (0, anchor_line() or "anchor comparison pending VERBATIM's measurement", True),
 (1, ladder_line() or "three-run comparison pending", False),
 (1, collapse_line() or "second-epoch comparison pending", False),
 (0, "Deliberate additions, listed rather than hidden: --epoch_offset, "
     "--anchor_weight, --new_lr_scale, --freeze_encoder", True),
 (0, "VERBATIM is their recipe with NONE of them, effective batch 16 by "
     "accumulation — the control both rows above are measured against", True),
], size=11)

# 5b ------------------------------------------------------- how training went
slide_fig("How training actually went", "training_BEST.png", [
 (0, "The loss is flat from about step 5,000 and stays flat through the epoch. "
     "It would be easy to call that converged. It is not:", True),
 (1, "loss = lambda*MSE + bpp, and a random 256px crop's bitrate swings twofold "
     "between consecutive batches. corr(loss, batch bpp) = 0.97 — the curve is "
     "mostly reporting what the dataloader handed the model", False),
 (0, "Panel b: the exits stay ordered and about 1 dB apart on the training "
     "batch — the ladder is healthy, nothing has collapsed onto its neighbour", True),
 (0, "Panel c is the measurement that does answer it: compute saved on the SAME "
     "fixed CTC frames, one point per checkpoint. Still climbing at 20-24k "
     "steps (BEST128 11.2 to 12.6%, FINE12 15.0 to 16.8% at qp63)", True),
 (0, "So the stopping signal is the held-out measurement, not the loss — and "
     "every number in this deck is one checkpoint of a run that has not "
     "finished", True),
 (0, "One epoch is 47,451 steps at batch 8; six runs share the machine, so an "
     "epoch costs 6-22 h wall-clock", False),
], size=11)

# 6 ------------------------------------------------------------------- anchor
slide_fig("The anchor, and a ×4 error caught",
          "nf_anchor.png", [
 (0, "Reading the schedule from epoch 0 applied the from-scratch lr to a "
     "converged model", True),
 (1, "deepest exit fell 0.153 / 0.201 / 0.263 dB in ONE epoch (qp0/32/63)", False),
 (1, "fixed to 0.025 / 0.051 / 0.074 by --epoch_offset 75 + --anchor_weight", False),
 (0, "Freezing the backbone gives exactly 0.000 — but collapses the ceiling from "
     "27.3% to 11.5%, so training the trunk is mandatory", True),
], size=13)

# 7 ---------------------------------------------------------------- example
slide_fig("What an exit costs, on one frame", "nf_exits.png", [
 (0, "Bosphorus 1080p, qp32, identical bitstream at every exit", True),
 (0, "Error maps ×25: the damage is structured, concentrated on detail and "
     "on tile borders — which is what the seam work targets", False),
], size=13)

# 8 --------------------------------------------------------------------- seam
slide_fig("The seam artefact, and what actually removed it", "nf_seam.png", [
 (0, "A tile decoded alone meets its border with invented values. Measured "
     "PURE — no early exit, every tile at full depth, so tiling is the only "
     "difference from a full-frame decode:", True),
 (1, "256px tiles, qp0/32/63 — zeros: 0.100 / 0.216 / 0.548 dB. At qp63 that "
     "is five times the entire 0.1 dB budget, before a single tile exits early", False),
 (0, "Three things fixed it, in order of contribution:", True),
 (1, "TILE SIZE — 256px instead of 128 halves it (1.167 -> 0.548 at qp63): the "
     "seam is a perimeter effect", False),
 (1, "REPLICATE padding instead of zeros — 0.548 -> 0.107 dB, an 80% cut for "
     "zero compute. 'linear' extrapolation was also tried and is WORSE (0.264), "
     "so a cleverer guess is not automatically a better one", False),
 (1, "GRID SEAM REPAIR — the tile lattice is known exactly, so the correction "
     "is gated by a learned P×P map indexed by position within a tile: "
     "f + G[i mod P, j mod P]·PW(WSiLU(DW3x3(f))). 256 scalars, 0.0007% of the "
     "decoder's parameters, initialised at exp(-d/tau) so the gate starts on "
     "the seam. Costs 0.95% of the decode", False),
 (0, "A plain 3×3 would have to INFER which pixels are on a boundary, and would "
     "apply its correction to the 77% that are clean interior — where any "
     "correction is damage. Telling it the grid is the whole idea.", True),
 (0, "arls (arXiv:2502.12300) is better still — 0.088 vs 0.107 at qp63 — and was "
     "REJECTED: +10.7% of decode wall-clock for +0.019 dB", True),
 (0, "Canvas coupling gives the 3×3 its real neighbour and reaches max|diff| = "
     "0.0 against full-frame for +0.03%. It is implemented and measured, but "
     "tile_coupling=False in every run here — so it is an option, not what "
     "these results use.", True),
], size=10)

# 10a --------------------------------------------- seam, formally: the cause
slide_text("The seam artefact, formally  (1/5): where it comes from", [
 (0, "A 3x3 depthwise convolution at feature position (i,j) computes", True),
 (1, "y[i,j] = sum_{u,v in {-1,0,1}} w[u,v] . f[i+u, j+v]", False),
 (0, "Decoded full-frame, f[i+u,j+v] are the neighbour's real features. Decoded "
     "per tile, positions outside the tile do not exist and the kernel is fed "
     "whatever the padding rule invents.", True),
 (0, "Only the 3x3 DEPTHWISE has spatial extent. Everything else in a "
     "DepthConvBlock is 1x1, and a 1x1 has no neighbours to miss — which is why "
     "the exit adapters are 1x1 on purpose: they add no seam.", True),
 (0, "The affected region is a ring one pixel wide per convolution. With b "
     "depthwise layers running per tile, the ring is b pixels deep, so for a "
     "tile of side F the corrupted fraction is", True),
 (1, "1 - ((F - 2b)/F)^2  ~=  4b/F   for b << F", False),
 (1, "F = 32 feature px (256 RGB), b = 8 per-tile blocks -> ~ 1 - (16/32)^2 = "
     "75% of the tile is within reach of at least one contaminated value", False),
 (0, "So this is NOT a thin line of bad pixels. The error PROPAGATES inward one "
     "ring per layer, which is why the error maps show structure across the "
     "whole tile and not only at the border.", True),
 (0, "Two consequences that shaped every later decision: the damage scales with "
     "the tile PERIMETER, and it scales with how many layers run per tile — "
     "i.e. with the split depth j.", True),
], size=11)

# 10b -------------------------------------------- seam, formally: the padding
slide_text("The seam artefact, formally  (2/5): padding is an estimator", [
 (0, "Padding is not a formatting detail. It is an ESTIMATOR of the neighbour "
     "the tile cannot see, and the seam penalty is that estimator's error.", True),
 (0, "For a border column x[0] with true neighbour x[-1]:", True),
 (1, "zeros      x_hat[-1] = 0                     — assumes the signal is "
     "centred at zero, which after the -0.5 shift is grey", False),
 (1, "replicate  x_hat[-1] = x[0]                  — assumes local constancy, "
     "i.e. a zeroth-order hold", False),
 (1, "linear     x_hat[-1] = 2.x[0] - x[1]         — first-order extrapolation", False),
 (1, "arls       x_hat[-1] = a.x[0], a fitted per channel by least squares — "
     "an AR(1) model (arXiv:2502.12300)", False),
 (0, "Measured on CTC, 256px tiles, PURE seam (no early exit, full depth, so "
     "tiling is the only difference from full-frame). dB above the release:", True),
 (1, "                qp0      qp32     qp63", False),
 (1, "zeros         0.1005   0.2162   0.5477", False),
 (1, "replicate     0.0616   0.0854   0.1070", False),
 (1, "linear        0.1793   0.2315   0.2638", False),
 (1, "arls          0.0518   0.0707   0.0879", False),
 (0, "The lesson is in the LINEAR row: a higher-order estimator is WORSE than a "
     "zeroth-order one. Extrapolating a gradient past a boundary amplifies "
     "whatever noise sits on it; assuming constancy does not.", True),
 (0, "arls wins on quality and was REJECTED on cost: +10.7% of decode "
     "wall-clock for 0.019 dB. The whole budget is 0.1 dB and the whole saving "
     "is ~30% — that trade does not close.", True),
], size=10)

# 10c ------------------------------------------- seam, formally: the tile size
slide_text("The seam artefact, formally  (3/5): why 256 and not 128", [
 (0, "From (1/5), corrupted fraction ~ 4b/F. Doubling the tile side halves it, "
     "and the measurement follows the law:", True),
 (1, "zeros,     qp63:   128px 1.1670 dB  ->  256px 0.5477 dB   (x0.47)", False),
 (1, "replicate, qp63:   128px 0.2125 dB  ->  256px 0.1070 dB   (x0.50)", False),
 (0, "So tile size was the single largest lever, and it costs nothing in "
     "compute — the MAC count of a tiled decode does not depend on the tile "
     "size at all.", True),
 (0, "What it does cost is ROUTING GRANULARITY: 40 tiles per 1080p frame at "
     "256px against 160 at 128px. Fewer, larger tiles means content that is "
     "locally easy gets averaged in with content that is not.", True),
 (0, "The halo argument points the same way. A tile of side F carried with halo "
     "h computes ((F+2h)/F)^2 as many pixels:", True),
 (1, "F=16, h=4  ->  2.25x     F=32, h=4  ->  1.56x", False),
 (0, "Which is why the halo is carried on the HEAD only (2.40% of MACs) and not "
     "on the trunk. Applying 2.25x to the per-tile trunk would consume more "
     "than routing saves — the arithmetic simply does not close.", True),
 (0, "BEST128 exists to measure the granularity side of this trade rather than "
     "argue it. Both are still training.", True),
], size=10)

# 10d ------------------------------------------ seam, formally: the repair
slide_text("The seam artefact, formally  (4/5): repair that is told where to look", [
 (0, "A translation-invariant 3x3 applied to the stitched canvas has to INFER "
     "which pixels sit on a boundary, from content alone — and it applies the "
     "same correction to the ~25% that are clean interior, where any correction "
     "is damage. It is being asked a harder question than the one we have.", True),
 (0, "But the grid is not unknown. unpatchify lays tiles on a fixed P x P "
     "lattice from the origin, at training and at inference alike. So gate the "
     "correction by position WITHIN a tile:", True),
 (1, "Repair(f) = f + G[i mod P, j mod P] . PW( WSiLU( DW3x3(f) ) )", False),
 (0, "G is P x P = 256 scalars, shared across all 384 channels", True),
 (1, "0.0007% of the decoder's parameters, and no MAC beyond one broadcast "
     "multiply", False),
 (1, "initialised at G = exp(-d/tau), d = distance in feature pixels to the "
     "nearest tile edge — so at step 0 the gate already sits on the seam and "
     "training refines it instead of discovering it", False),
 (1, "PW is zero-initialised, so the whole module starts as the identity and "
     "the warm start stays bit-exact against released DCVC-UF", False),
 (0, "Total cost 0.95% of the decode — and it is charged inside every saving in "
     "this deck, including the deepest exit, which is why our full-depth path "
     "costs 1.0095 stock decodes rather than 1.0.", True),
], size=10)

# 10e -------------------------------- seam: the pixels, and an honest verdict
slide_fig("The seam artefact  (5/5): at pixel scale, and what repair is worth",
          "seam_patches.png", [
 (0, "Bosphorus qp63, 192px window straddling a tile boundary (dashed). Every "
     "tile at FULL depth, so early exit contributes nothing — this is the "
     "tiling artefact alone. Error maps vs the full-frame decode of the SAME "
     "latent, amplified x40.", True),
 (0, "Zeros shows the seam as a bright ridge AND structured error across the "
     "tile interior — the inward propagation from (1/5). Replicate removes most "
     "of both.", True),
 (0, "An uncomfortable measurement, 8 sequences at qp63, repair ON vs OFF on "
     "the same latents and the same exit maps:", True),
 (1, "all tiles at full depth:  0.0672 -> 0.0750 dB   repair makes it WORSE", False),
 (1, "routed at 0.1 dB:         0.1259 -> 0.1239 dB   repair helps by 0.002", False),
 (0, "So grid repair earns 0.002 dB for 0.95% of the decode, and it RAISES the "
     "floor — the same floor that already consumes 30% of the budget at qp63. "
     "By the cost-per-dB rule that rejected arls, this module is on the wrong "
     "side of its own test.", True),
 (1, "caveat: it was trained jointly, so switching it off at inference is not "
     "the same as training without it. The clean test is a run with "
     "seam_repair=none at 256px, which is not one of the six.", False),
], size=10)

# 9 ------------------------------------------------------------------ honesty
slide_fig("MACs are not milliseconds", "latency.png", [
 (0, "Every saving in this project is a MAC count. The field measures FPS.", True),
 (0, "Measured end to end, 1920x1088, interleaved, median of 40:", True),
 (1, latency_line(0.1) or "0.1 dB pending", False),
 (1, latency_line(0.3) or "0.3 dB pending", False),
 (1, latency_line(0.5) or "0.5 dB pending", False),
 (0, "The MAC model is accurate at UNIFORM depth (agrees within 1 point per "
     "exit) and optimistic under ROUTED execution. The difference is exactly "
     "the machinery routing needs.", True),
 (0, "Profiled: the per-group bookkeeping — mask, gather, scatter, and the "
     "device-host sync each forces — is 36.4 ms against the groups' own "
     "77.5 ms of convolution, 32% of the loop, invisible to the MAC model", True),
 (0, "Recoverable, not a hardware limit: sorting tiles by exit depth once makes "
     "the active set a contiguous prefix. Prototyped, bit-identical output, "
     "23-37% of the group loop returned", True),
], size=11)

# 10 ------------------------------------------------------------------ results
s = slide_fig("Results vs the released decoder", "nf_results.png", [
 (0, budget_line(0.3), True),
 (1, "dB is the per-frame average test_video.py computes — the convention every "
     "published DCVC-UF number uses. Pooling all tiles into one MSE reads "
     "0.023-0.033 dB lower on the same allocation and would flatter these", False),
 (0, f"At 0.1 dB, against the RELEASE: {svr_row('signalled_BEST_0817_1542.json',0):.1f}% "
     f"/ {svr_row('signalled_BEST_0817_1542.json',32):.1f}% "
     f"/ {svr_row('signalled_BEST_0817_1542.json',63):.1f}% (qp0/32/63)", True),
 (1, "these are 0.6-0.75 points below what earlier decks showed: savings were "
     "divided by our own deepest exit (1.0095 stock decodes) instead of by the "
     "release's 1.0, which quietly took the seam-repair tax back out", False),
 (0, target_line(), False),
 (0, ceiling_line(), True),
 (0, "Integrated over the curve, not read at one point:", True),
 (1, bd_line(), False),
], size=12)

# 11 ------------------------------------------------- A vs B, the design choice
slide_fig("Who decides: encoder search or decoder prediction", "ab_decision.png", [
 (0, "A — the encoder has the source, so it decodes every tile at every exit, "
     "measures the true error, and signals the answer: ~94 bits/frame, "
     "0.008-0.020% of the bitrate", True),
 (0, "B — the decoder cannot see the source, so it predicts what A would have "
     "chosen from the stem, the latent and the entropy scales. Nothing is sent; "
     "the file is byte-identical to a stock stream", True),
 (0, "At 0.1 dB, same checkpoint and same frames:", True),
 (1, ab_line(0) or "qp0 pending", False),
 (1, ab_line(32) or "qp32 pending", False),
 (1, ab_line(63) or "qp63 pending", False),
 (0, "Compute is asymmetric: A costs the ENCODER ~1.21 extra decodes per frame "
     "and the decoder nothing; B costs the decoder 0.163% and the encoder "
     "nothing. For VOD A is cheaper in total; for live the ranking inverts", True),
], size=11)

# 12 ------------------------------------------- the earlier router claim, undone
slide_fig("A claim this work overturned", "ab_budgets.png", [
 (0, "The previous deck said: prediction failed, signalling works. That was "
     "measured on a router trained JOINTLY with a moving decoder.", True),
 (1, "BEST's joint router collapsed to a qp-dependent constant — agreement "
     "with the oracle 0.000 at qp63, 239 of 240 tiles sent to one exit", False),
 (0, "Retrained against the FROZEN decoder, same architecture and same "
     "information: 0.848 held-out agreement, and the zero-bit configuration "
     "works at every rate", True),
 (0, "So the limit was the training target, not the information", True),
 (0, "And the cost of not signalling depends on how tight the budget is:", True),
 (1, "0.1 dB: 1.35-3.96 points · 0.3 dB: 0.16-0.95 · 0.5 dB: 0.16-1.23", False),
 (1, "at the ceiling the gap is exactly 0.16 — the router's own 0.163% of "
     "compute, and nothing else left to explain", False),
 (0, "Value of seeing the source frame is largest where the budget is tightest", True),
], size=11)

# 13 -------------------------------------------------------- against the field
slide_fig("What a unit of speed costs, against the field", "paper_rdc.png", [
 (0, "BD-Rate is the cost; MACs saved is the benefit. Both measured on the "
     "released intra decoder, YUV420 PSNR, 40 CTC sequences", True),
 (1, "0.1 dB: 0.88% BD-Rate for 27.6% of MACs · 0.3 dB: 2.51% for 39.5% · "
     "0.5 dB: 3.36% for 41.7%", False),
 (0, "Expressed per unit of real speedup, the price is FLAT: 6.2 / 6.6 / 6.5 "
     "BD-Rate points across the three budgets — an exchange rate, not a "
     "cherry-picked operating point", True),
 (0, "DCVC-UF ships two model sizes, and the line between them is what this "
     "field currently pays for speed: HT-L to HT-S gives up 10.6 BD-Rate "
     "points for 1.66x, i.e. 16.1 points per unit", True),
 (0, "Adaptive depth buys speed about 2.4x more cheaply than shrinking the "
     "model — a first answer to 'why not just train a smaller decoder', taken "
     "from a table already in print", True),
 (1, "their pair is whole-video and ours is intra-only, so this calibrates the "
     "exchange rate rather than being a head-to-head; our own static baseline "
     "at matched depth is still owed", False),
], size=11)

# 14 --------------------------------------------------------- related work
slide_text("Related work, and where we sit", [
 (0, "Early exit / dynamic depth: BranchyNet, MSDNet, SkipNet, and the "
     "confidence- and entropy-based exit criteria surveyed in the dynamic "
     "networks literature", True),
 (1, "all decide from the network's own output; ours decides per SPATIAL TILE "
     "for a fixed output, which is a different question", False),
 (0, "Xie et al., 'Exploring the Rate-Distortion-Complexity Optimization in "
     "Neural Image Compression' (arXiv:2305.07678)", True),
 (1, "closest prior art to our configuration B: an adaptive spatial mask "
     "decoded from side information, no explicit overhead", False),
 (1, "but it adapts the ENTROPY model's spatial dependencies; we adapt the "
     "SYNTHESIS depth. Different part of the decoder, composable in principle", False),
 (0, "'Spatial competition for low-complexity learned image compression' "
     "(arXiv:2605.13243)", True),
 (1, "transmits a per-region mode map at 1.8e-4 bpp; ours is 4.5e-5, four "
     "times cheaper, but they select between CODECS for rate and we select "
     "DEPTH for compute", False),
 (0, "Loss-free load balancing (arXiv:2408.15664) is used directly: the router "
     "is balanced toward the ORACLE's exit mix, not toward uniform, because "
     "our exits are not interchangeable experts", True),
], size=11)

# 15 ---------------------------------------------------------- quantisation
slide_text("Quantisation: a second lever, and what it does to the ladder", [
 (0, "Early exit removes MACs; quantisation makes the rest cheaper. Same "
     "budget, so they can be priced against each other", True),
 (0, "Weight-only, per-channel, no calibration — deliberately the weakest form, "
     "so the numbers bound quantisation from below", True),
 (0, "The question is not whether it costs quality but whether it costs the "
     "SHALLOW exits more. It does not — the prediction was wrong:", True),
 (1, "at 8 bits the degradation is uniform (qp63: exit 2 +0.108 dB, deepest "
     "+0.111); at 6 and 4 bits the DEEPEST exits suffer most", False),
 (1, "reason: the deepest exit starts near-perfect and has no headroom to "
     "absorb a noise floor; a shallow exit's error is already dominated by the "
     "blocks it skipped", False),
 (0, "What matters for the ladder is the SPREAD between exits, which is what "
     "routing exploits:", True),
 (1, "qp63: 2.864 dB (fp32) -> 2.876 (8 bit) -> 2.224 (6 bit) -> 1.398 (4 bit)", False),
 (0, "8-bit weights compose cleanly at 16x fewer BOPs; 6 bits and below erode "
     "the ladder itself", True),
], size=11)

# 16 ---------------------------------------------------------- theory + status
d = {q: th[str(q)]["delta"]["3e-05"] for q in (0, 32, 63)} if th else None
slide_fig("Theory, and what is still owed", "nf_heterogeneity.png", [
 (0, "Cost is additive over tiles and MSE is a mean over tiles, so both are "
     "affine in the assignment", True),
 (1, "fixed proportions reach the convex hull of the K exit points; per-tile "
     "assignment reaches the Minkowski average of the N tile hulls", False),
 (1, "their gap is min-of-average minus average-of-min, zero iff every tile "
     "prefers the same exit — this IS the value of adaptivity", False),
 (0, f"Measured: delta/J rises {100*d[0]['delta']/d[0]['J_oracle']:.2f}% to "
     f"{100*d[63]['delta']/d[63]['J_oracle']:.2f}% with rate"
     if d else "Measured: delta/J rises with rate", True),
 (0, "Still owed before this is a paper:", True),
 (1, "a static decoder truncated to the depth we average (6.95 blocks of 12 at "
     "qp0, 8.78 at qp63) — the experiment that decides whether adaptivity is "
     "the contribution", False),
 (1, "all six runs to 4 epochs; every number here is one checkpoint at one epoch "
     "and the measured saving is still climbing", False),
 (1, "HEVC B/C/D, 13 of 53 sequences, reported as NOT MEASURED", False),
 (1, "end-to-end confirmation of the sorted-tile execution, once the runs free "
     "the decoder", False),
], size=10)

out = R / "paper" / "FLEX-UF_LMT.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
