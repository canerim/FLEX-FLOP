"""Every slide figure, in English, to Nature's conventions.

Numbers are read from results/*.json so a figure cannot disagree with the
experiment it reports. Style comes from scripts/naturestyle.py.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import naturestyle as ns
ns.apply()
R = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------- denominator
def _D():
    """Our deepest exit's cost in stock decodes -- 1.0095, not 1.0.

    Curves divide their saving by this, which answers "what does early exiting
    save against our own full-depth path". Every caption in this deck says
    "against the released decoder", whose denominator is exactly 1. The two
    differ by 0.6-0.75 points, and a figure showing one while the text quotes
    the other is the kind of inconsistency a reader is right to distrust.
    """
    import json as _j
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    cfg = FlexUFConfig(**_j.loads((R / "runs/BEST/meta.json").read_text())["config"])
    return float(exit_costs(cfg, "head")[-1])


_DEEP = None


def sv(row_or_value):
    """Saving against the RELEASE, from a row dict or a bare percentage."""
    global _DEEP
    if _DEEP is None:
        _DEEP = _D()
    if isinstance(row_or_value, dict):
        v = row_or_value.get("saving_pct_vs_release")
        if v is not None:
            return v
        v = row_or_value.get("saving_pct")
    else:
        v = row_or_value
    return None if v is None else 100.0 - (100.0 - v) * _DEEP


def J(p):
    f = R / "results" / p
    return json.loads(f.read_text()) if f.exists() else None
def save(fig, name):
    fig.savefig(R / "results" / name, dpi=300, bbox_inches="tight")
    print(f"  {name}")

# a — where the decode's compute sits, b — what causes the seam
fig, (a, b) = plt.subplots(1, 2, figsize=(ns.W2, 1.55))
a.barh([0], [8.16], color=ns.BLUE); a.barh([0], [89.44], left=8.16, color=ns.ORANGE)
a.barh([0], [2.40], left=97.6, color=ns.GREEN)
a.text(52.9, 0, "12 trunk blocks  89.4%", ha="center", va="center", color="white", fontsize=6.5)
a.text(4.1, -0.45, "upsample 8.2%", ha="center", fontsize=6, color=ns.INK2)
a.text(98.8, -0.45, "head 2.4%", ha="right", fontsize=6, color=ns.INK2)
a.set_xlim(0, 100); a.set_ylim(-0.8, 0.5); a.set_yticks([]); a.grid(False)
a.set_xlabel("Share of decode (%)"); ns.panel(a, "a", dx=-0.05, dy=1.28)
a.set_title("453.5 GMAC per 1080p frame", loc="left", pad=3)
b.barh([1, 0], [99.66, 0.34], color=[ns.BLUE, ns.VERM], height=.55)
b.set_yticks([0, 1]); b.set_yticklabels(["3×3 depthwise", "1×1 pointwise"])
b.set_xlim(0, 100); b.set_xlabel("Share of one DepthConvBlock (%)")
b.text(3, 0, "0.34% — the sole cause of the seam", va="center", fontsize=6, color=ns.VERM)
ns.panel(b, "b", dx=-0.28, dy=1.28)
b.set_title("Within a block", loc="left", pad=3)
save(fig, "nf_macs.png")

# warm start
#
# This must come from the WARM START, not from a trained checkpoint. The axis
# says "no training" and the slide's whole claim is that the ladder begins at
# the released decoder -- so plotting a trained run's per-exit dB here was
# mislabelled, and it had been since the figure was written. Switching why_qp
# to BEST made the numbers further from the truth rather than introducing the
# error.
w = J("why_qp_warmstart.json") or J("why_qp.json")
if not J("why_qp_warmstart.json"):
    print("  WARNING: nf_warmstart falls back to a TRAINED checkpoint; the "
          "'no training' label is then wrong. Run scripts/why_qp.py on "
          "runs/warmstart/ckpt_warmstart.pth.tar --out results/why_qp_warmstart.json")
r0 = [x for x in w["rows"] if x["qp"] == 0][0]
fig, a = plt.subplots(figsize=(ns.W15, 1.5))
# What training has to close, alongside where it starts. The warm start's
# deepest exit is the release exactly; its exit 2 is 2.80 dB behind, and BEST
# brings that to 0.167 -- a 17-fold reduction. Showing the start alone
# understated what training does; showing a trained run and labelling it "no
# training", which this figure did, understated it by nine times.
lab = ["deepest exit\n(step 0)", "exit 4", "exit 3", "exit 2"]
val = [0.0] + [-r0["db_per_exit"][k] for k in (4, 3, 2)]
_bt = J("why_qp.json")
_tr = None
if _bt and J("why_qp_warmstart.json"):
    _t0 = [x for x in _bt["rows"] if x["qp"] == 0][0]
    _tr = [0.0] + [-_t0["db_per_exit"][k] for k in (4, 3, 2)]
h = .34
# No label= on these: matplotlib takes the FIRST colour of a per-bar list for
# the legend swatch, which made both entries green when the bars are grey and
# blue. Explicit handles below.
a.barh([i - h / 2 for i in range(4)], val, h, color=[ns.GREEN] + [ns.INK2] * 3)
if _tr:
    a.barh([i + h / 2 for i in range(4)], _tr, h,
           color=[ns.GREEN] + [ns.BLUE] * 3)
for i, v in enumerate(val):
    # The zero bar has no length, so its annotation goes to the RIGHT of the
    # axis origin where the row is, not floating at the top of the panel.
    a.text(v - .05 if v else .06, i - h / 2 if v else i,
           "max|diff| = 0.0" if v == 0 else f"{v:.3f}",
           va="center", ha="right" if v else "left", fontsize=6,
           color=ns.GREEN if v == 0 else ns.INK2,
           fontweight="bold" if v == 0 else "normal")
if _tr:
    for i, v in enumerate(_tr):
        if v:
            a.text(v - .04, i + h / 2, f"{v:.3f}", va="center", ha="right",
                   fontsize=6, color=ns.BLUE)
a.set_yticks(range(4)); a.set_yticklabels(lab); a.invert_yaxis()
a.set_xlim(-3.35, .75)
a.set_xlabel("dB below released DCVC-UF (qp 0)")
from matplotlib.patches import Patch
a.legend(handles=[Patch(facecolor=ns.INK2, label="warm start, no training"),
                  Patch(facecolor=ns.BLUE, label="BEST, one epoch")],
         # y is inverted, so "lower right" is the exit-2 row -- where the bars
         # are. The empty space is the deepest-exit row on the left.
         loc="upper left", fontsize=6, framealpha=.9)
a.grid(axis="y", visible=False)
save(fig, "nf_warmstart.png")

# schedule
sys.path.insert(0, str(R))
from train_flexuf_image import get_training_strategy
st = get_training_strategy(); lr = [e[1] for e in st]
fig, a = plt.subplots(figsize=(ns.W2, 1.7))
a.step(range(len(lr)), lr, where="post", color=ns.BLUE, lw=1.2)
a.set_yscale("log"); a.set_ylabel("Learning rate"); a.set_xlabel("Epoch in Microsoft's schedule")
a.axvspan(90, 105, color=ns.ORANGE, alpha=.13, lw=0)
a.text(90, 9e-4, " 512×512 phase", ha="left", va="center", fontsize=6, color=ns.ORANGE)
# labels staggered in y: the entry points at 75/90/99 are close enough on a
# 105-epoch axis that side-by-side text overlaps at 5.5 pt.
for x, y, nm, c, ha in ((0, 3.2e-4, "e1–e4 read here\ndrift 0.20 dB", ns.VERM, "left"),
                        (75, 3.2e-4, "BEST, BEST128,\nCONTROL, FINE12", ns.BLUE, "right"),
                        (90, 4.0e-5, "VERBATIM", ns.GREEN, "right"),
                        (99, 4.5e-6, "RECIPE512", ns.ORANGE, "right")):
    a.axvline(x, color=c, ls="--", lw=.7)
    a.annotate(nm, (x, y), fontsize=6, color=c, ha=ha, va="center",
               xytext=(3 if ha == "left" else -3, 0), textcoords="offset points")
a.set_ylim(3e-7, 1.6e-3)
save(fig, "nf_schedule.png")

# anchor
fig, a = plt.subplots(figsize=(ns.W15, 1.6))
xs = np.arange(3); w_ = .26
for off, v, c, l in ((-w_-.01, [-.153, -.201, -.263], ns.VERM, "schedule read from epoch 0"),
                     (0, [-.025, -.051, -.074], ns.ORANGE, "offset 75 + anchor term"),
                     (w_+.01, [0, 0, 0], ns.GREEN, "backbone frozen")):
    a.bar(xs + off, v, w_, color=c, label=l)
    for x, y in zip(xs + off, v):
        if y: a.text(x, y - .006, f"{y:+.3f}", ha="center", va="top", fontsize=6, color=ns.INK2)
a.set_xticks(xs); a.set_xticklabels(["qp 0", "qp 32", "qp 63"]); a.set_ylim(-.30, .05)
a.set_ylabel("dB vs released DCVC-UF"); a.legend(loc="lower left", ncol=1)
save(fig, "nf_anchor.png")

# seam
fig, a = plt.subplots(figsize=(ns.W15, 1.6))
m = ["zeros", "replicate", "linear", "arls", "canvas\ncoupling"]
p128 = [1.1670, .2125, .5021, .1785, 0.0]; p256 = [.5477, .1070, .2638, .0879, 0.0]
xs = np.arange(5); w_ = .38
a.bar(xs - w_/2 - .01, p128, w_, color=ns.BLUE, label="128 px tiles")
a.bar(xs + w_/2 + .01, p256, w_, color=ns.GREEN, label="256 px tiles")
for x, v in zip(xs - w_/2 - .01, p128): a.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=6, color=ns.INK2)
for x, v in zip(xs + w_/2 + .01, p256): a.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=6, color=ns.INK2)
a.annotate("no seam:\nidentical to full-frame", xy=(4, .05), xytext=(3.05, .62),
           fontsize=6, color=ns.GREEN, ha="center",
           arrowprops=dict(arrowstyle="->", color=ns.GREEN, lw=.7))
a.set_xticks(xs); a.set_xticklabels(m); a.set_ylim(0, 1.35)
a.set_ylabel("Seam penalty, qp 63 (dB)"); a.legend()
save(fig, "nf_seam.png")

# MAC vs wall clock
fig, a = plt.subplots(figsize=(ns.W15, 1.6))
ex = ["exit 2", "exit 3", "exit 4", "exit 5"]
mac = [43.4, 28.6, 13.8, 0.0]; wall = [43.6, 28.8, 14.9, 0.5]
xs = np.arange(4); w_ = .38
a.bar(xs - w_/2 - .01, mac, w_, color=ns.BLUE, label="MAC model")
a.bar(xs + w_/2 + .01, wall, w_, color=ns.ORANGE, label="Wall clock, 1080p")
for x, (u, v) in enumerate(zip(mac, wall)):
    a.text(x, max(u, v) + 1.2, f"{v-u:+.1f} pt", ha="center", fontsize=6, color=ns.INK2)
a.set_xticks(xs); a.set_xticklabels(ex); a.set_ylim(0, 50)
a.set_ylabel("Compute saved (%)"); a.legend()
save(fig, "nf_cost.png")

# results: two cuts
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

_c = J(CURVE) or J(BASE_CURVE)
pc = _c["rows"]
qps = sorted({r["qp"] for r in pc})
# Both cuts interpolate along the frontier rather than picking the nearest
# sweep sample. The deck's bullets already interpolate, and reading the figure
# off the grid put the two 1.5 points apart on the SAME slide -- the text said
# 33% at 0.1 dB where the panel showed 29%, purely because the sweep had no
# sample sitting on the budget.
def _front(q, key="db_vs_uf_per_frame"):
    # Per-frame decibels: the convention published DCVC-UF numbers use. See
    # scripts/db_convention.py -- pooling reads a quarter to a third of a
    # 0.1 dB budget lower on an identical allocation.
    best = {}
    for r in pc:
        if r["qp"] != q:
            continue
        k = round(sv(r), 6)
        v = r.get(key, r["db_vs_uf"])
        if k not in best or v < best[k]:
            best[k] = v
    pts = sorted(best.items())
    return [p[0] for p in pts], [p[1] for p in pts]

def db_at(q, s):
    S, D = _front(q)
    return float(np.interp(s, S, D, left=np.nan, right=np.nan))

def sv_at(q, d):
    S, D = _front(q)
    return float(np.interp(d, D, S, left=np.nan, right=np.nan))
fig, (a, b) = plt.subplots(1, 2, figsize=(ns.W2, 2.0))
for c, s in zip([ns.BLUE, ns.GREEN, ns.ORANGE], (15, 20, 30)):
    a.plot(qps, [db_at(q, s) for q in qps], color=c, marker="o", label=f"{s}% saved")
a.axhline(.1, color=ns.INK2, lw=.6, ls=(0, (3, 2)))
a.text(63, .105, "0.1 dB", ha="right", fontsize=6, color=ns.INK2)
a.set_xticks(qps); a.set_xlabel("QP (low = low bitrate)")
a.set_ylabel("dB below released DCVC-UF"); a.legend(loc="upper left")
ns.panel(a, "a"); a.set_title("Fixed saving → quality cost", loc="left", pad=3)
# Budgets every rate's frontier actually spans, computed rather than fixed.
#
# 0.1/0.2/0.3 dB was fine for the baseline. BEST's whole qp0 frontier spans
# 0.197 dB, so its 0.3 dB point does not exist and the curve simply started at
# qp16 -- which looks like lost data rather than a curve running off the end.
_top = min(max(_front(q)[1]) for q in qps)
_budgets = [round(x, 3) for x in (0.05, _top / 2, _top * 0.95)]
for c, d in zip([ns.PURPLE, ns.VERM, ns.BLUE], _budgets):
    b.plot(qps, [sv_at(q, d) for q in qps], color=c, marker="s",
           label=f"{d:.2f} dB budget")
b.set_xticks(qps); b.set_xlabel("QP"); b.set_ylabel("Compute saved (%)")
# Lower left: with the curves interpolated they now run high across the whole
# axis, and an upper-right legend sits on the 0.3 dB line.
b.set_ylim(0, 46); b.legend(loc="lower left")
ns.panel(b, "b"); b.set_title("Fixed budget → saving", loc="left", pad=3)
save(fig, "nf_results.png")

# adaptivity gain
th = J("theory_check.json")
fig, a = plt.subplots(figsize=(ns.W15, 1.7))
for lam, c, mk in (("1e-05", ns.INK2, "s"), ("3e-05", ns.BLUE, "o"), ("1e-04", ns.ORANGE, "^")):
    y = [100 * th[str(q)]["delta"][lam]["delta"] / th[str(q)]["delta"][lam]["J_oracle"] for q in qps]
    a.plot(qps, y, color=c, marker=mk, label=f"$\\lambda$ = {lam}")
a.set_xticks(qps); a.set_xlabel("QP"); a.set_ylabel(r"$\Delta / J$  (%)")
a.legend(loc="upper left")
a.annotate("adaptivity is worth\nmore at high rate", (48, 3.52), fontsize=6, color=ns.BLUE,
           xytext=(-72, 10), textcoords="offset points",
           arrowprops=dict(arrowstyle="->", color=ns.BLUE, lw=.7))
save(fig, "nf_theory.png")

# 2x2 isolation
iso = {k: J(f"iso_{k}.json") for k in
       ("A_128w_128t", "B_128w_256t", "C_256w_256t", "D_256w_128t")}
if all(iso.values()):
    fig, a = plt.subplots(figsize=(ns.W15, 1.7))
    order = [("A_128w_128t", "128 px weights\n128 px tiles"), ("B_128w_256t", "128 px weights\n256 px tiles"),
             ("D_256w_128t", "256 px weights\n128 px tiles"), ("C_256w_256t", "256 px weights\n256 px tiles")]
    xs = np.arange(4); w_ = .26
    for off, q, c in ((-w_-.01, 0, ns.BLUE), (0, 32, ns.ORANGE), (w_+.01, 63, ns.GREEN)):
        y = [next(sv(r) for r in iso[k]["rows"] if r["qp"] == q) for k, _ in order]
        a.bar(xs + off, y, w_, color=c, label=f"qp {q}")
        for x, v in zip(xs + off, y): a.text(x, v + .5, f"{v:.1f}", ha="center", fontsize=6, color=ns.INK2)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in order]); a.set_ylim(0, 30)
    a.set_ylabel("Compute saved at ≤0.1 dB (%)"); a.legend(ncol=3, loc="upper right")
    save(fig, "nf_isolation.png")
print("done")

# router: three ways to choose the exit
sg = J(SIGNALLED) or J(BASE_SIGNALLED)
if sg and pc:
    fig, a = plt.subplots(figsize=(ns.W15, 2.0))
    sgm = {r["qp"]: r for r in sg["rows"]}
    for c, q in zip(ns.SERIES, sorted(sgm)):
        # Per-frame, to match the signalled markers plotted on the same axes.
        # These were pooled, which put two conventions on one plot and moved
        # the oracle bound 0.023-0.033 dB left of where the markers live.
        pts = sorted([(r.get("db_vs_uf_per_frame", r["db_vs_uf"]),
                       sv(r)) for r in pc if r["qp"] == q])
        pts = [p for p in pts if -.02 <= p[0] <= .30]
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=.8,
               ls=(0, (4, 2)), alpha=.85)
        a.plot(sgm[q]["db_vs_uf"], sv(sgm[q]), marker="o", ms=4.5, color=c,
               markeredgecolor="white", markeredgewidth=.6)
        a.annotate(f"qp {q}", (sgm[q]["db_vs_uf"], sv(sgm[q])), fontsize=6,
                   color=c, textcoords="offset points", xytext=(5, -2))
    a.plot(0.367, 24.8, marker="X", ms=6, color=ns.VERM, markeredgecolor="white",
           markeredgewidth=.6)
    a.annotate("predicting router (qp 32)\n24.8% for 0.367 dB — off the frontier",
               (0.367, 24.8), xytext=(0.30, 6.0), fontsize=6, color=ns.VERM,
               ha="center", va="center",
               arrowprops=dict(arrowstyle="-", color=ns.VERM, lw=.4,
                               shrinkA=2, shrinkB=3))
    a.axvline(.1, color=ns.INK2, lw=.6, ls=(0, (3, 2)))
    a.text(.105, 1, "0.1 dB", fontsize=6, color=ns.INK2, rotation=90, va="bottom")
    a.set_xlim(-.01, .42); a.set_ylim(0, 34)
    a.set_xlabel("dB below released DCVC-UF"); a.set_ylabel("Compute saved (%)")
    from matplotlib.lines import Line2D
    a.legend(handles=[
        Line2D([], [], color=ns.INK2, ls=(0, (4, 2)), lw=.8, label="oracle bound"),
        Line2D([], [], color=ns.INK2, marker="o", ls="", ms=4, label="signalled system"),
        Line2D([], [], color=ns.VERM, marker="X", ls="", ms=5, label="predicting router"),
    ], loc="upper left")
    save(fig, "nf_router.png")

# Where the tiles actually exit, and why lowering the split depth is the wrong
# lever. The ceiling (all tiles at the shallowest exit) is set by j: ~43% at
# j=2, 58.1% at j=1, 72.9% at j=0. Raising it only helps if the shallowest exit
# is ALREADY saturated -- and it is, at low rate, where 34% of tiles take it.
# At qp 63 only 10% can afford it and the mass sits at exits 4-5, so an even
# cheaper exit would be selected by almost nobody. The gap at high rate is
# shallow-exit QUALITY, not the number of exits below.
pc2 = _c
if pc2:
    fig, a = plt.subplots(figsize=(ns.W15, 1.9))
    qps = [0, 16, 32, 48, 63]
    rows = []
    for q in qps:
        cand = [r for r in pc2["rows"] if r["qp"] == q
                and r.get("db_vs_uf_per_frame", r["db_vs_uf"]) <= 0.105]
        rows.append(max(cand, key=lambda r: sv(r)))
    # Exits 0, 1 and 2 cost the SAME under j=2 -- the first j groups run
    # full-frame for every tile, so the cost vector is 0.5716 three times over.
    # argmin therefore scatters the cheapest allocation across three indices.
    # Plotting index 2 alone showed 29.5% where 33.8% of tiles were actually at
    # that operating point, and the bars summed to 95%.
    #
    # The scatter is not a tie-break artefact: argmin returns 0 or 1 only when
    # that exit's MSE is strictly lower than exit 2's on that tile. On average
    # exit 2 is far better (0.167 dB against 0.409 and 0.606 at qp0), but for
    # about an eighth of the tiles that take the cheapest cost, a SHALLOWER exit
    # reconstructs better. The adapters are trained per exit, so the ladder is
    # monotone in the mean and not tile by tile. It costs nothing -- the three
    # share a cost, and the allocation simply takes whichever wins.
    import sys as _sys
    _sys.path.insert(0, str(R))
    from flexuf.config import FlexUFConfig as _C
    from flexuf.cost import exit_costs as _ec
    _cfg = _C(**json.loads((R / "results" / CURVE).read_text()).get("config", {})) \
        if False else None
    cols = [ns.BLUE, ns.SKY, ns.GREEN, ns.YELLOW, ns.ORANGE, ns.VERM]
    bot = np.zeros(len(qps))
    x = np.arange(len(qps))
    _merged = [0, 1, 2]
    for k in range(2, 6):
        idx = _merged if k == 2 else [k]
        v = np.array([100 * sum(r["hist"][i] for i in idx) / sum(r["hist"])
                      for r in rows])
        a.bar(x, v, .62, bottom=bot, color=cols[k],
              label="exit 2 (cheapest)" if k == 2 else f"exit {k}",
              edgecolor="white", linewidth=.4)
        for xi, (b_, v_) in enumerate(zip(bot, v)):
            if v_ > 6:
                # Okabe-Ito yellow is the only light swatch here; the other
                # three need white text to stay legible.
                a.text(xi, b_ + v_ / 2, f"{v_:.0f}", ha="center", va="center",
                       fontsize=6, color=ns.INK if k == 3 else "white")
        bot += v
    a.set_xticks(x); a.set_xticklabels([f"qp {q}" for q in qps])
    a.set_ylabel("Share of tiles (%)"); a.set_ylim(0, 100)
    a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=4, frameon=False)
    a.set_title("Exit 3 carries the allocation; the cheapest exit collapses with "
                "rate", fontsize=6, color=ns.INK2, loc="left")
    save(fig, "nf_usage.png")

# The trade-off in two integrated numbers, plus what the anchor is worth.
bd = J("bd_saving.json")
if bd:
    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.0))
    rs = bd["rows"]
    x = np.arange(len(rs))
    lab = [f"qp {r['qp']}" for r in rs]
    got = [sv(r["bd_saving_pct"]) for r in rs]
    extra = [(sv(r["bd_saving_pct_zero_drift"]) - sv(r["bd_saving_pct"]))
             if r["bd_saving_pct_zero_drift"] is not None else 0.0 for r in rs]

    ax[0].bar(x, got, .62, color=ns.BLUE, label="measured")
    ax[0].bar(x, extra, .62, bottom=got, color="none", edgecolor=ns.BLUE,
              hatch="////", linewidth=.5, label="if the deepest exit did not drift")
    for xi, (g, e) in enumerate(zip(got, extra)):
        ax[0].text(xi, g / 2, f"{g:.0f}", ha="center", va="center", fontsize=6,
                   color="white")
        if e:
            ax[0].text(xi, g + e + 1.0, f"+{e:.1f}", ha="center", fontsize=6,
                       color=ns.BLUE)
    lo, hi = bd["db_interval"]
    ax[0].set_ylabel("BD-saving (%)")
    ax[0].set_ylim(0, 48)
    ax[0].legend(loc="upper right", fontsize=6)
    ax[0].grid(axis="y", visible=False)

    ax[1].plot(x, [r["bd_quality_db"] for r in rs], marker="o", ms=4,
               color=ns.VERM, lw=1.0)
    for xi, r in enumerate(rs):
        ax[1].annotate(f"{r['bd_quality_db']:.3f}", (xi, r["bd_quality_db"]),
                       fontsize=6, color=ns.VERM, textcoords="offset points",
                       xytext=(0, 6), ha="center")
    slo, shi = bd["saving_interval"]
    ax[1].set_ylabel("BD-quality (dB)")
    ax[1].set_ylim(0, .17)

    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(lab)
    ns.panel(ax[0], "a"); ns.panel(ax[1], "b")
    # Intervals belong in the titles: a BD number without the interval it was
    # integrated over is not a number, and the y-label is not wide enough to
    # carry it without colliding with the panel letter.
    ax[0].set_title(f"What a budget buys — mean saving over dB in "
                    f"[{lo:.3f}, {hi:.3f}]", fontsize=6, color=ns.INK2, loc="left")
    ax[1].set_title(f"What a saving costs — mean dB over saving in "
                    f"[{slo:.0f}%, {shi:.0f}%]", fontsize=6, color=ns.INK2,
                    loc="left")
    fig.tight_layout()
    save(fig, "nf_tradeoff.png")

# Content dependence: the mean is not what a clip gets.
ps = J("per_sequence.json")
if ps:
    fig, a = plt.subplots(figsize=(ns.W15, 2.0))
    rows = ps["rows"]
    xs = np.arange(len(rows))
    for i, r in enumerate(rows):
        v = np.array([sv(q) for q in r["per_sequence"]])
        # Jittered strip, deterministic: index-derived offsets, since the run
        # must reproduce and Math.random-style jitter would not.
        off = (np.arange(len(v)) % 9 - 4) / 22.0
        a.scatter(np.full_like(v, i) + off, v, s=2.5, color=ns.SKY,
                  linewidths=0, alpha=.85, zorder=2)
        a.plot([i - .34, i + .34], [r["median"]] * 2, color=ns.BLUE, lw=1.4,
               zorder=3)
        a.plot([i, i], [r["p25"], r["p75"]], color=ns.BLUE, lw=.7, zorder=3)
        a.annotate(f"{r['max'] / r['min']:.1f}×", (i, 46), fontsize=6,
                   color=ns.VERM, ha="center")
    a.set_xticks(xs); a.set_xticklabels([f"qp {r['qp']}" for r in rows])
    a.set_ylabel("Compute saved at 0.10 dB (%)")
    a.set_ylim(0, 50)
    # "best/worst" belongs in the title: as a standalone label it landed on the
    # first ratio it was meant to explain.
    a.set_title("One dot per CTC sequence; bar is the median, whisker the IQR; "
                "red is best/worst", fontsize=6, color=ns.INK2, loc="left")
    save(fig, "nf_content.png")

# Heterogeneity, at two granularities. Same claim, two independent measurements:
# the theory says the value of adaptivity comes from tiles disagreeing about
# which exit they want, and that disagreement should grow with rate. Panel a is
# that quantity. Panel b is the consequence one level up -- whole sequences
# pulling apart at the same operating point.
th2 = J("theory_check.json")
ps2 = J("per_sequence.json")
if th2 and ps2:
    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.0))
    qs = [r["qp"] for r in ps2["rows"]]

    for c, lam, nm in ((ns.BLUE, "3e-05", "λ = 3×10⁻⁵"),
                       (ns.ORANGE, "1e-05", "λ = 10⁻⁵"),
                       (ns.GREEN, "1e-04", "λ = 10⁻⁴")):
        v = [100 * th2[str(q)]["delta"][lam]["delta"]
             / th2[str(q)]["delta"][lam]["J_oracle"] for q in qs]
        ax[0].plot(range(len(qs)), v, marker="o", ms=3.5, color=c, lw=1.0, label=nm)
    ax[0].set_ylabel("Adaptivity gain Δ/J (%)")
    ax[0].legend(loc="upper left")
    ax[0].set_title("Tiles: value of routing at all", fontsize=6, color=ns.INK2,
                    loc="left")

    for i, r in enumerate(ps2["rows"]):
        v = np.array([sv(q) for q in r["per_sequence"]])
        off = (np.arange(len(v)) % 9 - 4) / 22.0
        ax[1].scatter(np.full_like(v, i) + off, v, s=2.0, color=ns.SKY,
                      linewidths=0, alpha=.85, zorder=2)
        ax[1].plot([i - .34, i + .34], [r["median"]] * 2, color=ns.BLUE, lw=1.3,
                   zorder=3)
        ax[1].plot([i, i], [r["p25"], r["p75"]], color=ns.BLUE, lw=.7, zorder=3)
        ax[1].annotate(f"{r['max'] / r['min']:.1f}×", (i, 46.5), fontsize=6,
                       color=ns.VERM, ha="center")
    ax[1].set_ylabel("Compute saved at 0.10 dB (%)")
    ax[1].set_ylim(0, 51)
    ax[1].set_title("Sequences: what a single clip gets (red = best/worst)",
                    fontsize=6, color=ns.INK2, loc="left")

    for a_ in ax:
        a_.set_xticks(range(len(qs))); a_.set_xticklabels([f"qp {q}" for q in qs])
    ns.panel(ax[0], "a"); ns.panel(ax[1], "b")
    fig.tight_layout()
    save(fig, "nf_heterogeneity.png")

# What a second epoch did to the run without distillation or scaled adapters.
c0, c1 = J("why_qp_CONTROL_ep0.json"), J("why_qp_CONTROL_ep1.json")
if c0 and c1:
    A = {r["qp"]: r["db_per_exit"] for r in c0["rows"]}
    B = {r["qp"]: r["db_per_exit"] for r in c1["rows"]}
    qs = sorted(set(A) & set(B))
    fig, ax = plt.subplots(1, len(qs), figsize=(ns.W2, 1.9), sharey=True)
    ax = np.atleast_1d(ax)
    ks = [2, 3, 4, 5]
    for a_, q in zip(ax, qs):
        x = np.arange(len(ks))
        w = .36
        a_.bar(x - w / 2, [A[q][k] for k in ks], w, color=ns.INK2,
               label="after epoch 0")
        a_.bar(x + w / 2, [B[q][k] for k in ks], w, color=ns.VERM,
               label="after epoch 1")
        a_.set_xticks(x)
        a_.set_xticklabels([f"exit {k}" if k < 5 else "deepest" for k in ks],
                           fontsize=6)
        a_.set_title(f"qp {q}", fontsize=6, color=ns.INK2, loc="left")
        a_.grid(axis="y", visible=False)
    ax[0].set_ylabel("dB below released DCVC-UF")
    ax[0].legend(loc="upper right", fontsize=6)
    # The point is the SHAPE: the deepest bar pair is level, every shallower
    # pair opens up, and it opens wider the shallower it is.
    # Title states the observation, not a cause. The obvious reading -- that
    # the missing additions are responsible -- was tested against VERBATIM,
    # which carries fewer of them and improved instead.
    fig.suptitle("CONTROL: a second epoch degrades every exit except the "
                 "deepest. VERBATIM, with fewer additions, improved — cause open",
                 fontsize=6, color=ns.INK2, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "nf_collapse.png")
