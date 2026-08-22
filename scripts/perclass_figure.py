"""Saving by CTC class, against the one variable that predicts it: tile count.

The headline is a mean over four resolutions, and the mean hides the strongest
dependence in the whole method. A 256 px tile is 40 tiles on a 1080p frame and
2 at 416x240, and with two tiles there is barely an allocation to make.

Operating point is global: lambda is bisected once so the whole test set lands on
the budget, then each class is reported at that lambda -- the deployment case,
not a per-class re-tuning.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply(ns.for_column())
d = json.load(open(R / "results/per_class_RECIPE512.json"))
ORDER = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
COL = {"MCL-JCV": ns.BLUE, "UVG": ns.SKY, "HEVC_B": ns.GREEN,
       "HEVC_E": ns.YELLOW, "HEVC_C": ns.ORANGE, "HEVC_D": ns.VERM}

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.9))

# ---- a: saving vs rate, one line per class, at 0.1 dB ----------------------
a = ax[0]
rows = [r for r in d["rows"] if abs(r["budget_db"] - 0.1) < 1e-9]
qps = sorted(r["qp"] for r in rows)
for c in ORDER:
    ys = []
    for q in qps:
        r = next(x for x in rows if x["qp"] == q)
        ys.append(r["per_class"][c]["saving"] if c in r["per_class"] else np.nan)
    n = rows[0]["per_class"][c]["n"]
    t = rows[0]["per_class"][c]["tiles"]
    a.plot(qps, ys, marker="o", ms=3.5, lw=1.2, color=COL[c],
           # Short in the key, full in the caption. Six full class names in a
           # panel a third of a column wide do not fit at legible type, and
           # the mapping is one line of caption.
           label=c.replace("HEVC_", "").replace("MCL-JCV", "MCL")
                  .replace("_", " "))
a.set_xlabel("qp"); a.set_ylabel("decoder MACs saved (%)")
# Two columns across the top of the panel, with the y limit raised to make
# room for them. Inside the axes at column type the six class names reached
# the y-axis ticks; below the axes they reached the x label.
# No key here. Panel c is a bar per class with the class names along its axis
# and the same colours, so the mapping is already in the figure once; a second
# copy in a panel a third of a column wide is what kept colliding with itself.
a.set_ylim(0, 38)
a.set_title("0.1 dB, global operating point",
            fontsize=ns.fs(6), color=ns.INK2, loc="left")
ns.panel(a, "a")

# ---- b: saving against tile count ------------------------------------------
b = ax[1]
for q, mk in zip(qps, ("o", "s", "^")):
    r = next(x for x in rows if x["qp"] == q)
    xs = [r["per_class"][c]["tiles"] for c in ORDER if c in r["per_class"]]
    ys = [r["per_class"][c]["saving"] for c in ORDER if c in r["per_class"]]
    b.plot(xs, ys, mk, ms=5, ls="none", label=f"qp {q}",
           color={0: ns.BLUE, 32: ns.ORANGE, 63: ns.VERM}[q])
r0 = rows[0]
# Classes that share a tile count land on the same vertical, so their labels
# have to be fanned out. The offsets used to be a hand-written table keyed by
# class name, tuned to where the points sat on an earlier checkpoint; when the
# savings moved, two of them collided again. Computed from the data instead:
# within a group, labels stack downwards in the order of their savings.
_groups = {}
for c in ORDER:
    if c in r0["per_class"]:
        _groups.setdefault(r0["per_class"][c]["tiles"], []).append(c)
_dys = {}
for _x, _cs in _groups.items():
    _cs.sort(key=lambda c: -r0["per_class"][c]["saving"])
    for _i, _c in enumerate(_cs):
        _dys[_c] = 7 - _i * 13
for c in ORDER:
    if c in r0["per_class"]:
        v = r0["per_class"][c]
        b.annotate(c.replace("HEVC_", "").replace("MCL-JCV", "MCL"),
                   (v["tiles"], v["saving"]), fontsize=ns.fs(6), color=ns.INK2,
                   textcoords="offset points", xytext=(4, _dys[c]))
b.set_xscale("log"); b.set_xticks([2, 8, 15, 40])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.set_xlabel("tiles per frame"); b.set_ylabel("MACs saved (%)")
b.legend(loc="lower right", fontsize=ns.fs(6))
# Centred, and the y limit lifted to keep it clear of the highest end-label:
# left-aligned it sat on MCL, which is the top point of the panel.
b.set_title("Saving against tile count",
            fontsize=ns.fs(6), color=ns.INK2, loc="center", pad=9)
ns.panel(b, "b", dx=-0.22)

# ---- c: what each class actually spends ------------------------------------
c_ = ax[2]
r = next(x for x in rows if x["qp"] == qps[0])
names = [c for c in ORDER if c in r["per_class"]]
db = [r["per_class"][c]["db"] for c in names]
c_.barh(range(len(names)), db, color=[COL[c] for c in names], height=0.62)
c_.axvline(0.1, color=ns.INK, lw=0.9, ls=(0, (4, 2)))
c_.text(0.1, len(names) - 0.35, " budget", fontsize=ns.fs(6), color=ns.INK, va="top")
for i, (c, v) in enumerate(zip(names, db)):
    c_.text(v + 0.003, i, f"{r['per_class'][c]['res']}", fontsize=ns.fs(6),
            va="center", color=ns.INK2)
c_.set_yticks(range(len(names)))
c_.set_yticklabels([c.replace("_", " ") for c in names], fontsize=ns.fs(6))
c_.set_xlabel("dB spent")
c_.set_xlim(0, max(db) * 1.45)
c_.set_title("dB spent per class", fontsize=ns.fs(6), color=ns.INK2, loc="left")
ns.panel(c_, "c", dx=-0.30)

fig.tight_layout(w_pad=2.2, h_pad=1.2)
out = R / "docs/figures/perclass.png"
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "perclass.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
print(f"  -> {out}")
