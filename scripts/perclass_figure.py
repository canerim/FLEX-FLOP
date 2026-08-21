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
ns.apply()

d = json.load(open(R / "results/per_class_RECIPE512.json"))
ORDER = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
COL = {"MCL-JCV": ns.BLUE, "UVG": ns.SKY, "HEVC_B": ns.GREEN,
       "HEVC_E": ns.YELLOW, "HEVC_C": ns.ORANGE, "HEVC_D": ns.VERM}

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.6))

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
           label=f"{c.replace('_',' ')}  ({t} tiles, n={n})")
a.set_xlabel("qp"); a.set_ylabel("decoder MACs saved (%)")
a.legend(loc="upper right", fontsize=6, ncol=1, columnspacing=0.8,
             frameon=False, handlelength=1.2, labelspacing=0.25,
             borderpad=0.1)
a.set_ylim(0, 42)
a.set_title("0.1 dB, global operating point",
            fontsize=6, color=ns.INK2, loc="left")
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
for c in ORDER:
    if c in r0["per_class"]:
        v = r0["per_class"][c]
        # Three classes share 40 tiles, so their labels landed on top of one
        # another at the right edge. Fan them out vertically by the order they
        # appear, which is also the order of their savings.
        _dy = {"UVG": -7, "HEVC_B": -1}.get(c, 3)
        b.annotate(c.replace("HEVC_", "").replace("MCL-JCV", "MCL"),
                   (v["tiles"], v["saving"]), fontsize=6, color=ns.INK2,
                   textcoords="offset points", xytext=(4, _dy))
b.set_xscale("log"); b.set_xticks([2, 8, 15, 40])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.set_xlabel("tiles per frame at 256 px"); b.set_ylabel("MACs saved (%)")
b.legend(loc="lower right", fontsize=6)
b.set_title("Saving against tile count",
            fontsize=6, color=ns.INK2, loc="left")
ns.panel(b, "b", dx=-0.22)

# ---- c: what each class actually spends ------------------------------------
c_ = ax[2]
r = next(x for x in rows if x["qp"] == qps[0])
names = [c for c in ORDER if c in r["per_class"]]
db = [r["per_class"][c]["db"] for c in names]
c_.barh(range(len(names)), db, color=[COL[c] for c in names], height=0.62)
c_.axvline(0.1, color=ns.INK, lw=0.9, ls=(0, (4, 2)))
c_.text(0.1, len(names) - 0.35, " budget", fontsize=6, color=ns.INK, va="top")
for i, (c, v) in enumerate(zip(names, db)):
    c_.text(v + 0.003, i, f"{r['per_class'][c]['res']}", fontsize=6,
            va="center", color=ns.INK2)
c_.set_yticks(range(len(names)))
c_.set_yticklabels([c.replace("_", " ") for c in names], fontsize=6)
c_.set_xlabel("dB spent, global operating point")
c_.set_xlim(0, max(db) * 1.45)
c_.set_title("dB spent per class", fontsize=6, color=ns.INK2, loc="left")
ns.panel(c_, "c", dx=-0.30)

fig.tight_layout()
out = R / "docs/figures/perclass.png"
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "perclass.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
print(f"  -> {out}")
