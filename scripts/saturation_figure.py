"""Where a quality budget actually does something, per rate.

Three regions, and only the middle one is a design choice:

  below the floor        no allocation meets the budget at all -- the tiling
                         penalty alone already exceeds it
  floor .. saturation    the budget genuinely buys compute; this is the band
                         every reported operating point lives in
  above saturation       every tile is already on the cheapest rung, saving is
                         pinned at the ceiling, and more dB buys nothing
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "RECIPE512"
def _pick(*names):
    for n in names:
        if (R / "results" / n).exists():
            print(f"  reading {n}")
            return json.load(open(R / "results" / n))
    raise SystemExit(f"none of {names} exists")


d = _pick(f"saturation_{TAG}_ctc53.json", f"saturation_{TAG}.json")
rows = d["rows"]
qp = np.array([r["qp"] for r in rows], float)
sat = np.array([r["saturation_db"] for r in rows])
flo = np.array([r["floor_db"] for r in rows])
CEIL = d["ceiling_pct"]

fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.8))

# ---- a: the three regions --------------------------------------------------
a = ax[0]
a.fill_between(qp, 0, flo, color=ns.VERM, alpha=0.16, lw=0)
a.fill_between(qp, flo, sat, color=ns.GREEN, alpha=0.16, lw=0)
a.fill_between(qp, sat, 0.42, color="#bbbbbb", alpha=0.22, lw=0)
a.plot(qp, flo, marker="^", color=ns.GREEN, lw=1.2,
       label="floor")
a.plot(qp, sat, marker="o", color=ns.BLUE, lw=1.2,
       label=f"saturation — every tile at exit {d['exit_j']}")
a.axhline(0.1, color=ns.INK, lw=0.8, ls=(0, (4, 2)), label="0.1 dB budget")

a.annotate(f"{sat[0]:.3f} dB", (qp[0], sat[0]), fontsize=ns.fs(6), color=ns.BLUE,
           textcoords="offset points", xytext=(3, 5))
a.annotate(f"{sat[-1]:.3f} dB", (qp[-1], sat[-1]), fontsize=ns.fs(6), color=ns.BLUE,
           textcoords="offset points", xytext=(-4, 5), ha="right")



a.set_ylim(0, 0.42); a.set_xlim(qp[0], qp[-1])
a.set_xlabel("qp   (0 = lowest rate  →  63 = highest)")
a.set_ylabel("quality budget, dB below the release")
a.legend(loc="center left", fontsize=ns.fs(6), bbox_to_anchor=(0.02, 0.62))
ns.panel(a, "a")

# ---- b: what a budget is worth, and where it stops being worth anything ----
b = ax[1]
b.plot(qp, sat - flo, marker="s", color=ns.ORANGE, lw=1.2)
b.set_xlabel("qp"); b.set_ylabel("width of the usable band (dB)")
b.set_xlim(qp[0], qp[-1])
used = (0.1 - flo) / (sat - flo)
b2 = b.twinx()
b2.plot(qp, 100 * used, marker="o", color=ns.BLUE, lw=1.0, ls=(0, (3, 2)))
b2.set_ylabel("share of the band 0.1 dB uses (%)", color=ns.BLUE,
               fontsize=ns.fs(6))
b2.tick_params(axis="y", colors=ns.BLUE)
b2.grid(False)
for x, u in zip(qp[::2], (100 * used)[::2]):
    b2.annotate(f"{u:.0f}%", (x, u), fontsize=ns.fs(6), color=ns.BLUE,
                textcoords="offset points", xytext=(0, -9), ha="center")
ns.panel(b, "b", dx=-0.20)

fig.tight_layout()
# Both directories. build_pdf reads paper/figures, and the f-string filename
# also hid this figure from check_figs_fresh, which looks for a literal name.
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / f"saturation_{TAG}.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    print(f"  -> {_d.relative_to(R)}/saturation_{TAG}.png")
