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
ns.apply()

TAG = sys.argv[1] if len(sys.argv) > 1 else "RECIPE512"
d = json.load(open(R / f"results/saturation_{TAG}.json"))
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
       label="floor — every tile at full depth")
a.plot(qp, sat, marker="o", color=ns.BLUE, lw=1.2,
       label=f"saturation — every tile at exit {d['exit_j']}")
a.axhline(0.1, color=ns.INK, lw=0.8, ls=(0, (4, 2)))
a.text(qp[-1], 0.104, "the 0.1 dB budget this project works to", fontsize=5,
       color=ns.INK, ha="right")
a.annotate(f"{sat[0]:.3f} dB", (qp[0], sat[0]), fontsize=5.5, color=ns.BLUE,
           textcoords="offset points", xytext=(3, 5))
a.annotate(f"{sat[-1]:.3f} dB", (qp[-1], sat[-1]), fontsize=5.5, color=ns.BLUE,
           textcoords="offset points", xytext=(-4, 5), ha="right")
a.text(31, 0.385, "nothing left to buy — the budget is being thrown away",
       fontsize=5.2, ha="center", color="#666666")
a.text(31, (flo[4] + sat[4]) / 2 + 0.01, "where a budget buys compute",
       fontsize=5.2, ha="center", color="#1a6b52")
a.text(31, 0.014, "infeasible — tiling alone already costs this much",
       fontsize=5.2, ha="center", color=ns.VERM)
a.set_ylim(0, 0.42); a.set_xlim(qp[0], qp[-1])
a.set_xlabel("qp   (0 = lowest rate  →  63 = highest)")
a.set_ylabel("quality budget, dB below the release")
a.legend(loc="center left", fontsize=5, bbox_to_anchor=(0.02, 0.62))
a.set_title(f"{TAG}: the working range of a quality budget",
            fontsize=6, color=ns.INK2, loc="left")
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
               fontsize=6)
b2.tick_params(axis="y", colors=ns.BLUE)
b2.grid(False)
for x, u in zip(qp[::2], (100 * used)[::2]):
    b2.annotate(f"{u:.0f}%", (x, u), fontsize=5, color=ns.BLUE,
                textcoords="offset points", xytext=(0, -9), ha="center")
b.set_title(f"The band WIDENS with rate — the ladder has more to offer at high\n"
            f"rate, and 0.1 dB uses less and less of it. Ceiling is {CEIL:.2f}%\n"
            "throughout the grey region of panel a.",
            fontsize=6, color=ns.INK2, loc="left")
ns.panel(b, "b", dx=-0.20)

fig.tight_layout()
out = R / f"docs/figures/saturation_{TAG}.png"
fig.savefig(out, dpi=300)
print(f"  -> {out}")
