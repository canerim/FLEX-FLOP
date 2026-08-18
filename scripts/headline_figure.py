"""The two operating points that matter, side by side.

0.1 dB is the budget this project works to. 0.1818 dB is where the K = 6 ladder
stops responding at the lowest rate: every tile is already on the cheapest rung
it has, so saving is pinned at the architectural ceiling and more dB buys
nothing. Showing both makes the shape of the trade visible -- one is a slope, the
other is a wall.

Everything here is measured on the DEPLOYED tiled decode path, and every dB is
what a decoder delivers, verified by decoding the chosen exit map (flexuf/eval.py).
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
def _pick(*names):
    for n in names:
        if (R / "results" / n).exists():
            print(f"  reading {n}")
            return json.load(open(R / "results" / n))
    raise SystemExit(f"none of {names} exists")


fine = _pick(f"signalled_{TAG}_fine_ctc53.json", f"signalled_{TAG}_fine.json")
sat = _pick(f"saturation_{TAG}_ctc53.json", f"saturation_{TAG}.json")
CEIL = sat["ceiling_pct"]
SATD = {r["qp"]: r["saturation_db"] for r in sat["rows"]}
FLOOR = {r["qp"]: r["floor_db"] for r in sat["rows"]}

def at(b):
    return {r["qp"]: r for r in fine["rows"]
            if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}

B1, B2 = fine["budgets"][0], fine["budgets"][-1]
lo, hi = at(B1), at(B2)
qp = np.array(sorted(lo), float)

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.8))

# ---- a: saving against rate at both budgets --------------------------------
a = ax[0]
a.axhline(CEIL, color=ns.INK, lw=0.8, ls=(0, (4, 2)))

y2 = [hi[q]["saving_pct_vs_release"] for q in qp]
y1 = [lo[q]["saving_pct_vs_release"] for q in qp]
a.fill_between(qp, y1, y2, color=ns.ORANGE, alpha=0.13, lw=0)
a.plot(qp, y2, marker="o", color=ns.ORANGE, lw=1.3,
       label=f"{B2:.4f} dB — where qp 0 saturates")
a.plot(qp, y1, marker="s", color=ns.BLUE, lw=1.3, label=f"{B1:.2f} dB — the target")
a.annotate(f"{y2[0]:.1f}", (qp[0], y2[0]), fontsize=5.5, color=ns.ORANGE,
           textcoords="offset points", xytext=(4, -9))
a.annotate(f"{y1[0]:.1f}", (qp[0], y1[0]), fontsize=5.5, color=ns.BLUE,
           textcoords="offset points", xytext=(4, -9))
a.annotate(f"{y1[-1]:.1f}", (qp[-1], y1[-1]), fontsize=5.5, color=ns.BLUE,
           textcoords="offset points", xytext=(-3, -9), ha="right")
a.annotate(f"{y2[-1]:.1f}", (qp[-1], y2[-1]), fontsize=5.5, color=ns.ORANGE,
           textcoords="offset points", xytext=(-3, 5), ha="right")
a.set_xlabel("qp   (0 = lowest rate  →  63 = highest)")
a.set_ylabel("decoder MACs saved vs the release (%)")
a.set_xlim(qp[0], qp[-1]); a.set_ylim(0, CEIL * 1.18)
a.legend(loc="lower left", fontsize=5)
a.set_title(f"{fine['n_sequences']} CTC sequences",
            fontsize=6, color=ns.INK2, loc="left")
ns.panel(a, "a")

# ---- b: where the budget stops doing anything ------------------------------
b = ax[1]
sd = np.array([SATD[q] for q in qp]); fd = np.array([FLOOR[q] for q in qp])
b.fill_between(qp, 0, fd, color=ns.VERM, alpha=0.15, lw=0)
b.fill_between(qp, fd, sd, color=ns.GREEN, alpha=0.15, lw=0)
b.fill_between(qp, sd, 0.40, color="#bbbbbb", alpha=0.20, lw=0)
b.plot(qp, fd, marker="^", color=ns.GREEN, lw=1.2, label="floor: tiling alone")
b.plot(qp, sd, marker="o", color=ns.ORANGE, lw=1.2, label="saturation: ceiling reached")
b.axhline(B1, color=ns.BLUE, lw=1.0, ls=(0, (4, 2)))
b.text(qp[-1], B1 + 0.006, f"{B1:.2f} dB", fontsize=5, color=ns.BLUE, ha="right")
b.axhline(B2, color=ns.ORANGE, lw=0.8, ls=(0, (1, 2)))
b.text(qp[0] + 1, B2 + 0.006, f"{B2:.4f} dB", fontsize=5, color=ns.ORANGE)


b.set_xlabel("qp"); b.set_ylabel("quality budget, dB below the release")
b.set_xlim(qp[0], qp[-1]); b.set_ylim(0, 0.40)
b.legend(loc="upper left", fontsize=5, bbox_to_anchor=(0.0, 0.93))
b.set_title("Floor, saturation, and the two budgets",
            fontsize=6, color=ns.INK2, loc="left")
ns.panel(b, "b", dx=-0.22)

# ---- c: what the extra dB is worth --------------------------------------
c = ax[2]
gain = np.array(y2) - np.array(y1)
c.bar(qp, gain, width=5.0, color=ns.PURPLE)
for x, g in zip(qp, gain):
    c.annotate(f"{g:.1f}", (x, g), fontsize=5, ha="center", va="bottom",
               color=ns.INK, textcoords="offset points", xytext=(0, 1.5))
c.set_xlabel("qp")
c.set_ylabel(f"extra saving from {B1:.2f} → {B2:.4f} dB (pts)")
c.set_xlim(qp[0] - 4, qp[-1] + 4)
c.set_title("Extra saving from the looser budget",
            fontsize=6, color=ns.INK2, loc="left")
ns.panel(c, "c", dx=-0.24)

fig.tight_layout()
out = R / f"docs/figures/headline_{TAG}.png"
fig.savefig(out, dpi=300)
print(f"  -> {out}")
