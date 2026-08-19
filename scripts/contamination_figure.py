"""Seam against per-tile depth: what predicts it and what does not."""
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

d = json.load(open(R / "results/contamination_law.json"))
b = np.array(d["b"])
qps = list(d["seam_db"])
COL = {"0": ns.BLUE, "32": ns.ORANGE, "63": ns.VERM}

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.4))

# a: the measurement, log-log, with the fitted power law
for q in qps:
    y = np.array(d["seam_db"][q])
    A = np.polyfit(np.log(b), np.log(y), 1)
    ax[0].loglog(b, y, "o", ms=4.5, color=COL[q], label=f"q{q}")
    bb = np.linspace(b.min(), b.max(), 50)
    ax[0].loglog(bb, np.exp(A[1]) * bb ** A[0], "-", lw=1.0, color=COL[q])
ax[0].set_xlabel("per-tile blocks $b$"); ax[0].set_ylabel("seam (dB)")
ax[0].set_xticks([2, 4, 8, 12])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].legend(fontsize=5.5, loc="upper left")
ns.panel(ax[0], "a")

# b: both models against the data, one free scale each
y = np.array(d["seam_db"]["63"])
for name, m, c in (("area", np.array(d["area_model"]), ns.GREEN),
                   ("$b^2$", (b / b.max()) ** 2, ns.PURPLE)):
    s = (m @ y) / (m @ m)
    ax[1].plot(b, s * m, "-o", ms=4, lw=1.1, color=c, label=name)
ax[1].plot(b, y, "s", ms=5, color=ns.INK, label="measured")
ax[1].set_xlabel("per-tile blocks $b$"); ax[1].set_ylabel("seam at q63 (dB)")
ax[1].legend(fontsize=5.5, loc="upper left")
ns.panel(ax[1], "b", dx=-0.24)

# c: relative error of each model, all rates
# Three models, not two: the area fraction, a fixed square, and a power with the
# exponent fitted. The fitted exponent is what the paper's "roughly the square"
# claim rests on -- a FIXED square is a good deal worse than it, and saying so is
# the difference between a law and a slogan.
w = 0.27
for i, q in enumerate(qps):
    y = np.array(d["seam_db"][q])
    errs = []
    for m in (np.array(d["area_model"]), (b / b.max()) ** 2):
        s = (m @ y) / (m @ m)
        errs.append(100 * np.mean(np.abs(s * m - y) / y))
    A_ = np.vstack([np.log(b), np.ones_like(b)]).T
    al, c0 = np.linalg.lstsq(A_, np.log(y), rcond=None)[0]
    errs.append(100 * np.mean(np.abs(np.exp(c0) * b ** al - y) / y))
    ax[2].bar(i - w, errs[0], w, color=ns.GREEN)
    ax[2].bar(i, errs[1], w, color=ns.PURPLE)
    ax[2].bar(i + w, errs[2], w, color=ns.ORANGE)
    for xx, e in ((i - w, errs[0]), (i, errs[1]), (i + w, errs[2])):
        ax[2].annotate(f"{e:.0f}", (xx, e), fontsize=5.5, ha="center",
                       va="bottom", textcoords="offset points", xytext=(0, 1.5))
ax[2].set_xticks(range(len(qps))); ax[2].set_xticklabels([f"q{q}" for q in qps])
ax[2].set_ylabel("mean relative error (%)")
ax[2].set_ylim(0, 310)
ax[2].bar(0, 0, color=ns.GREEN, label="area")
ax[2].bar(0, 0, color=ns.PURPLE, label="$b^2$")
ax[2].bar(0, 0, color=ns.ORANGE, label=r"$b^{\alpha}$, $\alpha$ fitted")
ax[2].legend(fontsize=5.5, loc="upper right")
ns.panel(ax[2], "c", dx=-0.26)

fig.tight_layout()
fig.savefig(R / "docs/figures/contamination.png", dpi=300)
print("  -> docs/figures/contamination.png")
