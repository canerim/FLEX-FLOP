"""Plot the RD curves and the saving available at chosen quality points.

Two panels, because the two questions are different:

  left   Rate-distortion. Where each trained model sits — PSNR against bpp. This
         is where "is the codec any good" is answered, and it is the panel that
         shows why the from-scratch model cannot be compared to real DCVC-UF at a
         given QP: it learned its own quantisation mapping, so the same QP index
         lands at a different rate.

  right  What early exit buys, per model, at qp63. Saving against dB given up,
         measured through the deployed patched path. The two models sit in
         completely different regimes and the reason matters — see the caption.

Colours are the reference palette's categorical slots 1-3 in fixed order.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
C_REL, C_OURS, C_WS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

rel = json.loads((ROOT / "results/rd_release_dense.json").read_text())["dense"]
ours = json.loads((ROOT / "results/rd_e1_ours.json").read_text())["dense"]

# Ladder measured through forward_routed at qp63: (saving %, dB given up).
# From-scratch e1 at epoch 3, and the warm-started release with adapters still
# at zero-init.
LADDER_OURS = [(0.0, 0.0432), (13.98, 0.1006), (28.88, 0.2137), (43.79, 0.1787)]
LADDER_WS = [(0.0, 0.7076), (13.98, 2.5705), (28.88, 4.9893), (43.79, 6.2736)]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6))
for a in (ax, bx):
    a.grid(True, color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)

# ---- left: rate-distortion ------------------------------------------------
ax.plot([r["bpp"] for r in rel], [r["psnr_611"] for r in rel], "-o",
        color=C_REL, lw=2, ms=8, label="released DCVC-UF (encoder = Microsoft's)")
ax.plot([r["bpp"] for r in ours], [r["psnr_611"] for r in ours], "-o",
        color=C_OURS, lw=2, ms=8, label="ours, trained from scratch (encoder also trained)")
for r in rel[::2]:
    ax.annotate(f"qp{r['qp']}", (r["bpp"], r["psnr_611"]), textcoords="offset points",
                xytext=(4, -12), fontsize=8, color=C_REL)
for r in ours:
    ax.annotate(f"qp{r['qp']}", (r["bpp"], r["psnr_611"]), textcoords="offset points",
                xytext=(4, 7), fontsize=8, color=C_OURS)

# the honest comparison is vertical at matched rate, not at matched QP
b = 0.783
p_ours = 39.92
p_rel = 40.72 + (b - 0.6278) / (0.8287 - 0.6278) * (41.86 - 40.72)
ax.annotate("", xy=(b, p_rel), xytext=(b, p_ours),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.4))
ax.annotate(f"{p_rel - p_ours:.2f} dB at matched bpp", (b, (p_rel + p_ours) / 2),
            textcoords="offset points", xytext=(-140, -4), fontsize=9, color=INK)

ax.set_xlabel("bits per pixel", color=INK2, fontsize=10)
ax.set_ylabel("PSNR (6:1:1 YUV), dB", color=INK2, fontsize=10)
ax.set_title("Rate-distortion — held-out Open Images", color=INK, fontsize=11.5, pad=10)
ax.legend(frameon=False, fontsize=9, loc="lower right")

# ---- right: what early exit costs ----------------------------------------
bx.plot([s for s, _ in LADDER_OURS], [d for _, d in LADDER_OURS], "-o",
        color=C_OURS, lw=2, ms=8, label="ours, from scratch (epoch 3)")
bx.plot([s for s, _ in LADDER_WS], [d for _, d in LADDER_WS], "-o",
        color=C_WS, lw=2, ms=8, label="released weights, adapters untrained")
for (s, d), k in zip(LADDER_OURS, [5, 4, 3, 2]):
    bx.annotate(f"exit {k}", (s, d), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color=C_OURS)
for (s, d), k in zip(LADDER_WS, [5, 4, 3, 2]):
    bx.annotate(f"exit {k}", (s, d), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color=C_WS)
bx.axhline(0.1, color=INK2, lw=1, ls="--")
bx.annotate("0.1 dB target", (1, 0.13), fontsize=9, color=INK2)
bx.set_xlabel("decoder compute saved (%)", color=INK2, fontsize=10)
bx.set_ylabel("PSNR given up (dB)", color=INK2, fontsize=10)
bx.set_title("What early exit costs, qp63, deployed patched path",
             color=INK, fontsize=11.5, pad=10)
bx.legend(frameon=False, fontsize=9, loc="upper left")

fig.text(0.5, -0.06,
         "The two right-hand curves are not a like-for-like race. The from-scratch model gives up "
         "little when truncated because it is\nundertrained and already soft — there is less detail "
         "left to lose — while sitting 1.68 dB below real DCVC-UF at matched rate.\n"
         "The released weights give up a lot because they are sharp AND their adapters are still at "
         "zero-init; closing that is what Stage A does.",
         ha="center", fontsize=9, color=INK2)

fig.tight_layout()
out = ROOT / "results/rd_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {out}")

# ---- the numbers, at chosen quality points --------------------------------
print("\nSAVING AT CHOSEN QUALITY POINTS (qp63, deployed path)\n")
print(f"  {'dB budget':>10} {'ours (from scratch)':>22} {'released + untrained adapters':>32}")
for budget in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0):
    def best(ladder):
        ok = [s for s, d in ladder if d <= budget]
        return f"{max(ok):.1f}%" if ok else "—"
    print(f"  {budget:>9.2f} {best(LADDER_OURS):>22} {best(LADDER_WS):>32}")
print("\n  '—' means no exit meets that budget.")
