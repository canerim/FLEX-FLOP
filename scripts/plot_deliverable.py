"""The deliverable figure: dense vs routed, PSNR against bpp, one set of axes.

Both curves come from the SAME model and the same bitstream — only the synthesis
transform differs — so the vertical gap between them is attributable to the
decoder and to nothing else. The bpp axis is shared by construction.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/rd_warmstart.json").read_text())
dense, routed = d["dense"], d["routed"]
C_D, C_R = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6))
for a in (ax, bx):
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)

ax.plot([r["bpp"] for r in dense], [r["psnr_611"] for r in dense], "-o",
        color=C_D, lw=1.8, ms=5, label="Dense decoder (released DCVC-UF)")
ax.plot([r["bpp"] for r in routed], [r["psnr_611"] for r in routed], "-o",
        color=C_R, lw=1.8, ms=5, label="Multi-Exit Decoder (routed, top-1)")
for r in routed:
    ax.annotate(f"QP{r['qp']}", (r["bpp"], r["psnr_611"]), textcoords="offset points",
                xytext=(4, -13), fontsize=8, color=C_R)
ax.set_xlabel("bits per pixel — identical for both curves by construction",
              color=INK2, fontsize=10)
ax.set_ylabel("PSNR (6:1:1 YUV), dB", color=INK2, fontsize=10)
ax.set_title("Rate-distortion — held-out Open Images", color=INK, fontsize=11.5, pad=10)
ax.legend(frameon=False, fontsize=9, loc="lower right")

sv = [r["saving_pct"] for r in routed]
db = [dn["psnr_611"] - rt["psnr_611"] for dn, rt in zip(dense, routed)]
bx.plot(sv, db, "-o", color=C_R, lw=1.8, ms=5)
for r, s, y in zip(routed, sv, db):
    bx.annotate(f"QP{r['qp']}", (s, y), textcoords="offset points",
                xytext=(6, 5), fontsize=8, color=C_R)
bx.axhline(0.1, color=INK2, lw=1, ls="--")
bx.annotate("0.1 dB target", (min(sv), 0.13), fontsize=9, color=INK2)
bx.axvline(0, color=INK2, lw=1)
bx.set_xlabel("decoder compute saved (%)  —  negative means it costs more",
              color=INK2, fontsize=10)
bx.set_ylabel("PSNR given up vs the dense decoder (dB)", color=INK2, fontsize=10)
bx.set_title("What routing buys, per rate point", color=INK, fontsize=11.5, pad=10)

fig.subplots_adjust(bottom=0.30)
fig.text(0.5, -0.20,
    "Both decoders are run on the SAME batch, so every point shares one bitstream and the bpp is identical by construction — an\n"
    "earlier version swept them separately and the random crops differed, putting up to 0.005 bpp between curves that read the same bits.\n"
    "The router adapts to rate on its own: 24.7% saved at QP0 where a sparse latent makes early exits cheap, nothing at QP63 where a\n"
    "dense one makes them expensive. Saving turns slightly negative at high rate because the seam-repair pass is billed at 0.95% and,\n"
    "in this checkpoint, is still at zero-init identity.",
    ha="center", fontsize=9, color=INK2)
fig.tight_layout(rect=[0, 0.04, 1, 1])
out = ROOT / "results/rd_deliverable.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
print(f"wrote {out}")
