"""Which ladder is right depends on the budget, and the two curves cross.

The table this replaces reports four ladders at four budgets, and the fact
worth seeing in it is a crossing: the coarse ladder wins where the budget is
tight and the fine one wins where it is loose, because a finer ladder has a
higher ceiling and a higher floor. A table makes the reader find that; a plot
shows it.

Marked honestly. Only RECIPE512 is measured with hooks at every budget, and the
runs are on their own latest checkpoints rather than on one, so this compares
ladders and training together. The figure says so rather than leaving it to the
caption.

    python scripts/ladder_crossover.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()

STYLE = {
    "RECIPE512": (ns.BLUE, "-", "o", "K=6, j=2, 256 px"),
    "BEST": ("#6baed6", "-", "s", "K=6, j=2, 256 px, second run"),
    "BEST128": (ns.GREEN, "--", "^", "K=6, j=2, 128 px"),
    "FINE12": (ns.VERM, "-", "D", "K=12, j=4, 128 px"),
}


def main():
    tex = (ROOT / "paper/tables/runs.tex").read_text()
    buds, rows = [], []
    for line in tex.splitlines():
        if "dB &" in line or "\\,dB" in line and "Ladder" in line:
            buds = [float(x) for x in re.findall(r"([\d.]+)\\,dB", line)]
        if line.strip().startswith(("RECIPE512", "BEST", "FINE12")):
            cells = [c.strip() for c in line.split("&")]
            tag = cells[0]
            vals = []
            for c in cells[3:]:
                c = re.sub(r"\\textbf\{|\}|\$\^?\{?\\(ast|dagger)\}?\$?|\\\\", "", c)
                c = c.replace("$", "").replace("^", "").strip()
                vals.append(float(c) if re.fullmatch(r"[\d.]+", c) else None)
            ceil = re.sub(r"[^\d.]", "", cells[2])
            rows.append((tag, float(ceil), vals, "\\dagger" in line))
    if not buds or not rows:
        raise SystemExit("could not parse paper/tables/runs.tex")

    fig, ax = plt.subplots(figsize=(ns.W1, 1.62))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.4, length=2, width=0.5)

    for tag, ceil, vals, modelled in rows:
        col, ls, mk, lab = STYLE.get(tag, (ns.INK2, "-", "o", tag))
        xs = [b for b, v in zip(buds, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, ls, marker=mk, ms=2.6, lw=0.9, color=col,
                label=lab + ("*" if modelled else ""))
        ax.axhline(ceil, color=col, lw=0.5, ls=":", alpha=0.6)
    ax.text(buds[-1], rows[-1][1] + 0.6, "ceiling of the fine ladder",
            fontsize=4.6, color=ns.VERM, ha="right")
    ax.text(buds[-1], rows[0][1] - 2.4, "ceiling of the coarse one",
            fontsize=4.6, color=ns.BLUE, ha="right")

    ax.set_xlabel("distortion budget (dB)")
    ax.set_ylabel("decoder MACs saved (%)")
    ax.set_xticks(buds)
    ax.legend(frameon=False, fontsize=4.7, handlelength=1.6, borderpad=0,
              loc="lower right")
    ax.text(0.02, 0.95, "* saving from the arithmetic model, not hooks; only "
            "the first row\nis hook-counted at every budget, and each run is on "
            "its own\nlatest checkpoint",
            transform=ax.transAxes, fontsize=4.3, color=ns.INK2, va="top")
    out = ROOT / "docs/figures/ladder_crossover.png"
    for _d in (ROOT / "docs/figures", ROOT / "paper/figures"):
        _d.mkdir(parents=True, exist_ok=True)
        fig.savefig(_d / "ladder_crossover.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(f"  budgets {buds}")
    for tag, ceil, vals, m in rows:
        print(f"    {tag:<12} ceiling {ceil:5.1f}  {vals}  "
              f"{'modelled' if m else 'hook-counted'}")
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
