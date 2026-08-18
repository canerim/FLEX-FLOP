"""Typeset formulas as images, so a slide can carry real mathematics.

python-pptx has no equation support: anything written in a text box is a string
in the body font, and `argmin_k` set in Calibri is not a formula, it is a
transcription of one. Rendering through matplotlib's mathtext gives the same
typesetting rules LaTeX uses -- proper operators, correct spacing around
relations, real subscripts -- and the result drops into a slide as a picture.

Transparent background and a generous DPI, so the glyphs stay sharp when the
slide is projected and the surrounding slide colour shows through.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "eq"
OUT.mkdir(parents=True, exist_ok=True)


def eq(tex: str, size: int = 22, color: str = "#1a1a1a") -> Path:
    """Render one display formula. Returns the PNG path.

    Cached on a hash of the arguments: a deck is rebuilt many times while its
    text is edited, and re-rasterising unchanged mathematics each time is pure
    latency.
    """
    key = hashlib.md5(f"{tex}|{size}|{color}".encode()).hexdigest()[:16]
    p = OUT / f"{key}.png"
    if p.exists():
        return p
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(p, dpi=300, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for s in (r"k_t^{\star}(\lambda)=\arg\min_k\;\{D[t,k]+\lambda\,C_k\}",
              r"\hat{k}_t=\arg\max_k\;\{\log\mathrm{softmax}(z_t)_k-\beta\,C_k\}"):
        print(" ", eq(s))
