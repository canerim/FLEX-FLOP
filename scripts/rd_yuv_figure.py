"""Rate-distortion per colour component, one panel each, against the release.

The convention is the one a codec group reads without a caption: rate on x,
PSNR on y, one panel for Y, U and V, every curve's BD-rate against the anchor
in brackets beside its name. The anchor here is the released DCVC-UF decoder
itself, so a bracket is not one codec against another -- it is what decoding
the same bitstream with fewer operations costs, in the currency of rate.

What the three panels are for. The budget is set on DCVC-UF's 6:1:1 weighted
PSNR, in which chroma carries a seventh of the weight. A weighted mean can sit
at 0.1 dB while luma and chroma move apart underneath it, and no number the
paper otherwise reports would show it. Splitting the metric is the only way to
see whether the budget is quietly spending chroma to buy luma.

    python scripts/rd_yuv_figure.py [--out docs/figures/rd_yuv.png]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import naturestyle as ns          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from paper_metrics import bd_rate  # noqa: E402

COMPONENTS = [("psnr_y", "PSNR-Y"), ("psnr_u", "PSNR-U"), ("psnr_v", "PSNR-V")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/rd_yuv_PAPER.json")
    ap.add_argument("--out", default="docs/figures/rd_yuv.png")
    a = ap.parse_args()
    ns.apply()

    d = json.loads((ROOT / a.src).read_text())
    rows = d["rows"]
    qs = sorted({r["qp"] for r in rows})
    rel = {r["qp"]: r for r in rows if r["config"] == "release"}
    budgets = sorted({r["budget_db"] for r in rows
                      if r["budget_db"] is not None})
    ours = {b: {r["qp"]: r for r in rows if r["budget_db"] == b}
            for b in budgets}

    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.3))
    for i, (key, title) in enumerate(COMPONENTS):
        x0 = np.array([rel[q]["bpp"] for q in qs])
        y0 = np.array([rel[q][key] for q in qs])
        ax[i].plot(x0, y0, "-o", color=ns.BLACK, lw=1.3, ms=3.5, zorder=4,
                   label="DCVC-UF (anchor)")
        for n, b in enumerate(budgets):
            g = ours[b]
            got = [q for q in qs if q in g]
            if len(got) < 2:
                continue
            # The map is the only thing our side adds to the file, so the rate
            # axis moves by the map's bits and nothing else.
            x = np.array([g[q]["bpp"] for q in got])
            y = np.array([g[q][key] for q in got])
            lbl = f"{b:g} dB"
            if len(got) == len(qs):
                bd = bd_rate(x0, y0, x, y)
                lbl += f" [{bd:+.2f}%]"
            ax[i].plot(x, y, "--", color=ns.SERIES[n % len(ns.SERIES)],
                       lw=1.0, marker="s", ms=2.6, label=lbl)
        ax[i].set_title(title, fontsize=8, pad=3)
        ax[i].set_xlabel("rate (bpp)")
        if i == 0:
            ax[i].set_ylabel("PSNR (dB)")
        ax[i].legend(fontsize=5.6, loc="lower right", frameon=False,
                     handlelength=1.5, labelspacing=0.25, borderpad=0)
        ns.panel(ax[i], "abc"[i], dx=-0.24)

    fig.tight_layout(w_pad=1.4)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"  -> {a.out}")

    # The number the figure exists to expose, printed so it cannot be missed.
    print(f"\n  BD-rate against the released decoder, per component")
    print(f"  {'budget':>8}" + "".join(f"{t:>12}" for _, t in COMPONENTS)
          + f"{'6:1:1':>12}")
    for b in budgets:
        g = ours[b]
        got = [q for q in qs if q in g]
        if len(got) < len(qs):
            continue
        x0 = np.array([rel[q]["bpp"] for q in got])
        x = np.array([g[q]["bpp"] for q in got])
        cells = []
        for key, _ in COMPONENTS + [("psnr_611", "6:1:1")]:
            y0 = np.array([rel[q][key] for q in got])
            y = np.array([g[q][key] for q in got])
            cells.append(f"{bd_rate(x0, y0, x, y):>+11.2f}%")
        print(f"  {b:>8.2f}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
