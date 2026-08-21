"""The curve a compression reader looks for first: bits against quality, ours
on the released codec's own axes.

Everything else in this paper is relative, and for the claim that is right --
the bitstream does not change, so the only honest comparison is against the
decoder that reads it. But a reader coming from the compression literature
wants to see the two rate-distortion curves on one plot before believing any
of it, and until now the paper never drew them.

Panel a is that plot, and the finding is that there is nothing to see: at a
0.1 dB budget the two curves are indistinguishable at the scale a codec is
normally read at. Panel b is the same data with the vertical axis blown up
into the difference, which is where the budget lives, and the decoder
arithmetic saved beside it -- what is given up against what is bought.

    python scripts/rd_vs_uf_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import naturestyle as ns          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from savings import sv, pick      # noqa: E402

ns.apply()
OUT = (ROOT / "docs/figures", ROOT / "paper/figures")
BUDGETS = [0.1, 0.2, 0.3]


def main() -> int:
    rd = json.loads((ROOT / "results/rd_absolute_PAPER.json").read_text())
    ref = {r["qp"]: r for r in rd["rows"]}
    qps = sorted(ref)
    sig, sig_f = pick("signalled_RECIPE512_ctc53.json")
    by = {}
    for r in sig["rows"]:
        by.setdefault(round(r["budget_db"], 3), {})[r["qp"]] = r

    bpp = np.array([ref[q]["bpp"] for q in qps])
    rel = np.array([ref[q]["psnr_release"] for q in qps])

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))

    # a -- the two curves on the axes a codec is read on
    ax[0].plot(bpp, rel, "-o", color=ns.BLACK, lw=1.3, ms=4,
               label="DCVC-UF (released)", zorder=3)
    for n, b in enumerate(BUDGETS):
        rows = by.get(b, {})
        got = [q for q in qps if q in rows]
        if not got:
            continue
        y = np.array([ref[q]["psnr_release"] - rows[q]["db_vs_uf"] for q in got])
        x = np.array([ref[q]["bpp"] + rows[q].get("bpp_added", 0.0) for q in got])
        ax[0].plot(x, y, "--", color=ns.SERIES[n], lw=1.1,
                   label=f"FLEX-UF, {b:g} dB budget")
    ax[0].set_xlabel("rate (bits per pixel)")
    ax[0].set_ylabel("PSNR (dB), 6:1:1 on YUV420")
    ax[0].legend(fontsize=6, loc="lower right", frameon=False,
                 handlelength=1.6, labelspacing=0.3)
    ns.panel(ax[0], "a")

    # b -- the same difference, magnified, against what it buys
    ax2 = ax[1].twinx()
    for n, b in enumerate(BUDGETS):
        rows = by.get(b, {})
        got = [q for q in qps if q in rows]
        if not got:
            continue
        x = np.array([ref[q]["bpp"] for q in got])
        ax[1].plot(x, [rows[q]["db_vs_uf"] for q in got], "-o",
                   color=ns.SERIES[n], lw=1.1, ms=3.5, label=f"{b:g} dB")
        ax2.plot(x, [sv(rows[q]) for q in got], ":s", color=ns.SERIES[n],
                 lw=1.0, ms=3.0, alpha=0.85)
    ax[1].set_xlabel("rate (bits per pixel)")
    ax[1].set_ylabel("quality given up (dB)")
    ax2.set_ylabel("decoder MACs saved (%)")
    ax2.grid(False)
    ax[1].legend(fontsize=6, loc="upper left", frameon=False, ncol=3,
                 columnspacing=0.8, handlelength=1.4)
    ns.panel(ax[1], "b", dx=-0.20)

    fig.tight_layout(w_pad=2.0)
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "rd_vs_uf.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")

    # The numbers the caption quotes, written rather than read off the plot.
    bd, _ = pick("bdrate.json")
    stat = {"reference_file": "results/rd_absolute_PAPER.json",
            "signalled_file": sig_f,
            "bpp_low": float(bpp[0]), "bpp_high": float(bpp[-1]),
            "psnr_low": float(rel[0]), "psnr_high": float(rel[-1]),
            "max_gap_db_at_tenth": max(
                by[0.1][q]["db_vs_uf"] for q in qps if q in by.get(0.1, {})),
            "bd_rate_rows": bd.get("rows", []) if bd else []}
    (ROOT / "results/rd_vs_uf.json").write_text(json.dumps(stat, indent=2))
    print(f"  wrote rd_vs_uf.png   release {rel[0]:.2f}-{rel[-1]:.2f} dB over "
          f"{bpp[0]:.3f}-{bpp[-1]:.3f} bpp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
