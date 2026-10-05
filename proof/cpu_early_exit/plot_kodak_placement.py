"""Single-column vector figure for complete Kodak placement replay."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "proof/cpu_early_exit/results/kodak24_qp5_placement_20261005"
OUT = ROOT / "docs/figures/kodak-placement-20261005"
QPS = [0, 16, 32, 48, 63]


def main() -> None:
    data = json.loads((BASE / "summary.json").read_text())
    rows = data["per_image_qp"]
    assert len(rows) == 120 and all(len([r for r in rows if r["qp"] == q]) == 24 for q in QPS)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .65, "xtick.major.width": .55, "ytick.major.width": .55,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    navy, teal, muted = "#17324d", "#087e83", "#9aabb4"
    fig, ax = plt.subplots(figsize=(3.45, 2.46))
    fig.subplots_adjust(left=.165, right=.98, bottom=.23, top=.92)
    for j, qp in enumerate(QPS):
        subset = sorted((r for r in rows if r["qp"] == qp), key=lambda r: r["image"])
        gain = np.asarray([r["rgb_gain_db"] for r in subset])
        jitter = np.linspace(-1.9, 1.9, 24)
        nonuniform = np.asarray([r["nonuniform_map"] for r in subset])
        ax.scatter(qp + jitter[nonuniform], gain[nonuniform], s=11,
                   color=muted, alpha=.72, linewidths=0, zorder=2)
        if (~nonuniform).any():
            ax.scatter(qp + jitter[~nonuniform], gain[~nonuniform], s=13,
                       facecolors="white", edgecolors=muted, linewidths=.65, zorder=2)
        stat = data["per_qp"][str(qp)]
        lo, hi = stat["rgb_image_cluster_ci95_db"]
        mean = stat["mean_rgb_gain_db"]
        ax.vlines(qp, lo, hi, color=teal, linewidth=2.0, zorder=4)
        ax.hlines([lo, hi], qp-1.25, qp+1.25, color=teal, linewidth=1.0, zorder=4)
        ax.scatter(qp, mean, s=38, color=navy, edgecolors="white", linewidths=.7, zorder=5)
    ax.axhline(0, color=navy, linewidth=.7, alpha=.67, zorder=1)
    ax.set_xlim(-5, 68)
    ax.set_xticks(QPS)
    ax.set_xlabel("Quality index (QP)", labelpad=6)
    ax.set_ylabel("Original-map placement gain (dB)", labelpad=7)
    ax.set_title("Same depth counts, different tile placement", loc="left",
                 color=navy, fontsize=9.2, fontweight="semibold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6ecef", linewidth=.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color=navy)
    ax.text(.98, .02, "24 images / QP", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color="#5e717e")
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"kodak_placement.{suffix}", dpi=350,
                    bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
