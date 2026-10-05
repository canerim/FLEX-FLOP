"""Two-panel vector comparison of Kodak fixed-cost placement controls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RANDOM = ROOT / "proof/cpu_early_exit/results/kodak24_qp5_placement_20261005/summary.json"
BAYER = ROOT / "proof/cpu_early_exit/results/kodak24_qp5_bayer_histogram_20261005/summary.json"
OUT = ROOT / "docs/figures/kodak-placement-controls-20261005"
QPS = (0, 16, 32, 48, 63)


def main() -> None:
    random = json.loads(RANDOM.read_text())
    bayer = json.loads(BAYER.read_text())
    assert len(random["per_image_qp"]) == len(bayer["per_image_qp"]) == 120
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8.5, "xtick.labelsize": 7.6,
                         "ytick.labelsize": 7.6, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    navy, teal, orange, muted = "#17324d", "#087e83", "#c47535", "#9aabb4"
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True)
    fig.subplots_adjust(left=.083, right=.985, bottom=.235, top=.84, wspace=.15)
    max_point = max(r["rgb_gain_db"] for source in (random, bayer)
                    for r in source["per_image_qp"])
    min_point = min(r["rgb_gain_db"] for source in (random, bayer)
                    for r in source["per_image_qp"])
    lo = min(-.006, min_point-.005)
    hi = max(.08, max_point+.007)
    controls = [(random, "a  Three random permutations", teal,
                 "mean_rgb_gain_db", "nonuniform_map"),
                (bayer, "b  Fixed Bayer placement", orange,
                 "mean_router_gain_rgb_db", "same_map")]
    for ax, (source, title, accent, mean_key, state_key) in zip(axes, controls):
        for qp in QPS:
            group = sorted((r for r in source["per_image_qp"] if r["qp"] == qp),
                           key=lambda r: r["image"])
            assert len(group) == 24
            gains = np.asarray([r["rgb_gain_db"] for r in group])
            unchanged = np.asarray([not r[state_key] if state_key == "nonuniform_map"
                                    else r[state_key] for r in group])
            jitter = np.linspace(-1.55, 1.55, 24)
            ax.scatter(qp+jitter[~unchanged], gains[~unchanged], s=9.5,
                       color=muted, alpha=.62, linewidths=0, zorder=2)
            if unchanged.any():
                ax.scatter(qp+jitter[unchanged], gains[unchanged], s=12,
                           facecolors="white", edgecolors=muted,
                           linewidths=.55, zorder=2)
            stat = source["per_qp"][str(qp)]
            low, high = stat["rgb_image_cluster_ci95_db"]
            ax.vlines(qp, low, high, color=accent, linewidth=1.8, zorder=4)
            ax.hlines([low, high], qp-1.1, qp+1.1,
                      color=accent, linewidth=.85, zorder=4)
            ax.scatter(qp, stat[mean_key], s=32, color=navy,
                       edgecolors="white", linewidths=.65, zorder=5)
        ax.axhline(0, color=navy, linewidth=.7, alpha=.65, zorder=1)
        ax.set_xlim(-5, 68)
        ax.set_ylim(lo, hi)
        ax.set_xticks(QPS)
        ax.set_xlabel("Quality index (QP)", labelpad=5)
        ax.set_title(title, loc="left", color=navy,
                     fontsize=9.0, fontweight="semibold", pad=6)
        ax.yaxis.grid(True, color="#e6ecef", linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, width=.55, color=navy)
    axes[0].set_ylabel("Router placement gain (dB)", labelpad=5)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"kodak_placement_controls.{suffix}", dpi=350,
                    bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
