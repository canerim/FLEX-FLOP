"""Show the measured architecture/implementation split without a 3x-only claim."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results/cohort_20261003"
OUT = DATA / "latency_decomposition"
SEQUENCES = ("BQMall", "FourPeople", "videoSRC05")
LABELS = ("480p", "720p", "1080p")
QPS = (16, 32, 48)


def main() -> None:
    cases = {(seq, qp): json.loads((DATA / f"{seq}_qp{qp}.json").read_text())
             for seq in SEQUENCES for qp in QPS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": .7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.85),
                                 gridspec_kw={"width_ratios": [1.08, 1]})
    xs = np.arange(3)
    arms = (("released_d12", "D12 · PyTorch", "#bf6752"),
            ("e15_stock", "e15 · PyTorch", "#478b93"),
            ("e15_triton", "e15 · Triton", "#254d68"))
    for j, (key, label, color) in enumerate(arms):
        vals = [cases[seq, 32]["median_wall_ms"][key] for seq in SEQUENCES]
        ax.bar(xs + (j - 1) * .23, vals, .20, color=color, label=label)
    ax.set(xticks=xs, xticklabels=LABELS, ylabel="Decoder synthesis time (ms)")
    ax.set_title("a   Same-QP latency at QP 32", loc="left", fontsize=9,
                 fontweight="bold", pad=9)
    ax.grid(axis="y", color="#dfe7e8", linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")

    colors = {16: "#b75d49", 32: "#37848c", 48: "#294e69"}
    for qp in QPS:
        stock = [cases[seq, qp]["median_speedup_released_vs_e15_stock_wall"]
                 for seq in SEQUENCES]
        total = [cases[seq, qp]["median_speedup_wall"] for seq in SEQUENCES]
        bx.plot(xs, stock, marker="o", markersize=4, linewidth=1.25,
                color=colors[qp], linestyle="--")
        bx.plot(xs, total, marker="o", markersize=4, linewidth=1.5,
                color=colors[qp], label=f"QP {qp}")
    bx.set(xticks=xs, xticklabels=LABELS, ylim=(.95, 3.15),
           ylabel="Released D12 / e15 time (×)")
    bx.axhline(1, color="#819195", linewidth=.65)
    bx.grid(axis="y", color="#dfe7e8", linewidth=.5)
    bx.set_title("b   Implementation contribution", loc="left", fontsize=9,
                 fontweight="bold", pad=9)
    bx.legend(frameon=False, fontsize=7.2, loc="center", ncol=3,
              bbox_to_anchor=(.51, .52))
    fig.subplots_adjust(left=.09, right=.99, top=.80, bottom=.25, wspace=.37)
    fig.text(.015, .08, "Dashed: e15 PyTorch. Solid: e15 Triton. Released D12 uses stock PyTorch in both ratios.",
             fontsize=6.7, color="#435a60")
    fig.text(.015, .015,
             "Nine first-frame CTC workloads · 20 paired trials each · idle A6000 · shared latent · same-QP synthesis only",
             fontsize=6.4, color="#697d82")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    med_stock = statistics.median(cases[seq, qp]["median_speedup_released_vs_e15_stock_wall"]
                                  for seq in SEQUENCES for qp in QPS)
    med_total = statistics.median(cases[seq, qp]["median_speedup_wall"]
                                  for seq in SEQUENCES for qp in QPS)
    print(f"{OUT}: stock {med_stock:.3f}x; stock-vs-Triton {med_total:.3f}x")


if __name__ == "__main__":
    main()
